#include "VolumetricReSTIRSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"

#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>

// PIX for Windows
#ifdef USE_PIX
#define USE_PIX_SUPPORTED_ARCHITECTURE
#include "debug/pix3.h"
#endif

using Microsoft::WRL::ComPtr;

/**
 * Initialize the volumetric ReSTIR system
 */
bool VolumetricReSTIRSystem::Initialize(Device* device, ResourceManager* resources, uint32_t width, uint32_t height) {
    if (m_initialized) {
        LOG_WARN("VolumetricReSTIRSystem already initialized");
        return true;
    }

    if (!device || !resources) {
        LOG_ERROR("Invalid device or resource manager");
        return false;
    }

    m_device = device;
    m_resources = resources;
    m_width = width;
    m_height = height;

    LOG_INFO("Initializing Volumetric ReSTIR System (Phase 1 - RIS only)");
    LOG_INFO("  Resolution: {}x{}", width, height);
    LOG_INFO("  Random walks per pixel (M): {}", m_randomWalksPerPixel);
    LOG_INFO("  Max bounces (K): {}", m_maxBounces);

    // Create GPU resources
    if (!CreateReservoirBuffers()) {
        LOG_ERROR("Failed to create reservoir buffers");
        return false;
    }

    if (!CreatePiecewiseConstantVolume()) {
        LOG_ERROR("Failed to create piecewise-constant volume");
        return false;
    }

    if (!CreatePipelines()) {
        LOG_ERROR("Failed to create compute pipelines");
        return false;
    }

    m_initialized = true;
    m_frameCount = 0;

    LOG_INFO("Volumetric ReSTIR System initialized successfully");
    LOG_INFO("  Memory: Reservoirs = {:.2f} MB", (width * height * sizeof(VolumetricReservoir) * 2) / (1024.0f * 1024.0f));

    return true;
}

/**
 * Create ping-pong reservoir buffers
 *
 * Memory: 2 × (width × height × 64 bytes)
 * Example: 1920×1080 @ 64 bytes = 264 MB total
 */
bool VolumetricReSTIRSystem::CreateReservoirBuffers() {
    const uint64_t reservoirCount = static_cast<uint64_t>(m_width) * m_height;
    const uint64_t bufferSize = reservoirCount * sizeof(VolumetricReservoir);

    LOG_INFO("Creating reservoir buffers:");
    LOG_INFO("  Count: {} reservoirs", reservoirCount);
    LOG_INFO("  Size per buffer: {:.2f} MB", bufferSize / (1024.0f * 1024.0f));

    for (int i = 0; i < 2; i++) {
        // Buffer description
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = bufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // Heap properties (default heap for GPU access)
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        // Create buffer
        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer[i])
        );

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create reservoir buffer {}: HRESULT 0x{:08X}", i, static_cast<uint32_t>(hr));
            return false;
        }

        // Set debug name
        std::wstring name = L"VolumetricReSTIR_ReservoirBuffer_" + std::to_wstring(i);
        m_reservoirBuffer[i]->SetName(name.c_str());

        // Create SRV (Structured Buffer<VolumetricReservoir>)
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(reservoirCount);
        srvDesc.Buffer.StructureByteStride = sizeof(VolumetricReservoir);

        m_reservoirSRV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateShaderResourceView(
            m_reservoirBuffer[i].Get(),
            &srvDesc,
            m_reservoirSRV[i]
        );

        // Create UAV (RWStructuredBuffer<VolumetricReservoir>)
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(reservoirCount);
        uavDesc.Buffer.StructureByteStride = sizeof(VolumetricReservoir);

        m_reservoirUAV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateUnorderedAccessView(
            m_reservoirBuffer[i].Get(),
            nullptr,
            &uavDesc,
            m_reservoirUAV[i]
        );
    }

    LOG_INFO("Reservoir buffers created successfully");
    return true;
}

/**
 * Create piecewise-constant volume (Mip 2) for T* transmittance
 *
 * This is a 3D texture storing approximate transmittance for candidate generation.
 * Resolution: Typically 64×64×64 or 128×128×128
 * Format: R16_FLOAT (sufficient for [0,1] transmittance values)
 *
 * Memory: 64³ × 2 bytes = 512 KB
 */
bool VolumetricReSTIRSystem::CreatePiecewiseConstantVolume() {
    // Mip 2 resolution (coarse grid for cheap T* lookups)
    const uint32_t volumeSize = 64; // 64×64×64 grid

    LOG_INFO("Creating piecewise-constant volume (Mip 2):");
    LOG_INFO("  Resolution: {}³", volumeSize);

    // 3D texture description
    D3D12_RESOURCE_DESC volumeDesc = {};
    volumeDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    volumeDesc.Width = volumeSize;
    volumeDesc.Height = volumeSize;
    volumeDesc.DepthOrArraySize = volumeSize;
    volumeDesc.MipLevels = 1;
    volumeDesc.Format = DXGI_FORMAT_R16_FLOAT;
    volumeDesc.SampleDesc.Count = 1;
    volumeDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    volumeDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Heap properties
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    // Create 3D texture
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &volumeDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_volumeMip2)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Mip 2 volume: HRESULT 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_volumeMip2->SetName(L"VolumetricReSTIR_VolumeMip2");

    // Create SRV (Texture3D<float>)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture3D.MipLevels = 1;
    srvDesc.Texture3D.MostDetailedMip = 0;

    m_volumeMip2SRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_volumeMip2.Get(),
        &srvDesc,
        m_volumeMip2SRV
    );

    // Pre-compute GPU handle to avoid GetGPUHandle() call during rendering
    m_volumeMip2SRV_GPU = m_resources->GetGPUHandle(m_volumeMip2SRV);
    LOG_INFO("Volume Mip 2 GPU handle: 0x{:016X}", m_volumeMip2SRV_GPU.ptr);

    const float memoryMB = (volumeSize * volumeSize * volumeSize * 2) / (1024.0f * 1024.0f);
    LOG_INFO("Mip 2 volume created successfully ({:.2f} MB)", memoryMB);

    // Allocate descriptor table for shading pass (2 contiguous descriptors)
    // [0]: Reservoir SRV (t2)
    // [1]: Output texture UAV (u0) - will be filled later
    m_shadingTableStart = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE secondDescriptor = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_shadingTableGPU = m_resources->GetGPUHandle(m_shadingTableStart);

    // Verify descriptors are contiguous
    UINT descriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (secondDescriptor.ptr != m_shadingTableStart.ptr + descriptorSize) {
        LOG_WARN("Shading table descriptors are not contiguous! This may cause issues.");
    }

    // Copy reservoir SRV to table[0] - this is permanent
    m_device->GetDevice()->CopyDescriptorsSimple(1, m_shadingTableStart, m_reservoirSRV[0],
                                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Note: table[1] (output UAV) will be set dynamically in ShadeSelectedPaths()
    LOG_INFO("Shading descriptor table allocated (2 descriptors)");

    // Create constant buffer for path generation (256 bytes aligned)
    D3D12_HEAP_PROPERTIES uploadHeapProps = {};
    uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC cbDesc = {};
    cbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cbDesc.Width = 256; // PathGenerationConstants size (aligned to 256 bytes)
    cbDesc.Height = 1;
    cbDesc.DepthOrArraySize = 1;
    cbDesc.MipLevels = 1;
    cbDesc.Format = DXGI_FORMAT_UNKNOWN;
    cbDesc.SampleDesc.Count = 1;
    cbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    cbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &cbDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_pathGenConstantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create path generation constant buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_pathGenConstantBuffer->SetName(L"PathGeneration Constants");
    LOG_INFO("Path generation constant buffer created (256 bytes)");

    // Create constant buffer for shading (256 bytes aligned)
    hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &cbDesc,  // Same size as path gen constants (256 bytes)
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_shadingConstantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create shading constant buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_shadingConstantBuffer->SetName(L"Shading Constants");
    LOG_INFO("Shading constant buffer created (256 bytes)");

    return true;
}

/**
 * Create compute pipelines for path generation and shading
 *
 * Phase 1 pipelines:
 * - Path generation (RIS candidate sampling)
 * - Final shading (path evaluation)
 *
 * Phase 2-3 pipelines (future):
 * - Spatial reuse
 * - Temporal reuse
 */
bool VolumetricReSTIRSystem::CreatePipelines() {
    LOG_INFO("Creating Volumetric ReSTIR compute pipelines...");

    auto d3dDevice = m_device->GetDevice();
    HRESULT hr;

    // ===================================================================
    // Path Generation Pipeline
    // ===================================================================

    // Create root signature for path generation
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[5];

        // b0: PathGenerationConstants (constant buffer descriptor instead of root constants)
        // This saves root signature space: CBV = 2 DWORDs vs 64 DWORDs for inline constants
        rootParams[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t0: Particle BLAS (SRV, acceleration structure)
        rootParams[1].InitAsShaderResourceView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t1: Particle buffer (SRV, structured buffer)
        rootParams[2].InitAsShaderResourceView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t2: Volume Mip 2 (descriptor table for Texture3D + sampler)
        CD3DX12_DESCRIPTOR_RANGE1 srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        rootParams[3].InitAsDescriptorTable(1, &srvRange, D3D12_SHADER_VISIBILITY_ALL);

        // u0: Reservoir buffer (UAV, structured buffer)
        rootParams[4].InitAsUnorderedAccessView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // Static sampler for volume texture
        D3D12_STATIC_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MipLODBias = 0.0f;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderRegister = 0; // s0
        sampler.RegisterSpace = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(_countof(rootParams), rootParams, 1, &sampler,
                            D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;

        hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Path generation root signature serialization failed: {}",
                         static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }

        hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                           IID_PPV_ARGS(&m_pathGenerationRS));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create path generation root signature: 0x{:08X}", static_cast<uint32_t>(hr));
            return false;
        }

        m_pathGenerationRS->SetName(L"VolumetricReSTIR PathGeneration RS");
        LOG_INFO("Path generation root signature created");
    }

    // Load path generation shader and create PSO
    {
        std::ifstream shaderFile("shaders/volumetric_restir/path_generation.dxil", std::ios::binary);
        if (!shaderFile.is_open()) {
            LOG_ERROR("Failed to load path_generation.dxil");
            LOG_ERROR("  Shader must be compiled first!");
            return false;
        }

        std::vector<uint8_t> shaderBytecode((std::istreambuf_iterator<char>(shaderFile)),
                                           std::istreambuf_iterator<char>());
        shaderFile.close();
        LOG_INFO("Loaded path generation shader: {} bytes", shaderBytecode.size());

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_pathGenerationRS.Get();
        psoDesc.CS.pShaderBytecode = shaderBytecode.data();
        psoDesc.CS.BytecodeLength = shaderBytecode.size();

        hr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pathGenerationPSO));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create path generation PSO: 0x{:08X}", static_cast<uint32_t>(hr));
            return false;
        }

        m_pathGenerationPSO->SetName(L"VolumetricReSTIR PathGeneration PSO");
        LOG_INFO("Path generation PSO created");
    }

    // ===================================================================
    // Shading Pipeline
    // ===================================================================

    // Create root signature for shading
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[4];

        // b0: ShadingConstants (constant buffer descriptor)
        // This saves root signature space: CBV = 2 DWORDs vs 64 DWORDs for inline constants
        rootParams[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t0: Particle BLAS (SRV, acceleration structure)
        rootParams[1].InitAsShaderResourceView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t1: Particle buffer (SRV, structured buffer)
        rootParams[2].InitAsShaderResourceView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t2: Reservoir buffer (SRV, structured buffer)
        // u0: Output texture (UAV, RWTexture2D)
        CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        rootParams[3].InitAsDescriptorTable(2, ranges, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(_countof(rootParams), rootParams, 0, nullptr,
                            D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;

        hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Shading root signature serialization failed: {}",
                         static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }

        hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                           IID_PPV_ARGS(&m_shadingRS));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create shading root signature: 0x{:08X}", static_cast<uint32_t>(hr));
            return false;
        }

        m_shadingRS->SetName(L"VolumetricReSTIR Shading RS");
        LOG_INFO("Shading root signature created");
    }

    // Load shading shader and create PSO
    {
        std::ifstream shaderFile("shaders/volumetric_restir/shading.dxil", std::ios::binary);
        if (!shaderFile.is_open()) {
            LOG_ERROR("Failed to load shading.dxil");
            LOG_ERROR("  Shader must be compiled first!");
            return false;
        }

        std::vector<uint8_t> shaderBytecode((std::istreambuf_iterator<char>(shaderFile)),
                                           std::istreambuf_iterator<char>());
        shaderFile.close();
        LOG_INFO("Loaded shading shader: {} bytes", shaderBytecode.size());

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_shadingRS.Get();
        psoDesc.CS.pShaderBytecode = shaderBytecode.data();
        psoDesc.CS.BytecodeLength = shaderBytecode.size();

        hr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_shadingPSO));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create shading PSO: 0x{:08X}", static_cast<uint32_t>(hr));
            return false;
        }

        m_shadingPSO->SetName(L"VolumetricReSTIR Shading PSO");
        LOG_INFO("Shading PSO created");
    }

    LOG_INFO("Volumetric ReSTIR pipelines created successfully");
    return true;
}

/**
 * Generate candidate paths and perform RIS (Phase 1)
 *
 * For each pixel:
 * 1. Generate M random walks through the volume
 * 2. Use regular tracking (piecewise-constant volume, Mip 2)
 * 3. Perform weighted reservoir sampling (select 1 path)
 * 4. Store winner in reservoir buffer
 */
void VolumetricReSTIRSystem::GenerateCandidates(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBVH,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    const DirectX::XMFLOAT3& cameraPos,
    const DirectX::XMMATRIX& viewMatrix,
    const DirectX::XMMATRIX& projMatrix,
    uint32_t frameIndex)
{
    if (!m_initialized) {
        LOG_ERROR("VolumetricReSTIRSystem not initialized");
        return;
    }

    // CRITICAL: Check for null TLAS (would cause GPU hang)
    if (!particleBVH) {
        LOG_ERROR("VolumetricReSTIR: particleBVH is null - cannot generate candidates");
        LOG_ERROR("  Make sure RT lighting system is initialized");
        return;
    }

    if (!particleBuffer) {
        LOG_ERROR("VolumetricReSTIR: particleBuffer is null - cannot generate candidates");
        return;
    }

    // PIX event marker
    // TODO: Re-enable after fixing PIX includes
    // PIXBeginEvent(commandList, PIX_COLOR_INDEX(10), "VolumetricReSTIR Path Generation");

    // CRITICAL: Set descriptor heaps (required for descriptor table bindings)
    ID3D12DescriptorHeap* heaps[] = { m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
    commandList->SetDescriptorHeaps(1, heaps);

    // Set pipeline state
    commandList->SetPipelineState(m_pathGenerationPSO.Get());
    commandList->SetComputeRootSignature(m_pathGenerationRS.Get());

    // Prepare constants
    PathGenerationConstants constants = {};
    constants.screenWidth = m_width;
    constants.screenHeight = m_height;
    constants.particleCount = particleCount;
    constants.randomWalksPerPixel = m_randomWalksPerPixel;
    constants.maxBounces = m_maxBounces;
    constants.frameIndex = frameIndex;
    constants.cameraPos = cameraPos;

    // Convert matrices to XMFLOAT4X4 for constant buffer upload
    DirectX::XMStoreFloat4x4(&constants.viewMatrix, viewMatrix);
    DirectX::XMStoreFloat4x4(&constants.projMatrix, projMatrix);

    // Compute inverse view-projection matrix
    DirectX::XMMATRIX viewProj = DirectX::XMMatrixMultiply(viewMatrix, projMatrix);
    DirectX::XMMATRIX invViewProj = DirectX::XMMatrixInverse(nullptr, viewProj);
    DirectX::XMStoreFloat4x4(&constants.invViewProjMatrix, invViewProj);

    // Upload constants to constant buffer
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, 0 }; // Not reading, only writing
    m_pathGenConstantBuffer->Map(0, &readRange, &mappedData);
    memcpy(mappedData, &constants, sizeof(PathGenerationConstants));
    m_pathGenConstantBuffer->Unmap(0, nullptr);

    // Bind resources
    // Root parameter 0: Constant buffer
    commandList->SetComputeRootConstantBufferView(0, m_pathGenConstantBuffer->GetGPUVirtualAddress());

    // Root parameter 1: t0 - Particle BLAS (SRV)
    commandList->SetComputeRootShaderResourceView(1, particleBVH->GetGPUVirtualAddress());

    // Root parameter 2: t1 - Particle buffer (SRV)
    commandList->SetComputeRootShaderResourceView(2, particleBuffer->GetGPUVirtualAddress());

    // Root parameter 3: t2 - Volume Mip 2 (descriptor table)
    commandList->SetComputeRootDescriptorTable(3, m_volumeMip2SRV_GPU);

    // Root parameter 4: u0 - Reservoir buffer (UAV)
    commandList->SetComputeRootUnorderedAccessView(4, m_reservoirBuffer[m_currentBufferIndex]->GetGPUVirtualAddress());

    // Dispatch compute shader
    uint32_t dispatchX = (m_width + 7) / 8;   // 8×8 thread groups
    uint32_t dispatchY = (m_height + 7) / 8;

    // Log first dispatch for debugging
    static bool loggedFirstDispatch = false;
    if (!loggedFirstDispatch) {
        LOG_INFO("VolumetricReSTIR GenerateCandidates first dispatch:");
        LOG_INFO("  Resolution: {}x{}", m_width, m_height);
        LOG_INFO("  Dispatch: {}x{} thread groups ({}x{} threads)",
                 dispatchX, dispatchY, dispatchX * 8, dispatchY * 8);
        LOG_INFO("  Reservoirs: {} (should match thread count)",
                 m_width * m_height);
        loggedFirstDispatch = true;
    }

    commandList->Dispatch(dispatchX, dispatchY, 1);

    // UAV barrier (ensure writes complete before shading)
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = m_reservoirBuffer[m_currentBufferIndex].Get();
    commandList->ResourceBarrier(1, &barrier);

    // TODO: Re-enable after fixing PIX includes
    // PIXEndEvent(commandList);

    m_frameCount++;
}

/**
 * Perform spatial reuse (Phase 2 - not yet implemented)
 */
void VolumetricReSTIRSystem::SpatialReuse(ID3D12GraphicsCommandList* commandList) {
    // Phase 2 implementation pending
}

/**
 * Perform temporal reuse (Phase 3 - not yet implemented)
 */
void VolumetricReSTIRSystem::TemporalReuse(ID3D12GraphicsCommandList* commandList, const DirectX::XMFLOAT3& cameraPos) {
    // Phase 3 implementation pending
}

/**
 * Shade the selected paths (final rendering)
 *
 * For each pixel:
 * 1. Read reservoir winner
 * 2. Evaluate path contribution f̂(λ)/p̂(λ)
 * 3. Apply transmittance T(x₀→x₁) for visibility
 * 4. Write final color to output texture
 */
void VolumetricReSTIRSystem::ShadeSelectedPaths(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* outputTexture,
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAV,
    ID3D12Resource* particleBVH,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    const DirectX::XMFLOAT3& cameraPos,
    const DirectX::XMMATRIX& viewMatrix,
    const DirectX::XMMATRIX& projMatrix)
{
    if (!m_initialized) {
        LOG_ERROR("VolumetricReSTIRSystem not initialized");
        return;
    }

    // Log first call to verify it's being called
    static bool loggedFirstCall = false;
    if (!loggedFirstCall) {
        LOG_INFO("VolumetricReSTIR: ShadeSelectedPaths called for first time");
        loggedFirstCall = true;
    }

    // PIX event marker
    // TODO: Re-enable after fixing PIX includes
    // PIXBeginEvent(commandList, PIX_COLOR_INDEX(11), "VolumetricReSTIR Shading");

    // Transition reservoir buffer to SRV state (read-only)
    D3D12_RESOURCE_BARRIER barrierToSRV = {};
    barrierToSRV.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierToSRV.Transition.pResource = m_reservoirBuffer[m_currentBufferIndex].Get();
    barrierToSRV.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrierToSRV.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrierToSRV.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &barrierToSRV);

    // CRITICAL: Set descriptor heaps (required for descriptor table bindings)
    ID3D12DescriptorHeap* heaps[] = { m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
    commandList->SetDescriptorHeaps(1, heaps);

    // Set pipeline state
    commandList->SetPipelineState(m_shadingPSO.Get());
    commandList->SetComputeRootSignature(m_shadingRS.Get());

    // Prepare constants with data from caller
    struct ShadingConstants {
        uint32_t screenWidth;
        uint32_t screenHeight;
        uint32_t particleCount;
        uint32_t padding0;
        DirectX::XMFLOAT3 cameraPos;
        float padding1;
        DirectX::XMFLOAT4X4 viewMatrix;
        DirectX::XMFLOAT4X4 projMatrix;
        DirectX::XMFLOAT4X4 invViewProjMatrix;
    };

    ShadingConstants constants = {};
    constants.screenWidth = m_width;
    constants.screenHeight = m_height;
    constants.particleCount = particleCount;
    constants.cameraPos = cameraPos;
    DirectX::XMStoreFloat4x4(&constants.viewMatrix, viewMatrix);
    DirectX::XMStoreFloat4x4(&constants.projMatrix, projMatrix);

    // Compute inverse view-projection for ray reconstruction
    DirectX::XMMATRIX viewProj = viewMatrix * projMatrix;
    DirectX::XMMATRIX invViewProj = DirectX::XMMatrixInverse(nullptr, viewProj);
    DirectX::XMStoreFloat4x4(&constants.invViewProjMatrix, invViewProj);

    // Upload constants to constant buffer
    void* shadingMappedData = nullptr;
    D3D12_RANGE shadingReadRange = { 0, 0 }; // Not reading, only writing
    m_shadingConstantBuffer->Map(0, &shadingReadRange, &shadingMappedData);
    memcpy(shadingMappedData, &constants, sizeof(ShadingConstants));
    m_shadingConstantBuffer->Unmap(0, nullptr);

    // Bind resources
    // Root parameter 0: Constant buffer
    commandList->SetComputeRootConstantBufferView(0, m_shadingConstantBuffer->GetGPUVirtualAddress());

    // Root parameter 1: t0 - Particle BLAS (SRV)
    if (particleBVH) {
        commandList->SetComputeRootShaderResourceView(1, particleBVH->GetGPUVirtualAddress());
    }

    // Root parameter 2: t1 - Particle buffer (SRV)
    commandList->SetComputeRootShaderResourceView(2, particleBuffer->GetGPUVirtualAddress());

    // Root parameter 3: Descriptor table (t2: reservoir SRV, u0: output texture UAV)
    // Update output UAV in descriptor table slot [1]
    UINT descriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE outputUAVSlot = m_shadingTableStart;
    outputUAVSlot.ptr += descriptorSize;

    // Create UAV descriptor for output texture at table[1]
    D3D12_UNORDERED_ACCESS_VIEW_DESC outUAVDesc = {};
    outUAVDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    outUAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    outUAVDesc.Texture2D.MipSlice = 0;
    outUAVDesc.Texture2D.PlaneSlice = 0;
    m_device->GetDevice()->CreateUnorderedAccessView(
        outputTexture,
        nullptr,
        &outUAVDesc,
        outputUAVSlot
    );

    // Bind the descriptor table (starts at table[0], includes both SRV and UAV)
    commandList->SetComputeRootDescriptorTable(3, m_shadingTableGPU);

    // Dispatch compute shader
    uint32_t dispatchX = (m_width + 7) / 8;   // 8×8 thread groups
    uint32_t dispatchY = (m_height + 7) / 8;

    // DEBUG: Log dispatch parameters on first call
    static bool loggedDispatch = false;
    if (!loggedDispatch) {
        LOG_INFO("VolumetricReSTIR ShadeSelectedPaths dispatch: {}x{} thread groups ({}x{} pixels, {}x{} threads total)",
                 dispatchX, dispatchY, m_width, m_height, dispatchX * 8, dispatchY * 8);
        LOG_INFO("Output texture resource: 0x{:016X}", reinterpret_cast<uintptr_t>(outputTexture));
        loggedDispatch = true;
    }

    commandList->Dispatch(dispatchX, dispatchY, 1);

    // CRITICAL: UAV barrier to ensure output texture writes are visible
    D3D12_RESOURCE_BARRIER barriers[2] = {};

    // Barrier 1: UAV barrier for output texture (ensure writes complete)
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers[0].UAV.pResource = outputTexture;

    // Barrier 2: Transition reservoir buffer back to UAV state
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = m_reservoirBuffer[m_currentBufferIndex].Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    commandList->ResourceBarrier(2, barriers);

    // TODO: Re-enable after fixing PIX includes
    // PIXEndEvent(commandList);
}
