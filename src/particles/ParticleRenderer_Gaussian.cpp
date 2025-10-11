#include "ParticleRenderer_Gaussian.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <d3dcompiler.h>
#include <fstream>

ParticleRenderer_Gaussian::~ParticleRenderer_Gaussian() {
    // ComPtr cleanup automatic
}

bool ParticleRenderer_Gaussian::Initialize(Device* device,
                                           ResourceManager* resources,
                                           uint32_t particleCount,
                                           uint32_t screenWidth,
                                           uint32_t screenHeight) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;

    LOG_INFO("Initializing 3D Gaussian Splatting renderer...");
    LOG_INFO("  Particles: {}", particleCount);
    LOG_INFO("  Resolution: {}x{}", screenWidth, screenHeight);
    LOG_INFO("  Reusing existing RTLightingSystem BLAS/TLAS");

    // Create constant buffer for RenderConstants
    // Use upload heap so we can map/write every frame
    const UINT constantBufferSize = (sizeof(RenderConstants) + 255) & ~255;  // Align to 256 bytes
    D3D12_HEAP_PROPERTIES uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create constant buffer");
        return false;
    }

    // Map the constant buffer (keep it mapped for the lifetime of the buffer)
    hr = m_constantBuffer->Map(0, nullptr, &m_constantBufferMapped);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map constant buffer");
        return false;
    }

    LOG_INFO("Created constant buffer: {} bytes (aligned from {} bytes)",
             constantBufferSize, sizeof(RenderConstants));

    if (!CreateOutputTexture(screenWidth, screenHeight)) {
        return false;
    }

    // Create ReSTIR reservoir buffers (ping-pong for temporal reuse)
    // Reservoir struct: float3 lightPos (12B) + float weightSum (4B) + uint M (4B) + float W (4B) = 24 bytes
    // Pad to 32 bytes for cache alignment
    const uint32_t reservoirElementSize = 32;  // bytes per pixel
    const uint32_t reservoirBufferSize = screenWidth * screenHeight * reservoirElementSize;

    LOG_INFO("Creating ReSTIR reservoir buffers...");
    LOG_INFO("  Resolution: {}x{} pixels", screenWidth, screenHeight);
    LOG_INFO("  Element size: {} bytes", reservoirElementSize);
    LOG_INFO("  Buffer size: {} MB per buffer", reservoirBufferSize / (1024 * 1024));
    LOG_INFO("  Total memory: {} MB (2x buffers)", (reservoirBufferSize * 2) / (1024 * 1024));

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC reservoirBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
        reservoirBufferSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    for (int i = 0; i < 2; i++) {
        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &reservoirBufferDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_reservoirBuffer[i])
        );

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create reservoir buffer {}", i);
            return false;
        }

        // Create SRV (for reading previous frame)
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = screenWidth * screenHeight;
        srvDesc.Buffer.StructureByteStride = reservoirElementSize;

        m_reservoirSRV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateShaderResourceView(
            m_reservoirBuffer[i].Get(),
            &srvDesc,
            m_reservoirSRV[i]
        );
        m_reservoirSRVGPU[i] = m_resources->GetGPUHandle(m_reservoirSRV[i]);

        // Create UAV (for writing current frame)
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = screenWidth * screenHeight;
        uavDesc.Buffer.StructureByteStride = reservoirElementSize;

        m_reservoirUAV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateUnorderedAccessView(
            m_reservoirBuffer[i].Get(),
            nullptr,
            &uavDesc,
            m_reservoirUAV[i]
        );
        m_reservoirUAVGPU[i] = m_resources->GetGPUHandle(m_reservoirUAV[i]);

        LOG_INFO("Created reservoir buffer {}: SRV=0x{:016X}, UAV=0x{:016X}",
                 i, m_reservoirSRVGPU[i].ptr, m_reservoirUAVGPU[i].ptr);
    }

    if (!CreatePipeline()) {
        return false;
    }

    LOG_INFO("Gaussian Splatting renderer initialized successfully");
    return true;
}

bool ParticleRenderer_Gaussian::CreateOutputTexture(uint32_t width, uint32_t height) {
    // Create UAV texture for Gaussian rendering output
    // TODO: Use R16G16B16A16_FLOAT to prevent color banding (needs format check)
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // Reverted - R16 causing crash
    texDesc.SampleDesc.Count = 1;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_outputTexture)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian output texture");
        return false;
    }

    // Create UAV descriptor (typed UAVs require descriptor table binding)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;

    m_outputUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateUnorderedAccessView(
        m_outputTexture.Get(),
        nullptr,
        &uavDesc,
        m_outputUAV
    );

    // Get GPU handle for binding in Render()
    m_outputUAVGPU = m_resources->GetGPUHandle(m_outputUAV);

    LOG_INFO("Created Gaussian output texture: {}x{}", width, height);
    LOG_INFO("  UAV CPU handle: 0x{:016X}", m_outputUAV.ptr);
    LOG_INFO("  UAV GPU handle: 0x{:016X}", m_outputUAVGPU.ptr);

    if (m_outputUAVGPU.ptr == 0) {
        LOG_ERROR("CRITICAL: GPU handle is null!");
        return false;
    }

    return true;
}

bool ParticleRenderer_Gaussian::CreatePipeline() {
    // Load Gaussian raytrace compute shader
    std::ifstream shaderFile("shaders/particles/particle_gaussian_raytrace.dxil", std::ios::binary);
    if (!shaderFile.is_open()) {
        LOG_ERROR("Failed to load particle_gaussian_raytrace.dxil");
        LOG_ERROR("  Make sure shader is compiled!");
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> computeShader;
    HRESULT hr = D3DCreateBlob(shaderData.size(), &computeShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob");
        return false;
    }
    memcpy(computeShader->GetBufferPointer(), shaderData.data(), shaderData.size());
    LOG_INFO("Loaded Gaussian shader: {} bytes", shaderData.size());

    // Root signature matches particle_gaussian_raytrace.hlsl
    // b0: GaussianConstants (CBV - no size limit!)
    // t0: StructuredBuffer<Particle> g_particles
    // t1: Buffer<float4> g_rtLighting
    // t2: RaytracingAccelerationStructure g_particleBVH (TLAS from RTLightingSystem!)
    // t3: StructuredBuffer<Reservoir> g_prevReservoirs (previous frame, read-only)
    // u0: RWTexture2D<float4> g_output (descriptor table - typed UAV requirement)
    // u1: RWStructuredBuffer<Reservoir> g_currentReservoirs (current frame, write)
    CD3DX12_DESCRIPTOR_RANGE1 uavRanges[2];
    uavRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D
    uavRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // u1: RWStructuredBuffer

    CD3DX12_ROOT_PARAMETER1 rootParams[7];
    rootParams[0].InitAsConstantBufferView(0);             // b0 - CBV (no DWORD limit!)
    rootParams[1].InitAsShaderResourceView(0);             // t0 - particles
    rootParams[2].InitAsShaderResourceView(1);             // t1 - rtLighting
    rootParams[3].InitAsShaderResourceView(2);             // t2 - TLAS
    rootParams[4].InitAsShaderResourceView(3);             // t3 - previous reservoirs (SRV)
    rootParams[5].InitAsDescriptorTable(1, &uavRanges[0]); // u0 - output texture
    rootParams[6].InitAsDescriptorTable(1, &uavRanges[1]); // u1 - current reservoirs (UAV)

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(7, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Root signature serialization failed: {}", (char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_rootSignature));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create root signature");
        return false;
    }

    // Create compute PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian PSO");
        LOG_ERROR("  HRESULT: 0x{:08X}", static_cast<unsigned int>(hr));

        // Try to get debug messages
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_device->GetDevice()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 numMessages = infoQueue->GetNumStoredMessages();
            LOG_ERROR("  D3D12 Debug Messages ({} messages):", numMessages);
            for (UINT64 i = 0; i < numMessages && i < 10; i++) {
                SIZE_T messageLength = 0;
                infoQueue->GetMessage(i, nullptr, &messageLength);
                if (messageLength > 0) {
                    std::vector<char> messageData(messageLength);
                    D3D12_MESSAGE* message = (D3D12_MESSAGE*)messageData.data();
                    if (SUCCEEDED(infoQueue->GetMessage(i, message, &messageLength))) {
                        LOG_ERROR("    [{}]: {}", i, message->pDescription);
                    }
                }
            }
        }
        return false;
    }

    LOG_INFO("Gaussian pipeline created");
    return true;
}

void ParticleRenderer_Gaussian::Render(ID3D12GraphicsCommandList4* cmdList,
                                       ID3D12Resource* particleBuffer,
                                       ID3D12Resource* rtLightingBuffer,
                                       ID3D12Resource* tlas,
                                       const RenderConstants& constants) {
    if (!cmdList || !particleBuffer || !rtLightingBuffer || !m_resources) {
        LOG_ERROR("Gaussian Render: null resource!");
        return;
    }

    // Set descriptor heap (required for descriptor tables!)
    ID3D12DescriptorHeap* heap = m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (heap) {
        // Cast to base ID3D12GraphicsCommandList for SetDescriptorHeaps
        ID3D12GraphicsCommandList* baseList = static_cast<ID3D12GraphicsCommandList*>(cmdList);
        baseList->SetDescriptorHeaps(1, &heap);
    }

    // Set Gaussian raytrace pipeline
    cmdList->SetPipelineState(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSignature.Get());

    // Upload constants to GPU (copy to mapped constant buffer)
    memcpy(m_constantBufferMapped, &constants, sizeof(RenderConstants));

    // Bind constant buffer view (no size limit!)
    cmdList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Set resources
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, rtLightingBuffer->GetGPUVirtualAddress());

    // Set TLAS for RayQuery operations
    if (!tlas) {
        LOG_ERROR("TLAS is null! RayQuery will fail");
        return;
    }
    cmdList->SetComputeRootShaderResourceView(3, tlas->GetGPUVirtualAddress());

    // ReSTIR: Bind previous frame's reservoir (read-only SRV)
    uint32_t prevIndex = 1 - m_currentReservoirIndex;  // Ping-pong
    cmdList->SetComputeRootShaderResourceView(4, m_reservoirBuffer[prevIndex]->GetGPUVirtualAddress());

    // Bind output texture (UAV descriptor table)
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAVHandle = m_resources->GetGPUHandle(m_outputUAV);
    if (outputUAVHandle.ptr == 0) {
        LOG_ERROR("GPU handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(5, outputUAVHandle);

    // ReSTIR: Bind current frame's reservoir (write UAV descriptor table)
    D3D12_GPU_DESCRIPTOR_HANDLE currentReservoirUAVHandle = m_reservoirUAVGPU[m_currentReservoirIndex];
    if (currentReservoirUAVHandle.ptr == 0) {
        LOG_ERROR("Reservoir UAV handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(6, currentReservoirUAVHandle);

    // Dispatch (8x8 thread groups)
    uint32_t dispatchX = (constants.screenWidth + 7) / 8;
    uint32_t dispatchY = (constants.screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);

    // Add UAV barriers to ensure compute shader completes before next frame
    D3D12_RESOURCE_BARRIER uavBarriers[2] = {};
    uavBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarriers[0].UAV.pResource = m_outputTexture.Get();
    uavBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarriers[1].UAV.pResource = m_reservoirBuffer[m_currentReservoirIndex].Get();
    cmdList->ResourceBarrier(2, uavBarriers);

    // ReSTIR: Swap reservoir buffers for next frame (ping-pong)
    m_currentReservoirIndex = 1 - m_currentReservoirIndex;
}
