#include "NanoVDBSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"

#include <d3dcompiler.h>  // For D3DReadFileToBlob

using namespace DirectX;

NanoVDBSystem::~NanoVDBSystem() {
    Shutdown();
}

bool NanoVDBSystem::Initialize(Device* device, ResourceManager* resources,
                                uint32_t screenWidth, uint32_t screenHeight) {
    m_device = device;
    m_resources = resources;
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;

    LOG_INFO("[NanoVDB] Initializing volumetric rendering system...");

    // Create constant buffer (256-byte aligned for D3D12)
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(256);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create constant buffer");
        return false;
    }

    // Create compute pipeline
    if (!CreateComputePipeline()) {
        LOG_ERROR("[NanoVDB] Failed to create compute pipeline");
        return false;
    }

    LOG_INFO("[NanoVDB] System initialized successfully");
    return true;
}

void NanoVDBSystem::Shutdown() {
    m_gridBuffer.Reset();
    m_constantBuffer.Reset();
    m_rootSignature.Reset();
    m_pipelineState.Reset();
    m_hasGrid = false;
    m_enabled = false;

    LOG_INFO("[NanoVDB] System shutdown");
}

bool NanoVDBSystem::CreateFogSphere(float radius, DirectX::XMFLOAT3 center,
                                     float voxelSize, float halfWidth) {
    LOG_INFO("[NanoVDB] Creating procedural fog sphere: radius={:.1f}, center=({:.1f}, {:.1f}, {:.1f})",
             radius, center.x, center.y, center.z);

    // For the prototype, we don't actually create a NanoVDB grid
    // Instead, the shader uses procedural density calculation
    // This lets us test the rendering pipeline before integrating full NanoVDB

    // Store sphere parameters for shader
    m_sphereCenter = center;
    m_sphereRadius = radius;

    // Store grid bounds (used by shader for AABB culling)
    float boundsPadding = halfWidth * voxelSize;
    m_gridWorldMin = { center.x - radius - boundsPadding,
                       center.y - radius - boundsPadding,
                       center.z - radius - boundsPadding };
    m_gridWorldMax = { center.x + radius + boundsPadding,
                       center.y + radius + boundsPadding,
                       center.z + radius + boundsPadding };

    // Estimate voxel count for display purposes
    float gridSize = 2.0f * (radius + boundsPadding);
    int voxelsPerSide = static_cast<int>(gridSize / voxelSize);
    m_voxelCount = static_cast<uint32_t>(voxelsPerSide * voxelsPerSide * voxelsPerSide);

    // Mark as having a grid (even though it's procedural)
    m_hasGrid = true;
    m_gridSizeBytes = m_voxelCount * sizeof(float);  // Estimate

    LOG_INFO("[NanoVDB] Procedural fog sphere created!");
    LOG_INFO("  Bounds: [({:.1f}, {:.1f}, {:.1f}) - ({:.1f}, {:.1f}, {:.1f})]",
             m_gridWorldMin.x, m_gridWorldMin.y, m_gridWorldMin.z,
             m_gridWorldMax.x, m_gridWorldMax.y, m_gridWorldMax.z);
    LOG_INFO("  Estimated voxels: {} (procedural - no GPU memory used)", m_voxelCount);

    return true;
}

bool NanoVDBSystem::LoadFromFile(const std::string& filepath) {
    LOG_INFO("[NanoVDB] Loading grid from file: {}", filepath.c_str());

    // TODO: Implement file loading using nanovdb::io::readGrid
    LOG_INFO("[NanoVDB] File loading not yet implemented (using procedural density)");
    return false;
}

bool NanoVDBSystem::UploadGridToGPU(const void* gridData, uint64_t sizeBytes) {
    // For prototype, grid data is procedural - nothing to upload
    LOG_INFO("[NanoVDB] UploadGridToGPU called - procedural mode, skipping upload");
    return true;
}

bool NanoVDBSystem::CreateComputePipeline() {
    LOG_INFO("[NanoVDB] Creating compute pipeline...");

    // Root signature with:
    // [0] CBV - NanoVDBConstants (b0)
    // [1] SRV table - Light buffer (t1)
    // [2] SRV table - Depth buffer (t2)
    // [3] UAV table - Output texture (u0)

    CD3DX12_DESCRIPTOR_RANGE1 lightSrvRange;
    lightSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // t1: light buffer

    CD3DX12_DESCRIPTOR_RANGE1 depthSrvRange;
    depthSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // t2: depth buffer

    CD3DX12_DESCRIPTOR_RANGE1 uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: output

    CD3DX12_ROOT_PARAMETER1 rootParams[4];
    rootParams[0].InitAsConstantBufferView(0);               // b0: constants
    rootParams[1].InitAsDescriptorTable(1, &lightSrvRange);  // t1: lights
    rootParams[2].InitAsDescriptorTable(1, &depthSrvRange);  // t2: depth
    rootParams[3].InitAsDescriptorTable(1, &uavRange);       // u0: output

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(4, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("[NanoVDB] Root signature serialization failed: {}",
                      static_cast<const char*>(error->GetBufferPointer()));
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(
        0,
        signature->GetBufferPointer(),
        signature->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create root signature");
        return false;
    }

    // Load compute shader
    ComPtr<ID3DBlob> shaderBlob;
    hr = D3DReadFileToBlob(L"build/bin/Debug/shaders/volumetric/nanovdb_raymarch.dxil", &shaderBlob);
    if (FAILED(hr)) {
        LOG_INFO("[NanoVDB] Shader not found at build path, trying alternate...");
        hr = D3DReadFileToBlob(L"shaders/volumetric/nanovdb_raymarch.dxil", &shaderBlob);
        if (FAILED(hr)) {
            LOG_INFO("[NanoVDB] Shader not found, pipeline creation deferred");
            // This is expected on first run before shader is compiled
            return true;  // Return true to allow initialization to complete
        }
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = { shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize() };

    hr = m_device->GetDevice()->CreateComputePipelineState(
        &psoDesc,
        IID_PPV_ARGS(&m_pipelineState)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create compute pipeline state");
        return false;
    }

    LOG_INFO("[NanoVDB] Compute pipeline created successfully");
    return true;
}

void NanoVDBSystem::Render(
    ID3D12GraphicsCommandList* commandList,
    const DirectX::XMMATRIX& viewProj,
    const DirectX::XMFLOAT3& cameraPos,
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAV,
    D3D12_GPU_DESCRIPTOR_HANDLE lightSRV,
    D3D12_GPU_DESCRIPTOR_HANDLE depthSRV,
    uint32_t lightCount,
    ID3D12DescriptorHeap* descriptorHeap,
    uint32_t renderWidth,
    uint32_t renderHeight,
    float time) {

    if (!m_enabled || !m_hasGrid || !m_pipelineState) {
        return;  // Nothing to render
    }

    // Update constant buffer with actual render dimensions (important for DLSS)
    NanoVDBConstants constants = {};

    // Calculate inverse view-projection
    // NOTE: DirectXMath is row-major, HLSL with row_major keyword expects row-major
    // Do NOT transpose - store directly for row_major HLSL compatibility
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);
    XMStoreFloat4x4(&constants.invViewProj, invViewProj);

    constants.cameraPos = cameraPos;
    constants.densityScale = m_densityScale;
    constants.gridWorldMin = m_gridWorldMin;
    constants.emissionStrength = m_emissionStrength;
    constants.gridWorldMax = m_gridWorldMax;
    constants.absorptionCoeff = m_absorptionCoeff;
    constants.sphereCenter = m_sphereCenter;
    constants.scatteringCoeff = m_scatteringCoeff;
    constants.sphereRadius = m_sphereRadius;
    constants.maxRayDistance = m_maxRayDistance;
    constants.stepSize = m_stepSize;
    constants.lightCount = lightCount;
    constants.screenWidth = renderWidth;   // Use actual render dimensions
    constants.screenHeight = renderHeight; // (may differ from native due to DLSS)
    constants.time = time;                 // Animation time for procedural effects
    constants.debugMode = m_debugMode ? 1 : 0;

    // Map and update constants
    void* mappedData = nullptr;
    m_constantBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, &constants, sizeof(constants));
    m_constantBuffer->Unmap(0, nullptr);

    // Set descriptor heap (required for shader-visible descriptors)
    ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
    commandList->SetDescriptorHeaps(1, heaps);

    // Set pipeline state
    commandList->SetComputeRootSignature(m_rootSignature.Get());
    commandList->SetPipelineState(m_pipelineState.Get());

    // Bind resources:
    // [0] CBV - NanoVDBConstants (b0)
    // [1] SRV table - Light buffer (t1)
    // [2] SRV table - Depth buffer (t2)
    // [3] UAV table - Output texture (u0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    commandList->SetComputeRootDescriptorTable(1, lightSRV);
    commandList->SetComputeRootDescriptorTable(2, depthSRV);
    commandList->SetComputeRootDescriptorTable(3, outputUAV);

    // Dispatch compute shader using actual render dimensions
    uint32_t groupsX = (renderWidth + 7) / 8;
    uint32_t groupsY = (renderHeight + 7) / 8;
    commandList->Dispatch(groupsX, groupsY, 1);

    // Log first few frames for verification
    static int renderCount = 0;
    renderCount++;
    if (renderCount <= 3) {
        LOG_INFO("[NanoVDB] Rendered frame (groups: {} x {}, res: {}x{}, time: {:.2f})",
                 groupsX, groupsY, renderWidth, renderHeight, time);
    }
}
