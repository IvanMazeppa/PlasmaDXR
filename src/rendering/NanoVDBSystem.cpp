#include "NanoVDBSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"

#include <d3dcompiler.h>  // For D3DReadFileToBlob
#include <filesystem>     // For directory iteration (animation loading)
#include <algorithm>      // For std::sort
#include <fstream>        // For file reading

// NanoVDB file I/O
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/io/IO.h>

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
    m_hasFileGrid = false;
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

    // Check file extension - we only support .nvdb (NanoVDB native format)
    // OpenVDB (.vdb) files need to be converted first
    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    if (ext == "vdb") {
        LOG_ERROR("[NanoVDB] OpenVDB (.vdb) files not directly supported!");
        LOG_ERROR("[NanoVDB] Please convert to .nvdb format first:");
        LOG_ERROR("[NanoVDB]   - Use nanovdb_convert tool (build from external/nanovdb)");
        LOG_ERROR("[NanoVDB]   - Or use Blender's OpenVDB export with NanoVDB option");
        return false;
    }

    try {
        // Read the NanoVDB file into a GridHandle
        // This loads the entire grid into host memory
        LOG_INFO("[NanoVDB] Reading NanoVDB file...");
        nanovdb::GridHandle<nanovdb::HostBuffer> handle =
            nanovdb::io::readGrid<nanovdb::HostBuffer>(filepath, 0, 1);  // grid index 0, verbose=1

        if (!handle) {
            LOG_ERROR("[NanoVDB] Failed to read grid from file (handle is empty)");
            return false;
        }

        // Get buffer info
        uint64_t bufferSize = handle.bufferSize();
        const void* bufferData = handle.data();

        LOG_INFO("[NanoVDB] Grid loaded: {} bytes ({:.2f} MB)",
                 bufferSize, bufferSize / (1024.0 * 1024.0));

        // Try to access as float grid (most common for density fields)
        const nanovdb::NanoGrid<float>* floatGrid = handle.grid<float>();
        if (floatGrid) {
            LOG_INFO("[NanoVDB] Grid type: Float (density field)");
            LOG_INFO("[NanoVDB] Grid name: {}", floatGrid->gridName());

            // Get world bounding box
            auto worldBBox = floatGrid->worldBBox();
            m_gridWorldMin = {
                static_cast<float>(worldBBox.min()[0]),
                static_cast<float>(worldBBox.min()[1]),
                static_cast<float>(worldBBox.min()[2])
            };
            m_gridWorldMax = {
                static_cast<float>(worldBBox.max()[0]),
                static_cast<float>(worldBBox.max()[1]),
                static_cast<float>(worldBBox.max()[2])
            };

            // Get voxel size
            float voxelSize = static_cast<float>(floatGrid->voxelSize()[0]);
            LOG_INFO("[NanoVDB] Voxel size: {:.4f}", voxelSize);

            // Get active voxel count
            uint64_t activeVoxels = floatGrid->activeVoxelCount();
            m_voxelCount = static_cast<uint32_t>(std::min(activeVoxels, static_cast<uint64_t>(UINT32_MAX)));
            LOG_INFO("[NanoVDB] Active voxels: {}", activeVoxels);

            LOG_INFO("[NanoVDB] World bounds: ({:.1f}, {:.1f}, {:.1f}) to ({:.1f}, {:.1f}, {:.1f})",
                     m_gridWorldMin.x, m_gridWorldMin.y, m_gridWorldMin.z,
                     m_gridWorldMax.x, m_gridWorldMax.y, m_gridWorldMax.z);
        } else {
            // Try other grid types (Vec3f for velocity, FogVolume, etc.)
            LOG_WARN("[NanoVDB] Grid is not a float type. Attempting generic access...");

            // Get base grid data for bounds
            const nanovdb::GridData* gridData = reinterpret_cast<const nanovdb::GridData*>(bufferData);
            if (gridData) {
                auto worldBBox = gridData->mWorldBBox;
                m_gridWorldMin = {
                    static_cast<float>(worldBBox.mCoord[0][0]),
                    static_cast<float>(worldBBox.mCoord[0][1]),
                    static_cast<float>(worldBBox.mCoord[0][2])
                };
                m_gridWorldMax = {
                    static_cast<float>(worldBBox.mCoord[1][0]),
                    static_cast<float>(worldBBox.mCoord[1][1]),
                    static_cast<float>(worldBBox.mCoord[1][2])
                };
                LOG_INFO("[NanoVDB] Grid type: {}", static_cast<int>(gridData->mGridType));
            }
        }

        // Upload to GPU
        if (!UploadGridToGPU(bufferData, bufferSize)) {
            LOG_ERROR("[NanoVDB] Failed to upload grid to GPU");
            return false;
        }

        // Calculate and store original grid center (needed for repositioning)
        m_originalGridCenter = {
            (m_gridWorldMin.x + m_gridWorldMax.x) * 0.5f,
            (m_gridWorldMin.y + m_gridWorldMax.y) * 0.5f,
            (m_gridWorldMin.z + m_gridWorldMax.z) * 0.5f
        };

        // Set procedural sphere center to grid center for fallback
        m_sphereCenter = m_originalGridCenter;
        float dx = m_gridWorldMax.x - m_gridWorldMin.x;
        float dy = m_gridWorldMax.y - m_gridWorldMin.y;
        float dz = m_gridWorldMax.z - m_gridWorldMin.z;
        m_sphereRadius = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);

        // Reset grid offset since this is a fresh load
        m_gridOffset = { 0.0f, 0.0f, 0.0f };

        m_hasGrid = true;
        m_hasFileGrid = true;  // Mark as file-loaded grid
        m_gridSizeBytes = bufferSize;

        LOG_INFO("[NanoVDB] Grid loaded and uploaded successfully!");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("[NanoVDB] Failed to load grid: {}", e.what());
        return false;
    }
}

bool NanoVDBSystem::UploadGridToGPU(const void* gridData, uint64_t sizeBytes) {
    if (!gridData || sizeBytes == 0) {
        LOG_ERROR("[NanoVDB] UploadGridToGPU: Invalid grid data");
        return false;
    }

    LOG_INFO("[NanoVDB] Uploading {} bytes ({:.2f} MB) to GPU...",
             sizeBytes, sizeBytes / (1024.0 * 1024.0));

    // Release any existing grid buffer
    m_gridBuffer.Reset();

    // Create a default heap buffer for the grid data
    // We use a ByteAddressBuffer (raw buffer) for flexible access
    D3D12_HEAP_PROPERTIES defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,  // Initial state for upload
        nullptr,
        IID_PPV_ARGS(&m_gridBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create GPU buffer for grid data (HRESULT: 0x{:08X})", hr);
        return false;
    }

    m_gridBuffer->SetName(L"NanoVDB Grid Buffer");

    // Create an upload buffer to transfer data to GPU
    ComPtr<ID3D12Resource> uploadBuffer;
    D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create upload buffer (HRESULT: 0x{:08X})", hr);
        m_gridBuffer.Reset();
        return false;
    }

    // Map the upload buffer and copy data
    void* mappedData = nullptr;
    hr = uploadBuffer->Map(0, nullptr, &mappedData);
    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to map upload buffer (HRESULT: 0x{:08X})", hr);
        m_gridBuffer.Reset();
        return false;
    }

    memcpy(mappedData, gridData, sizeBytes);
    uploadBuffer->Unmap(0, nullptr);

    // Create a command list to perform the copy
    ComPtr<ID3D12CommandAllocator> cmdAllocator;
    ComPtr<ID3D12GraphicsCommandList> cmdList;

    hr = m_device->GetDevice()->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&cmdAllocator)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create command allocator for upload");
        m_gridBuffer.Reset();
        return false;
    }

    hr = m_device->GetDevice()->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        cmdAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&cmdList)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create command list for upload");
        m_gridBuffer.Reset();
        return false;
    }

    // Copy from upload buffer to default buffer
    cmdList->CopyBufferRegion(m_gridBuffer.Get(), 0, uploadBuffer.Get(), 0, sizeBytes);

    // Transition to shader resource state
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_gridBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &barrier);

    // Close and execute
    cmdList->Close();

    ID3D12CommandList* cmdLists[] = { cmdList.Get() };
    m_device->GetCommandQueue()->ExecuteCommandLists(1, cmdLists);

    // Wait for upload to complete
    // Create a fence to synchronize
    ComPtr<ID3D12Fence> fence;
    hr = m_device->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create fence for upload sync");
        // Continue anyway - buffer might still work
    } else {
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        m_device->GetCommandQueue()->Signal(fence.Get(), 1);
        if (fence->GetCompletedValue() < 1) {
            fence->SetEventOnCompletion(1, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        CloseHandle(fenceEvent);
    }

    // Create SRV for the grid buffer as StructuredBuffer<uint>
    // PNanoVDB.h expects StructuredBuffer<uint> (pnanovdb_buf_t) in HLSL mode
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;  // Structured buffers use UNKNOWN format
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = static_cast<UINT>(sizeBytes / 4);  // Number of uint32 elements
    srvDesc.Buffer.StructureByteStride = sizeof(uint32_t);  // 4 bytes per element
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;  // Not a raw buffer

    // Allocate descriptor from ResourceManager
    m_gridBufferSRV_CPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(m_gridBuffer.Get(), &srvDesc, m_gridBufferSRV_CPU);
    m_gridBufferSRV_GPU = m_resources->GetGPUHandle(m_gridBufferSRV_CPU);

    LOG_INFO("[NanoVDB] Grid buffer uploaded successfully!");
    LOG_INFO("[NanoVDB] Buffer elements: {} uint32s", sizeBytes / 4);

    return true;
}

bool NanoVDBSystem::CreateComputePipeline() {
    LOG_INFO("[NanoVDB] Creating compute pipeline...");

    // Root signature with:
    // [0] CBV - NanoVDBConstants (b0)
    // [1] SRV table - Grid buffer (t0) - NanoVDB raw data
    // [2] SRV table - Light buffer (t1)
    // [3] SRV table - Depth buffer (t2)
    // [4] UAV table - Output texture (u0)

    CD3DX12_DESCRIPTOR_RANGE1 gridSrvRange;
    gridSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // t0: grid buffer

    CD3DX12_DESCRIPTOR_RANGE1 lightSrvRange;
    lightSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // t1: light buffer

    CD3DX12_DESCRIPTOR_RANGE1 depthSrvRange;
    depthSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // t2: depth buffer

    CD3DX12_DESCRIPTOR_RANGE1 uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: output

    CD3DX12_ROOT_PARAMETER1 rootParams[5];
    rootParams[0].InitAsConstantBufferView(0);               // b0: constants
    rootParams[1].InitAsDescriptorTable(1, &gridSrvRange);   // t0: grid
    rootParams[2].InitAsDescriptorTable(1, &lightSrvRange);  // t1: lights
    rootParams[3].InitAsDescriptorTable(1, &depthSrvRange);  // t2: depth
    rootParams[4].InitAsDescriptorTable(1, &uavRange);       // u0: output

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(5, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

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
    constants.useGridBuffer = (m_hasFileGrid && m_gridBuffer) ? 1 : 0;
    constants.gridOffset = m_gridOffset;  // Transform sampling position back to original grid space

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
    // [1] SRV table - Grid buffer (t0) - NanoVDB raw data
    // [2] SRV table - Light buffer (t1)
    // [3] SRV table - Depth buffer (t2)
    // [4] UAV table - Output texture (u0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Bind grid buffer if we have one loaded from file
    if (m_gridBuffer) {
        commandList->SetComputeRootDescriptorTable(1, m_gridBufferSRV_GPU);
    }

    commandList->SetComputeRootDescriptorTable(2, lightSRV);
    commandList->SetComputeRootDescriptorTable(3, depthSRV);
    commandList->SetComputeRootDescriptorTable(4, outputUAV);

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

// ============================================================================
// ANIMATION SUPPORT IMPLEMENTATION
// ============================================================================

bool NanoVDBSystem::LoadAnimationSequence(const std::vector<std::string>& filepaths) {
    if (filepaths.empty()) {
        LOG_ERROR("[NanoVDB] LoadAnimationSequence: No filepaths provided");
        return false;
    }

    LOG_INFO("[NanoVDB] Loading animation sequence with {} frames...", filepaths.size());

    // Clear existing animation frames
    m_animFrames.clear();
    m_animCurrentFrame = 0;
    m_animAccumulator = 0.0f;

    size_t loadedCount = 0;
    auto* d3dDevice = m_device->GetDevice();

    for (size_t i = 0; i < filepaths.size(); ++i) {
        const auto& filepath = filepaths[i];

        try {
            // Read file
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                LOG_WARN("[NanoVDB] Failed to open frame {}: {}", i, filepath);
                continue;
            }

            std::streamsize fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> fileData(fileSize);
            if (!file.read(fileData.data(), fileSize)) {
                LOG_WARN("[NanoVDB] Failed to read frame {}: {}", i, filepath);
                continue;
            }

            const void* bufferData = fileData.data();
            uint64_t bufferSize = static_cast<uint64_t>(fileSize);

            // Create GPU buffer for this frame
            AnimationFrame frame;
            frame.sizeBytes = bufferSize;

            D3D12_HEAP_PROPERTIES defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

            HRESULT hr = d3dDevice->CreateCommittedResource(
                &defaultHeapProps,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&frame.buffer));

            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create GPU buffer for frame {}", i);
                continue;
            }

            // Create upload buffer
            D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            ComPtr<ID3D12Resource> uploadBuffer;
            hr = d3dDevice->CreateCommittedResource(
                &uploadHeapProps,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer));

            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create upload buffer for frame {}", i);
                continue;
            }

            // Copy data to upload buffer
            void* mappedData = nullptr;
            uploadBuffer->Map(0, nullptr, &mappedData);
            memcpy(mappedData, bufferData, bufferSize);
            uploadBuffer->Unmap(0, nullptr);

            // Create command allocator and list for upload (same pattern as UploadGridToGPU)
            ComPtr<ID3D12CommandAllocator> cmdAllocator;
            ComPtr<ID3D12GraphicsCommandList> cmdList;

            hr = d3dDevice->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&cmdAllocator));
            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create command allocator for frame {}", i);
                continue;
            }

            hr = d3dDevice->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                cmdAllocator.Get(), nullptr,
                IID_PPV_ARGS(&cmdList));
            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create command list for frame {}", i);
                continue;
            }

            // Copy and transition
            cmdList->CopyBufferRegion(frame.buffer.Get(), 0, uploadBuffer.Get(), 0, bufferSize);

            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                frame.buffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmdList->ResourceBarrier(1, &barrier);
            cmdList->Close();

            // Execute and wait
            ID3D12CommandList* cmdLists[] = { cmdList.Get() };
            m_device->GetCommandQueue()->ExecuteCommandLists(1, cmdLists);

            // Create fence and wait for upload
            ComPtr<ID3D12Fence> fence;
            hr = d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
            if (SUCCEEDED(hr)) {
                HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                m_device->GetCommandQueue()->Signal(fence.Get(), 1);
                if (fence->GetCompletedValue() < 1) {
                    fence->SetEventOnCompletion(1, fenceEvent);
                    WaitForSingleObject(fenceEvent, INFINITE);
                }
                CloseHandle(fenceEvent);
            }

            // Create SRV for this frame (StructuredBuffer<uint> like existing code)
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Buffer.FirstElement = 0;
            srvDesc.Buffer.NumElements = static_cast<UINT>(bufferSize / sizeof(uint32_t));
            srvDesc.Buffer.StructureByteStride = sizeof(uint32_t);
            srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

            frame.srvCPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            d3dDevice->CreateShaderResourceView(frame.buffer.Get(), &srvDesc, frame.srvCPU);
            frame.srvGPU = m_resources->GetGPUHandle(frame.srvCPU);

            m_animFrames.push_back(std::move(frame));
            loadedCount++;

            if (loadedCount % 5 == 0 || i == filepaths.size() - 1) {
                LOG_INFO("[NanoVDB] Loaded {}/{} frames...", loadedCount, filepaths.size());
            }

        } catch (const std::exception& e) {
            LOG_WARN("[NanoVDB] Exception loading frame {}: {}", i, e.what());
            continue;
        }
    }

    if (loadedCount > 0) {
        LOG_INFO("[NanoVDB] Animation loaded: {} frames ({:.1f} MB total)",
                 loadedCount,
                 m_animFrames.size() > 0 ?
                     (m_animFrames[0].sizeBytes * loadedCount) / (1024.0 * 1024.0) : 0.0);

        // Use first frame as current grid
        if (!m_animFrames.empty()) {
            m_gridBuffer = m_animFrames[0].buffer;
            m_gridBufferSRV_GPU = m_animFrames[0].srvGPU;
            m_gridBufferSRV_CPU = m_animFrames[0].srvCPU;
            m_gridSizeBytes = m_animFrames[0].sizeBytes;
            m_hasGrid = true;
            m_hasFileGrid = true;

            // Extract grid bounds from the first frame using NanoVDB
            try {
                nanovdb::GridHandle<nanovdb::HostBuffer> handle =
                    nanovdb::io::readGrid<nanovdb::HostBuffer>(filepaths[0], 0, 0);
                if (handle) {
                    const nanovdb::NanoGrid<float>* floatGrid = handle.grid<float>();
                    if (floatGrid) {
                        auto worldBBox = floatGrid->worldBBox();
                        m_gridWorldMin = {
                            static_cast<float>(worldBBox.min()[0]),
                            static_cast<float>(worldBBox.min()[1]),
                            static_cast<float>(worldBBox.min()[2])
                        };
                        m_gridWorldMax = {
                            static_cast<float>(worldBBox.max()[0]),
                            static_cast<float>(worldBBox.max()[1]),
                            static_cast<float>(worldBBox.max()[2])
                        };
                        // Store original center for repositioning
                        m_originalGridCenter = {
                            (m_gridWorldMin.x + m_gridWorldMax.x) * 0.5f,
                            (m_gridWorldMin.y + m_gridWorldMax.y) * 0.5f,
                            (m_gridWorldMin.z + m_gridWorldMax.z) * 0.5f
                        };
                        m_gridOffset = { 0.0f, 0.0f, 0.0f };
                        LOG_INFO("[NanoVDB] Animation bounds: ({:.2f},{:.2f},{:.2f}) to ({:.2f},{:.2f},{:.2f})",
                                 m_gridWorldMin.x, m_gridWorldMin.y, m_gridWorldMin.z,
                                 m_gridWorldMax.x, m_gridWorldMax.y, m_gridWorldMax.z);
                    }
                }
            } catch (const std::exception& e) {
                LOG_WARN("[NanoVDB] Could not extract animation bounds: {}", e.what());
            }
        }
        return true;
    }

    LOG_ERROR("[NanoVDB] Failed to load any animation frames");
    return false;
}

size_t NanoVDBSystem::LoadAnimationFromDirectory(const std::string& directory, const std::string& pattern) {
    namespace fs = std::filesystem;

    std::vector<std::string> filepaths;

    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".nvdb") {
                    filepaths.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[NanoVDB] Failed to enumerate directory {}: {}", directory, e.what());
        return 0;
    }

    if (filepaths.empty()) {
        LOG_WARN("[NanoVDB] No .nvdb files found in {}", directory);
        return 0;
    }

    // Sort files by name (assumes numbered sequence like smoke_0001.nvdb)
    std::sort(filepaths.begin(), filepaths.end());

    LOG_INFO("[NanoVDB] Found {} .nvdb files in {}", filepaths.size(), directory);

    if (LoadAnimationSequence(filepaths)) {
        return m_animFrames.size();
    }
    return 0;
}

void NanoVDBSystem::SetAnimationFrame(size_t frame) {
    if (m_animFrames.empty()) return;

    frame = frame % m_animFrames.size();  // Clamp/wrap
    m_animCurrentFrame = frame;

    // Update current grid buffer to this frame
    m_gridBuffer = m_animFrames[frame].buffer;
    m_gridBufferSRV_GPU = m_animFrames[frame].srvGPU;
    m_gridBufferSRV_CPU = m_animFrames[frame].srvCPU;
    m_gridSizeBytes = m_animFrames[frame].sizeBytes;
}

void NanoVDBSystem::UpdateAnimation(float deltaTime) {
    if (!m_animPlaying || m_animFrames.empty() || m_animFPS <= 0.0f) {
        return;
    }

    m_animAccumulator += deltaTime;
    float frameTime = 1.0f / m_animFPS;

    while (m_animAccumulator >= frameTime) {
        m_animAccumulator -= frameTime;

        size_t nextFrame = m_animCurrentFrame + 1;

        if (nextFrame >= m_animFrames.size()) {
            if (m_animLoop) {
                nextFrame = 0;
            } else {
                m_animPlaying = false;
                return;
            }
        }

        SetAnimationFrame(nextFrame);
    }
}
