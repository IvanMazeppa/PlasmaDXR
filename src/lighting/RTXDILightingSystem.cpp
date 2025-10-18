#include "RTXDILightingSystem.h"
#include "core/Device.h"
#include "utils/ResourceManager.h"
#include "utils/Logger.h"
#include "utils/d3dx12/d3dx12.h"

#include <d3dcompiler.h>
#include <fstream>
#include <vector>

// RTXDI SDK headers (Milestone 1 complete - we can now include these!)
// #include <Rtxdi/DI/ReSTIRDI.h>
// #include <Rtxdi/RtxdiParameters.h>

bool RTXDILightingSystem::Initialize(Device* device, ResourceManager* resources,
                                     uint32_t width, uint32_t height) {
    LOG_INFO("Initializing RTXDI Lighting System...");

    m_device = device;
    m_resources = resources;
    m_width = width;
    m_height = height;

    auto d3dDevice = m_device->GetDevice();

    // === Milestone 2: Light Grid Construction ===

    LOG_INFO("Creating RTXDI light grid...");
    LOG_INFO("  Grid dimensions: {}x{}x{} = {} cells", GRID_CELLS_X, GRID_CELLS_Y, GRID_CELLS_Z, TOTAL_GRID_CELLS);
    LOG_INFO("  Cell size: {} units", CELL_SIZE);
    LOG_INFO("  World bounds: {} to {}", WORLD_MIN, WORLD_MAX);
    LOG_INFO("  Max lights per cell: {}", MAX_LIGHTS_PER_CELL);

    // Create light grid buffer (27,000 cells × 128 bytes = 3.375 MB)
    const uint64_t lightGridBufferSize = TOTAL_GRID_CELLS * sizeof(LightGridCell);

    D3D12_RESOURCE_DESC lightGridDesc = {};
    lightGridDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightGridDesc.Width = lightGridBufferSize;
    lightGridDesc.Height = 1;
    lightGridDesc.DepthOrArraySize = 1;
    lightGridDesc.MipLevels = 1;
    lightGridDesc.Format = DXGI_FORMAT_UNKNOWN;
    lightGridDesc.SampleDesc.Count = 1;
    lightGridDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    lightGridDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = d3dDevice->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &lightGridDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_lightGridBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create light grid buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_lightGridBuffer->SetName(L"RTXDI Light Grid");
    LOG_INFO("Created light grid buffer: {:.2f} MB", lightGridBufferSize / (1024.0f * 1024.0f));

    // Create SRV for light grid (shader reads)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = TOTAL_GRID_CELLS;
    srvDesc.Buffer.StructureByteStride = sizeof(LightGridCell);

    m_lightGridSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    d3dDevice->CreateShaderResourceView(m_lightGridBuffer.Get(), &srvDesc, m_lightGridSRV);

    // Create UAV for light grid (compute shader writes)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = TOTAL_GRID_CELLS;
    uavDesc.Buffer.StructureByteStride = sizeof(LightGridCell);

    m_lightGridUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    d3dDevice->CreateUnorderedAccessView(m_lightGridBuffer.Get(), nullptr, &uavDesc, m_lightGridUAV);

    LOG_INFO("Light grid descriptors created (SRV + UAV)");

    // Create light buffer (16 lights × 32 bytes = 512 bytes - matches multi-light system)
    const uint64_t lightBufferSize = 16 * 32;  // Max 16 lights, 32 bytes each

    D3D12_RESOURCE_DESC lightDesc = {};
    lightDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightDesc.Width = lightBufferSize;
    lightDesc.Height = 1;
    lightDesc.DepthOrArraySize = 1;
    lightDesc.MipLevels = 1;
    lightDesc.Format = DXGI_FORMAT_UNKNOWN;
    lightDesc.SampleDesc.Count = 1;
    lightDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    lightDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = d3dDevice->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &lightDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_lightBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create light buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_lightBuffer->SetName(L"RTXDI Light Buffer");

    // Create SRV for light buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC lightSRVDesc = {};
    lightSRVDesc.Format = DXGI_FORMAT_UNKNOWN;
    lightSRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    lightSRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    lightSRVDesc.Buffer.FirstElement = 0;
    lightSRVDesc.Buffer.NumElements = 16;
    lightSRVDesc.Buffer.StructureByteStride = 32;

    m_lightBufferSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    d3dDevice->CreateShaderResourceView(m_lightBuffer.Get(), &lightSRVDesc, m_lightBufferSRV);

    LOG_INFO("Light buffer created: {} bytes (max 16 lights)", lightBufferSize);

    // === Milestone 2.2: Light Grid Build Shader ===
    LOG_INFO("Loading light grid build compute shader...");

    // Load light grid build shader
    std::ifstream shaderFile("shaders/rtxdi/light_grid_build_cs.dxil", std::ios::binary);
    if (!shaderFile) {
        LOG_ERROR("Failed to open light_grid_build_cs.dxil");
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    ComPtr<ID3DBlob> shaderBlob;
    hr = D3DCreateBlob(shaderData.size(), &shaderBlob);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for light grid build shader");
        return false;
    }
    memcpy(shaderBlob->GetBufferPointer(), shaderData.data(), shaderData.size());
    LOG_INFO("Light grid build shader loaded: {} bytes", shaderData.size());

    // Create root signature
    // b0: GridConstants (8 DWORDs)
    // t0: StructuredBuffer<Light> g_lights
    // u0: RWStructuredBuffer<LightGridCell> g_lightGrid
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[3];
        rootParams[0].InitAsConstants(8, 0);  // b0: GridConstants (8 DWORDs = 32 bytes)
        rootParams[1].InitAsShaderResourceView(0);  // t0: lights
        rootParams[2].InitAsUnorderedAccessView(0);  // u0: light grid

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(3, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> signature, error;
        hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Light grid root signature serialization failed: {}", (char*)error->GetBufferPointer());
            }
            return false;
        }

        hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                            IID_PPV_ARGS(&m_lightGridBuildRS));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create light grid root signature");
            return false;
        }

        LOG_INFO("Light grid root signature created");
    }

    // Create compute PSO
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_lightGridBuildRS.Get();
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(shaderBlob.Get());

        hr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_lightGridBuildPSO));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create light grid build PSO: 0x{:08X}", static_cast<uint32_t>(hr));
            return false;
        }

        LOG_INFO("Light grid build PSO created");
    }

    // === Milestone 3: DXR Pipeline Setup ===
    // TODO: Create state object (raygen + miss + closesthit + callable)
    // TODO: Build shader binding table (SBT)

    // === Milestone 4: Reservoir Buffers ===
    // TODO: Create reservoir buffers (2× for ping-pong)
    // Size: width * height * sizeof(RTXDI_DIReservoir)
    // Example: 1920×1080 × 64 bytes = 126 MB per buffer

    m_initialized = true;
    LOG_INFO("RTXDI Lighting System initialized successfully!");
    LOG_INFO("  Milestone 2.1: Buffers created ✅");
    LOG_INFO("  Milestone 2.2: Light grid build shader ready ✅");
    LOG_INFO("  Next: Milestone 2.3 - PIX validation");

    return true;
}

void RTXDILightingSystem::UpdateLightGrid(const void* lights, uint32_t lightCount,
                                          ID3D12GraphicsCommandList* commandList) {
    if (!m_initialized) {
        return;
    }

    if (!m_lightGridBuildPSO || !m_lightGridBuildRS) {
        LOG_ERROR("Light grid build PSO/RS not initialized");
        return;
    }

    if (lightCount == 0 || lightCount > 16) {
        LOG_WARN("Invalid light count: {} (must be 1-16)", lightCount);
        return;
    }

    // === Milestone 2.2 Implementation: Light Grid Build ===

    // 1. Upload lights to GPU buffer (CPU → GPU)
    {
        D3D12_RANGE readRange = { 0, 0 };  // Not reading
        void* mappedData = nullptr;

        // Transition light buffer to COPY_DEST
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_lightBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        commandList->ResourceBarrier(1, &barrier);

        // Create upload buffer for lights
        D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(16 * 32);  // Max 16 lights × 32 bytes

        ComPtr<ID3D12Resource> uploadBuffer;
        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &uploadDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer)
        );

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create light upload buffer");
            return;
        }

        // Map and copy
        hr = uploadBuffer->Map(0, &readRange, &mappedData);
        if (SUCCEEDED(hr)) {
            memcpy(mappedData, lights, lightCount * 32);
            uploadBuffer->Unmap(0, nullptr);
        }

        // Copy to GPU buffer
        commandList->CopyBufferRegion(m_lightBuffer.Get(), 0, uploadBuffer.Get(), 0, lightCount * 32);

        // Transition to NON_PIXEL_SHADER_RESOURCE for compute shader read
        barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_lightBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        commandList->ResourceBarrier(1, &barrier);
    }

    // 2. Transition light grid to UNORDERED_ACCESS
    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_lightGridBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        commandList->ResourceBarrier(1, &barrier);
    }

    // 3. Set compute pipeline
    commandList->SetPipelineState(m_lightGridBuildPSO.Get());
    commandList->SetComputeRootSignature(m_lightGridBuildRS.Get());

    // 4. Set grid constants (b0)
    struct GridConstants {
        uint32_t gridCellsX;
        uint32_t gridCellsY;
        uint32_t gridCellsZ;
        uint32_t lightCount;
        float worldMin;
        float worldMax;
        float cellSize;
        uint32_t maxLightsPerCell;
    } constants;

    constants.gridCellsX = GRID_CELLS_X;
    constants.gridCellsY = GRID_CELLS_Y;
    constants.gridCellsZ = GRID_CELLS_Z;
    constants.lightCount = lightCount;
    constants.worldMin = WORLD_MIN;
    constants.worldMax = WORLD_MAX;
    constants.cellSize = CELL_SIZE;
    constants.maxLightsPerCell = MAX_LIGHTS_PER_CELL;

    commandList->SetComputeRoot32BitConstants(0, 8, &constants, 0);

    // 5. Bind light buffer SRV (t0)
    commandList->SetComputeRootShaderResourceView(1, m_lightBuffer->GetGPUVirtualAddress());

    // 6. Bind light grid UAV (u0)
    commandList->SetComputeRootUnorderedAccessView(2, m_lightGridBuffer->GetGPUVirtualAddress());

    // 7. Dispatch compute shader
    // Grid: 30×30×30 cells
    // Thread group: 8×8×8 threads
    // Dispatch: (30/8, 30/8, 30/8) = (4, 4, 4) groups (rounds up)
    uint32_t dispatchX = (GRID_CELLS_X + 7) / 8;
    uint32_t dispatchY = (GRID_CELLS_Y + 7) / 8;
    uint32_t dispatchZ = (GRID_CELLS_Z + 7) / 8;

    commandList->Dispatch(dispatchX, dispatchY, dispatchZ);

    // 8. UAV barrier on light grid (ensure writes complete before next use)
    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_lightGridBuffer.Get());
        commandList->ResourceBarrier(1, &barrier);
    }

    // 9. Transition light grid back to COMMON
    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_lightGridBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON
        );
        commandList->ResourceBarrier(1, &barrier);
    }
}

void RTXDILightingSystem::ComputeLighting(ID3D12GraphicsCommandList* commandList,
                                         ID3D12Resource* particleBuffer,
                                         ID3D12Resource* outputBuffer) {
    if (!m_initialized) {
        return;
    }

    // === Milestone 4 Implementation ===
    // TODO: Bind resources (light grid, reservoirs, particles)
    // TODO: DispatchRays (raygen shader with RTXDI sampling)
    // TODO: Temporal reuse (merge with previous frame reservoirs)
    // TODO: Write output light samples

    // For now, this is a no-op (Milestones 3-4 not yet implemented)
}

void RTXDILightingSystem::DumpBuffers(ID3D12GraphicsCommandList* commandList,
                                      const std::string& outputDir,
                                      uint32_t frameNum) {
    if (!m_initialized) {
        LOG_WARN("RTXDI not initialized, skipping buffer dump");
        return;
    }

    LOG_INFO("Dumping RTXDI buffers to: {}", outputDir);

    auto d3dDevice = m_device->GetDevice();

    // Create readback resources
    D3D12_RESOURCE_DESC gridDesc = m_lightGridBuffer->GetDesc();
    D3D12_RESOURCE_DESC lightDesc = m_lightBuffer->GetDesc();

    D3D12_HEAP_PROPERTIES readbackHeap = {};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;

    // Light grid readback buffer
    ComPtr<ID3D12Resource> gridReadback;
    D3D12_RESOURCE_DESC gridReadbackDesc = {};
    gridReadbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    gridReadbackDesc.Width = gridDesc.Width;
    gridReadbackDesc.Height = 1;
    gridReadbackDesc.DepthOrArraySize = 1;
    gridReadbackDesc.MipLevels = 1;
    gridReadbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    gridReadbackDesc.SampleDesc.Count = 1;
    gridReadbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    gridReadbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = d3dDevice->CreateCommittedResource(
        &readbackHeap,
        D3D12_HEAP_FLAG_NONE,
        &gridReadbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&gridReadback)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create grid readback buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return;
    }

    // Light buffer readback buffer
    ComPtr<ID3D12Resource> lightReadback;
    D3D12_RESOURCE_DESC lightReadbackDesc = {};
    lightReadbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightReadbackDesc.Width = lightDesc.Width;
    lightReadbackDesc.Height = 1;
    lightReadbackDesc.DepthOrArraySize = 1;
    lightReadbackDesc.MipLevels = 1;
    lightReadbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    lightReadbackDesc.SampleDesc.Count = 1;
    lightReadbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    lightReadbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = d3dDevice->CreateCommittedResource(
        &readbackHeap,
        D3D12_HEAP_FLAG_NONE,
        &lightReadbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&lightReadback)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create light readback buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return;
    }

    // Copy GPU buffers to readback buffers
    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_lightGridBuffer.Get(),
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_lightBuffer.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)
        };
        commandList->ResourceBarrier(2, barriers);

        commandList->CopyBufferRegion(gridReadback.Get(), 0, m_lightGridBuffer.Get(), 0, gridDesc.Width);
        commandList->CopyBufferRegion(lightReadback.Get(), 0, m_lightBuffer.Get(), 0, lightDesc.Width);

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_lightGridBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_lightBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        commandList->ResourceBarrier(2, barriers);
    }

    // Execute and wait for completion
    m_device->WaitForGPU();

    // Map and write to files
    void* gridData = nullptr;
    void* lightData = nullptr;

    hr = gridReadback->Map(0, nullptr, &gridData);
    if (SUCCEEDED(hr)) {
        std::string gridPath = outputDir + "/g_lightGrid.bin";
        std::ofstream gridFile(gridPath, std::ios::binary);
        if (gridFile.is_open()) {
            gridFile.write(static_cast<const char*>(gridData), gridDesc.Width);
            gridFile.close();
            LOG_INFO("Dumped light grid: {} ({:.2f} MB)", gridPath, gridDesc.Width / (1024.0f * 1024.0f));
        }
        gridReadback->Unmap(0, nullptr);
    }

    hr = lightReadback->Map(0, nullptr, &lightData);
    if (SUCCEEDED(hr)) {
        std::string lightPath = outputDir + "/g_lights.bin";
        std::ofstream lightFile(lightPath, std::ios::binary);
        if (lightFile.is_open()) {
            lightFile.write(static_cast<const char*>(lightData), lightDesc.Width);
            lightFile.close();
            LOG_INFO("Dumped lights: {} ({} bytes)", lightPath, lightDesc.Width);
        }
        lightReadback->Unmap(0, nullptr);
    }

    LOG_INFO("RTXDI buffer dump complete");
}
