#include "RTXDILightingSystem.h"
#include "core/Device.h"
#include "utils/ResourceManager.h"
#include "utils/Logger.h"
#include "utils/d3dx12/d3dx12.h"

#include <d3dcompiler.h>
#include <fstream>
#include <vector>

// PIX event markers (conditionally compiled)
#ifdef USE_PIX
#define USE_PIX_SUPPORTED_ARCHITECTURE
#include "debug/pix3.h"
#endif

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
    LOG_INFO("Creating DXR pipeline...");

    if (!CreateDebugOutputBuffer()) {
        LOG_ERROR("Failed to create debug output buffer");
        return false;
    }

    if (!CreateDXRPipeline()) {
        LOG_ERROR("Failed to create DXR pipeline");
        return false;
    }

    if (!CreateShaderBindingTable()) {
        LOG_ERROR("Failed to create shader binding table");
        return false;
    }

    LOG_INFO("DXR pipeline created successfully!");

    // === Milestone 4: Reservoir Buffers ===
    // TODO: Create reservoir buffers (2× for ping-pong)
    // Size: width * height * sizeof(RTXDI_DIReservoir)
    // Example: 1920×1080 × 64 bytes = 126 MB per buffer

    m_initialized = true;
    LOG_INFO("RTXDI Lighting System initialized successfully!");
    LOG_INFO("  Milestone 2.1: Buffers created ✅");
    LOG_INFO("  Milestone 2.2: Light grid build shader ready ✅");
    LOG_INFO("  Milestone 3: DXR pipeline ready ✅");
    LOG_INFO("  Next: Milestone 4 - First visual test");

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

    // 1. Upload lights to GPU buffer (CPU → GPU) using ResourceManager
    {
        // Use ResourceManager's upload heap instead of creating a new buffer every frame
        uint32_t lightDataSize = lightCount * 32;
        auto uploadAllocation = m_resources->AllocateUpload(lightDataSize, 256);

        if (!uploadAllocation.cpuAddress) {
            LOG_ERROR("Failed to allocate upload memory for lights");
            return;
        }

        // Copy light data to upload heap
        memcpy(uploadAllocation.cpuAddress, lights, lightDataSize);

        // === DEBUG: Validate source and uploaded data ===
        static bool logged = false;
        if (!logged) {
            // Check source data
            const float* sourceData = reinterpret_cast<const float*>(lights);
            LOG_INFO("  [VALIDATION] Source light 0: pos=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
                     sourceData[0], sourceData[1], sourceData[2], sourceData[3]);

            // Check uploaded data
            const float* uploadedData = reinterpret_cast<const float*>(uploadAllocation.cpuAddress);
            LOG_INFO("  [VALIDATION] Uploaded light 0: pos=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
                     uploadedData[0], uploadedData[1], uploadedData[2], uploadedData[3]);

            LOG_INFO("  [VALIDATION] Upload heap: resource={:p}, offset={}, cpuAddr={:p}",
                     static_cast<void*>(uploadAllocation.resource), uploadAllocation.offset,
                     uploadAllocation.cpuAddress);

            logged = true;
        }

        // Transition light buffer to COPY_DEST
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_lightBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        commandList->ResourceBarrier(1, &barrier);

        // Copy from upload heap to GPU buffer
        commandList->CopyBufferRegion(
            m_lightBuffer.Get(), 0,
            uploadAllocation.resource, uploadAllocation.offset,
            lightDataSize
        );

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

    // === DEBUG: Log compute dispatch parameters ===
    static bool dispatchLogged = false;
    if (!dispatchLogged) {
        LOG_INFO("  [VALIDATION] Compute constants: gridCells=({},{},{}), lightCount={}, cellSize={:.1f}",
                 constants.gridCellsX, constants.gridCellsY, constants.gridCellsZ,
                 constants.lightCount, constants.cellSize);
        dispatchLogged = true;
    }

    commandList->SetComputeRoot32BitConstants(0, 8, &constants, 0);

    // 5. Bind light buffer SRV (t0)
    D3D12_GPU_VIRTUAL_ADDRESS lightBufferGPU = m_lightBuffer->GetGPUVirtualAddress();
    commandList->SetComputeRootShaderResourceView(1, lightBufferGPU);

    // 6. Bind light grid UAV (u0)
    D3D12_GPU_VIRTUAL_ADDRESS lightGridGPU = m_lightGridBuffer->GetGPUVirtualAddress();
    commandList->SetComputeRootUnorderedAccessView(2, lightGridGPU);

    // === DEBUG: Log GPU addresses ===
    static bool addressesLogged = false;
    if (!addressesLogged) {
        LOG_INFO("  [VALIDATION] Light buffer GPU address: 0x{:016X}", lightBufferGPU);
        LOG_INFO("  [VALIDATION] Light grid GPU address: 0x{:016X}", lightGridGPU);
        addressesLogged = true;
    }

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

// === Milestone 3 Implementation ===

bool RTXDILightingSystem::CreateDebugOutputBuffer() {
    auto d3dDevice = m_device->GetDevice();

    // Create debug output texture (R32G32B32A32_FLOAT, screen resolution)
    D3D12_RESOURCE_DESC outputDesc = {};
    outputDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    outputDesc.Width = m_width;
    outputDesc.Height = m_height;
    outputDesc.DepthOrArraySize = 1;
    outputDesc.MipLevels = 1;
    outputDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    outputDesc.SampleDesc.Count = 1;
    outputDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    outputDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = d3dDevice->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &outputDesc,
        D3D12_RESOURCE_STATE_COMMON,  // Start in COMMON, transition on first use
        nullptr,
        IID_PPV_ARGS(&m_debugOutputBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create debug output buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_debugOutputBuffer->SetName(L"RTXDI Debug Output");

    // Create UAV
    // Create SRV (for reading in temporal accumulation shader)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    m_debugOutputSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    d3dDevice->CreateShaderResourceView(m_debugOutputBuffer.Get(), &srvDesc, m_debugOutputSRV);

    // Create UAV (for raygen shader writes)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;

    m_debugOutputUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    d3dDevice->CreateUnorderedAccessView(m_debugOutputBuffer.Get(), nullptr, &uavDesc, m_debugOutputUAV);

    LOG_INFO("Debug output buffer created: {}x{} (R32G32B32A32_FLOAT)", m_width, m_height);

    // === M5: Create temporal accumulation buffers (PING-PONG) ===
    LOG_INFO("Creating RTXDI temporal accumulation buffers (ping-pong)...");

    // Same format as debug output (R32G32B32A32_FLOAT)
    // R: accumulated light index, G: sample count, B: reserved, A: frame ID
    D3D12_RESOURCE_DESC accumDesc = {};
    accumDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    accumDesc.Width = m_width;
    accumDesc.Height = m_height;
    accumDesc.DepthOrArraySize = 1;
    accumDesc.MipLevels = 1;
    accumDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    accumDesc.SampleDesc.Count = 1;
    accumDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    accumDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Create TWO buffers to avoid read-write hazards (ping-pong technique)
    for (int i = 0; i < 2; i++) {
        hr = d3dDevice->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &accumDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,  // Start as UAV
            nullptr,
            IID_PPV_ARGS(&m_accumulatedBuffer[i])
        );

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create accumulated buffer {}: 0x{:08X}", i, static_cast<uint32_t>(hr));
            return false;
        }

        std::wstring name = L"RTXDI Accumulated Samples " + std::to_wstring(i);
        m_accumulatedBuffer[i]->SetName(name.c_str());

        // Create SRV (for reading as previous frame)
        D3D12_SHADER_RESOURCE_VIEW_DESC accumSrvDesc = {};
        accumSrvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        accumSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        accumSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        accumSrvDesc.Texture2D.MipLevels = 1;

        m_accumulatedSRV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        d3dDevice->CreateShaderResourceView(m_accumulatedBuffer[i].Get(), &accumSrvDesc, m_accumulatedSRV[i]);

        // Create UAV (for writing as current frame)
        D3D12_UNORDERED_ACCESS_VIEW_DESC accumUavDesc = {};
        accumUavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        accumUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        accumUavDesc.Texture2D.MipSlice = 0;

        m_accumulatedUAV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        d3dDevice->CreateUnorderedAccessView(m_accumulatedBuffer[i].Get(), nullptr, &accumUavDesc, m_accumulatedUAV[i]);
    }

    uint64_t accumBufferSize = m_width * m_height * 16 * 2;  // 16 bytes per pixel × 2 buffers
    LOG_INFO("Accumulated buffers created (ping-pong): {:.2f} MB total", accumBufferSize / (1024.0f * 1024.0f));

    // Create M5 temporal accumulation pipeline
    if (!CreateTemporalAccumulationPipeline()) {
        LOG_ERROR("Failed to create temporal accumulation pipeline");
        return false;
    }

    return true;
}

bool RTXDILightingSystem::CreateDXRPipeline() {
    auto d3dDevice = m_device->GetDevice();

    // Query for ID3D12Device5 (required for DXR 1.1)
    ComPtr<ID3D12Device5> device5;
    HRESULT hr = d3dDevice->QueryInterface(IID_PPV_ARGS(&device5));
    if (FAILED(hr)) {
        LOG_ERROR("ID3D12Device5 not available (DXR 1.1 required)");
        return false;
    }

    // Load shader libraries
    std::ifstream raygenFile("shaders/rtxdi/rtxdi_raygen.dxil", std::ios::binary);
    if (!raygenFile) {
        LOG_ERROR("Failed to open rtxdi_raygen.dxil");
        return false;
    }

    std::ifstream missFile("shaders/rtxdi/rtxdi_miss.dxil", std::ios::binary);
    if (!missFile) {
        LOG_ERROR("Failed to open rtxdi_miss.dxil");
        return false;
    }

    std::vector<char> raygenData((std::istreambuf_iterator<char>(raygenFile)), std::istreambuf_iterator<char>());
    std::vector<char> missData((std::istreambuf_iterator<char>(missFile)), std::istreambuf_iterator<char>());

    ComPtr<ID3DBlob> raygenBlob, missBlob;
    hr = D3DCreateBlob(raygenData.size(), &raygenBlob);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create raygen blob");
        return false;
    }
    memcpy(raygenBlob->GetBufferPointer(), raygenData.data(), raygenData.size());

    hr = D3DCreateBlob(missData.size(), &missBlob);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create miss blob");
        return false;
    }
    memcpy(missBlob->GetBufferPointer(), missData.data(), missData.size());

    LOG_INFO("Loaded shaders: raygen={} bytes, miss={} bytes", raygenData.size(), missData.size());

    // Create global root signature
    // b0: GridConstants (8 DWORDs)
    // t0: StructuredBuffer<LightGridCell> g_lightGrid
    // t1: StructuredBuffer<Light> g_lights
    // u0: RWTexture2D<float4> g_output
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[4];
        rootParams[0].InitAsConstants(8, 0);  // b0: GridConstants
        rootParams[1].InitAsShaderResourceView(0);  // t0: light grid
        rootParams[2].InitAsShaderResourceView(1);  // t1: lights

        CD3DX12_DESCRIPTOR_RANGE1 uavRange;
        uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: output
        rootParams[3].InitAsDescriptorTable(1, &uavRange);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(4, rootParams, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

        ComPtr<ID3DBlob> signature, error;
        hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("DXR root signature serialization failed: {}", (char*)error->GetBufferPointer());
            }
            return false;
        }

        hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                            IID_PPV_ARGS(&m_dxrGlobalRS));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create DXR root signature");
            return false;
        }

        LOG_INFO("DXR global root signature created");
    }

    // Build state object
    std::vector<D3D12_STATE_SUBOBJECT> subobjects;

    // DXIL library for raygen
    D3D12_EXPORT_DESC raygenExport = {};
    raygenExport.Name = L"RayGen";
    raygenExport.ExportToRename = nullptr;
    raygenExport.Flags = D3D12_EXPORT_FLAG_NONE;

    D3D12_DXIL_LIBRARY_DESC raygenLibDesc = {};
    raygenLibDesc.DXILLibrary.pShaderBytecode = raygenBlob->GetBufferPointer();
    raygenLibDesc.DXILLibrary.BytecodeLength = raygenBlob->GetBufferSize();
    raygenLibDesc.NumExports = 1;
    raygenLibDesc.pExports = &raygenExport;

    D3D12_STATE_SUBOBJECT raygenSubobj = {};
    raygenSubobj.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    raygenSubobj.pDesc = &raygenLibDesc;
    subobjects.push_back(raygenSubobj);

    // DXIL library for miss
    D3D12_EXPORT_DESC missExport = {};
    missExport.Name = L"Miss";
    missExport.ExportToRename = nullptr;
    missExport.Flags = D3D12_EXPORT_FLAG_NONE;

    D3D12_DXIL_LIBRARY_DESC missLibDesc = {};
    missLibDesc.DXILLibrary.pShaderBytecode = missBlob->GetBufferPointer();
    missLibDesc.DXILLibrary.BytecodeLength = missBlob->GetBufferSize();
    missLibDesc.NumExports = 1;
    missLibDesc.pExports = &missExport;

    D3D12_STATE_SUBOBJECT missSubobj = {};
    missSubobj.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    missSubobj.pDesc = &missLibDesc;
    subobjects.push_back(missSubobj);

    // Shader config (payload size = 16 bytes, attributes = 8 bytes)
    D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
    shaderConfig.MaxPayloadSizeInBytes = 16;
    shaderConfig.MaxAttributeSizeInBytes = 8;

    D3D12_STATE_SUBOBJECT shaderConfigSubobj = {};
    shaderConfigSubobj.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shaderConfigSubobj.pDesc = &shaderConfig;
    subobjects.push_back(shaderConfigSubobj);

    // Pipeline config (max recursion depth = 1)
    D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
    pipelineConfig.MaxTraceRecursionDepth = 1;

    D3D12_STATE_SUBOBJECT pipelineConfigSubobj = {};
    pipelineConfigSubobj.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    pipelineConfigSubobj.pDesc = &pipelineConfig;
    subobjects.push_back(pipelineConfigSubobj);

    // Global root signature
    D3D12_STATE_SUBOBJECT globalRSSubobj = {};
    globalRSSubobj.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    globalRSSubobj.pDesc = m_dxrGlobalRS.GetAddressOf();
    subobjects.push_back(globalRSSubobj);

    // Create state object
    D3D12_STATE_OBJECT_DESC stateObjectDesc = {};
    stateObjectDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    stateObjectDesc.NumSubobjects = static_cast<UINT>(subobjects.size());
    stateObjectDesc.pSubobjects = subobjects.data();

    hr = device5->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&m_dxrStateObject));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create DXR state object: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    LOG_INFO("DXR state object created ({} subobjects)", subobjects.size());

    return true;
}

bool RTXDILightingSystem::CreateShaderBindingTable() {
    auto d3dDevice = m_device->GetDevice();

    // Query state object properties
    ComPtr<ID3D12StateObjectProperties> stateObjectProps;
    HRESULT hr = m_dxrStateObject->QueryInterface(IID_PPV_ARGS(&stateObjectProps));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to query state object properties");
        return false;
    }

    // Get shader identifiers
    void* raygenID = stateObjectProps->GetShaderIdentifier(L"RayGen");
    void* missID = stateObjectProps->GetShaderIdentifier(L"Miss");

    if (!raygenID || !missID) {
        LOG_ERROR("Failed to get shader identifiers");
        return false;
    }

    const uint32_t shaderIDSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

    // Calculate record sizes (aligned to D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT)
    m_raygenRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    m_raygenRecordSize = (m_raygenRecordSize + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1) &
                         ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1);

    m_missRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    m_missRecordSize = (m_missRecordSize + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1) &
                       ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1);

    m_hitRecordSize = 0;  // No hit group for Milestone 3

    // Calculate SBT size (aligned to D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT)
    uint64_t raygenTableSize = m_raygenRecordSize;
    uint64_t missTableSize = m_missRecordSize;
    uint64_t hitTableSize = 0;

    raygenTableSize = (raygenTableSize + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1) &
                      ~(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1);
    missTableSize = (missTableSize + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1) &
                    ~(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1);

    uint64_t sbtSize = raygenTableSize + missTableSize;

    // Create upload buffer for SBT
    D3D12_RESOURCE_DESC sbtDesc = {};
    sbtDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    sbtDesc.Width = sbtSize;
    sbtDesc.Height = 1;
    sbtDesc.DepthOrArraySize = 1;
    sbtDesc.MipLevels = 1;
    sbtDesc.Format = DXGI_FORMAT_UNKNOWN;
    sbtDesc.SampleDesc.Count = 1;
    sbtDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    sbtDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES uploadHeap = {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

    hr = d3dDevice->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &sbtDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_sbtBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create SBT buffer: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_sbtBuffer->SetName(L"RTXDI Shader Binding Table");

    // Map and fill SBT
    uint8_t* sbtData = nullptr;
    hr = m_sbtBuffer->Map(0, nullptr, reinterpret_cast<void**>(&sbtData));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map SBT buffer");
        return false;
    }

    // Raygen record
    memcpy(sbtData, raygenID, shaderIDSize);

    // Miss record
    memcpy(sbtData + raygenTableSize, missID, shaderIDSize);

    m_sbtBuffer->Unmap(0, nullptr);

    LOG_INFO("Shader binding table created: {} bytes (raygen={}, miss={})",
             sbtSize, raygenTableSize, missTableSize);

    return true;
}

bool RTXDILightingSystem::CreateTemporalAccumulationPipeline() {
    LOG_INFO("Creating RTXDI temporal accumulation pipeline...");

    auto d3dDevice = m_device->GetDevice();

    // Root signature:
    // [0] b0 - Accumulation constants (48 DWORDs = 192 bytes) - Phase 4 M5 fix: prevViewProj + invViewProj (no viewProj)
    // [1] t0 - RTXDI output (Texture2D - requires descriptor table!)
    // [2] t1 - Previous accumulated (Texture2D - requires descriptor table!)
    // [3] u0 - Current accumulated (RWTexture2D - requires descriptor table!)
    // [4] t2 - RT Depth buffer (Texture2D<float> - Phase 4 M5 fix for depth-based reprojection)
    // Total: 48 + 4 (descriptor tables) = 52 DWORDs (within 64 DWORD limit)

    CD3DX12_DESCRIPTOR_RANGE srvRanges[3];
    srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // t0: Texture2D<float4>
    srvRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // t1: Texture2D<float4>
    srvRanges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // t2: Texture2D<float> (depth)

    CD3DX12_DESCRIPTOR_RANGE uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D<float4>

    CD3DX12_ROOT_PARAMETER rootParams[5];
    rootParams[0].InitAsConstants(48, 0);  // b0: AccumulationConstants (48 DWORDs - removed viewProj)
    rootParams[1].InitAsDescriptorTable(1, &srvRanges[0]);  // t0: RTXDI output
    rootParams[2].InitAsDescriptorTable(1, &srvRanges[1]);  // t1: Prev accumulated
    rootParams[3].InitAsDescriptorTable(1, &uavRange);  // u0: Curr accumulated
    rootParams[4].InitAsDescriptorTable(1, &srvRanges[2]);  // t2: RT depth buffer

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init(_countof(rootParams), rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Failed to serialize temporal accumulation root signature: {}",
                     static_cast<const char*>(error->GetBufferPointer()));
        }
        return false;
    }

    hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                        IID_PPV_ARGS(&m_temporalAccumulateRS));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create temporal accumulation root signature: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_temporalAccumulateRS->SetName(L"RTXDI Temporal Accumulation RS");

    // Load shader bytecode
    std::ifstream shaderFile("shaders/rtxdi/rtxdi_temporal_accumulate.dxil", std::ios::binary);
    if (!shaderFile.is_open()) {
        LOG_ERROR("Failed to load rtxdi_temporal_accumulate.dxil");
        LOG_ERROR("  Make sure shader is compiled!");
        return false;
    }

    std::vector<uint8_t> shaderBytecode((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    shaderFile.close();
    LOG_INFO("Loaded temporal accumulation shader: {} bytes", shaderBytecode.size());

    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_temporalAccumulateRS.Get();
    psoDesc.CS.pShaderBytecode = shaderBytecode.data();
    psoDesc.CS.BytecodeLength = shaderBytecode.size();
    psoDesc.NodeMask = 0;  // Single GPU
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    LOG_INFO("Creating PSO with:");
    LOG_INFO("  Root signature: {}", (void*)psoDesc.pRootSignature);
    LOG_INFO("  Shader size: {} bytes", psoDesc.CS.BytecodeLength);

    hr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_temporalAccumulatePSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create temporal accumulation PSO: HRESULT = 0x{:08X}", static_cast<uint32_t>(hr));

        // Check for common D3D12 errors
        if (hr == E_INVALIDARG) {
            LOG_ERROR("  E_INVALIDARG - Invalid argument (likely root signature mismatch)");
        } else if (hr == E_OUTOFMEMORY) {
            LOG_ERROR("  E_OUTOFMEMORY - Out of memory");
        } else if (hr == 0x887A0005) {  // DXGI_ERROR_DEVICE_REMOVED
            LOG_ERROR("  DXGI_ERROR_DEVICE_REMOVED - GPU device removed");
        }

        return false;
    }

    m_temporalAccumulatePSO->SetName(L"RTXDI Temporal Accumulation PSO");

    LOG_INFO("Temporal accumulation pipeline created");
    return true;
}

void RTXDILightingSystem::DispatchRays(ID3D12GraphicsCommandList4* commandList, uint32_t width, uint32_t height, uint32_t frameIndex) {
    if (!m_initialized || !m_dxrStateObject || !m_sbtBuffer) {
        LOG_ERROR("DXR pipeline not initialized");
        return;
    }

    // Transition debug output buffer to UAV for raygen shader write
    // Track state explicitly to ensure correct transitions
    D3D12_RESOURCE_STATES beforeState = m_debugOutputInSRVState
        ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE  // Previous frame ended with SRV read
        : D3D12_RESOURCE_STATE_COMMON;                     // First frame (initial creation state)

    D3D12_RESOURCE_BARRIER preBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_debugOutputBuffer.Get(),
        beforeState,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    commandList->ResourceBarrier(1, &preBarrier);

    // Set pipeline state
    commandList->SetPipelineState1(m_dxrStateObject.Get());

    // Set compute root signature
    commandList->SetComputeRootSignature(m_dxrGlobalRS.Get());

    // Set grid constants (b0) - Milestone 4: Added frameIndex for temporal random variation
    struct GridConstants {
        uint32_t screenWidth;
        uint32_t screenHeight;
        uint32_t gridCellsX;
        uint32_t gridCellsY;
        uint32_t gridCellsZ;
        float worldMin;
        float cellSize;
        uint32_t frameIndex;      // NEW M4: For random number generation
    } constants;

    constants.screenWidth = width;
    constants.screenHeight = height;
    constants.gridCellsX = GRID_CELLS_X;
    constants.gridCellsY = GRID_CELLS_Y;
    constants.gridCellsZ = GRID_CELLS_Z;
    constants.worldMin = WORLD_MIN;
    constants.cellSize = CELL_SIZE;
    constants.frameIndex = frameIndex;  // NEW M4: Pass frame counter for temporal variation

    commandList->SetComputeRoot32BitConstants(0, 8, &constants, 0);

    // Bind light grid (t0)
    commandList->SetComputeRootShaderResourceView(1, m_lightGridBuffer->GetGPUVirtualAddress());

    // Bind lights (t1)
    commandList->SetComputeRootShaderResourceView(2, m_lightBuffer->GetGPUVirtualAddress());

    // Bind debug output (u0) - requires descriptor heap
    auto descriptorHeap = m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
    commandList->SetDescriptorHeaps(1, heaps);

    // Get GPU handle for debug output UAV
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    uint32_t descriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Calculate offset to our UAV descriptor (from CPU handle)
    SIZE_T cpuOffset = m_debugOutputUAV.ptr - descriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr;
    uint32_t descriptorIndex = static_cast<uint32_t>(cpuOffset / descriptorSize);

    gpuHandle.ptr += descriptorIndex * descriptorSize;
    commandList->SetComputeRootDescriptorTable(3, gpuHandle);

    // Build dispatch rays descriptor
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};

    // Raygen shader table
    dispatchDesc.RayGenerationShaderRecord.StartAddress = m_sbtBuffer->GetGPUVirtualAddress();
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = m_raygenRecordSize;

    // Miss shader table
    uint64_t raygenTableSize = (m_raygenRecordSize + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1) &
                               ~(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT - 1);
    dispatchDesc.MissShaderTable.StartAddress = m_sbtBuffer->GetGPUVirtualAddress() + raygenTableSize;
    dispatchDesc.MissShaderTable.SizeInBytes = m_missRecordSize;
    dispatchDesc.MissShaderTable.StrideInBytes = m_missRecordSize;

    // Hit group table (empty for Milestone 3)
    dispatchDesc.HitGroupTable.StartAddress = 0;
    dispatchDesc.HitGroupTable.SizeInBytes = 0;
    dispatchDesc.HitGroupTable.StrideInBytes = 0;

    // Dispatch dimensions
    dispatchDesc.Width = width;
    dispatchDesc.Height = height;
    dispatchDesc.Depth = 1;

    // Dispatch rays
    commandList->DispatchRays(&dispatchDesc);

    // UAV barrier on debug output (ensure writes complete)
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_debugOutputBuffer.Get());
    commandList->ResourceBarrier(1, &barrier);

    // CRITICAL FIX: Transition debug output from UAV to SRV for Gaussian renderer read
    // Without this transition, the Gaussian renderer cannot read the RTXDI output (resource state violation)
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_debugOutputBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    commandList->ResourceBarrier(1, &barrier);

    // Track state for next frame
    m_debugOutputInSRVState = true;
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

// === M5: Temporal Accumulation Dispatch (Phase 4 M5 Fix: Depth-Based Reprojection) ===
// NOTE: viewProj removed to fit within 64-DWORD root constant limit
void RTXDILightingSystem::DispatchTemporalAccumulation(
    ID3D12GraphicsCommandList* commandList,
    const DirectX::XMFLOAT3& cameraPos,
    const DirectX::XMFLOAT4X4& prevViewProj,
    const DirectX::XMFLOAT4X4& invViewProj,  // Phase 4 M5: For depth unprojection
    D3D12_GPU_DESCRIPTOR_HANDLE depthBufferSRV,  // Phase 4 M5: RT depth buffer from Gaussian renderer
    uint32_t frameIndex
) {
    if (!m_temporalAccumulatePSO || !m_temporalAccumulateRS) {
        LOG_ERROR("Temporal accumulation pipeline not initialized");
        return;
    }

    // PIX marker for debugging
#ifdef USE_PIX
    //     PIXBeginEvent(commandList, PIX_COLOR_INDEX(4), "RTXDI M5 Temporal Accumulation");
#endif

    // === 1. Transition resources ===
    std::vector<D3D12_RESOURCE_BARRIER> barriers;

    // Debug output (RTXDI raygen output) → SRV for read
    if (!m_debugOutputInSRVState) {
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            m_debugOutputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        ));
        m_debugOutputInSRVState = true;
    }

    // PING-PONG: Determine which buffers to read/write
    uint32_t prevIndex = 1 - m_currentAccumIndex;  // Previous frame's buffer (read as SRV)
    uint32_t currIndex = m_currentAccumIndex;      // Current frame's buffer (write as UAV)

    // Current accumulated buffer: If in SRV state from previous frame, transition to UAV for writing
    if (m_accumulatedInSRVState[currIndex]) {
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            m_accumulatedBuffer[currIndex].Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        ));
        m_accumulatedInSRVState[currIndex] = false;
    }

    if (!barriers.empty()) {
        commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }

    // === 2. Set pipeline ===
    commandList->SetPipelineState(m_temporalAccumulatePSO.Get());
    commandList->SetComputeRootSignature(m_temporalAccumulateRS.Get());

    // === 3. Set constants (Phase 4 M5 Fix: prevViewProj + invViewProj, viewProj removed) ===
    struct AccumulationConstants {
        uint32_t screenWidth;
        uint32_t screenHeight;
        uint32_t frameIndex;
        uint32_t maxSamples;
        float resetThreshold;
        float padding1;
        float padding2;
        float padding3;
        DirectX::XMFLOAT3 cameraPos;
        uint32_t padding4;
        DirectX::XMFLOAT3 prevCameraPos;
        uint32_t forceReset;
        DirectX::XMFLOAT4X4 prevViewProj;   // For reprojection to previous frame
        DirectX::XMFLOAT4X4 invViewProj;    // Phase 4 M5: Inverse ViewProj for depth unprojection
        // NOTE: viewProj removed to fit within 64-DWORD root constant limit
    } constants;

    constants.screenWidth = m_width;
    constants.screenHeight = m_height;
    constants.frameIndex = frameIndex;
    constants.maxSamples = m_maxSamples;
    constants.resetThreshold = m_resetThreshold;
    constants.padding1 = 0.0f;
    constants.padding2 = 0.0f;
    constants.padding3 = 0.0f;
    constants.cameraPos = cameraPos;
    constants.padding4 = 0;
    constants.prevCameraPos = m_prevCameraPos;
    constants.forceReset = m_forceReset ? 1 : 0;
    constants.prevViewProj = prevViewProj;
    constants.invViewProj = invViewProj;  // Phase 4 M5: For depth-based world position reconstruction

    // Root constants size: 16 (base) + 16 (prevViewProj) + 16 (invViewProj) = 48 DWORDs
    commandList->SetComputeRoot32BitConstants(0, 48, &constants, 0);

    // === 4. Bind resources (Descriptor Tables for Texture2D/RWTexture2D) ===
    // PING-PONG BUFFERS: Read from previous frame, write to current frame
    // (prevIndex and currIndex already declared in section 1)

    // Convert CPU descriptor handles to GPU handles
    D3D12_GPU_DESCRIPTOR_HANDLE debugOutputGPU = m_resources->GetGPUHandle(m_debugOutputSRV);
    D3D12_GPU_DESCRIPTOR_HANDLE prevAccumGPU = m_resources->GetGPUHandle(m_accumulatedSRV[prevIndex]);  // Read from prev
    D3D12_GPU_DESCRIPTOR_HANDLE currAccumGPU = m_resources->GetGPUHandle(m_accumulatedUAV[currIndex]);  // Write to curr

    commandList->SetComputeRootDescriptorTable(1, debugOutputGPU);  // t0: Current RTXDI output
    commandList->SetComputeRootDescriptorTable(2, prevAccumGPU);     // t1: PREVIOUS frame's accumulated (SRV)
    commandList->SetComputeRootDescriptorTable(3, currAccumGPU);     // u0: CURRENT frame's accumulated (UAV)
    commandList->SetComputeRootDescriptorTable(4, depthBufferSRV);   // t2: RT depth buffer (Phase 4 M5 fix)

    // === 5. Dispatch compute shader ===
    uint32_t dispatchX = (m_width + 7) / 8;   // 8×8 thread groups
    uint32_t dispatchY = (m_height + 7) / 8;
    commandList->Dispatch(dispatchX, dispatchY, 1);

    // === 6. UAV barrier (ensure writes complete on current frame's buffer) ===
    D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_accumulatedBuffer[currIndex].Get());
    commandList->ResourceBarrier(1, &uavBarrier);

    // === 6b. Transition current buffer from UAV to SRV (for Gaussian renderer to read) ===
    D3D12_RESOURCE_BARRIER toSrvBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_accumulatedBuffer[currIndex].Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    commandList->ResourceBarrier(1, &toSrvBarrier);
    m_accumulatedInSRVState[currIndex] = true;  // Track state

    // === 7. SWAP BUFFERS (ping-pong for next frame) ===
    m_currentAccumIndex = 1 - m_currentAccumIndex;  // Toggle 0↔1

    // === 8. Update camera state for next frame ===
    m_prevCameraPos = cameraPos;
    // m_prevFrameIndex tracked externally or not critical for reset logic anymore
    m_forceReset = false;  // Clear reset flag

#ifdef USE_PIX
    //     PIXEndEvent(commandList);
#endif
}
