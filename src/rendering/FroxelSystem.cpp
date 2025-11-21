#include "FroxelSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <stdexcept>
#include <fstream>
#include <vector>

using namespace DirectX;

FroxelSystem::FroxelSystem(Device* device, ResourceManager* resources)
    : m_device(device)
    , m_resources(resources)
    , m_screenWidth(0)
    , m_screenHeight(0)
    , m_debugVisualization(false)
    , m_initialized(false)
{
    // Default grid parameters (can be adjusted via SetGridDimensions/SetWorldBounds)
    m_gridParams.gridMin = XMFLOAT3(-1500.0f, -1500.0f, -1500.0f);
    m_gridParams.gridMax = XMFLOAT3(1500.0f, 1500.0f, 1500.0f);
    m_gridParams.gridDimensions = XMUINT3(160, 90, 64);  // 921,600 voxels
    m_gridParams.densityMultiplier = 1.0f;
    m_gridParams.particleCount = 0;

    UpdateGridParams();
}

FroxelSystem::~FroxelSystem()
{
    Shutdown();
}

bool FroxelSystem::Initialize(uint32_t width, uint32_t height)
{
    if (m_initialized) {
        LOG_WARN("FroxelSystem already initialized");
        return true;
    }

    m_screenWidth = width;
    m_screenHeight = height;

    LOG_INFO("Initializing Froxel System:");
    LOG_INFO("  Grid Dimensions: {}x{}x{} = {} voxels",
             m_gridParams.gridDimensions.x,
             m_gridParams.gridDimensions.y,
             m_gridParams.gridDimensions.z,
             m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y * m_gridParams.gridDimensions.z);
    LOG_INFO("  World Bounds: [{:.1f}, {:.1f}, {:.1f}] to [{:.1f}, {:.1f}, {:.1f}]",
             m_gridParams.gridMin.x, m_gridParams.gridMin.y, m_gridParams.gridMin.z,
             m_gridParams.gridMax.x, m_gridParams.gridMax.y, m_gridParams.gridMax.z);
    LOG_INFO("  Voxel Size: {:.2f} × {:.2f} × {:.2f}",
             m_gridParams.voxelSize.x, m_gridParams.voxelSize.y, m_gridParams.voxelSize.z);

    try {
        CreateResources();
        CreatePipelineStates();
        m_initialized = true;
        LOG_INFO("Froxel System initialized successfully");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize Froxel System: {}", e.what());
        return false;
    }
}

void FroxelSystem::Shutdown()
{
    if (!m_initialized) return;

    m_densityGrid.Reset();
    m_lightingGrid.Reset();
    m_injectDensityPSO.Reset();
    m_lightVoxelsPSO.Reset();
    m_injectDensityRootSig.Reset();
    m_lightVoxelsRootSig.Reset();

    m_initialized = false;
    LOG_INFO("Froxel System shutdown");
}

void FroxelSystem::CreateResources()
{
    auto device = m_device->GetDevice();

    // === Create 3D Density Grid (R16_FLOAT) ===
    D3D12_RESOURCE_DESC densityDesc = {};
    densityDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    densityDesc.Width = m_gridParams.gridDimensions.x;
    densityDesc.Height = m_gridParams.gridDimensions.y;
    densityDesc.DepthOrArraySize = m_gridParams.gridDimensions.z;
    densityDesc.MipLevels = 1;
    densityDesc.Format = DXGI_FORMAT_R16_FLOAT;
    densityDesc.SampleDesc.Count = 1;
    densityDesc.SampleDesc.Quality = 0;
    densityDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    densityDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &densityDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_densityGrid)
    );

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create density grid texture");
    }

    m_densityGrid->SetName(L"Froxel Density Grid");

    // === Create 3D Lighting Grid (R16G16B16A16_FLOAT) ===
    D3D12_RESOURCE_DESC lightingDesc = densityDesc;
    lightingDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

    hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &lightingDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_lightingGrid)
    );

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create lighting grid texture");
    }

    m_lightingGrid->SetName(L"Froxel Lighting Grid");

    // === Create UAV and SRV descriptors ===

    // Density Grid UAV (for writing in inject_density.hlsl)
    D3D12_UNORDERED_ACCESS_VIEW_DESC densityUavDesc = {};
    densityUavDesc.Format = DXGI_FORMAT_R16_FLOAT;
    densityUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
    densityUavDesc.Texture3D.MipSlice = 0;
    densityUavDesc.Texture3D.FirstWSlice = 0;
    densityUavDesc.Texture3D.WSize = m_gridParams.gridDimensions.z;

    m_densityGridUAVCPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device->CreateUnorderedAccessView(m_densityGrid.Get(), nullptr, &densityUavDesc, m_densityGridUAVCPU);
    m_densityGridUAVGPU = m_resources->GetGPUHandle(m_densityGridUAVCPU);

    // Density Grid SRV (for reading in light_voxels.hlsl)
    D3D12_SHADER_RESOURCE_VIEW_DESC densitySrvDesc = {};
    densitySrvDesc.Format = DXGI_FORMAT_R16_FLOAT;
    densitySrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    densitySrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    densitySrvDesc.Texture3D.MipLevels = 1;

    D3D12_CPU_DESCRIPTOR_HANDLE densitySRVCpu = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device->CreateShaderResourceView(m_densityGrid.Get(), &densitySrvDesc, densitySRVCpu);
    m_densityGridSRVGPU = m_resources->GetGPUHandle(densitySRVCpu);

    // Lighting Grid UAV (for writing in light_voxels.hlsl)
    D3D12_UNORDERED_ACCESS_VIEW_DESC lightingUavDesc = {};
    lightingUavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    lightingUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
    lightingUavDesc.Texture3D.MipSlice = 0;
    lightingUavDesc.Texture3D.FirstWSlice = 0;
    lightingUavDesc.Texture3D.WSize = m_gridParams.gridDimensions.z;

    m_lightingGridUAVCPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device->CreateUnorderedAccessView(m_lightingGrid.Get(), nullptr, &lightingUavDesc, m_lightingGridUAVCPU);
    m_lightingGridUAVGPU = m_resources->GetGPUHandle(m_lightingGridUAVCPU);

    // Lighting Grid SRV (for reading in particle_gaussian_raytrace.hlsl)
    D3D12_SHADER_RESOURCE_VIEW_DESC lightingSrvDesc = {};
    lightingSrvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    lightingSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    lightingSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    lightingSrvDesc.Texture3D.MipLevels = 1;

    D3D12_CPU_DESCRIPTOR_HANDLE lightingSRVCpu = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device->CreateShaderResourceView(m_lightingGrid.Get(), &lightingSrvDesc, lightingSRVCpu);
    m_lightingGridSRVGPU = m_resources->GetGPUHandle(lightingSRVCpu);

    // === Create Constant Buffer for FroxelParams ===
    const UINT constantBufferSize = (sizeof(GridParams) + 255) & ~255;  // Align to 256 bytes
    D3D12_HEAP_PROPERTIES uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);

    hr = device->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &cbDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)
    );

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create froxel constant buffer");
    }

    m_constantBuffer->SetName(L"Froxel Constant Buffer");

    // Map constant buffer (keep it mapped for lifetime)
    hr = m_constantBuffer->Map(0, nullptr, &m_constantBufferMapped);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to map froxel constant buffer");
    }

    LOG_INFO("Created froxel constant buffer: {} bytes (aligned from {} bytes)",
             constantBufferSize, sizeof(GridParams));

    LOG_INFO("Created froxel grid textures ({}x{}x{} voxels)",
             m_gridParams.gridDimensions.x,
             m_gridParams.gridDimensions.y,
             m_gridParams.gridDimensions.z);
    LOG_INFO("  Density Grid: R16_FLOAT ({:.2f} MB)",
             (m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y *
              m_gridParams.gridDimensions.z * 2) / (1024.0f * 1024.0f));
    LOG_INFO("  Lighting Grid: R16G16B16A16_FLOAT ({:.2f} MB)",
             (m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y *
              m_gridParams.gridDimensions.z * 8) / (1024.0f * 1024.0f));
    LOG_INFO("  Created 4 descriptor views (2 UAVs, 2 SRVs)");
}

void FroxelSystem::CreatePipelineStates()
{
    auto d3dDevice = m_device->GetDevice();

    // === Load Inject Density Shader ===
    std::ifstream injectDensityFile("shaders/froxel/inject_density.dxil", std::ios::binary);
    if (!injectDensityFile) {
        throw std::runtime_error("Failed to load inject_density.dxil - run CMake to compile shaders");
    }

    std::vector<char> injectDensityBytecode(
        (std::istreambuf_iterator<char>(injectDensityFile)),
        std::istreambuf_iterator<char>()
    );
    injectDensityFile.close();

    // === Load Light Voxels Shader ===
    std::ifstream lightVoxelsFile("shaders/froxel/light_voxels.dxil", std::ios::binary);
    if (!lightVoxelsFile) {
        throw std::runtime_error("Failed to load light_voxels.dxil - run CMake to compile shaders");
    }

    std::vector<char> lightVoxelsBytecode(
        (std::istreambuf_iterator<char>(lightVoxelsFile)),
        std::istreambuf_iterator<char>()
    );
    lightVoxelsFile.close();

    // === Create Root Signature for Inject Density ===
    // Root Parameters:
    //   b0: FroxelParams constant buffer
    //   t0: Particle buffer (SRV) - root descriptor to avoid heap allocation
    //   u0: Density grid (UAV)

    CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0

    CD3DX12_ROOT_PARAMETER1 rootParams[3];
    rootParams[0].InitAsConstantBufferView(0);  // b0: FroxelParams
    rootParams[1].InitAsShaderResourceView(0);  // t0: Particles (root descriptor - no heap allocation!)
    rootParams[2].InitAsDescriptorTable(1, &ranges[0]);  // u0: Density grid

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC injectRootSigDesc;
    injectRootSigDesc.Init_1_1(_countof(rootParams), rootParams, 0, nullptr,
                               D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3DX12SerializeVersionedRootSignature(&injectRootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                        &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Root signature serialization error: {}", (char*)error->GetBufferPointer());
        }
        throw std::runtime_error("Failed to serialize inject density root signature");
    }

    hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                      IID_PPV_ARGS(&m_injectDensityRootSig));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create inject density root signature");
    }

    m_injectDensityRootSig->SetName(L"Froxel Inject Density Root Signature");

    // === Create Root Signature for Light Voxels ===
    // Root Parameters:
    //   b0: FroxelParams constant buffer
    //   t0: Density grid (SRV)
    //   t1: Light buffer (SRV) - root descriptor to avoid heap allocation
    //   t2: Particle BVH (acceleration structure)
    //   u0: Lighting grid (UAV)

    CD3DX12_DESCRIPTOR_RANGE1 lightRanges[2];
    lightRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // t0
    lightRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0

    CD3DX12_ROOT_PARAMETER1 lightRootParams[5];
    lightRootParams[0].InitAsConstantBufferView(0);  // b0: FroxelParams
    lightRootParams[1].InitAsDescriptorTable(1, &lightRanges[0]);  // t0: Density grid
    lightRootParams[2].InitAsShaderResourceView(1);  // t1: Lights (root descriptor - no heap allocation!)
    lightRootParams[3].InitAsShaderResourceView(2);  // t2: BVH (acceleration structure - must be root descriptor)
    lightRootParams[4].InitAsDescriptorTable(1, &lightRanges[1]);  // u0: Lighting grid

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC lightRootSigDesc;
    lightRootSigDesc.Init_1_1(_countof(lightRootParams), lightRootParams, 0, nullptr,
                              D3D12_ROOT_SIGNATURE_FLAG_NONE);

    hr = D3DX12SerializeVersionedRootSignature(&lightRootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Root signature serialization error: {}", (char*)error->GetBufferPointer());
        }
        throw std::runtime_error("Failed to serialize light voxels root signature");
    }

    hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                      IID_PPV_ARGS(&m_lightVoxelsRootSig));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create light voxels root signature");
    }

    m_lightVoxelsRootSig->SetName(L"Froxel Light Voxels Root Signature");

    // === Create Pipeline State for Inject Density ===
    D3D12_COMPUTE_PIPELINE_STATE_DESC injectPsoDesc = {};
    injectPsoDesc.pRootSignature = m_injectDensityRootSig.Get();
    injectPsoDesc.CS = { injectDensityBytecode.data(), injectDensityBytecode.size() };

    hr = d3dDevice->CreateComputePipelineState(&injectPsoDesc, IID_PPV_ARGS(&m_injectDensityPSO));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create inject density PSO");
    }

    m_injectDensityPSO->SetName(L"Froxel Inject Density PSO");

    // === Create Pipeline State for Light Voxels ===
    D3D12_COMPUTE_PIPELINE_STATE_DESC lightPsoDesc = {};
    lightPsoDesc.pRootSignature = m_lightVoxelsRootSig.Get();
    lightPsoDesc.CS = { lightVoxelsBytecode.data(), lightVoxelsBytecode.size() };

    hr = d3dDevice->CreateComputePipelineState(&lightPsoDesc, IID_PPV_ARGS(&m_lightVoxelsPSO));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create light voxels PSO");
    }

    m_lightVoxelsPSO->SetName(L"Froxel Light Voxels PSO");

    LOG_INFO("Froxel pipeline states created successfully");
    LOG_INFO("  Inject Density: {} bytes", injectDensityBytecode.size());
    LOG_INFO("  Light Voxels: {} bytes", lightVoxelsBytecode.size());
}

void FroxelSystem::InjectDensity(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount)
{
    if (!m_initialized) {
        LOG_ERROR("FroxelSystem not initialized");
        return;
    }

    // Update particle count and upload constant buffer
    m_gridParams.particleCount = particleCount;
    memcpy(m_constantBufferMapped, &m_gridParams, sizeof(GridParams));

    // Set pipeline state and root signature
    commandList->SetPipelineState(m_injectDensityPSO.Get());
    commandList->SetComputeRootSignature(m_injectDensityRootSig.Get());

    // Root param 0: Constant buffer (b0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Root param 1: Particle buffer SRV (t0) - use root descriptor (no heap allocation!)
    commandList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());

    // Root param 2: Density grid UAV (u0)
    commandList->SetComputeRootDescriptorTable(2, m_densityGridUAVGPU);

    // Dispatch density injection compute shader
    // Thread groups: (particleCount + 255) / 256
    uint32_t threadGroups = (particleCount + 255) / 256;

    LOG_INFO("FROXEL: Injecting density for {} particles ({} thread groups)", particleCount, threadGroups);

    commandList->Dispatch(threadGroups, 1, 1);

    LOG_INFO("FROXEL: Density injection dispatch complete");
}

void FroxelSystem::LightVoxels(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    ID3D12Resource* lightBuffer,
    uint32_t lightCount,
    ID3D12Resource* particleBVH)
{
    if (!m_initialized) {
        LOG_ERROR("FroxelSystem not initialized");
        return;
    }

    // Set pipeline state and root signature
    commandList->SetPipelineState(m_lightVoxelsPSO.Get());
    commandList->SetComputeRootSignature(m_lightVoxelsRootSig.Get());

    // Root param 0: Constant buffer (b0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Root param 1: Density grid SRV (t0)
    commandList->SetComputeRootDescriptorTable(1, m_densityGridSRVGPU);

    // Root param 2: Light buffer SRV (t1) - use root descriptor (no heap allocation!)
    commandList->SetComputeRootShaderResourceView(2, lightBuffer->GetGPUVirtualAddress());

    // Root param 3: Particle BVH (t2) - acceleration structure (root descriptor)
    commandList->SetComputeRootShaderResourceView(3, particleBVH->GetGPUVirtualAddress());

    // Root param 4: Lighting grid UAV (u0)
    commandList->SetComputeRootDescriptorTable(4, m_lightingGridUAVGPU);

    // Thread groups: Dispatch 8×8×8 thread groups to cover entire grid
    uint32_t groupsX = (m_gridParams.gridDimensions.x + 7) / 8;
    uint32_t groupsY = (m_gridParams.gridDimensions.y + 7) / 8;
    uint32_t groupsZ = (m_gridParams.gridDimensions.z + 7) / 8;

    LOG_INFO("FROXEL: Lighting {} voxels with {} lights ({}×{}×{} thread groups)",
              m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y * m_gridParams.gridDimensions.z,
              lightCount,
              groupsX, groupsY, groupsZ);

    commandList->Dispatch(groupsX, groupsY, groupsZ);

    LOG_INFO("FROXEL: Voxel lighting dispatch complete");
}

void FroxelSystem::ClearGrid(ID3D12GraphicsCommandList* commandList)
{
    if (!m_initialized) return;

    LOG_DEBUG("Clearing froxel grids");

    // Clear density grid (R16_FLOAT - single channel)
    float densityClearValue[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    commandList->ClearUnorderedAccessViewFloat(
        m_densityGridUAVGPU,
        m_densityGridUAVCPU,
        m_densityGrid.Get(),
        densityClearValue,
        0,
        nullptr
    );

    // Clear lighting grid (R16G16B16A16_FLOAT - RGBA)
    float lightingClearValue[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    commandList->ClearUnorderedAccessViewFloat(
        m_lightingGridUAVGPU,
        m_lightingGridUAVCPU,
        m_lightingGrid.Get(),
        lightingClearValue,
        0,
        nullptr
    );
}

void FroxelSystem::SetGridDimensions(uint32_t x, uint32_t y, uint32_t z)
{
    if (m_initialized) {
        LOG_WARN("Cannot change grid dimensions after initialization");
        return;
    }

    m_gridParams.gridDimensions = XMUINT3(x, y, z);
    UpdateGridParams();

    LOG_INFO("Grid dimensions set to {}x{}x{} = {} voxels",
             x, y, z, x * y * z);
}

void FroxelSystem::SetWorldBounds(const XMFLOAT3& min, const XMFLOAT3& max)
{
    m_gridParams.gridMin = min;
    m_gridParams.gridMax = max;
    UpdateGridParams();

    LOG_INFO("World bounds set to [{:.1f}, {:.1f}, {:.1f}] to [{:.1f}, {:.1f}, {:.1f}]",
             min.x, min.y, min.z, max.x, max.y, max.z);
}

void FroxelSystem::UpdateGridParams()
{
    // Compute voxel size
    m_gridParams.voxelSize.x = (m_gridParams.gridMax.x - m_gridParams.gridMin.x) / m_gridParams.gridDimensions.x;
    m_gridParams.voxelSize.y = (m_gridParams.gridMax.y - m_gridParams.gridMin.y) / m_gridParams.gridDimensions.y;
    m_gridParams.voxelSize.z = (m_gridParams.gridMax.z - m_gridParams.gridMin.z) / m_gridParams.gridDimensions.z;
}
