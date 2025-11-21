#include "FroxelSystem.h"
#include "../core/Device.h"
#include "../utils/ShaderManager.h"
#include "../utils/Logger.h"
#include <stdexcept>

using namespace DirectX;

FroxelSystem::FroxelSystem(Device* device, ShaderManager* shaderManager)
    : m_device(device)
    , m_shaderManager(shaderManager)
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
        LOG_WARNING("FroxelSystem already initialized");
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
    auto device = m_device->GetD3DDevice();

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
    // Note: Actual descriptor creation should be done by ResourceManager
    // This is a placeholder - you'll need to integrate with your descriptor heap system
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
}

void FroxelSystem::CreatePipelineStates()
{
    // Load and compile shaders
    auto injectDensityShader = m_shaderManager->LoadComputeShader(L"shaders/froxel/inject_density.hlsl");

    if (!injectDensityShader) {
        throw std::runtime_error("Failed to load inject_density.hlsl");
    }

    // Create root signature for density injection
    // TODO: Implement root signature creation
    // For now, this is a placeholder

    LOG_INFO("Froxel pipeline states created");
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

    // Update particle count
    m_gridParams.particleCount = particleCount;

    // TODO: Implement density injection dispatch
    // This will dispatch the inject_density.hlsl compute shader

    // Dispatch density injection compute shader
    // Thread groups: (particleCount + 255) / 256
    uint32_t threadGroups = (particleCount + 255) / 256;

    LOG_DEBUG("Injecting density for {} particles ({} thread groups)", particleCount, threadGroups);

    // commandList->Dispatch(threadGroups, 1, 1);
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

    // TODO: Implement voxel lighting dispatch
    // This will dispatch the light_voxels.hlsl compute shader

    // Thread groups: Dispatch 8×8×8 thread groups to cover entire grid
    uint32_t groupsX = (m_gridParams.gridDimensions.x + 7) / 8;
    uint32_t groupsY = (m_gridParams.gridDimensions.y + 7) / 8;
    uint32_t groupsZ = (m_gridParams.gridDimensions.z + 7) / 8;

    LOG_DEBUG("Lighting {} voxels with {} lights ({} thread groups)",
              m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y * m_gridParams.gridDimensions.z,
              lightCount,
              groupsX * groupsY * groupsZ);

    // commandList->Dispatch(groupsX, groupsY, groupsZ);
}

void FroxelSystem::ClearGrid(ID3D12GraphicsCommandList* commandList)
{
    if (!m_initialized) return;

    // Clear density grid to zero
    // TODO: Use ClearUnorderedAccessViewFloat
    float clearValue[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    LOG_DEBUG("Clearing froxel grids");

    // commandList->ClearUnorderedAccessViewFloat(...);
}

void FroxelSystem::SetGridDimensions(uint32_t x, uint32_t y, uint32_t z)
{
    if (m_initialized) {
        LOG_WARNING("Cannot change grid dimensions after initialization");
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
