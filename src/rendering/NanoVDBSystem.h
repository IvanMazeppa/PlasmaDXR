#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <string>
#include <vector>

// Forward declarations
class Device;
class ResourceManager;

using Microsoft::WRL::ComPtr;

/**
 * NanoVDB Volumetric Rendering System
 *
 * Renders continuous volumetric effects (gas clouds, nebulae, explosions)
 * using NVIDIA's NanoVDB sparse volumetric data structure.
 *
 * Architecture:
 * - Loads or generates NanoVDB grids (fog volumes)
 * - Uploads grid data to GPU as StructuredBuffer<uint>
 * - HLSL shader uses PNanoVDB.h for ray marching
 * - Composites with existing Gaussian particle render
 *
 * Use Cases:
 * - Static nebulae (Crab, Horsehead) - load from .nvdb files
 * - Ambient dust clouds - procedurally generated spheres
 * - Explosions/supernovae - pre-baked animation sequences
 *
 * Based on: NVIDIA NanoVDB (Museth, SIGGRAPH 2021)
 */
class NanoVDBSystem {
public:
    NanoVDBSystem() = default;
    ~NanoVDBSystem();

    // Disable copy/move
    NanoVDBSystem(const NanoVDBSystem&) = delete;
    NanoVDBSystem& operator=(const NanoVDBSystem&) = delete;

    /**
     * Initialize the NanoVDB system
     *
     * @param device D3D12 device
     * @param resources Resource manager for buffer allocation
     * @param screenWidth Render target width
     * @param screenHeight Render target height
     * @return true if initialization succeeded
     */
    bool Initialize(Device* device, ResourceManager* resources,
                    uint32_t screenWidth, uint32_t screenHeight);

    /**
     * Shutdown and release resources
     */
    void Shutdown();

    /**
     * Create a procedural fog sphere for testing
     *
     * @param radius Sphere radius in world units
     * @param center Sphere center in world space
     * @param voxelSize Size of each voxel in world units (smaller = more detail)
     * @param halfWidth Fog falloff width in voxels
     * @return true if creation succeeded
     */
    bool CreateFogSphere(float radius, DirectX::XMFLOAT3 center,
                         float voxelSize = 5.0f, float halfWidth = 3.0f);

    /**
     * Load a NanoVDB grid from file
     *
     * @param filepath Path to .nvdb file
     * @return true if load succeeded
     */
    bool LoadFromFile(const std::string& filepath);

    /**
     * Render the volumetric grid
     *
     * @param commandList Command list for GPU work
     * @param viewProj View-projection matrix
     * @param cameraPos Camera position in world space
     * @param outputUAV GPU descriptor handle for output texture UAV (u0)
     * @param lightSRV GPU descriptor handle for light buffer SRV (t1)
     * @param depthSRV GPU descriptor handle for depth buffer SRV (t2) for occlusion
     * @param lightCount Number of active lights
     * @param descriptorHeap The descriptor heap to set for shader access
     * @param renderWidth Actual render target width (may differ from native due to DLSS)
     * @param renderHeight Actual render target height
     * @param time Animation time in seconds for procedural effects
     */
    void Render(
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
        float time = 0.0f);

    /**
     * Enable/disable the system
     */
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    /**
     * Configuration parameters
     */
    void SetDensityScale(float scale) { m_densityScale = scale; }
    float GetDensityScale() const { return m_densityScale; }

    void SetEmissionStrength(float strength) { m_emissionStrength = strength; }
    float GetEmissionStrength() const { return m_emissionStrength; }

    void SetAbsorptionCoeff(float coeff) { m_absorptionCoeff = coeff; }
    float GetAbsorptionCoeff() const { return m_absorptionCoeff; }

    void SetScatteringCoeff(float coeff) { m_scatteringCoeff = coeff; }
    float GetScatteringCoeff() const { return m_scatteringCoeff; }

    void SetMaxRayDistance(float dist) { m_maxRayDistance = dist; }
    float GetMaxRayDistance() const { return m_maxRayDistance; }

    void SetStepSize(float size) { m_stepSize = size; }
    float GetStepSize() const { return m_stepSize; }

    void SetDebugMode(bool debug) { m_debugMode = debug; }
    bool GetDebugMode() const { return m_debugMode; }

    // Sphere parameters (from CreateFogSphere)
    DirectX::XMFLOAT3 GetSphereCenter() const { return m_sphereCenter; }
    float GetSphereRadius() const { return m_sphereRadius; }

    // Grid info
    bool HasGrid() const { return m_hasGrid; }
    bool HasFileGrid() const { return m_hasFileGrid; }
    uint64_t GetGridSizeBytes() const { return m_gridSizeBytes; }
    uint32_t GetVoxelCount() const { return m_voxelCount; }

    // Grid bounds (for positioning/scaling loaded grids)
    DirectX::XMFLOAT3 GetGridWorldMin() const { return m_gridWorldMin; }
    DirectX::XMFLOAT3 GetGridWorldMax() const { return m_gridWorldMax; }
    void SetGridWorldMin(const DirectX::XMFLOAT3& min) { m_gridWorldMin = min; }
    void SetGridWorldMax(const DirectX::XMFLOAT3& max) { m_gridWorldMax = max; }

    // Scale grid bounds uniformly around center
    void ScaleGridBounds(float scale) {
        DirectX::XMFLOAT3 center = {
            (m_gridWorldMin.x + m_gridWorldMax.x) * 0.5f,
            (m_gridWorldMin.y + m_gridWorldMax.y) * 0.5f,
            (m_gridWorldMin.z + m_gridWorldMax.z) * 0.5f
        };
        DirectX::XMFLOAT3 halfExtent = {
            (m_gridWorldMax.x - m_gridWorldMin.x) * 0.5f * scale,
            (m_gridWorldMax.y - m_gridWorldMin.y) * 0.5f * scale,
            (m_gridWorldMax.z - m_gridWorldMin.z) * 0.5f * scale
        };
        m_gridWorldMin = { center.x - halfExtent.x, center.y - halfExtent.y, center.z - halfExtent.z };
        m_gridWorldMax = { center.x + halfExtent.x, center.y + halfExtent.y, center.z + halfExtent.z };
    }

    // Move grid center
    void SetGridCenter(const DirectX::XMFLOAT3& newCenter) {
        DirectX::XMFLOAT3 halfExtent = {
            (m_gridWorldMax.x - m_gridWorldMin.x) * 0.5f,
            (m_gridWorldMax.y - m_gridWorldMin.y) * 0.5f,
            (m_gridWorldMax.z - m_gridWorldMin.z) * 0.5f
        };
        m_gridWorldMin = { newCenter.x - halfExtent.x, newCenter.y - halfExtent.y, newCenter.z - halfExtent.z };
        m_gridWorldMax = { newCenter.x + halfExtent.x, newCenter.y + halfExtent.y, newCenter.z + halfExtent.z };
        m_sphereCenter = newCenter;  // Also update sphere center for fallback
    }

    // GPU resources for external binding
    ID3D12Resource* GetGridBuffer() const { return m_gridBuffer.Get(); }
    D3D12_GPU_DESCRIPTOR_HANDLE GetGridBufferSRV() const { return m_gridBufferSRV_GPU; }

private:
    bool CreateComputePipeline();
    bool UploadGridToGPU(const void* gridData, uint64_t sizeBytes);

    // Shader constant buffer structure
    struct NanoVDBConstants {
        DirectX::XMFLOAT4X4 invViewProj;    // 64 bytes
        DirectX::XMFLOAT3 cameraPos;         // 12 bytes
        float densityScale;                  // 4 bytes
        DirectX::XMFLOAT3 gridWorldMin;      // 12 bytes (grid bounds)
        float emissionStrength;              // 4 bytes
        DirectX::XMFLOAT3 gridWorldMax;      // 12 bytes
        float absorptionCoeff;               // 4 bytes
        DirectX::XMFLOAT3 sphereCenter;      // 12 bytes (procedural sphere center)
        float scatteringCoeff;               // 4 bytes
        float sphereRadius;                  // 4 bytes (procedural sphere radius)
        float maxRayDistance;                // 4 bytes
        float stepSize;                      // 4 bytes
        uint32_t lightCount;                 // 4 bytes
        uint32_t screenWidth;                // 4 bytes
        uint32_t screenHeight;               // 4 bytes
        float time;                          // 4 bytes
        uint32_t debugMode;                  // 4 bytes (0=normal, 1=debug solid color)
        uint32_t useGridBuffer;              // 4 bytes (0=procedural, 1=file-loaded grid)
        float padding[2];                    // 8 bytes to align to 256
    };  // Total: 176 bytes (padded to 256 for cbuffer)

    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Grid data
    bool m_hasGrid = false;
    bool m_hasFileGrid = false;  // True if grid was loaded from file (vs procedural)
    uint64_t m_gridSizeBytes = 0;
    uint32_t m_voxelCount = 0;
    DirectX::XMFLOAT3 m_gridWorldMin = { -500, -500, -500 };
    DirectX::XMFLOAT3 m_gridWorldMax = { 500, 500, 500 };

    // GPU resources
    ComPtr<ID3D12Resource> m_gridBuffer;              // NanoVDB grid data
    ComPtr<ID3D12Resource> m_constantBuffer;          // Shader constants
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    // Descriptors
    D3D12_GPU_DESCRIPTOR_HANDLE m_gridBufferSRV_GPU = {};
    D3D12_CPU_DESCRIPTOR_HANDLE m_gridBufferSRV_CPU = {};

    // Configuration
    bool m_enabled = false;
    float m_densityScale = 1.0f;
    float m_emissionStrength = 0.5f;
    float m_absorptionCoeff = 0.1f;
    float m_scatteringCoeff = 0.5f;
    float m_maxRayDistance = 2000.0f;
    float m_stepSize = 5.0f;
    bool m_debugMode = false;

    // Sphere parameters (set by CreateFogSphere)
    DirectX::XMFLOAT3 m_sphereCenter = { 0.0f, 0.0f, 0.0f };
    float m_sphereRadius = 200.0f;

    uint32_t m_screenWidth = 1920;
    uint32_t m_screenHeight = 1080;
};
