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
     * @param outputTexture Texture to render into (or composite with)
     * @param lightBuffer Light data for illumination
     * @param lightCount Number of active lights
     */
    void Render(
        ID3D12GraphicsCommandList* commandList,
        const DirectX::XMMATRIX& viewProj,
        const DirectX::XMFLOAT3& cameraPos,
        ID3D12Resource* outputTexture,
        ID3D12Resource* lightBuffer,
        uint32_t lightCount);

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

    // Grid info
    bool HasGrid() const { return m_hasGrid; }
    uint64_t GetGridSizeBytes() const { return m_gridSizeBytes; }
    uint32_t GetVoxelCount() const { return m_voxelCount; }

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
        float scatteringCoeff;               // 4 bytes
        float maxRayDistance;                // 4 bytes
        float stepSize;                      // 4 bytes
        uint32_t lightCount;                 // 4 bytes
        uint32_t screenWidth;                // 4 bytes
        uint32_t screenHeight;               // 4 bytes
        float time;                          // 4 bytes
        float padding;                       // 4 bytes
    };  // Total: 144 bytes (should be 256-aligned for cbuffer)

    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Grid data
    bool m_hasGrid = false;
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

    uint32_t m_screenWidth = 1920;
    uint32_t m_screenHeight = 1080;
};
