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

    // Move grid center - updates AABB and calculates offset for shader
    void SetGridCenter(const DirectX::XMFLOAT3& newCenter) {
        DirectX::XMFLOAT3 halfExtent = {
            (m_gridWorldMax.x - m_gridWorldMin.x) * 0.5f,
            (m_gridWorldMax.y - m_gridWorldMin.y) * 0.5f,
            (m_gridWorldMax.z - m_gridWorldMin.z) * 0.5f
        };
        m_gridWorldMin = { newCenter.x - halfExtent.x, newCenter.y - halfExtent.y, newCenter.z - halfExtent.z };
        m_gridWorldMax = { newCenter.x + halfExtent.x, newCenter.y + halfExtent.y, newCenter.z + halfExtent.z };
        m_sphereCenter = newCenter;  // Also update sphere center for fallback

        // Calculate offset from original grid center (for shader sampling)
        m_gridOffset = {
            newCenter.x - m_originalGridCenter.x,
            newCenter.y - m_originalGridCenter.y,
            newCenter.z - m_originalGridCenter.z
        };
    }

    // ========================================================================
    // ANIMATION SUPPORT
    // ========================================================================

    /**
     * Load an animation sequence from multiple .nvdb files
     * @param filepaths Vector of paths to .nvdb files (in order)
     * @return true if at least one frame loaded successfully
     */
    bool LoadAnimationSequence(const std::vector<std::string>& filepaths);

    /**
     * Load animation from a directory pattern (e.g., "smoke_*.nvdb")
     * @param directory Directory containing .nvdb files
     * @param pattern Glob pattern for matching files
     * @return Number of frames loaded
     */
    size_t LoadAnimationFromDirectory(const std::string& directory, const std::string& pattern = "*.nvdb");

    // Animation playback control
    void SetAnimationPlaying(bool playing) { m_animPlaying = playing; }
    bool IsAnimationPlaying() const { return m_animPlaying; }

    void SetAnimationSpeed(float fps) { m_animFPS = fps; }
    float GetAnimationSpeed() const { return m_animFPS; }

    void SetAnimationLoop(bool loop) { m_animLoop = loop; }
    bool GetAnimationLoop() const { return m_animLoop; }

    void SetAnimationFrame(size_t frame);
    size_t GetAnimationFrame() const { return m_animCurrentFrame; }
    size_t GetAnimationFrameCount() const { return m_animFrames.size(); }

    // Call each frame to advance animation based on delta time
    void UpdateAnimation(float deltaTime);

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
        DirectX::XMFLOAT3 gridOffset;        // 12 bytes (offset from original grid position)
        float padding;                       // 4 bytes to align to 256
    };  // Total: 192 bytes (padded to 256 for cbuffer)

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

    // Grid repositioning (for file-loaded grids)
    DirectX::XMFLOAT3 m_originalGridCenter = { 0.0f, 0.0f, 0.0f };  // Center when file was loaded
    DirectX::XMFLOAT3 m_gridOffset = { 0.0f, 0.0f, 0.0f };          // Current offset from original

    uint32_t m_screenWidth = 1920;
    uint32_t m_screenHeight = 1080;

    // ========================================================================
    // ANIMATION STATE
    // ========================================================================

    // Animation frame data - each frame is a separate GPU buffer + SRV
    struct AnimationFrame {
        ComPtr<ID3D12Resource> buffer;
        D3D12_GPU_DESCRIPTOR_HANDLE srvGPU = {};
        D3D12_CPU_DESCRIPTOR_HANDLE srvCPU = {};
        uint64_t sizeBytes = 0;
    };
    std::vector<AnimationFrame> m_animFrames;

    // Playback state
    bool m_animPlaying = false;
    bool m_animLoop = true;
    float m_animFPS = 24.0f;              // Frames per second
    float m_animAccumulator = 0.0f;       // Time accumulator for frame advancement
    size_t m_animCurrentFrame = 0;
};
