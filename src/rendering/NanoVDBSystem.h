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
    // Material type enum for volumetric rendering behavior
    // Controls emission, scattering, and color behavior
    // Must be declared before methods that use it
    enum class NanoVDBMaterialType : uint32_t {
        SMOKE = 0,      // Neutral grey scattering, NO emission (cold particulate)
        FIRE = 1,       // Temperature-based orange/red emission
        PLASMA = 2,     // Hot blue-white emission (high temperature)
        NEBULA = 3,     // Custom albedo color for emission tint
        GAS_CLOUD = 4   // Similar to smoke but with slight emission
    };

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

    // Material type - controls emission/scattering behavior
    void SetMaterialType(NanoVDBMaterialType type) { m_materialType = type; }
    NanoVDBMaterialType GetMaterialType() const { return m_materialType; }

    // Albedo - base color for scattering and emission tint
    void SetAlbedo(const DirectX::XMFLOAT3& albedo) { m_albedo = albedo; }
    DirectX::XMFLOAT3 GetAlbedo() const { return m_albedo; }

    // Sphere parameters (from CreateFogSphere)
    DirectX::XMFLOAT3 GetSphereCenter() const { return m_sphereCenter; }
    float GetSphereRadius() const { return m_sphereRadius; }

    // Grid info
    bool HasGrid() const { return m_hasGrid; }
    bool HasFileGrid() const { return m_hasFileGrid; }
    uint64_t GetGridSizeBytes() const { return m_gridSizeBytes; }
    uint32_t GetVoxelCount() const { return m_voxelCount; }

    // Grid metadata (Phase 1: Blender NanoVDB visibility fix)
    const std::string& GetGridName() const { return m_gridName; }
    uint32_t GetGridType() const { return m_gridType; }
    const std::string& GetGridTypeName() const { return m_gridTypeName; }
    const std::string& GetLastError() const { return m_lastError; }
    bool HasError() const { return !m_lastError.empty(); }
    void ClearError() { m_lastError.clear(); }

    // Grid type constants (from PNanoVDB.h)
    static constexpr uint32_t GRID_TYPE_UNKNOWN = 0;
    static constexpr uint32_t GRID_TYPE_FLOAT = 1;
    static constexpr uint32_t GRID_TYPE_DOUBLE = 2;
    static constexpr uint32_t GRID_TYPE_INT16 = 3;
    static constexpr uint32_t GRID_TYPE_INT32 = 4;
    static constexpr uint32_t GRID_TYPE_INT64 = 5;
    static constexpr uint32_t GRID_TYPE_VEC3F = 6;
    static constexpr uint32_t GRID_TYPE_VEC3D = 7;
    static constexpr uint32_t GRID_TYPE_MASK = 8;
    static constexpr uint32_t GRID_TYPE_HALF = 9;       // 16-bit float - common from Blender Half precision
    static constexpr uint32_t GRID_TYPE_UINT32 = 10;
    static constexpr uint32_t GRID_TYPE_BOOL = 11;
    static constexpr uint32_t GRID_TYPE_RGBA8 = 12;
    static constexpr uint32_t GRID_TYPE_FP4 = 13;
    static constexpr uint32_t GRID_TYPE_FP8 = 14;
    static constexpr uint32_t GRID_TYPE_FP16 = 15;      // Another 16-bit format - from Blender Mini precision
    static constexpr uint32_t GRID_TYPE_FPN = 16;
    static constexpr uint32_t GRID_TYPE_VEC4F = 17;
    static constexpr uint32_t GRID_TYPE_VEC4D = 18;

    // Helper to check if grid type is shader-compatible
    bool IsGridTypeSupported() const { return m_gridType == GRID_TYPE_FLOAT || m_gridType == GRID_TYPE_HALF || m_gridType == GRID_TYPE_FP16; }

    // ========================================================================
    // PHASE 2: GRID ENUMERATION AND SELECTION
    // ========================================================================

    /**
     * Information about a single grid within a NanoVDB file
     */
    struct GridInfo {
        std::string name;           // Grid name (e.g., "density", "temperature")
        uint32_t type;              // Grid type ID (GRID_TYPE_FLOAT, etc.)
        std::string typeName;       // Human-readable type name
        uint32_t index;             // Index in the file (0-based)
        bool isCompatible;          // True if shader can render this grid type
    };

    /**
     * Enumerate all grids in a NanoVDB file
     * @param filepath Path to .nvdb file
     * @return Vector of GridInfo for each grid in the file
     */
    static std::vector<GridInfo> EnumerateGrids(const std::string& filepath);

    /**
     * Load a specific grid by name from a NanoVDB file
     * @param filepath Path to .nvdb file
     * @param gridName Name of grid to load (empty = prefer "density", then first float grid)
     * @return true if load succeeded
     */
    bool LoadFromFile(const std::string& filepath, const std::string& gridName);

    /**
     * Get list of grids in the currently loaded file
     */
    const std::vector<GridInfo>& GetAvailableGrids() const { return m_availableGrids; }

    /**
     * Get index of currently selected grid
     */
    uint32_t GetSelectedGridIndex() const { return m_selectedGridIndex; }

    // Grid bounds (for positioning/scaling loaded grids)
    DirectX::XMFLOAT3 GetGridWorldMin() const { return m_gridWorldMin; }
    DirectX::XMFLOAT3 GetGridWorldMax() const { return m_gridWorldMax; }
    void SetGridWorldMin(const DirectX::XMFLOAT3& min) { m_gridWorldMin = min; }
    void SetGridWorldMax(const DirectX::XMFLOAT3& max) { m_gridWorldMax = max; }

    // Scale grid bounds uniformly around center
    // NOTE: This also tracks cumulative scale factor needed for shader coordinate transform
    // The shader needs to scale sample positions back to original grid space
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
        // Track cumulative scale for shader coordinate transform
        m_gridScale *= scale;
    }

    // Get current cumulative scale factor
    float GetGridScale() const { return m_gridScale; }

    // Reset scale to 1.0 (restores original grid bounds)
    void ResetGridScale() {
        if (m_gridScale != 1.0f) {
            float resetScale = 1.0f / m_gridScale;
            ScaleGridBounds(resetScale);
        }
        m_gridScale = 1.0f;
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
    static std::string GridTypeToString(uint32_t gridType);

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
        uint32_t gridType;                   // 4 bytes - grid value type (1=FLOAT, 9=HALF, 15=FP16)
        uint32_t materialType;               // 4 bytes - material behavior (0=SMOKE, 1=FIRE, 2=PLASMA, etc.)
        DirectX::XMFLOAT3 albedo;            // 12 bytes - base color for scattering/emission tint
        float gridScale;                     // 4 bytes - cumulative scale factor applied to grid bounds
        DirectX::XMFLOAT3 originalGridCenter; // 12 bytes - original grid center before scaling/repositioning
    };  // Total: 224 bytes (padded to 256 for cbuffer)

    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Grid data
    bool m_hasGrid = false;
    bool m_hasFileGrid = false;  // True if grid was loaded from file (vs procedural)
    uint64_t m_gridSizeBytes = 0;
    uint32_t m_voxelCount = 0;
    DirectX::XMFLOAT3 m_gridWorldMin = { -500, -500, -500 };
    DirectX::XMFLOAT3 m_gridWorldMax = { 500, 500, 500 };

    // Grid metadata (Phase 1: Blender NanoVDB visibility fix)
    std::string m_gridName;           // Name of loaded grid (e.g., "density", "temperature")
    uint32_t m_gridType = 0;          // Grid value type (GRID_TYPE_FLOAT, GRID_TYPE_HALF, etc.)
    std::string m_gridTypeName;       // Human-readable type name
    std::string m_lastError;          // Last error message for UI display
    std::string m_loadedFilePath;     // Path of currently loaded file

    // Phase 2: Grid enumeration and selection
    std::vector<GridInfo> m_availableGrids;  // All grids in current file
    uint32_t m_selectedGridIndex = 0;        // Index of currently loaded grid

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

    // Material type - controls emission/scattering behavior
    // SMOKE (0): Neutral grey scattering, NO emission - for cold particulate matter
    // FIRE (1): Temperature-based orange/red blackbody emission
    // PLASMA (2): Hot blue-white emission
    // NEBULA (3): Custom albedo-tinted emission
    // GAS_CLOUD (4): Slight emission with scattering
    NanoVDBMaterialType m_materialType = NanoVDBMaterialType::SMOKE;
    DirectX::XMFLOAT3 m_albedo = { 0.9f, 0.9f, 0.9f };  // Base color (grey-white for smoke)

    // Sphere parameters (set by CreateFogSphere)
    DirectX::XMFLOAT3 m_sphereCenter = { 0.0f, 0.0f, 0.0f };
    float m_sphereRadius = 200.0f;

    // Grid repositioning and scaling (for file-loaded grids)
    DirectX::XMFLOAT3 m_originalGridCenter = { 0.0f, 0.0f, 0.0f };  // Center when file was loaded
    DirectX::XMFLOAT3 m_gridOffset = { 0.0f, 0.0f, 0.0f };          // Current offset from original
    float m_gridScale = 1.0f;                                        // Cumulative scale factor (for shader coordinate transform)

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
