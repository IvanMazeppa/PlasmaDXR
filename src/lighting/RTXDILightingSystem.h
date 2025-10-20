#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <vector>
#include <string>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;

using Microsoft::WRL::ComPtr;

/**
 * RTXDI (RTX Direct Illumination) Lighting System
 *
 * Replaces multi-light brute-force approach with NVIDIA's production-grade
 * ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling for Direct Illumination).
 *
 * Architecture:
 * 1. Light Grid (ReGIR) - Spatial acceleration structure
 * 2. Reservoir Sampling - Importance-weighted light selection
 * 3. Temporal Reuse - Merge with previous frame reservoirs
 * 4. Spatial Reuse - Merge with neighbor reservoirs
 *
 * Phase 4 Integration Roadmap:
 * - Milestone 1: SDK linked ✅ (COMPLETE)
 * - Milestone 2: Light grid construction
 * - Milestone 3: DXR pipeline setup
 * - Milestone 4: First visual test (basic temporal reuse)
 * - Full Feature: Add spatial reuse + optimization
 */
class RTXDILightingSystem {
public:
    RTXDILightingSystem() = default;
    ~RTXDILightingSystem() = default;

    // Disable copy/move
    RTXDILightingSystem(const RTXDILightingSystem&) = delete;
    RTXDILightingSystem& operator=(const RTXDILightingSystem&) = delete;

    /**
     * Initialize RTXDI lighting system
     *
     * @param device D3D12 device
     * @param resources Resource manager for buffer allocation
     * @param width Screen width (for reservoir buffers)
     * @param height Screen height (for reservoir buffers)
     * @return true if initialization succeeded
     */
    bool Initialize(Device* device, ResourceManager* resources, uint32_t width, uint32_t height);

    /**
     * Update light grid from current light array
     *
     * @param lights Array of lights to build grid from
     * @param lightCount Number of lights
     * @param commandList Command list for GPU work
     */
    void UpdateLightGrid(const void* lights, uint32_t lightCount, ID3D12GraphicsCommandList* commandList);

    /**
     * Perform RTXDI sampling and resampling
     *
     * @param commandList Command list for GPU work
     * @param particleBuffer Particle data (for visibility testing)
     * @param outputBuffer Output buffer for selected light samples
     */
    void ComputeLighting(ID3D12GraphicsCommandList* commandList,
                        ID3D12Resource* particleBuffer,
                        ID3D12Resource* outputBuffer);

    /**
     * Check if RTXDI is ready for use
     */
    bool IsReady() const { return m_initialized; }

    /**
     * Dump RTXDI buffers for analysis
     *
     * @param commandList Command list for GPU readback
     * @param outputDir Output directory for buffer files
     * @param frameNum Frame number for filename
     */
    void DumpBuffers(ID3D12GraphicsCommandList* commandList,
                    const std::string& outputDir,
                    uint32_t frameNum);

    /**
     * Dispatch DXR rays for light grid sampling (Milestone 4: Reservoir Sampling)
     *
     * @param commandList Command list for GPU work
     * @param width Screen width
     * @param height Screen height
     * @param frameIndex Frame counter for temporal random variation
     */
    void DispatchRays(ID3D12GraphicsCommandList4* commandList, uint32_t width, uint32_t height, uint32_t frameIndex);

    /**
     * Get RTXDI debug output buffer (selected light indices per pixel)
     * R channel: asfloat(lightIndex) - 0-15 or 0xFFFFFFFF if no lights
     * G/B channels: debug data (cell index, light count)
     *
     * @return Debug output buffer resource (raw M4 output)
     */
    ID3D12Resource* GetDebugOutputBuffer() const { return m_debugOutputBuffer.Get(); }

    /**
     * Get M5 accumulated buffer (temporally smoothed light selection)
     * This is the buffer that should be used for rendering after M5 temporal accumulation
     * Returns the current frame's output buffer
     *
     * @return Accumulated buffer resource (M5 output)
     */
    ID3D12Resource* GetAccumulatedBuffer() const { return m_accumulatedBuffer[m_currentAccumIndex].Get(); }

private:
    // === DXR Pipeline Creation (Milestone 3) ===
    bool CreateDXRPipeline();
    bool CreateShaderBindingTable();
    bool CreateDebugOutputBuffer();

    // === M5 Pipeline Creation ===
    bool CreateTemporalAccumulationPipeline();

    // Initialization state
    bool m_initialized = false;

    // Device references
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Screen dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // === Milestone 2: Light Grid (ReGIR) ===

    // Light grid parameters
    static constexpr uint32_t GRID_CELLS_X = 30;     // 600 units / 20 = 30 cells
    static constexpr uint32_t GRID_CELLS_Y = 30;
    static constexpr uint32_t GRID_CELLS_Z = 30;
    static constexpr uint32_t TOTAL_GRID_CELLS = GRID_CELLS_X * GRID_CELLS_Y * GRID_CELLS_Z;  // 27,000 cells

    static constexpr float WORLD_MIN = -1500.0f;     // World bounds (expanded for RTXDI presets)
    static constexpr float WORLD_MAX = 1500.0f;
    static constexpr float CELL_SIZE = 100.0f;       // 100 units per cell (3000 / 30)
    static constexpr uint32_t MAX_LIGHTS_PER_CELL = 16;

    // Light grid cell structure (128 bytes)
    struct LightGridCell {
        uint32_t lightIndices[16];  // Indices into light array (64 bytes)
        float lightWeights[16];     // Importance weights (64 bytes)
    };

    // Light grid GPU resources
    ComPtr<ID3D12Resource> m_lightGridBuffer;           // Light grid cells
    D3D12_CPU_DESCRIPTOR_HANDLE m_lightGridSRV;         // For shader reads
    D3D12_CPU_DESCRIPTOR_HANDLE m_lightGridUAV;         // For compute shader writes

    // Light buffer (uploaded from multi-light system)
    ComPtr<ID3D12Resource> m_lightBuffer;               // Current frame lights
    D3D12_CPU_DESCRIPTOR_HANDLE m_lightBufferSRV;       // For light grid build shader

    // Light grid build compute shader
    ComPtr<ID3D12PipelineState> m_lightGridBuildPSO;
    ComPtr<ID3D12RootSignature> m_lightGridBuildRS;

    // === Milestone 3: DXR Pipeline ===
    ComPtr<ID3D12StateObject> m_dxrStateObject;
    ComPtr<ID3D12RootSignature> m_dxrGlobalRS;

    // Shader binding table (SBT)
    ComPtr<ID3D12Resource> m_sbtBuffer;
    uint64_t m_raygenRecordSize = 0;
    uint64_t m_missRecordSize = 0;
    uint64_t m_hitRecordSize = 0;

    // Debug output buffer (for visualization)
    ComPtr<ID3D12Resource> m_debugOutputBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE m_debugOutputSRV;  // For reading in M5 temporal accumulation
    D3D12_CPU_DESCRIPTOR_HANDLE m_debugOutputUAV;  // For writing from raygen shader
    bool m_debugOutputInSRVState = false;  // Track resource state for proper transitions

    // === Milestone 5: Temporal Accumulation ===
    // Ping-pong buffers to avoid read-write hazards
    ComPtr<ID3D12Resource> m_accumulatedBuffer[2];        // Dual temporal accumulation buffers
    D3D12_CPU_DESCRIPTOR_HANDLE m_accumulatedSRV[2];      // SRVs for both buffers
    D3D12_CPU_DESCRIPTOR_HANDLE m_accumulatedUAV[2];      // UAVs for both buffers
    uint32_t m_currentAccumIndex = 0;                     // Ping-pong index (swaps 0↔1 each frame)

    ComPtr<ID3D12PipelineState> m_temporalAccumulatePSO;  // Temporal accumulation compute PSO
    ComPtr<ID3D12RootSignature> m_temporalAccumulateRS;   // Root signature

    // Camera state for reset detection
    DirectX::XMFLOAT3 m_prevCameraPos = {0, 0, 0};
    uint32_t m_prevFrameIndex = 0;
    bool m_forceReset = false;                            // Manual reset trigger

    // Accumulation parameters
    uint32_t m_maxSamples = 8;                            // Default: 8 samples (Performance mode)
    float m_resetThreshold = 10.0f;                       // Camera movement threshold (units)

    // === Future: Spatial Reuse (M6) ===
    // TODO: Add spatial resampling passes
    // TODO: Add visibility reuse cache

public:
    // M5 Temporal Accumulation API
    void DispatchTemporalAccumulation(ID3D12GraphicsCommandList* commandList,
                                      const DirectX::XMFLOAT3& cameraPos,
                                      uint32_t frameIndex);

    // Getters/setters for M5 parameters
    uint32_t GetMaxSamples() const { return m_maxSamples; }
    void SetMaxSamples(uint32_t maxSamples) { m_maxSamples = maxSamples; m_forceReset = true; }

    float GetResetThreshold() const { return m_resetThreshold; }
    void SetResetThreshold(float threshold) { m_resetThreshold = threshold; }

    void ForceReset() { m_forceReset = true; }

    // Get accumulated buffer SRV for Gaussian renderer binding (current frame's output)
    D3D12_CPU_DESCRIPTOR_HANDLE GetAccumulatedBufferSRV() const { return m_accumulatedSRV[m_currentAccumIndex]; }
};
