#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <vector>

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

private:
    // Initialization state
    bool m_initialized = false;

    // Device references
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Screen dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // === Milestone 2: Light Grid (ReGIR) ===
    // TODO: Add light grid buffer
    // TODO: Add light grid parameters (cell size, world bounds)
    // TODO: Add compute shader for grid construction

    // === Milestone 3: DXR Pipeline ===
    // TODO: Add state object
    // TODO: Add shader binding table (SBT)
    // TODO: Add raygen/miss/closesthit shader resources

    // === Milestone 4: Reservoir Sampling ===
    // TODO: Add reservoir buffers (ping-pong)
    // TODO: Add RTXDI context parameters
    // TODO: Add temporal reuse logic

    // === Future: Spatial Reuse ===
    // TODO: Add spatial resampling passes
    // TODO: Add visibility reuse cache
};
