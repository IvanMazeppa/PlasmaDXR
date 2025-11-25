#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;

using Microsoft::WRL::ComPtr;

/**
 * Hybrid Probe Grid System for Volumetric Particle Scattering
 *
 * Pre-computes lighting at sparse grid points (probes), particles interpolate
 * between them using trilinear interpolation. Designed to avoid atomic contention
 * issues that plague Volumetric ReSTIR with dense particle distributions.
 *
 * Architecture:
 * - 32³ probe grid (32,768 probes) covering world space [-1500, +1500]
 * - Each probe stores spherical harmonics L2 (9 RGB coefficients = 27 floats)
 * - Probes updated every 4 frames (temporal amortization)
 * - Particles query 8 nearest probes per frame (trilinear interpolation)
 *
 * Performance:
 * - Update cost: ~0.5-1.0ms per frame (amortized)
 * - Query cost: ~0.2-0.3ms per frame (cache-friendly reads)
 * - Zero atomic operations (no contention!)
 *
 * Based on: Irradiance Probes (McGuire et al. 2017) + DDGI (Majercik et al. 2019)
 */
class ProbeGridSystem {
public:
    ProbeGridSystem() = default;
    ~ProbeGridSystem() = default;

    // Disable copy/move
    ProbeGridSystem(const ProbeGridSystem&) = delete;
    ProbeGridSystem& operator=(const ProbeGridSystem&) = delete;

    /**
     * Initialize the probe grid system
     *
     * @param device D3D12 device
     * @param resources Resource manager for buffer allocation
     * @return true if initialization succeeded
     */
    bool Initialize(Device* device, ResourceManager* resources);

    /**
     * Update probe lighting (call every frame, internally amortizes)
     *
     * @param commandList Command list for GPU work
     * @param particleTLAS Acceleration structure for ray queries
     * @param particleBuffer Particle data buffer
     * @param particleCount Number of particles
     * @param lightBuffer Light data buffer
     * @param lightCount Number of lights
     * @param frameIndex Frame counter for temporal amortization
     */
    void UpdateProbes(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* particleTLAS,
        ID3D12Resource* particleBuffer,
        uint32_t particleCount,
        ID3D12Resource* lightBuffer,
        uint32_t lightCount,
        uint32_t frameIndex);

    /**
     * Get probe buffer SRV for particle shader sampling
     *
     * @return GPU descriptor handle for probe buffer SRV
     */
    D3D12_GPU_DESCRIPTOR_HANDLE GetProbeBufferSRV() const { return m_probeBufferSRV_GPU; }
    ID3D12Resource* GetProbeBuffer() const { return m_probeBuffer.Get(); }

    /**
     * Enable/disable probe grid system
     */
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    /**
     * Configuration parameters
     */
    void SetGridSize(uint32_t size);
    void SetRaysPerProbe(uint32_t rays) { m_raysPerProbe = rays; }
    void SetUpdateInterval(uint32_t frames) { m_updateInterval = frames; }
    void SetProbeIntensity(float intensity) { m_probeIntensity = intensity; }

    uint32_t GetGridSize() const { return m_gridSize; }
    uint32_t GetRaysPerProbe() const { return m_raysPerProbe; }
    uint32_t GetUpdateInterval() const { return m_updateInterval; }
    float GetProbeIntensity() const { return m_probeIntensity; }

    /**
     * Get probe grid parameters for shader binding
     */
    struct ProbeGridParams {
        DirectX::XMFLOAT3 gridMin;       // World-space grid min (-1500, -1500, -1500)
        float gridSpacing;               // Distance between probes (93.75 units for 32³)
        uint32_t gridSize;               // Grid dimension (32)
        uint32_t totalProbes;            // Total probe count (32,768)
        uint32_t padding0;
        uint32_t padding1;
    };

    ProbeGridParams GetProbeGridParams() const;

private:
    /**
     * Probe data structure (GPU-side)
     *
     * Each probe stores:
     * - World-space position (12 bytes)
     * - Spherical harmonics L2 irradiance (27 floats × 4 bytes = 108 bytes)
     * - Last update frame (4 bytes)
     * Total: 124 bytes per probe
     *
     * For 32³ grid: 32,768 probes × 124 bytes = 4.06 MB
     *
     * Note: Aligned to 16 bytes for GPU efficiency
     */
    struct Probe {
        DirectX::XMFLOAT3 position;          // World-space probe location (12 bytes)
        uint32_t lastUpdateFrame;            // Frame when last updated (4 bytes)

        // Spherical harmonics L2 (9 coefficients × RGB)
        // L0: 1 coefficient (DC term)
        // L1: 3 coefficients (linear terms)
        // L2: 5 coefficients (quadratic terms)
        DirectX::XMFLOAT3 sh[9];             // 9 × 12 bytes = 108 bytes

        uint32_t padding[1];                 // Align to 128 bytes (128 = 16×8)
    };
    static_assert(sizeof(Probe) == 128, "Probe must be 128 bytes for GPU alignment");

    /**
     * Constant buffer for probe update shader
     */
    struct ProbeUpdateConstants {
        DirectX::XMFLOAT3 gridMin;           // Grid world-space minimum
        float gridSpacing;                   // Distance between probes

        uint32_t gridSize;                   // Grid dimension (48)
        uint32_t raysPerProbe;               // Rays to cast per probe (16)
        uint32_t particleCount;              // Number of particles
        uint32_t lightCount;                 // Number of lights

        uint32_t frameIndex;                 // Frame counter for temporal amortization
        uint32_t updateInterval;             // Frames between full grid updates (4)
        float probeIntensity;                // Intensity multiplier (200-2000, runtime configurable)
        float particleRadius;                // Base particle radius for intersection tests
    };

    // Resource creation helpers
    bool CreateProbeBuffer();
    bool CreatePipelines();

    // Initialization state
    bool m_initialized = false;
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Configuration
    bool m_enabled = true;
    uint32_t m_gridSize = 48;                // 48³ grid (110,592 probes) - Phase 2 resolution upgrade
    uint32_t m_raysPerProbe = 16;            // Phase 2: Increased from 1 to 16 for quality (original was 64)
    uint32_t m_updateInterval = 4;           // Update 1/4 of grid per frame
    float m_probeIntensity = 800.0f;         // Intensity multiplier (200-2000, runtime configurable)

    // World-space grid parameters
    DirectX::XMFLOAT3 m_gridMin = DirectX::XMFLOAT3(-1500.0f, -1500.0f, -1500.0f);
    DirectX::XMFLOAT3 m_gridMax = DirectX::XMFLOAT3(1500.0f, 1500.0f, 1500.0f);
    float m_gridSpacing = 62.5f;             // 3000 / 48 = 62.5 units per cell

    // GPU Resources
    ComPtr<ID3D12Resource> m_probeBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE m_probeBufferSRV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_probeBufferSRV_GPU;
    D3D12_CPU_DESCRIPTOR_HANDLE m_probeBufferUAV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_probeBufferUAV_GPU;

    // Constant buffer for probe update
    ComPtr<ID3D12Resource> m_updateConstantBuffer;

    // Compute pipeline for probe updates
    ComPtr<ID3D12PipelineState> m_updateProbePSO;
    ComPtr<ID3D12RootSignature> m_updateProbeRS;

    // Statistics
    uint32_t m_frameCount = 0;
    uint32_t m_probesUpdatedLastFrame = 0;
};
