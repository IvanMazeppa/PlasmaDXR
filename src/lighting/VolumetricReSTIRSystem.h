#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;

using Microsoft::WRL::ComPtr;

/**
 * Volumetric ReSTIR System
 *
 * Implements spatiotemporal reservoir resampling for volumetric path tracing.
 * Based on "Fast Volume Rendering with Spatiotemporal Reservoir Resampling"
 * (Lin, Wyman, Yuksel 2021)
 *
 * Phases:
 * - Phase 1: RIS-only candidate generation (no reuse)
 * - Phase 2: Spatial reuse with Talbot MIS
 * - Phase 3: Temporal reuse (fixes RTXDI patchwork)
 * - Phase 4: Optimization and unbiased final shading
 *
 * Current Status: Phase 1 - Infrastructure Setup
 */
class VolumetricReSTIRSystem {
public:
    VolumetricReSTIRSystem() = default;
    ~VolumetricReSTIRSystem() = default;

    // Disable copy/move
    VolumetricReSTIRSystem(const VolumetricReSTIRSystem&) = delete;
    VolumetricReSTIRSystem& operator=(const VolumetricReSTIRSystem&) = delete;

    /**
     * Initialize the volumetric ReSTIR system
     *
     * @param device D3D12 device
     * @param resources Resource manager for buffer allocation
     * @param width Screen width
     * @param height Screen height
     * @return true if initialization succeeded
     */
    bool Initialize(Device* device, ResourceManager* resources, uint32_t width, uint32_t height);

    /**
     * Generate candidate paths and perform RIS (Phase 1)
     *
     * @param commandList Command list for GPU work
     * @param particleBVH Acceleration structure for ray queries
     * @param particleBuffer Particle data
     * @param particleCount Number of particles
     * @param cameraPos Camera position
     * @param viewMatrix View matrix
     * @param projMatrix Projection matrix
     * @param frameIndex Frame counter for randomization
     */
    void GenerateCandidates(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* particleBVH,
        ID3D12Resource* particleBuffer,
        uint32_t particleCount,
        const DirectX::XMFLOAT3& cameraPos,
        const DirectX::XMMATRIX& viewMatrix,
        const DirectX::XMMATRIX& projMatrix,
        uint32_t frameIndex);

    /**
     * Perform spatial reuse (Phase 2 - not yet implemented)
     */
    void SpatialReuse(ID3D12GraphicsCommandList* commandList);

    /**
     * Perform temporal reuse (Phase 3 - not yet implemented)
     */
    void TemporalReuse(ID3D12GraphicsCommandList* commandList, const DirectX::XMFLOAT3& cameraPos);

    /**
     * Shade the selected paths (final rendering)
     *
     * @param commandList Command list
     * @param outputTexture Output render target
     */
    void ShadeSelectedPaths(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* outputTexture);

    /**
     * Get current reservoir buffer (for reading in shaders)
     */
    ID3D12Resource* GetReservoirBuffer() const { return m_reservoirBuffer[m_currentBufferIndex].Get(); }

    /**
     * Enable/disable phases
     */
    void SetEnableSpatialReuse(bool enable) { m_enableSpatialReuse = enable; }
    void SetEnableTemporalReuse(bool enable) { m_enableTemporalReuse = enable; }

    /**
     * Configuration parameters
     */
    void SetRandomWalksPerPixel(uint32_t M) { m_randomWalksPerPixel = M; }
    void SetMaxBounces(uint32_t K) { m_maxBounces = K; }
    void SetSpatialNeighbors(uint32_t N) { m_spatialNeighbors = N; }
    void SetTemporalClampFactor(float Q) { m_temporalClampFactor = Q; }

private:
    // === Phase 1: Data Structures ===

    /**
     * Path vertex (position + direction)
     * Stored implicitly via distances and directions from camera
     */
    struct PathVertex {
        float z;           // Distance from previous vertex (4 bytes)
        float omega[3];    // Direction to next vertex (12 bytes)
                          // Total: 16 bytes per vertex
    };

    /**
     * Volumetric Reservoir (per-pixel)
     * Stores selected path from weighted reservoir sampling
     *
     * Layout (64 bytes total for K=3):
     * - Path description: 4 + 48 = 52 bytes
     * - RIS state: 4 + 4 + 4 = 12 bytes
     */
    struct VolumetricReservoir {
        uint32_t pathLength;              // Number of scattering events k (4 bytes)
        PathVertex vertices[3];           // Up to 3 bounces (48 bytes for K=3)
        float wsum;                       // Cumulative weight sum (4 bytes)
        float M;                          // Total candidates seen (4 bytes)
        uint32_t flags;                   // Path type and metadata (4 bytes)
                                          // Bit 0: isScatteringPath (vs emission)
                                          // Bits 1-31: reserved
        uint32_t padding;                 // Align to 64 bytes (4 bytes)
    };

    /**
     * Constants for path generation
     */
    struct PathGenerationConstants {
        uint32_t screenWidth;
        uint32_t screenHeight;
        uint32_t particleCount;
        uint32_t randomWalksPerPixel;     // M (default: 4)

        uint32_t maxBounces;              // K (default: 3)
        uint32_t frameIndex;              // For randomization
        uint32_t padding0;
        uint32_t padding1;

        DirectX::XMFLOAT3 cameraPos;
        float padding2;

        DirectX::XMFLOAT4X4 viewMatrix;
        DirectX::XMFLOAT4X4 projMatrix;
        DirectX::XMFLOAT4X4 invViewProjMatrix;
    };

    // === Resource Creation ===
    bool CreateReservoirBuffers();
    bool CreatePiecewiseConstantVolume();
    bool CreatePipelines();

    // Initialization state
    bool m_initialized = false;
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // === Configuration Parameters ===
    uint32_t m_randomWalksPerPixel = 4;   // M (paper uses 4)
    uint32_t m_maxBounces = 3;            // K (paper uses 3)
    uint32_t m_spatialNeighbors = 3;      // N for Phase 2
    float m_temporalClampFactor = 4.0f;   // Q for Phase 3

    // === Phase Control ===
    bool m_enableSpatialReuse = false;    // Phase 2 (not yet implemented)
    bool m_enableTemporalReuse = false;   // Phase 3 (not yet implemented)

    // === GPU Resources ===

    // Reservoir buffers (ping-pong for temporal reuse)
    ComPtr<ID3D12Resource> m_reservoirBuffer[2];
    D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirSRV[2];
    D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirUAV[2];
    uint32_t m_currentBufferIndex = 0;

    // Piecewise-constant volume for T* (Mip 2)
    ComPtr<ID3D12Resource> m_volumeMip2;
    D3D12_CPU_DESCRIPTOR_HANDLE m_volumeMip2SRV;

    // === Compute Pipelines ===

    // Phase 1: Path generation and RIS
    ComPtr<ID3D12PipelineState> m_pathGenerationPSO;
    ComPtr<ID3D12RootSignature> m_pathGenerationRS;

    // Final shading
    ComPtr<ID3D12PipelineState> m_shadingPSO;
    ComPtr<ID3D12RootSignature> m_shadingRS;

    // Phase 2: Spatial reuse (future)
    ComPtr<ID3D12PipelineState> m_spatialReusePSO;
    ComPtr<ID3D12RootSignature> m_spatialReuseRS;

    // Phase 3: Temporal reuse (future)
    ComPtr<ID3D12PipelineState> m_temporalReusePSO;
    ComPtr<ID3D12RootSignature> m_temporalReuseRS;

    // === Statistics ===
    uint32_t m_frameCount = 0;
};
