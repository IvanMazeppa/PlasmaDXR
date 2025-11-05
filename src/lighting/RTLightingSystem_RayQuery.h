#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <vector>
#include <string>

// RTLightingSystem using DXR 1.1 RayQuery (Inline Raytracing)
// Simpler than traditional DXR pipeline - no state objects or SBT needed
// Uses compute shader with RayQuery objects for hardware-accelerated ray tracing

class Device;
class ResourceManager;

class RTLightingSystem_RayQuery {
public:
    struct LightingConstants {
        uint32_t particleCount;
        uint32_t raysPerParticle;      // 8=high, 4=medium, 2=low quality
        float maxLightingDistance;      // Ray TMax (e.g., 20.0)
        float lightingIntensity;        // Global intensity multiplier

        // Dynamic emission parameters
        DirectX::XMFLOAT3 cameraPosition; // Camera position for distance-based effects
        uint32_t frameCount;              // Frame counter for temporal effects
        float emissionStrength;           // Global emission multiplier (0.0-1.0)
        float emissionThreshold;          // Temperature threshold for emission (K)
        float rtSuppression;              // How much RT lighting suppresses emission (0.0-1.0)
        float temporalRate;               // Temporal modulation frequency
    };

    struct AABBConstants {
        uint32_t particleCount;
        float particleRadius;
        // Phase 1.5 Adaptive Particle Radius
        uint32_t enableAdaptiveRadius;
        float adaptiveInnerZone;
        float adaptiveOuterZone;
        float adaptiveInnerScale;
        float adaptiveOuterScale;
        float densityScaleMin;
        float densityScaleMax;
        float padding;
    };

public:
    RTLightingSystem_RayQuery() = default;
    ~RTLightingSystem_RayQuery();

    // Initialize RT pipeline
    bool Initialize(Device* device, ResourceManager* resources, uint32_t particleCount);
    void Shutdown();

    // Main pipeline: AABB generation → BLAS/TLAS build → RT lighting compute
    void ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                        ID3D12Resource* particleBuffer,
                        uint32_t particleCount,
                        const DirectX::XMFLOAT3& cameraPosition);

    // Get output lighting buffer
    ID3D12Resource* GetLightingBuffer() const { return m_lightingBuffer.Get(); }

    // Get acceleration structures (for reuse by Gaussian renderer)
    // Phase 1: Returns appropriate TLAS based on particle count
    // <=2044 particles: Probe grid TLAS (single BLAS)
    // >2044 particles: Combined TLAS (2 BLAS instances)
    ID3D12Resource* GetTLAS() const {
        // If combined TLAS exists (>2044 particles), return it
        if (m_topLevelAS && m_directRTAS.tlas) {
            return m_topLevelAS.Get();  // Combined TLAS (2 instances)
        }
        // Otherwise return probe grid TLAS (<=2044 particles)
        return m_probeGridAS.tlas.Get();  // Single instance TLAS
    }

    // Dual AS support (Phase 1)
    ID3D12Resource* GetProbeGridTLAS() const { return m_probeGridAS.tlas.Get(); }
    ID3D12Resource* GetDirectRTTLAS() const { return m_directRTAS.tlas.Get(); }

    // Settings
    void SetRaysPerParticle(uint32_t rays) { m_raysPerParticle = rays; }
    void SetMaxLightingDistance(float dist) { m_maxLightingDistance = dist; }
    void SetLightingIntensity(float intensity) { m_lightingIntensity = intensity; }

    // Dynamic emission settings
    void SetEmissionStrength(float strength) { m_emissionStrength = strength; }
    void SetEmissionThreshold(float threshold) { m_emissionThreshold = threshold; }
    void SetRTSuppression(float suppression) { m_rtSuppression = suppression; }
    void SetTemporalRate(float rate) { m_temporalRate = rate; }

    // Phase 1.5 Adaptive Particle Radius
    void SetAdaptiveRadiusEnabled(bool enabled) { m_enableAdaptiveRadius = enabled; }
    void SetAdaptiveInnerZone(float zone) { m_adaptiveInnerZone = zone; }
    void SetAdaptiveOuterZone(float zone) { m_adaptiveOuterZone = zone; }
    void SetAdaptiveInnerScale(float scale) { m_adaptiveInnerScale = scale; }
    void SetAdaptiveOuterScale(float scale) { m_adaptiveOuterScale = scale; }
    void SetDensityScaleMin(float min) { m_densityScaleMin = min; }
    void SetDensityScaleMax(float max) { m_densityScaleMax = max; }

private:
    // ========================================================================
    // Dual AS Architecture - Type Definitions (Phase 1)
    // ========================================================================
    struct AccelerationStructureSet {
        Microsoft::WRL::ComPtr<ID3D12Resource> aabbBuffer;          // Per-particle AABBs
        Microsoft::WRL::ComPtr<ID3D12Resource> lightingBuffer;      // Per-particle lighting output
        Microsoft::WRL::ComPtr<ID3D12Resource> blas;                // Bottom-level AS
        Microsoft::WRL::ComPtr<ID3D12Resource> tlas;                // Top-level AS
        Microsoft::WRL::ComPtr<ID3D12Resource> blasScratch;         // BLAS build scratch
        Microsoft::WRL::ComPtr<ID3D12Resource> tlasScratch;         // TLAS build scratch
        Microsoft::WRL::ComPtr<ID3D12Resource> instanceDesc;        // TLAS instance descriptor
        uint32_t startParticle;                                     // First particle in this set
        uint32_t particleCount;                                     // Number of particles in this set
        size_t blasSize;                                            // BLAS size in bytes
        size_t tlasSize;                                            // TLAS size in bytes
    };

    static constexpr uint32_t PROBE_GRID_PARTICLE_LIMIT = 2044;    // Max particles before bug threshold

    // ========================================================================
    // Private Functions
    // ========================================================================
    bool LoadShaders();
    bool CreateRootSignatures();
    bool CreatePipelineStates();
    bool CreateAccelerationStructures();

    // Dual AS helpers (Phase 1)
    bool CreateAccelerationStructureSet(AccelerationStructureSet& asSet, uint32_t particleCount, const std::string& namePrefix);
    void GenerateAABBs_Dual(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer, uint32_t totalParticleCount);
    void BuildBLAS_ForSet(ID3D12GraphicsCommandList4* cmdList, AccelerationStructureSet& asSet, uint32_t particleOffset);
    void BuildTLAS_ForSet(ID3D12GraphicsCommandList4* cmdList, AccelerationStructureSet& asSet);
    void BuildCombinedTLAS(ID3D12GraphicsCommandList4* cmdList);  // Combined TLAS with 2 instances for full visibility

    // Legacy functions (will be removed after migration)
    void GenerateAABBs(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer);
    void BuildBLAS(ID3D12GraphicsCommandList4* cmdList);
    void BuildTLAS(ID3D12GraphicsCommandList4* cmdList);
    void DispatchRayQueryLighting(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer, const DirectX::XMFLOAT3& cameraPosition);

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    uint32_t m_particleCount = 0;

    // Settings
    uint32_t m_raysPerParticle = 16;         // Minimum for acceptable quality without denoising
    float m_maxLightingDistance = 100.0f;    // Reduced from 500 to limit ray distance
    float m_lightingIntensity = 1.0f;        // Global intensity multiplier
    float m_particleRadius = 5.0f;           // Matches visual particle size (reduced from 25.0)
    uint32_t m_frameCount = 0;               // Frame counter for temporal effects

    // Dynamic emission settings
    float m_emissionStrength = 0.25f;        // Global emission multiplier (tuned for balance)
    float m_emissionThreshold = 22000.0f;    // Only particles >22000K emit significantly
    float m_rtSuppression = 0.7f;            // RT lighting suppresses 70% of emission
    float m_temporalRate = 0.03f;            // Subtle temporal pulse rate

    // Phase 1.5 Adaptive Particle Radius
    bool m_enableAdaptiveRadius = true;
    float m_adaptiveInnerZone = 100.0f;
    float m_adaptiveOuterZone = 300.0f;
    float m_adaptiveInnerScale = 0.5f;
    float m_adaptiveOuterScale = 2.0f;
    float m_densityScaleMin = 0.3f;
    float m_densityScaleMax = 3.0f;

    // Compute shaders (RayQuery approach - no lib_6_x needed!)
    Microsoft::WRL::ComPtr<ID3DBlob> m_aabbGenShader;
    Microsoft::WRL::ComPtr<ID3DBlob> m_rayQueryLightingShader;

    // Root signatures
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_aabbGenRootSig;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rayQueryLightingRootSig;

    // Pipeline state objects (compute shaders)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_aabbGenPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_rayQueryLightingPSO;

    // Buffers (LEGACY - will be removed after migration)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_aabbBuffer;          // Per-particle AABBs
    Microsoft::WRL::ComPtr<ID3D12Resource> m_lightingBuffer;      // Per-particle lighting output

    // ============================================================================
    // Dual AS Members (Phase 1 - ACTIVE)
    // ============================================================================
    AccelerationStructureSet m_probeGridAS;   // AS #1: Particles 0-2043 (Probe Grid volumetric GI)
    AccelerationStructureSet m_directRTAS;    // AS #2: Particles 2044+ (Direct RT lighting)

    // ============================================================================
    // Legacy Acceleration Structures (Phase 0 - WILL BE REMOVED AFTER MIGRATION)
    // ============================================================================
    Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;       // BLAS for particles
    Microsoft::WRL::ComPtr<ID3D12Resource> m_topLevelAS;          // TLAS
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_tlasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_instanceDescsBuffer;

    size_t m_blasSize = 0;
    size_t m_tlasSize = 0;
};
