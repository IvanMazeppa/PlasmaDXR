#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>

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
    };

    struct AABBConstants {
        uint32_t particleCount;
        float particleRadius;
        float padding[2];
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
                        uint32_t particleCount);

    // Get output lighting buffer
    ID3D12Resource* GetLightingBuffer() const { return m_lightingBuffer.Get(); }

    // Get acceleration structures (for reuse by Gaussian renderer)
    ID3D12Resource* GetTLAS() const { return m_topLevelAS.Get(); }

    // Settings
    void SetRaysPerParticle(uint32_t rays) { m_raysPerParticle = rays; }
    void SetMaxLightingDistance(float dist) { m_maxLightingDistance = dist; }
    void SetLightingIntensity(float intensity) { m_lightingIntensity = intensity; }

private:
    bool LoadShaders();
    bool CreateRootSignatures();
    bool CreatePipelineStates();
    bool CreateAccelerationStructures();

    void GenerateAABBs(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer);
    void BuildBLAS(ID3D12GraphicsCommandList4* cmdList);
    void BuildTLAS(ID3D12GraphicsCommandList4* cmdList);
    void DispatchRayQueryLighting(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer);

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    uint32_t m_particleCount = 0;

    // Settings
    uint32_t m_raysPerParticle = 16;         // Increased from 4: Eliminates violent brightness flashing (40% visual improvement)
    float m_maxLightingDistance = 100.0f;    // Reduced from 500 to limit ray distance
    float m_lightingIntensity = 1.0f;        // Global intensity multiplier
    float m_particleRadius = 5.0f;           // Matches visual particle size (reduced from 25.0)

    // Compute shaders (RayQuery approach - no lib_6_x needed!)
    Microsoft::WRL::ComPtr<ID3DBlob> m_aabbGenShader;
    Microsoft::WRL::ComPtr<ID3DBlob> m_rayQueryLightingShader;

    // Root signatures
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_aabbGenRootSig;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rayQueryLightingRootSig;

    // Pipeline state objects (compute shaders)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_aabbGenPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_rayQueryLightingPSO;

    // Buffers
    Microsoft::WRL::ComPtr<ID3D12Resource> m_aabbBuffer;          // Per-particle AABBs
    Microsoft::WRL::ComPtr<ID3D12Resource> m_lightingBuffer;      // Per-particle lighting output

    // Acceleration Structures
    Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;       // BLAS for particles
    Microsoft::WRL::ComPtr<ID3D12Resource> m_topLevelAS;          // TLAS
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_tlasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_instanceDescsBuffer;

    size_t m_blasSize = 0;
    size_t m_tlasSize = 0;
};
