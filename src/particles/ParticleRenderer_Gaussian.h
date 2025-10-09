#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;

// ParticleRenderer_Gaussian - 3D Gaussian Splatting via Ray Tracing
// Renders particles as volumetric ellipsoids using inline ray tracing

class ParticleRenderer_Gaussian {
public:
    struct RenderConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT4X4 invViewProj;
        DirectX::XMFLOAT3 cameraPos;
        float time;
        DirectX::XMFLOAT3 cameraRight;
        float particleSize;
        DirectX::XMFLOAT3 cameraUp;
        float padding;
        DirectX::XMFLOAT3 cameraForward;
        uint32_t screenWidth;
        uint32_t screenHeight;
        float fovY;
        float aspectRatio;
        float padding2;

        // Physical emission toggles and strengths
        bool usePhysicalEmission = false;
        float emissionStrength = 1.0f;
        bool useDopplerShift = false;
        float dopplerStrength = 1.0f;
        bool useGravitationalRedshift = false;
        float redshiftStrength = 1.0f;
    };

public:
    ParticleRenderer_Gaussian() = default;
    ~ParticleRenderer_Gaussian();

    bool Initialize(Device* device,
                   ResourceManager* resources,
                   uint32_t particleCount);

    void Render(ID3D12GraphicsCommandList* cmdList,
               ID3D12Resource* particleBuffer,
               ID3D12Resource* rtLightingBuffer,
               const RenderConstants& constants);

    // Rebuild BLAS when particles change significantly
    void RebuildAccelerationStructure(ID3D12GraphicsCommandList* cmdList,
                                      ID3D12Resource* particleBuffer);

private:
    bool CreateRayTracingPipeline();
    bool CreateAccelerationStructures();

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    uint32_t m_particleCount = 0;

    // Ray tracing pipeline
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_rayTracingPSO;

    // Acceleration structures for ray tracing
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasBuffer;  // Bottom-level (particle AABBs)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_tlasBuffer;  // Top-level
    Microsoft::WRL::ComPtr<ID3D12Resource> m_aabbBuffer;  // AABB geometry

    // Output texture
    Microsoft::WRL::ComPtr<ID3D12Resource> m_outputTexture;
    D3D12_CPU_DESCRIPTOR_HANDLE m_outputUAV;

    // Scratch buffers for AS builds
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_tlasScratch;
};
