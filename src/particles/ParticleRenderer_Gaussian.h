#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;

// ParticleRenderer_Gaussian - 3D Gaussian Splatting via Inline Ray Tracing
// REUSES existing RTLightingSystem's BLAS/TLAS - no duplicate infrastructure!
// Just a compute shader that ray traces Gaussians and outputs to texture

class ParticleRenderer_Gaussian {
public:
    struct RenderConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT4X4 invViewProj;
        DirectX::XMFLOAT3 cameraPos;
        float particleRadius;
        DirectX::XMFLOAT3 cameraRight;
        float time;
        DirectX::XMFLOAT3 cameraUp;
        uint32_t screenWidth;
        DirectX::XMFLOAT3 cameraForward;
        uint32_t screenHeight;
        float fovY;
        float aspectRatio;
        uint32_t particleCount;
        float padding;

        // Physical emission toggles and strengths
        uint32_t usePhysicalEmission;
        float emissionStrength;
        uint32_t useDopplerShift;
        float dopplerStrength;
        uint32_t useGravitationalRedshift;
        float redshiftStrength;

        // RT system toggles for performance
        uint32_t useShadowRays;
        uint32_t useInScattering;
        uint32_t usePhaseFunction;
        float phaseStrength;
        float inScatterStrength;
        float rtLightingStrength;
        uint32_t useAnisotropicGaussians;
        float anisotropyStrength;
    };

public:
    ParticleRenderer_Gaussian() = default;
    ~ParticleRenderer_Gaussian();

    bool Initialize(Device* device,
                   ResourceManager* resources,
                   uint32_t particleCount,
                   uint32_t screenWidth,
                   uint32_t screenHeight);

    void Render(ID3D12GraphicsCommandList4* cmdList,
               ID3D12Resource* particleBuffer,
               ID3D12Resource* rtLightingBuffer,
               ID3D12Resource* tlas,  // From RTLightingSystem!
               const RenderConstants& constants);

    // Get output texture to copy to backbuffer
    ID3D12Resource* GetOutputTexture() const { return m_outputTexture.Get(); }

private:
    bool CreatePipeline();
    bool CreateOutputTexture(uint32_t width, uint32_t height);

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    uint32_t m_particleCount = 0;

    // Simple compute shader pipeline
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso;

    // Constant buffer (replaces root constants to avoid 64 DWORD limit)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_constantBuffer;
    void* m_constantBufferMapped = nullptr;

    // Output texture (UAV) - bound via descriptor table (typed UAV requirement)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_outputTexture;
    D3D12_CPU_DESCRIPTOR_HANDLE m_outputUAV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_outputUAVGPU;
};
