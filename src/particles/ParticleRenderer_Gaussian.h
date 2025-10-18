#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <vector>

// Forward declarations
class Device;
class ResourceManager;

// ParticleRenderer_Gaussian - 3D Gaussian Splatting via Inline Ray Tracing
// REUSES existing RTLightingSystem's BLAS/TLAS - no duplicate infrastructure!
// Just a compute shader that ray traces Gaussians and outputs to texture

class ParticleRenderer_Gaussian {
public:
    // Light structure matching HLSL (32 bytes)
    struct Light {
        DirectX::XMFLOAT3 position;        // 12 bytes
        float intensity;                   // 4 bytes
        DirectX::XMFLOAT3 color;          // 12 bytes
        float radius;                      // 4 bytes
    };

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
        float emissionBlendFactor;  // 0.0 = pure artistic, 1.0 = pure physical
        float padding2;

        // RT system toggles for performance
        uint32_t useShadowRays;
        uint32_t useInScattering;
        uint32_t usePhaseFunction;
        float phaseStrength;
        float inScatterStrength;
        float rtLightingStrength;
        uint32_t useAnisotropicGaussians;
        float anisotropyStrength;

        // ReSTIR toggles and parameters
        uint32_t useReSTIR;
        uint32_t restirInitialCandidates;  // M = number of candidates to test (16-32)
        uint32_t frameIndex;               // For temporal validation
        float restirTemporalWeight;        // How much to trust previous frame (0-1)

        // Multi-light system
        uint32_t lightCount;               // Number of active lights (0-16)
        DirectX::XMFLOAT3 padding3;        // Padding for alignment

        // PCSS soft shadow system
        uint32_t shadowRaysPerLight;       // 1 (performance), 4 (balanced), 8 (quality)
        uint32_t enableTemporalFiltering;  // Temporal accumulation for soft shadows
        float temporalBlend;               // Blend factor for temporal filtering (0.0-1.0)
        float padding4;                    // Alignment
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

    // Resize output textures and buffers when window size changes
    bool Resize(uint32_t newWidth, uint32_t newHeight);

    // Update lights (call this before Render if lights changed)
    void UpdateLights(const std::vector<Light>& lights);

    // Get output texture to copy to backbuffer
    ID3D12Resource* GetOutputTexture() const { return m_outputTexture.Get(); }

    // Get output SRV for blit pass (HDR→SDR conversion)
    D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const { return m_outputSRVGPU; }

    // Get ReSTIR reservoir buffers for debugging/analysis
    ID3D12Resource* GetCurrentReservoirs() const {
        return m_reservoirBuffer[m_currentReservoirIndex].Get();
    }
    ID3D12Resource* GetPrevReservoirs() const {
        return m_reservoirBuffer[1 - m_currentReservoirIndex].Get();
    }

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
    D3D12_CPU_DESCRIPTOR_HANDLE m_outputSRV;      // SRV for blit pass (read HDR in pixel shader)
    D3D12_GPU_DESCRIPTOR_HANDLE m_outputSRVGPU;

    // ReSTIR reservoir buffers (ping-pong between frames)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_reservoirBuffer[2];  // Double-buffered
    D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirSRV[2];                // SRV for reading previous frame
    D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirSRVGPU[2];
    D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirUAV[2];                // UAV for writing current frame
    D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirUAVGPU[2];
    uint32_t m_currentReservoirIndex = 0;                         // Which buffer is current (0 or 1)
    uint32_t m_screenWidth = 0;
    uint32_t m_screenHeight = 0;

    // Multi-light system
    Microsoft::WRL::ComPtr<ID3D12Resource> m_lightBuffer;         // Structured buffer for lights
    D3D12_CPU_DESCRIPTOR_HANDLE m_lightSRV;                       // SRV for shader access (t4)
    D3D12_GPU_DESCRIPTOR_HANDLE m_lightSRVGPU;
    void* m_lightBufferMapped = nullptr;                          // CPU access for updates

    // PCSS soft shadow system (temporal filtering)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_shadowBuffer[2];     // Ping-pong shadow buffers (R16_FLOAT)
    D3D12_CPU_DESCRIPTOR_HANDLE m_shadowSRV[2];                   // SRV for reading previous shadow
    D3D12_GPU_DESCRIPTOR_HANDLE m_shadowSRVGPU[2];
    D3D12_CPU_DESCRIPTOR_HANDLE m_shadowUAV[2];                   // UAV for writing current shadow
    D3D12_GPU_DESCRIPTOR_HANDLE m_shadowUAVGPU[2];
    uint32_t m_currentShadowIndex = 0;                            // Which buffer is current (0 or 1)
};
