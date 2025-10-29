#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <vector>

// Forward declarations
class Device;
class ResourceManager;

#ifdef ENABLE_DLSS
#include "../dlss/DLSSSystem.h"  // Need full definition for DLSSQualityMode enum
#endif

// ParticleRenderer_Gaussian - 3D Gaussian Splatting via Inline Ray Tracing
// REUSES existing RTLightingSystem's BLAS/TLAS - no duplicate infrastructure!
// Just a compute shader that ray traces Gaussians and outputs to texture

class ParticleRenderer_Gaussian {
public:
    // Light structure matching HLSL (64 bytes with god ray parameters)
    struct Light {
        // === Base Light Properties (32 bytes) ===
        DirectX::XMFLOAT3 position;        // 12 bytes
        float intensity;                   // 4 bytes
        DirectX::XMFLOAT3 color;          // 12 bytes
        float radius;                      // 4 bytes

        // === God Ray Parameters (32 bytes) ===
        float enableGodRays;              // 4 bytes (0.0=disabled, 1.0=enabled)
        float godRayIntensity;            // 4 bytes (brightness 0.0-10.0)
        float godRayLength;               // 4 bytes (beam length 100.0-5000.0 units)
        float godRayFalloff;              // 4 bytes (radial falloff 0.1-10.0, higher=sharper)
        DirectX::XMFLOAT3 godRayDirection; // 12 bytes (normalized beam direction)
        float godRayConeAngle;            // 4 bytes (half-angle in radians 0.0-1.57)
        float godRayRotationSpeed;        // 4 bytes (rotation rad/s, 0=static)
        float _padding;                   // 4 bytes (GPU alignment)
        // Total: 64 bytes (GPU-aligned)
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

        // Multi-light system
        uint32_t lightCount;               // Number of active lights (0-16)
        DirectX::XMFLOAT3 padding3;        // Padding for alignment

        // PCSS soft shadow system
        uint32_t shadowRaysPerLight;       // 1 (performance), 4 (balanced), 8 (quality)
        uint32_t enableTemporalFiltering;  // Temporal accumulation for soft shadows
        float temporalBlend;               // Blend factor for temporal filtering (0.0-1.0)
        uint32_t useRTXDI;                 // 0=multi-light (13 lights), 1=RTXDI (1 sampled light)
        uint32_t debugRTXDISelection;      // DEBUG: Visualize selected light index (0=off, 1=on)
        DirectX::XMFLOAT3 debugPadding;    // Padding for alignment

        // === God Ray System (Phase 5 Milestone 5.3c) ===
        float godRayDensity;               // Global god ray density (0.0-1.0, ambient medium)
        float godRayStepMultiplier;        // Ray march step multiplier (0.5-2.0, quality vs speed)
        DirectX::XMFLOAT2 godRayPadding;   // Padding for alignment

        // === Phase 1 Lighting Fix ===
        float rtMinAmbient;                // Global ambient term (0.0-0.2) to prevent completely black particles
        DirectX::XMFLOAT3 lightingPadding; // Padding for alignment

        // === Phase 1.5 Adaptive Particle Radius ===
        uint32_t enableAdaptiveRadius;     // Toggle for density/distance-based radius scaling
        float adaptiveInnerZone;           // Distance threshold for inner shrinking (0-200 units)
        float adaptiveOuterZone;           // Distance threshold for outer expansion (200-600 units)
        float adaptiveInnerScale;          // Min scale for inner dense regions (0.1-1.0)
        float adaptiveOuterScale;          // Max scale for outer sparse regions (1.0-3.0)
        float densityScaleMin;             // Min density scale clamp (0.1-1.0)
        float densityScaleMax;             // Max density scale clamp (1.0-5.0)
        float adaptivePadding;             // Padding for alignment
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
               const RenderConstants& constants,
               ID3D12Resource* rtxdiOutputBuffer = nullptr);  // RTXDI selected lights (optional)

    // Resize output textures and buffers when window size changes
    bool Resize(uint32_t newWidth, uint32_t newHeight);

    // Update lights (call this before Render if lights changed)
    void UpdateLights(const std::vector<Light>& lights);

    // Get output texture to copy to backbuffer
    ID3D12Resource* GetOutputTexture() const { return m_outputTexture.Get(); }

    // Get output SRV for blit pass (HDR→SDR conversion)
#ifdef ENABLE_DLSS
    D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const {
        // If DLSS succeeded, return upscaled output, otherwise render-res output
        if (m_dlssSystem && m_dlssFeatureCreated && m_upscaledOutputSRVGPU.ptr != 0) {
            return m_upscaledOutputSRVGPU;
        }
        return m_outputSRVGPU;
    }
#else
    D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const { return m_outputSRVGPU; }
#endif

#ifdef ENABLE_DLSS
    // Set DLSS system reference for lazy feature creation
    void SetDLSSSystem(DLSSSystem* dlss, uint32_t width, uint32_t height) {
        m_dlssSystem = dlss;
        m_dlssWidth = width;
        m_dlssHeight = height;
    }
#endif

private:
    bool CreatePipeline();
    bool CreateOutputTexture(uint32_t width, uint32_t height);

#ifdef ENABLE_DLSS
    bool CreateMotionVectorPipeline();
#endif

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

    // RTXDI output buffer cache (to prevent descriptor leak)
    D3D12_CPU_DESCRIPTOR_HANDLE m_rtxdiSRV = {};                  // Cached SRV for RTXDI output
    D3D12_GPU_DESCRIPTOR_HANDLE m_rtxdiSRVGPU = {};               // Cached GPU handle

#ifdef ENABLE_DLSS
    // DLSS Super Resolution system (lazy feature creation)
    DLSSSystem* m_dlssSystem = nullptr;       // Not owned (pointer to Application's DLSSSystem)
    bool m_dlssFeatureCreated = false;        // Track lazy creation
    uint32_t m_dlssWidth = 0;                 // Feature creation width
    uint32_t m_dlssHeight = 0;                // Feature creation height

    // Super Resolution parameters
    DLSSSystem::DLSSQualityMode m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Balanced;
    uint32_t m_renderWidth = 0;   // Internal render resolution (e.g., 1280×720)
    uint32_t m_renderHeight = 0;
    uint32_t m_outputWidth = 0;   // Final display resolution (e.g., 1920×1080)
    uint32_t m_outputHeight = 0;
    bool m_dlssFirstFrame = true; // For reset flag
    float m_dlssSharpness = 0.0f; // DLSS sharpness setting

    // Motion vector buffer for DLSS temporal denoising (RG16_FLOAT)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_motionVectorBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE m_motionVectorSRV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_motionVectorSRVGPU;
    D3D12_CPU_DESCRIPTOR_HANDLE m_motionVectorUAV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_motionVectorUAVGPU;

    // Upscaled output texture (DLSS writes here at full resolution)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_upscaledOutputTexture;
    D3D12_CPU_DESCRIPTOR_HANDLE m_upscaledOutputSRV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_upscaledOutputSRVGPU;

    // Depth buffer for DLSS (R32_FLOAT, optional for SR)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_depthBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE m_depthUAV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_depthUAVGPU;
    D3D12_CPU_DESCRIPTOR_HANDLE m_depthSRV;
    D3D12_GPU_DESCRIPTOR_HANDLE m_depthSRVGPU;

    // Motion vector compute pipeline
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_motionVectorPSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_motionVectorRootSig;
#endif
};
