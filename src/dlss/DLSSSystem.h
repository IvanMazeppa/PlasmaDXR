#pragma once

#ifdef ENABLE_DLSS

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>

// DLSS SDK headers
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_params.h"

using Microsoft::WRL::ComPtr;

class DLSSSystem {
public:
    DLSSSystem();
    ~DLSSSystem();

    // Lifecycle
    bool Initialize(ID3D12Device* device, const wchar_t* appDataPath);
    void Shutdown();

    // Ray Reconstruction
    struct RayReconstructionParams {
        ID3D12Resource* inputNoisySignal;      // Shadow rays (low sample count)
        ID3D12Resource* inputDiffuseAlbedo;    // Particle color/albedo
        ID3D12Resource* inputSpecularAlbedo;   // Specular reflectance (optional)
        ID3D12Resource* inputNormals;          // Surface normals (optional for particles)
        ID3D12Resource* inputRoughness;        // Surface roughness (optional)
        ID3D12Resource* inputMotionVectors;    // Particle motion vectors
        ID3D12Resource* outputDenoisedSignal;  // Denoised result

        uint32_t width;
        uint32_t height;
        float jitterOffsetX;   // TAA jitter (we don't use TAA, set to 0)
        float jitterOffsetY;
    };

    bool CreateRayReconstructionFeature(ID3D12GraphicsCommandList* cmdList, uint32_t width, uint32_t height);
    bool EvaluateRayReconstruction(
        ID3D12GraphicsCommandList* cmdList,
        const RayReconstructionParams& params
    );

    // Feature detection
    bool IsRayReconstructionSupported() const { return m_rrSupported; }

    // Settings
    void SetDenoiserStrength(float strength) { m_denoiserStrength = strength; }
    float GetDenoiserStrength() const { return m_denoiserStrength; }

private:
    // DLSS handles
    NVSDK_NGX_Handle* m_rrFeature = nullptr;
    NVSDK_NGX_Parameter* m_params = nullptr;

    // Device
    ComPtr<ID3D12Device> m_device;

    // State
    bool m_initialized = false;
    bool m_rrSupported = false;
    float m_denoiserStrength = 1.0f;

    // Feature info
    uint32_t m_featureWidth = 0;
    uint32_t m_featureHeight = 0;
};

#endif // ENABLE_DLSS
