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

    // DLSS Quality Modes
    enum class DLSSQualityMode {
        Quality,        // 1.56× boost (67% render resolution)
        Balanced,       // 2.25× boost (58% render resolution) - RECOMMENDED
        Performance,    // 4.0× boost (50% render resolution)
        UltraPerf       // 7.1× boost (33% render resolution)
    };

    // Super Resolution (AI Upscaling)
    struct SuperResolutionParams {
        ID3D12Resource* inputColor;         // Low-res render (e.g., 720p)
        ID3D12Resource* outputUpscaled;     // High-res output (e.g., 1080p)
        ID3D12Resource* inputMotionVectors; // Optional (zeros acceptable)
        ID3D12Resource* inputDepth;         // Optional (improves quality)

        uint32_t renderWidth;      // Input resolution (e.g., 1280)
        uint32_t renderHeight;     // Input resolution (e.g., 720)
        uint32_t outputWidth;      // Output resolution (e.g., 1920)
        uint32_t outputHeight;     // Output resolution (e.g., 1080)

        float jitterOffsetX;       // Set to 0 (no TAA)
        float jitterOffsetY;
        float sharpness;           // 0.0-1.0, default 0.0
        int reset;                 // 1 = clear history, 0 = accumulate
    };

    bool CreateSuperResolutionFeature(
        ID3D12GraphicsCommandList* cmdList,
        uint32_t renderWidth,
        uint32_t renderHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        DLSSQualityMode qualityMode
    );

    bool EvaluateSuperResolution(
        ID3D12GraphicsCommandList* cmdList,
        const SuperResolutionParams& params
    );

    // Feature detection
    bool IsSuperResolutionSupported() const { return m_dlssSupported; }

    // Helper to expose parameters for NGX_DLSS_GET_OPTIMAL_SETTINGS
    NVSDK_NGX_Parameter* GetParameters() const { return m_params; }

    // Settings
    void SetSharpness(float sharpness) { m_sharpness = sharpness; }
    float GetSharpness() const { return m_sharpness; }

private:
    // DLSS handles
    NVSDK_NGX_Handle* m_dlssFeature = nullptr;  // Super Resolution feature
    NVSDK_NGX_Parameter* m_params = nullptr;

    // Device
    ComPtr<ID3D12Device> m_device;

    // State
    bool m_initialized = false;
    bool m_dlssSupported = false;
    float m_sharpness = 0.0f;

    // Feature info
    uint32_t m_renderWidth = 0;
    uint32_t m_renderHeight = 0;
    uint32_t m_outputWidth = 0;
    uint32_t m_outputHeight = 0;
};

#endif // ENABLE_DLSS
