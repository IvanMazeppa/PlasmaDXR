#include "DLSSSystem.h"

#ifdef ENABLE_DLSS

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>  // For GetModuleFileNameW
#include "../utils/Logger.h"
#include <stdexcept>
#include <filesystem>  // For absolute path resolution
#include "nvsdk_ngx_helpers.h"  // For NGX_DLSS_GET_OPTIMAL_SETTINGS

// DLSS Project ID for development (narrow string, NOT wide string)
// IMPORTANT: ProjectID must be UUID/GUID format (hexadecimal characters only!)
// Generated UUID for PlasmaDX-Clean: 05227611-97e2-4462-af0b-ab7d47f06a86
#define DLSS_PROJECT_ID "05227611-97e2-4462-af0b-ab7d47f06a86"
#define DLSS_ENGINE_VERSION "1.0.0"

DLSSSystem::DLSSSystem() = default;

DLSSSystem::~DLSSSystem() {
    Shutdown();
}

bool DLSSSystem::Initialize(ID3D12Device* device, const wchar_t* appDataPath) {
    if (m_initialized) {
        LOG_WARN("DLSS already initialized");
        return true;
    }

    if (!device) {
        LOG_ERROR("DLSS: Invalid device pointer");
        return false;
    }

    m_device = device;

    // Configure NGX common features (logging, DLL paths)
    // IMPORTANT: Use absolute path - relative paths can fail NGX DLL loading
    static wchar_t exeDir[MAX_PATH] = {};
    if (exeDir[0] == 0) {
        // Get the directory where the executable is located
        GetModuleFileNameW(nullptr, exeDir, MAX_PATH);
        wchar_t* lastSlash = wcsrchr(exeDir, L'\\');
        if (lastSlash) *lastSlash = L'\0';  // Remove executable name, keep directory
        LOG_INFO("DLSS: Using DLL path: {}", std::filesystem::path(exeDir).string());
    }
    const wchar_t* dllPaths[] = { exeDir };  // Absolute path to exe directory
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.LoggingInfo.LoggingCallback = nullptr;  // Use default file logging
    featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;  // Enable debugging
    featureInfo.LoggingInfo.DisableOtherLoggingSinks = false;
    featureInfo.PathListInfo.Path = dllPaths;
    featureInfo.PathListInfo.Length = 1;

    // Initialize NGX SDK with Project ID
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
        DLSS_PROJECT_ID,                   // const char* (narrow string!)
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,      // Custom engine
        DLSS_ENGINE_VERSION,                // const char* engine version
        appDataPath,                        // const wchar_t* app data path
        device,                             // ID3D12Device*
        &featureInfo,                       // FeatureCommonInfo (with logging!)
        NVSDK_NGX_Version_API               // SDK version
    );

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to initialize NGX SDK: 0x{}", hexStr);
        return false;
    }

    LOG_INFO("DLSS: NGX SDK initialized successfully");

    // Get capability parameters
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_params);
    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to get capability parameters: 0x{}", hexStr);
        NVSDK_NGX_D3D12_Shutdown1(device);
        return false;
    }

    // Check for driver updates
    int needsUpdatedDriver = 0;
    unsigned int minDriverVersionMajor = 0;
    unsigned int minDriverVersionMinor = 0;

    result = m_params->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);

    if (needsUpdatedDriver) {
        m_params->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minDriverVersionMajor);
        m_params->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minDriverVersionMinor);

        LOG_WARN("DLSS: Driver update required. Minimum version: {}.{}",
                 minDriverVersionMajor, minDriverVersionMinor);

        NVSDK_NGX_D3D12_Shutdown1(device);
        return false;
    }

    // Check DLSS Super Resolution availability
    int dlssAvailable = 0;
    result = m_params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);

    m_dlssSupported = (dlssAvailable != 0);

    if (!m_dlssSupported) {
        LOG_WARN("DLSS: Super Resolution not supported on this GPU");
        NVSDK_NGX_D3D12_Shutdown1(device);
        return false;
    }

    LOG_INFO("DLSS: Super Resolution supported");
    m_initialized = true;
    return true;
}

void DLSSSystem::Shutdown() {
    if (!m_initialized) return;

    // Release DLSS Super Resolution feature
    if (m_dlssFeature) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_dlssFeature);
        m_dlssFeature = nullptr;
    }

    // Shutdown NGX (use Shutdown1 with device pointer, NOT deprecated Shutdown())
    if (m_device) {
        NVSDK_NGX_D3D12_Shutdown1(m_device.Get());
    }

    m_initialized = false;
    LOG_INFO("DLSS: Shutdown complete");
}

bool DLSSSystem::CreateSuperResolutionFeature(
    ID3D12GraphicsCommandList* cmdList,
    uint32_t renderWidth,
    uint32_t renderHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    DLSSQualityMode qualityMode
) {
    if (!cmdList) {
        LOG_ERROR("DLSS: Command list is required for feature creation");
        return false;
    }

    if (!m_initialized || !m_dlssSupported) {
        LOG_ERROR("DLSS: Not initialized or DLSS not supported");
        return false;
    }

    // Release existing feature if resolution or quality mode changed
    if (m_dlssFeature && (m_renderWidth != renderWidth || m_renderHeight != renderHeight ||
                          m_outputWidth != outputWidth || m_outputHeight != outputHeight ||
                          m_qualityMode != qualityMode)) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_dlssFeature);
        m_dlssFeature = nullptr;
        LOG_INFO("DLSS: Released feature due to resolution or quality mode change");
    }

    if (m_dlssFeature) {
        return true; // Already created for this resolution and quality mode
    }

    // Convert quality mode to NGX enum
    NVSDK_NGX_PerfQuality_Value perfQuality;
    switch (qualityMode) {
        case DLSSQualityMode::Quality:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            break;
        case DLSSQualityMode::Balanced:
            perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
            break;
        case DLSSQualityMode::Performance:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
            break;
        case DLSSQualityMode::UltraPerf:
            perfQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
            break;
        default:
            perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
    }

    // Use m_params for creation (already allocated in Initialize)
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Width, renderWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Height, renderHeight);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutWidth, outputWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutHeight, outputHeight);
    NVSDK_NGX_Parameter_SetI(m_params, NVSDK_NGX_Parameter_PerfQualityValue, perfQuality);

    // HDR flag (our renderer uses HDR R16G16B16A16_FLOAT)
    NVSDK_NGX_Parameter_SetI(m_params, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                             NVSDK_NGX_DLSS_Feature_Flags_IsHDR);

    // Create Super Resolution feature (NOT RayReconstruction!)
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_CreateFeature(
        cmdList,
        NVSDK_NGX_Feature_SuperSampling,  // ← SUPER RESOLUTION, not Ray Reconstruction
        m_params,
        &m_dlssFeature
    );

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to create Super Resolution feature: 0x{}", hexStr);
        return false;
    }

    m_renderWidth = renderWidth;
    m_renderHeight = renderHeight;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;
    m_qualityMode = qualityMode;  // Store quality mode

    LOG_INFO("DLSS: Super Resolution feature created successfully!");
    LOG_INFO("  Render: {}x{}, Output: {}x{}", renderWidth, renderHeight, outputWidth, outputHeight);
    return true;
}

bool DLSSSystem::EvaluateSuperResolution(
    ID3D12GraphicsCommandList* cmdList,
    const SuperResolutionParams& params
) {
    if (!m_dlssFeature) {
        LOG_ERROR("DLSS: Super Resolution feature not created");
        return false;
    }

    if (!cmdList) {
        LOG_ERROR("DLSS: Invalid command list");
        return false;
    }

    if (!params.inputColor || !params.outputUpscaled) {
        LOG_ERROR("DLSS: Missing required input/output buffers");
        return false;
    }

    // Set input color (render resolution)
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, NVSDK_NGX_Parameter_Color,
                                         params.inputColor);

    // Set output upscaled (target resolution)
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params, NVSDK_NGX_Parameter_Output,
                                         params.outputUpscaled);

    // Motion vectors (optional, zeros OK)
    if (params.inputMotionVectors) {
        NVSDK_NGX_Parameter_SetD3d12Resource(m_params, NVSDK_NGX_Parameter_MotionVectors,
                                             params.inputMotionVectors);

        // CRITICAL: MV scaling with RENDER resolution (not output resolution!)
        float mvScaleX = 1.0f / params.renderWidth;
        float mvScaleY = 1.0f / params.renderHeight;
        NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_MV_Scale_X, mvScaleX);
        NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_MV_Scale_Y, mvScaleY);
    }

    // Depth (optional, improves quality)
    if (params.inputDepth) {
        NVSDK_NGX_Parameter_SetD3d12Resource(m_params, NVSDK_NGX_Parameter_Depth,
                                             params.inputDepth);
    }

    // Resolution parameters
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Width, params.renderWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Height, params.renderHeight);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutWidth, params.outputWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutHeight, params.outputHeight);

    // Sharpness
    NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Sharpness, params.sharpness);

    // Jitter (always 0 for non-TAA)
    NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Jitter_Offset_X, params.jitterOffsetX);
    NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Jitter_Offset_Y, params.jitterOffsetY);

    // Reset flag
    NVSDK_NGX_Parameter_SetI(m_params, NVSDK_NGX_Parameter_Reset, params.reset);

    // Evaluate DLSS Super Resolution
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_EvaluateFeature(
        cmdList,
        m_dlssFeature,
        m_params,
        nullptr
    );

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Super Resolution evaluation failed: 0x{}", hexStr);
        return false;
    }

    return true;
}

#endif // ENABLE_DLSS
