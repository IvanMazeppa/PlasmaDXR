#include "DLSSSystem.h"

#ifdef ENABLE_DLSS

#include "../utils/Logger.h"
#include <stdexcept>
#include "nvsdk_ngx_defs_dlssd.h"  // Ray Reconstruction definitions

// DLSS Project ID for development (narrow string, NOT wide string)
// IMPORTANT: ProjectID must be UUID/GUID format (hexadecimal characters only!)
// Generated UUID for PlasmaDX-Clean: a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d
#define DLSS_PROJECT_ID "a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
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
    const wchar_t* dllPaths[] = { L"." };  // Search local directory first
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

    // Check DLSS-RR (Ray Reconstruction / Denoiser) availability
    int rrAvailable = 0;
    result = m_params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &rrAvailable);

    m_rrSupported = (rrAvailable != 0);

    if (!m_rrSupported) {
        LOG_WARN("DLSS: Ray Reconstruction not supported on this GPU");
        NVSDK_NGX_D3D12_Shutdown1(device);
        return false;
    }

    LOG_INFO("DLSS: Ray Reconstruction supported");
    m_initialized = true;
    return true;
}

void DLSSSystem::Shutdown() {
    if (!m_initialized) return;

    // Release Ray Reconstruction feature
    if (m_rrFeature) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_rrFeature);
        m_rrFeature = nullptr;
    }

    // Shutdown NGX (use Shutdown1 with device pointer, NOT deprecated Shutdown())
    if (m_device) {
        NVSDK_NGX_D3D12_Shutdown1(m_device.Get());
    }

    m_initialized = false;
    LOG_INFO("DLSS: Shutdown complete");
}

bool DLSSSystem::CreateRayReconstructionFeature(ID3D12GraphicsCommandList* cmdList, uint32_t width, uint32_t height) {
    if (!cmdList) {
        LOG_ERROR("DLSS: Command list is required for feature creation");
        return false;
    }

    if (!m_initialized || !m_rrSupported) {
        LOG_ERROR("DLSS: Not initialized or RR not supported");
        return false;
    }

    // Release existing feature if resolution changed
    if (m_rrFeature && (m_featureWidth != width || m_featureHeight != height)) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_rrFeature);
        m_rrFeature = nullptr;
    }

    if (m_rrFeature) {
        return true; // Already created for this resolution
    }

    // Set creation parameters
    NVSDK_NGX_Parameter* creationParams = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_AllocateParameters(&creationParams);

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to allocate creation parameters: 0x{}", hexStr);
        return false;
    }

    // Set required parameters for Ray Reconstruction (MUST use typed setters!)
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_Width, width);
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_Height, height);
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_OutWidth, width);
    NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_OutHeight, height);
    NVSDK_NGX_Parameter_SetI(creationParams, NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_Balanced);
    // CRITICAL: Must set MVLowRes flag for Ray Reconstruction
    NVSDK_NGX_Parameter_SetI(creationParams, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                             NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
    // CRITICAL: Denoise mode MUST be set to DLUnified for Ray Reconstruction
    NVSDK_NGX_Parameter_SetI(creationParams, NVSDK_NGX_Parameter_DLSS_Denoise_Mode,
                             NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);

    // Create Ray Reconstruction feature with VALID command list
    result = NVSDK_NGX_D3D12_CreateFeature(
        cmdList,  // MUST be valid command list!
        NVSDK_NGX_Feature_RayReconstruction,
        creationParams,
        &m_rrFeature
    );

    NVSDK_NGX_D3D12_DestroyParameters(creationParams);

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to create Ray Reconstruction feature: 0x{}", hexStr);
        return false;
    }

    m_featureWidth = width;
    m_featureHeight = height;

    LOG_INFO("DLSS: Ray Reconstruction feature created ({}x{})", width, height);
    return true;
}

bool DLSSSystem::EvaluateRayReconstruction(
    ID3D12GraphicsCommandList* cmdList,
    const RayReconstructionParams& params
) {
    if (!m_rrFeature) {
        LOG_ERROR("DLSS: Ray Reconstruction feature not created");
        return false;
    }

    if (!cmdList) {
        LOG_ERROR("DLSS: Invalid command list");
        return false;
    }

    if (!params.inputNoisySignal || !params.outputDenoisedSignal) {
        LOG_ERROR("DLSS: Missing required input/output buffers");
        return false;
    }

    // Set evaluation parameters
    NVSDK_NGX_Parameter* evalParams = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_AllocateParameters(&evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Failed to allocate eval parameters: 0x{}", hexStr);
        return false;
    }

    // Input/Output resources
    evalParams->Set(NVSDK_NGX_Parameter_Color, params.inputNoisySignal);
    evalParams->Set(NVSDK_NGX_Parameter_Output, params.outputDenoisedSignal);

    // Optional inputs (improve quality if provided)
    if (params.inputDiffuseAlbedo) {
        evalParams->Set(NVSDK_NGX_Parameter_GBuffer_Albedo, params.inputDiffuseAlbedo);
    }
    if (params.inputNormals) {
        evalParams->Set(NVSDK_NGX_Parameter_GBuffer_Normals, params.inputNormals);
    }
    if (params.inputRoughness) {
        evalParams->Set(NVSDK_NGX_Parameter_GBuffer_Roughness, params.inputRoughness);
    }
    if (params.inputMotionVectors) {
        evalParams->Set(NVSDK_NGX_Parameter_MotionVectors, params.inputMotionVectors);
    }

    // Jitter offset (we don't use TAA, set to 0)
    evalParams->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, params.jitterOffsetX);
    evalParams->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, params.jitterOffsetY);

    // Denoiser strength (parameter name is "Denoise" not "Denoise_Strength")
    evalParams->Set(NVSDK_NGX_Parameter_Denoise, m_denoiserStrength);

    // Evaluate
    result = NVSDK_NGX_D3D12_EvaluateFeature(
        cmdList,
        m_rrFeature,
        evalParams,
        nullptr  // Callback (not needed)
    );

    NVSDK_NGX_D3D12_DestroyParameters(evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Ray Reconstruction evaluation failed: 0x{}", hexStr);
        return false;
    }

    return true;
}

#endif // ENABLE_DLSS
