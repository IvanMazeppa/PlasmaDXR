#include "DLSSSystem.h"

#ifdef ENABLE_DLSS

#include "../utils/Logger.h"
#include <stdexcept>

// DLSS Project ID for development (placeholder - official app ID would come from NVIDIA)
#define DLSS_PROJECT_ID L"PlasmaDX-Clean"

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

    // Initialize NGX SDK
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
        DLSS_PROJECT_ID,
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,  // Custom engine
        NVSDK_NGX_ENGINE_VERSION(1, 0, 0),
        appDataPath,
        device
    );

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to initialize NGX SDK: 0x{:X}", static_cast<uint32_t>(result));
        return false;
    }

    LOG_INFO("DLSS: NGX SDK initialized successfully");

    // Get capability parameters
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_params);
    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to get capability parameters: 0x{:X}", static_cast<uint32_t>(result));
        NVSDK_NGX_D3D12_Shutdown();
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

        NVSDK_NGX_D3D12_Shutdown();
        return false;
    }

    // Check DLSS-RR (Ray Reconstruction / Denoiser) availability
    int rrAvailable = 0;
    result = m_params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &rrAvailable);

    m_rrSupported = (rrAvailable != 0);

    if (!m_rrSupported) {
        LOG_WARN("DLSS: Ray Reconstruction not supported on this GPU");
        NVSDK_NGX_D3D12_Shutdown();
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

    // Shutdown NGX
    NVSDK_NGX_D3D12_Shutdown();

    m_initialized = false;
    LOG_INFO("DLSS: Shutdown complete");
}

bool DLSSSystem::CreateRayReconstructionFeature(uint32_t width, uint32_t height) {
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
        LOG_ERROR("DLSS: Failed to allocate creation parameters: 0x{:X}", static_cast<uint32_t>(result));
        return false;
    }

    // Set required parameters for Ray Reconstruction
    creationParams->Set(NVSDK_NGX_Parameter_Width, width);
    creationParams->Set(NVSDK_NGX_Parameter_Height, height);
    creationParams->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                       NVSDK_NGX_DLSS_Feature_Flags_IsHDR);  // HDR support

    // Create Ray Reconstruction feature
    result = NVSDK_NGX_D3D12_CreateFeature(
        nullptr,  // Command list (can be nullptr for creation)
        NVSDK_NGX_Feature_RayReconstruction,
        creationParams,
        &m_rrFeature
    );

    NVSDK_NGX_D3D12_DestroyParameters(creationParams);

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to create Ray Reconstruction feature: 0x{:X}", static_cast<uint32_t>(result));
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
        LOG_ERROR("DLSS: Failed to allocate eval parameters: 0x{:X}", static_cast<uint32_t>(result));
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

    // Denoiser strength
    evalParams->Set(NVSDK_NGX_Parameter_Denoise_Strength, m_denoiserStrength);

    // Evaluate
    result = NVSDK_NGX_D3D12_EvaluateFeature(
        cmdList,
        m_rrFeature,
        evalParams,
        nullptr  // Callback (not needed)
    );

    NVSDK_NGX_D3D12_DestroyParameters(evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Ray Reconstruction evaluation failed: 0x{:X}", static_cast<uint32_t>(result));
        return false;
    }

    return true;
}

#endif // ENABLE_DLSS
