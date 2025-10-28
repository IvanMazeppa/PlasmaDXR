# DLSS 4.0 Integration Plan for PlasmaDX-Clean

**Date:** 2025-10-28
**Branch:** 0.10.12
**SDK Location:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/`
**Target:** Integrate NVIDIA DLSS 4.0 Ray Reconstruction for 2-4× raytracing performance

---

## Executive Summary

Integrate NVIDIA DLSS 4.0 SDK with focus on **Ray Reconstruction** (DLSS-RR) to dramatically reduce shadow ray count while improving visual quality through AI-powered denoising.

**Key Benefits:**
- Reduce shadow rays from 8 → 1-2 per light
- AI denoising better than temporal filtering
- 2-4× raytracing performance improvement
- Better quality at lower ray counts

---

## DLSS SDK Structure (Verified)

### Headers
- **Location:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/include/`
- **Main Header:** `nvsdk_ngx.h`
- **Ray Reconstruction:** `nvsdk_ngx_defs_dlssd.h` (DLSS-D = Denoiser)

### Libraries (VS2022 Compatible)
- **Dynamic:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib`
- **Debug:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib`

### Runtime DLLs
- **Development:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/lib/Windows_x86_64/dev/`
  - `nvngx_dlss.dll` - Super Resolution
  - `nvngx_dlssd.dll` - Ray Reconstruction (Denoiser) ← **WE USE THIS**
  - `nvngx_dlssg.dll` - Frame Generation

- **Release:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/lib/Windows_x86_64/rel/`

### Documentation
- **Ray Reconstruction Guide:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/doc/DLSS-RR Integration Guide.pdf`
- **General Guide:** `/mnt/d/Users/dilli/AndroidStudioProjects/dlss/doc/DLSS_Programming_Guide_Release.pdf`

---

## Integration Architecture

### Clean Module Pattern

Following PlasmaDX-Clean's architecture, create a dedicated DLSS subsystem:

```
src/
└── dlss/
    ├── DLSSSystem.h          # DLSS context, initialization, evaluation
    └── DLSSSystem.cpp        # Implementation
```

**Why separate module:**
- Single Responsibility Principle
- Easy to disable/remove
- No god object anti-pattern
- Matches existing: `RTLightingSystem`, `RTXDILightingSystem`, `AdaptiveQualitySystem`

---

## Implementation Steps

### **Phase 1: Build System Integration (Day 1)**

#### 1.1 Update CMakeLists.txt

Add DLSS SDK paths and link libraries:

```cmake
# DLSS SDK Configuration
set(DLSS_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dlss")

if(NOT EXISTS "${DLSS_SDK_DIR}/include/nvsdk_ngx.h")
    message(WARNING "DLSS SDK not found at ${DLSS_SDK_DIR}. DLSS features will be disabled.")
    set(ENABLE_DLSS OFF)
else()
    message(STATUS "DLSS SDK found: ${DLSS_SDK_DIR}")
    set(ENABLE_DLSS ON)
endif()

# Add DLSS sources (if enabled)
if(ENABLE_DLSS)
    list(APPEND SOURCES
        src/dlss/DLSSSystem.cpp
    )
    list(APPEND HEADERS
        src/dlss/DLSSSystem.h
    )
endif()

# Include DLSS headers
if(ENABLE_DLSS)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${DLSS_SDK_DIR}/include
    )
endif()

# Link DLSS library
if(ENABLE_DLSS)
    target_link_libraries(${PROJECT_NAME}
        ${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d$<$<CONFIG:Debug>:_dbg>.lib
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        ENABLE_DLSS
    )
endif()

# Copy DLSS DLLs to output
if(ENABLE_DLSS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_SDK_DIR}/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlssd.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMENT "Copying DLSS Ray Reconstruction DLL"
    )
endif()
```

#### 1.2 Verify Build

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
cmake -S . -B build
cmake --build build --config Debug
```

---

### **Phase 2: DLSSSystem Class (Day 1-2)**

#### 2.1 Create Header (`src/dlss/DLSSSystem.h`)

```cpp
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

    bool CreateRayReconstructionFeature(uint32_t width, uint32_t height);
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
```

#### 2.2 Create Implementation (`src/dlss/DLSSSystem.cpp`)

```cpp
#include "DLSSSystem.h"

#ifdef ENABLE_DLSS

#include "../utils/Logger.h"
#include <stdexcept>

// DLSS Application ID (placeholder - need to get from NVIDIA)
// For development, use project ID instead
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

    m_device = device;

    // Initialize NGX SDK
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
        DLSS_PROJECT_ID,
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,  // Custom engine
        NVSDK_NGX_ENGINE_VERSION_1_0_0,
        appDataPath,
        device
    );

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to initialize NGX SDK: {}", result);
        return false;
    }

    LOG_INFO("DLSS: NGX SDK initialized successfully");

    // Get capability parameters
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_params);
    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to get capability parameters: {}", result);
        NVSDK_NGX_D3D12_Shutdown();
        return false;
    }

    // Check Ray Reconstruction support
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

    // Check DLSS-RR (Denoiser) availability
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
        LOG_ERROR("DLSS: Failed to allocate creation parameters");
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
        LOG_ERROR("DLSS: Failed to create Ray Reconstruction feature: {}", result);
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

    // Set evaluation parameters
    NVSDK_NGX_Parameter* evalParams = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_AllocateParameters(&evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to allocate eval parameters");
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
        LOG_ERROR("DLSS: Ray Reconstruction evaluation failed: {}", result);
        return false;
    }

    return true;
}

#endif // ENABLE_DLSS
```

---

### **Phase 3: Application Integration (Day 2-3)**

#### 3.1 Update Application.h

Add DLSS system member:

```cpp
#ifdef ENABLE_DLSS
#include "../dlss/DLSSSystem.h"
#endif

class Application {
    // ... existing members ...

private:
    #ifdef ENABLE_DLSS
    std::unique_ptr<DLSSSystem> m_dlssSystem;
    #endif

    // DLSS settings
    bool m_enableDLSS = true;           // Master toggle
    float m_dlssDenoiserStrength = 1.0f;
};
```

#### 3.2 Update Application.cpp - Initialization

```cpp
bool Application::Initialize(HINSTANCE hInstance, int nCmdShow, int argc, char** argv) {
    // ... existing initialization ...

    #ifdef ENABLE_DLSS
    // Initialize DLSS system
    m_dlssSystem = std::make_unique<DLSSSystem>();

    std::wstring appDataPath = L"./"; // Or use proper path
    if (m_dlssSystem->Initialize(m_device->GetDevice(), appDataPath.c_str())) {
        LOG_INFO("DLSS System initialized successfully");

        // Create Ray Reconstruction feature
        if (m_dlssSystem->CreateRayReconstructionFeature(m_width, m_height)) {
            LOG_INFO("DLSS Ray Reconstruction ready");
        } else {
            LOG_WARN("DLSS Ray Reconstruction creation failed");
            m_enableDLSS = false;
        }
    } else {
        LOG_WARN("DLSS System initialization failed - features disabled");
        m_enableDLSS = false;
    }
    #endif

    return true;
}
```

---

### **Phase 4: Shadow Ray Integration (Day 3-4)**

This is where the magic happens! We replace high-sample-count shadow rays with DLSS-denoised low-sample-count rays.

**Current Pipeline:**
```
Shadow Rays (8 samples) → Temporal Filtering → Final Shadow
```

**DLSS Pipeline:**
```
Shadow Rays (1-2 samples) → DLSS Ray Reconstruction → Final Shadow
```

#### 4.1 Modify Gaussian Renderer

In `ParticleRenderer_Gaussian.cpp`, add DLSS path:

```cpp
void ParticleRenderer_Gaussian::Render(/* params */) {
    // ... existing code ...

    #ifdef ENABLE_DLSS
    if (m_enableDLSS && dlssSystem && dlssSystem->IsRayReconstructionSupported()) {
        // Low sample count (1-2 rays per light)
        RenderShadowRays(cmdList, /*rayCount=*/1);

        // Apply DLSS Ray Reconstruction
        DLSSSystem::RayReconstructionParams rrParams = {};
        rrParams.inputNoisySignal = m_shadowBuffer.Get();
        rrParams.outputDenoisedSignal = m_shadowBufferDenoised.Get();
        rrParams.inputMotionVectors = m_motionVectorBuffer.Get();
        rrParams.width = m_width;
        rrParams.height = m_height;
        rrParams.jitterOffsetX = 0.0f;
        rrParams.jitterOffsetY = 0.0f;

        dlssSystem->EvaluateRayReconstruction(cmdList, rrParams);

        // Use denoised buffer
        useShadowBuffer = m_shadowBufferDenoised.Get();
    } else
    #endif
    {
        // Traditional path (8 rays + temporal filtering)
        RenderShadowRays(cmdList, /*rayCount=*/8);
        useShadowBuffer = m_shadowBuffer.Get();
    }

    // Final composition
    CompositeFinalImage(cmdList, useShadowBuffer);
}
```

---

### **Phase 5: ImGui Controls (Day 4)**

Add DLSS controls to ImGui interface in `Application.cpp`:

```cpp
void Application::RenderImGui() {
    // ... existing ImGui code ...

    #ifdef ENABLE_DLSS
    if (ImGui::CollapsingHeader("DLSS Settings")) {
        if (m_dlssSystem && m_dlssSystem->IsRayReconstructionSupported()) {
            bool dlssToggled = ImGui::Checkbox("Enable DLSS Ray Reconstruction", &m_enableDLSS);

            if (m_enableDLSS) {
                float strength = m_dlssSystem->GetDenoiserStrength();
                if (ImGui::SliderFloat("Denoiser Strength", &strength, 0.0f, 2.0f)) {
                    m_dlssSystem->SetDenoiserStrength(strength);
                }

                ImGui::Text("Shadow Rays: 1-2 (DLSS denoised)");
            } else {
                ImGui::Text("Shadow Rays: 8 (traditional)");
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                              "DLSS Not Available");
            ImGui::Text("Requires: RTX GPU with latest drivers");
        }
    }
    #endif
}
```

---

## Testing & Validation

### Performance Benchmarks

Compare before/after with PIX captures:

**Test Scenarios:**
1. **Baseline:** 10K particles, 13 lights, 8 shadow rays
2. **DLSS 1-ray:** 10K particles, 13 lights, 1 shadow ray + DLSS
3. **DLSS 2-ray:** 10K particles, 13 lights, 2 shadow rays + DLSS

**Expected Results:**
- 1-ray + DLSS: 2-3× faster than 8-ray traditional
- 2-ray + DLSS: 1.5-2× faster, better quality
- Quality: DLSS 2-ray ≥ Traditional 8-ray

### Visual Quality Comparison

Use MCP screenshot comparison tool:

```bash
# Capture baseline (8 rays traditional)
# Press F2 to capture

# Enable DLSS, capture (1 ray + DLSS)
# Press F2 to capture

# Use MCP tool
compare_screenshots_ml(
    before_path="baseline_8ray.bmp",
    after_path="dlss_1ray.bmp",
    save_heatmap=True
)
```

---

## Timeline

**Day 1 (Today):**
- ✅ Explore SDK structure
- ✅ Create integration plan
- ⏳ Update CMakeLists.txt
- ⏳ Create DLSSSystem.h/cpp skeleton

**Day 2:**
- Implement DLSSSystem initialization
- Implement Ray Reconstruction creation
- Test initialization on RTX 4060 Ti

**Day 3:**
- Integrate with Gaussian renderer
- Modify shadow ray pipeline
- Create shadow buffer resources

**Day 4:**
- Add ImGui controls
- Performance benchmarking
- Visual quality validation

**Day 5:**
- Bug fixes and optimization
- Update CLAUDE.md
- Documentation

---

## Known Issues & Workarounds

### Issue: Application ID Required

**Problem:** DLSS SDK requires an application ID from NVIDIA

**Workaround:** Use `NVSDK_NGX_D3D12_Init_with_ProjectID` with custom project ID
- Allowed for development/testing
- Production release would need official app ID from NVIDIA

### Issue: Driver Version Requirements

**Problem:** DLSS 4.0 requires recent drivers (531.00+)

**Solution:** Feature detection already handles this
- Check `NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver`
- Gracefully disable if not supported

---

## Success Criteria

✅ **Build Success:**
- CMake configures without errors
- Project compiles with DLSS SDK linked
- DLLs copy to output directory

✅ **Runtime Success:**
- DLSS initializes on RTX 4060 Ti
- Ray Reconstruction feature creates successfully
- No crashes during evaluation

✅ **Performance Success:**
- 2× minimum fps improvement (DLSS 1-ray vs traditional 8-ray)
- 4× ideal fps improvement

✅ **Quality Success:**
- DLSS 2-ray quality ≥ Traditional 8-ray
- No visual artifacts
- Smooth temporal stability

---

## Next Steps After DLSS

Once DLSS is working:

1. **Variable Rate Shading** (3 days) - Quick 40% boost
2. **Temporal Upsampling** (2 weeks) - 4K output quality
3. **ReSTIR GI** (4 weeks) - Global illumination

---

**Last Updated:** 2025-10-28
**Status:** Ready to begin implementation
**Branch:** 0.10.12 (fully backed up)
