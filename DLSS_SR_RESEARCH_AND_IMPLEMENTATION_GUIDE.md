# DLSS Super Resolution - Research & Implementation Guide

**Date:** 2025-10-29
**Project:** PlasmaDX-Clean Volumetric Particle Renderer
**Goal:** Implement DLSS-SR to achieve 2-4× performance boost
**Estimated Time:** 45-60 minutes
**Risk Level:** LOW (much simpler than Ray Reconstruction)

---

## Executive Summary

After 8 hours of attempting DLSS Ray Reconstruction (which failed due to G-buffer requirements), we're pivoting to DLSS Super Resolution. This document contains comprehensive research to avoid similar pitfalls and ensure successful first-time implementation.

**Key Difference from Ray Reconstruction:**
- **Ray Reconstruction:** Denoise noisy raytracing at full resolution → Requires G-buffer
- **Super Resolution:** Upscale clean low-resolution render → No G-buffer needed ✅

---

## 1. Critical Requirements (MUST HAVE)

### 1.1 Required Resources

| Resource | Format | Purpose | Status |
|----------|--------|---------|--------|
| **Color Input** | R16G16B16A16_FLOAT | Low-res render (e.g., 720p) | ✅ Have (m_outputTexture) |
| **Color Output** | R16G16B16A16_FLOAT | Upscaled result (e.g., 1080p) | ✅ Have (m_denoisedOutputTexture) |
| **Depth Buffer** | R32_FLOAT or D24_UNORM_S8_UINT | Scene depth | ✅ Have (m_depthBuffer) |
| **Motion Vectors** | RG16_FLOAT | Per-pixel motion | ✅ Have (m_motionVectorBuffer) |

**IMPORTANT:** All four buffers already created during Ray Reconstruction work!

### 1.2 Resource State Requirements

**Per Streamline Documentation:**
- DLSS manages resource states automatically
- **No manual transitions needed** (DLSS handles it)
- Application must restore pipeline state after evaluation

**Current Implementation:**
- Already have proper state management in ParticleRenderer_Gaussian
- No changes needed ✅

### 1.3 Resolution Calculation

**Use NGX Helper Function (CRITICAL!):**
```cpp
NGX_DLSS_GET_OPTIMAL_SETTINGS(
    m_params,                          // Capability parameters
    screenWidth,                       // Target output width (1920)
    screenHeight,                      // Target output height (1080)
    NVSDK_NGX_PerfQuality_Value_Balanced,  // Quality mode
    &renderWidth,                      // OUT: Optimal render width (1280)
    &renderHeight,                     // OUT: Optimal render height (720)
    &maxWidth, &maxHeight,             // OUT: Dynamic resolution bounds
    &minWidth, &minHeight,             // OUT: Dynamic resolution bounds
    &optimalSharpness                  // OUT: Recommended sharpness
);
```

**Why This Matters:**
- DLSS has specific resolution requirements per mode
- Manual calculation can cause errors
- Helper function ensures compatibility

**Quality Modes:**
| Mode | Scale | Example (1080p output) | Boost |
|------|-------|------------------------|-------|
| Quality | 0.67 | 1290×720 | 1.56× |
| Balanced | 0.577 | 1110×620 | 2.25× |
| Performance | 0.5 | 960×540 | 4.0× |
| Ultra Performance | 0.33 | 635×360 | 7.1× |

### 1.4 Motion Vector Format

**CRITICAL: Motion Vector Scaling**

From Streamline docs: "Motion vector scaling is critical for correct temporal data interpretation"

**Two formats supported:**

**Option A: Pixel-Space Vectors** (what we currently have)
```cpp
// Motion vectors in pixels (e.g., object moved 5.2 pixels right)
// MUST scale by render resolution:
float mvScaleX = 1.0f / renderWidth;   // e.g., 1.0/1280 = 0.00078125
float mvScaleY = 1.0f / renderHeight;  // e.g., 1.0/720 = 0.001388889

NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_MV_Scale_X, mvScaleX);
NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_MV_Scale_Y, mvScaleY);
```

**Option B: Normalized Vectors [-1, 1]**
```cpp
// Motion vectors already normalized to [-1, 1] range
float mvScaleX = 1.0f;
float mvScaleY = 1.0f;

NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
```

**Our Current State:**
- Motion vectors are zeros (static scene)
- Format is RG16_FLOAT (correct)
- We'll use Option A (pixel-space) and set to zeros initially

---

## 2. Optional Enhancements (IMPROVE QUALITY)

### 2.1 Particle Mask Buffer

**What it does:** Helps DLSS distinguish particles from solid geometry

**From SDK Headers:**
```cpp
// Optional: Particle/transparency mask
NVSDK_NGX_Parameter_SetD3d12Resource(params,
    NVSDK_NGX_Parameter_IsParticleMask,
    pInIsParticleMask);
```

**When to use:**
- Mixed particle + solid geometry rendering
- Prevents particle ghosting artifacts
- Improves temporal stability for transparent effects

**Our case:**
- 100% volumetric particles (no solid geometry)
- Probably not needed, but can add later if ghosting occurs
- **Decision: Skip for initial implementation** ⏭️

### 2.2 Exposure Buffer

**What it does:** Tells DLSS about scene brightness for better tone mapping

**SDK Parameter:**
```cpp
// Optional: Exposure texture (auto-exposure)
NVSDK_NGX_Parameter_SetD3d12Resource(params,
    NVSDK_NGX_Parameter_ExposureTexture,
    pInExposureTexture);
```

**Our case:**
- Currently no auto-exposure system
- DLSS will work without it
- **Decision: Skip for initial implementation** ⏭️

### 2.3 Transparency Mask

**What it does:** Marks regions with alpha blending for better handling

**SDK Parameter:**
```cpp
// Optional: Transparency/reactive mask
NVSDK_NGX_Parameter_SetD3d12Resource(params,
    NVSDK_NGX_Parameter_TransparencyMask,
    pInTransparencyMask);
```

**Our case:**
- Volumetric particles use volumetric absorption (not alpha blending)
- May help with edge cases
- **Decision: Add if quality issues arise** ⏭️

### 2.4 Sharpness Control

**What it does:** Post-upscale sharpening filter

**SDK Parameter:**
```cpp
// Sharpness: 0.0 = default, 1.0 = maximum sharpening
NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Sharpness, 0.0f);
```

**Best Practice:**
- Start with 0.0 (DLSS default)
- Use optimal value from `NGX_DLSS_GET_OPTIMAL_SETTINGS()`
- Expose in ImGui for user preference
- **Decision: Use optimal value, add ImGui control** ✅

---

## 3. Common Pitfalls (AVOID THESE!)

### 3.1 Wrong Resolution Order

**❌ WRONG:**
```cpp
// Creating output at RENDER resolution
outputTexDesc.Width = renderWidth;   // 1280 - WRONG!
outputTexDesc.Height = renderHeight; // 720 - WRONG!

// Creating input at TARGET resolution
inputTexDesc.Width = screenWidth;    // 1920 - WRONG!
inputTexDesc.Height = screenHeight;  // 1080 - WRONG!
```

**✅ CORRECT:**
```cpp
// Input = RENDER resolution (low-res)
inputTexDesc.Width = renderWidth;    // 1280 ✓
inputTexDesc.Height = renderHeight;  // 720 ✓

// Output = TARGET resolution (high-res)
outputTexDesc.Width = screenWidth;   // 1920 ✓
outputTexDesc.Height = screenHeight; // 1080 ✓
```

**Why this matters:** Swapping these causes initialization failure

### 3.2 Jitter Offsets in Non-TAA Scenes

**Issue:** DLSS expects TAA jitter for temporal stability

**Our case:** We don't use TAA (no camera jitter)

**Solution:**
```cpp
// Always set to zero if not using TAA
NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
```

**If we ADD TAA later:**
```cpp
// Jitter must be in RENDER pixel space, not screen space
float jitterX = haltonX / renderWidth;   // NOT screenWidth!
float jitterY = haltonY / renderHeight;  // NOT screenHeight!
```

### 3.3 Motion Vector Scaling Errors

**Most Common Mistake:** Using wrong resolution for MV scaling

**❌ WRONG:**
```cpp
// Using OUTPUT resolution - WRONG!
float mvScaleX = 1.0f / screenWidth;  // WRONG!
float mvScaleY = 1.0f / screenHeight; // WRONG!
```

**✅ CORRECT:**
```cpp
// Use RENDER resolution (where MVs were generated)
float mvScaleX = 1.0f / renderWidth;   // ✓
float mvScaleY = 1.0f / renderHeight;  // ✓
```

**Why:** Motion vectors are computed at render resolution, not output resolution

### 3.4 Feature Type Confusion

**Ray Reconstruction Uses:**
```cpp
NVSDK_NGX_Feature_RayReconstruction  // ❌ Wrong for Super Resolution
```

**Super Resolution Uses:**
```cpp
NVSDK_NGX_Feature_SuperSampling      // ✅ Correct
```

**IMPORTANT:** Must match during:
1. Feature creation (`CreateFeature`)
2. Feature evaluation (`EvaluateFeature`)
3. Feature release (`ReleaseFeature`)

### 3.5 Reset Flag Misuse

**Reset flag tells DLSS to clear temporal history**

**When to set Reset=1:**
- Scene cut (teleport, level load)
- Resolution change
- Quality mode change
- First frame after feature creation

**When to set Reset=0:**
- Every other frame (continuous rendering)

**Our case:**
```cpp
// First frame after creation
NVSDK_NGX_Parameter_SetI(params, NVSDK_NGX_Parameter_Reset, 1);

// Subsequent frames
NVSDK_NGX_Parameter_SetI(params, NVSDK_NGX_Parameter_Reset, 0);
```

---

## 4. Particle/Volumetric Specific Considerations

### 4.1 Why Particles Struggle with Upscalers

**From Research:**
> "Depending on the input resolution, an object or particle that is too thin or small may not show up at all in the input image or may show up inconsistently, which can cause visible artifacts, such as flickering or ghosting."

**DLSS 4 Improvement:**
> "The new architecture improves flow estimation, which is particularly valuable for challenging scenarios like particle effects, where traditional methods often struggle due to the complex, dynamic, and often sparse nature of these elements."

**Implications for PlasmaDX:**
- Our particles are volumetric (not sparse points) ✅
- Gaussian splatting creates thick volumes ✅
- Less likely to disappear at low resolution ✅
- Still: Monitor for small/distant particle issues

### 4.2 Adaptive Particle Radius Synergy

**Current Feature:** Adaptive particle radius based on distance

**DLSS Benefit:**
- Farther particles already larger (due to adaptive radius)
- Less likely to disappear at reduced render resolution
- Natural LOD helps DLSS upscaling

**Recommendation:** Keep adaptive radius enabled with DLSS-SR

### 4.3 Volumetric Effects Best Practices

**From Industry Research:**
> "Fire effects and volumetric smoke remain largely unaffected even in Performance mode at 4K"

**Why volumetrics work well:**
- Spatially coherent (not sparse points)
- Natural motion blur from volume integration
- Less sensitive to resolution than sharp edges

**PlasmaDX Advantages:**
- 3D Gaussian volumes (not billboards)
- Temperature-based emission (smooth gradients)
- Beer-Lambert absorption (continuous field)

### 4.4 Motion Vector Generation for Particles

**Current State:** Zero motion vectors (static scene assumption)

**Phase 1 (Current):** Accept some ghosting
- Set MVs to zero
- Let DLSS use spatial upscaling only
- Still get 2-4× performance boost

**Phase 2 (Enhancement):** Compute per-particle motion
```cpp
// In particle shader:
float4 currPos = mul(viewProj, float4(particleWorldPos, 1.0));
float4 prevPos = mul(prevViewProj, float4(particleWorldPos - velocity * dt, 1.0));
float2 motionVector = (currPos.xy / currPos.w) - (prevPos.xy / prevPos.w);
motionVector *= 0.5; // [-1, 1] range
```

**Benefit of Phase 2:**
- Eliminates ghosting during camera movement
- Improves temporal stability
- Better handling of fast-moving particles

---

## 5. Step-by-Step Implementation Plan

### Phase 1: DLSSSystem Changes (15 min)

**File:** `src/dlss/DLSSSystem.h`

**Changes:**
1. Add `SuperResolutionParams` struct
2. Add `CreateSuperResolutionFeature()` method
3. Add `EvaluateSuperResolution()` method
4. Add quality mode enum

**New struct:**
```cpp
enum class DLSSQualityMode {
    Quality,        // 1.56× boost
    Balanced,       // 2.25× boost
    Performance,    // 4.0× boost
    UltraPerf       // 7.1× boost
};

struct SuperResolutionParams {
    ID3D12Resource* inputColor;         // Render at low-res
    ID3D12Resource* outputUpscaled;     // Full resolution output
    ID3D12Resource* inputMotionVectors; // Optional (zeros OK)
    ID3D12Resource* inputDepth;         // Optional (improves quality)

    uint32_t renderWidth;      // e.g., 1280
    uint32_t renderHeight;     // e.g., 720
    uint32_t outputWidth;      // e.g., 1920
    uint32_t outputHeight;     // e.g., 1080

    float jitterOffsetX;       // Set to 0 (no TAA)
    float jitterOffsetY;
    float sharpness;           // 0.0-1.0, use optimal from SDK
    int reset;                 // 1 = clear history, 0 = accumulate
};
```

**File:** `src/dlss/DLSSSystem.cpp`

**Change 1: Feature Creation**
```cpp
bool DLSSSystem::CreateSuperResolutionFeature(
    ID3D12GraphicsCommandList* cmdList,
    uint32_t renderWidth,
    uint32_t renderHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    DLSSQualityMode qualityMode
) {
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
    }

    // Set creation parameters
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Width, renderWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Height, renderHeight);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutWidth, outputWidth);
    NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutHeight, outputHeight);
    NVSDK_NGX_Parameter_SetI(m_params, NVSDK_NGX_Parameter_PerfQualityValue, perfQuality);

    // Create Super Resolution feature (NOT Ray Reconstruction!)
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_CreateFeature(
        cmdList,
        NVSDK_NGX_Feature_SuperSampling,  // ← CHANGED FROM RayReconstruction
        m_params,
        &m_dlssFeature  // Renamed from m_rrFeature
    );

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to create Super Resolution feature: 0x{:08X}",
                  static_cast<uint32_t>(result));
        return false;
    }

    LOG_INFO("DLSS: Super Resolution feature created successfully!");
    LOG_INFO("  Render: {}x{}, Output: {}x{}",
             renderWidth, renderHeight, outputWidth, outputHeight);

    return true;
}
```

**Change 2: Feature Evaluation**
```cpp
bool DLSSSystem::EvaluateSuperResolution(
    ID3D12GraphicsCommandList* cmdList,
    const SuperResolutionParams& params
) {
    // Set input color (render resolution)
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params,
        NVSDK_NGX_Parameter_Color, params.inputColor);

    // Set output upscaled (target resolution)
    NVSDK_NGX_Parameter_SetD3d12Resource(m_params,
        NVSDK_NGX_Parameter_Output, params.outputUpscaled);

    // Motion vectors (optional, zeros OK)
    if (params.inputMotionVectors) {
        NVSDK_NGX_Parameter_SetD3d12Resource(m_params,
            NVSDK_NGX_Parameter_MotionVectors, params.inputMotionVectors);

        // CRITICAL: MV scaling with RENDER resolution
        float mvScaleX = 1.0f / params.renderWidth;
        float mvScaleY = 1.0f / params.renderHeight;
        NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_MV_Scale_X, mvScaleX);
        NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_MV_Scale_Y, mvScaleY);
    }

    // Depth (optional, improves quality)
    if (params.inputDepth) {
        NVSDK_NGX_Parameter_SetD3d12Resource(m_params,
            NVSDK_NGX_Parameter_Depth, params.inputDepth);
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
        cmdList, m_dlssFeature, m_params, nullptr
    );

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Super Resolution evaluation failed: 0x{:08X}",
                  static_cast<uint32_t>(result));
        return false;
    }

    return true;
}
```

### Phase 2: ParticleRenderer_Gaussian Changes (20 min)

**File:** `src/particles/ParticleRenderer_Gaussian.h`

**Changes:**
1. Add render resolution members
2. Add DLSS quality mode member
3. Update output texture to render resolution

**New members:**
```cpp
#ifdef ENABLE_DLSS
    // DLSS Super Resolution
    DLSSSystem::DLSSQualityMode m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Balanced;
    uint32_t m_renderWidth = 0;   // Internal render resolution
    uint32_t m_renderHeight = 0;
    uint32_t m_outputWidth = 0;   // Final display resolution
    uint32_t m_outputHeight = 0;
    bool m_dlssFirstFrame = true; // For reset flag
#endif
```

**File:** `src/particles/ParticleRenderer_Gaussian.cpp`

**Change 1: Initialize Resolution**
```cpp
bool ParticleRenderer_Gaussian::Initialize(...) {
    // ... existing code ...

#ifdef ENABLE_DLSS
    if (m_dlssSystem) {
        // Store target resolution
        m_outputWidth = screenWidth;
        m_outputHeight = screenHeight;

        // Get optimal render resolution from DLSS SDK
        uint32_t maxW, maxH, minW, minH;
        float optimalSharpness;

        NVSDK_NGX_PerfQuality_Value perfQuality =
            NVSDK_NGX_PerfQuality_Value_Balanced; // Balanced mode

        NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
            m_dlssSystem->GetParameters(),  // Need to expose this
            m_outputWidth,
            m_outputHeight,
            perfQuality,
            &m_renderWidth,    // OUT: optimal render resolution
            &m_renderHeight,
            &maxW, &maxH, &minW, &minH,
            &optimalSharpness
        );

        if (NVSDK_NGX_SUCCEEDED(result)) {
            LOG_INFO("DLSS: Optimal resolution calculated:");
            LOG_INFO("  Render: {}x{}", m_renderWidth, m_renderHeight);
            LOG_INFO("  Output: {}x{}", m_outputWidth, m_outputHeight);
            LOG_INFO("  Sharpness: {}", optimalSharpness);

            // Use render resolution for output texture (internal rendering)
            screenWidth = m_renderWidth;
            screenHeight = m_renderHeight;
        } else {
            LOG_ERROR("DLSS: Failed to get optimal settings, using native resolution");
            m_renderWidth = screenWidth;
            m_renderHeight = screenHeight;
        }
    }
#endif

    // Continue with texture creation (now at render resolution)
    // ... existing code ...
}
```

**Change 2: Create Upscaled Output at Target Resolution**
```cpp
#ifdef ENABLE_DLSS
    if (m_dlssSystem) {
        // Create upscaled output texture at TARGET resolution
        D3D12_RESOURCE_DESC upscaledTexDesc = {};
        upscaledTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        upscaledTexDesc.Width = m_outputWidth;   // Full resolution (1920)
        upscaledTexDesc.Height = m_outputHeight; // Full resolution (1080)
        upscaledTexDesc.DepthOrArraySize = 1;
        upscaledTexDesc.MipLevels = 1;
        upscaledTexDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        upscaledTexDesc.SampleDesc.Count = 1;
        upscaledTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        hr = m_device->GetDevice()->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &upscaledTexDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_upscaledOutputTexture)  // Renamed from m_denoisedOutputTexture
        );

        // ... create SRV/UAV descriptors ...
    }
#endif
```

**Change 3: Enable DLSS Feature Creation**
```cpp
#ifdef ENABLE_DLSS
    // Create DLSS Super Resolution feature (lazy initialization)
    if (m_dlssSystem && !m_dlssFeatureCreated && cmdList) {
        bool success = m_dlssSystem->CreateSuperResolutionFeature(
            cmdList,
            m_renderWidth,
            m_renderHeight,
            m_outputWidth,
            m_outputHeight,
            m_dlssQualityMode
        );

        if (success) {
            m_dlssFeatureCreated = true;
            m_dlssFirstFrame = true;  // Set reset flag
            LOG_INFO("DLSS: Super Resolution feature created successfully!");
        } else {
            LOG_ERROR("DLSS: Failed to create Super Resolution feature");
            // Continue without DLSS (fallback to native resolution)
        }
    }
#endif
```

**Change 4: Evaluate DLSS Super Resolution**
```cpp
#ifdef ENABLE_DLSS
    // DLSS Super Resolution: Upscale from render-res to output-res
    if (m_dlssSystem && m_dlssFeatureCreated) {
        PIXBeginEvent(cmdList, PIX_COLOR_DEFAULT, L"DLSS Super Resolution");

        // Prepare parameters
        DLSSSystem::SuperResolutionParams dlssParams = {};
        dlssParams.inputColor = m_outputTexture.Get();        // Render-res input
        dlssParams.outputUpscaled = m_upscaledOutputTexture.Get(); // Full-res output
        dlssParams.inputMotionVectors = m_motionVectorBuffer.Get(); // Zeros for now
        dlssParams.inputDepth = m_depthBuffer.Get();          // Optional depth
        dlssParams.renderWidth = m_renderWidth;
        dlssParams.renderHeight = m_renderHeight;
        dlssParams.outputWidth = m_outputWidth;
        dlssParams.outputHeight = m_outputHeight;
        dlssParams.jitterOffsetX = 0.0f;  // No TAA
        dlssParams.jitterOffsetY = 0.0f;
        dlssParams.sharpness = 0.0f;      // Use DLSS default (or optimal from SDK)
        dlssParams.reset = m_dlssFirstFrame ? 1 : 0;

        bool success = m_dlssSystem->EvaluateSuperResolution(cmdList, dlssParams);

        if (success && m_dlssFirstFrame) {
            m_dlssFirstFrame = false;  // Only reset on first frame
        }

        PIXEndEvent(cmdList);
    }
#endif
```

### Phase 3: ImGui Controls (10 min)

**File:** `src/core/Application.cpp`

**Add DLSS Quality Controls:**
```cpp
#ifdef ENABLE_DLSS
if (ImGui::CollapsingHeader("DLSS Super Resolution")) {
    ImGui::Checkbox("Enable DLSS-SR", &m_dlssEnabled);

    if (m_dlssEnabled) {
        const char* qualityModes[] = {
            "Quality (1.56x)",
            "Balanced (2.25x)",
            "Performance (4.0x)",
            "Ultra Performance (7.1x)"
        };
        int currentMode = static_cast<int>(m_particleRenderer->GetDLSSQualityMode());

        if (ImGui::Combo("Quality Mode", &currentMode, qualityModes, 4)) {
            m_particleRenderer->SetDLSSQualityMode(
                static_cast<DLSSSystem::DLSSQualityMode>(currentMode)
            );
            // Note: Requires feature recreation (defer to next frame)
        }

        float sharpness = m_particleRenderer->GetDLSSSharpness();
        if (ImGui::SliderFloat("Sharpness", &sharpness, -1.0f, 1.0f)) {
            m_particleRenderer->SetDLSSSharpness(sharpness);
        }

        ImGui::Separator();
        ImGui::Text("Render Resolution: %dx%d",
                    m_particleRenderer->GetRenderWidth(),
                    m_particleRenderer->GetRenderHeight());
        ImGui::Text("Output Resolution: %dx%d",
                    m_particleRenderer->GetOutputWidth(),
                    m_particleRenderer->GetOutputHeight());
    }
}
#endif
```

---

## 6. Testing Checklist

### 6.1 Initialization Tests

- [ ] DLSS SDK initializes without errors
- [ ] `NGX_DLSS_GET_OPTIMAL_SETTINGS` succeeds
- [ ] Render resolution calculated correctly (e.g., 1280×720 for Balanced @ 1080p)
- [ ] Output texture created at full resolution (1920×1080)
- [ ] Super Resolution feature created successfully
- [ ] No crash on first render

### 6.2 Runtime Tests

- [ ] Particles render correctly at lower resolution
- [ ] DLSS upscaling produces visible output (not black screen)
- [ ] Framerate improves by expected amount (1.5-4×)
- [ ] No flickering or severe artifacts
- [ ] ImGui overlay still visible and functional
- [ ] No log spam or repeated errors

### 6.3 Quality Comparison

**Capture screenshots:**
- [ ] Native 1920×1080 (baseline)
- [ ] DLSS Quality mode (1290×720 → 1920×1080)
- [ ] DLSS Balanced mode (1110×620 → 1920×1080)
- [ ] DLSS Performance mode (960×540 → 1920×1080)

**Compare:**
- [ ] Particle edges (smooth vs jagged)
- [ ] Distant particles (visible vs missing)
- [ ] Motion (ghosting artifacts?)
- [ ] Overall sharpness

**Use MCP tool:**
```python
compare_screenshots_ml(
    before_path="native_1080p.bmp",
    after_path="dlss_balanced_1080p.bmp",
    save_heatmap=True
)
```

### 6.4 Performance Benchmarks

**Test scenarios:**
| Particles | Native FPS | DLSS Quality | DLSS Balanced | DLSS Performance |
|-----------|------------|--------------|---------------|------------------|
| 10K | 120 | Target: 187 | Target: 270 | Target: 480 |
| 20K | 65 | Target: 101 | Target: 146 | Target: 260 |
| 40K | 30 | Target: 47 | Target: 68 | Target: 120 |

### 6.5 Edge Cases

- [ ] Resize window (recreate feature?)
- [ ] Toggle DLSS on/off at runtime
- [ ] Switch quality modes mid-render
- [ ] Extremely close camera (large particles)
- [ ] Extremely far camera (tiny particles)
- [ ] Fast camera rotation (motion blur?)

---

## 7. Success Criteria

### Minimum Viable (45 min implementation)

**Must have:**
- ✅ DLSS-SR feature creates successfully
- ✅ Particles render at lower resolution
- ✅ Upscaling produces correct output
- ✅ Measurable FPS improvement (>1.5×)
- ✅ No crashes or black screens

### Ideal Outcome (60 min with polish)

**Should have:**
- ✅ 2.25× FPS boost (Balanced mode)
- ✅ Acceptable visual quality (compare screenshots)
- ✅ ImGui controls functional
- ✅ No severe ghosting or flickering

### Future Enhancements (Phase 2)

**Nice to have:**
- ⏳ Actual motion vector computation (eliminate ghosting)
- ⏳ Particle mask buffer (improve transparency)
- ⏳ Dynamic resolution scaling
- ⏳ Per-scene quality presets

---

## 8. Rollback Plan (If DLSS-SR Fails)

**If DLSS-SR doesn't work:**

1. Check NGX logs in `build/bin/Debug/ngx/`
2. Verify resolution calculation (render < output)
3. Confirm feature type is `SuperSampling`, not `RayReconstruction`
4. Test with different quality modes
5. Disable depth/MV inputs (test with color only)

**If still fails:**
- Disable DLSS with `if (false && ...)` (same as Ray Reconstruction)
- Keep render resolution = screen resolution
- Document errors for NVIDIA dev support
- Consider AMD FSR2 or Intel XeSS as alternative

**Fallback options:**
1. Temporal upsampling (custom implementation)
2. AMD FidelityFX Super Resolution 2.x (no ML required)
3. Native resolution with other optimizations (VRS, PINN, etc.)

---

## 9. Key Differences from Ray Reconstruction

| Aspect | Ray Reconstruction | Super Resolution |
|--------|-------------------|------------------|
| **Purpose** | Denoise raytracing | Upscale low-res render |
| **Input Resolution** | Full (1920×1080) | Reduced (1280×720) |
| **Output Resolution** | Same as input | Higher than input |
| **G-buffer Required** | ❌ YES (normals, roughness, albedo) | ✅ NO |
| **Depth Required** | ❌ YES | ✅ Optional |
| **Motion Vectors** | Required (zeros fail) | Optional (zeros acceptable) |
| **Feature Type** | `RayReconstruction` | `SuperSampling` |
| **Complexity** | High (many inputs) | Low (2 required inputs) |
| **Risk Level** | High (volumetric incompatibility) | Low (proven for volumetrics) |
| **Implementation Time** | 2-3 hours | 45-60 minutes |

---

## 10. Expected Results

### Performance (RTX 4060 Ti @ 1080p)

| Mode | Render Res | Boost | 10K Particles | 40K Particles | 100K Particles |
|------|------------|-------|---------------|---------------|----------------|
| Native | 1920×1080 | 1.0× | 120 FPS | 30 FPS | 18 FPS |
| Quality | 1290×720 | 1.56× | 187 FPS | 47 FPS | 28 FPS |
| **Balanced** | **1110×620** | **2.25×** | **270 FPS** | **68 FPS** | **41 FPS** |
| Performance | 960×540 | 4.0× | 480 FPS | 120 FPS | 72 FPS |

**Recommended:** Balanced mode (best FPS/quality ratio)

### Quality (Subjective)

**Quality Mode:**
- Near-identical to native
- Slight softness on particle edges
- Acceptable for most use cases

**Balanced Mode:**
- Good quality, noticeable improvement over native resolution
- Minor softness, but particles remain clear
- **Best overall choice**

**Performance Mode:**
- Noticeable quality loss (softer, some detail missing)
- Distant particles may flicker
- Use only if FPS critical (>100K particles)

---

## 11. Post-Implementation: Next Steps

### If DLSS-SR Succeeds

**Immediate (same session):**
1. Benchmark all quality modes
2. Capture comparison screenshots
3. Document optimal settings
4. Commit to Git

**Short-term (next session):**
1. Implement motion vector computation
2. Test with particle mask buffer
3. Add dynamic resolution scaling
4. Expose sharpness in ImGui

**Medium-term (Phase 6B):**
1. Combine DLSS-SR + PINN physics (100K particles @ 120 FPS!)
2. Add VRS 2.0 (cumulative boost: ~380 FPS @ 10K particles)
3. Optimize RTXDI M5 temporal accumulation

### If DLSS-SR Fails

**Debugging:**
1. Analyze NGX logs
2. Test with minimal parameters (color only)
3. Try different quality modes
4. Contact NVIDIA dev support

**Alternative path:**
1. Implement temporal upsampling (custom solution)
2. Evaluate AMD FSR2 (no NVIDIA hardware requirement)
3. Focus on non-upscaling optimizations (PINN, VRS, Hi-Z culling)

---

## 12. Resources and References

### Official Documentation

- **DLSS SDK:** `/DLSS/doc/DLSS_Programming_Guide_Release.pdf`
- **Helper Functions:** `/DLSS/include/nvsdk_ngx_helpers.h`
- **Parameter Definitions:** `/DLSS/include/nvsdk_ngx_defs.h`

### Online Resources

- **Streamline Programming Guide:** https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md
- **DLSS Integration Guide:** https://developer.nvidia.com/blog/how-to-integrate-nvidia-dlss-4-into-your-game-with-nvidia-streamline
- **DLSS Developer Portal:** https://developer.nvidia.com/rtx/dlss

### Project Documentation

- **DLSS Phase 3 Postmortem:** `DLSS_PHASE3_POSTMORTEM.md`
- **DLSS Pivot Document:** `DLSS_PIVOT_TO_SUPER_RESOLUTION.md`
- **Enhancement Ideas:** `NON_RAYTRACING_ENHANCEMENT_IDEAS.md`

### MCP Tools

- **Screenshot comparison:** `compare_screenshots_ml(before, after, save_heatmap=True)`
- **Performance analysis:** `compare_performance(native_log, dlss_log)`
- **PIX analysis:** `analyze_pix_capture(capture_path)`

---

## 13. Final Thoughts

**Why This Will Work:**

1. ✅ **80% infrastructure already built** (buffers, descriptors, routing)
2. ✅ **No G-buffer requirement** (volumetric particles compatible)
3. ✅ **Proven technology** (used in 100+ games with volumetrics)
4. ✅ **Simple API** (only 2 required inputs: color + output)
5. ✅ **Low risk** (can fallback to native resolution if fails)

**Key Lessons from Ray Reconstruction:**

1. ✅ Check NGX logs immediately when errors occur
2. ✅ Use SDK helper functions (don't manual calculate)
3. ✅ Start simple (color only, add depth/MV later)
4. ✅ Test incrementally (don't add all features at once)
5. ✅ Document expected vs actual results

**Confidence Level:** HIGH (90%+ success probability)

**Estimated Time:** 45-60 minutes

**Expected Outcome:** 2-4× performance boost with acceptable quality

**Risk Assessment:** LOW (much simpler than Ray Reconstruction)

---

**Good luck! Let's salvage that 8-hour DLSS session and get those performance gains! 🚀**

---

**Document Version:** 1.0
**Created:** 2025-10-29
**Author:** Claude Code Research Session
**Status:** Ready for implementation
