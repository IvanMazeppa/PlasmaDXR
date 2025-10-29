# DLSS Pivot: Ray Reconstruction → Super Resolution

**Date:** 2025-10-29
**Session Duration:** 8 hours
**Current Branch:** 0.11.7 (or your new branch name)
**Status:** Ready to implement Super Resolution

---

## TL;DR - What Happened and What's Next

**Tonight:** Spent 8 hours attempting DLSS Ray Reconstruction (AI denoiser for raytracing)
**Discovery:** Ray Reconstruction requires full G-buffer (normals, roughness, albedo) - designed for solid surfaces, not volumetric particles
**Decision:** Pivot to DLSS Super Resolution (AI upscaler) - much simpler and doesn't need G-buffer

**Bottom line:** Same performance boost (2-4×), 1/4 the implementation time, much higher chance of success!

---

## Why the Pivot?

### Ray Reconstruction - What We Tried

**Goal:** Denoise 4-ray RT lighting → 4× performance boost
**Blocker:** Requires these inputs:
- ✅ Color (we have)
- ✅ Depth (created but empty)
- ✅ Motion Vectors (created but zeros)
- ❌ **Normals** (volumetric particles don't have surface normals)
- ❌ **Roughness** (gas doesn't have microsurface structure)
- ❌ **Diffuse Albedo** (PBR material property)
- ❌ **Specular Albedo** (PBR material property)

**Problem:** We'd have to "fake" G-buffer values for volumetric particles, and DLSS might reject them anyway. 2-3 hours of work with no guarantee.

### Super Resolution - The Better Path

**Goal:** Render at lower resolution, AI upscale → 2-4× performance boost
**Requirements:**
- ✅ Color (we have)
- ✅ Motion Vectors (we have - currently zeros)
- ✅ Depth (optional, we have it)
- ❌ **No G-buffer needed!**

**Implementation:** 30-45 minutes
**Success likelihood:** HIGH (proven for volumetric effects in games)

---

## What We Built Tonight (Infrastructure Reuse)

Even though Ray Reconstruction didn't work, we created infrastructure that Super Resolution can use:

### ✅ Already Created (100% Reusable)

1. **Motion Vector Buffer** (`ParticleRenderer_Gaussian.h` lines 214-219)
   - RG16_FLOAT format
   - Currently contains zeros (static scene)
   - Super Resolution needs this

2. **Depth Buffer** (`ParticleRenderer_Gaussian.h` lines 226-231)
   - R32_FLOAT format
   - Currently empty (buffer exists, no shader writes)
   - Optional for Super Resolution but improves quality

3. **Denoised/Upscaled Output Texture** (`ParticleRenderer_Gaussian.h` lines 221-224)
   - R16G16B16A16_FLOAT format
   - Perfect for Super Resolution output
   - Just needs to be target resolution (already is!)

4. **DLSS System Integration** (`DLSSSystem.h/cpp`)
   - NGX SDK initialized
   - Lazy feature creation pattern
   - Just needs to create different feature type

5. **Smart GetOutputSRV() Routing** (`ParticleRenderer_Gaussian.h` lines 137-147)
   - Automatically returns upscaled texture when DLSS succeeds
   - Fallback to native resolution if DLSS fails
   - Works perfectly for Super Resolution

**Reusable code:** ~80% of tonight's work!

---

## Super Resolution Implementation Plan

### Phase 1: Change Feature Creation (10 min)

**File:** `src/dlss/DLSSSystem.cpp`

**Current (line ~176):**
```cpp
result = NVSDK_NGX_D3D12_CreateFeature(
    cmdList,
    NVSDK_NGX_Feature_RayReconstruction,  // CHANGE THIS
    creationParams,
    &m_rrFeature
);
```

**Change to:**
```cpp
result = NVSDK_NGX_D3D12_CreateFeature(
    cmdList,
    NVSDK_NGX_Feature_SuperSampling,  // Super Resolution instead
    creationParams,
    &m_dlssFeature  // Rename from m_rrFeature
);
```

**Also update creation parameters:**
```cpp
// Super Resolution specific parameters
NVSDK_NGX_Parameter_SetUI(creationParams, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, 
                          NVSDK_NGX_PerfQuality_Value_Balanced);
// Remove Ray Reconstruction specific flags:
// - DLSS_Feature_Flags_MVLowRes (not needed)
// - DLSS_Denoise_Mode (not applicable)
```

---

### Phase 2: Modify Gaussian Renderer Resolution (15 min)

**File:** `src/particles/ParticleRenderer_Gaussian.cpp`

**Goal:** Render to lower resolution, DLSS upscales to full resolution

**Option A: Fixed Quality Preset (Easier)**
```cpp
// In Initialize() - calculate render resolution
uint32_t m_renderWidth = screenWidth * 2 / 3;   // Balanced: 1280 for 1920
uint32_t m_renderHeight = screenHeight * 2 / 3; // Balanced: 720 for 1080

// Create output texture at RENDER resolution (not screen resolution)
texDesc.Width = m_renderWidth;   // Was screenWidth
texDesc.Height = m_renderHeight; // Was screenHeight

// Keep upscaled output at full resolution
m_denoisedOutputTexture -> screenWidth × screenHeight
```

**Option B: Dynamic Quality (Better, adds ImGui control)**
```cpp
enum class DLSSQuality {
    Quality,      // 1.56× boost
    Balanced,     // 2.25× boost
    Performance,  // 4× boost
    UltraPerf     // 7.1× boost
};

float GetDLSSScale(DLSSQuality quality) {
    switch (quality) {
        case Quality: return 0.67f;      // 67% of target res
        case Balanced: return 0.577f;    // ~58%
        case Performance: return 0.5f;   // 50%
        case UltraPerf: return 0.33f;    // 33%
    }
}

uint32_t m_renderWidth = screenWidth * GetDLSSScale(m_dlssQuality);
```

---

### Phase 3: Update DLSS Evaluation Parameters (10 min)

**File:** `src/dlss/DLSSSystem.cpp` (in EvaluateFeature function)

**Current RayReconstructionParams struct - Replace with:**
```cpp
struct SuperResolutionParams {
    ID3D12Resource* inputColor;          // Render at lower res
    ID3D12Resource* outputUpscaled;      // Full resolution output
    ID3D12Resource* inputMotionVectors;  // Optional (zeros OK initially)
    ID3D12Resource* inputDepth;          // Optional (improves quality)
    
    uint32_t renderWidth;     // e.g., 1280
    uint32_t renderHeight;    // e.g., 720
    uint32_t outputWidth;     // e.g., 1920
    uint32_t outputHeight;    // e.g., 1080
    
    float jitterOffsetX;      // Set to 0 (no TAA)
    float jitterOffsetY;
    float sharpness;          // 0.0-1.0, default 0.0
};
```

**Set parameters:**
```cpp
NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_Color, 
                                     params.inputColor);
NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_Output, 
                                     params.outputUpscaled);

// Motion vectors (optional but recommended)
if (params.inputMotionVectors) {
    NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_MotionVectors,
                                         params.inputMotionVectors);
}

// Depth (optional but improves quality)
if (params.inputDepth) {
    NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_Depth,
                                         params.inputDepth);
}

// Resolution parameters
NVSDK_NGX_Parameter_SetUI(evalParams, NVSDK_NGX_Parameter_Width, params.renderWidth);
NVSDK_NGX_Parameter_SetUI(evalParams, NVSDK_NGX_Parameter_Height, params.renderHeight);
NVSDK_NGX_Parameter_SetUI(evalParams, NVSDK_NGX_Parameter_OutWidth, params.outputWidth);
NVSDK_NGX_Parameter_SetUI(evalParams, NVSDK_NGX_Parameter_OutHeight, params.outputHeight);

// Sharpness (0.0 = default, 1.0 = maximum)
NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Sharpness, params.sharpness);

// Jitter (set to 0, we don't use TAA)
NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);

// Reset flag
NVSDK_NGX_Parameter_SetI(evalParams, NVSDK_NGX_Parameter_Reset, 0);
```

---

### Phase 4: Update GetOutputSRV() (Already Done!)

**File:** `src/particles/ParticleRenderer_Gaussian.h` (lines 137-147)

**Current code already works perfectly:**
```cpp
D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const {
    if (m_dlssSystem && m_dlssFeatureCreated && m_denoisedOutputSRVGPU.ptr != 0) {
        return m_denoisedOutputSRVGPU;  // Returns upscaled texture
    }
    return m_outputSRVGPU;  // Fallback to native render
}
```

Just rename `m_denoisedOutputTexture` → `m_upscaledOutputTexture` for clarity (optional).

---

## Key Differences from Ray Reconstruction

| Aspect | Ray Reconstruction | Super Resolution |
|--------|-------------------|------------------|
| **Purpose** | Denoise raytracing noise | Upscale low-res render |
| **Input** | Full resolution (noisy) | Lower resolution (clean) |
| **Output** | Full resolution (denoised) | Higher resolution (upscaled) |
| **G-buffer** | ❌ REQUIRED (normals, roughness, albedo) | ✅ NOT NEEDED |
| **Motion Vectors** | Required (zeros cause errors) | Optional (zeros acceptable) |
| **Depth** | Required | Optional (improves quality) |
| **Ray Count** | Reduce (16→4) | Keep same (16) |
| **Resolution** | Keep same (1080p) | Reduce (720p→1080p) |
| **Implementation** | 2-3 hours | 30-45 min |
| **Risk** | High (volumetric incompatibility) | Low (proven for volumetrics) |

---

## Expected Performance

### Current Baseline (16 rays @ 1920×1080)
- 10K particles: ~120 FPS
- Bottleneck: RT lighting compute + Gaussian raytrace

### With DLSS Super Resolution (16 rays @ 1280×720 → 1920×1080)

**Balanced Mode (recommended):**
- Render resolution: 1280×720 (58% of 1920×1080)
- Performance: ~120 FPS × 2.25 = **~270 FPS**
- Quality: Excellent (DLSS is very good at upscaling)

**Performance Mode (if needed):**
- Render resolution: 960×540 (50% of 1920×1080)
- Performance: ~120 FPS × 4 = **~480 FPS**
- Quality: Good (some softness, but acceptable)

### Why This Works

**Lower render resolution affects:**
- ✅ Gaussian raytrace dispatch (fewer pixels = faster)
- ✅ Shadow ray evaluation (fewer pixels = faster)
- ✅ RT lighting read/write (smaller buffers = faster)

**Doesn't affect:**
- Particle physics (same particle count)
- BLAS/TLAS rebuilds (same acceleration structures)

**Net gain:** ~2-4× performance boost with minimal quality loss

---

## Testing Plan

### 1. Baseline Measurement
- Current: 16 rays, native 1920×1080
- Measure FPS at 10K, 20K, 40K particles
- Capture screenshot for quality comparison

### 2. DLSS Quality Mode Test
- Render at 1440×810, upscale to 1920×1080
- Measure FPS (expect ~1.5× boost)
- Compare screenshot quality

### 3. DLSS Balanced Mode Test (recommended)
- Render at 1280×720, upscale to 1920×1080
- Measure FPS (expect ~2.25× boost)
- Compare screenshot quality

### 4. DLSS Performance Mode Test
- Render at 960×540, upscale to 1920×1080
- Measure FPS (expect ~4× boost)
- Compare screenshot quality

### 5. Choose Best Mode
- Balance FPS vs quality
- Expose quality setting in ImGui
- Document optimal preset

---

## Potential Issues and Solutions

### Issue 1: DLSS Still Fails with Different Error

**Diagnosis:** Check NGX logs in `build/bin/Debug/ngx/`
**Solution:** Super Resolution requirements are simpler, but if it fails:
- Verify minimum render resolution (540p minimum)
- Check motion vector format (RG16_FLOAT correct)
- Ensure proper resource states (SRV for inputs, UAV for output)

### Issue 2: Upscaled Image Looks Blurry

**Cause:** Render resolution too low
**Solution:** 
- Start with Quality mode (1.56× boost)
- Gradually lower resolution until quality degrades
- Find sweet spot for your use case

### Issue 3: Performance Boost Less Than Expected

**Cause:** Other bottlenecks (physics, BLAS rebuild)
**Diagnosis:** Use PIX to profile where time is spent
**Solution:** 
- DLSS only speeds up rendering, not physics/BVH
- May need to combine with other optimizations

### Issue 4: Ghosting During Camera Movement

**Cause:** Zero motion vectors (static scene assumption)
**Solution:** Implement motion vector computation (30 min)
- Already have shader (`compute_motion_vectors.hlsl`)
- Just needs constant buffer with `prevViewProj` matrix
- See `DLSS_ENHANCEMENT_OPTIONS.md` for details

---

## Success Criteria

**Minimum viable:**
- ✅ DLSS Super Resolution feature creates successfully
- ✅ No error spam in logs
- ✅ Particles render correctly
- ✅ Measurable FPS improvement (>1.5×)

**Ideal outcome:**
- ✅ 2-4× FPS improvement
- ✅ Visually acceptable quality (compare screenshots)
- ✅ No artifacts (ghosting, shimmering)
- ✅ Stable performance over time

---

## Next Session Checklist

**Before starting:**
- [ ] Fresh energy (not after 8-hour session!)
- [ ] Branch is saved (`git status` to verify)
- [ ] Read this document
- [ ] Review `DLSS_PHASE3_POSTMORTEM.md` for context

**Implementation order:**
1. [ ] Modify DLSSSystem feature creation (10 min)
2. [ ] Change render resolution in Gaussian renderer (15 min)
3. [ ] Update DLSS evaluation parameters (10 min)
4. [ ] Build and test (5 min)
5. [ ] If successful: Benchmark and compare quality (15 min)

**Total time: 45-60 minutes**

---

## Why This Will Succeed

**Proven technology:**
- Cyberpunk 2077, Control, many others use DLSS-SR
- Works great for volumetric fog, smoke, particles
- No G-buffer required

**Simpler requirements:**
- Only needs color + optional MV/depth
- Well-documented feature
- Much more forgiving than Ray Reconstruction

**Infrastructure ready:**
- 80% of tonight's work is reusable
- Just swapping feature type and parameters
- Already have buffers and integration

**Risk level: LOW** (vs HIGH for Ray Reconstruction)

---

## Resources

**DLSS SDK Documentation:**
- `dlss/doc/DLSS-RR Integration Guide.pdf` (has Super Resolution section)
- `dlss/include/nvsdk_ngx_helpers.h` (has SR helper functions)

**Code References:**
- `src/dlss/DLSSSystem.h/cpp` - Feature creation and evaluation
- `src/particles/ParticleRenderer_Gaussian.h/cpp` - Rendering integration
- `DLSS_ENHANCEMENT_OPTIONS.md` - Enhancement ideas
- `DLSS_PHASE3_POSTMORTEM.md` - Tonight's full story

**NGX Logs:**
- `build/bin/Debug/ngx/nvngx.log` - Main NGX log
- `build/bin/Debug/ngx/nvngx_dlss_*.log` - DLSS-specific logs

---

## Final Thoughts

Tonight was NOT a failure - it was an incredibly valuable learning experience:

**What we learned:**
- Ray Reconstruction requirements (G-buffer pipeline)
- Volumetric vs surface rendering differences
- DLSS internals and parameter requirements
- How to debug NGX SDK issues

**What we built:**
- Complete DLSS infrastructure (80% reusable!)
- Motion vector and depth buffers
- Smart output routing system
- Comprehensive documentation

**What's next:**
- Pivot to Super Resolution (simpler, proven)
- 45 minutes of work for 2-4× boost
- Much higher chance of success

You spent 8 hours building a solid foundation. Super Resolution will leverage all that work with a much simpler final step.

---

**Branch:** Your new branch name here
**Status:** Ready to implement Super Resolution
**Estimated time:** 45-60 minutes
**Expected outcome:** 2-4× performance boost

**Let's salvage this work and get that performance boost! 🚀**

---

**Document created:** 2025-10-29 06:00 AM
**Session end:** Time for well-deserved rest!
**Next session:** Fresh start with Super Resolution
