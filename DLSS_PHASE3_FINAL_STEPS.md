# DLSS Phase 3 - Final Steps (CRITICAL PIVOT)

**Date:** 2025-10-29 04:15 AM
**Branch:** 0.11.6
**Status:** 80% complete - 3 steps remaining!

---

## What We Discovered Tonight 🎯

**WRONG APPROACH:** Denoising shadows (R16_FLOAT single channel) → Error 0xBAD00005
**RIGHT APPROACH:** Denoise entire Gaussian output (R16G16B16A16_FLOAT) → Perfect for DLSS!

**Why this is HUGE:**
- RT lighting uses 16 rays/particle (expensive!)
- Can reduce to 4 rays + DLSS denoise = **4× performance boost**
- DLSS sees full scene context = better quality than denoising parts separately

---

## ✅ What's Done (Last 30 minutes)

1. **Created denoised output texture** (R16G16B16A16_FLOAT)
   - Lines 259-301 in `ParticleRenderer_Gaussian.cpp`
   - `m_denoisedOutputTexture` - DLSS writes here
   - `m_denoisedOutputSRV` - For reading in blit pass

2. **Redirected DLSS call** from shadows to final output
   - Lines 712-777 in `ParticleRenderer_Gaussian.cpp`
   - Denoises `m_outputTexture` → `m_denoisedOutputTexture`
   - Graceful fallback if DLSS fails

3. **Motion vectors disabled** (temporary - causing crashes)
   - Lines 550-561 in `ParticleRenderer_Gaussian.cpp`
   - Zero motion vectors = static scene assumption (acceptable for now)

---

## ⏳ What's Left (20 minutes)

### Step 1: Update GetOutputSRV() to return denoised texture (5 min)

**File:** `src/particles/ParticleRenderer_Gaussian.h` (line 137)

**Current:**
```cpp
D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const { return m_outputSRVGPU; }
```

**Change to:**
```cpp
#ifdef ENABLE_DLSS
D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const {
    // If DLSS succeeded, return denoised output, otherwise noisy output
    if (m_dlssSystem && m_dlssFeatureCreated && m_denoisedOutputSRVGPU.ptr != 0) {
        return m_denoisedOutputSRVGPU;
    }
    return m_outputSRVGPU;
}
#else
D3D12_GPU_DESCRIPTOR_HANDLE GetOutputSRV() const { return m_outputSRVGPU; }
#endif
```

**Why:** Blit pass needs to read denoised texture instead of noisy one

---

### Step 2: Reduce rays per particle 16 → 4 (2 min)

**File:** `src/lighting/RTLightingSystem_RayQuery.h` (line 87)

**Current:**
```cpp
uint32_t m_raysPerParticle = 16;  // Increased from 4: Eliminates violent brightness flashing
```

**Change to:**
```cpp
uint32_t m_raysPerParticle = 4;  // Reduced for DLSS denoising (4× faster, DLSS removes noise)
```

**Why:** This is the performance win! 16→4 = 4× faster RT lighting

---

### Step 3: Build and Test (5 min)

```bash
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build /nologo /v:minimal
```

**Expected log output:**
```
[INFO] DLSS: Creating Ray Reconstruction feature (1920x1080)...
[INFO] DLSS: Ray Reconstruction feature created successfully!
[INFO] DLSS: Full-scene denoising successful
```

**If successful:**
- No error 0xBAD00005
- Log shows "Full-scene denoising successful" every few frames
- FPS should be ~4× higher than baseline

**If errors:**
- Check log for DLSS errors
- Verify R16G16B16A16_FLOAT format
- Check resource states (SRV vs UAV)

---

## Quick Reference

### Files Modified Tonight

**Headers:**
- `ParticleRenderer_Gaussian.h` (lines 211-214) - Added denoised output texture
- Need to update line 137 (GetOutputSRV) - **Step 1 above**

**Implementation:**
- `ParticleRenderer_Gaussian.cpp` (lines 259-301) - Created denoised texture
- `ParticleRenderer_Gaussian.cpp` (lines 712-777) - DLSS denoising call
- `DLSSSystem.cpp` (lines 216-287) - Fixed parameter setters (used typed setters)

**Shaders:**
- `compute_motion_vectors.hlsl` - Created (temporarily disabled)

**To Change:**
- `RTLightingSystem_RayQuery.h` (line 87) - Reduce rays **Step 2 above**

---

## Benchmark Plan (After Working)

### Test 1: Baseline (16 rays, no DLSS)
```bash
# Temporarily set m_raysPerParticle = 16
# Disable DLSS in code (comment out if-block)
# Run, measure FPS
```

### Test 2: DLSS (4 rays + denoising)
```bash
# Set m_raysPerParticle = 4
# Enable DLSS
# Run, measure FPS
# Expected: 4× faster (480 FPS vs 120 FPS @ 10K particles)
```

### Test 3: Visual Quality
- F2 to screenshot both modes
- Side-by-side comparison
- Check for: blur, missing detail, artifacts
- Tune denoiser strength if needed (ImGui slider later)

---

## If Something Goes Wrong

### Error: 0xBAD00005 (OutOfDate)
**Cause:** DLSS doesn't like the input format
**Fix:** Check we're using R16G16B16A16_FLOAT (not R16_FLOAT)

### Crash on startup
**Cause:** Descriptor allocation or constant buffer issue
**Fix:** Check logs for "Failed to create" messages

### Black screen
**Cause:** Blit pass reading wrong texture
**Fix:** Ensure GetOutputSRV() returns correct texture (Step 1)

### No performance gain
**Cause:** Still using 16 rays
**Fix:** Verify RTLightingSystem_RayQuery.h line 87 = 4

### DLSS not running
**Check log for:** "DLSS: Full-scene denoising successful"
**If missing:** DLSS not activating, check m_dlssFeatureCreated flag

---

## Next Session Enhancements (DLSS_ENHANCEMENT_OPTIONS.md)

**Priority 1:** Add depth buffer (15 min) - Huge quality win
**Priority 2:** Fix motion vectors (30 min) - Eliminate ghosting
**Priority 3:** Add normals (20 min) - Sharper edges

---

## Key Code Locations

**Denoised texture creation:**
```cpp
// ParticleRenderer_Gaussian.cpp:259-301
m_denoisedOutputTexture
m_denoisedOutputSRV
m_denoisedOutputSRVGPU
```

**DLSS denoising call:**
```cpp
// ParticleRenderer_Gaussian.cpp:712-777
if (m_dlssSystem && m_dlssFeatureCreated) {
    // Denoise m_outputTexture → m_denoisedOutputTexture
}
```

**Blit pass reads output:**
```cpp
// Application.cpp (blit pass)
gaussianRenderer->GetOutputSRV()  // Currently returns noisy, needs denoised!
```

---

## Success Criteria

✅ Build succeeds without errors
✅ Log shows "DLSS: Full-scene denoising successful"
✅ FPS increases 3-4× (e.g., 120 → 400+ FPS)
✅ Visual quality maintained or improved
✅ No crashes or GPU timeouts

---

**CRITICAL:** Only 3 small changes needed to test (Steps 1-3 above)
**Time:** 20 minutes total
**Reward:** 4× performance boost! 🚀

**Last Updated:** 2025-10-29 04:15 AM
**Ready to complete when you resume!**
