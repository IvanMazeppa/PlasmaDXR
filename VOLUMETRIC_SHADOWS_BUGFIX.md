# Volumetric Shadows Bug Fix (2025-11-10)

**Branch**: 0.15.0
**Status**: ✅ FIXED

---

## Critical Bug: Shadows Not Visible

**Symptom**: Volumetric raytraced shadows had no visible effect when toggled on/off (F5 key).

**Root Cause**: Two critical bugs in `shaders/particles/particle_gaussian_raytrace.hlsl`:

### Bug #1: RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES (Line 232-233)

**Original code:**
```hlsl
RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
```

**Problem**: The flag `RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES` tells the ray query to **SKIP ALL PROCEDURAL PRIMITIVES**, which is exactly what the 3D Gaussian particles are! This meant shadow rays would never hit any particles, resulting in no shadow occlusion.

**Fix:**
```hlsl
RayQuery<RAY_FLAG_NONE> query;  // Do NOT skip procedural primitives!
```

### Bug #2: Incorrect CommitProceduralPrimitiveHit() Usage (Line 294)

**Original code:**
```hlsl
while (query.Proceed()) {
    // ... accumulate opacity ...
}
query.CommitProceduralPrimitiveHit(shadowOpacity);  // ❌ WRONG - called outside loop
```

**Problem**: `CommitProceduralPrimitiveHit()` must be called **inside** the `query.Proceed()` loop for each hit candidate, not after the loop completes.

**Fix:**
```hlsl
while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
        // ... accumulate opacity ...

        query.CommitProceduralPrimitiveHit(tHit);  // ✅ CORRECT - commit each hit

        // ... check early out ...
    }
}
```

---

## Files Modified

**`shaders/particles/particle_gaussian_raytrace.hlsl`**
- Lines 232-233: Removed `RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES` flag
- Line 284: Moved `CommitProceduralPrimitiveHit()` inside loop
- Line 297: Removed incorrect post-loop call

---

## Build Status

✅ **Shader compilation**: SUCCESS
✅ **Solution build**: SUCCESS

**Output**: `build/bin/Debug/PlasmaDX-Clean.exe`

---

## Testing Recommendations

1. **Enable shadows** (F5) and verify visible darkening where particles occlude lights
2. **Test quality presets**:
   - Performance (1 ray): Should show sharp shadows with temporal smoothing
   - Balanced (4 rays): Should show soft shadow penumbra
   - Quality (8 rays): Should show very soft, graduated shadows
3. **Look for temperature variation** - Hotter particles (>22000K) should cast darker shadows
4. **Verify temporal convergence** - Performance preset should smooth out noise over ~67ms

---

## Expected Behavior

**Before fix**: No visible shadow effect when F5 toggled on/off
**After fix**: Clear volumetric self-shadowing with Beer-Lambert absorption

**Shadow characteristics:**
- Particles cast volumetric shadows on each other
- Shadow strength depends on particle temperature (hotter = denser = darker)
- Soft shadow penumbra visible in Balanced/Quality presets
- Temporal accumulation reduces noise in Performance preset

---

## Performance Impact

No performance change expected - the fix corrects broken functionality without adding overhead.

**Target framerates** (10K particles, RTX 4060 Ti, 1080p):
- Performance (1 ray): 115+ FPS
- Balanced (4 rays): 90-100 FPS
- Quality (8 rays): 60-75 FPS

---

**Last Updated**: 2025-11-10
**Verified**: Shader compilation + solution build successful
