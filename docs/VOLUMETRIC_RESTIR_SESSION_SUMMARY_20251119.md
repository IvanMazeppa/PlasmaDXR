# Volumetric ReSTIR Session Summary - 2025-11-19

## Overview
Third attempt at implementing Volumetric ReSTIR. Gemini 3 Pro designed a new approach that bypasses the density grid atomic contention issue.

---

## The New Approach (Gemini-Designed)

### Root Cause Identified
The original implementation crashed due to **atomic contention** in `PopulateVolumeMip2`:
- Thousands of threads fighting for same voxel addresses using `InterlockedMax`
- Caused GPU TDR timeouts with 10K+ particles

### Solution: Direct BVH Path Generation
Instead of building a 32³ density grid, use the **RT BVH directly** for path generation:
- No atomics = no contention = no TDR crashes
- Hardware-accelerated RayQuery for particle queries
- Scalable to 100K particles

### Files Modified by Gemini
1. **`src/lighting/VolumetricReSTIRSystem.cpp`** - Disabled `PopulateVolumeMip2` (early return)
2. **`shaders/volumetric_restir/path_generation.hlsl`** - Real implementation using `QueryNearestParticle`
3. **`shaders/volumetric_restir/shading.hlsl`** - Updated with `QueryParticleFromRay`

---

## Fixes Applied During This Session

### Fix 1: Variable Redefinition (C++ Compilation Error)
**Issue:** `srvBarrier` defined twice in `VolumetricReSTIRSystem.cpp`
**Fix:** Renamed second instance to `finalSrvBarrier` in dead code after early return

### Fix 2: Root Signature Mismatch (PSO Creation Failed)
**Issue:** Root signature only had 2 parameters (b0, u0) but shader needed 6 resources
**Fix:** Updated root signature to include:
- Slot 0: b0 - Constant buffer
- Slot 1: Descriptor table (t0=BVH, t1=particles, t2=volume)
- Slot 2: u0 - Reservoir UAV
- Static sampler s0

### Fix 3: Descriptor Allocation Missing
**Issue:** No descriptors allocated for path generation SRV table
**Fix:** Added `m_pathGenSrvTableCPU`/`m_pathGenSrvTableGPU` members and allocated 3 contiguous SRV descriptors in Initialize()

### Fix 4: Resource Binding in GenerateCandidates
**Issue:** Only bound slots 0 and 1, not the descriptor table
**Fix:** Updated to create SRVs for BVH/particles dynamically and bind descriptor table at slot 1, UAV at slot 2

### Fix 5: Particle Struct Mismatch (Critical!)
**Issue:** Shader Particle struct was 80 bytes with ellipsoid axes, but C++ struct is 48 bytes
**C++ layout (48 bytes):**
- position (12) + temperature (4) + velocity (12) + density (4) + albedo (12) + materialType (4)

**Fix:** Updated both `path_generation.hlsl` and `shading.hlsl` to match C++ struct

### Fix 6: Resolution Mismatch (Frozen Output)
**Issue:** VolumetricReSTIR rendered at 640×360 but wrote to 2560×1440 texture
- Only top-left corner was written, rest showed stale Gaussian frame
**Fix:** Changed to full resolution (2560×1440) for Phase 1 testing

---

## Current State

### What Works
- System initializes successfully
- PSOs created without errors
- No crashes or TDR timeouts
- Frame loop continues running
- Physics simulation continues
- Higher FPS when ReSTIR enabled (100 FPS vs 80 FPS legacy)

### What Doesn't Work
- **Black screen** when VolumetricReSTIR is enabled
- No visible particles/lighting

---

## Analysis: Why Black Screen

### Most Likely Cause: RayQuery Not Finding Particles

The `QueryNearestParticle` function in `path_generation.hlsl` has this code:

```hlsl
while (q.Proceed()) {
    if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
        q.CommitProceduralPrimitiveHit(q.CandidateProceduralPrimitiveNonOpaque());
    }
}
```

**Problem:** `CommitProceduralPrimitiveHit()` takes a `float hitT` (hit distance), but `CandidateProceduralPrimitiveNonOpaque()` returns a `bool`. This passes 0.0 or 1.0 as the hit distance, which is completely wrong.

**Correct usage:**
```hlsl
q.CommitProceduralPrimitiveHit(q.CandidateRayT());
```

### Secondary Issues

1. **No intersection testing** - For procedural primitives (ellipsoid particles), you need actual ray-ellipsoid intersection testing. The current code commits the AABB hit without verifying the ray actually intersects the particle geometry.

2. **Reservoir always empty** - If `QueryNearestParticle` never returns true, `GenerateCandidatePath` returns false, pathLength stays 0, and `ShadeSelectedPaths` outputs black for pixels with `pathLength == 0`.

3. **ComputeTargetPDF issues** - Even if paths are generated, the target PDF computation also uses broken `QueryNearestParticle` calls.

---

## Next Steps to Fix Black Screen

### Priority 1: Fix CommitProceduralPrimitiveHit
Change in both `path_generation.hlsl` and `shading.hlsl`:
```hlsl
// WRONG
q.CommitProceduralPrimitiveHit(q.CandidateProceduralPrimitiveNonOpaque());

// CORRECT
q.CommitProceduralPrimitiveHit(q.CandidateRayT());
```

### Priority 2: Add Ray-Ellipsoid Intersection (Optional for Phase 1)
For Phase 1, using AABB approximation may be acceptable. Full ellipsoid intersection can be added later for visual quality.

### Priority 3: Debug Output
Add diagnostic output to verify:
- Are RayQueries finding any particles?
- What are the hit distances?
- Are reservoirs being populated with valid paths?

---

## Performance Notes

- Full resolution (2560×1440) uses 3.7M shader invocations
- Legacy RayQuery renderer: ~80 FPS
- VolumetricReSTIR: ~100 FPS (doing less work since output is black)
- Memory: 478 MB for reservoir buffers at full resolution

---

## Code Locations

### Key Files
- `src/lighting/VolumetricReSTIRSystem.cpp` - Main system implementation
- `src/lighting/VolumetricReSTIRSystem.h` - Header with descriptor handles
- `shaders/volumetric_restir/path_generation.hlsl` - Path generation shader
- `shaders/volumetric_restir/shading.hlsl` - Final shading shader
- `shaders/volumetric_restir/volumetric_restir_common.hlsl` - Shared definitions
- `src/core/Application.cpp` - Integration and rendering loop (lines 336-362 for init, 1000-1030 for rendering)

### Root Signature Layout
```
Slot 0: CBV (b0) - PathGenerationConstants
Slot 1: Descriptor Table - t0 (BVH), t1 (particles), t2 (volume)
Slot 2: UAV (u0) - Reservoir buffer
Static Sampler: s0
```

---

## Session Summary

| Item | Status |
|------|--------|
| TDR crashes | FIXED - bypassed atomic contention |
| PSO creation | FIXED - root signature matches shader |
| Particle struct | FIXED - 48 bytes matches C++ |
| Resolution | FIXED - full res, fills output texture |
| RayQuery hits | BROKEN - `CommitProceduralPrimitiveHit` has wrong parameter |
| Visual output | BLACK - no particles rendered |

**Next session: Fix the `CommitProceduralPrimitiveHit` calls to use `q.CandidateRayT()` instead of `q.CandidateProceduralPrimitiveNonOpaque()`**

---

## Document History
- Created: 2025-11-19
- Author: Claude Code session with Ben
