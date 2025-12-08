# Performance Optimizations Implemented - 2025-12-08

**Branch:** 0.22.6 (AABB fix) → current (BLAS update)
**Developer:** Ben + Claude Code

---

## Summary

Two major performance optimizations implemented targeting ray intersection overhead:

| Optimization | File(s) | Expected Gain |
|--------------|---------|---------------|
| AABB Proportional Sizing | gaussian_common.hlsl | **+50-100% FPS** (reduced ray tests) |
| BLAS Incremental Update | RTLightingSystem_RayQuery.cpp/.h | **+15-20 FPS** (~1.3ms saved) |

---

## Optimization 1: AABB Proportional Sizing

### Problem Identified
PIX GPU captures revealed that **all particle AABBs were identical massive sizes** (800×800×800 units) regardless of `particleRadius` setting. This caused:
- 1.3 billion+ ray-AABB intersection tests per frame
- GPU at 96% utilization with only 51 FPS
- Changing `particleRadius` from 1 to 50 had zero effect on AABB size

### Root Cause
1. **Fixed maxAllowedRadius cap** (line 116): `float maxAllowedRadius = 100.0;` was hardcoded, not proportional to `baseRadius`
2. **4σ multiplier** (line 185): `maxRadius * 4.0` created 400-unit half-extents regardless of particle size

### Fix Applied

**File:** `shaders/particles/gaussian_common.hlsl`

**Change 1:** Made radius cap proportional to baseRadius (lines 121, 128):
```hlsl
// OLD: float maxAllowedRadius = 100.0;
// NEW:
float maxAllowedRadius = max(baseRadius * 4.0, 24.0); // 4× base radius, min 24 units
```

**Change 2:** Reduced σ multiplier from 4.0 to 2.5 (line 187):
```hlsl
// OLD: float maxRadius = max(...) * 4.0; // 4σ = 99.99% coverage
// NEW:
float maxRadius = max(scale.x, max(scale.y, scale.z)) * 2.5; // 2.5σ = 98.8% coverage
```

### Impact

| particleRadius | OLD AABB | NEW AABB | Volume Reduction |
|----------------|----------|----------|------------------|
| 24 (user max)  | 800³     | 480³     | **78%** |
| 5 (default)    | 800³     | 120³     | **99.7%** |
| 1 (minimum)    | 800³     | 120³     | **99.7%** |

### Trade-offs
- Minor edge clipping visible with adaptive radius enabled (acceptable)
- Lighting intensity slightly reduced (mitigated with runtime controls)
- 2.5σ covers 98.8% vs 99.99% for 4σ (visually imperceptible)

---

## Optimization 2: BLAS Incremental Update

### Problem Identified
Full BLAS rebuild every frame consuming ~2.1ms, even though particle positions only change slightly between frames.

### Fix Applied

**Files:**
- `src/lighting/RTLightingSystem_RayQuery.h` (line 128)
- `src/lighting/RTLightingSystem_RayQuery.cpp` (lines 272-287, 767-798)

**Change 1:** Added tracking flag to AccelerationStructureSet struct:
```cpp
bool blasBuiltOnce = false;  // BLAS update optimization (2025-12-08)
```

**Change 2:** Added ALLOW_UPDATE flag at BLAS creation (prebuild):
```cpp
blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
```

**Change 3:** Modified BuildBLAS_ForSet() to use PERFORM_UPDATE after first frame:
```cpp
if (asSet.blasBuiltOnce) {
    // Update path: reuse existing BVH structure
    blasInputs.Flags = ... | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    blasDesc.SourceAccelerationStructureData = asSet.blas->GetGPUVirtualAddress();
} else {
    // Initial build
    blasInputs.Flags = ... | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
}
// ...
asSet.blasBuiltOnce = true;
```

### Expected Impact
- BLAS build time: 2.1ms → ~0.8ms (62% reduction)
- FPS gain: +15-20 FPS
- Both ProbeGridAS and DirectRTAS benefit independently

---

## Testing Checklist

- [ ] Verify no visual artifacts (cube clipping)
- [ ] Confirm FPS improvement in PIX
- [ ] Test with various particleRadius values (1, 5, 24)
- [ ] Verify BLAS update path activates after first frame (log message)
- [ ] Test particle count changes (should trigger rebuild, not update)
- [ ] Confirm no TDR/device removal errors

---

## Rollback Instructions

If issues occur:

**AABB Fix:** Revert `gaussian_common.hlsl` changes:
```hlsl
// Revert line 121/128 to:
float maxAllowedRadius = 100.0;

// Revert line 187 to:
float maxRadius = max(scale.x, max(scale.y, scale.z)) * 4.0;
```

**BLAS Update:** Remove `blasBuiltOnce` checks in `BuildBLAS_ForSet()` and revert to simple:
```cpp
blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
```

---

## Related Documentation

- Original analysis: `optimization/OPTIMIZATION_ACTION_PLAN.md`
- PIX captures: `screenshots/Screenshot 2025-12-08 01*.png` (AABB visualization)
- Branch 0.22.6: Contains AABB fix only

---

**Last Updated:** 2025-12-08
**Status:** Implemented, awaiting testing
