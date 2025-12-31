# BUG Analysis: Water Mesh RT Hit Detection Failure

**Date:** 2025-12-30
**Status:** ROOT CAUSE IDENTIFIED
**Analyzed By:** Claude Code

---

## Executive Summary

**ROOT CAUSE FOUND:** The `BuildCombinedTLAS()` function is only called when there are overflow particles (`directRTCount > 0`) OR ground plane is enabled. When water mesh is enabled WITHOUT these conditions, `BuildCombinedTLAS()` is never called, and the water mesh BLAS is never added to the TLAS.

---

## Root Cause Analysis

### The Bug Location

**File:** `src/lighting/RTLightingSystem_RayQuery.cpp`
**Line:** 1295

```cpp
// 4. Build combined TLAS (if we have overflow particles OR ground plane)
bool buildCombinedTLAS = (directRTCount > 0 || (m_groundPlane.enabled && m_groundPlane.blas));
if (buildCombinedTLAS) {
    BuildCombinedTLAS(cmdList, true);  // skipBarrier=true
}
```

### What's Wrong

The condition `buildCombinedTLAS` does NOT check for `m_waterMesh.enabled`. It only triggers on:
1. `directRTCount > 0` - overflow particles (>2044 particles)
2. `m_groundPlane.enabled && m_groundPlane.blas` - ground plane

When water mesh is enabled but:
- Particle count is ≤2044 (no overflow)
- Ground plane is disabled

Then `BuildCombinedTLAS()` is never called, and water mesh is never added to the TLAS.

### Evidence from Logs

**Working log (203858):**
```
[20:39:06] Water mesh BLAS buffers created successfully
[20:39:06] Water mesh connected to RT lighting system
[20:39:06] Water mesh loaded successfully: 24 KB
[20:39:06] Water mesh added to TLAS (InstanceID=3, vertices=544, triangles=960)  ← PRESENT
```

**Broken log (205114):**
```
[20:51:45] Water mesh BLAS buffers created successfully
[20:51:45] Water mesh connected to RT lighting system
[20:51:45] Water mesh loaded successfully: 24 KB
[Missing "Water mesh added to TLAS" message]  ← MISSING
```

The difference: In the working log, ground plane was likely enabled, triggering `BuildCombinedTLAS()`. In the broken logs, ground plane was disabled and particle count was 1.

---

## The Fix

**Line 1295 needs to include water mesh in the condition:**

```cpp
// BEFORE (buggy):
bool buildCombinedTLAS = (directRTCount > 0 || (m_groundPlane.enabled && m_groundPlane.blas));

// AFTER (fixed):
bool buildCombinedTLAS = (directRTCount > 0 ||
                          (m_groundPlane.enabled && m_groundPlane.blas) ||
                          (m_waterMesh.enabled && m_waterMesh.blas));
```

---

## File References

| File | Line | Description |
|------|------|-------------|
| `src/lighting/RTLightingSystem_RayQuery.cpp` | 1295 | **BUG:** Missing water mesh in buildCombinedTLAS condition |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | 1024-1033 | Water mesh correctly added in BuildCombinedTLAS() |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | 1505-1581 | SetWaterMesh() - correctly creates BLAS |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | 1583-1617 | BuildWaterMeshBLAS() - builds BLAS correctly |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | 1228-1229 | Water mesh BLAS build called correctly |
| `src/core/Application.cpp` | 7339-7344 | LoadWaterMesh() correctly calls SetWaterMesh + SetWaterMeshEnabled |
| `shaders/particles/particle_gaussian_raytrace.hlsl` | 1293-1319 | Shader water hit detection - correct |
| `shaders/materials/water_material.hlsl` | 12 | INSTANCE_ID_WATER = 3 - correct |

---

## Why This Bug Occurred

The code comment on lines 1287-1290 documents only 3 instances:
```
// Build a COMBINED TLAS with 2-3 instances:
//   Instance 0: Probe Grid BLAS (particles 0-2043)
//   Instance 1: Direct RT BLAS (particles 2044+)
//   Instance 2: Ground Plane BLAS (optional)
```

Instance 3 (Water Mesh) was added later but the condition on line 1295 was never updated.

---

## Verification Steps

1. Add water mesh to the condition on line 1295
2. Run with particle count = 1, ground plane disabled
3. Load water mesh (Bowl or Sphere)
4. Verify "Water mesh added to TLAS" log message appears
5. Verify water geometry is visible (not just cyan debug tint)

---

## Additional Notes

### The Shader Logic is Correct

The shader correctly:
1. Checks `enableWaterMesh != 0` (line 1293)
2. Checks `query.CommittedInstanceID() == INSTANCE_ID_WATER` (line 1295)
3. Uses `ShadeWater()` for reflection/refraction (lines 1311-1318)

### The BLAS Build is Correct

`BuildWaterMeshBLAS()` correctly:
1. Builds triangle BLAS with proper geometry desc
2. Uses correct vertex stride (6 floats = pos + normal)
3. Gets called in `BuildAllBLAS()` (line 1228-1229)

### The Problem is TLAS-Level

The water BLAS exists but is never added to the TLAS because `BuildCombinedTLAS()` is never called.

---

## Secondary Issue: Comment Update Needed

Update line 1287-1292 to include Instance 3:
```cpp
// Build a COMBINED TLAS with 2-4 instances:
//   Instance 0: Probe Grid BLAS (particles 0-2043)
//   Instance 1: Direct RT BLAS (particles 2044+)
//   Instance 2: Ground Plane BLAS (optional)
//   Instance 3: Water Mesh BLAS (optional)
```

---

**Analysis Complete - Ready for Implementation**
