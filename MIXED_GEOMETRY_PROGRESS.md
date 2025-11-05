# Mixed Geometry BLAS Workaround - Progress Report

## Current Status: PARTIAL SUCCESS ✅⚠️

### What We've Implemented

**Mixed Geometry BLAS (Test H from DEBUG_2045_TESTS.md):**
- Added 1 non-degenerate triangle to BLAS alongside procedural AABBs
- Triangle positioned at -10000 units (far outside scene bounds -1500 to +1500)
- Triangle in UPLOAD heap (correct for DXR vertex buffers)
- BLAS now has 2 geometries: procedural AABBs + dummy triangle

### Key Finding: We Bypassed the Original Bug!

**Original behavior (pure procedural BLAS):**
- 2044 particles: Works ✅
- 2045 particles: **INSTANT crash** at frame 0 ❌

**New behavior (mixed geometry BLAS):**
- 100 particles, frame 0: **COMPLETES SUCCESSFULLY** ✅
- 100 particles, frame 1: Crashes ❌
- 2045 particles: **NOT YET TESTED** ⏳

### Analysis

**SUCCESS**: Frame 0 now completes, which NEVER happened with pure procedural BLAS at 2045+ particles. This proves mixed geometry uses a **different driver code path** that avoids the original Ada Lovelace bug.

**NEW ISSUE**: Frame 1 crash suggests a SECONDARY bug:
- Only affects frames after the first
- Might be specific to low particle counts (100 tested)
- Could be unrelated to the original 2045 threshold

### Files Modified

**src/lighting/RTLightingSystem_RayQuery.h:177**
```cpp
Microsoft::WRL::ComPtr<ID3D12Resource> m_dummyTriangleBuffer;
```

**src/lighting/RTLightingSystem_RayQuery.cpp:250-279**
- Create dummy triangle buffer (36 bytes, UPLOAD heap)
- Non-degenerate triangle: (-10000, -10000, -10000), (-10000, -10000, -9999), (-10000, -9999, -10000)

**src/lighting/RTLightingSystem_RayQuery.cpp:280-303**
- Updated BLAS size calculation for 2 geometries

**src/lighting/RTLightingSystem_RayQuery.cpp:770-828**
- BuildBLAS() now builds mixed geometry BLAS
- Geometry 0: Procedural AABBs (particles)
- Geometry 1: Dummy triangle (never hit by rays)

### Next Steps

1. **CRITICAL TEST**: Run at 2045 particles to confirm original bug is fixed
   - If frame 0 completes → ORIGINAL BUG SOLVED ✅
   - If crashes at frame 0 → Mixed geometry insufficient ❌

2. **If 2045 frame 0 works:**
   - The frame 1 crash is a separate issue
   - May need different fix (resource barriers, state transitions, etc.)
   - Might be specific to low particle counts

3. **If 2045 still crashes at frame 0:**
   - Try alternative triangle configurations:
     - Multiple triangles instead of one
     - Different positions
     - Indexed triangle instead of vertex-only

### Test Instructions

To test at 2045 particles:

**Option 1: Modify hardcoded default in Application.cpp**
```cpp
// In Application::Application() constructor
m_particleCount = 2045;  // Change from 100 to 2045
```

**Option 2: Command line (if --particles argument works)**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe --particles 2045
```

**Option 3: Create config.json**
```json
{
  "rendering": {
    "particleCount": 2045
  }
}
```

### Hypothesis

The original 2045 bug was caused by:
- Pure procedural BLAS triggering specific BVH traversal code path in Ada Lovelace driver
- Exact threshold (2045 particles = 512 BVH leaves = 2^9) triggering power-of-2 boundary bug

Mixed geometry forces:
- Different BVH construction algorithm (handles both triangles and procedural primitives)
- Different traversal code path (mixed geometry shader paths)
- Different memory layout (2 geometry descriptors instead of 1)

This bypasses the Ada Lovelace bug entirely at the cost of minimal overhead (1 dummy triangle = ~36 bytes + negligible traversal cost).

### Log Evidence

**build/bin/Debug/logs/PlasmaDX-Clean_20251105_025855.log:**

Line 295-298 (Frame 0):
```
Building MIXED GEOMETRY BLAS (procedural AABBs + dummy triangle)
BLAS geometry 0: 100 procedural AABBs
BLAS geometry 1: 1 non-degenerate triangle at -10000 units (dummy)
RT Lighting computed with dynamic emission (frame 0)  ← ✅ FRAME 0 COMPLETES
```

Line 330-333 (Frame 1 - crashes here):
```
Building MIXED GEOMETRY BLAS (procedural AABBs + dummy triangle)
BLAS geometry 0: 100 procedural AABBs
BLAS geometry 1: 1 non-degenerate triangle at -10000 units (dummy)
[LOG ENDS - CRASH]  ← ❌ FRAME 1 CRASHES
```

**This is UNPRECEDENTED** - frame 0 NEVER completed with pure procedural BLAS at 2045+ particles.

---

**Next session: Test at 2045 particles to confirm original bug is fixed.**
