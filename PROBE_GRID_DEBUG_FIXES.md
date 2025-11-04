# Probe Grid Debug Fixes - Session 4

**Date:** 2025-11-04
**Status:** Three immediate fixes applied, ready for testing

---

## Changes Applied

Following the immediate debugging steps from `PROBE_GRID_2045_CRASH_INVESTIGATION.md`, I've implemented three critical fixes to isolate the GPU hang cause:

### 1. RayQuery Iteration Limit (CRITICAL)

**File:** `shaders/probe_grid/update_probes.hlsl:247-281`

Added iteration counter to prevent infinite loops in `while(q.Proceed())`:

```hlsl
// CRITICAL FIX: Add iteration limit to prevent infinite loops causing TDR timeout
uint iterationCount = 0;
const uint MAX_ITERATIONS = 1000;

while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
    iterationCount++;

    // ... existing intersection testing code
}

// Diagnostic: If we hit the iteration limit, mark this probe with red (timeout)
if (iterationCount >= MAX_ITERATIONS) {
    totalIrradiance = float3(10.0, 0.0, 0.0); // Bright red = timeout detected
}
```

**Purpose:**
- Prevents GPU hang from infinite `Proceed()` loops
- Diagnostic: If you see bright red probes in the scene, the shader is hitting the iteration limit
- 1000 iterations should be more than enough for normal traversal

**Expected Outcome:**
- If crash stops: Confirms infinite loop was the root cause
- If crash continues: Rules out infinite loop, points to other issues (resource binding, TLAS state, etc.)

---

### 2. Explicit TLAS Barrier

**File:** `src/core/Application.cpp:659-666`

Added explicit UAV barrier on TLAS before probe grid dispatch:

```cpp
// CRITICAL FIX: Add explicit TLAS barrier before probe grid dispatch
// Ensures TLAS is in correct state after RT lighting build
if (m_probeGridSystem && m_rtLighting) {
    D3D12_RESOURCE_BARRIER tlasBarrier = {};
    tlasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    tlasBarrier.UAV.pResource = m_rtLighting->GetTLAS();
    cmdList->ResourceBarrier(1, &tlasBarrier);
}
```

**Purpose:**
- Ensures TLAS build completes before probe grid reads it
- Prevents race condition between RT lighting (TLAS build) and probe grid (TLAS read)
- Forces GPU to flush all pending writes to TLAS

**Expected Outcome:**
- If crash stops: Confirms resource state mismatch was the issue
- If crash continues: TLAS state is not the problem

---

### 3. Extreme Ray Count Test

**File:** `src/lighting/ProbeGridSystem.h:168`

Reduced rays per probe from 16 → 1 (extreme test):

```cpp
uint32_t m_raysPerProbe = 1;  // EXTREME TEST: 1 ray per probe (was 16, original 64)
```

**Purpose:**
- Minimizes GPU work per probe (8,192 probes × 1 ray = 8,192 rays/frame)
- If crash persists with 1 ray, it's NOT a ray count issue
- If crash stops, ray count is a factor (but 16 was already very low)

**Ray Count History:**
- Original: 64 rays per probe (524K rays/frame)
- First reduction: 16 rays per probe (131K rays/frame) - still crashed
- Current test: 1 ray per probe (8K rays/frame) - testing now

**Expected Outcome:**
- If crash stops: Ray count is the issue (but 1 ray/probe is not viable for production)
- If crash continues: Rules out ray count entirely

---

## Test Plan

### What to Test:

1. **Launch at 2045 particles** (the known failure point)
2. **Watch for visual artifacts:**
   - Bright red probes = Iteration limit hit (infinite loop detected)
   - Normal colored probes = Shader completing successfully
3. **Monitor for GPU hang:**
   - Still crashes with 5-second pause → Not fixed yet
   - No crash → One of the three fixes solved it
4. **Check log output** in `build/bin/Debug/logs/`

### Interpreting Results:

**Scenario A: Crash stops, no red probes**
- ✅ TLAS barrier or ray count fixed it
- Next step: Increase rays to 4, 8, 16 to find threshold
- Use TLAS barrier permanently

**Scenario B: Crash stops, red probes visible**
- ✅ Iteration limit prevented hang
- Root cause: Infinite loop in `while(q.Proceed())`
- Next step: Investigate why `Proceed()` isn't terminating
- Possible causes:
  - Invalid TLAS structure
  - Malformed ray (NaN/Inf in direction/origin)
  - Driver bug with specific traversal pattern

**Scenario C: Crash continues (same 5-second TDR)**
- ❌ Not infinite loop, not TLAS barrier, not ray count
- Root cause is deeper:
  - Resource binding mismatch (null pointers?)
  - Shader compilation issue (invalid bytecode?)
  - Descriptor heap exhaustion
  - TLAS corruption at specific particle counts
- Next steps from investigation doc:
  - Add diagnostic logging (TLAS GPU address, resource states)
  - Implement particle batching (~2000 per batch)
  - Consider TraceRay() instead of RayQuery

**Scenario D: Different crash behavior**
- Note exact differences (timing, symptoms, logs)
- May indicate progress toward solution

---

## Files Modified

### Shaders:
- `shaders/probe_grid/update_probes.hlsl` - Added iteration limit and timeout diagnostic

### C++ Source:
- `src/core/Application.cpp:659-666` - Added TLAS barrier before probe dispatch
- `src/lighting/ProbeGridSystem.h:168` - Reduced rays per probe to 1

### Build Output:
- `build/bin/Debug/shaders/probe_grid/update_probes.dxil` - Recompiled with iteration limit
- `build/bin/Debug/PlasmaDX-Clean.exe` - Rebuilt with TLAS barrier

---

## What These Fixes Test

| Fix | Tests For | If It Stops Crash | If Crash Continues |
|-----|-----------|-------------------|-------------------|
| **Iteration Limit** | Infinite `Proceed()` loops | Root cause found | Not infinite loop |
| **TLAS Barrier** | Resource state mismatch | Root cause found | TLAS state is correct |
| **1 Ray/Probe** | Ray count threshold | Ray count is factor | Not ray count issue |

---

## Next Steps (If Still Crashes)

Based on investigation document recommendations:

### Short-Term (4-6 hours):
1. **Implement Particle Batching** (most robust solution)
   - Split particles into groups of ~2000
   - Each batch gets separate BLAS/TLAS
   - Probe shader traces against all batches
   - Avoids all power-of-2 thresholds
   - Scalable to 100K+ particles

2. **Add Diagnostic Logging:**
   - Log TLAS GPU address before probe dispatch
   - Verify all resource pointers are non-null
   - Check resource states

### Alternative Approaches:
3. **Limit Probe Updates Per Frame** (1-2 hours)
   - Update only 1024 probes/frame instead of 8192
   - May stay under TDR threshold

4. **Use TraceRay() Instead of RayQuery** (8-12 hours)
   - Full DXR pipeline with raygen/hit/miss shaders
   - May avoid RayQuery-specific driver bugs

---

## Expected Timeline

**Testing Phase:** 5-10 minutes
**Analysis:** 10-15 minutes
**If Batching Needed:** 4-6 hours
**Total:** 5 minutes (best case) to 7 hours (batching implementation)

---

## Success Criteria

✅ **Full Success:** No crash at 2045 particles, probes working correctly
⚠️ **Partial Success:** Crash isolated to specific fix (iteration limit hit, for example)
❌ **No Progress:** Still crashes with same symptoms

---

## Commit Message (If Successful)

```
fix: Add three critical fixes for probe grid GPU hang at 2045+ particles

- Add RayQuery iteration limit (1000) to prevent infinite loops
- Add explicit TLAS UAV barrier before probe dispatch
- Test with 1 ray/probe to isolate ray count factor

Addresses GPU hang/TDR timeout at 2045+ particles.
See PROBE_GRID_DEBUG_FIXES.md for test plan.
```

---

**Last Updated:** 2025-11-04 (Session 4)
**Status:** Three fixes applied, build successful, ready for testing
**Test Configuration:** 2045 particles, 1 ray/probe, iteration limit 1000
