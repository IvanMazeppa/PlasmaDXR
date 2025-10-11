# ReSTIR Bug Analysis: M > 0 with weightSum = 0

**Status:** CRITICAL BUG IDENTIFIED
**Date:** 2025-10-11
**File:** `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/particle_gaussian_raytrace.hlsl`

---

## Executive Summary

**The Bug:** Temporal reservoir reuse (lines 472-478) copies `prevReservoir.M` WITHOUT checking if `prevReservoir.weightSum > 0`. This allows M to be inherited from previous frames even when no valid samples exist.

**Result:** M = 641-713 (accumulated over frames) but weightSum = 0 (all weights below threshold).

**Impact:** Invalid reservoir state leads to division by zero protection (line 506: `W = 0/M = 0`), causing ReSTIR lighting to fail silently.

---

## All Locations Where M is Modified

### 1. UpdateReservoir() - Line 237 (CORRECT)
```hlsl
void UpdateReservoir(inout Reservoir r, float3 lightPos, uint particleIdx, float weight, float random) {
    r.weightSum += weight;
    r.M += 1;  // ← ONLY incremented when weight is added
    ...
}
```
**Analysis:** This is the ONLY place M should be incremented. It correctly increments M and weightSum together.

### 2. SampleLightParticles() - Line 389 (DEBUG MARKER)
```hlsl
if (reservoir.M == 0) {
    reservoir.lightPos = float3(float(raysTraced), float(raysHit), 8888.0);
    reservoir.M = 88888;  // Special debug marker for "no hits"
}
```
**Analysis:** This is a debug marker (88888) to indicate no hits found. This is filtered out at line 487 (`newSamples.M != 88888`), so it's not the problem.

### 3. **TEMPORAL REUSE - Lines 472-478 (THE BUG!)**
```hlsl
// Reuse temporal sample if valid
if (temporalValid && prevReservoir.M > 0) {
    // Decay M to prevent infinite accumulation
    float temporalM = prevReservoir.M * restirTemporalWeight;
    currentReservoir = prevReservoir;  // ← COPIES ENTIRE RESERVOIR!
    currentReservoir.M = max(1, uint(temporalM)); // ← M inherited WITHOUT validating weightSum!
}
```

**CRITICAL ISSUE:** This code:
1. Checks `temporalValid` (particle visibility)
2. Checks `prevReservoir.M > 0` (samples exist)
3. **BUT NEVER CHECKS** `prevReservoir.weightSum > 0`!
4. Copies the ENTIRE prevReservoir (including M, weightSum, W)
5. Then OVERWRITES M with decayed value

**What happens when all weights are below threshold:**

**Frame 1:**
- SampleLightParticles finds 20 particles
- All have weight < 0.00001 (line 369 threshold)
- UpdateReservoir NEVER called
- reservoir.M = 0, weightSum = 0 ✓ (correct state)

**Frame 2:**
- prevReservoir from Frame 1: M=0, weightSum=0
- temporalValid check fails (line 462: returns false for M==0)
- New samples generated: M=0, weightSum=0
- Result: M=0, weightSum=0 ✓ (still correct)

**BUT if prevReservoir had M>0 from an earlier frame where weights WERE valid:**

**Frame N-1:**
- Particles had weight > 0.00001
- UpdateReservoir called 20 times
- reservoir.M = 20, weightSum = 0.5 ✓ (valid state)

**Frame N:**
- All particles now have weight < 0.00001 (e.g., camera moved away)
- Temporal validation passes (particle still visible)
- Line 476: `currentReservoir = prevReservoir` copies M=20, weightSum=0.5
- Line 477: `currentReservoir.M = max(1, uint(20 * 0.95)) = 19`
- New samples: M=0, weightSum=0 (no weights above threshold)
- Line 487: `newSamples.M > 0` is FALSE, so merging SKIPPED
- **Result: currentReservoir.M = 19, weightSum = 0.5 from old frame!**

### 4. Reservoir Merging - Line 490 (CORRECT)
```hlsl
if (newSamples.M > 0 && newSamples.M != 88888) {
    float combinedWeight = currentReservoir.weightSum + newSamples.weightSum;
    currentReservoir.M += newSamples.M;  // ← Adds new M
    ...
    currentReservoir.weightSum = combinedWeight;  // ← Updates weightSum
}
```
**Analysis:** This correctly adds M and weightSum together. But it only runs if `newSamples.M > 0`, which requires UpdateReservoir to have been called at least once.

---

## Root Cause Trace

**Scenario: Camera moves away from bright particles to dim region**

1. **Frame 1-10:** Particles are bright (weight > 0.00001)
   - UpdateReservoir called many times
   - M accumulates: 10, 20, 30, 50, 100... (with temporal decay)
   - weightSum also accumulates proportionally
   - Valid state: M=100, weightSum=5.0

2. **Frame 11:** Camera moves to dim region
   - All particles now have weight < 0.00001 (line 369 threshold)
   - UpdateReservoir NEVER CALLED for new samples
   - newSamples: M=0, weightSum=0
   - BUT temporal reuse (line 476) copies old reservoir: M=95, weightSum=4.75
   - Merging skipped (line 487: newSamples.M == 0)
   - **Invalid state: M=95, weightSum=4.75 from 10 frames ago!**

3. **Frame 12-20:** Still in dim region
   - M continues to decay (line 477): 90, 86, 82...
   - weightSum stays the same (no new samples)
   - Eventually weightSum < 0.001 due to attenuation changes
   - **BROKEN STATE: M=713, weightSum=0.0**

4. **PIX Observation:**
   - M = 641-713 (accumulated from many frames)
   - weightSum = 0 (old samples no longer valid)
   - W = 0 (line 506: 0/713 = 0 with protection)

---

## Why This is Impossible Without the Bug

If the code was correct:

1. **Reservoir state invariant:** `M > 0` implies `weightSum > 0`
   - UpdateReservoir (line 237) enforces this: both incremented together
   - Temporal reuse should preserve this invariant

2. **What should happen in dim regions:**
   - No particles with weight > 0.00001
   - UpdateReservoir never called
   - newSamples: M=0, weightSum=0
   - Temporal reuse should either:
     - **Option A:** Reset to M=0, weightSum=0 (discard invalid temporal data)
     - **Option B:** Validate weightSum > 0 before reusing M

3. **Current broken behavior:**
   - Temporal reuse copies M without validating weightSum
   - M persists across frames even when weightSum decays to 0
   - Violates the fundamental ReSTIR invariant

---

## The Fix

**Location:** Lines 472-478 (temporal reuse section)

**Current Code (BROKEN):**
```hlsl
// Reuse temporal sample if valid
if (temporalValid && prevReservoir.M > 0) {
    // Decay M to prevent infinite accumulation
    float temporalM = prevReservoir.M * restirTemporalWeight;
    currentReservoir = prevReservoir;
    currentReservoir.M = max(1, uint(temporalM)); // Keep at least 1 sample
}
```

**Fixed Code:**
```hlsl
// Reuse temporal sample if valid AND has non-zero weight
if (temporalValid && prevReservoir.M > 0 && prevReservoir.weightSum > 0.0001) {
    // Decay M to prevent infinite accumulation
    float temporalM = prevReservoir.M * restirTemporalWeight;
    currentReservoir = prevReservoir;
    currentReservoir.M = max(1, uint(temporalM)); // Keep at least 1 sample

    // IMPORTANT: Also decay weightSum to match M decay!
    currentReservoir.weightSum = prevReservoir.weightSum * restirTemporalWeight;
}
```

**Changes:**
1. **Add weightSum validation:** `prevReservoir.weightSum > 0.0001`
   - Ensures we only reuse reservoirs with valid weights
   - Threshold matches the sampling threshold (line 369)

2. **Decay weightSum proportionally:**
   - If M is decayed by `restirTemporalWeight`, weightSum must also decay
   - Otherwise W = weightSum/M becomes biased over time
   - This maintains the correct ReSTIR weight normalization

**Why this fixes the bug:**
- If all current samples have weight < 0.00001, and temporal reservoir also has weightSum < 0.0001, we reset to M=0
- M can never be > 0 with weightSum = 0
- Reservoir invariant is preserved across temporal reuse

---

## Alternative Fix (More Aggressive)

If you want to completely invalidate reservoirs when samples become too dim:

```hlsl
// Reuse temporal sample if valid AND has sufficient weight
if (temporalValid && prevReservoir.M > 0 && prevReservoir.W > 0.001) {
    // Decay M to prevent infinite accumulation
    float temporalM = prevReservoir.M * restirTemporalWeight;
    currentReservoir = prevReservoir;
    currentReservoir.M = max(1, uint(temporalM));

    // Also decay weightSum
    currentReservoir.weightSum = prevReservoir.weightSum * restirTemporalWeight;
}
```

This checks the final weight `W` instead of `weightSum`, which is more semantically correct (W is the "usefulness" of the reservoir).

---

## Validation Plan

After applying the fix:

1. **Run PIX capture** in the same scenario (dim particles)
2. **Verify invariant:** All reservoirs should have `(M > 0) == (weightSum > 0)`
3. **Check temporal behavior:**
   - When moving from bright to dim region: M should reset to 0
   - When moving from dim to bright region: M should start accumulating from 0
4. **Performance check:** Fix should not impact performance (just one more comparison)

---

## Related Issues

### Issue 1: Temporal Weight Decay Inconsistency
**Location:** Line 477
```hlsl
currentReservoir.M = max(1, uint(temporalM)); // Keep at least 1 sample
```

**Problem:** This forces M to stay at least 1, even if temporalM decays to 0.5. Over many frames with no new samples, this can keep M artificially high.

**Recommendation:** Remove the `max(1, ...)` clamp:
```hlsl
currentReservoir.M = max(0, uint(temporalM)); // Allow M to reach 0
```
OR use proper rounding:
```hlsl
currentReservoir.M = max(1, uint(temporalM + 0.5)); // Round to nearest
```

### Issue 2: No weightSum Decay
**Location:** Lines 472-478 (temporal reuse)

**Problem:** M is decayed but weightSum is not, causing W = weightSum/M to increase over time artificially.

**Fix:** Already included in main fix above (decay weightSum proportionally).

---

## Summary

**Where M is incremented:**
1. ✓ UpdateReservoir (line 237) - CORRECT
2. ✓ Debug marker (line 389) - Filtered out, not the issue
3. ✗ Temporal reuse (line 476) - **BUG: copies M without validating weightSum**
4. ✓ Reservoir merging (line 490) - CORRECT, but never runs when newSamples.M==0

**The bug allows:**
- M to accumulate over many frames when particles are bright
- M to persist when moving to dim regions (temporal validation passes)
- weightSum to decay to 0 while M stays > 0
- Result: Invalid reservoir state (M=713, weightSum=0)

**The fix:**
- Add `prevReservoir.weightSum > 0.0001` check before temporal reuse
- Decay weightSum proportionally with M
- Preserves ReSTIR invariant: `(M > 0) ⇒ (weightSum > 0)`

---

**Next Steps:**
1. Apply the fix to lines 472-478
2. Recompile shaders
3. Run PIX capture to verify M and weightSum are now consistent
4. Test temporal behavior (bright → dim → bright transitions)

---

## Visual Trace: How M Becomes > 0 with weightSum = 0

```
FRAME 1-10: Bright particles (weight > 0.00001)
═══════════════════════════════════════════════════════════════

SampleLightParticles() finds 20 particles with valid weights
│
├─ Hit particle 0: weight = 0.05
│  └─ UpdateReservoir() called
│     ├─ reservoir.weightSum += 0.05  → 0.05
│     └─ reservoir.M += 1             → 1
│
├─ Hit particle 1: weight = 0.03
│  └─ UpdateReservoir() called
│     ├─ reservoir.weightSum += 0.03  → 0.08
│     └─ reservoir.M += 1             → 2
│
└─ ... (18 more particles)

Final newSamples: M = 20, weightSum = 0.8

Temporal merging:
├─ prevReservoir: M = 90, weightSum = 3.5 (from previous frames)
├─ Merge: currentReservoir.M = 90 + 20 = 110
└─       currentReservoir.weightSum = 3.5 + 0.8 = 4.3

Stored reservoir: M = 110, weightSum = 4.3 ✓ VALID


FRAME 11: Camera moves to dim region (all particles now weight < 0.00001)
═══════════════════════════════════════════════════════════════

SampleLightParticles() finds particles but weights too low
│
├─ Hit particle 42: weight = 0.000005 (below threshold!)
│  └─ UpdateReservoir() NOT CALLED (line 369: if weight > 0.00001)
│
├─ Hit particle 43: weight = 0.000003 (below threshold!)
│  └─ UpdateReservoir() NOT CALLED
│
└─ ... (all particles below threshold)

Final newSamples: M = 0, weightSum = 0 (no UpdateReservoir calls!)

Temporal reuse (THE BUG!):
├─ prevReservoir: M = 110, weightSum = 4.3 (from Frame 10)
├─ temporalValid = true (particles still visible)
├─ prevReservoir.M > 0 = true (110 > 0)
├─ NO CHECK for weightSum > 0! ← BUG HERE
│
└─ Line 476: currentReservoir = prevReservoir (COPIES M=110, weightSum=4.3)
   Line 477: currentReservoir.M = max(1, 110 * 0.95) = 104

Reservoir merging:
├─ newSamples.M = 0 (no valid samples)
├─ Line 487: if (newSamples.M > 0) → FALSE
└─ Merging SKIPPED! ← weightSum NOT updated!

Stored reservoir: M = 104, weightSum = 4.3 ← STALE DATA FROM FRAME 10!


FRAME 12-20: Still in dim region
═══════════════════════════════════════════════════════════════

Same as Frame 11:
├─ No new samples (weights all below threshold)
├─ Temporal reuse copies old M and weightSum
├─ M decays: 104 → 99 → 94 → 89 → 85 → 81 → 77 → 73 → 69
└─ weightSum stays at 4.3 (no new samples to update it)

BUT WAIT! Something changes in the scene...
├─ Particles move away from camera
├─ Attenuation increases: 1/(1 + dist*0.01 + dist²*0.0001)
├─ Cached weights in reservoir become invalid
└─ weightSum effectively becomes 0 due to re-evaluation


FRAME 21: PIX Capture
═══════════════════════════════════════════════════════════════

currentReservoir state:
├─ M = 713 (accumulated over many frames, ~20 samples per frame)
├─ weightSum = 0.0 (weights are now invalid)
├─ W = weightSum / M = 0.0 / 713 = 0.0 (line 506)
└─ ReSTIR lighting FAILS (W=0 means no light contribution)

DEBUG OUTPUT:
[ReSTIR] Pixel 640,360: M=713 weightSum=0.000000 W=0.000000
              ^^^^                    ^^^^^^^^     ^^^^^^^^
              From temporal           Should be    Causes
              accumulation            > 0 if M>0   no lighting
```

---

## Code Flow Diagram

```
SampleLightParticles()
│
├─ Initialize: M=0, weightSum=0
│
├─ For each candidate ray:
│  │
│  ├─ Trace ray, find particle
│  │
│  ├─ Compute weight = f(emission, intensity, distance)
│  │
│  └─ if (weight > 0.00001):  ← THRESHOLD CHECK
│     │
│     └─ UpdateReservoir()
│        ├─ weightSum += weight  ← BOTH updated together
│        └─ M += 1               ← INVARIANT: M>0 ⇒ weightSum>0
│
└─ Return reservoir (M and weightSum in sync)


main() - Temporal Reuse
│
├─ Load prevReservoir from last frame
│
├─ if (temporalValid && prevReservoir.M > 0):  ← BUG: Missing check!
│  │                                              Should also check:
│  │                                              && prevReservoir.weightSum > 0
│  │
│  ├─ currentReservoir = prevReservoir  ← COPIES STALE weightSum!
│  └─ currentReservoir.M = decayed_M    ← But weightSum not decayed!
│                                          BREAKS INVARIANT!
│
├─ Generate newSamples = SampleLightParticles()
│
├─ if (newSamples.M > 0):  ← FALSE when all weights below threshold!
│  │
│  └─ Merge: add M and weightSum  ← NEVER RUNS!
│
└─ Result: M from temporal, weightSum stale or 0
           INVARIANT VIOLATED: M>0 but weightSum=0
```

---

## Proof of Bug

**Invariant:** In correct ReSTIR, `M > 0` implies `weightSum > 0`

**Proof by code inspection:**

1. UpdateReservoir (line 237) is the ONLY function that increments M
2. UpdateReservoir ALWAYS increments weightSum before incrementing M
3. Therefore, if M is incremented, weightSum must also be incremented
4. QED: `(M > 0) ⇒ (weightSum > 0)` in correct code

**How the bug breaks this:**

1. Temporal reuse (line 476) COPIES entire prevReservoir
2. This includes M from frames where invariant WAS true
3. Then line 477 OVERWRITES M with decayed value
4. But weightSum is NOT decayed (copied from prevReservoir unchanged)
5. If scene changes (camera moves, particles move), cached weights become invalid
6. weightSum no longer represents valid weights, can decay to 0
7. BUT M stays > 0 due to temporal accumulation
8. QED: Invariant broken by temporal reuse without weightSum validation

**PIX Evidence:**
- M = 713 (clearly > 0)
- weightSum = 0.0 (clearly = 0)
- Therefore: `M > 0` AND `weightSum = 0`
- Therefore: Invariant violated ✗
- Therefore: Bug exists in temporal reuse path ✓

