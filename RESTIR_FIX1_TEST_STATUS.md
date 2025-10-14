# ReSTIR Fix 1 - Test Status
**Date:** 2025-10-13
**Fix Applied:** MIS Weight Formula Correction (W × M)

---

## Status: **READY FOR TESTING**

### What Happened

1. **23:51** - You ran test with OLD shader (before fix)
   - Log: `PlasmaDX-Clean_20251013_235110.log`
   - Shader timestamp: Oct 13 06:38
   - Result: No difference (expected - fix not applied yet!)

2. **23:58** - Shader recompiled with Fix 1
   - File: `shaders/particles/particle_gaussian_raytrace.dxil`
   - Timestamp: Oct 13 23:58
   - Change: Line 648 now uses `W × M` instead of `W × 16 / M`

3. **Now** - Ready for proper test with new shader

---

## Fix 1 Details

**Changed:** Line 648 in `particle_gaussian_raytrace.hlsl`

**Before (buggy):**
```hlsl
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
// Problem: Divides by M twice (W already = weightSum / M)
// Effect: Adjacent pixels with M=1 vs M=4 have 4× brightness difference → dots!
```

**After (fixed):**
```hlsl
float misWeight = currentReservoir.W * float(currentReservoir.M);
// Solution: Standard ReSTIR formula (W × M)
// Effect: Spatial consistency restored - M variance no longer affects brightness
```

---

## Testing Instructions

### Quick Test (5 minutes)

1. **Launch application** (shader will load the new compiled version)
2. **Enable ReSTIR** (F7 key)
3. **Navigate to problem area** (close to particles, orange indicator)
4. **Look for improvements:**
   - ✅ Fewer or no dots?
   - ✅ Colors more vibrant?
   - ✅ Smoother lighting across screen?

### Detailed Test (15 minutes)

Test at **4 distances** to validate fix across all scenarios:

| Distance | Expected Before Fix | Expected After Fix 1 |
|----------|-------------------|---------------------|
| **100 units** | Severe dots, washed out | Smooth, vibrant colors |
| **200 units** | Many dots, dull colors | Minor artifacts only |
| **400 units** | Some dots visible | Nearly perfect |
| **600 units** | Working (baseline) | Still working |

**Steps:**
1. Launch app, enable ReSTIR (F7)
2. Position camera at each distance
3. Take screenshot or note visual quality
4. Compare to before

---

## Expected Results

### If Fix 1 Works (Likely)

**Symptoms that should DISAPPEAR:**
- ✅ Millions of dots/sparkles when close
- ✅ Color washing out to dull tones
- ✅ Brightness flickering as M varies

**Symptoms that might PERSIST:**
- ⚠️ Slight temporal flicker (needs Fix 2)
- ⚠️ Slightly subdued colors (needs Fix 3 - clamp increase)

**If this happens:** Fix 1 solved the primary bug! Proceed to Fix 2 & 3 for polish.

---

### If Fix 1 Doesn't Work (Unlikely)

**Possible reasons:**
1. **Shader not loaded** - Check log for "Loaded Gaussian shader" message
2. **Wrong shader file** - Verify `particle_gaussian_raytrace.dxil` timestamp is 23:58
3. **ReSTIR disabled** - Make sure F7 shows "ReSTIR: ON"
4. **Different bug** - The dots aren't from ReSTIR MIS weight issue

**Debug steps:**
1. Check log file for shader load confirmation
2. Verify ReSTIR toggle messages in log
3. Create new PIX capture for agent analysis
4. Consider applying Fix 2 & 3 together

---

## Additional Fixes Available

If Fix 1 **helps but doesn't fully resolve** the issue, apply these:

### Fix 2: Distance-Adaptive Temporal Weight

**Problem:** Stale samples dominate when close (90% temporal weight too high)

**Location:** Line 487 in `particle_gaussian_raytrace.hlsl`

**Code:**
```hlsl
// Add BEFORE line 487:
float distToNearestParticle = length(currentReservoir.lightPos - cameraPos);
float adaptiveTemporalWeight = lerp(0.3, restirTemporalWeight,
                                    saturate(distToNearestParticle / 200.0));

// Then CHANGE line 487:
float temporalM = prevReservoir.M * adaptiveTemporalWeight; // Use adaptive instead of restirTemporalWeight
```

**Effect:**
- Close (< 200): Uses 0.3 temporal weight (favors fresh samples)
- Far (> 200): Uses full 0.9 temporal weight (stable reuse)
- Prevents stale light positions from causing flicker

---

### Fix 3: Increase Clamp Range

**Problem:** Clamp(0, 2.0) crushes highlights → color washing

**Location:** Line 651 in `particle_gaussian_raytrace.hlsl`

**Code:**
```hlsl
// CHANGE from:
misWeight = clamp(misWeight, 0.0, 2.0);

// TO:
misWeight = clamp(misWeight, 0.0, 10.0);  // Allow more dynamic range
```

**Effect:**
- Restores full color saturation
- Allows proper HDR brightness
- No more washed-out colors

---

## Validation Metrics

### If Fix 1 Works:

**Visual Confirmation:**
- Screen should look smooth and vibrant at all distances
- No more "salt and pepper" noise pattern
- Colors maintain saturation when close

**PIX Capture Analysis:**
If you want empirical proof, create new capture and check:
- M/W correlation should be **flat** (no inverse relationship)
- misWeight values should be **spatially coherent** (neighbors similar)
- Final illumination should be **uniform** where expected

---

## Next Steps

### Scenario A: Fix 1 Completely Resolves Issue ✅

1. ✅ Mark ReSTIR bug as **RESOLVED**
2. Document fix in version notes
3. Consider backporting fix to any other ReSTIR implementations
4. Move on to other features or optimizations

### Scenario B: Fix 1 Helps But Not Perfect ⚠️

1. Apply Fix 2 (distance-adaptive temporal weight)
2. Apply Fix 3 (increase clamp to 10.0)
3. Test again with all 3 fixes
4. Create validation capture set

### Scenario C: Fix 1 No Effect ❌

1. Verify shader actually loaded (check log)
2. Create new PIX capture at problem distance
3. Use PIX Agent v3 for deeper analysis
4. Consider alternative hypotheses:
   - Bug is in reservoir sampling code (lines 276-394)?
   - Bug is in temporal validation (line 248)?
   - Bug is elsewhere entirely?

---

## Test Checklist

**Before Testing:**
- [ ] Shader compiled at 23:58 (verify with `ls -lh shaders/particles/particle_gaussian_raytrace.dxil`)
- [ ] Application closed (to force shader reload)
- [ ] Clear any cached shaders if applicable

**During Testing:**
- [ ] Launch application
- [ ] Enable ReSTIR (F7 key)
- [ ] Verify log shows "ReSTIR: ON"
- [ ] Navigate to close distance (orange indicator)
- [ ] Observe dots vs smooth rendering

**After Testing:**
- [ ] Document results (screenshots helpful)
- [ ] Note any remaining artifacts
- [ ] Check log for any errors
- [ ] Decide: Fix 1 only, or apply Fix 2+3?

---

## Current Status

**Shader:** ✅ Compiled with Fix 1 (23:58)
**Test:** ❌ Not yet tested with new shader (previous test was before fix)
**Next Action:** **RUN APPLICATION NOW** and test at close distance

---

**Good luck!** The fix is mathematically sound and should resolve the primary bug. Report back with results!
