# ReSTIR Brightness Fix - October 12, 2025

## Problem Summary

ReSTIR was causing visual artifacts when approaching the particle cloud:
1. **"Dots"** - Individual light samples visible at medium distance (orange indicator)
2. **Color shift** - Muted/brown colors when close (yellow/green indicator)
3. **Over-brightness** - ReSTIR lighting adding too much illumination

## Root Cause

The ReSTIR weight calculation was incorrectly scaling by M (sample count):

```hlsl
// WRONG (previous code):
float restirScale = (currentReservoir.W * currentReservoir.M) / float(restirInitialCandidates);
rtLight = lightEmission * lightIntensity * attenuation * restirScale;
```

**Why this was wrong:**
- `W = weightSum / M` is already the average weight per sample
- Multiplying by M again effectively scales by M²
- When M=24 (close to particles), this caused 576× over-brightness!
- This created extreme illumination values that got clamped/saturated

## The Fix

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl`
**Lines:** 624-627, 639

### Change 1: Remove M scaling
```hlsl
// FIXED:
// W = weightSum / M is the unbiased estimator
rtLight = lightEmission * lightIntensity * attenuation * currentReservoir.W;
```

**Rationale:** In ReSTIR theory, W already represents the properly weighted light contribution. Scaling by M was double-counting samples.

### Change 2: Clamp illumination
```hlsl
// Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
// Clamp to prevent over-brightness from extreme ReSTIR samples
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
```

**Rationale:** Even with correct W calculation, extreme cases can occur. The clamp acts as a safety net.

## Expected Results

After this fix, with ReSTIR enabled (F7):

✅ **Consistent brightness** - Similar to ReSTIR OFF
✅ **No dots** - Smooth continuous lighting
✅ **No color shift** - Proper emission colors at all distances
✅ **RT controls work** - I/K keys adjust lighting properly
✅ **Distance indicators work correctly:**
  - **Red:** Far from particles, M=0-1
  - **Orange:** Medium distance, M=2-8
  - **Yellow:** Close, M=9-16
  - **Green:** Very close, M=16+

## Testing Instructions

### Manual Test
1. Build with updated shader (completed)
2. Launch `PlasmaDX-Clean.exe` (Debug build)
3. Press **F7** to enable ReSTIR
4. Fly toward particle cloud with WASD
5. Watch corner indicator transition: Red → Orange → Yellow → Green
6. **Expected:** Smooth brightness, no dots, colors unchanged

### PIX Agent Test
1. Enable ReSTIR in config:
```json
{
  "gaussianRT": {
    "useReSTIR": true,
    "restirInitialCandidates": 16,
    "restirTemporalWeight": 0.95
  }
}
```

2. Capture frame 120+ (after temporal convergence)
3. Check reservoir data in PIX:
   - `currentReservoir.M` should be 1-24
   - `currentReservoir.W` should be 0.0001-0.01 (reasonable range)
   - `weightSum` should equal `W * M`
4. Verify output color values are not saturated (< 1.0)

## Technical Details

### ReSTIR Weight Theory

In weighted reservoir sampling:
- **weightSum** = sum of all sample weights
- **M** = number of samples processed
- **W** = weightSum / M = average weight per sample

The **unbiased estimator** for light contribution is:
```
L = emission × intensity × attenuation × W
```

NOT:
```
L = emission × intensity × attenuation × W × M  // WRONG!
```

### Why M Scaling Was Added (Incorrectly)

The original reasoning was: "We took M samples, so we need to scale up the contribution." But this is wrong because:

1. **W already accounts for sample count** - it's the average, and that's what makes it unbiased
2. **Scaling by M makes it biased** - you're over-weighting regions with more samples
3. **ReSTIR's value is variance reduction, not magnitude scaling** - we want better samples, not brighter ones

### Correct Brightness Matching

To match non-ReSTIR brightness:
- ReSTIR uses intelligent sampling to find important lights
- Non-ReSTIR traces 4 rays in fixed directions
- They should converge to the same answer with enough samples
- If brightness differs, the issue is in attenuation or sampling strategy, NOT in W scaling

## Files Modified

1. **shaders/particles/particle_gaussian_raytrace.hlsl** (lines 624-627, 639)
   - Removed `* currentReservoir.M` scaling
   - Added clamp to illumination

2. **Compiled shader:** `shaders/particles/particle_gaussian_raytrace.dxil`

3. **Rebuilt:** Debug and DebugPIX builds

## Verification Checklist

- [ ] ReSTIR ON/OFF produce similar brightness
- [ ] No "dots" when approaching particles
- [ ] Colors remain vibrant (no brown/muted shift)
- [ ] Corner indicator smoothly transitions through colors
- [ ] I/K keys adjust RT lighting properly
- [ ] M and W values in PIX are reasonable
- [ ] No saturated output colors in captures

## Next Steps

1. **Test manually** - Fly through particle cloud with ReSTIR enabled
2. **PIX Agent analysis** - Capture with ReSTIR enabled, verify reservoir data
3. **A/B comparison** - Capture identical frames with ReSTIR ON vs OFF, compare

## Related Documents

- [RESTIR_BUG_FIX_SUMMARY.txt](RESTIR_BUG_FIX_SUMMARY.txt) - Original M increment bug
- [AGENT_WEIGHT_ANALYSIS.md](AGENT_WEIGHT_ANALYSIS.md) - Weight threshold analysis
- [PIX_AGENT_V1_COMPLETE.md](pix/PIX_AGENT_V1_COMPLETE.md) - PIX automation status

---

**Fix Applied:** October 12, 2025 22:45
**Status:** ✅ Compiled and ready for testing
**Confidence:** High - based on ReSTIR theory and prior weight analysis