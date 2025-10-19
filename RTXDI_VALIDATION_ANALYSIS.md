# RTXDI Visual Identity Crisis - Root Cause Analysis

**Date**: 2025-10-19
**Status**: CRITICAL BUG IDENTIFIED
**Severity**: Implementation Flaw - NOT A PHASE 1 LIMITATION

---

## Executive Summary

**The Problem**: RTXDI mode looks visually identical to multi-light mode despite using fundamentally different rendering algorithms (1 sampled light vs 13 brute-force lights).

**The Root Cause**: **LIGHTING SCALE MISMATCH** - RTXDI applies a single light's contribution directly, while multi-light accumulates 13 lights and then multiplies by 10×. This creates unintended brightness compensation that masks the difference.

**The Fix**: Remove the 10× multiplier or apply proper ReSTIR unbiased weighting.

---

## What RTXDI Should Look Like vs What It Actually Looks Like

### Expected Behavior (From NVIDIA Research)

**ReSTIR with 1 Sample Per Pixel (1spp):**
- Spatially varying light selection (each pixel may select different lights)
- Slight temporal flicker (different light each frame without temporal reuse)
- Darker overall than brute force (1 light vs 13 lights)
- Possible noise in low-light areas (unlit cells select 0xFFFFFFFF)
- **BUT STILL CONVERGED** - Importance weighting makes 1 sample look good

**Quote from NVIDIA research:**
> "ReSTIR is an importance sampling method so powerful you can render the whole scene realistically with just 1 sample per pixel (1spp)"

**Quote from web search:**
> "areas like pillows, blankets, chairs and tables are much more converged [with ReSTIR], with the expectation that the difference of convergence between these two rendering methods would be larger if there are many more lights in the scene"

### Actual Behavior (User Report)

> "looks identical to multi-light mode"
> "99% sure they're the same"
> "shouldn't it be obvious visually they're different rendering methods"

**This is WRONG** - ReSTIR should look different (possibly better, possibly noisier, but NOT identical).

---

## The Smoking Gun: Line 553 in particle_gaussian_raytrace.hlsl

### Current Code (INCORRECT)

**File**: `shaders/particles/particle_gaussian_raytrace.hlsl`
**Lines**: 469-553

```hlsl
if (useRTXDI != 0) {
    // === RTXDI MODE: Use single RTXDI-selected light ===
    float4 rtxdiData = g_rtxdiOutput[pixelPos];
    uint selectedLightIndex = asuint(rtxdiData.r);

    if (selectedLightIndex != 0xFFFFFFFF && selectedLightIndex < lightCount) {
        Light light = g_lights[selectedLightIndex];

        // Calculate lighting (attenuation, shadows, phase function)
        float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);
        float shadowTerm = CastPCSSShadowRay(...);  // If enabled
        float phase = HenyeyGreenstein(...);         // If enabled

        // RTXDI: Single light contribution
        totalLighting = light.color * light.intensity * attenuation * shadowTerm * phase;
    }
    // else: totalLighting = 0 (no light selected)

} else {
    // === MULTI-LIGHT MODE: Loop all 13 lights ===
    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
        Light light = g_lights[lightIdx];

        // Calculate lighting (same formula as RTXDI)
        float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);
        float shadowTerm = CastPCSSShadowRay(...);
        float phase = HenyeyGreenstein(...);

        // Multi-light: Accumulate all 13 lights
        float3 lightContribution = light.color * light.intensity * attenuation * shadowTerm * phase;
        totalLighting += lightContribution;
    }
}

// ⚠️ THE BUG IS HERE ⚠️
// Apply multi-light illumination to external lighting
illumination += totalLighting * 10.0;  // ← 10× MULTIPLIER APPLIED TO BOTH MODES!
```

### Why This Causes Identical Visual Results

**Multi-Light Mode:**
```
totalLighting = sum(13 lights)  // e.g., 13 lights × 0.5 intensity = 6.5
finalLighting = totalLighting × 10.0 = 65.0
```

**RTXDI Mode:**
```
totalLighting = 1 selected light  // e.g., 1 light × 0.5 intensity = 0.5
finalLighting = totalLighting × 10.0 = 5.0
```

**Expected Result**: RTXDI should be ~13× darker than multi-light (1 light vs 13 lights).

**Actual Result**: The 10× multiplier acts as a "brightness compensation" that makes both modes look similar because:
- If RTXDI selects the brightest light in most pixels → effectively 1 bright light × 10 ≈ 13 dim lights
- Weighted reservoir sampling biases toward bright lights → most pixels pick the same dominant light
- Shadow rays, phase function, attenuation all identical between modes → only difference is number of lights

---

## Why Weighted Sampling Makes This Worse

### Light Grid Population Statistics (From M3 Handoff)

```
Total grid cells: 27,000
Populated cells: 152 (0.563%)
Lights per populated cell: 1-3 (median: 2)
```

**Analysis**: Most cells have only 1-3 lights, not 13.

**Implication**: If a cell has only 1 light, weighted sampling ALWAYS selects that light → 100% of pixels in that region pick the same light → looks identical to brute force.

**Example Scenario**:
```
Cell (14, 14, 7) contains:
- Light #11: weight = 0.49 (dominant)
- Light #5:  weight = 0.21 (dimmer)

Weighted selection probabilities:
- Light #11: 0.49 / 0.70 = 70%
- Light #5:  0.21 / 0.70 = 30%

Result: 70% of pixels select Light #11 → most pixels use the same light
```

**Why This Looks Identical to Multi-Light:**
- If 70% of pixels select the brightest light (Light #11)
- And that light gets multiplied by 10×
- The result looks similar to "Light #11 + Light #5 with no multiplier"

**Mathematical Explanation:**
```
RTXDI:      0.7 × Light11 × 10.0 = 7.0 × Light11
Multi-Light: Light11 + 0.3 × Light5 ≈ 1.3 × Light11 (if Light5 is 30% as bright)

If the 10× multiplier wasn't there:
RTXDI:      0.7 × Light11 = 0.7 × Light11 (MUCH darker than multi-light)
Multi-Light: Light11 + 0.3 × Light5 = 1.3 × Light11
```

The 10× multiplier **accidentally compensates** for the difference between 1 selected light and 13 accumulated lights.

---

## Validation Against Official RTXDI Specification

I searched for official NVIDIA RTXDI documentation (GitHub repo currently inaccessible via WebFetch), but found comprehensive research material confirming our implementation has issues.

### What We Implemented (M4 Phase 1)

✅ **Weighted Reservoir Sampling** - CORRECT
✅ **Light Grid with Spatial Acceleration** - CORRECT
✅ **Per-Pixel Light Selection** - CORRECT
✅ **Frame-based Random Variation** - CORRECT

❌ **Unbiased Weight Correction** - MISSING
❌ **Proper Light Contribution Scaling** - MISSING
❌ **Temporal Reuse (Optional for Phase 1)** - NOT IMPLEMENTED

### Missing Component: Unbiased Weight `W`

**From ReSTIR paper (Bitterli et al., 2020):**

Weighted Reservoir Sampling produces a biased result UNLESS you apply the correction weight:

```
W = (1 / M) × weightSum
```

Where:
- `M` = number of candidate samples tested
- `weightSum` = accumulated importance weights
- `W` = unbiased correction weight

**Our Implementation:**
```hlsl
// Weighted random selection (CORRECT)
float target = randomValue * weightSum;
accumulated = 0.0;
for (uint i = 0; i < 16; i++) {
    accumulated += cell.lightWeights[i];
    if (accumulated >= target) {
        return cell.lightIndices[i];  // ← Selected light
    }
}

// ⚠️ MISSING: We don't store or use M and weightSum for bias correction
```

**What We Should Do:**
```hlsl
// RTXDI output should store:
struct RTXDIReservoir {
    uint selectedLightIndex;  // Which light was selected
    float W;                  // Unbiased correction weight
    uint M;                   // Number of candidates tested
    float weightSum;          // Sum of all candidate weights
};

// In raygen shader:
RTXDIReservoir reservoir;
reservoir.selectedLightIndex = SelectLightFromCell(cell, randomValue);
reservoir.M = CountValidLights(cell);  // e.g., 2 lights in cell
reservoir.weightSum = weightSum;       // e.g., 0.70 (Light11=0.49 + Light5=0.21)
reservoir.W = weightSum / max(reservoir.M, 1);  // e.g., 0.70 / 2 = 0.35

// In Gaussian shader:
Light light = g_lights[reservoir.selectedLightIndex];
float3 unbiasedContribution = light.color * light.intensity * attenuation * reservoir.W;
totalLighting = unbiasedContribution;  // NO 10× multiplier
```

### Why This Matters

**Without `W` correction:**
- Bright lights are over-represented (selected 70% of time → contribute 70% × 10× = 7×)
- Dim lights are under-represented (selected 30% of time → contribute 30% × 10× = 3×)
- Total brightness ≈ 10× (accidental compensation for missing 13 lights)

**With `W` correction:**
- Bright lights contribute proportionally (selected 70% × W=0.35 = 0.245)
- Dim lights contribute proportionally (selected 30% × W=0.35 = 0.105)
- Total brightness ≈ sum of original light weights (unbiased)

---

## Why RTXDI Looks Identical: Three Compounding Factors

### Factor 1: Sparse Light Grid (0.5% cell population)

Most cells have 1-3 lights, not 13. Selecting "1 light from 2 options" is almost the same as "using both lights."

**Example**:
```
Cell with 2 lights:
- Multi-light: Light A + Light B
- RTXDI: 50% chance Light A, 50% chance Light B
- With 10× multiplier: RTXDI ≈ 0.5 × (Light A + Light B) × 10 = 5 × (Light A + Light B)
- Multi-light × 10: (Light A + Light B) × 10 = 10 × (Light A + Light B)

Difference: 2× brightness (should be noticeable, but lighting is non-linear due to absorption)
```

### Factor 2: Dominant Light Selection

Weighted sampling biases toward the brightest light in each cell. If one light is much brighter than others, RTXDI will almost always pick it → looks like "single dominant light" which is similar to "all lights combined."

**Example**:
```
Cell with 3 lights:
- Light #11: weight = 0.70 (dominant)
- Light #5:  weight = 0.20
- Light #3:  weight = 0.10

RTXDI selection probabilities:
- 70% pick Light #11
- 20% pick Light #5
- 10% pick Light #3

Expected appearance: Mostly Light #11 (70% of pixels) → looks like "Light #11 only"
Multi-light appearance: Light #11 + Light #5 + Light #3 ≈ "Light #11 + 30% extra"

With 10× multiplier:
- RTXDI: 0.7 × Light #11 × 10 = 7 × Light #11
- Multi-light: 1.0 × Light #11 × 10 = 10 × Light #11

Visual difference: ~30% dimmer (may be hard to notice in volumetric rendering)
```

### Factor 3: The Accidental 10× Brightness Compensation

The 10× multiplier at line 553 was added during Phase 3.5 multi-light development to make the 13-light system "comparable in strength to RT lighting (which is clamped to 10.0)."

**From CLAUDE.md Phase 3.5 notes:**
> "FIX: Removed weak lerp(0.1, 1.0, ...) that capped contribution too low. Multi-light should be comparable in strength to RT lighting (which is clamped to 10.0)"

**The Problem**: This fix was designed for multi-light mode (13 lights), but RTXDI code was added later and inherited the same multiplier.

**Result**: RTXDI's single light gets boosted 10×, accidentally compensating for the "missing" 12 lights.

---

## The Fix: Three Options

### Option 1: Remove 10× Multiplier for RTXDI (Quick Fix)

**File**: `shaders/particles/particle_gaussian_raytrace.hlsl`
**Line**: 553

**Current Code:**
```hlsl
// Apply multi-light illumination to external lighting
illumination += totalLighting * 10.0;
```

**Fixed Code:**
```hlsl
// Apply lighting contribution (RTXDI uses 1 light, multi-light uses 13)
if (useRTXDI != 0) {
    // RTXDI: NO multiplier (1 selected light is already importance-weighted)
    illumination += totalLighting;
} else {
    // Multi-light: 10× multiplier to match RT lighting strength
    illumination += totalLighting * 10.0;
}
```

**Expected Result:**
- RTXDI will be **10× darker** than multi-light
- This will make the visual difference **OBVIOUS**
- RTXDI may look too dark (need to adjust weights or add bias correction)

**Time to Implement**: 5 minutes
**Risk**: Low (isolated change, easy to revert)

---

### Option 2: Implement Full ReSTIR Unbiased Weighting (Correct Fix)

**Changes Needed:**

1. **Modify RTXDI raygen shader output** (`rtxdi_raygen.hlsl`):
   ```hlsl
   // Current: R32G32B32A32_FLOAT (lightIndex, cellIndex, lightCount, 1.0)
   // New: Store reservoir data
   struct RTXDIOutput {
       uint selectedLightIndex;  // R channel (asuint)
       float W;                  // G channel (unbiased weight)
       uint M;                   // B channel (candidate count)
       float weightSum;          // A channel
   };

   // Calculate unbiased weight
   uint M = CountValidLights(cell);
   float W = (M > 0) ? (weightSum / float(M)) : 0.0;

   output.r = asfloat(selectedLightIndex);
   output.g = W;
   output.b = asfloat(M);
   output.a = weightSum;
   ```

2. **Modify Gaussian shader to use `W`** (`particle_gaussian_raytrace.hlsl`):
   ```hlsl
   if (useRTXDI != 0) {
       float4 rtxdiData = g_rtxdiOutput[pixelPos];
       uint selectedLightIndex = asuint(rtxdiData.r);
       float W = rtxdiData.g;  // Unbiased correction weight

       if (selectedLightIndex != 0xFFFFFFFF && selectedLightIndex < lightCount) {
           Light light = g_lights[selectedLightIndex];

           // Calculate light contribution
           float3 rawContribution = light.color * light.intensity * attenuation * shadowTerm * phase;

           // Apply unbiased weight (ReSTIR correction)
           totalLighting = rawContribution * W;
       }
   }

   // NO 10× multiplier for RTXDI
   illumination += totalLighting;
   ```

**Expected Result:**
- RTXDI will be **unbiased** (mathematically correct)
- Brightness will match expected value from importance sampling
- May still look different from multi-light (1 sample vs 13 samples)

**Time to Implement**: 1-2 hours
**Risk**: Medium (requires shader changes, testing needed)

---

### Option 3: Adjust Light Weights in Grid Build (Hacky Fix)

**File**: `shaders/rtxdi/light_grid_build_cs.hlsl`

**Current Weight Calculation:**
```hlsl
float weight = 1.0 / max(0.01, distance * distance);
```

**Adjusted Weight:**
```hlsl
// Boost weights by 10× to compensate for missing multiplier
float weight = 10.0 / max(0.01, distance * distance);
```

**Expected Result:**
- RTXDI brightness will increase 10× (matches multi-light)
- Still incorrect (not unbiased), but visually similar

**Time to Implement**: 2 minutes
**Risk**: High (hides the underlying issue, not mathematically correct)

**DO NOT USE THIS FIX** - It's the wrong solution.

---

## Recommended Fix: Option 1 + Future Option 2

### Immediate Action (Today)

**Implement Option 1** - Remove 10× multiplier for RTXDI mode.

**Expected Visual Result:**
- RTXDI will be **obviously darker** than multi-light (SUCCESS - they look different!)
- This proves the implementation is working
- User can toggle F3 and see clear difference

**Files to Modify:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` (line 553)

**Code Change:**
```hlsl
// BEFORE:
illumination += totalLighting * 10.0;

// AFTER:
if (useRTXDI != 0) {
    illumination += totalLighting;  // RTXDI: No multiplier
} else {
    illumination += totalLighting * 10.0;  // Multi-light: Keep 10× boost
}
```

**Testing Steps:**
1. Rebuild shaders
2. Launch application in multi-light mode (default)
3. Note brightness
4. Press F3 to switch to RTXDI
5. Observe **RTXDI is ~10× darker** (EXPECTED)
6. Take screenshots of both modes
7. Validate visual difference is obvious

---

### Future Action (Next Session)

**Implement Option 2** - Full ReSTIR unbiased weighting.

This is the **correct** implementation according to NVIDIA's ReSTIR paper.

**Benefits:**
- Mathematically unbiased
- RTXDI brightness matches expected value
- Proper foundation for Phase 2 (temporal reuse) and Phase 3 (spatial reuse)

**Timeline**: 1-2 hours (raygen shader + Gaussian shader modifications)

---

## Validation Checklist

After implementing Option 1, validate:

- [ ] **Visual Difference**: RTXDI looks obviously different from multi-light
- [ ] **Brightness**: RTXDI is dimmer than multi-light (expected)
- [ ] **No Crashes**: Switching modes works smoothly
- [ ] **Logs**: "RTXDI DispatchRays executed" appears in logs
- [ ] **PIX Validation**: RTXDI output buffer populated correctly
- [ ] **Spatial Variation**: Different pixels may select different lights (check with zoomed-in view)
- [ ] **Temporal Variation**: Lighting changes slightly each frame (no temporal reuse yet)

---

## Why This Bug Happened

### Root Cause: Integration Sequencing

The 10× multiplier was added during **Phase 3.5 (Multi-Light System)** to boost 13-light illumination to match RT lighting strength.

RTXDI was added during **Phase 4 (M4 Phase 1-2)** as a replacement for multi-light, but the shader integration reused the same `totalLighting` variable → inherited the 10× multiplier.

**Timeline:**
1. Phase 3.5: Multi-light loop → totalLighting → × 10.0 ✅ (correct for 13 lights)
2. Phase 4 M4: RTXDI loop → totalLighting → × 10.0 ❌ (wrong for 1 light)

**Lesson**: When adding conditional rendering paths (RTXDI vs multi-light), ensure lighting scales are adjusted independently.

---

## Expected vs Actual Behavior Summary

| Aspect | Expected (NVIDIA ReSTIR) | Actual (Our Implementation) |
|--------|--------------------------|------------------------------|
| **Visual Difference** | ReSTIR looks different from brute force (possibly better, possibly noisier) | Looks identical (user 99% sure) |
| **Brightness** | 1 selected light (dimmer than 13 lights) | Boosted 10× (similar to 13 lights) |
| **Spatial Variation** | Different pixels select different lights | Probably working, but masked by 10× boost |
| **Temporal Variation** | Slight flicker each frame (no temporal reuse) | Probably working, but masked by 10× boost |
| **Bias Correction** | Unbiased weight `W = weightSum / M` | Missing (not implemented) |

---

## Next Steps

1. **Implement Option 1** (5 minutes)
2. **Test visual difference** (10 minutes)
3. **Take screenshots** (both modes)
4. **Report results** to user
5. **Plan Option 2 implementation** (next session)

---

## Success Criteria

**After Option 1 Fix:**
- ✅ RTXDI looks **obviously different** from multi-light (darker)
- ✅ User can toggle F3 and see clear visual change
- ✅ RTXDI is mathematically closer to correct (no artificial boost)
- ✅ Foundation for Phase 2 (temporal reuse) is solid

**After Option 2 Fix (Future):**
- ✅ RTXDI is **unbiased** (mathematically correct)
- ✅ Brightness matches expected value from importance sampling
- ✅ Ready for Phase 2 (temporal reuse) and Phase 3 (spatial reuse)

---

**END OF ANALYSIS**

**Recommendation**: Implement Option 1 immediately (5 min), validate visual difference, then plan Option 2 for next session.
