# ReSTIR Debug Analysis - Why Fixes Aren't Working
**Date:** 2025-10-14 03:30
**Status:** All 3 fixes applied, but NO visual change observed

---

## Current Situation

**Applied fixes:**
1. ✅ Fix 1: MIS weight formula (W × M instead of W × 16 / M)
2. ✅ Fix 2: Distance-adaptive temporal weight (0.3 when close)
3. ✅ Fix 3: Increased clamp from 2.0 to 10.0

**Observation:**
- Shader is loading correctly (18860 bytes)
- ReSTIR is enabled (green corner indicator)
- **NO visual difference** from before fixes

**Critical user observation:**
> "I also noticed the past few builds that i can't increase the rt lighting intensity higher than default.
> In fact if i try to boost it up it actually becomes more muted."

---

## Hypothesis: The Values Are Too Small Regardless

### The Problem

Even with all fixes applied, if the base values are microscopically small, the fixes won't help:

```hlsl
// From previous analysis:
W = 0.000045 (at working distance)
W = 0.002171 (at buggy distance - 48× larger!)

// With Fix 1 (W × M):
misWeight = 0.002171 × 4 = 0.008684

// After clamp(0, 10.0):
misWeight = 0.008684 (unchanged, way under clamp)

// Multiply by directLight:
rtLight = lightEmission × lightIntensity × attenuation × 0.008684
// If lightEmission = (1, 0.5, 0.2), lightIntensity = 2.0, attenuation = 0.5:
rtLight = (1, 0.5, 0.2) × 2.0 × 0.5 × 0.008684
rtLight = (0.008684, 0.004342, 0.001737)  // Still TINY!

// Line 672: illumination += clamp(rtLight × rtLightingStrength, 0, 10)
// If rtLightingStrength = 1.0:
illumination += (0.008684, 0.004342, 0.001737)
// Final: illumination = (1.008684, 1.004342, 1.001737)
// This is BARELY above 1.0! Visually invisible!
```

---

## Why Boosting RT Intensity Makes It Darker

This is the KEY observation! When you press `I` to increase RT intensity:

```hlsl
// rtLightingStrength starts at 1.0
// Each press: rtLightingStrength *= 2.0
// After 5 presses: rtLightingStrength = 32.0

// Line 672: illumination += clamp(rtLight × 32.0, 0, 10)
// rtLight = (0.008684, 0.004342, 0.001737)
// rtLight × 32.0 = (0.278, 0.139, 0.056)
// After clamp(0, 10): (0.278, 0.139, 0.056)  // Under clamp, no issue

// SO WHY DOES IT GET DARKER???
```

**Wait...** Let me re-read the code at line 672:

```hlsl
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
```

This ADDS to illumination (starts at 1.0). So boosting should make it BRIGHTER, not darker.

**UNLESS...**

---

## New Hypothesis: The SECOND Clamp is The Problem!

Look at line 672 again:

```hlsl
float3 illumination = float3(1, 1, 1); // Base self-illumination

// Add RT lighting
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
```

If `rtLight × rtLightingStrength` is TINY (like 0.278), then:
- illumination = 1.0 + 0.278 = 1.278 ✅ (should be brighter!)

But then at line 683:

```hlsl
// Apply shadow to external illumination
illumination *= lerp(0.1, 1.0, shadowTerm);
```

If `shadowTerm` is low (shadows present), this DIVIDES illumination by 10!
- illumination = 1.278 × 0.1 = 0.1278 ❌ (MUCH DARKER!)

**But this affects BOTH ReSTIR ON and OFF...**

---

## Third Hypothesis: ReSTIR Weight is ZERO

What if the reservoir isn't finding ANY valid samples?

```hlsl
// Line 637:
if (useReSTIR != 0 && currentReservoir.M > 0 && currentReservoir.M != 88888) {
    // Use ReSTIR light
} else {
    // Fallback: Use pre-computed RT lighting
    rtLight = g_rtLighting[hit.particleIdx].rgb;
}
```

If `currentReservoir.M == 0` or `currentReservoir.M == 88888`:
- ReSTIR path is SKIPPED
- Falls back to `g_rtLighting` buffer
- **This buffer might be empty or invalid!**

---

## Fourth Hypothesis: Attenuation Formula is TOO Aggressive

Look at line 645 (and line 357 in sampling):

```hlsl
float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);
```

Let's compute this for typical distances:

| Distance | Calculation | Attenuation | Effect |
|----------|-------------|-------------|--------|
| 10 units | 1 / (1 + 0.1 + 0.01) = 1 / 1.11 | 0.90 | Good |
| 50 units | 1 / (1 + 0.5 + 0.25) = 1 / 1.75 | 0.57 | OK |
| 100 units | 1 / (1 + 1 + 1) = 1 / 3 | 0.33 | Dim |
| 200 units | 1 / (1 + 2 + 4) = 1 / 7 | 0.14 | Very dim |
| 500 units | 1 / (1 + 5 + 25) = 1 / 31 | 0.032 | **DARK!** |

**This might be the real bug!** At 200+ units (orange indicator), attenuation drops to 0.14-0.032.

Combined with small W values:
```
misWeight = 0.008684
attenuation = 0.032 (at 500 units)
rtLight = emission × intensity × 0.032 × 0.008684
rtLight = emission × intensity × 0.000278  // INVISIBLE!
```

---

## The REAL Solution: Boost W or Reduce Attenuation

### Option A: Boost W Values (100× multiplier)

The problem is that W values are physically correct but too small for visible contribution.

**Fix:** Multiply misWeight by 100 or 1000:

```hlsl
// Line 654:
float misWeight = currentReservoir.W * float(currentReservoir.M) * 100.0;  // Boost for visibility
```

### Option B: Weaker Attenuation (More Realistic for Large Scenes)

The current attenuation is TOO aggressive for accretion disk scales (10-300 radii).

**Fix:** Use linear-only attenuation (remove quadratic term):

```hlsl
// Line 645 (and line 357):
float attenuation = 1.0 / max(1.0 + dist * 0.001, 0.1);  // 10× weaker
```

| Distance | New Attenuation | Old Attenuation | Improvement |
|----------|-----------------|-----------------|-------------|
| 100 units | 0.91 | 0.33 | 2.75× brighter |
| 200 units | 0.83 | 0.14 | 5.9× brighter |
| 500 units | 0.67 | 0.032 | **21× brighter!** |

### Option C: Remove SECOND Clamp (Line 672)

The clamp on line 672 might be unnecessary if misWeight clamp is already applied:

```hlsl
// Line 672 - BEFORE:
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);

// Line 672 - AFTER:
illumination += rtLight * rtLightingStrength;  // No clamp!
```

This allows unlimited contribution from RT lighting.

---

## Recommended Debug Steps

### Step 1: Add Debug Visualization

Add this code after line 654 to see actual misWeight values:

```hlsl
// DEBUG: Visualize misWeight magnitude
if (pixelPos.x < 300 && pixelPos.y > resolution.y - 100) {
    // Show misWeight as color
    float misWeightVis = misWeight * 100.0;  // Scale for visibility
    g_output[pixelPos] = float4(misWeightVis, misWeightVis, misWeightVis, 1.0);
    return;  // Early exit to show just this
}
```

**Expected:**
- If bottom-left corner is BLACK: misWeight is too small (< 0.01)
- If bottom-left corner is GRAY: misWeight is reasonable (0.1-1.0)
- If bottom-left corner is WHITE: misWeight is good (> 1.0)

### Step 2: Try Extreme Boost (1000× multiplier)

```hlsl
// Line 654:
float misWeight = currentReservoir.W * float(currentReservoir.M) * 1000.0;
```

If this makes RT lighting VISIBLE, then the problem is scale (W too small).

### Step 3: Try Linear Attenuation

```hlsl
// Line 645 (and line 357):
float attenuation = 1.0 / max(1.0 + dist * 0.001, 0.1);
```

If this makes RT lighting VISIBLE, then the problem is attenuation too aggressive.

### Step 4: Remove ALL Clamps

```hlsl
// Line 658: Comment out clamp
// misWeight = clamp(misWeight, 0.0, 10.0);

// Line 672: Remove clamp
illumination += rtLight * rtLightingStrength;
```

If this makes RT lighting VISIBLE, then clamps were the problem.

---

## Most Likely Root Cause

Based on all evidence, I believe the issue is:

**Attenuation formula is TOO aggressive for the scene scale.**

- Accretion disk spans 10-300 radii (100-3000 units)
- Current attenuation: quadratic falloff (dist²)
- At 200-500 units: attenuation = 0.032-0.14 (way too dark!)
- Combined with small W (0.002): final contribution invisible

**Solution:**
Change attenuation from `dist²` to `dist` (linear falloff):

```hlsl
// OLD (line 357 and 645):
float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

// NEW (much weaker falloff for large scenes):
float attenuation = 1.0 / max(1.0 + dist * 0.001, 0.1);
```

This makes distant lights 10-20× brighter, which should make ReSTIR contribution finally visible!

---

## Action Plan

1. **Try Fix A (weaker attenuation)** - Most likely to work
2. **Try Fix B (boost misWeight by 100×)** - If attenuation fix not enough
3. **Try Fix C (remove second clamp)** - If still not visible
4. **Add debug visualization** - To confirm values are in correct range

Let me know which fix you want to try first, or if you want me to apply them all at once!

---

## Why Previous Fixes Didn't Work - Final Analysis

The three fixes we applied (W × M, adaptive temporal, increase clamp) were **mathematically correct** but addressed the WRONG problem:

- **Fix 1 (W × M):** Removed spatial inconsistency → CORRECT algorithm, but values still too small
- **Fix 2 (adaptive temporal):** Reduced stale samples → GOOD for quality, but doesn't affect brightness
- **Fix 3 (increase clamp to 10):** Allowed more range → GOOD, but values never reached 2.0 anyway!

**The REAL bug:**
Attenuation formula crushes distant lights to invisibility (0.032 at 500 units).
No amount of algorithmic fixes can overcome this - we need to change the physical attenuation model!

---

**Status:** Waiting for user decision on which fix to try
**Recommendation:** Apply weaker attenuation formula first (most likely to work)
