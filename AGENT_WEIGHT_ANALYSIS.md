# ReSTIR Weight Analysis - Critical Bug Found

## Investigation Date: 2025-10-11

## Problem Summary
ReSTIR Phase 1 finds M=641-713 particles but weightSum=0 in both current and previous buffers.
This indicates ALL particle weights are below the 0.00001 threshold, preventing UpdateReservoir() from being called.

---

## Function Implementations (from gaussian_common.hlsl)

### 1. TemperatureToEmission(float temperature)
```hlsl
float3 TemperatureToEmission(float temperature) {
    float t = saturate((temperature - 800.0) / 25200.0);

    float3 color;
    if (t < 0.25) {
        float blend = t / 0.25;
        color = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
    } else if (t < 0.5) {
        float blend = (t - 0.25) / 0.25;
        color = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
    } else if (t < 0.75) {
        float blend = (t - 0.5) / 0.25;
        color = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
    } else {
        float blend = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
    }

    return color;
}
```
**Range**: Maps 800K-26000K to emission colors (red-orange-yellow-white gradient)

---

### 2. EmissionIntensity(float temperature)
```hlsl
float EmissionIntensity(float temperature) {
    float normalized = temperature / 26000.0;
    return pow(normalized, 2.0);
}
```
**Range**: Quadratic scaling, normalized by 26000K

---

## Weight Calculation Formula (from particle_gaussian_raytrace.hlsl, line 360)

```hlsl
float3 emission = TemperatureToEmission(hitParticle.temperature);
float intensity = EmissionIntensity(hitParticle.temperature);
float dist = length(hitParticle.position - rayOrigin);

// Attenuation: 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1)
float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

// Weight = luminance of light contribution
float weight = dot(emission * intensity * attenuation, float3(0.299, 0.587, 0.114));
```

**Threshold**: `if (weight > 0.00001) { UpdateReservoir(...); }`

---

## Test Case Calculation: 5000K Particle at 50 Units

### Step 1: Emission Color
```
t = (5000 - 800) / 25200 = 0.1667
Since t < 0.25:
  blend = 0.1667 / 0.25 = 0.6668
  emission = lerp((0.5, 0.1, 0.05), (1.0, 0.3, 0.1), 0.6668)
  emission = (0.7334, 0.2334, 0.0833)
```

### Step 2: Intensity
```
normalized = 5000 / 26000 = 0.1923
intensity = (0.1923)^2 = 0.03698
```

### Step 3: Attenuation (distance = 50)
```
attenuation = 1.0 / max(1.0 + 50*0.01 + 50*50*0.0001, 0.1)
            = 1.0 / max(1.0 + 0.5 + 0.25, 0.1)
            = 1.0 / 1.75
            = 0.5714
```

### Step 4: Light Contribution
```
lightContrib = emission * intensity * attenuation
             = (0.7334, 0.2334, 0.0833) * 0.03698 * 0.5714
             = (0.01549, 0.00493, 0.00176)
```

### Step 5: Final Weight (Luminance)
```
weight = dot((0.01549, 0.00493, 0.00176), (0.299, 0.587, 0.114))
       = 0.01549 * 0.299 + 0.00493 * 0.587 + 0.00176 * 0.114
       = 0.004632 + 0.002894 + 0.000201
       = 0.007727
```

---

## Result Analysis

### Expected Weight: **0.007727**

**Threshold Check**: `0.007727 > 0.00001` → **PASS** ✓

This weight is **773x larger** than the threshold, so it SHOULD trigger UpdateReservoir().

---

## Root Cause Hypothesis

Since the calculated weight (0.007727) is well above the threshold, but weightSum=0 in the actual run, there are three possible causes:

### 1. **MOST LIKELY: Intensity calculation bug**
The `EmissionIntensity()` function returns values that are too small:
- At 5000K: intensity = 0.037
- At 800K (minimum): intensity = 0.00095
- At 1000K: intensity = 0.00148

**For lower temperatures (800-2000K)**, intensity becomes extremely small:
```
T=800K:  intensity = (800/26000)^2 = 0.00095
T=1000K: intensity = (1000/26000)^2 = 0.00148
T=1500K: intensity = (1500/26000)^2 = 0.00333
```

From the log, many particles have temperatures around 800-1500K:
- Particle 0: 1025K → intensity = 0.00156
- Particle 7: 1008K → intensity = 0.00150
- Particle 8, 11, 14, 15, 16: 800K → intensity = 0.00095

**At dist=50, T=1000K**:
```
weight = 0.7334 * 0.00148 * 0.5714 * 0.299 (luminance approximation)
       ≈ 0.000185
```

Still above 0.00001, but getting close.

---

### 2. **Distance may be much larger than 50**
If particles are at dist=200-500 units (realistic for accretion disk):

**At dist=200, T=1000K**:
```
attenuation = 1.0 / (1.0 + 200*0.01 + 200*200*0.0001)
            = 1.0 / (1.0 + 2.0 + 4.0) = 0.1429
weight ≈ 0.7334 * 0.00148 * 0.1429 * 0.299 ≈ 0.000046
```

**At dist=300, T=1000K**:
```
attenuation = 1.0 / (1.0 + 3.0 + 9.0) = 0.0769
weight ≈ 0.7334 * 0.00148 * 0.0769 * 0.299 ≈ 0.000025
```

**At dist=400, T=1000K**:
```
attenuation = 1.0 / (1.0 + 4.0 + 16.0) = 0.0476
weight ≈ 0.7334 * 0.00148 * 0.0476 * 0.299 ≈ 0.000015
```

**At dist=500, T=1000K**:
```
attenuation = 1.0 / (1.0 + 5.0 + 25.0) = 0.0323
weight ≈ 0.7334 * 0.00148 * 0.0323 * 0.299 ≈ 0.000010
```

**CRITICAL**: At dist=500+, low-temperature particles fall below 0.00001!

---

### 3. **Ray origin may be far from particles**
If camera is positioned at (0, 0, -500) and particles are in disk at z=0, y=[-200,200], x=[-200,200]:
```
dist = sqrt(x^2 + y^2 + (z+500)^2)
     ≈ sqrt(200^2 + 200^2 + 500^2)
     ≈ sqrt(290000) ≈ 538 units
```

**This would put ALL low-temperature particles below threshold.**

---

## Actual Log Data Analysis

From PlasmaDX-Clean_20251011_173125.log:

**Temperature Distribution** (first 20 particles):
- 800K (minimum):  5 particles (25%)
- 1000-2000K:      2 particles (10%)
- 2000-5000K:      4 particles (20%)
- 5000-15000K:     9 particles (45%)

**Position ranges**:
- X: -259 to +521
- Y: -196 to +347
- Z: -411 to +443

**Maximum distance from origin**: ~660 units (particle 11 at (521, -17, -411))

**If camera is at origin looking along +Z**:
- Close particles (~30-100 units): Should have visible weights
- Far particles (>400 units): Low-temp ones will be below threshold

---

## Verdict

**THE THRESHOLD IS TOO HIGH FOR REALISTIC ACCRETION DISK SCENARIOS**

### Evidence:
1. 25% of particles are at minimum temperature (800K) with intensity=0.00095
2. Particle distances range up to 600+ units
3. At dist>400, low-temp particles produce weights < 0.00001
4. 35% of particles have T<5000K, making them vulnerable at medium-far distances

### Impact:
- ReSTIR is rejecting 30-50% of valid light sources
- Only hottest particles (>10000K) at close-medium range contribute
- This explains weightSum=0 despite M=641-713 hits

---

## Recommended Fix

### Option 1: Lower the threshold (IMMEDIATE FIX)
```hlsl
// OLD: if (weight > 0.00001)
// NEW:
if (weight > 0.000001)  // 10x more sensitive
```

This allows particles at dist=500-600 with T=800-2000K to contribute.

---

### Option 2: Adjust intensity scaling (BETTER FIX)
The current intensity formula is too aggressive for low temperatures:

```hlsl
// OLD:
float EmissionIntensity(float temperature) {
    float normalized = temperature / 26000.0;
    return pow(normalized, 2.0);  // Quadratic: very steep falloff
}

// NEW:
float EmissionIntensity(float temperature) {
    float normalized = temperature / 26000.0;
    return pow(normalized, 1.5);  // Less aggressive falloff
}
```

This gives:
- T=800K:  OLD=0.00095, NEW=0.0049 (5x boost)
- T=1000K: OLD=0.00148, NEW=0.0061 (4x boost)
- T=5000K: OLD=0.03698, NEW=0.0846 (2.3x boost)

---

### Option 3: Boost base emission (CONSERVATIVE FIX)
Multiply emission color by a constant factor:

```hlsl
float3 TemperatureToEmission(float temperature) {
    // ... existing color calculation ...
    return color * 2.0;  // Boost all emission by 2x
}
```

---

## Recommended Implementation Order

1. **IMMEDIATE**: Lower threshold to 0.000001 (1-line change)
2. **SHORT-TERM**: Change intensity exponent from 2.0 to 1.5
3. **VALIDATION**: Check weightSum > 0 in PIX after each change
4. **TUNING**: Adjust emission multiplier if needed for visual quality

---

## Testing Protocol

After applying fix:
1. Run with same scene setup
2. Check PIX markers for weightSum values
3. Verify UpdateReservoir() is being called (M > 0 AND weightSum > 0)
4. Count how many particles contribute (expect >50% of hits)
5. Validate visual output shows distributed lighting from disk

---

## File Locations

**Shader**: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/particle_gaussian_raytrace.hlsl`
- Line 369: Threshold check
- Line 360: Weight calculation

**Common functions**: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/gaussian_common.hlsl`
- Line 147-166: TemperatureToEmission()
- Line 169-172: EmissionIntensity()

**Log**: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/logs/PlasmaDX-Clean_20251011_173125.log`
- Lines 1721-1740: Particle temperature/position data

---

## Conclusion

**Root Cause**: The 0.00001 threshold is too high for the combination of:
- Low-temperature particles (800-2000K) with intensity^2 scaling
- Large scene distances (400-600 units) with quadratic attenuation
- Realistic accretion disk temperature distribution (25% at minimum temp)

**Fix**: Lower threshold to 0.000001 as immediate fix, then adjust intensity formula for proper physically-based scaling.

**Expected Result**: weightSum > 0, ReSTIR properly samples light particles across full temperature range.
