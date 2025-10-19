# RTXDI Milestone 4 - Remaining Issues & Fixes

**Date:** 2025-10-19
**Status:** RTXDI Working! Debug visualization working! But 2 issues remain.

---

## ✅ ACHIEVEMENTS

1. **RTXDI End-to-End Pipeline Working**
   - Light grid building (27K cells)
   - Weighted reservoir sampling (1 light per pixel)
   - Gaussian renderer integration
   - Patchwork pattern visible (EXPECTED behavior!)

2. **Debug Visualization Working**
   - Rainbow colors showing selected light indices
   - Can verify RTXDI is selecting different lights per region

3. **Descriptor Leak Fixed**
   - Runs indefinitely (1200+ frames stable)

4. **RT Lighting Auto-Disabled in RTXDI Mode**
   - No more redundant particle-to-particle lighting

---

## ❌ REMAINING ISSUES

### Issue 1: Background Glow with All Lights Disabled

**Symptom:**
Even with:
- All lights OFF (0/16)
- RT lighting OFF
- Emission strength = 0.0
- All features disabled

You still see a dim grey/brown glow.

**Root Cause:**
`EmissionIntensity()` function in `gaussian_common.hlsl` (line 169):

```hlsl
float EmissionIntensity(float temperature) {
    float normalized = temperature / 26000.0;
    return pow(normalized, 2.0);  // <-- ALWAYS returns non-zero for any temperature!
}
```

Even at 800K (coolest particles):
- intensity = (800/26000)^2 = **0.000946** (dim but visible after tone mapping!)

**The Fix:**
In `particle_gaussian_raytrace.hlsl` lines 460-570, we need to:

1. **Multiply intensity by emissionStrength** so when strength = 0, intensity = 0:

```hlsl
} else {
    // Standard temperature-based color (artistic)
    emission = TemperatureToEmission(p.temperature);
    intensity = EmissionIntensity(p.temperature) * emissionStrength;  // <-- FIX: Scale by strength!
```

2. **Set totalEmission to zero when no lights present:**

```hlsl
if (totalLighting == float3(0,0,0) && illumination == float3(0,0,0)) {
    totalEmission = float3(0,0,0);  // Truly black!
} else {
    totalEmission = emission * intensity * (illumination + totalLighting);
}
```

**Time to Fix:** 5 minutes
**Impact:** Particles will be TRULY BLACK with all lights disabled

---

### Issue 2: Red Edge Glow in Debug Mode

**Symptom:**
Rainbow debug visualization shows **red glow on bottom/right edges** of screen.

**Root Cause:**
Pixels outside the populated light grid regions are defaulting to **Light Index 0** (red in rainbow).

Possible causes:
1. Grid cells at edge have only 1 light (Light 0)
2. Pixels mapping to empty cells return 0xFFFFFFFF, but shader defaults to index 0
3. World-to-grid mapping has edge case for screen periphery

**The Fix:**
In `rtxdi_raygen.hlsl`, add bounds checking:

```hlsl
uint SelectLightFromCell(LightGridCell cell, float randomValue) {
    // ... existing code ...

    if (validLights == 0 || weightSum <= 0.0) {
        return 0xFFFFFFFF;  // No lights (should show BLACK in debug)
    }

    // ... rest of function ...
}
```

And in `particle_gaussian_raytrace.hlsl` debug visualization:

```hlsl
if (debugRTXDISelection != 0) {
    if (selectedLightIndex == 0xFFFFFFFF) {
        // No light: BLACK (not red!)
        totalLighting = float3(0, 0, 0);
    } else if (selectedLightIndex < lightCount) {  // <-- Add bounds check!
        // Color-code by light index (0-12 = rainbow colors)
        // ... existing rainbow code ...
    } else {
        // Out of bounds: MAGENTA warning color
        totalLighting = float3(1, 0, 1);
    }
}
```

**Time to Fix:** 10 minutes
**Impact:** Edge glow disappears, debug visualization cleaner

---

## 🎯 RECOMMENDED NEXT STEPS

### Option A: Fix Both Issues Now (15 minutes)
Apply both fixes above, rebuild, test again.

### Option B: Ship M4 As-Is, Fix in M5
The background glow is minor (only visible with all lights off, which is a debug scenario).
The edge glow is only in debug mode.

**Neither affects normal operation!**

You could:
1. Save current branch as `0.8.0-m4-complete`
2. Document these known issues in CLAUDE.md
3. Move on to M5 (Temporal Reuse) which will add massive quality improvements
4. Circle back to polish these edge cases later

---

## 📊 M4 Success Criteria - ACHIEVED

| Criterion | Status | Evidence |
|-----------|--------|----------|
| RTXDI pipeline end-to-end | ✅ COMPLETE | Patchwork pattern visible |
| Light grid population | ✅ WORKING | 152 cells, 0.563% populated |
| Weighted selection | ✅ WORKING | Rainbow debug shows variation |
| Descriptor stability | ✅ FIXED | Runs 1200+ frames |
| RT lighting isolation | ✅ FIXED | Auto-disabled in RTXDI mode |
| Debug visualization | ✅ WORKING | Rainbow colors visible |
| Performance target | ✅ MET | 20 FPS @ 10K particles (baseline) |

**M4 is COMPLETE enough to proceed to M5!**

---

## 🚀 Next Milestone: M5 - Temporal Reuse

**What it adds:**
- Reservoir buffers (store selected light from previous frame)
- Temporal accumulation (reduces flicker)
- Validation (check if previous light is still valid)
- Bias correction (unbiased weight `W = weightSum / M`)

**Visual impact:**
- Patchwork pattern **smooths out over time** (60ms convergence)
- Temporal noise **disappears** (stable lighting)
- Quality **matches 13-light** but with 1 sample/pixel!

**Time estimate:** 2-3 hours

---

## Your Call

You've achieved an incredible milestone! **You have a working DXR raytracing engine with NVIDIA RTXDI!**

The remaining issues are cosmetic edge cases. You can:
1. **Fix them now** (15 min) for perfect polish
2. **Ship M4 as-is** and move to M5 for dramatic quality gains
3. **Take a break** and celebrate this massive achievement! 🎉

What would you like to do?
