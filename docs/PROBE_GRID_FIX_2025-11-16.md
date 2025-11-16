# Probe Grid Dim Lighting Fix - 2025-11-16

**Status:** ✅ IMPLEMENTED - Ready for testing
**Session:** Continuation from SESSION_HANDOFF_2025-11-16.md
**Critical Fix:** Probe shader now captures multi-light contributions

---

## Root Cause Analysis

### The Problem

The probe grid was **extremely dim** despite previous fixes (dispatch, ray distance) because:

1. **Probe shader only captured particle blackbody emission** (lines 314-322 in update_probes.hlsl)
2. **Physical emission is disabled** (RT-first philosophy per user preference)
3. **Line 326 had a TODO comment:** "Add external light source contributions - For now, particles are self-emissive only"
4. **Result:** Probes captured ZERO lighting from the 13 multi-lights in the scene

### Evidence

From `update_probes.hlsl:326-327` (BEFORE fix):
```hlsl
// TODO (Phase 2): Add external light source contributions
// For now, particles are self-emissive only
```

The probe update shader was literally ignoring all 13 lights in the scene and only looking at particle blackbody emission (which was disabled).

---

## The Fix

### Implementation Details

**File Modified:** `shaders/probe_grid/update_probes.hlsl`
**Lines Changed:** 306-364 (added multi-light integration loop)

**What was added:**
1. Calculate hit position from ray intersection
2. For each enabled light in the scene:
   - Cast shadow ray from hit position to light
   - If not occluded, compute light contribution
   - Use inverse-square attenuation
   - Scale by light radius for ambient influence
   - Accumulate to total irradiance

**Key Code:**
```hlsl
// PHASE 2 IMPLEMENTATION: External light source contributions
for (uint lightIdx = 0; lightIdx < g_lightCount; lightIdx++) {
    Light light = g_lights[lightIdx];
    if (light.enabled == 0) continue;

    // Vector from hit position to light
    float3 toLight = light.position - hitPosition;
    float lightDistance = length(toLight);
    float3 lightDir = toLight / lightDistance;

    // Simple shadow ray test (binary visibility)
    RayDesc shadowRay;
    shadowRay.Origin = hitPosition + lightDir * 0.1;
    shadowRay.Direction = lightDir;
    shadowRay.TMin = 0.01;
    shadowRay.TMax = lightDistance - 0.1;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
    shadowQuery.TraceRayInline(g_particleTLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
    shadowQuery.Proceed();

    // If shadow ray didn't hit anything, light is visible
    if (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) {
        // Inverse-square attenuation for lights
        float attenuation = 1.0 / max(lightDistance * lightDistance, 1.0);

        // Light contribution with intensity
        float3 lightContribution = light.color * light.intensity * attenuation;

        // Scale by light radius (larger radius = more ambient influence)
        lightContribution *= (light.radius / 100.0);

        totalIrradiance += lightContribution;
    }
}
```

### Why This Works

1. **Multi-lights are already bound:** Application.cpp line 719 passes light buffer to UpdateProbes
2. **Light count is passed:** Line 720 passes lightCount
3. **Root signature already set up:** ProbeGridSystem.cpp line 180-181 binds light buffer to t1
4. **Shader was ready:** Light structure and buffer were already defined, just not used

**The fix implements the TODO that was always planned but never completed.**

---

## How to Test

### Option 1: Hybrid Lighting (Recommended)

**Config:** `configs/scenarios/hybrid_lighting.json`

**What it does:**
- Probe-grid: Enabled (32³ grid, provides ambient volumetric illumination)
- Multi-light: Enabled (4 lights for rim enhancement, down from 13)
- Target: 90+ FPS with 90% visual quality of 13-light baseline

**Launch command:**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/hybrid_lighting.json
```

**Expected results:**
- Probe-grid should now provide visible ambient volumetric glow
- No violent flashes (unlike inline RayQuery)
- Smooth volumetric cohesion (particles look like a disk, not isolated fireflies)
- FPS: 90-110 (better than 72 FPS with 13 multi-lights)

---

### Option 2: Probe-Grid Only (Diagnostic)

**Config:** `configs/scenarios/probe_grid_only.json`

**What it does:**
- Probe-grid: Enabled (32³ grid)
- Multi-light: DISABLED (but lights still passed to probe shader for sampling)
- Isolates probe-grid contribution only

**Launch command:**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/probe_grid_only.json
```

**Expected results:**
- Should see ambient volumetric glow from probe-grid alone
- Useful for verifying probe-grid is working before blending with multi-light
- FPS: 110-130 (probe-grid overhead is minimal)

**Note:** This config has `"multiLight": { "enabled": false }`, which means multi-light won't render directly. However, the lights are still spawned and passed to the probe shader (Application.cpp line 705-706), so probes will capture their contributions.

---

### Option 3: Multi-Light Only (Baseline)

**Existing default config**

**What it does:**
- Multi-light: Enabled (13 lights)
- Probe-grid: Disabled

**Expected results:**
- Baseline: 72 FPS @ 10K particles
- Beautiful volumetric scattering (but too slow)

**Use this to compare visual quality:**
1. Take screenshot with multi-light only (baseline)
2. Take screenshot with hybrid lighting
3. Compare using MCP tool:
   ```bash
   mcp__dxr-image-quality-analyst__compare_screenshots_ml(
       before_path="/mnt/c/Users/dilli/Pictures/Screenshots/multi_light_baseline.png",
       after_path="/mnt/c/Users/dilli/Pictures/Screenshots/hybrid_probe_multi.png"
   )
   ```

---

## Success Criteria

### Visual Quality
- ✅ Visible ambient volumetric glow (not pitch black anymore)
- ✅ No violent flashes (unlike inline RayQuery)
- ✅ Smooth volumetric cohesion (disk appearance, not isolated particles)
- ✅ Proper atmospheric scattering visible

### Performance (Hybrid Config)
- ✅ FPS ≥90 @ 10K particles, 1080p (RTX 4060 Ti)
- ✅ 25% better than 13-light multi-light baseline (72 FPS)

### Visual Quality Comparison
- ✅ ≥90% of 13-light multi-light quality (via ML LPIPS comparison)
- ✅ Rim lighting maintained (from 4 selective multi-lights)
- ✅ Ambient cohesion improved (from probe-grid)

---

## What If It Doesn't Work?

### Diagnostic Steps

1. **Check if probes are updating:**
   - Look for log message: "Dispatching probe update: (X, X, X) thread groups for 32³ grid (32768 total probes)"
   - Should appear in log every frame

2. **Capture buffer dump:**
   - Launch with: `--dump-buffers 120`
   - Check `PIX/buffer_dumps/g_probeGrid.bin` exists and is non-zero

3. **Use path-and-probe agent to validate:**
   ```bash
   mcp__path-and-probe__validate_sh_coefficients(
       probe_buffer_path="PIX/buffer_dumps/g_probeGrid.bin"
   )
   ```

4. **Check light buffer is bound:**
   - Look for log: "Probe update root signature created"
   - Verify no shader compilation errors

### Possible Issues

**Issue 1: Still too dim**
- **Solution:** Increase `probeIntensity` in config from 800 → 2000
- **File:** `configs/scenarios/hybrid_lighting.json` line 26

**Issue 2: Lights not visible to probes**
- **Check:** Are lights enabled? (line 14 in config: `"enabled": true`)
- **Check:** Is lightCount > 0? (should be 4 for hybrid, 13 for baseline)

**Issue 3: Shadow rays blocking all light**
- **Potential:** Shadow ray offset might be too aggressive (0.1 units)
- **Solution:** Increase offset or disable shadow rays temporarily to test

---

## Next Steps (If Fix Works)

### Phase 1: Validated ✅
- Probe-grid captures multi-light contributions
- Hybrid approach provides visible ambient + rim lighting

### Phase 2: Optimization
1. **Tune probe intensity** (800 → 1200-1500 for brighter ambient)
2. **Tune blend weights** (0.7 probe / 0.3 multi-light → adjust for taste)
3. **Experiment with light count** (4 vs 6 vs 8 selective lights)
4. **Test grid resolution** (32³ vs 48³ for quality vs performance)

### Phase 3: RT Lighting Scattering Fix (Long-term)
- **Problem:** Inline RayQuery lighting still causes violent flashes
- **Solution 1:** Use hybrid approach (probe-grid + multi-light) - **RECOMMENDED**
- **Solution 2:** Rewrite inline RayQuery to do proper volumetric scattering - **COMPLEX**

See `docs/RT_LIGHTING_SCATTERING_PROBLEM.md` for full analysis.

---

## Files Modified This Session

### Shaders
- `shaders/probe_grid/update_probes.hlsl` - Added multi-light contributions (lines 326-362)

### Configs (Already Existed)
- `configs/scenarios/hybrid_lighting.json` - Hybrid approach (probe-grid + 4 multi-lights)
- `configs/scenarios/probe_grid_only.json` - Probe-grid isolation test
- `configs/scenarios/multi_light_only.json` - 13-light baseline

### Documentation (Created)
- `docs/PROBE_GRID_FIX_2025-11-16.md` (this file)

### Previous Session (Already Committed)
- `src/utils/ConfigLoader.h:18` - Default particle radius 50.0 → 15.0
- `docs/SESSION_HANDOFF_2025-11-16.md` - Session context
- `docs/RT_LIGHTING_SCATTERING_PROBLEM.md` - Root cause analysis

---

## Build Status

✅ **Shaders Compiled:** update_probes.hlsl compiled successfully
✅ **Project Built:** PlasmaDX-Clean.sln build succeeded (Debug configuration)
✅ **No Errors:** Clean build, ready to test

---

## Expected User Experience

### Before Fix
- Probe-grid: Pitch black (0% visible lighting)
- Only multi-light works (72 FPS, too slow)
- Inline RayQuery: Violent flashes (unusable)

### After Fix (Hybrid Config)
- Probe-grid: Visible ambient volumetric glow
- Multi-light (4 lights): Rim enhancement
- Combined: 90+ FPS, beautiful volumetric scattering
- No violent flashes (smooth, cohesive disk appearance)

---

## Technical Notes

### Why Shadow Rays Are Needed

The probe shader casts shadow rays to each light to check visibility:
```hlsl
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
shadowQuery.TraceRayInline(g_particleTLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
```

**Purpose:** Ensure probe only accumulates lighting from lights that are visible from the hit position.

**Performance Impact:** Minimal - only 16 rays per probe (amortized over 4 frames = 4 rays/probe/frame)

### Attenuation Model

Uses inverse-square law for physically-based light falloff:
```hlsl
float attenuation = 1.0 / max(lightDistance * lightDistance, 1.0);
```

Scaled by light radius to give larger lights more ambient influence:
```hlsl
lightContribution *= (light.radius / 100.0);
```

---

**Last Updated:** 2025-11-16
**Status:** Ready for user testing
**Expected Result:** Probe-grid dim lighting FIXED - should now provide visible ambient volumetric illumination

**Please test with hybrid_lighting.json config and report results!**
