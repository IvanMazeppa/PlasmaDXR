# RT Lighting Scattering Problem - Root Cause Analysis

**Date:** 2025-11-16
**Status:** 🔴 CRITICAL - Core visual quality blocker
**Priority:** HIGHEST

---

## Problem Statement

The core visual quality issue with PlasmaDX is NOT the Gaussian particle structure or material diversity - it's that **the RT lighting system cannot scatter light volumetrically**.

### User Description (Direct Quote)

> "the inline RQ lighting causes violent flashes unless it gets turned down and can't scatter light based on the way it's implemented (each particle simply gets brighter when hit with a ray, this was the first RT lighting effect i got working but it's shortcomings have been a nightmare to get right)"

---

## Technical Root Cause

### Current Implementation (Broken)

**File:** `src/lighting/RTLightingSystem_RayQuery.cpp`

**What it does:**
```cpp
// Simplified pseudocode
for each particle:
    if ray_hits_particle(light, particle):
        particle.brightness += light.intensity
```

**Problems:**
1. ❌ Particles just get brighter - no volumetric scattering
2. ❌ Causes violent flashes when particles are directly lit
3. ❌ No Beer-Lambert law absorption
4. ❌ No Henyey-Greenstein phase function scattering
5. ❌ No atmospheric glow or volumetric cohesion
6. ❌ Particles look isolated, not like a continuous disk

**Result:** Unusable for production rendering

---

### What SHOULD Happen (Proper Volumetric Scattering)

```cpp
for each particle:
    float3 accumulated_radiance = 0.0;

    for each light:
        // Ray march through particle volume
        float3 ray_origin = particle.position;
        float3 ray_direction = normalize(light.position - particle.position);

        // Integrate along ray path
        float optical_depth = 0.0;
        float3 scattered_radiance = 0.0;

        for (int step = 0; step < NUM_STEPS; step++) {
            float3 sample_pos = ray_origin + ray_direction * (step * step_size);

            // Beer-Lambert law for absorption
            float density = evaluate_density(sample_pos, particle);
            optical_depth += density * step_size;
            float transmittance = exp(-optical_depth);

            // Henyey-Greenstein phase function for scattering
            float phase = henyey_greenstein(cos_theta, g);
            scattered_radiance += light.color * light.intensity * phase * transmittance * density * step_size;
        }

        accumulated_radiance += scattered_radiance;
    }

    particle.radiance = accumulated_radiance;
```

**This provides:**
- ✅ Smooth volumetric scattering
- ✅ Atmospheric glow
- ✅ Proper light absorption (Beer-Lambert)
- ✅ Directional scattering (phase function)
- ✅ No violent flashes
- ✅ Volumetric cohesion

---

## What DOES Work

### Multi-Light System ✅

**File:** Multi-light rendering path (exact file TBD)

**Performance:** 72 FPS @ 10K particles, 13 lights, 1080p

**Quality:** Beautiful volumetric scattering, proper atmospheric glow, rim lighting

**Why it works:**
- Implements proper volumetric scattering
- Uses Henyey-Greenstein phase function correctly
- Creates cohesive volumetric disk appearance

**Why we can't just use it:**
- Only 72 FPS (target: 90+ FPS)
- Too expensive for 16+ lights
- Need better performance without sacrificing quality

---

## Potential Solutions

### Solution 1: Fix Probe-Grid (RECOMMENDED)

**Status:** Partially implemented, currently extremely dim

**What it is:**
- 48³ grid (110,592 probes) covering [-1500, +1500] world space
- Each probe stores spherical harmonics (SH) irradiance
- Provides ambient volumetric illumination via trilinear interpolation

**Why it might work:**
- Proper volumetric ambient lighting (if we fix dim issue)
- Good performance (amortized over many particles)
- No violent flashes (smooth interpolation)

**Current Issues:**
- ✅ Dispatch fixed (all 110,592 probes updating)
- ✅ Ray distance fixed (200 → 2000 units)
- ❌ **Still extremely dim** - unknown root cause

**Next Steps:**
1. Validate probe buffers are non-zero (not just dispatched)
2. Check if shader is writing correct irradiance values
3. Verify SH accumulation logic in `update_probes.hlsl`
4. Test with intensity boost (800 → 5000?)
5. Check attenuation formula (inverse-square too aggressive at distance?)

**Agent:** `path-and-probe` (6 diagnostic tools available)

---

### Solution 2: Hybrid Approach (BEST LONG-TERM)

**Concept:**
- Probe-grid for ambient volumetric illumination (base lighting)
- 4-6 selective multi-lights for rim lighting and hero highlights

**Benefits:**
- Best visual quality (combines both systems' strengths)
- Performance: Probe-grid (~5ms) + 6 lights (~50% of 13-light cost) = 90+ FPS target
- No violent flashes (probe-grid handles base, multi-light only for accents)
- Proper volumetric scattering (multi-light) + smooth ambient (probe-grid)

**Prerequisites:**
- Probe-grid must work (currently dim)
- Need to implement light blending (probe-grid + multi-light)
- Need to tune which lights are "hero" lights vs. ambient

**Implementation Complexity:** Medium (1-2 sessions after probe-grid fixed)

---

### Solution 3: Rewrite Inline RayQuery Scattering (NOT RECOMMENDED)

**What it would take:**
- Complete shader rewrite of `RTLightingSystem_RayQuery.cpp`
- Implement proper volumetric ray marching (see pseudocode above)
- Add Beer-Lambert absorption
- Add Henyey-Greenstein phase function
- Tune for performance (ray steps, sample count)

**Estimated Effort:** 2-3 sessions (complex shader work)

**Risk:**
- May still not match multi-light quality
- Performance unknown (could be worse than multi-light)
- High complexity, high chance of new bugs

**Recommendation:** Do NOT pursue unless probe-grid fails

---

## Recommended Action Plan

### Phase 1: Fix Probe-Grid Dim Lighting (IMMEDIATE)

**Goal:** Get probe-grid providing visible ambient illumination

**Time estimate:** 1-2 hours

**Steps:**
1. Capture fresh buffer dump with current state
2. Use `path-and-probe` agent to validate probe buffers:
   ```bash
   mcp__path-and-probe__validate_sh_coefficients(
     probe_buffer_path="PIX/buffer_dumps/g_probeGrid.bin"
   )
   ```
3. If buffers zeroed: Debug shader write issue
4. If buffers non-zero but dim: Analyze attenuation formula
5. Test with increased intensity (800 → 5000)
6. Verify irradiance accumulation in `update_probes.hlsl`

**Success Criteria:**
- Probe-grid provides visible ambient illumination
- No violent flashes
- Smooth volumetric glow visible in screenshot

---

### Phase 2: Test Hybrid Approach (After Phase 1)

**Goal:** Combine probe-grid + selective multi-lights

**Time estimate:** 1-2 hours

**Steps:**
1. Keep probe-grid enabled (now working from Phase 1)
2. Enable 4-6 multi-lights (instead of 13)
3. Position lights for rim lighting and hero highlights
4. Capture screenshot and compare to 13-light baseline
5. Measure FPS (target: 90+)

**Success Criteria:**
- Visual quality ≥90% of 13-light multi-light baseline
- FPS ≥90 (10% better than current 72 FPS)
- No violent flashes
- Beautiful volumetric scattering maintained

---

### Phase 3: Tune and Optimize (Ongoing)

**Goal:** Maximize visual quality while maintaining 90+ FPS

**Areas to tune:**
- Probe-grid intensity vs. multi-light intensity balance
- Which lights are "hero" lights (rim, highlight) vs. ambient
- Probe-grid resolution (48³ vs. 32³ vs. 64³)
- Light count (6 vs. 8 vs. 10 selective lights)

---

## Why This Matters (User Context)

**User Quote:**
> "i will say my mental health is bad, and this project is one of the few things keeping me going"

**Impact:**
- This project is therapeutically important to the user
- Visual quality is the core motivation (experimenting with raytracing)
- Needs to see achievable progress
- Current state ("far away from what I'm hoping for") is demoralizing

**What success looks like:**
- Beautiful volumetric scattering (like multi-light)
- 90+ FPS performance
- No violent flashes (unlike inline RayQuery)
- Atmospheric glow and cohesive disk appearance

**Why there's hope:**
- ✅ Problem is understood (not mysterious)
- ✅ Probe-grid is fixable (dispatch works, just debugging dim issue)
- ✅ Hybrid approach is viable path forward
- ✅ Multi-agent system providing real diagnostic value

---

## Related Documentation

- `docs/SESSION_HANDOFF_2025-11-16.md` - Complete session analysis
- `docs/AUTONOMOUS_MULTI_AGENT_RAG_GUIDE.md` - Multi-agent workflow
- `docs/MULTI_AGENT_ROADMAP.md` - Development priorities
- `agents/path-and-probe/README.md` - Probe-grid diagnostic tools

---

**Last Updated:** 2025-11-16
**Next Action:** Fix probe-grid dim lighting (Phase 1)
**Expected Timeline:** 1-2 hours to debug and fix
