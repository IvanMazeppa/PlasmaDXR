# Screen-Space Shadow System - Critical Issues Diagnosis

**Date:** 2025-11-07
**Branch:** 0.14.2
**Status:** 🚨 MULTIPLE CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

Three critical issues discovered during testing:

1. ✅ **IDENTIFIED:** Particle count crash (2045-4991 range) - NVIDIA BVH builder bug
2. ✅ **IDENTIFIED:** All particles green in debug mode - depth buffer not detecting occlusions
3. ❌ **NEW:** Inline RQ lighting broken - needs investigation

---

## Issue 1: Particle Count Crash (2045-4991)

### Symptoms

- Crashes with >2044 particles (TDR, 5-second pause)
- Same crash pattern as original probe grid bug
- Works fine with ≤2044 particles

### Root Cause

**THIS IS NOT A DEPTH PRE-PASS BUG!**

From `PROBE_GRID_PHASE_2_SUCCESS.md`, the crash is an **NVIDIA BVH builder threshold bug**:

| Particle Range | Status | Notes |
|----------------|--------|-------|
| ≤2044 | ✅ Works | Single BLAS, no dual AS needed |
| 2045-4991 | ❌ **CRASHES** | NVIDIA internal BVH threshold bug |
| ≥4992 | ✅ Works | Dual AS with proper Direct RT BLAS size |

**Critical detail:** The difference between 4991 (crash) and 4992 (works) is **1 particle** (2947 vs 2948 in Direct RT BLAS). This confirms an internal NVIDIA driver threshold.

### Attempted Fix (WRONG)

I initially tried skipping particles 0-2043 in `depth_prepass.hlsl`:
```hlsl
// WRONG FIX (reverted):
if (particleIdx < 2044) {
    return;  // Skip probe grid particles
}
```

**Why this was wrong:**
- If you have 2500 particles, this skips 2044 particles
- Only 456 particles (2044-2499) rendered to depth buffer
- **This is why everything was green** - too few particles for occlusion!

### Correct Solution

**AVOID THE 2045-4991 PARTICLE RANGE ENTIRELY**

1. **For development/testing:** Use exactly 2000 particles (safe, below limit)
2. **For production:** Use ≥5000 particles (well above 4992 threshold)

**Example config:**
```json
{
    "particleCount": 2000,  // Testing (below 2045)
    // OR
    "particleCount": 10000  // Production (above 4992)
}
```

---

## Issue 2: All Particles Green (No Shadows)

### Symptoms

- Debug visualization shows all green particles
- `shadowTerm = 1.0` (fully lit, no occlusion)
- Only visible with multi-light system
- No effect with probe grid or inline RQ

### Root Causes (Multiple)

#### Cause 2A: Particle Count in Forbidden Range

If testing with 2500-4000 particles:
1. App crashes before shadows can work
2. OR my incorrect "skip particles" fix made only 456 particles render to depth buffer
3. With so few particles, no occlusions detected → all green

**Solution:** Test with 2000 or 10000 particles, not 2500.

#### Cause 2B: Debug Visualization Not Integrated with Probe Grid

The debug visualization only works in the **multi-light loop**:

```hlsl
// particle_gaussian_raytrace.hlsl line 1127
if (debugScreenSpaceShadows != 0 && lightIdx == 0) {
    float3 debugColor = float3(1.0 - shadowTerm, shadowTerm, 0.0);
    totalLighting = debugColor * 50.0;  // Override lighting
    break;
}
```

This code is **inside the multi-light loop**, not the probe grid or inline RQ paths!

**Why you see nothing without multi-light:** Debug visualization doesn't execute in those code paths.

#### Cause 2C: Depth Buffer Not Populated

Possible reasons:
1. Depth pre-pass shader fails silently
2. Particles outside screen bounds
3. Projection matrix incorrect
4. Clear shader not running

**Diagnosis needed:** Parse `g_shadowDepth.bin` from PIX to verify depth values.

### Testing Plan

1. **Set particle count to 2000** (below 2045 threshold)
2. **Enable multi-light mode** (not probe grid)
3. **Add 1-2 lights close together** for occlusion testing
4. **Enable debug visualization**
5. **Look for red/yellow particles** (not just green)

If still all green with 2000 particles + multi-light:
- Depth buffer is NOT being populated
- Depth pre-pass shader has a bug

---

## Issue 3: Inline RQ Lighting Broken

### Symptoms

- "normal particle-to-particle inline RQ lighting has stopped working"
- Probe grid still functions
- Visual artifact issues persist

### Investigation Needed

I did not intentionally modify inline RQ lighting code. Possible causes:

1. **Constant buffer alignment issue:** Added `debugScreenSpaceShadows` constant, may have broken alignment
2. **Root signature mismatch:** 12-parameter root signature might conflict
3. **State transition missing:** Depth pre-pass UAV barrier might interfere

### Files to Check

1. **ParticleRenderer_Gaussian.cpp** - RenderConstants structure alignment
2. **particle_gaussian_raytrace.hlsl** - Cbuffer layout matches C++ struct
3. **Application.cpp** - Constant upload size matches struct size

### Debug Steps

1. Check constant buffer size:
```cpp
sizeof(RenderConstants) % 16 == 0  // Must be 16-byte aligned
```

2. Check root signature parameter count matches SetComputeRoot calls

3. Verify inline RQ code path still executes:
```hlsl
// Add logging to inline RQ section
if (useVolumetricRT != 0) {
    // This code path should execute for inline RQ
}
```

---

## Depth Buffer Analysis (PIX Capture)

### Files Captured

From PIX dump (2025-11-07 04:22):

1. **g_shadowDepth.bin** (4.9 MB) - Shadow depth buffer (Phase 2)
2. **g_depthBuffer.bin** (4.9 MB) - Unknown depth buffer (needs investigation)
3. **g_particleLighting.bin** (32 KB = 2000 particles)
4. **g_probeGrid.bin** (14 MB = 110,592 probes, 48³ grid)

### Probe Grid Status

From user report:
> "you can see how sparsely populated the g_probeGrid buffer is with nothing at all for irradiance_1 and irradiance_2, and almost nothing for irradiance_0."

This suggests probe grid is enabled but barely populated. Possible reasons:
1. Particles outside grid bounds (-1500 to +1500 range)
2. Rays per probe too low (should be 16)
3. Update interval too high (should be 4 frames)
4. Intensity scale too low (should be 800.0)

---

## Recommended Next Steps

### Priority 1: Fix Particle Count Issue

**User action:**
```bash
# Edit config.json or command line
"particleCount": 2000  // Safe testing value (below 2045)
```

Retest:
1. Verify no TDR crash
2. Check if debug visualization shows red/yellow particles
3. Verify inline RQ lighting works again

### Priority 2: Debug Depth Buffer Population

If still all green with 2000 particles:

**Parse depth buffer:**
```python
# Use parsing script from SHADOW_IMPLEMENTATION_STATUS.md
python3 parse_depth_buffer.py g_shadowDepth.bin

# Expected output if working:
# - Valid depth pixels: 5-20% of screen
# - Depth range: 0.1-0.9
# - Far plane pixels (1.0): 80-95%

# If NOT working:
# - Valid depth pixels: 0%
# - All pixels = 1.0 (far plane)
```

### Priority 3: Fix Inline RQ Lighting

**Check constant buffer alignment:**
```cpp
// In ParticleRenderer_Gaussian.h
static_assert(sizeof(RenderConstants) % 16 == 0, "RenderConstants must be 16-byte aligned");
```

**Verify upload size:**
```cpp
// In Application.cpp, check SetComputeRoot32BitConstants call:
cmdList->SetComputeRoot32BitConstants(
    0,
    sizeof(RenderConstants) / 4,  // MUST match actual struct size!
    &gaussianConstants,
    0
);
```

---

## Files Modified (This Session)

### Reverted Changes

1. **shaders/shadows/depth_prepass.hlsl** - Removed incorrect "skip probe particles" fix

### Still Active

2. **shaders/particles/particle_gaussian_raytrace.hlsl** - Debug visualization (lines 92-93, 1126-1133)
3. **src/particles/ParticleRenderer_Gaussian.h** - `debugScreenSpaceShadows` constant (line 125)
4. **src/core/Application.h** - Debug toggle member (line 279)
5. **src/core/Application.cpp** - Debug constant upload (line 908) and ImGui (lines 2876-2887)

---

## Critical Questions for User

1. **What particle count are you testing with?** (Probably 2500-4000, which is in forbidden range)
2. **Is multi-light mode enabled?** (Debug visualization only works there)
3. **Did inline RQ work BEFORE this session?** (To confirm I broke it vs pre-existing issue)
4. **Can you test with exactly 2000 particles?** (Below 2045 threshold)
5. **Can you test with exactly 10000 particles?** (Above 4992 threshold)

---

## Expected Behavior After Fixes

### With 2000 Particles + Multi-Light

- ✅ No TDR crash
- ✅ Debug visualization shows green (lit) and red (shadowed) particles
- ✅ Inline RQ lighting works
- ✅ Probe grid works
- ✅ Screen-space shadows visible

### With 10000 Particles + Multi-Light

- ✅ No TDR crash (dual AS kicks in properly)
- ✅ Debug visualization works
- ✅ All lighting modes functional
- ✅ Production-ready performance

---

## Conclusion

The core issue is **testing in the NVIDIA BVH builder forbidden range (2045-4991 particles)**. This causes crashes and makes debugging impossible.

My attempted fix (skipping particles) was wrong and made shadows worse.

**Solution:** Use 2000 or 10000 particles for testing, NEVER 2500-4000.

The inline RQ issue needs separate investigation - likely a constant buffer alignment problem from adding the debug constant.
