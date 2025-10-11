# ReSTIR Phase 1 Debug Briefing
**Date**: 2025-10-11 17:32
**Status**: BLOCKING BUG - ReSTIR darkens scene and doesn't respond to RT controls

## Current State

### What Works
- ✅ ReSTIR toggle (F7) activates correctly (confirmed in logs)
- ✅ Particles are being found by ray queries (M=512-713 in PIX)
- ✅ Temporal weight control (CTRL+F7) works
- ✅ Physical emission controls work (particle temperature, phase function)
- ✅ Debug indicators show correct state (green when finding hits)

### What's Broken
- ❌ **Scene darkens immediately when ReSTIR enabled**
- ❌ **Colors shift to brown/muted hues**
- ❌ **weightSum always = 0** in reservoir (despite finding particles)
- ❌ **RT lighting strength controls (I/K) don't work when ReSTIR ON**
- ❌ **M decays rapidly from ~512 to 0** (then indicator goes orange/red)

## Key Evidence from PIX

### When ReSTIR First Enabled (Green Indicator)
```
Reservoir Entry:
{
  lightPos = { 15, 0, 999 },
  weightSum = 0,           // ❌ PROBLEM: Should be non-zero!
  M = 641,                 // ✅ Finding particles
  W = 0,                   // ❌ Zero because weightSum is 0
  particleIdx = 0,
  pad = 0
}
```

### After A Few Frames (Orange/Red Indicator)
```
Reservoir Entry:
{
  lightPos = { x, y, 999 },
  weightSum = 0,           // Still zero
  M = 0,                   // Decayed to zero
  W = 0,
  particleIdx = 0,
  pad = 0
}
```

## Code Changes Made (Didn't Fix Issue)

### 1. Fixed Ray Origin
**Problem**: Using near plane position instead of camera position
**Fix**: Changed from `ray.Origin` to `cameraPos`
**Result**: Particles ARE now being found (M > 0), but weightSum still 0

### 2. Fixed Attenuation Formula
**Problem**: Too harsh inverse-square for large accretion disk
**Fix**: Changed to `1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1)`
**Result**: No change - still darkens

### 3. Lowered Weight Threshold
**Problem**: Threshold too high (0.001)
**Fix**: Lowered to 0.00001
**Result**: No change

### 4. Matched Sampling and Rendering Attenuation
**Problem**: Different formulas in sampling vs rendering
**Fix**: Made both use the same gentle attenuation
**Result**: No change

## Critical Clue

The fact that **physical emission controls work** but **RT lighting controls don't** suggests:

```hlsl
// Line 623 in shader - This line is the issue
illumination += rtLight * rtLightingStrength;
```

When ReSTIR is ON:
- `rtLight` comes from the reservoir calculation (lines 603-614)
- If `weightSum = 0`, then `W = 0`, so `rtLight ≈ 0`
- Therefore changing `rtLightingStrength` has no effect (0 × anything = 0)

When ReSTIR is OFF:
- `rtLight` comes from pre-computed buffer (line 617)
- That buffer has real values, so controls work

## Root Cause Hypothesis

The problem is in `UpdateReservoir` or the weight calculation in `SampleLightParticles`:

```hlsl
// Lines 350-366 - Weight calculation
float3 emission = TemperatureToEmission(hitParticle.temperature);
float intensity = EmissionIntensity(hitParticle.temperature);
float dist = length(hitParticle.position - rayOrigin);
float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);
float weight = dot(emission * intensity * attenuation, float3(0.299, 0.587, 0.114));

if (weight > 0.00001) {
    float random = Hash(pixelIndex * numCandidates + i + frameIndex * 2000);
    UpdateReservoir(reservoir, hitParticle.position, hitParticleIdx, weight, random);
}
```

**Possible issues:**
1. `TemperatureToEmission()` or `EmissionIntensity()` returning zero/near-zero
2. `UpdateReservoir()` not properly accumulating `weightSum`
3. Hash function producing bad random values
4. Weight calculation fundamentally wrong for this scene scale

## Files to Investigate

1. **[particle_gaussian_raytrace.hlsl](shaders/particles/particle_gaussian_raytrace.hlsl)**
   - Lines 275-383: `SampleLightParticles` function
   - Lines 195-220: `UpdateReservoir` function
   - Lines 603-614: Reservoir light usage in main loop

2. **[particle_common.hlsl](shaders/particles/particle_common.hlsl)**
   - `TemperatureToEmission()` function
   - `EmissionIntensity()` function

3. **PIX Captures**
   - Inspect actual particle temperatures
   - Check if ANY particles have high enough temp to emit light

## Agent Tasks

### Agent 1: Investigate Weight Calculation
- Read `TemperatureToEmission()` and `EmissionIntensity()`
- Calculate expected weight for typical particle (temp ~1000-10000K)
- Determine if weights should be non-zero for the scene

### Agent 2: Investigate UpdateReservoir
- Examine the `UpdateReservoir` function implementation
- Verify it's correctly accumulating `weightSum`
- Check the weighted reservoir sampling algorithm

### Agent 3: Create Diagnostic Shader
- Add debug output to write calculated weights to a buffer
- Log actual temperature values of hit particles
- Verify the weight > 0.00001 condition is being met

### Agent 4: Review Original Design
- Check the RESTIR_RAYQUERY_DIAGNOSIS.md document
- Verify we implemented the fixes correctly
- Identify any steps we skipped

## Next Steps

**Option 1**: Deploy agents to systematically investigate
**Option 2**: Add aggressive debug output to understand why weightSum = 0
**Option 3**: Temporarily bypass weight threshold to force all hits to update reservoir

## Log Files
- Current session: `logs/PlasmaDX-Clean_20251011_173125.log`
- Shows ReSTIR toggling, but no weight/particle data

## Expected Behavior When Fixed

When ReSTIR is enabled (F7):
1. Scene brightness should MATCH ReSTIR OFF state
2. Colors should remain vibrant (no brown shift)
3. I/K controls should adjust RT lighting strength
4. PIX should show `weightSum > 0` and `W > 0`
5. M should stay stable (not decay to 0)
6. Lighting should converge faster (10-60× improvement)
