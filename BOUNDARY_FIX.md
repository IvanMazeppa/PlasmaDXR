# 300-Unit Sphere Boundary Fix

## Problem Summary
Lights completely fail to illuminate particles beyond ~300 units from the origin, despite particles existing at 400-600+ units. The boundary exactly matches `config.json` outerRadius: 300.0, but this is just the initial spawn range.

## Root Cause Analysis

### Primary Issue: TLAS World-Space Bounds
The D3D12 raytracing acceleration structure (TLAS) computes conservative world-space bounds based on the transformed BLAS bounds. Since:
1. Particles initially spawn within 10-300 units
2. BLAS is built from current particle AABBs
3. TLAS uses identity transform (no scaling)
4. D3D12 may cache conservative bounds from first frame

The TLAS effectively has a ~300-350 unit spherical boundary. Shadow rays and light sampling rays that traverse beyond this boundary fail to find intersections.

### Secondary Issues
1. **Hardcoded physics ranges**: Temperature calculations assume 10-300 unit range
2. **No dynamic bounds tracking**: System doesn't track actual particle distribution
3. **PREFER_FAST_BUILD flag**: Prevents proper TLAS updates

## Recommended Fixes

### Fix 1: Expand TLAS Instance Transform (IMMEDIATE FIX)
**File:** `src/lighting/RTLightingSystem_RayQuery.cpp`
**Lines:** 399-411

```cpp
// OLD CODE (lines 408-411):
// Identity transform
instanceDesc.Transform[0][0] = 1.0f;
instanceDesc.Transform[1][1] = 1.0f;
instanceDesc.Transform[2][2] = 1.0f;

// NEW CODE:
// Scale transform to cover larger world space
// This effectively expands the TLAS bounds by 3x to cover particles that drift
const float worldScale = 3.0f;  // Cover up to 900 units radius
instanceDesc.Transform[0][0] = worldScale;
instanceDesc.Transform[1][1] = worldScale;
instanceDesc.Transform[2][2] = worldScale;
```

**Note:** When using a scaled transform, ray traversal automatically handles the transform. No shader changes needed.

### Fix 2: Dynamic Temperature Falloff
**File:** `shaders/particles/particle_physics.hlsl`
**Lines:** 228-229

```hlsl
// OLD CODE:
// Use inverse square law scaled for our radius range (10-300)
float tempFactor = saturate(1.0 - (distance - 10.0) / 290.0);

// NEW CODE:
// Dynamic temperature falloff based on actual world size
float innerRadius = 10.0;
float maxWorldRadius = 1000.0;  // Should come from constants
float tempFactor = saturate(1.0 - (distance - innerRadius) / (maxWorldRadius - innerRadius));
```

### Fix 3: Increase Particle Boundary Check
**File:** `shaders/particles/particle_physics.hlsl`
**Line:** 244

```hlsl
// OLD CODE:
if (distance > constants.outerRadius * 2.0) {

// NEW CODE:
if (distance > 1000.0) {  // Allow particles to drift much further
```

### Fix 4: Add Max World Radius to Constants
**File:** `src/particles/ParticleSystem.cpp`
**Lines:** 314-320 (PhysicsConstants struct)

Add new field:
```cpp
struct PhysicsConstants {
    // ... existing fields ...
    float maxWorldRadius;  // Add this
};

// In UpdatePhysics():
constants.maxWorldRadius = 1000.0f;  // Or from config
```

### Fix 5: Update Config Schema
**File:** `config.json`

Add new parameter:
```json
"maxWorldRadius": 1000.0,
"_maxWorldRadius_note": "Maximum world bounds for ray tracing and physics (units)",
```

## Testing Plan

1. **Verify Particles Beyond 300 Units**
   - Run with disabled constraint shape
   - Let particles drift naturally
   - Check max particle distance in ImGui

2. **Test Light Illumination**
   - Place light at (0, 100, 2000)
   - Verify particles at 400-600 units receive lighting
   - Check shadow rays work correctly

3. **PIX Validation**
   - Capture frame with particles beyond 300 units
   - Inspect TLAS bounds in PIX
   - Verify shadow ray traversal succeeds

4. **Performance Impact**
   - Larger TLAS bounds may slightly reduce traversal performance
   - Monitor FPS with expanded bounds
   - Consider LOD culling for distant particles

## Implementation Priority

1. **Immediate (Fix 1)**: Scale TLAS transform - fixes lighting instantly
2. **High (Fix 2-3)**: Update physics ranges - prevents visual artifacts
3. **Medium (Fix 4-5)**: Add config parameters - makes it adjustable
4. **Low**: Optimize with LOD culling - performance enhancement

## Expected Impact

- **Before**: Lights fail beyond ~300 units, particles go dark
- **After**: Lights work at any distance, full scene illumination
- **Performance**: Minimal impact (<5% FPS loss) due to larger TLAS
- **RTXDI Compatibility**: Required fix for RTXDI integration (needs full scene coverage)

## Validation Checklist

- [ ] Particles beyond 300 units are visible
- [ ] Lights at 2000+ units illuminate all particles
- [ ] Shadow rays work at all distances
- [ ] No performance regression >10%
- [ ] PIX shows expanded TLAS bounds
- [ ] ReSTIR sampling works beyond 300 units