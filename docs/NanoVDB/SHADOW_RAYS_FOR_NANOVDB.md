# Shadow Rays for NanoVDB Volumetric Self-Shadowing

## Current State

**Working**: Shadow rays in `particle_gaussian_raytrace.hlsl` for Gaussian particles
**Not Working**: Shadow rays have no effect on NanoVDB volumes (rays pass through without attenuation)

**Why**: The shadow ray system traces against the particle TLAS (acceleration structure), but NanoVDB volumes are rendered separately via ray marching - they're not in the TLAS.

---

## How Shadow Rays Work (Current Gaussian Implementation)

From `particle_gaussian_raytrace.hlsl`:

```hlsl
// Shadow ray setup
float shadowFactor = 1.0;  // Fully lit by default

// For each light, trace shadow ray toward light
RayDesc shadowRay;
shadowRay.Origin = hitPoint;
shadowRay.Direction = normalize(lightPos - hitPoint);
shadowRay.TMin = 0.001;
shadowRay.TMax = distance(hitPoint, lightPos);

// Query TLAS for occluders
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
shadowQuery.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, shadowRay);
shadowQuery.Proceed();

if (shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    shadowFactor = 0.0;  // Fully shadowed
}

// Apply shadow to lighting
float3 lighting = lightColor * lightIntensity * shadowFactor;
```

This works because Gaussian particles are represented as procedural primitives in the TLAS.

---

## Option A: Volumetric Self-Shadowing (In-Shader)

The most physically correct approach: during ray marching, accumulate optical depth toward each light.

### Implementation in `nanovdb_raymarch.hlsl`:

```hlsl
// During primary ray march, at each sample point:
float3 ComputeLightingWithShadows(float3 samplePos, float density) {
    float3 totalLight = float3(0, 0, 0);

    for (uint i = 0; i < lightCount; i++) {
        Light light = g_lights[i];
        float3 lightDir = normalize(light.position - samplePos);
        float lightDist = distance(light.position, samplePos);

        // March toward light, accumulating optical depth
        float shadowOpticalDepth = 0.0;
        float shadowStepSize = stepSize * 2.0;  // Coarser for performance
        uint maxShadowSteps = 32;  // Limit for performance

        float3 shadowPos = samplePos;
        for (uint s = 0; s < maxShadowSteps; s++) {
            shadowPos += lightDir * shadowStepSize;

            // Check if still inside volume bounds
            if (!IsInsideAABB(shadowPos, gridWorldMin, gridWorldMax)) break;
            if (distance(shadowPos, samplePos) > lightDist) break;

            // Sample density at shadow position
            float shadowDensity = SampleNanoVDBDensity(shadowPos);
            shadowOpticalDepth += shadowDensity * absorptionCoeff * shadowStepSize;
        }

        // Beer-Lambert transmittance toward light
        float shadowTransmittance = exp(-shadowOpticalDepth);

        // Accumulate light contribution
        float falloff = 1.0 / (1.0 + lightDist * lightDist * 0.0001);
        totalLight += light.color * light.intensity * falloff * shadowTransmittance;
    }

    return totalLight;
}
```

### Pros:
- Physically correct volumetric self-shadowing
- Soft shadows naturally emerge from density falloff
- No TLAS changes required

### Cons:
- **Performance**: Shadow march per sample per light (expensive!)
- With 16 lights and 64 primary steps: 16 * 64 * 32 = 32,768 shadow samples/pixel
- Need aggressive optimization (temporal accumulation, fewer shadow steps, skip low-density samples)

---

## Option B: Add NanoVDB AABB to TLAS

Add the NanoVDB bounding box as a procedural primitive to the TLAS, then existing shadow rays will test against it.

### C++ Changes (RTLightingSystem or NanoVDBSystem):

```cpp
// Create AABB instance for NanoVDB bounds
D3D12_RAYTRACING_AABB nanoVDBAABB;
nanoVDBAABB.MinX = m_gridWorldMin.x;
nanoVDBAABB.MinY = m_gridWorldMin.y;
nanoVDBAABB.MinZ = m_gridWorldMin.z;
nanoVDBAABB.MaxX = m_gridWorldMax.x;
nanoVDBAABB.MaxY = m_gridWorldMax.y;
nanoVDBAABB.MaxZ = m_gridWorldMax.z;

// Add to BLAS as procedural primitive
// Then include in TLAS with unique instance ID (e.g., 0xFFFFFFFF)
```

### Shader Changes:

```hlsl
// In Gaussian or shadow shader, check instance ID
if (shadowQuery.CommittedInstanceID() == 0xFFFFFFFF) {
    // Hit NanoVDB AABB - need to estimate attenuation
    // This is where it gets tricky: TLAS only gives hit/miss, not density
    shadowFactor *= 0.3;  // Approximate attenuation (not physically correct)
}
```

### Pros:
- Reuses existing shadow ray infrastructure
- Single ray test per light (fast)

### Cons:
- Only gives binary hit/miss for entire volume (not density-based)
- Approximate attenuation (not physically correct)
- Need to coordinate TLAS between particle system and NanoVDB system

---

## Option C: Hybrid Approach (Recommended)

Use TLAS for distant/secondary shadows, in-shader marching for hero volumes.

### Implementation Strategy:

1. **For distant lights or background volumes**:
   - Use TLAS AABB test with fixed attenuation factor
   - Fast, good enough for non-hero elements

2. **For hero volume (single NanoVDB asset)**:
   - Use in-shader shadow marching
   - Limit to 2-4 key lights only
   - Temporal accumulation to amortize cost

3. **LOD shadow quality**:
   ```hlsl
   if (distanceToCamera < 500.0) {
       // Full shadow marching for nearby volumes
       shadowTransmittance = MarchShadowRay(samplePos, lightDir, 32);
   } else {
       // Approximate for distant volumes
       shadowTransmittance = EstimateAttenuation(samplePos, lightDir);
   }
   ```

---

## Performance Estimates (RTX 4060 Ti @ 1080p)

| Method | Shadow Rays/Pixel | Est. Cost | Quality |
|--------|-------------------|-----------|---------|
| Current (no NanoVDB shadows) | 0 | 0 ms | N/A |
| Option A (full in-shader) | 16 lights * 32 steps | +8-12 ms | Excellent |
| Option A (4 lights, 16 steps) | 4 * 16 = 64 | +1-2 ms | Good |
| Option B (TLAS only) | 16 | +0.5 ms | Approximate |
| Option C (hybrid) | 4 + 12 | +1.5 ms | Good |

**Current baseline**: ~100-120 FPS with shadows disabled on NanoVDB
**Target**: Maintain 60+ FPS with volumetric self-shadowing

---

## Implementation Plan

### Phase 1: Single-Light Shadow Marching (Quick Win)
1. Add `ComputeShadowAttenuation()` to `nanovdb_raymarch.hlsl`
2. Use only the brightest/closest light
3. Limit to 16 shadow steps
4. Add ImGui toggle: "NanoVDB Self-Shadows"

### Phase 2: Multi-Light with Temporal Accumulation
1. Rotate through lights across frames (light 0 on frame 0, light 1 on frame 1, etc.)
2. Blend with previous frame's shadow result
3. Converges to full multi-light shadow over ~16 frames

### Phase 3: TLAS Integration (Optional)
1. Add NanoVDB AABB to particle TLAS
2. Use for binary occlusion test in Gaussian renderer
3. Particles casting shadows on volumes and vice versa

---

## Quick Start: Single-Light Shadow (Phase 1)

Add to `nanovdb_raymarch.hlsl` in the ray march loop:

```hlsl
// After sampling density at current step
if (density > 0.01 && lightCount > 0) {
    // Only shadow from light 0 for now
    Light mainLight = g_lights[0];
    float3 toLight = normalize(mainLight.position - worldPos);

    // Quick shadow march (16 steps)
    float shadowOpticalDepth = 0.0;
    float3 sp = worldPos;
    for (uint s = 0; s < 16; s++) {
        sp += toLight * stepSize * 2.0;
        if (!IsInsideAABB(sp)) break;
        shadowOpticalDepth += SampleNanoVDBDensity(sp) * absorptionCoeff * stepSize * 2.0;
    }

    float shadowTrans = exp(-shadowOpticalDepth);
    inScatter *= shadowTrans;  // Attenuate in-scattering by shadow
}
```

---

## References

- "Production Volume Rendering" - SIGGRAPH 2017 Course
- "A Survey on Participating Media Rendering Techniques" - CGF 2020
- PlasmaDX Gaussian shadows: `shaders/particles/particle_gaussian_raytrace.hlsl`
- NanoVDB ray marching: `shaders/volumetric/nanovdb_raymarch.hlsl`
