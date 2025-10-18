# DXR Volumetric Particle Rendering - Implementation Quick Start Guide

## Current System Status
- **Working**: DXR 1.1 RayQuery system with 12.5M rays/frame
- **Working**: BLAS/TLAS for 20,000 particle AABBs
- **Problem**: Flat/2D appearance, brown monotone colors, no volumetric shading

## Recommended Implementation Order

### Phase 1: Shadow Rays (8-16 hours) [IMMEDIATE FIX]
**File**: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/agent/AdvancedTechniqueWebSearches/shadowing/volumetric_particles/DXR_Shadow_Rays_Implementation.md`

**Quick Implementation**:
```hlsl
// Add to your existing particle shader
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;

float ComputeShadow(float3 pos, float3 lightPos, float3 normal)
{
    float3 toLight = normalize(lightPos - pos);
    RayDesc ray;
    ray.Origin = pos + normal * 0.001f;
    ray.Direction = toLight;
    ray.TMin = 0.001f;
    ray.TMax = length(lightPos - pos);

    shadowQuery.TraceRayInline(g_AccelStruct, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    shadowQuery.Proceed();

    return shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.3f : 1.0f;
}
```

**Expected Result**: Immediate depth perception, particles will cast shadows on each other

### Phase 2: Volumetric Scattering (24-32 hours) [CORE VOLUMETRIC FIX]
**File**: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/agent/AdvancedTechniqueWebSearches/ray_tracing/volumetric_particles/Volumetric_Scattering_Illumination.md`

**Key Components**:
1. Beer-Lambert absorption: `exp(-density * distance)`
2. Henyey-Greenstein phase function for anisotropic scattering
3. Ray marching through particle volume with density accumulation

**Minimal Implementation**:
```hlsl
float3 VolumetricLighting(float3 pos, float3 viewDir)
{
    float density = SampleParticleDensity(pos);
    float3 accumulated = 0;

    // For each light
    float3 lightDir = normalize(lightPos - pos);
    float transmittance = exp(-density * distance * 0.5f);
    float phase = (1 - g*g) / (4*PI * pow(1 + g*g - 2*g*cosTheta, 1.5));

    accumulated += lightColor * transmittance * phase * density;
    return accumulated;
}
```

**Expected Result**: Proper 3D volumetric appearance with light scattering through particles

### Phase 3: 3D Gaussian Ray Tracing (32-40 hours) [OPTIMIZATION]
**File**: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/agent/AdvancedTechniqueWebSearches/ray_tracing/volumetric_particles/3D_Gaussian_Ray_Tracing.md`

**Why This Helps**: Your existing AABB TLAS is perfect for this. Properly evaluates 3D Gaussian particles with anisotropic shapes.

**Key Integration**:
```hlsl
// Evaluate 3D Gaussian at intersection
float EvaluateGaussian(float3 localPos, float3x3 invCovariance)
{
    float exponent = -0.5f * dot(localPos, mul(invCovariance, localPos));
    return exp(exponent);
}
```

### Phase 4: Inter-Particle Bouncing (48-56 hours) [ADVANCED]
**File**: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/agent/AdvancedTechniqueWebSearches/ai_lighting/Secondary_Ray_Inter_Particle_Bouncing.md`

**Only After Phases 1-3 Work**: This adds global illumination between particles

### Phase 5: ReSTIR Integration (40-48 hours) [PERFORMANCE]
**File**: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/agent/AdvancedTechniqueWebSearches/efficiency_optimizations/ReSTIR_Particle_Integration.md`

**When Needed**: If performance is poor after implementing phases 1-3

## Critical Fix for Brown Monotone Issue

The brown monotone problem is likely caused by:
1. **Missing shadow rays** - causing all particles to receive full light
2. **No volumetric absorption** - light passes through without attenuation
3. **Incorrect alpha blending** - particles not sorted/blended properly

**Immediate Debug Test**:
```hlsl
// Add this to verify your RT is actually changing colors
float3 debugColor = float3(
    shadowFactor,           // Red = shadow
    density * 10.0f,        // Green = density
    dot(normal, lightDir)   // Blue = lighting angle
);
```

## Performance Budget Allocation

With 12.5M rays/frame budget:
- **Primary rays**: 60% (7.5M) - existing
- **Shadow rays**: 25% (3.1M) - new
- **Secondary bounces**: 15% (1.9M) - optional

## Common Integration Points

### 1. Modify Your Ray Generation Shader
```hlsl
[shader("raygeneration")]
void ParticleRayGen()
{
    // Your existing primary ray code

    // ADD: Shadow ray after primary hit
    float shadow = ComputeShadow(hit.pos, lightPos, hit.normal);

    // ADD: Volumetric scattering
    float3 scatter = VolumetricLighting(hit.pos, -rayDir);

    // MODIFY: Final color computation
    finalColor = directLight * shadow + scatter;
}
```

### 2. Add to Your Constant Buffer
```hlsl
cbuffer VolumetricParams : register(b3)
{
    float g_Density;         // 0.5
    float g_Absorption;      // 0.3
    float g_Scattering;      // 0.7
    float g_Anisotropy;      // 0.8 (forward scattering)
}
```

### 3. Leverage Existing TLAS
Your 20,000 AABBs are already perfect - just add intersection shaders for volumetric evaluation.

## Validation Checklist

- [ ] Shadow rays reduce brightness on occluded particles
- [ ] Particles show depth-based color variation
- [ ] Light attenuates through particle volume
- [ ] Particles glow with inner scattering
- [ ] No more flat/2D appearance
- [ ] Colors vary based on density and lighting angle

## Emergency Fallback

If performance tanks, use hybrid approach:
```hlsl
if (distanceToCamera < 100.0f)
{
    // Full volumetric for close particles
    color = VolumetricScattering(...);
}
else
{
    // Simple shading for distant particles
    color = BasicLambert(...) * shadowFactor;
}
```

## Contact Points for Issues

- **Shadow rays not working**: Check RAY_FLAG settings, verify TLAS is built correctly
- **Still flat appearance**: Ensure normal vectors are correct, check phase function
- **Performance issues**: Reduce march steps, use temporal accumulation
- **Color issues**: Verify Beer-Lambert coefficients, check scattering albedo

Start with Phase 1 (Shadow Rays) - you should see immediate improvement within 2-3 hours of implementation.