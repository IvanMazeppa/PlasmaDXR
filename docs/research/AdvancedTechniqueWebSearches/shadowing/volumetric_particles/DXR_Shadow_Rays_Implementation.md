# DXR Shadow Rays for Volumetric Particles

## Source
- Technology: DirectX Raytracing (DXR) 1.1 with RayQuery
- Date: October 2025
- Platform: DXR Tier 1.1 with Shader Model 6.5

## Summary
Shadow rays in DXR for volumetric particles enable self-shadowing and inter-particle occlusion by casting secondary rays from particle surfaces toward light sources. Using the RayQuery inline raytracing API, we can efficiently determine visibility without the overhead of dynamic shader scheduling, making it ideal for volumetric particle systems with thousands of elements.

The technique leverages hardware-accelerated ray-triangle intersection with proper BLAS/TLAS structures already built for particle AABBs, enabling real-time shadow computation for 20,000+ particles.

## Key Innovation
Using RayQuery's template parameters for optimized shadow determination combined with Beer-Lambert law for volumetric absorption creates physically accurate self-shadowing in particle clouds without requiring full path tracing.

## Implementation Details

### Algorithm
```hlsl
// Step 1: Initialize RayQuery with shadow-specific flags
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
         RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> shadowQuery;

// Step 2: For each shaded particle point
float3 ComputeParticleShadow(float3 particlePos, float3 lightPos, float3 normal)
{
    // Calculate shadow ray direction
    float3 toLight = normalize(lightPos - particlePos);
    float lightDistance = length(lightPos - particlePos);

    // Offset ray origin to avoid self-intersection
    float3 rayOrigin = particlePos + normal * 0.001f; // Small epsilon offset

    // Initialize ray query
    RayDesc shadowRay;
    shadowRay.Origin = rayOrigin;
    shadowRay.Direction = toLight;
    shadowRay.TMin = 0.001f;
    shadowRay.TMax = lightDistance;

    // Trace shadow ray
    shadowQuery.TraceRayInline(
        g_AccelStruct,      // TLAS
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xFF,               // Instance mask
        shadowRay
    );

    // Process query
    shadowQuery.Proceed();

    // Check for occlusion
    if (shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // In shadow - apply volumetric absorption
        float hitDistance = shadowQuery.CommittedRayT();
        float absorption = BeerLambert(g_AbsorptionCoeff, hitDistance);
        return float3(absorption, absorption, absorption);
    }

    return float3(1, 1, 1); // Fully lit
}

// Beer-Lambert absorption for volumetric particles
float BeerLambert(float absorptionCoeff, float distance)
{
    return exp(-absorptionCoeff * distance);
}
```

### Code Snippets

#### Multi-sample Shadow Rays for Soft Shadows
```hlsl
float3 ComputeSoftShadows(float3 pos, float3 lightPos, float lightRadius, uint samples)
{
    float3 shadowAccum = 0;

    for (uint i = 0; i < samples; i++)
    {
        // Jitter light position for area light sampling
        float2 xi = Hammersley2D(i, samples);
        float3 lightSample = lightPos + SampleDiskUniform(xi) * lightRadius;

        // Cast shadow ray to sampled light position
        shadowAccum += ComputeParticleShadow(pos, lightSample, normal);
    }

    return shadowAccum / float(samples);
}
```

#### Inter-Particle Shadowing with Distance Attenuation
```hlsl
float ComputeInterParticleShadow(float3 origin, float3 direction, float maxDistance)
{
    RayQuery<RAY_FLAG_NONE> query;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001f;
    ray.TMax = maxDistance;

    query.TraceRayInline(g_AccelStruct, 0, 0xFF, ray);

    float totalTransmittance = 1.0f;

    // Accumulate absorption through multiple particle hits
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            // Get particle density at hit point
            float density = GetParticleDensityAt(query.CandidatePrimitiveIndex());
            float thickness = EstimateParticleThickness(query.CandidateTriangleBarycentrics());

            // Apply Beer-Lambert absorption
            totalTransmittance *= BeerLambert(density * g_ExtinctionCoeff, thickness);

            // Early termination if too opaque
            if (totalTransmittance < 0.01f)
            {
                query.CommitProceduralPrimitiveHit(query.CandidateRayT());
                break;
            }
        }
    }

    return totalTransmittance;
}
```

### Data Structures
```hlsl
// Per-particle shadow data buffer
struct ParticleShadowData
{
    float3 shadowTransmittance;
    float  ambientOcclusion;
    uint   shadowRayCount;
    float3 accumulatedIndirect;
};

StructuredBuffer<ParticleShadowData> g_ParticleShadows : register(t10);

// Shadow ray configuration
cbuffer ShadowConfig : register(b2)
{
    uint   g_ShadowSamples;
    float  g_ShadowBias;
    float  g_MaxShadowDistance;
    float  g_AbsorptionCoeff;
    float  g_ExtinctionCoeff;
    float  g_ScatteringAlbedo;
};
```

### Pipeline Integration

1. **Build BLAS/TLAS** with particle AABBs (already done)
2. **Add shadow pass** after primary ray hit determination
3. **Store shadow results** in per-particle buffer
4. **Apply shadows** during final shading

```hlsl
// Integration in main particle shader
[shader("raygeneration")]
void ParticleRayGen()
{
    // ... existing ray generation code ...

    if (primaryHit.valid)
    {
        // Compute direct lighting with shadows
        float3 directLight = 0;
        for (uint i = 0; i < g_LightCount; i++)
        {
            float3 shadow = ComputeParticleShadow(
                primaryHit.position,
                g_Lights[i].position,
                primaryHit.normal
            );

            directLight += g_Lights[i].color * shadow *
                          ComputeBRDF(primaryHit.normal, toLight, toView);
        }

        // Store result
        g_Output[DispatchRaysIndex().xy] = float4(directLight, 1.0);
    }
}
```

## Performance Metrics
- **GPU Time**: ~0.5-2ms for 20,000 particles with single shadow ray per particle
- **Memory Usage**: 32 bytes per particle for shadow data
- **Quality Metrics**: Eliminates flat appearance, adds depth perception

## Hardware Requirements
- **Minimum GPU**: RTX 2060 / RX 6600 (DXR 1.1 support)
- **Optimal GPU**: RTX 4070 or better (higher RT core count)

## Implementation Complexity
- **Estimated Dev Time**: 8-16 hours for basic implementation
- **Risk Level**: Low (uses existing TLAS structure)
- **Dependencies**: DXR 1.1, Shader Model 6.5

## Related Techniques
- Beer-Lambert volumetric absorption
- Soft shadow sampling
- ReSTIR for importance sampling
- Temporal shadow denoising

## Notes for PlasmaDX Integration
- Leverage existing 12.5M rays/frame budget by allocating 20-30% for shadow rays
- Use particle AABBs already in TLAS for shadow intersection
- Consider temporal accumulation for soft shadows to amortize cost
- Implement LOD system: fewer shadow samples for distant particles
- Cache shadow results when particles are static