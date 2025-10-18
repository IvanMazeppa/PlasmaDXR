# Secondary Ray Inter-Particle Light Bouncing

## Source
- Technology: DXR 1.1 with Multi-bounce Path Tracing
- References: NVIDIA OptiX Global Illumination, DXR Tier 1.1 Specification
- Date: October 2025
- Conference/Journal: Real-time Ray Tracing Best Practices

## Summary
Inter-particle light bouncing simulates how light scatters between particles in a volumetric system, creating color bleeding, indirect illumination, and realistic light diffusion through the medium. This technique uses secondary rays to capture light transport between particles, essential for achieving the volumetric "glow" effect in dense particle clouds like plasma or fire.

The implementation leverages DXR 1.1's RayQuery to trace secondary rays from particle surfaces to neighboring particles, accumulating indirect illumination contributions without the overhead of full recursive ray tracing.

## Key Innovation
Hierarchical light propagation through particle neighborhoods using importance-sampled secondary rays, combined with irradiance caching to amortize the cost of expensive multi-bounce calculations across multiple frames.

## Implementation Details

### Algorithm
```hlsl
// Main inter-particle light bouncing algorithm
float3 ComputeInterParticleBounce(float3 particlePos, float3 normal, uint particleIdx)
{
    const uint MAX_BOUNCES = 3;
    const uint SAMPLES_PER_BOUNCE = 8;

    float3 indirectLight = 0;
    float3 throughput = 1.0;

    for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        float3 bounceContribution = 0;

        // Sample hemisphere around particle
        for (uint sample = 0; sample < SAMPLES_PER_BOUNCE; sample++)
        {
            // Importance sample cosine-weighted hemisphere
            float2 xi = Hammersley2D(sample, SAMPLES_PER_BOUNCE);
            float3 sampleDir = SampleCosineHemisphere(xi, normal);

            // Trace ray to find neighboring particles
            RayQuery<RAY_FLAG_NONE> query;

            RayDesc ray;
            ray.Origin = particlePos + normal * 0.01f; // Bias
            ray.Direction = sampleDir;
            ray.TMin = 0.01f;
            ray.TMax = g_BounceRadius; // Limit search radius

            query.TraceRayInline(g_AccelStruct, 0, 0xFF, ray);

            float3 sampleRadiance = 0;
            float totalWeight = 0;

            // Process all hit particles along ray
            while (query.Proceed())
            {
                if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                {
                    uint hitParticleIdx = query.CandidatePrimitiveIndex();

                    // Skip self
                    if (hitParticleIdx == particleIdx)
                        continue;

                    float t = query.CandidateRayT();
                    float3 hitPos = ray.Origin + ray.Direction * t;

                    // Get particle properties
                    Particle hitParticle = g_Particles[hitParticleIdx];

                    // Evaluate particle radiance
                    float3 particleRadiance = EvaluateParticleRadiance(
                        hitParticle, hitPos, -sampleDir
                    );

                    // Distance attenuation
                    float attenuation = 1.0f / (1.0f + t * t);

                    // Gaussian weight based on particle overlap
                    float gaussianWeight = EvaluateGaussianOverlap(
                        particlePos, hitPos,
                        g_Particles[particleIdx].scale,
                        hitParticle.scale
                    );

                    sampleRadiance += particleRadiance * attenuation * gaussianWeight;
                    totalWeight += gaussianWeight;
                }
            }

            // Normalize and accumulate
            if (totalWeight > 0)
            {
                sampleRadiance /= totalWeight;
                bounceContribution += sampleRadiance;
            }
        }

        // Average samples and apply throughput
        bounceContribution /= float(SAMPLES_PER_BOUNCE);
        indirectLight += throughput * bounceContribution;

        // Russian roulette for path termination
        float continueProbability = min(0.95f, Luminance(throughput));
        if (Random() > continueProbability)
            break;

        throughput *= g_ScatteringAlbedo / continueProbability;
    }

    return indirectLight;
}
```

### Code Snippets

#### Particle Radiance Evaluation with BRDF
```hlsl
float3 EvaluateParticleRadiance(Particle p, float3 hitPos, float3 viewDir)
{
    // Direct illumination on particle
    float3 directLight = 0;
    for (uint i = 0; i < g_LightCount; i++)
    {
        float3 lightDir = normalize(g_Lights[i].position - hitPos);
        float visibility = TraceShadowRay(hitPos, lightDir);

        // Particle BRDF (simplified Lambertian + emission)
        float3 brdf = p.color / PI;
        directLight += g_Lights[i].color * brdf * visibility * max(0, dot(p.normal, lightDir));
    }

    // Add particle emission
    float3 emission = p.color * p.emissionStrength;

    // Add cached indirect (from previous bounce or frame)
    float3 cachedIndirect = SampleIrradianceCache(hitPos, p.normal);

    return directLight + emission + cachedIndirect * 0.5f;
}
```

#### Hierarchical Particle Neighbor Query
```hlsl
struct ParticleNeighborhood
{
    uint indices[32];
    float weights[32];
    uint count;
};

ParticleNeighborhood GetParticleNeighbors(float3 pos, float radius)
{
    ParticleNeighborhood neighbors;
    neighbors.count = 0;

    // Use spatial hash grid for acceleration
    uint3 gridCell = GetGridCell(pos);

    // Check 3x3x3 neighborhood
    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++)
    {
        uint3 cell = gridCell + uint3(dx, dy, dz);
        uint cellHash = HashCell(cell);

        // Iterate particles in cell
        uint particleCount = g_GridCellCounts[cellHash];
        uint startIdx = g_GridCellStarts[cellHash];

        for (uint i = 0; i < particleCount && neighbors.count < 32; i++)
        {
            uint pIdx = g_GridParticleIndices[startIdx + i];
            Particle p = g_Particles[pIdx];

            float dist = length(p.center - pos);
            if (dist < radius && dist > 0.001f)
            {
                neighbors.indices[neighbors.count] = pIdx;
                neighbors.weights[neighbors.count] = 1.0f / (1.0f + dist);
                neighbors.count++;
            }
        }
    }

    return neighbors;
}
```

#### Irradiance Caching for Temporal Coherence
```hlsl
// Irradiance cache structure
struct IrradianceProbe
{
    float3 position;
    float3 normal;
    float3 irradiance;
    float  radius;
    uint   age;
};

RWStructuredBuffer<IrradianceProbe> g_IrradianceCache : register(u2);

float3 SampleIrradianceCache(float3 pos, float3 normal)
{
    float3 accumulated = 0;
    float totalWeight = 0;

    const uint CACHE_SIZE = 256;
    for (uint i = 0; i < CACHE_SIZE; i++)
    {
        IrradianceProbe probe = g_IrradianceCache[i];

        // Check if probe is valid and nearby
        if (probe.age < 10) // Fresh enough
        {
            float distance = length(probe.position - pos);
            if (distance < probe.radius)
            {
                // Weight by distance and normal alignment
                float distWeight = exp(-distance / probe.radius);
                float normalWeight = saturate(dot(probe.normal, normal));
                float weight = distWeight * normalWeight;

                accumulated += probe.irradiance * weight;
                totalWeight += weight;
            }
        }
    }

    return totalWeight > 0 ? accumulated / totalWeight : 0;
}

void UpdateIrradianceCache(float3 pos, float3 normal, float3 irradiance)
{
    // Find oldest or invalid probe to replace
    uint replaceIdx = 0;
    uint maxAge = 0;

    for (uint i = 0; i < 256; i++)
    {
        if (g_IrradianceCache[i].age > maxAge)
        {
            maxAge = g_IrradianceCache[i].age;
            replaceIdx = i;
        }
    }

    // Update probe
    IrradianceProbe newProbe;
    newProbe.position = pos;
    newProbe.normal = normal;
    newProbe.irradiance = irradiance;
    newProbe.radius = g_ProbeRadius;
    newProbe.age = 0;

    g_IrradianceCache[replaceIdx] = newProbe;
}
```

#### Photon Mapping-Inspired Gathering
```hlsl
float3 GatherPhotons(float3 pos, float3 normal, float gatherRadius)
{
    RayQuery<RAY_FLAG_NONE> query;

    // Cast gathering rays in hemisphere
    const uint GATHER_SAMPLES = 16;
    float3 gathered = 0;

    for (uint i = 0; i < GATHER_SAMPLES; i++)
    {
        float2 xi = Hammersley2D(i, GATHER_SAMPLES);
        float3 dir = SampleUniformHemisphere(xi, normal);

        RayDesc ray;
        ray.Origin = pos + normal * 0.01f;
        ray.Direction = dir;
        ray.TMin = 0;
        ray.TMax = gatherRadius;

        query.TraceRayInline(g_AccelStruct, 0, 0xFF, ray);

        // Gather energy from all intersected particles
        while (query.Proceed())
        {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint pIdx = query.CandidatePrimitiveIndex();
                float t = query.CandidateRayT();

                // Kernel-based density estimation
                float kernel = 1.0f - (t / gatherRadius);
                kernel = kernel * kernel; // Quadratic falloff

                gathered += g_Particles[pIdx].color *
                           g_Particles[pIdx].emissionStrength * kernel;
            }
        }
    }

    return gathered / (PI * gatherRadius * gatherRadius);
}
```

### Data Structures
```hlsl
// Inter-particle bounce configuration
cbuffer BounceConfig : register(b5)
{
    float g_BounceRadius;         // Maximum bounce search distance
    uint  g_MaxBounces;           // Number of bounce iterations
    uint  g_SamplesPerBounce;     // Samples per bounce
    float g_ScatteringAlbedo;     // Energy conservation factor
    float g_RussianRouletteThreshold;
    float g_ProbeRadius;          // Irradiance cache probe radius
    uint  g_FrameCount;           // For temporal accumulation
};

// Spatial acceleration structure
StructuredBuffer<uint> g_GridCellStarts : register(t10);
StructuredBuffer<uint> g_GridCellCounts : register(t11);
StructuredBuffer<uint> g_GridParticleIndices : register(t12);
```

### Pipeline Integration
```hlsl
[shader("raygeneration")]
void InterParticleBounceRayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    Ray ray = GenerateCameraRay(pixel);

    // Find primary particle hit
    RayHit hit = TracePrimaryRay(ray);

    if (hit.valid && hit.isParticle)
    {
        // Compute direct lighting
        float3 direct = ComputeDirectLighting(hit.position, hit.normal);

        // Compute inter-particle bounces
        float3 indirect = ComputeInterParticleBounce(
            hit.position,
            hit.normal,
            hit.primitiveIndex
        );

        // Update irradiance cache
        UpdateIrradianceCache(hit.position, hit.normal, indirect);

        // Combine and tone map
        float3 finalColor = direct + indirect;
        finalColor = ACESFilm(finalColor);

        g_Output[pixel] = float4(finalColor, 1.0);
    }
}
```

## Performance Metrics
- **GPU Time**: 3-8ms for 3 bounces with 8 samples each
- **Memory Usage**: 128 bytes per particle for neighbor data + 4KB irradiance cache
- **Quality Metrics**: Eliminates energy loss, creates realistic glow
- **Convergence**: 5-10 frames for stable indirect illumination

## Hardware Requirements
- **Minimum GPU**: RTX 3070 (for acceptable performance with bounces)
- **Optimal GPU**: RTX 4080+ (high RT core count for secondary rays)

## Implementation Complexity
- **Estimated Dev Time**: 48-56 hours
- **Risk Level**: High (complex interactions, needs careful tuning)
- **Dependencies**: Spatial hash grid, irradiance caching system

## Related Techniques
- Photon mapping
- Irradiance caching
- Path tracing
- Light propagation volumes
- Spherical harmonics

## Notes for PlasmaDX Integration
- Start with single bounce, profile before adding more
- Use spatial hash grid to limit neighbor searches
- Cache indirect lighting across frames for temporal stability
- Consider hybrid: full bounces for hero particles, cached for background
- Importance sample based on particle emission strength
- Use LOD: fewer samples for distant inter-particle bounces
- Monitor total ray count: 20K particles × 8 samples × 3 bounces = 480K secondary rays
- Combine with ReSTIR for better sample efficiency