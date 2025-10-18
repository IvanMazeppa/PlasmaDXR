# ReSTIR Integration for Volumetric Particle Systems

## Source
- Paper: Fast Volume Rendering with Spatiotemporal Reservoir Resampling
- Authors: Daqi Lin, Chris Wyman, Cem Yuksel
- Date: SIGGRAPH Asia 2021
- Additional: Spatiotemporal Reservoir Resampling for Real-time Ray Tracing (NVIDIA 2020)

## Summary
ReSTIR (Reservoir-based Spatiotemporal Importance Resampling) dramatically improves convergence for volumetric particle rendering by reusing samples across space and time. Instead of independently sampling each pixel, ReSTIR maintains reservoirs that accumulate and share high-quality samples between neighboring pixels and frames, reducing noise and improving performance by 10-100x.

For particle systems, ReSTIR enables efficient sampling of complex lighting interactions, including multiple scattering events and indirect illumination, without the computational cost of brute-force path tracing.

## Key Innovation
Extending ReSTIR from surface direct illumination to volumetric path space, maintaining unbiased results while using biased approximations during resampling evaluation. This allows combining cheap approximations for sample reuse with accurate methods for final shading.

## Implementation Details

### Algorithm
```hlsl
// Core ReSTIR algorithm for volumetric particles
struct Reservoir
{
    float3 samplePosition;     // Sampled scattering position
    float3 sampleDirection;     // Scattering direction
    float  samplePdf;          // PDF of the sample
    float  weightSum;          // Sum of weights (W)
    uint   M;                  // Number of samples seen
    float3 radiance;           // Cached radiance value
};

// Main ReSTIR integration
float3 ReSTIRVolumetric(uint2 pixel, float3 rayOrigin, float3 rayDir)
{
    // Step 1: Generate new candidate sample
    Reservoir candidate = GenerateCandidate(rayOrigin, rayDir);

    // Step 2: Load temporal reservoir from previous frame
    Reservoir temporal = LoadTemporalReservoir(pixel);

    // Step 3: Combine with temporal history
    Reservoir combined = CombineReservoirs(candidate, temporal);

    // Step 4: Spatial reuse from neighboring pixels
    combined = SpatialResampling(pixel, combined);

    // Step 5: Final shading with unbiased evaluation
    float3 finalRadiance = ShadeReservoir(combined);

    // Step 6: Store for next frame
    StoreTemporalReservoir(pixel, combined);

    return finalRadiance;
}
```

### Code Snippets

#### Candidate Generation with Importance Sampling
```hlsl
Reservoir GenerateCandidate(float3 origin, float3 direction)
{
    Reservoir r;
    r.M = 1;

    // Importance sample scattering position along ray
    float t = SampleExponential(Random(), g_ExtinctionCoeff);
    r.samplePosition = origin + direction * t;

    // Importance sample scattering direction (phase function)
    float2 xi = Random2D();
    r.sampleDirection = SampleHenyeyGreenstein(xi, -direction, g_Anisotropy);

    // Compute PDF
    r.samplePdf = g_ExtinctionCoeff * exp(-g_ExtinctionCoeff * t) *
                  HenyeyGreenstein(dot(-direction, r.sampleDirection), g_Anisotropy);

    // Estimate radiance (can use biased approximation here)
    r.radiance = EstimateRadianceFast(r.samplePosition, r.sampleDirection);

    // Target PDF proportional to radiance
    float targetPdf = Luminance(r.radiance);
    r.weightSum = targetPdf / r.samplePdf;

    return r;
}
```

#### Temporal Resampling
```hlsl
Reservoir CombineReservoirs(Reservoir r1, Reservoir r2)
{
    Reservoir combined;
    combined.M = r1.M + r2.M;

    // Stream r1 into combined
    combined.samplePosition = r1.samplePosition;
    combined.sampleDirection = r1.sampleDirection;
    combined.samplePdf = r1.samplePdf;
    combined.weightSum = r1.weightSum;
    combined.radiance = r1.radiance;

    // Stream r2 with probability proportional to weight
    float w2 = Luminance(r2.radiance) * r2.M / max(0.001, r2.samplePdf);
    combined.weightSum += w2;

    float probability = w2 / max(0.001, combined.weightSum);
    if (Random() < probability)
    {
        combined.samplePosition = r2.samplePosition;
        combined.sampleDirection = r2.sampleDirection;
        combined.samplePdf = r2.samplePdf;
        combined.radiance = r2.radiance;
    }

    // Clamp M to prevent unbounded growth
    combined.M = min(combined.M, 20);

    return combined;
}
```

#### Spatial Resampling with Visibility Check
```hlsl
Reservoir SpatialResampling(uint2 pixel, Reservoir initial)
{
    const uint NEIGHBOR_COUNT = 5;
    Reservoir result = initial;

    for (uint i = 0; i < NEIGHBOR_COUNT; i++)
    {
        // Select neighbor with spatial jitter
        int2 offset = GetSpatialOffset(i);
        uint2 neighbor = pixel + offset;

        // Boundary check
        if (any(neighbor >= g_Resolution))
            continue;

        Reservoir neighborReservoir = LoadSpatialReservoir(neighbor);

        // Visibility/validity check
        if (!IsVisible(initial.samplePosition, neighborReservoir.samplePosition))
            continue;

        // Jacobian for volume density changes
        float jacobian = ComputeJacobian(initial, neighborReservoir);

        // Stream neighbor sample
        float w = Luminance(neighborReservoir.radiance) * jacobian *
                 neighborReservoir.M / max(0.001, neighborReservoir.samplePdf);

        result.weightSum += w;

        float probability = w / max(0.001, result.weightSum);
        if (Random() < probability)
        {
            result.samplePosition = neighborReservoir.samplePosition;
            result.sampleDirection = neighborReservoir.sampleDirection;
            result.samplePdf = neighborReservoir.samplePdf;
            result.radiance = neighborReservoir.radiance;
        }

        result.M += neighborReservoir.M;
    }

    return result;
}
```

#### Unbiased Final Shading
```hlsl
float3 ShadeReservoir(Reservoir r)
{
    // Recompute radiance with unbiased method for final shading
    float3 unbiasedRadiance = 0;

    // Direct lighting
    for (uint i = 0; i < g_LightCount; i++)
    {
        float3 lightDir = normalize(g_Lights[i].position - r.samplePosition);
        float visibility = TraceVisibilityRay(r.samplePosition, lightDir);

        float phase = HenyeyGreenstein(dot(r.sampleDirection, lightDir), g_Anisotropy);
        unbiasedRadiance += g_Lights[i].color * visibility * phase;
    }

    // Indirect lighting (trace continuation ray)
    if (g_EnableIndirect)
    {
        RayQuery<RAY_FLAG_NONE> query;
        RayDesc ray;
        ray.Origin = r.samplePosition;
        ray.Direction = r.sampleDirection;
        ray.TMin = 0.001f;
        ray.TMax = g_MaxTraceDistance;

        query.TraceRayInline(g_AccelStruct, 0, 0xFF, ray);
        query.Proceed();

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            // Recursive evaluation or environment lookup
            float3 hitRadiance = EvaluateIndirectLighting(query);
            unbiasedRadiance += hitRadiance * g_ScatteringAlbedo;
        }
    }

    // Apply MIS weight
    float misWeight = r.weightSum / max(0.001, r.M * r.samplePdf);
    return unbiasedRadiance * misWeight;
}
```

#### Temporal Reprojection
```hlsl
Reservoir LoadTemporalReservoir(uint2 currentPixel)
{
    // Reproject current pixel to previous frame
    float3 worldPos = ReconstructWorldPosition(currentPixel);
    float4 prevClip = mul(g_PrevViewProj, float4(worldPos, 1.0));
    float2 prevUV = prevClip.xy / prevClip.w * 0.5 + 0.5;
    uint2 prevPixel = prevUV * g_Resolution;

    // Boundary and disocclusion check
    if (any(prevPixel >= g_Resolution))
        return InitializeEmptyReservoir();

    Reservoir temporal = g_PrevReservoirs[prevPixel];

    // Validate temporal sample
    float3 currentDensity = SampleParticleDensity(temporal.samplePosition);
    float3 prevDensity = g_PrevDensityBuffer[prevPixel].rgb;

    // Discard if scene changed significantly
    if (abs(currentDensity - prevDensity) > g_DisocclusionThreshold)
        return InitializeEmptyReservoir();

    return temporal;
}
```

### Data Structures
```hlsl
// ReSTIR buffers
RWStructuredBuffer<Reservoir> g_CurrentReservoirs : register(u0);
StructuredBuffer<Reservoir> g_PrevReservoirs : register(t20);
RWTexture2D<float4> g_PrevDensityBuffer : register(u1);

// ReSTIR configuration
cbuffer ReSTIRConfig : register(b4)
{
    uint  g_MaxM;                    // Max samples in reservoir
    float g_TemporalReuse;           // Temporal reuse factor
    uint  g_SpatialNeighbors;        // Number of spatial neighbors
    float g_SpatialRadius;           // Spatial resampling radius
    float g_DisocclusionThreshold;   // Temporal validation threshold
    uint  g_FrameIndex;              // Current frame number
    float g_TargetPdfScale;          // Scale factor for target PDF
};
```

### Pipeline Integration
```hlsl
[shader("raygeneration")]
void ReSTIRParticleRayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;

    // Initialize or update random seed
    InitializeRandom(pixel, g_FrameIndex);

    // Generate primary ray
    Ray ray = GenerateCameraRay(pixel);

    // Find volume intersection
    float2 tMinMax = IntersectVolumeBounds(ray);

    if (tMinMax.y > tMinMax.x)
    {
        // Apply ReSTIR for volumetric scattering
        float3 radiance = ReSTIRVolumetric(
            pixel,
            ray.origin + ray.direction * tMinMax.x,
            ray.direction
        );

        // Temporal accumulation for additional denoising
        float3 prev = g_PrevOutput[pixel].rgb;
        float blend = g_TemporalReuse;

        g_Output[pixel] = float4(lerp(radiance, prev, blend), 1.0);
    }
}
```

## Performance Metrics
- **GPU Time**: 1-3ms for 1080p with 5 spatial neighbors
- **Convergence**: 10-100x faster than naive Monte Carlo
- **Memory Usage**: 48 bytes per pixel for reservoir storage
- **Quality Metrics**: Near ground-truth with 20x fewer samples

## Hardware Requirements
- **Minimum GPU**: RTX 3070 (8GB VRAM for reservoir buffers)
- **Optimal GPU**: RTX 4090 (massive parallel throughput)

## Implementation Complexity
- **Estimated Dev Time**: 40-48 hours
- **Risk Level**: High (complex algorithm, needs careful debugging)
- **Dependencies**: Temporal reprojection, motion vectors

## Related Techniques
- RIS (Resampled Importance Sampling)
- WRS (Weighted Reservoir Sampling)
- Multiple Importance Sampling (MIS)
- Temporal Accumulation

## Notes for PlasmaDX Integration
- Start with temporal reuse only, add spatial after validation
- Use power heuristic for MIS weights
- Implement streaming RIS to reduce memory bandwidth
- Consider hash-grid spatial structure for neighbor queries
- Bias correction is critical for moving particles
- Can combine with denoising for even better quality
- Monitor reservoir M values - if too high, indicates poor sampling