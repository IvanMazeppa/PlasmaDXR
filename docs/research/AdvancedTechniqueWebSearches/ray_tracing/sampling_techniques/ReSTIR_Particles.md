# ReSTIR for Particle Systems

## Source
- Paper/Article: [A Gentle Introduction to ReSTIR Path Reuse in Real-Time](https://dl.acm.org/doi/10.1145/3587423.3595511)
- Authors: Chris Wyman, NVIDIA
- Date: SIGGRAPH 2023-2024
- Conference/Journal: ACM SIGGRAPH 2023 Courses

## Summary
ReSTIR (Reservoir-based Spatiotemporal Importance Resampling) enables efficient reuse of samples across pixels and frames, achieving 100× efficiency improvements over traditional path tracing. For particle systems, ReSTIR can dramatically reduce noise in particle-to-particle lighting calculations by reusing visibility and lighting samples across neighboring particles and temporal frames.

The technique works by maintaining reservoirs of light samples that are repeatedly resampled and combined spatially and temporally. This is particularly effective for particle systems where neighboring particles often have similar lighting conditions.

## Key Innovation
The ability to reuse expensive ray tracing samples across particles and frames while maintaining an unbiased Monte Carlo estimator. This allows particle systems with complex inter-particle lighting to achieve real-time performance with minimal noise.

## Implementation Details

### Algorithm
```hlsl
// Simplified ReSTIR for particle lighting
struct Reservoir {
    float3 lightSample;      // Selected light direction/position
    float weightSum;          // Sum of weights seen
    float M;                  // Number of samples seen
    float W;                  // Final weight
};

// Per-particle reservoir update
Reservoir UpdateReservoir(Reservoir r, float3 sample, float weight, float randomVal) {
    r.weightSum += weight;
    r.M += 1.0;

    // Stochastically update sample
    if (randomVal < weight / r.weightSum) {
        r.lightSample = sample;
    }

    return r;
}

// Combine reservoirs from neighboring particles
Reservoir CombineReservoirs(Reservoir r1, Reservoir r2, float randomVal) {
    Reservoir combined;
    combined.weightSum = r1.weightSum + r2.weightSum;
    combined.M = r1.M + r2.M;

    if (randomVal < r1.weightSum / combined.weightSum) {
        combined.lightSample = r1.lightSample;
    } else {
        combined.lightSample = r2.lightSample;
    }

    return combined;
}
```

### Code Snippets
```hlsl
// Temporal reuse for particles
[numthreads(256, 1, 1)]
void ParticleReSTIRTemporal(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_NumParticles) return;

    Particle p = g_Particles[particleIdx];

    // Get previous frame's reservoir
    Reservoir prevReservoir = g_PrevReservoirs[particleIdx];
    Reservoir currReservoir = g_CurrReservoirs[particleIdx];

    // Validate temporal sample is still valid
    float3 prevSamplePos = prevReservoir.lightSample;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    RayDesc ray;
    ray.Origin = p.position;
    ray.Direction = normalize(prevSamplePos - p.position);
    ray.TMin = 0.001;
    ray.TMax = length(prevSamplePos - p.position);

    q.TraceRayInline(g_AccelStruct, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_NOTHING) {
        // Temporal sample is valid, combine
        float random = Hash(particleIdx + g_FrameCount);
        currReservoir = CombineReservoirs(currReservoir, prevReservoir, random);
    }

    // Normalize weights
    currReservoir.W = currReservoir.weightSum / max(1.0, currReservoir.M);
    g_OutputReservoirs[particleIdx] = currReservoir;
}

// Spatial reuse across neighboring particles
[numthreads(256, 1, 1)]
void ParticleReSTIRSpatial(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_NumParticles) return;

    Particle p = g_Particles[particleIdx];
    Reservoir reservoir = g_InputReservoirs[particleIdx];

    // Find neighboring particles within radius
    const float searchRadius = 0.1; // In world units
    const uint maxNeighbors = 8;

    for (uint i = 0; i < maxNeighbors; i++) {
        uint neighborIdx = GetNeighborParticle(particleIdx, i, searchRadius);
        if (neighborIdx == INVALID_INDEX) continue;

        Particle neighbor = g_Particles[neighborIdx];
        Reservoir neighborReservoir = g_InputReservoirs[neighborIdx];

        // Validate neighbor's sample for current particle
        float3 samplePos = neighborReservoir.lightSample;
        if (IsVisible(p.position, samplePos)) {
            float random = Hash(particleIdx * maxNeighbors + i);
            reservoir = CombineReservoirs(reservoir, neighborReservoir, random);
        }
    }

    g_OutputReservoirs[particleIdx] = reservoir;
}
```

### Data Structures
```hlsl
// Buffer layouts for ReSTIR particle lighting
StructuredBuffer<Reservoir> g_PrevReservoirs : register(t0);
StructuredBuffer<Reservoir> g_CurrReservoirs : register(t1);
RWStructuredBuffer<Reservoir> g_OutputReservoirs : register(u0);
RaytracingAccelerationStructure g_AccelStruct : register(t2);

// Spatial acceleration structure for neighbor queries
StructuredBuffer<uint> g_ParticleGrid : register(t3);
StructuredBuffer<uint2> g_GridOffsets : register(t4); // Start/count per cell
```

### Pipeline Integration
1. **Initial Sampling**: Generate initial light samples per particle
2. **Temporal Reuse**: Combine with previous frame's reservoirs
3. **Spatial Reuse**: Exchange samples with neighboring particles
4. **Final Shading**: Use reservoir samples for lighting calculation

## Performance Metrics
- GPU Time: 2-5ms for 100K particles with 3 spatial passes
- Memory Usage: ~16MB for reservoirs (100K particles)
- Quality Metrics: 6-60× variance reduction vs. naive sampling

## Hardware Requirements
- Minimum GPU: RTX 2060 or AMD RX 6600 (DXR 1.1 support)
- Optimal GPU: RTX 4070 or better for inline ray tracing performance

## Implementation Complexity
- Estimated Dev Time: 2-3 days for basic implementation
- Risk Level: Medium (temporal artifacts possible)
- Dependencies: DXR 1.1, Compute Shader 5.0+

## Related Techniques
- [World-Space ReSTIR](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14974)
- [Area ReSTIR for Antialiasing](https://research.nvidia.com/publication/2024-08_area-restir)

## Notes for PlasmaDX Integration
- Perfect fit for accretion disk particle-to-particle lighting
- Can reuse existing RayQuery infrastructure
- Consider world-space grid for spatial neighbor queries
- Implement progressive quality levels for performance scaling