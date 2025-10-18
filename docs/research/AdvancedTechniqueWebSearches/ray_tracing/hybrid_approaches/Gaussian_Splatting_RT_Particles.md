# 3D Gaussian Ray Tracing for Particle Systems

## Source
- Paper/Article: [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/)
- Authors: Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, Zan Gojcic (NVIDIA)
- Date: SIGGRAPH Asia 2024
- Conference/Journal: ACM Transactions on Graphics

## Summary
3D Gaussian Ray Tracing replaces traditional rasterization-based rendering of particle systems with ray tracing, leveraging modern RT cores for improved quality and flexibility. Instead of projecting particles to screen-space tiles, this approach builds a BVH over particle bounding volumes and traces rays directly.

This enables proper handling of secondary rays (reflections, shadows), distorted cameras, and stochastic sampling while maintaining real-time performance. The technique is particularly effective for volumetric phenomena like accretion disks where particles represent density distributions.

## Key Innovation
Encapsulating Gaussian particles with bounding primitives that leverage fast ray-triangle intersections on RT cores, then shading batches of intersections in depth order. This combines the flexibility of ray tracing with the efficiency of particle-based representations.

## Implementation Details

### Algorithm
```hlsl
// Gaussian particle representation
struct GaussianParticle {
    float3 position;
    float3 scale;
    float4 rotation;  // Quaternion
    float3 sh_dc;     // Spherical harmonic DC term
    float opacity;
};

// Ray-Gaussian intersection
float2 RayGaussianIntersection(float3 rayOrigin, float3 rayDir, GaussianParticle g) {
    // Transform ray to Gaussian's local space
    float3 localOrigin = TransformToLocal(rayOrigin - g.position, g.rotation, g.scale);
    float3 localDir = TransformDirToLocal(rayDir, g.rotation, g.scale);

    // Solve quadratic for ellipsoid intersection
    float a = dot(localDir, localDir);
    float b = 2.0 * dot(localOrigin, localDir);
    float c = dot(localOrigin, localOrigin) - 1.0; // Unit sphere in local space

    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0) return float2(-1, -1);

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-b - sqrtDisc) / (2.0 * a);
    float t2 = (-b + sqrtDisc) / (2.0 * a);

    return float2(t1, t2);
}

// Gaussian evaluation at point
float EvaluateGaussian(float3 point, GaussianParticle g) {
    float3 diff = point - g.position;
    float3 localDiff = TransformToLocal(diff, g.rotation, g.scale);
    float dist2 = dot(localDiff, localDiff);
    return exp(-0.5 * dist2);
}
```

### Code Snippets
```hlsl
// Main ray tracing kernel for Gaussian particles
[numthreads(8, 8, 1)]
void TraceGaussianParticles(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= g_Resolution)) return;

    // Generate primary ray
    float2 uv = (id.xy + 0.5) / g_Resolution;
    RayDesc ray = GenerateCameraRay(uv);

    // Batch processing for efficiency
    const uint BATCH_SIZE = 16;
    uint hitParticles[BATCH_SIZE];
    float hitDistances[BATCH_SIZE];
    uint hitCount = 0;

    // Trace against BVH of particle bounding boxes
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_ParticleBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint particleIdx = q.CandidatePrimitiveIndex();
            GaussianParticle particle = g_Particles[particleIdx];

            // Detailed intersection test
            float2 t = RayGaussianIntersection(ray.Origin, ray.Direction, particle);
            if (t.x > ray.TMin && t.x < ray.TMax) {
                // Insert into sorted batch
                InsertSorted(hitParticles, hitDistances, hitCount, particleIdx, t.x, BATCH_SIZE);
            }
        }
    }

    // Shade batch in depth order
    float3 color = 0;
    float transmittance = 1.0;

    for (uint i = 0; i < hitCount; i++) {
        GaussianParticle p = g_Particles[hitParticles[i]];
        float3 hitPoint = ray.Origin + ray.Direction * hitDistances[i];

        // Evaluate Gaussian density
        float density = EvaluateGaussian(hitPoint, p) * p.opacity;

        // Volume rendering equation
        float3 emission = p.sh_dc * density;
        color += transmittance * emission;
        transmittance *= exp(-density * STEP_SIZE);

        if (transmittance < 0.001) break;
    }

    g_Output[id.xy] = float4(color, 1.0 - transmittance);
}

// Optimized batch shading for multiple intersections
void ShadeBatch(inout float3 color, inout float transmittance, uint batchStart, uint batchEnd) {
    // Process intersections in groups for better memory coherence
    [unroll(4)]
    for (uint i = batchStart; i < batchEnd; i += 4) {
        float4 densities = 0;
        float3 emissions[4];

        // Gather 4 particles at once
        [unroll(4)]
        for (uint j = 0; j < 4 && i + j < batchEnd; j++) {
            uint idx = g_HitBuffer[i + j].particleIdx;
            GaussianParticle p = g_Particles[idx];
            float3 hitPoint = g_HitBuffer[i + j].hitPoint;

            densities[j] = EvaluateGaussian(hitPoint, p) * p.opacity;
            emissions[j] = p.sh_dc * densities[j];
        }

        // Accumulate contributions
        [unroll(4)]
        for (uint j = 0; j < 4 && i + j < batchEnd; j++) {
            color += transmittance * emissions[j];
            transmittance *= exp(-densities[j] * STEP_SIZE);
        }
    }
}
```

### Data Structures
```hlsl
// Acceleration structure for particles
RaytracingAccelerationStructure g_ParticleBVH : register(t0);

// Particle data
StructuredBuffer<GaussianParticle> g_Particles : register(t1);
StructuredBuffer<float3x3> g_CovarianceMatrices : register(t2); // Precomputed

// Hit processing buffers
struct HitRecord {
    uint particleIdx;
    float3 hitPoint;
    float distance;
};
RWStructuredBuffer<HitRecord> g_HitBuffer : register(u0);

// Output
RWTexture2D<float4> g_Output : register(u1);
```

### Pipeline Integration
```hlsl
// Build TLAS for particles
void BuildParticleAccelStruct() {
    // For each particle, create bounding box geometry
    for (uint i = 0; i < numParticles; i++) {
        GaussianParticle p = particles[i];

        // Conservative bounding box (3 standard deviations)
        float3 extent = p.scale * 3.0;
        AABB bounds;
        bounds.min = p.position - extent;
        bounds.max = p.position + extent;

        // Add to BLAS
        AddAABBToAccelStruct(bounds, i);
    }

    // Build TLAS
    BuildTLAS();
}
```

## Performance Metrics
- GPU Time: 5-10ms for 100K particles at 1080p
- Memory Usage: ~50MB for BVH + particle data
- Quality Metrics: Eliminates rasterization artifacts, proper depth ordering

## Hardware Requirements
- Minimum GPU: RTX 3060 (for reasonable RT performance)
- Optimal GPU: RTX 4080 or better (for real-time 100K+ particles)

## Implementation Complexity
- Estimated Dev Time: 3-5 days
- Risk Level: Medium-High (requires BVH construction)
- Dependencies: DXR 1.1, RT cores strongly recommended

## Related Techniques
- [3DGUT - Gaussian Unscented Transform](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
- [VK Gaussian Splatting Sample](https://github.com/nvpro-samples/vk_gaussian_splatting)

## Notes for PlasmaDX Integration
- Excellent fit for volumetric accretion disk rendering
- Can handle millions of particles with proper LOD
- Supports proper transparency sorting automatically
- Enable secondary rays for inter-particle reflections/shadows
- Consider hybrid approach: rasterize distant particles, ray trace near ones