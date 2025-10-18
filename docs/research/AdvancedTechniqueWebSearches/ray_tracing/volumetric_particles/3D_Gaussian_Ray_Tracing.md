# 3D Gaussian Ray Tracing for Particle Systems

## Source
- Paper: 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes
- Authors: Nicolas Moenne-Loccoz et al.
- Date: ACM SIGGRAPH 2024
- Conference/Journal: ACM Transactions on Graphics

## Summary
3D Gaussian ray tracing leverages hardware RT cores to efficiently render volumetric particles represented as anisotropic 3D Gaussians. Instead of rasterizing particles, this approach builds a BVH over Gaussian primitives and traces rays, enabling proper handling of transparency, shadows, reflections, and complex camera models. The technique achieves 25x speedup over naive implementations by using bounding mesh proxies and ordered intersection batching.

This approach is particularly suited for your existing DXR 1.1 infrastructure, as it uses the same BLAS/TLAS acceleration structures you've already built for 20,000 particle AABBs.

## Key Innovation
Encapsulating 3D Gaussian particles with tight bounding meshes (AABBs or oriented boxes) to leverage fast hardware ray-triangle intersection, combined with efficient batch shading of semi-transparent intersections in depth order.

## Implementation Details

### Algorithm
```hlsl
// Core 3D Gaussian ray tracing with DXR 1.1
struct GaussianIntersection
{
    float t;              // Ray parameter
    uint primitiveIndex;  // Gaussian particle index
    float alpha;          // Opacity at intersection
    float3 color;         // Pre-integrated color
};

// Main ray tracing function for Gaussians
float3 TraceGaussianRay(float3 origin, float3 direction)
{
    // Use RayQuery to find all particle intersections
    RayQuery<RAY_FLAG_NONE> query;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;

    query.TraceRayInline(g_AccelStruct, 0, 0xFF, ray);

    // Collect intersections with Gaussians
    const uint MAX_INTERSECTIONS = 32;
    GaussianIntersection intersections[MAX_INTERSECTIONS];
    uint intersectionCount = 0;

    while (query.Proceed() && intersectionCount < MAX_INTERSECTIONS)
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint particleIdx = query.CandidatePrimitiveIndex();
            float t = query.CandidateRayT();

            // Evaluate Gaussian at intersection point
            float3 hitPoint = origin + direction * t;
            GaussianIntersection isect = EvaluateGaussian(particleIdx, hitPoint, direction);

            if (isect.alpha > 0.01f) // Skip nearly transparent
            {
                isect.t = t;
                isect.primitiveIndex = particleIdx;
                intersections[intersectionCount++] = isect;
            }
        }
    }

    // Sort intersections by depth
    SortIntersectionsByDepth(intersections, intersectionCount);

    // Alpha-blend in order
    return BlendGaussians(intersections, intersectionCount, origin, direction);
}
```

### Code Snippets

#### 3D Gaussian Evaluation
```hlsl
GaussianIntersection EvaluateGaussian(uint particleIdx, float3 worldPos, float3 viewDir)
{
    Particle p = g_Particles[particleIdx];

    // Transform to Gaussian local space
    float3 localPos = worldPos - p.center;

    // Anisotropic Gaussian evaluation
    float3x3 covariance = ConstructCovariance(p.scale, p.rotation);
    float3x3 invCov = InvertMatrix3x3(covariance);

    float exponent = -0.5f * dot(localPos, mul(invCov, localPos));
    float gaussianValue = exp(exponent);

    // View-dependent opacity (for splats)
    float3 normal = GetGaussianNormal(p, viewDir);
    float viewAlignment = saturate(dot(normal, -viewDir));

    GaussianIntersection result;
    result.alpha = p.opacity * gaussianValue * viewAlignment;
    result.color = p.color * p.intensity;

    // Add volumetric lighting
    result.color *= ComputeVolumetricLighting(worldPos, normal);

    return result;
}

// Construct 3D covariance matrix from scale and rotation
float3x3 ConstructCovariance(float3 scale, float4 quat)
{
    // Convert quaternion to rotation matrix
    float3x3 R = QuaternionToMatrix(quat);

    // Scaling matrix
    float3x3 S = float3x3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z
    );

    // Covariance = R * S * S^T * R^T
    float3x3 SS = mul(S, transpose(S));
    return mul(mul(R, SS), transpose(R));
}
```

#### Optimized AABB Intersection for Gaussians
```hlsl
[shader("intersection")]
void GaussianIntersection()
{
    uint particleIdx = PrimitiveIndex();
    Particle p = g_Particles[particleIdx];

    Ray ray = GetCurrentRay();

    // Quick AABB test
    float3 halfExtents = p.scale * 3.0f; // 3-sigma bounds
    float2 tMinMax = IntersectAABB(ray.origin, ray.direction, p.center, halfExtents);

    if (tMinMax.x < tMinMax.y)
    {
        // Refined intersection: find where Gaussian alpha > threshold
        const uint SAMPLES = 8;
        float tStep = (tMinMax.y - tMinMax.x) / SAMPLES;

        for (uint i = 0; i < SAMPLES; i++)
        {
            float t = tMinMax.x + i * tStep;
            float3 pos = ray.origin + ray.direction * t;

            GaussianIntersection isect = EvaluateGaussian(particleIdx, pos, ray.direction);

            if (isect.alpha > 0.01f)
            {
                // Report intersection
                ReportHit(t, 0, isect);
                break;
            }
        }
    }
}
```

#### Depth-Ordered Blending
```hlsl
float3 BlendGaussians(GaussianIntersection intersections[32], uint count,
                      float3 rayOrigin, float3 rayDir)
{
    float3 accumulatedColor = 0;
    float accumulatedAlpha = 0;

    for (uint i = 0; i < count; i++)
    {
        GaussianIntersection isect = intersections[i];

        // Alpha blending
        float alpha = isect.alpha * (1.0f - accumulatedAlpha);
        accumulatedColor += isect.color * alpha;
        accumulatedAlpha += alpha;

        // Early termination
        if (accumulatedAlpha > 0.99f)
            break;

        // Add shadows from this Gaussian
        if (i < count - 1) // Not the last one
        {
            float3 pos = rayOrigin + rayDir * isect.t;
            float shadowFactor = ComputeGaussianSelfShadow(pos, isect.primitiveIndex);
            accumulatedColor *= shadowFactor;
        }
    }

    return accumulatedColor;
}
```

#### Self-Shadowing Between Gaussians
```hlsl
float ComputeGaussianSelfShadow(float3 pos, uint excludeIdx)
{
    float shadow = 1.0f;

    for (uint lightIdx = 0; lightIdx < g_LightCount; lightIdx++)
    {
        float3 lightDir = normalize(g_Lights[lightIdx].position - pos);

        // Trace shadow ray through Gaussians
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;

        RayDesc shadowRay;
        shadowRay.Origin = pos + lightDir * 0.01f; // Bias
        shadowRay.Direction = lightDir;
        shadowRay.TMin = 0.001f;
        shadowRay.TMax = length(g_Lights[lightIdx].position - pos);

        shadowQuery.TraceRayInline(g_AccelStruct, 0, 0xFF, shadowRay);

        float occlusion = 0;
        while (shadowQuery.Proceed())
        {
            if (shadowQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint particleIdx = shadowQuery.CandidatePrimitiveIndex();
                if (particleIdx != excludeIdx)
                {
                    float t = shadowQuery.CandidateRayT();
                    float3 hitPos = pos + lightDir * t;

                    // Accumulate Gaussian occlusion
                    GaussianIntersection occluder = EvaluateGaussian(particleIdx, hitPos, lightDir);
                    occlusion += occluder.alpha * 0.5f; // Soft shadow
                }
            }
        }

        shadow *= saturate(1.0f - occlusion);
    }

    return shadow;
}
```

### Data Structures
```hlsl
// 3D Gaussian particle representation
struct GaussianParticle
{
    float3   center;        // World position
    float4   rotation;      // Quaternion
    float3   scale;         // Anisotropic scale
    float3   color;         // Base color (RGB)
    float    opacity;       // Base opacity
    float    intensity;     // Emission strength
    float3x3 viewProjJacobian; // For perspective correction
};

StructuredBuffer<GaussianParticle> g_Particles : register(t0);

// Acceleration structure for Gaussians
RaytracingAccelerationStructure g_AccelStruct : register(t1);

// K-buffer for sorting intersections
struct KBuffer
{
    GaussianIntersection hits[64];
    uint hitCount;
};

RWStructuredBuffer<KBuffer> g_KBuffers : register(u0);
```

### Pipeline Integration
```hlsl
// Build BLAS for Gaussian bounding boxes
D3D12_RAYTRACING_GEOMETRY_DESC CreateGaussianGeometry()
{
    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.AABBs.AABBCount = particleCount;
    geomDesc.AABBs.AABBs.StartAddress = aabbBuffer->GetGPUVirtualAddress();
    geomDesc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

    return geomDesc;
}

// Ray generation for Gaussian particles
[shader("raygeneration")]
void GaussianParticleRayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    float2 uv = (pixel + 0.5f) / DispatchRaysDimensions().xy;

    Ray ray = GenerateCameraRay(uv);

    // Trace through Gaussian particles
    float3 color = TraceGaussianRay(ray.origin, ray.direction);

    // Add environment if not fully opaque
    float3 environment = g_EnvironmentMap.SampleLevel(g_Sampler, ray.direction, 0).rgb;
    color = lerp(environment, color, g_ParticleOpacity);

    g_Output[pixel] = float4(color, 1.0f);
}
```

## Performance Metrics
- **GPU Time**: 1.82ms @ 549 FPS (10 Gaussians/ray) to 2.36ms @ 434 FPS (14 Gaussians/ray)
- **Memory Usage**: 64-96 bytes per Gaussian particle
- **Quality Metrics**: Comparable to rasterization with proper transparency handling
- **Speedup**: 25x faster than naive ray-Gaussian intersection

## Hardware Requirements
- **Minimum GPU**: RTX 2070 (RT cores required)
- **Optimal GPU**: RTX 4080+ (higher RT core throughput)

## Implementation Complexity
- **Estimated Dev Time**: 32-40 hours
- **Risk Level**: Medium (leverages existing TLAS infrastructure)
- **Dependencies**: DXR 1.1, procedural primitive support

## Related Techniques
- Gaussian Splatting (rasterization approach)
- Neural Radiance Fields
- Volume ray marching
- Bounding Volume Hierarchies (BVH)

## Notes for PlasmaDX Integration
- Your existing 20,000 AABBs in TLAS are perfect for this approach
- Use procedural primitive intersection shaders for refined hit tests
- Can combine with existing RayQuery infrastructure
- Consider LOD: use simpler spheres for distant particles
- Batch intersection processing reduces memory bandwidth
- Pre-sort particles by depth for coherent blending
- Use temporal caching for stable results across frames
- Shadow rays can reuse the same Gaussian evaluation code