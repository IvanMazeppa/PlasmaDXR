// 3D Gaussian Splatting Ray Tracing
// Replaces billboard rasterization with volumetric ray-traced Gaussians
// Proper depth sorting, transparency, and volumetric appearance

#include "gaussian_common.hlsl"

cbuffer CameraConstants : register(b0)
{
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
    float3 cameraPos;
    float padding0;
    float3 cameraRight;
    float padding1;
    float3 cameraUp;
    float padding2;
    float2 resolution;
    float2 invResolution;
};

cbuffer GaussianConstants : register(b1)
{
    float baseParticleRadius;
    uint maxIntersectionsPerRay;
    float volumeStepSize;
    float densityMultiplier;
};

// Input: Particle buffer
StructuredBuffer<Particle> g_particles : register(t0);

// Input: RT lighting (from particle-to-particle lighting pass)
StructuredBuffer<float4> g_rtLighting : register(t1);

// Input: Ray tracing acceleration structure
RaytracingAccelerationStructure g_particleBVH : register(t2);

// Output: Final rendered image
RWTexture2D<float4> g_output : register(u0);

// Hit record for batch processing
struct HitRecord {
    uint particleIdx;
    float tNear;
    float tFar;
    float sortKey; // For depth sorting
};

// Generate camera ray from pixel coordinates
RayDesc GenerateCameraRay(float2 pixelPos) {
    // NDC coordinates (-1 to 1)
    float2 ndc = (pixelPos + 0.5) * invResolution * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for D3D

    // Unproject to world space
    float4 nearPoint = mul(float4(ndc, 0.0, 1.0), invViewProj);
    float4 farPoint = mul(float4(ndc, 1.0, 1.0), invViewProj);

    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    RayDesc ray;
    ray.Origin = nearPoint.xyz;
    ray.Direction = normalize(farPoint.xyz - nearPoint.xyz);
    ray.TMin = 0.01;
    ray.TMax = 10000.0;

    return ray;
}

// Insert hit into sorted list (simple insertion sort for small batches)
void InsertHit(inout HitRecord hits[64], inout uint hitCount, uint particleIdx, float tNear, float tFar, uint maxHits) {
    if (hitCount >= maxHits) return;

    HitRecord newHit;
    newHit.particleIdx = particleIdx;
    newHit.tNear = tNear;
    newHit.tFar = tFar;
    newHit.sortKey = tNear; // Sort by entry distance

    // Insertion sort (simple for small batches)
    uint insertPos = hitCount;
    for (uint i = 0; i < hitCount; i++) {
        if (newHit.sortKey < hits[i].sortKey) {
            insertPos = i;
            break;
        }
    }

    // Shift elements
    for (uint i = hitCount; i > insertPos; i--) {
        hits[i] = hits[i - 1];
    }

    hits[insertPos] = newHit;
    hitCount++;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;

    // Early exit if outside render target
    if (any(pixelPos >= (uint2)resolution))
        return;

    // Generate primary ray
    RayDesc ray = GenerateCameraRay((float2)pixelPos);

    // Collect all Gaussian intersections via RayQuery
    const uint MAX_HITS = 64; // Batch size
    HitRecord hits[MAX_HITS];
    uint hitCount = 0;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    // Process all AABB candidates
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = query.CandidatePrimitiveIndex();

            // Read particle
            Particle p = g_particles[particleIdx];

            // Compute Gaussian parameters
            float3 scale = ComputeGaussianScale(p, baseParticleRadius);
            float3x3 rotation = ComputeGaussianRotation(p.velocity);

            // Detailed Gaussian-ellipsoid intersection
            float2 t = RayGaussianIntersection(ray.Origin, ray.Direction, p.position, scale, rotation);

            // Valid intersection?
            if (t.x > ray.TMin && t.x < ray.TMax && t.y > t.x) {
                // Commit the AABB hit (required for procedural primitives)
                query.CommitProceduralPrimitiveHit(t.x);

                // Store in hit list
                InsertHit(hits, hitCount, particleIdx, t.x, t.y, MAX_HITS);
            }
        }
    }

    // Volume rendering: march through sorted Gaussians
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;

    for (uint i = 0; i < hitCount; i++) {
        // Early exit if fully opaque
        if (transmittance < 0.001) break;

        HitRecord hit = hits[i];
        Particle p = g_particles[hit.particleIdx];

        // Gaussian parameters
        float3 scale = ComputeGaussianScale(p, baseParticleRadius);
        float3x3 rotation = ComputeGaussianRotation(p.velocity);

        // Ray-march through this Gaussian
        float tStart = max(hit.tNear, ray.TMin);
        float tEnd = min(hit.tFar, ray.TMax);

        uint steps = max(1, (uint)((tEnd - tStart) / volumeStepSize));
        float stepSize = (tEnd - tStart) / steps;

        for (uint step = 0; step < steps; step++) {
            float t = tStart + (step + 0.5) * stepSize;
            float3 pos = ray.Origin + ray.Direction * t;

            // Evaluate Gaussian density at this point
            float density = EvaluateGaussianDensity(pos, p.position, scale, rotation, p.density);
            density *= densityMultiplier;

            // Skip if negligible
            if (density < 0.001) continue;

            // Emission color from temperature
            float3 emission = TemperatureToEmission(p.temperature);
            float intensity = EmissionIntensity(p.temperature);

            // Add RT lighting from particle-to-particle pass
            float3 rtLight = g_rtLighting[hit.particleIdx].rgb;

            // Total emission (self-emission + RT lighting)
            float3 totalEmission = emission * intensity + rtLight;

            // Volume rendering equation
            float absorption = density * stepSize;
            accumulatedColor += transmittance * totalEmission * absorption;
            transmittance *= exp(-absorption);

            // Early exit
            if (transmittance < 0.001) break;
        }
    }

    // Background color (dark space)
    float3 backgroundColor = float3(0.0, 0.0, 0.1);
    float3 finalColor = accumulatedColor + transmittance * backgroundColor;

    // Tone mapping (Reinhard)
    finalColor = finalColor / (1.0 + finalColor);

    // Gamma correction
    finalColor = pow(finalColor, 1.0 / 2.2);

    g_output[pixelPos] = float4(finalColor, 1.0);
}