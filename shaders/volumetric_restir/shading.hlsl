/**
 * Volumetric ReSTIR - Final Shading Shader (Phase 1)
 *
 * Reads winning paths from reservoir buffer and evaluates their contributions
 * to produce the final rendered color.
 *
 * Algorithm:
 * 1. Read reservoir winner for pixel
 * 2. Reconstruct path from stored vertices
 * 3. Evaluate path contribution: f̂(λ) / p̂(λ) × W / M
 * 4. Apply transmittance T(x₀→x₁) for first segment
 * 5. Write final color to output texture
 */

#include "volumetric_restir_common.hlsl"

//=============================================================================
// Resources
//=============================================================================

// Constants
cbuffer ShadingConstants : register(b0) {
    uint g_screenWidth;
    uint g_screenHeight;
    uint g_particleCount;
    uint g_padding0;

    float3 g_cameraPos;
    float g_emissionIntensity;   // FIX 2025-11-19: Runtime tunable

    float g_particleRadius;      // FIX 2025-11-19: Runtime tunable
    float g_extinctionCoeff;     // FIX 2025-11-19: Runtime tunable
    float g_phaseG;              // FIX 2025-11-19: Runtime tunable
    float g_padding1;

    float4x4 g_viewMatrix;
    float4x4 g_projMatrix;
    float4x4 g_invViewProjMatrix;
};

// Particle acceleration structure
RaytracingAccelerationStructure g_particleBVH : register(t0);

// Particle data (48 bytes, matches ParticleSystem.h)
struct Particle {
    float3 position;       // 12 bytes (offset 0)
    float temperature;     // 4 bytes  (offset 12)
    float3 velocity;       // 12 bytes (offset 16)
    float density;         // 4 bytes  (offset 28)
    float3 albedo;         // 12 bytes (offset 32)
    uint materialType;     // 4 bytes  (offset 44)
};
StructuredBuffer<Particle> g_particles : register(t1);

// Reservoir buffer (input - read winners)
StructuredBuffer<VolumetricReservoir> g_reservoirs : register(t2);

// Output texture (RGBA render target)
RWTexture2D<float4> g_outputTexture : register(u0);

//=============================================================================
// Shading Helpers
//=============================================================================

/**
 * Ray-sphere intersection test
 *
 * @param rayOrigin Ray origin
 * @param rayDir Ray direction (normalized)
 * @param sphereCenter Sphere center
 * @param sphereRadius Sphere radius
 * @param[out] hitT Distance to hit (nearest intersection)
 * @return true if hit, false if miss
 */
bool RaySphereIntersection(
    float3 rayOrigin,
    float3 rayDir,
    float3 sphereCenter,
    float sphereRadius,
    out float hitT)
{
    float3 oc = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        hitT = 0.0;
        return false;
    }

    float sqrtD = sqrt(discriminant);
    float t0 = (-b - sqrtD) / (2.0 * a);
    float t1 = (-b + sqrtD) / (2.0 * a);

    // Return nearest positive hit
    if (t0 > 0.001) {
        hitT = t0;
        return true;
    } else if (t1 > 0.001) {
        hitT = t1;
        return true;
    }

    hitT = 0.0;
    return false;
}

/**
 * Query particle along path segment
 *
 * Re-traces the ray segment to identify which particle was hit.
 * This matches the logic used in path_generation.hlsl to ensure consistency.
 *
 * @param origin Segment start position
 * @param direction Segment direction
 * @param distance Segment length (hit distance)
 * @param[out] hitParticleIndex Index of hit particle
 * @return true if particle found
 */
bool QueryParticleFromRay(float3 origin, float3 direction, float distance, out uint hitParticleIndex) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001; // Same as generation
    ray.TMax = distance + 0.1; // Slight epsilon to ensure we hit it

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    // Find the closest hit (which should match the generated vertex)
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            // FIX 2025-11-19: Must perform actual intersection test for procedural primitives
            uint particleIdx = q.CandidatePrimitiveIndex();
            Particle p = g_particles[particleIdx];

            // Ray-sphere intersection using particle position
            // FIX 2025-11-19: Now uses runtime-tunable g_particleRadius
            float sphereRadius = g_particleRadius;
            float intersectT;
            if (RaySphereIntersection(origin, direction, p.position, sphereRadius, intersectT)) {
                // Valid intersection within ray bounds
                if (intersectT >= ray.TMin && intersectT <= ray.TMax) {
                    q.CommitProceduralPrimitiveHit(intersectT);
                }
            }
        }
    }

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ||
        q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        hitParticleIndex = q.CommittedPrimitiveIndex();
        return true;
    }

    return false;
}

/**
 * Evaluate particle emission (blackbody radiation)
 *
 * @param particle Particle data
 * @return Emitted radiance (RGB)
 */
float3 EvaluateParticleEmission(Particle particle) {
    float temp = particle.temperature;

    // Blackbody color (Wien's approximation)
    float3 color;

    if (temp < 3000.0) {
        float t = (temp - 1000.0) / 2000.0;
        color = float3(1.0, 0.2 + 0.5 * t, 0.0);
    } else if (temp < 6000.0) {
        float t = (temp - 3000.0) / 3000.0;
        color = float3(1.0, 0.7 + 0.3 * t, 0.3 * t);
    } else if (temp < 15000.0) {
        color = float3(1.0, 1.0, 1.0);
    } else {
        float t = min((temp - 15000.0) / 15000.0, 1.0);
        color = float3(1.0 - 0.3 * t, 1.0 - 0.3 * t, 1.0);
    }

    // Intensity (Stefan-Boltzmann law, T⁴)
    // FIX 2025-11-19: Now uses runtime-tunable g_emissionIntensity
    float intensity = pow(temp / 10000.0, 4.0) * g_emissionIntensity;

    return color * intensity;
}

/**
 * Evaluate transmittance along ray segment
 *
 * Uses Beer-Lambert law: T = exp(-∫ σ_t(s) ds)
 * For Phase 1, simplified to constant extinction.
 *
 * @param start Segment start position
 * @param end Segment end position
 * @return Transmittance [0, 1]
 */
float3 EvaluateTransmittance(float3 start, float3 end) {
    float distance = length(end - start);

    // Extinction coefficient (constant for Phase 1)
    // FIX 2025-11-19: Now uses runtime-tunable g_extinctionCoeff
    float extinction = g_extinctionCoeff;

    // Beer-Lambert law
    float transmittance = exp(-extinction * distance);

    return float3(transmittance, transmittance, transmittance);
}

/**
 * Evaluate path contribution
 *
 * Computes the Monte Carlo estimator for path λ:
 * L = f̂(λ) / p̂(λ) × W / M
 *
 * Where:
 * - f̂(λ) = target function (emission × phase × transmittance)
 * - p̂(λ) = target PDF (same as f̂ for unbiased estimation)
 * - W = reservoir weight sum
 * - M = total candidates seen
 *
 * @param reservoir Reservoir containing winning path
 * @param rayOrigin Camera ray origin
 * @param rayDirection Camera ray direction
 * @return Path radiance contribution
 */
float3 EvaluatePathContribution(
    VolumetricReservoir reservoir,
    float3 rayOrigin,
    float3 rayDirection)
{
    // Check for valid path
    if (reservoir.pathLength == 0 || reservoir.M == 0.0) {
        return float3(0, 0, 0);
    }

    float3 radiance = float3(0, 0, 0);

    // Reconstruct path positions
    float3 currentPos = rayOrigin;
    float3 currentDir = rayDirection;

    for (uint i = 0; i < reservoir.pathLength; i++) {
        PathVertex vertex = reservoir.vertices[i];

        // Move to vertex
        float3 prevPos = currentPos;
        currentPos += currentDir * vertex.z;

        // Query particle at vertex by re-tracing the segment
        uint particleIdx;
        if (QueryParticleFromRay(prevPos, currentDir, vertex.z, particleIdx)) {
            Particle particle = g_particles[particleIdx];

            // Evaluate emission
            float3 emission = EvaluateParticleEmission(particle);

            // Evaluate phase function
            // FIX 2025-11-19: Now uses runtime-tunable g_phaseG
            float cosTheta = dot(-currentDir, vertex.omega);
            float phase = HenyeyGreenstein(cosTheta, g_phaseG);

            // Evaluate transmittance along this segment
            float3 transmittance = EvaluateTransmittance(prevPos, currentPos);

            // Accumulate contribution
            radiance += emission * phase * transmittance;
        }

        // Update direction for next segment
        currentDir = vertex.omega;
    }

    // Apply MIS weight: W / M
    float misWeight = reservoir.wsum / max(reservoir.M, 1.0);
    radiance *= misWeight;

    return radiance;
}

//=============================================================================
// Main Compute Shader
//=============================================================================

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixelCoords = dispatchThreadID.xy;

    // Bounds check
    if (pixelCoords.x >= g_screenWidth || pixelCoords.y >= g_screenHeight) {
        return;
    }

    // Read reservoir winner for this pixel
    uint pixelIndex = pixelCoords.y * g_screenWidth + pixelCoords.x;
    VolumetricReservoir reservoir = g_reservoirs[pixelIndex];

    // Reconstruct camera ray
    float3 rayOrigin, rayDirection;
    ReconstructRay(
        pixelCoords,
        uint2(g_screenWidth, g_screenHeight),
        g_invViewProjMatrix,
        rayOrigin,
        rayDirection
    );

    // Evaluate path contribution
    float3 color = EvaluatePathContribution(reservoir, rayOrigin, rayDirection);

    // Tone mapping (simple Reinhard)
    color = color / (1.0 + color);

    // Gamma correction (approximate sRGB)
    color = pow(color, 1.0 / 2.2);

    // Write to output texture
    g_outputTexture[pixelCoords] = float4(color, 1.0);
}
