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
    float g_padding1;

    float4x4 g_viewMatrix;
    float4x4 g_projMatrix;
    float4x4 g_invViewProjMatrix;
};

// Particle acceleration structure
RaytracingAccelerationStructure g_particleBVH : register(t0);

// Particle data
struct Particle {
    float3 position;
    float radius;
    float3 velocity;
    float temperature;
    float3 ellipsoidAxis1;
    float padding0;
    float3 ellipsoidAxis2;
    float padding1;
    float3 ellipsoidAxis3;
    float padding2;
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
 * Query particle at position
 *
 * Uses inline RayQuery to find nearest particle within small radius.
 *
 * @param position Query position
 * @param[out] hitParticleIndex Index of nearest particle
 * @return true if particle found, false otherwise
 */
bool QueryParticleAtPosition(float3 position, out uint hitParticleIndex) {
    // Cast ray in arbitrary direction with short distance to detect nearby particle
    float3 direction = float3(0, 1, 0);
    float maxDistance = 10.0;

    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = maxDistance;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            q.CommitProceduralPrimitiveHit(q.CandidateProceduralPrimitiveNonOpaque());
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
    float intensity = pow(temp / 10000.0, 4.0) * 0.1;

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
    float extinction = 0.001; // Very low extinction (mostly transparent medium)

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

        // Query particle at vertex
        uint particleIdx;
        if (QueryParticleAtPosition(currentPos, particleIdx)) {
            Particle particle = g_particles[particleIdx];

            // Evaluate emission
            float3 emission = EvaluateParticleEmission(particle);

            // Evaluate phase function
            float cosTheta = dot(-currentDir, vertex.omega);
            float phase = HenyeyGreenstein(cosTheta, 0.7);

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
