/**
 * Volumetric ReSTIR - Path Generation Shader (Phase 1)
 *
 * Generates M candidate paths per pixel using regular tracking and
 * performs weighted reservoir sampling (RIS) to select the best path.
 *
 * This is the Phase 1 implementation: RIS only, no spatial/temporal reuse yet.
 *
 * Algorithm:
 * 1. For each pixel: reconstruct camera ray
 * 2. Generate M=4 random walks through volume
 * 3. Use regular tracking (distance sampling with Mip 2)
 * 4. Compute importance weight w = f̂(λ) / p(λ)
 * 5. Weighted reservoir sampling → select 1 winner
 * 6. Store winner in reservoir buffer
 */

#include "volumetric_restir_common.hlsl"

//=============================================================================
// Resources
//=============================================================================

// Constants (root constants, 256 bytes max)
cbuffer PathGenerationConstants : register(b0) {
    uint g_screenWidth;
    uint g_screenHeight;
    uint g_particleCount;
    uint g_randomWalksPerPixel;  // M (default: 4)

    uint g_maxBounces;           // K (default: 3)
    uint g_frameIndex;
    uint g_padding0;
    uint g_padding1;

    float3 g_cameraPos;
    float g_padding2;

    float4x4 g_viewMatrix;
    float4x4 g_projMatrix;
    float4x4 g_invViewProjMatrix;
};

// Particle acceleration structure (BLAS from RTLightingSystem)
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

// Piecewise-constant transmittance volume (Mip 2, coarse grid)
// IMPORTANT: Stored as UINT for atomic operations, convert to float with asfloat()
Texture3D<uint> g_volumeMip2 : register(t2);
SamplerState g_volumeSampler : register(s0);

// Output: reservoir buffer (RWStructuredBuffer)
RWStructuredBuffer<VolumetricReservoir> g_reservoirs : register(u0);

//=============================================================================
// Path Tracing Helpers
//=============================================================================

/**
 * Query particle at ray hit point
 *
 * Uses inline RayQuery to find nearest particle along ray direction.
 *
 * @param origin Ray origin
 * @param direction Ray direction (normalized)
 * @param maxDistance Maximum ray distance
 * @param[out] hitParticleIndex Index of hit particle (-1 if miss)
 * @param[out] hitDistance Distance to hit
 * @return true if hit, false if miss
 */
bool QueryNearestParticle(
    float3 origin,
    float3 direction,
    float maxDistance,
    out uint hitParticleIndex,
    out float hitDistance)
{
    // Setup ray query
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001; // Avoid self-intersection
    ray.TMax = maxDistance;

    // Create query
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(
        g_particleBVH,
        RAY_FLAG_NONE,
        0xFF, // Instance mask
        ray
    );

    // Process potential hits
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            // Procedural primitive hit (particle AABB)
            // Commit the hit (will be handled in intersection shader logic)
            q.CommitProceduralPrimitiveHit(q.CandidateProceduralPrimitiveNonOpaque());
        }
    }

    // Check for committed hit
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ||
        q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        hitParticleIndex = q.CommittedPrimitiveIndex();
        hitDistance = q.CommittedRayT();
        return true;
    }

    return false;
}

/**
 * Evaluate particle emission
 *
 * Returns radiance emitted by particle based on temperature (blackbody).
 *
 * @param particle Particle data
 * @return Emitted radiance (RGB)
 */
float3 EvaluateParticleEmission(Particle particle) {
    float temp = particle.temperature;

    // Blackbody radiation (Wien's approximation)
    // Cool red-orange (1000-3000K) → Yellow (3000-6000K) → White (6000-15000K) → Blue (15000+K)
    float3 color;

    if (temp < 3000.0) {
        // Red-orange
        float t = (temp - 1000.0) / 2000.0;
        color = float3(1.0, 0.2 + 0.5 * t, 0.0);
    } else if (temp < 6000.0) {
        // Yellow-white
        float t = (temp - 3000.0) / 3000.0;
        color = float3(1.0, 0.7 + 0.3 * t, 0.3 * t);
    } else if (temp < 15000.0) {
        // White
        color = float3(1.0, 1.0, 1.0);
    } else {
        // Blue-white
        float t = min((temp - 15000.0) / 15000.0, 1.0);
        color = float3(1.0 - 0.3 * t, 1.0 - 0.3 * t, 1.0);
    }

    // Intensity based on temperature (Stefan-Boltzmann)
    float intensity = pow(temp / 10000.0, 4.0) * 0.1;

    return color * intensity;
}

/**
 * Compute target PDF p̂(λ) for path λ
 *
 * Target function combines:
 * - Emission at scatter vertices
 * - Phase function alignment
 * - Transmittance along path
 *
 * This is the "ideal" PDF we want to importance sample.
 */
float ComputeTargetPDF(
    PathVertex vertices[3],
    uint pathLength,
    float3 rayOrigin,
    float3 rayDirection)
{
    if (pathLength == 0) {
        return 0.0; // Invalid path
    }

    float pdf = 1.0;

    // Reconstruct path positions
    float3 currentPos = rayOrigin;
    float3 currentDir = rayDirection;

    for (uint i = 0; i < pathLength; i++) {
        // Move to next vertex
        currentPos += vertices[i].z * currentDir;
        currentDir = vertices[i].omega;

        // Query particle at this vertex
        uint particleIdx;
        float hitDist;
        if (QueryNearestParticle(currentPos, currentDir, 10.0, particleIdx, hitDist)) {
            Particle p = g_particles[particleIdx];
            float3 emission = EvaluateParticleEmission(p);
            float emissionMagnitude = length(emission);

            // Weight by emission brightness
            pdf *= max(emissionMagnitude, 0.001);
        }
    }

    return pdf;
}

/**
 * Compute source PDF p(λ) for path λ
 *
 * Source PDF is the probability we generated this path using regular tracking.
 * For regular tracking with piecewise-constant volume: p(λ) = Π p(z_i)
 */
float ComputeSourcePDF(
    PathVertex vertices[3],
    uint pathLength,
    float3 rayOrigin)
{
    if (pathLength == 0) {
        return 0.0;
    }

    // For Phase 1: assume uniform PDF (will refine in later phases)
    // In practice, this comes from the distance sampling PDF
    return 1.0 / float(pathLength + 1);
}

/**
 * Generate one candidate path using random walk
 *
 * Performs regular tracking (distance sampling with Mip 2 volume) to generate
 * a random path through the participating medium.
 *
 * @param rayOrigin Camera ray origin
 * @param rayDirection Camera ray direction
 * @param rng Random number generator
 * @param[out] vertices Generated path vertices
 * @param[out] pathLength Number of vertices generated
 * @param[out] flags Path flags (bit 0: isScatteringPath)
 * @return true if valid path generated, false otherwise
 */
bool GenerateCandidatePath(
    float3 rayOrigin,
    float3 rayDirection,
    inout PCGState rng,
    out PathVertex vertices[3],
    out uint pathLength,
    out uint flags)
{
    pathLength = 0;
    flags = 0;

    float3 currentPos = rayOrigin;
    float3 currentDir = rayDirection;

    const float maxRayDistance = 3000.0; // Maximum ray travel distance

    // PHASE 1 STUB: Generate empty paths to test infrastructure
    // TODO: Implement full random walk with volume sampling
    // For now, just create zero-length paths to avoid GPU hangs
    pathLength = 0;
    flags = 0;

    // Return false (no valid path) - allows pipeline to run without GPU hang
    // This will result in black output, which is expected for Phase 1 stub
    return false;

    // DISABLED CODE BELOW - will be enabled in Phase 2 after volume population is validated
    #if 0
    for (uint bounce = 0; bounce < g_maxBounces; bounce++) {
        // Sample distance along ray using regular tracking
        float sampledDist;
        float pdf;

        if (!SampleDistanceRegular(
            currentPos,
            currentDir,
            g_volumeMip2,
            g_volumeSampler,
            maxRayDistance,
            rng,
            sampledDist,
            pdf))
        {
            // Ray escaped volume or hit max distance
            break;
        }

        // Move to scatter point
        currentPos += currentDir * sampledDist;

        // Sample new direction using phase function (Henyey-Greenstein, g=0.7)
        float3 newDir = SampleHenyeyGreenstein(currentDir, 0.7, rng);

        // Store vertex
        vertices[pathLength].z = sampledDist;
        vertices[pathLength].omega = newDir;

        pathLength++;

        // Update for next bounce
        currentDir = newDir;

        // Russian roulette (terminate low-contribution paths)
        if (bounce > 1 && PCGRandomFloat(rng) > 0.8) {
            break;
        }
    }
    #endif

    // Mark as scattering path if we have at least one bounce
    if (pathLength > 0) {
        flags |= 1; // Bit 0: isScatteringPath
        return true;
    }

    return false;
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

    // Initialize random number generator (unique per pixel + frame)
    PCGState rng = InitPCG(pixelCoords, g_frameIndex);

    // Reconstruct camera ray for this pixel
    float3 rayOrigin, rayDirection;
    ReconstructRay(
        pixelCoords,
        uint2(g_screenWidth, g_screenHeight),
        g_invViewProjMatrix,
        rayOrigin,
        rayDirection
    );

    // Initialize reservoir for this pixel
    VolumetricReservoir reservoir = InitReservoir();

    // Generate M candidate paths and perform weighted reservoir sampling
    for (uint candidateIdx = 0; candidateIdx < g_randomWalksPerPixel; candidateIdx++) {
        PathVertex candidateVertices[3];
        uint candidateLength;
        uint candidateFlags;

        // Generate candidate path using random walk
        if (GenerateCandidatePath(
            rayOrigin,
            rayDirection,
            rng,
            candidateVertices,
            candidateLength,
            candidateFlags))
        {
            // Compute importance weight w = p̂(λ) / p(λ)
            float targetPDF = ComputeTargetPDF(candidateVertices, candidateLength, rayOrigin, rayDirection);
            float sourcePDF = ComputeSourcePDF(candidateVertices, candidateLength, rayOrigin);

            float weight = targetPDF / max(sourcePDF, 0.0001);

            // Update reservoir with this candidate
            UpdateReservoir(
                reservoir,
                candidateVertices,
                candidateLength,
                candidateFlags,
                weight,
                rng
            );
        }
    }

    // Write winning path to reservoir buffer
    uint pixelIndex = pixelCoords.y * g_screenWidth + pixelCoords.x;
    g_reservoirs[pixelIndex] = reservoir;
}
