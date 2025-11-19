/**
 * Volumetric ReSTIR - Path Generation Shader (Phase 1)
 *
 * Generates M candidate paths per pixel using direct BVH regular tracking and
 * performs weighted reservoir sampling (RIS) to select the best path.
 *
 * FIX 2025-11-19: Replaced volumetric grid sampling (atomic contention) with 
 * direct RayQuery against particle BVH.
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

// Piecewise-constant transmittance volume (Mip 2, coarse grid)
// UNUSED IN FIX: We use BVH instead
Texture3D<uint> g_volumeMip2 : register(t2);
SamplerState g_volumeSampler : register(s0);

// Output: reservoir buffer (RWStructuredBuffer)
RWStructuredBuffer<VolumetricReservoir> g_reservoirs : register(u0);

//=============================================================================
// Path Tracing Helpers
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
            // FIX 2025-11-19: Must perform actual intersection test for procedural primitives
            uint particleIdx = q.CandidatePrimitiveIndex();
            Particle p = g_particles[particleIdx];

            // Ray-sphere intersection using particle position
            // FIX 2025-11-19: Use 50.0 to match particle radius setting (was 10.0, causing tiny particles)
            float sphereRadius = 50.0;
            float intersectT;
            if (RaySphereIntersection(origin, direction, p.position, sphereRadius, intersectT)) {
                // Valid intersection within ray bounds
                if (intersectT >= ray.TMin && intersectT <= ray.TMax) {
                    q.CommitProceduralPrimitiveHit(intersectT);
                }
            }
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
    // FIX 2025-11-19: Increased from 0.1 to 10.0 for visibility (was way too dim)
    float intensity = pow(temp / 10000.0, 4.0) * 10.0;

    return color * intensity;
}

/**
 * Compute target PDF p̂(λ) for path λ
 *
 * Target function combines:
 * - Emission at scatter vertices
 * - Phase function alignment
 * - Transmittance along path
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
 * Performs regular tracking (distance sampling with BVH RayQuery) to generate
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

    // Use hardware RayQuery to find next scattering event
    // This replaces the expensive/crashy volumetric grid marching
    
    // Single bounce for Phase 1 stability
    // We can increase this later
    const uint maxBounces = 1; 
    
    for (uint bounce = 0; bounce < maxBounces; bounce++) {
        uint hitIdx;
        float hitDist;
        
        if (QueryNearestParticle(currentPos, currentDir, maxRayDistance, hitIdx, hitDist)) {
            // We hit a particle! This is a valid candidate path.
            
            // Move to hit position
            currentPos += currentDir * hitDist;
            
            // Sample a new direction (Phase function)
            float3 newDir = SampleHenyeyGreenstein(-currentDir, 0.5, rng);
            
            // Store vertex
            // z: distance from previous point to this hit
            // omega: outgoing direction FROM this hit
            vertices[pathLength].z = hitDist;
            vertices[pathLength].omega = newDir;
            
            pathLength++;
            flags |= 1; // Valid scattering path
            
            // Stop after 1 bounce for now
            break;
        } else {
            // Ray missed everything - path terminates
            break;
        }
    }

    return pathLength > 0;
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