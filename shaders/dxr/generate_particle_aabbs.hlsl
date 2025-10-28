// DXR Per-Particle AABB Generation for 3D Gaussian Splatting
// Generates conservative AABBs for Gaussian ellipsoids
// Updates every frame as particles move

#include "../particles/gaussian_common.hlsl"

cbuffer AABBConstants : register(b0)
{
    uint particleCount;
    float particleRadius;  // Base particle radius for Gaussian sizing

    // Phase 1.5 Adaptive Particle Radius
    uint enableAdaptiveRadius;     // Toggle for density/distance-based radius scaling
    float adaptiveInnerZone;       // Distance threshold for inner shrinking
    float adaptiveOuterZone;       // Distance threshold for outer expansion
    float adaptiveInnerScale;      // Min scale for inner dense regions
    float adaptiveOuterScale;      // Max scale for outer sparse regions
    float densityScaleMin;         // Min density scale clamp
    float densityScaleMax;         // Max density scale clamp
    float padding;                 // Padding for alignment
};

StructuredBuffer<Particle> particles : register(t0);

// Output: Per-particle AABB buffer (D3D12_RAYTRACING_AABB format)
struct AABBOutput
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

RWStructuredBuffer<AABBOutput> particleAABBs : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIndex = dispatchThreadID.x;

    // Early exit if outside particle count
    if (particleIndex >= particleCount)
        return;

    // Read particle
    Particle p = particles[particleIndex];

    // Compute Gaussian AABB (conservative, axis-aligned bound)
    // Use maximum anisotropic bounds for conservative AABB (true, 3.0 max strength)
    AABB gaussianAABB = ComputeGaussianAABB(
        p, particleRadius, true, 3.0,
        enableAdaptiveRadius != 0,
        adaptiveInnerZone,
        adaptiveOuterZone,
        adaptiveInnerScale,
        adaptiveOuterScale,
        densityScaleMin,
        densityScaleMax
    );

    // Write to output buffer in D3D12 format
    AABBOutput aabb;
    aabb.minX = gaussianAABB.minPoint.x;
    aabb.minY = gaussianAABB.minPoint.y;
    aabb.minZ = gaussianAABB.minPoint.z;
    aabb.maxX = gaussianAABB.maxPoint.x;
    aabb.maxY = gaussianAABB.maxPoint.y;
    aabb.maxZ = gaussianAABB.maxPoint.z;

    particleAABBs[particleIndex] = aabb;
}
