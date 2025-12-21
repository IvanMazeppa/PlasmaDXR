// DXR Per-Particle AABB Generation for 3D Gaussian Splatting
// Generates conservative AABBs for Gaussian ellipsoids
// Updates every frame as particles move
// Now includes GPU-side frustum culling (2025-12-11)

#include "particles/gaussian_common.hlsl"

cbuffer AABBConstants : register(b0)
{
    // Original constants (12 DWORDs)
    uint particleCount;
    float particleRadius;  // Base particle radius for Gaussian sizing

    // Phase 1.5 - Dual AS Particle Offset (CRITICAL FIX)
    uint particleOffset;           // Start reading from this particle index (0 for probe grid, 2044 for Direct RT)
    uint padding1;                 // Alignment

    // Phase 1.5 Adaptive Particle Radius
    uint enableAdaptiveRadius;     // Toggle for density/distance-based radius scaling
    float adaptiveInnerZone;       // Distance threshold for inner shrinking
    float adaptiveOuterZone;       // Distance threshold for outer expansion
    float adaptiveInnerScale;      // Min scale for inner dense regions
    float adaptiveOuterScale;      // Max scale for outer sparse regions
    float densityScaleMin;         // Min density scale clamp
    float densityScaleMax;         // Max density scale clamp
    float padding2;                // Padding for alignment

    // Frustum Culling (2025-12-11 optimization) - 26 DWORDs
    float4 frustumPlanes[6];       // Left, Right, Bottom, Top, Near, Far (normalized, inward-facing)
    uint enableFrustumCulling;     // Toggle for frustum culling
    float frustumExpansion;        // Expand particle radius by this factor to avoid pop-in (1.5 default)
};

StructuredBuffer<Particle> particles : register(t0);

// Output: Per-particle AABB buffer (D3D12_RAYTRACING_AABB format)
struct AABBOutput
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

RWStructuredBuffer<AABBOutput> particleAABBs : register(u0);

// ============================================================================
// Frustum Culling Helper Functions
// ============================================================================

// Test if a sphere is completely outside a frustum plane
// Returns true if the sphere is entirely behind (outside) the plane
bool SphereOutsidePlane(float3 center, float radius, float4 plane)
{
    // plane.xyz = normal (points inward), plane.w = distance
    // Signed distance from center to plane
    float dist = dot(center, plane.xyz) + plane.w;

    // If center + radius is still behind plane, sphere is completely outside
    return dist < -radius;
}

// Test if a sphere is outside the view frustum
// Returns true if the sphere is completely outside ANY frustum plane
bool SphereOutsideFrustum(float3 center, float radius)
{
    // Expand radius to prevent pop-in at frustum edges
    float expandedRadius = radius * frustumExpansion;

    // Test against all 6 frustum planes
    // If sphere is completely behind any plane, it's outside the frustum
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        if (SphereOutsidePlane(center, expandedRadius, frustumPlanes[i]))
        {
            return true;  // Outside this plane, therefore outside frustum
        }
    }

    return false;  // Inside all planes, therefore visible
}

// ============================================================================
// Main AABB Generation
// ============================================================================

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIndex = dispatchThreadID.x;

    // Early exit if outside particle count
    if (particleIndex >= particleCount)
        return;

    // Read particle with offset (0 for probe grid, 2044 for Direct RT)
    // CRITICAL: This ensures Direct RT reads particles 2044-9999, not 0-9999 (no duplicates!)
    Particle p = particles[particleIndex + particleOffset];

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

    // ========================================================================
    // FRUSTUM CULLING (2025-12-11 optimization)
    // ========================================================================
    // If particle's bounding sphere is completely outside the frustum,
    // write a degenerate AABB (min > max) that DXR will treat as empty
    if (enableFrustumCulling)
    {
        // Compute bounding sphere radius from AABB extent
        float3 extent = gaussianAABB.maxPoint - gaussianAABB.minPoint;
        float boundingRadius = length(extent) * 0.5;

        if (SphereOutsideFrustum(p.position, boundingRadius))
        {
            // Write degenerate AABB - DXR handles this gracefully
            // Inverted bounds (min > max) result in zero-volume geometry
            AABBOutput degenerateAABB;
            degenerateAABB.minX = 1.0;
            degenerateAABB.maxX = 0.0;
            degenerateAABB.minY = 1.0;
            degenerateAABB.maxY = 0.0;
            degenerateAABB.minZ = 1.0;
            degenerateAABB.maxZ = 0.0;

            particleAABBs[particleIndex] = degenerateAABB;
            return;  // Skip normal AABB write
        }
    }

    // Write normal AABB to output buffer in D3D12 format
    AABBOutput aabb;
    aabb.minX = gaussianAABB.minPoint.x;
    aabb.minY = gaussianAABB.minPoint.y;
    aabb.minZ = gaussianAABB.minPoint.z;
    aabb.maxX = gaussianAABB.maxPoint.x;
    aabb.maxY = gaussianAABB.maxPoint.y;
    aabb.maxZ = gaussianAABB.maxPoint.z;

    particleAABBs[particleIndex] = aabb;
}
