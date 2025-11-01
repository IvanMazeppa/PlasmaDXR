/**
 * Volumetric ReSTIR - Volume Mip 2 Population Shader
 *
 * Splats particle density into a 64³ voxel grid for piecewise-constant
 * transmittance estimation (T*).
 *
 * Algorithm:
 * 1. For each particle, compute its world-space AABB
 * 2. Map AABB to voxel space (64³ grid covering scene bounds)
 * 3. For each overlapping voxel, accumulate density
 * 4. Density = particle volume contribution to voxel volume
 *
 * Output: R16_FLOAT 3D texture with extinction coefficient σ_t
 */

//=============================================================================
// Constants
//=============================================================================

cbuffer PopulationConstants : register(b0) {
    uint g_particleCount;
    uint g_volumeResolution;  // 64 for Mip 2
    uint g_padding0;
    uint g_padding1;

    float3 g_worldMin;        // Scene bounds min (-1500, -1500, -1500)
    float g_padding2;

    float3 g_worldMax;        // Scene bounds max (+1500, +1500, +1500)
    float g_padding3;

    float g_extinctionScale;  // Scale factor for extinction (default: 0.001)
    float g_padding4;
    float g_padding5;
    float g_padding6;
};

//=============================================================================
// Resources
//=============================================================================

// Particle data (same structure as Gaussian renderer)
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

StructuredBuffer<Particle> g_particles : register(t0);

// Output volume texture (R16_FLOAT)
RWTexture3D<float> g_volumeTexture : register(u0);

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * Convert world position to voxel coordinates [0, volumeResolution)
 *
 * @param worldPos World-space position
 * @return Voxel coordinates (may be out of bounds)
 */
int3 WorldToVoxel(float3 worldPos) {
    float3 normalized = (worldPos - g_worldMin) / (g_worldMax - g_worldMin);
    return int3(normalized * float(g_volumeResolution));
}

/**
 * Convert voxel coordinates to world-space position (voxel center)
 *
 * @param voxelCoords Voxel coordinates [0, volumeResolution)
 * @return World-space position
 */
float3 VoxelToWorld(int3 voxelCoords) {
    float3 normalized = (float3(voxelCoords) + 0.5) / float(g_volumeResolution);
    return g_worldMin + normalized * (g_worldMax - g_worldMin);
}

/**
 * Check if voxel is within bounds
 *
 * @param voxelCoords Voxel coordinates
 * @return true if in bounds [0, volumeResolution)
 */
bool IsVoxelInBounds(int3 voxelCoords) {
    return all(voxelCoords >= 0) && all(voxelCoords < int(g_volumeResolution));
}

/**
 * Compute particle density contribution to voxel
 *
 * Uses Gaussian falloff based on distance from particle center.
 *
 * @param particle Particle data
 * @param voxelCenter World-space position of voxel center
 * @return Density contribution (extinction coefficient σ_t)
 */
float ComputeDensityContribution(Particle particle, float3 voxelCenter) {
    float3 offset = voxelCenter - particle.position;
    float distance = length(offset);

    // Gaussian falloff: density = exp(-distance² / radius²)
    float radius = particle.radius;
    float falloff = exp(-distance * distance / (radius * radius));

    // Scale by extinction coefficient
    // Higher temperature = more extinction (hotter particles are denser)
    float tempFactor = particle.temperature / 10000.0; // Normalize around 10000K
    float extinction = g_extinctionScale * tempFactor * falloff;

    return extinction;
}

//=============================================================================
// Main Compute Shader
//=============================================================================

/**
 * Splat particle density into volume texture
 *
 * Each thread processes one particle, writing to all overlapping voxels.
 * Uses atomic adds to handle overlapping contributions.
 */
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIdx = dispatchThreadID.x;

    // Bounds check
    if (particleIdx >= g_particleCount) {
        return;
    }

    Particle particle = g_particles[particleIdx];

    // Compute particle AABB in world space
    float3 particleMin = particle.position - particle.radius;
    float3 particleMax = particle.position + particle.radius;

    // Convert to voxel space
    int3 voxelMin = WorldToVoxel(particleMin);
    int3 voxelMax = WorldToVoxel(particleMax);

    // Clamp to volume bounds
    voxelMin = max(voxelMin, int3(0, 0, 0));
    voxelMax = min(voxelMax, int3(g_volumeResolution - 1, g_volumeResolution - 1, g_volumeResolution - 1));

    // CRITICAL: Limit AABB size to prevent GPU timeout at high particle counts
    // At 2K+ particles with large radii, nested loop can write to 100+ voxels per particle
    // = 200K+ voxel writes per frame → TDR crash
    // Solution: Clamp to max 8×8×8 voxels per particle (512 voxels max)
    int3 aabbSize = voxelMax - voxelMin + 1;
    const int MAX_VOXELS_PER_AXIS = 8;

    if (aabbSize.x > MAX_VOXELS_PER_AXIS) {
        int3 center = (voxelMin + voxelMax) / 2;
        voxelMin.x = center.x - MAX_VOXELS_PER_AXIS / 2;
        voxelMax.x = center.x + MAX_VOXELS_PER_AXIS / 2;
    }
    if (aabbSize.y > MAX_VOXELS_PER_AXIS) {
        int3 center = (voxelMin + voxelMax) / 2;
        voxelMin.y = center.y - MAX_VOXELS_PER_AXIS / 2;
        voxelMax.y = center.y + MAX_VOXELS_PER_AXIS / 2;
    }
    if (aabbSize.z > MAX_VOXELS_PER_AXIS) {
        int3 center = (voxelMin + voxelMax) / 2;
        voxelMin.z = center.z - MAX_VOXELS_PER_AXIS / 2;
        voxelMax.z = center.z + MAX_VOXELS_PER_AXIS / 2;
    }

    // Splat density to all overlapping voxels (max 8×8×8 = 512 voxels per particle)
    for (int z = voxelMin.z; z <= voxelMax.z; z++) {
        for (int y = voxelMin.y; y <= voxelMax.y; y++) {
            for (int x = voxelMin.x; x <= voxelMax.x; x++) {
                int3 voxelCoords = int3(x, y, z);

                if (!IsVoxelInBounds(voxelCoords)) {
                    continue;
                }

                // Compute voxel center in world space
                float3 voxelCenter = VoxelToWorld(voxelCoords);

                // Compute density contribution from this particle
                float density = ComputeDensityContribution(particle, voxelCenter);

                // Write density to volume texture
                // Phase 1 simplified approach: Later particles overwrite earlier ones
                // Volume is pre-cleared to zero, so all voxels start at 0.0
                // This gives us SOME density data (last particle wins per voxel)
                // For Phase 2+, use R32_FLOAT + InterlockedAdd for proper accumulation
                if (density > 0.0001) {  // Only write if density is meaningful
                    g_volumeTexture[voxelCoords] = density;
                }
            }
        }
    }
}
