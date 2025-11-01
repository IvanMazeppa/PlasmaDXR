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
 * Output: R32_UINT 3D texture with extinction coefficient σ_t (stored as uint via asuint)
 *
 * IMPORTANT: Uses R32_UINT instead of R16_FLOAT to support atomic operations.
 * At >2044 particles, multiple threads write to the same voxel causing race
 * conditions. InterlockedMax prevents GPU hang at 2048 thread boundary.
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

// Output volume texture (R32_UINT for atomic operations)
// We use uint instead of float to support InterlockedMax for race-free writes
// Multiple particles can overlap the same voxel, causing race conditions
// at high particle counts (>2044). Atomics prevent GPU hang.
RWTexture3D<uint> g_volumeTexture : register(u0);

// Diagnostic counter buffer for debugging (4 uint32 counters)
// [0] = total threads executed
// [1] = early returns (bounds check failures)
// [2] = total voxel writes
// [3] = max voxels written by single particle
RWByteAddressBuffer g_diagnosticCounters : register(u1);

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
    uint dummy;

    // DIAGNOSTIC TEST: Write a sentinel value to prove shader is executing
    if (particleIdx == 0) {
        g_diagnosticCounters.Store(0, 0xDEADBEEF);  // Magic number to prove shader ran
    }
    GroupMemoryBarrierWithGroupSync();  // Ensure write completes

    // Count total threads executed
    g_diagnosticCounters.InterlockedAdd(0, 1, dummy);  // Offset 0 = counter[0]

    // Bounds check
    if (particleIdx >= g_particleCount) {
        // Count early returns
        g_diagnosticCounters.InterlockedAdd(4, 1, dummy);  // Offset 4 = counter[1]
        return;
    }

    Particle particle = g_particles[particleIdx];

    // Track voxel writes for this particle
    uint voxelWriteCount = 0;

    // Compute particle AABB in world space
    float3 particleMin = particle.position - particle.radius;
    float3 particleMax = particle.position + particle.radius;

    // Convert to voxel space
    int3 voxelMin = WorldToVoxel(particleMin);
    int3 voxelMax = WorldToVoxel(particleMax);

    // Clamp to volume bounds FIRST before any size calculations
    voxelMin = max(voxelMin, int3(0, 0, 0));
    voxelMax = min(voxelMax, int3(g_volumeResolution - 1, g_volumeResolution - 1, g_volumeResolution - 1));

    // CRITICAL FIX: Ensure voxelMin <= voxelMax after clamping
    // Edge case: If particle is completely outside volume, min > max after clamping
    // This creates invalid loop bounds → infinite loop → GPU hang
    if (any(voxelMin > voxelMax)) {
        return;  // Particle completely outside volume, skip it
    }

    // Limit AABB size to prevent excessive voxel writes
    // Max 8×8×8 = 512 voxels per particle to prevent GPU timeout
    const int MAX_VOXELS_PER_AXIS = 8;
    int3 aabbSize = voxelMax - voxelMin + 1;  // Now using CLAMPED coordinates

    if (aabbSize.x > MAX_VOXELS_PER_AXIS) {
        int center = (voxelMin.x + voxelMax.x) / 2;
        voxelMin.x = max(center - MAX_VOXELS_PER_AXIS / 2, 0);
        voxelMax.x = min(center + MAX_VOXELS_PER_AXIS / 2, g_volumeResolution - 1);
    }
    if (aabbSize.y > MAX_VOXELS_PER_AXIS) {
        int center = (voxelMin.y + voxelMax.y) / 2;
        voxelMin.y = max(center - MAX_VOXELS_PER_AXIS / 2, 0);
        voxelMax.y = min(center + MAX_VOXELS_PER_AXIS / 2, g_volumeResolution - 1);
    }
    if (aabbSize.z > MAX_VOXELS_PER_AXIS) {
        int center = (voxelMin.z + voxelMax.z) / 2;
        voxelMin.z = max(center - MAX_VOXELS_PER_AXIS / 2, 0);
        voxelMax.z = min(center + MAX_VOXELS_PER_AXIS / 2, g_volumeResolution - 1);
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

                // Write density to volume texture using atomic operations
                // CRITICAL FIX: Non-atomic writes cause race conditions at >2044 particles
                // Multiple threads writing to same voxel → GPU hang at 2048 thread boundary
                // Solution: Use InterlockedMax to ensure race-free writes (highest density wins)
                if (density > 0.0001) {  // Only write if density is meaningful
                    // Convert float density to uint using asuint (preserves bit pattern)
                    // This allows us to compare floats using integer atomic operations
                    uint densityAsUint = asuint(density);

                    // Atomic max: Highest density value wins per voxel
                    // This prevents race conditions while giving reasonable results
                    uint originalValue;
                    InterlockedMax(g_volumeTexture[voxelCoords], densityAsUint, originalValue);

                    // Count this voxel write
                    voxelWriteCount++;
                }
            }
        }
    }

    // Update diagnostic counters
    // Total voxel writes across all particles (offset 8 = counter[2])
    g_diagnosticCounters.InterlockedAdd(8, voxelWriteCount, dummy);

    // Track maximum voxels written by any single particle (offset 12 = counter[3])
    g_diagnosticCounters.InterlockedMax(12, voxelWriteCount, dummy);
}
