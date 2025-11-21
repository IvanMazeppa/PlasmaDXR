// Froxel Density Injection - Pass 1
// Converts particle positions to volumetric density field
// Each particle writes its density to the nearest voxel(s)

// Particle structure (matches C++ Particle)
struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
    float lifetime;
    uint materialType;
    float2 padding;
};

// Froxel grid parameters
cbuffer FroxelParams : register(b0)
{
    float3 gridMin;              // World-space minimum [-1500, -1500, -1500]
    float padding0;
    float3 gridMax;              // World-space maximum [1500, 1500, 1500]
    float padding1;
    uint3 gridDimensions;        // Grid size [160, 90, 64] = 921,600 voxels
    uint particleCount;
    float3 voxelSize;            // Computed: (gridMax - gridMin) / gridDimensions
    float densityMultiplier;     // Global density scale (default 1.0)
};

// Input: Particle buffer
StructuredBuffer<Particle> g_particles : register(t0);

// Output: 3D density grid (R16_FLOAT)
// NOTE: Using RWTexture3D instead of atomic operations for better performance
// We'll use additive blending in the shader for overlapping contributions
RWTexture3D<float> g_densityGrid : register(u0);

//------------------------------------------------------------------------------
// Density Injection Compute Shader
// Thread group: 256 particles per group (optimal for compute)
//------------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint particleIdx = threadID.x;

    // Bounds check
    if (particleIdx >= particleCount)
        return;

    // Read particle
    Particle p = g_particles[particleIdx];

    // Convert particle world position to grid coordinates
    float3 gridCoordFloat = (p.position - gridMin) / voxelSize;

    // Clamp to grid bounds (safety)
    gridCoordFloat = clamp(gridCoordFloat,
                           float3(0, 0, 0),
                           float3(gridDimensions) - 1.0);

    // Get integer voxel index (center voxel)
    int3 voxelIdx = int3(gridCoordFloat);

    // === TRILINEAR SPLATTING ===
    // Instead of dumping all density into one voxel, distribute it across
    // the 8 neighboring voxels using trilinear weights for smoother density field

    // Fractional part (for interpolation weights)
    float3 frac = gridCoordFloat - float3(voxelIdx);

    // Particle contribution (scaled by density and multiplier)
    float contribution = p.density * densityMultiplier * 0.01;

    // Distribute to 8 neighboring voxels (trilinear splatting)
    // This creates a smoother density field than single-voxel injection

    // Voxel (0,0,0) - base voxel
    float w000 = (1.0 - frac.x) * (1.0 - frac.y) * (1.0 - frac.z);
    g_densityGrid[voxelIdx + int3(0, 0, 0)] += contribution * w000;

    // Voxel (1,0,0)
    if (voxelIdx.x + 1 < int(gridDimensions.x)) {
        float w100 = frac.x * (1.0 - frac.y) * (1.0 - frac.z);
        g_densityGrid[voxelIdx + int3(1, 0, 0)] += contribution * w100;
    }

    // Voxel (0,1,0)
    if (voxelIdx.y + 1 < int(gridDimensions.y)) {
        float w010 = (1.0 - frac.x) * frac.y * (1.0 - frac.z);
        g_densityGrid[voxelIdx + int3(0, 1, 0)] += contribution * w010;
    }

    // Voxel (1,1,0)
    if (voxelIdx.x + 1 < int(gridDimensions.x) && voxelIdx.y + 1 < int(gridDimensions.y)) {
        float w110 = frac.x * frac.y * (1.0 - frac.z);
        g_densityGrid[voxelIdx + int3(1, 1, 0)] += contribution * w110;
    }

    // Voxel (0,0,1)
    if (voxelIdx.z + 1 < int(gridDimensions.z)) {
        float w001 = (1.0 - frac.x) * (1.0 - frac.y) * frac.z;
        g_densityGrid[voxelIdx + int3(0, 0, 1)] += contribution * w001;
    }

    // Voxel (1,0,1)
    if (voxelIdx.x + 1 < int(gridDimensions.x) && voxelIdx.z + 1 < int(gridDimensions.z)) {
        float w101 = frac.x * (1.0 - frac.y) * frac.z;
        g_densityGrid[voxelIdx + int3(1, 0, 1)] += contribution * w101;
    }

    // Voxel (0,1,1)
    if (voxelIdx.y + 1 < int(gridDimensions.y) && voxelIdx.z + 1 < int(gridDimensions.z)) {
        float w011 = (1.0 - frac.x) * frac.y * frac.z;
        g_densityGrid[voxelIdx + int3(0, 1, 1)] += contribution * w011;
    }

    // Voxel (1,1,1)
    if (voxelIdx.x + 1 < int(gridDimensions.x) &&
        voxelIdx.y + 1 < int(gridDimensions.y) &&
        voxelIdx.z + 1 < int(gridDimensions.z)) {
        float w111 = frac.x * frac.y * frac.z;
        g_densityGrid[voxelIdx + int3(1, 1, 1)] += contribution * w111;
    }
}
