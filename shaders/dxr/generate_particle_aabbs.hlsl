// DXR Per-Particle AABB Generation Compute Shader
// Generates individual AABBs for each particle for ray-traced particle-to-particle lighting
// Updates every frame as particles move

cbuffer AABBConstants : register(b0)
{
    uint particleCount;
    float particleRadius;  // Base particle radius for AABB sizing
    float2 padding;
};

// Input: Particle buffer (read particle positions)
struct Particle
{
    float3 position;    // Offset 0-11
    float temperature;  // Offset 12-15
    float3 velocity;    // Offset 16-27
    float density;      // Offset 28-31
};

StructuredBuffer<Particle> particles : register(t0);

// Output: Per-particle AABB buffer (D3D12_RAYTRACING_AABB format)
struct AABB
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

RWStructuredBuffer<AABB> particleAABBs : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIndex = dispatchThreadID.x;

    // Early exit if outside particle count
    if (particleIndex >= particleCount)
        return;

    // Read particle position
    Particle p = particles[particleIndex];
    float3 pos = p.position;

    // Generate AABB centered on particle position
    // Use fixed radius for now (can be made temperature-dependent later)
    AABB aabb;
    aabb.minX = pos.x - particleRadius;
    aabb.minY = pos.y - particleRadius;
    aabb.minZ = pos.z - particleRadius;
    aabb.maxX = pos.x + particleRadius;
    aabb.maxY = pos.y + particleRadius;
    aabb.maxZ = pos.z + particleRadius;

    // Write AABB to output buffer
    particleAABBs[particleIndex] = aabb;
}
