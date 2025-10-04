// Compute shader that merges RT lighting into particle data
// This is the WORKAROUND for mesh shaders not being able to read descriptor tables
// We pre-bake the lighting into the particle buffer

struct Particle {
    float3 position;
    float mass;
    float3 velocity;
    float temperature;
    float4 color;      // xyz = base color, w = alpha
};

struct ParticleWithLighting {
    float3 position;
    float mass;
    float3 velocity;
    float temperature;
    float4 color;       // xyz = base color, w = alpha
    float3 rtLighting;  // RT lighting contribution
    float pad;
};

// Input buffers
StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<float3> g_rtLighting : register(t1);

// Output buffer
RWStructuredBuffer<ParticleWithLighting> g_particlesWithLighting : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint particleIndex = id.x;

    // Read particle data
    Particle p = g_particles[particleIndex];

    // Read RT lighting (this is what mesh shaders can't do on buggy drivers)
    float3 rtLight = g_rtLighting[particleIndex];

    // Create merged particle
    ParticleWithLighting merged;
    merged.position = p.position;
    merged.mass = p.mass;
    merged.velocity = p.velocity;
    merged.temperature = p.temperature;
    merged.color = p.color;
    merged.rtLighting = rtLight;
    merged.pad = 0.0f;

    // Write to output buffer
    g_particlesWithLighting[particleIndex] = merged;
}