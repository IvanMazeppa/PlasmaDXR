// DXR 1.1 RT Lighting Shader for Particles
// PRIMARY GOAL: Output GREEN (0, 100, 0) to prove RT lighting works

// Ray payload for lighting calculations
struct RayPayload {
    float3 color;
    float shadow;
};

// Particle data
struct Particle {
    float3 position;
    float mass;
    float3 velocity;
    float temperature;
    float4 color;
};

// RT constants
cbuffer RTConstants : register(b0) {
    float4x4 g_viewProj;
    float3 g_cameraPos;
    float g_time;
    float3 g_lightDir;
    float g_lightIntensity;
    uint g_particleCount;
    uint g_frameIndex;
    uint g_enableShadows;
    float g_pad;
};

// Resources
RaytracingAccelerationStructure g_rtScene : register(t0);
StructuredBuffer<Particle> g_particles : register(t1);
RWStructuredBuffer<float3> g_rtLighting : register(u0);

[shader("raygeneration")]
void RayGen() {
    uint particleIndex = DispatchRaysIndex().x;

    if (particleIndex >= g_particleCount) {
        return;
    }

    // TEST: Output pure GREEN to verify RT lighting is working
    // If particles turn green, we know:
    // 1. RT pipeline is working
    // 2. Lighting buffer is being written
    // 3. Mesh shaders are reading the lighting (or compute fallback is working)

    float3 testColor = float3(0.0f, 100.0f, 0.0f);  // BRIGHT GREEN

    // For actual lighting, we would:
    // 1. Read particle position
    // 2. Cast rays to light sources
    // 3. Cast shadow rays
    // 4. Calculate particle-to-particle scattering

    // But for now, just output GREEN to prove the pipeline works
    g_rtLighting[particleIndex] = testColor;

    // Optional: Add some variation based on particle index for debugging
    if (particleIndex % 1000 == 0) {
        // Every 1000th particle gets RED instead (for visibility)
        g_rtLighting[particleIndex] = float3(100.0f, 0.0f, 0.0f);
    }
}

[shader("miss")]
void Miss(inout RayPayload payload) {
    // Sky/environment contribution
    payload.color = float3(0.1f, 0.1f, 0.2f);  // Dark blue space
    payload.shadow = 0.0f;  // No shadow
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // Particle hit - would calculate lighting here
    // For now, just return a color
    payload.color = float3(1.0f, 0.8f, 0.3f);  // Warm particle color
    payload.shadow = 1.0f;  // In shadow
}

// Once GREEN test works, implement real lighting:
// 1. Volumetric scattering within accretion disk
// 2. Blackbody radiation based on temperature
// 3. Gravitational lensing near black hole
// 4. Particle self-shadowing
// 5. Doppler shift for rotating disk