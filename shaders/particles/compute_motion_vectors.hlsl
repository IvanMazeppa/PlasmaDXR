// Motion vector compute shader
// Converts particle world-space velocities to screen-space motion vectors
// Used by DLSS Ray Reconstruction for temporal denoising

// Phase 2: Extended to 64 bytes for lifetime/pyro support
struct Particle {
    // === LEGACY FIELDS (32 bytes) ===
    float3 position;       // 12 bytes (offset 0) - World position
    float temperature;     // 4 bytes  (offset 12)
    float3 velocity;       // 12 bytes (offset 16) - World velocity (units/second)
    float density;         // 4 bytes  (offset 28)

    // === MATERIAL FIELDS (16 bytes) ===
    float3 albedo;         // 12 bytes (offset 32)
    uint materialType;     // 4 bytes  (offset 44)

    // === LIFETIME FIELDS (16 bytes) ===
    float lifetime;        // 4 bytes  (offset 48)
    float maxLifetime;     // 4 bytes  (offset 52)
    float spawnTime;       // 4 bytes  (offset 56)
    uint flags;            // 4 bytes  (offset 60)
};  // Total: 64 bytes

StructuredBuffer<Particle> g_particles : register(t0);
RWTexture2D<float2> g_motionVectors : register(u0);  // Output MV buffer

cbuffer MotionVectorConstants : register(b0) {
    row_major float4x4 viewProj;
    row_major float4x4 prevViewProj;
    float3 cameraPos;
    float deltaTime;
    uint screenWidth;
    uint screenHeight;
    uint particleCount;
    float padding;
};

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // Get pixel coordinates
    uint2 pixelCoord = DTid.xy;
    if (pixelCoord.x >= screenWidth || pixelCoord.y >= screenHeight)
        return;

    // Convert pixel to NDC
    float2 uv = (float2(pixelCoord) + 0.5f) / float2(screenWidth, screenHeight);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    // Find closest particle to this pixel (simple approach)
    // For better quality, could ray march and find first hit
    float minDist = 1e10;
    float3 closestVelocity = float3(0, 0, 0);
    float3 closestPos = float3(0, 0, 0);

    // OPTIMIZATION: Could use spatial acceleration here
    for (uint i = 0; i < particleCount; i++) {
        Particle p = g_particles[i];

        // Project particle to screen
        float4 clipPos = mul(viewProj, float4(p.position, 1.0f));
        clipPos /= clipPos.w;

        float2 screenPos = float2(clipPos.x, -clipPos.y);
        float dist = length(screenPos - ndc);

        if (dist < minDist) {
            minDist = dist;
            closestVelocity = p.velocity;
            closestPos = p.position;
        }
    }

    // Compute motion vector from velocity
    // Current position
    float4 currClip = mul(viewProj, float4(closestPos, 1.0f));
    currClip /= currClip.w;

    // Previous position (position - velocity * deltaTime)
    float3 prevPos = closestPos - closestVelocity * deltaTime;
    float4 prevClip = mul(prevViewProj, float4(prevPos, 1.0f));
    prevClip /= prevClip.w;

    // Motion vector in screen space (pixels)
    float2 motionVec = (currClip.xy - prevClip.xy) * float2(screenWidth, screenHeight) * 0.5f;

    // Write to output
    g_motionVectors[pixelCoord] = motionVec;
}
