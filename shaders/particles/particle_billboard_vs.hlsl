// Particle Billboard Vertex Shader
// Reads particle data and generates camera-facing billboard quads
// Applies RT lighting from lighting buffer

#include "plasma_emission.hlsl"

cbuffer CameraConstants : register(b0)
{
    row_major float4x4 viewProj;
    float3 cameraPos;
    float padding0;
    float3 cameraRight;
    float padding1;
    float3 cameraUp;
    float padding2;
};

cbuffer ParticleConstants : register(b1)
{
    float particleRadius;
    uint usePhysicalEmission;
    uint useDopplerShift;
    uint useGravitationalRedshift;
    float emissionStrength;
    float dopplerStrength;
    float redshiftStrength;
    float padding;
};

// Input: Particle data
struct Particle
{
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

StructuredBuffer<Particle> g_particles : register(t0);

// Input: RT lighting data (from RayQuery compute shader)
StructuredBuffer<float4> g_rtLighting : register(t1);

// Output to pixel shader
struct PixelInput
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;        // Particle base color
    float4 lighting : COLOR1;     // RT lighting contribution
    float alpha : COLOR2;
};

// Vertex shader entry point
// Uses SV_InstanceID for particle index and SV_VertexID for vertex within billboard
PixelInput main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    // Particle index comes from instance ID (one instance per particle)
    uint particleIdx = instanceID;

    // Vertex index is just the vertexID (0-5 for the 6 vertices of a quad)
    uint vertIdx = vertexID;

    // FIXED: Correct mapping for two counter-clockwise triangles
    // Triangle 0: vertices 0,1,2 -> corners BL,BR,TR (CCW)
    // Triangle 1: vertices 3,4,5 -> corners BL,TR,TL (CCW)
    uint cornerIdx;
    if (vertIdx == 0) cornerIdx = 0;      // BL (first triangle)
    else if (vertIdx == 1) cornerIdx = 1; // BR (first triangle)
    else if (vertIdx == 2) cornerIdx = 3; // TR (first triangle)
    else if (vertIdx == 3) cornerIdx = 0; // BL (second triangle)
    else if (vertIdx == 4) cornerIdx = 3; // TR (second triangle)
    else cornerIdx = 2;                    // TL (second triangle)

    // Read particle data
    Particle p = g_particles[particleIdx];

    // Read RT lighting for this particle
    float4 rtLight = g_rtLighting[particleIdx];

    // Compute particle color (with optional physical emission)
    float3 baseColor;

    if (usePhysicalEmission != 0) {
        // Physical blackbody emission (strength modulates intensity)
        baseColor = ComputePlasmaEmission(
            p.position,
            p.velocity,
            p.temperature,
            p.density,
            cameraPos
        );

        // Apply emission strength
        baseColor = lerp(float3(0.5, 0.5, 0.5), baseColor, emissionStrength);

        // Optional Doppler shift with strength
        if (useDopplerShift != 0) {
            float3 viewDir = normalize(cameraPos - p.position);
            baseColor = DopplerShift(baseColor, p.velocity, viewDir, dopplerStrength);
        }

        // Optional gravitational redshift with strength
        if (useGravitationalRedshift != 0) {
            float radius = length(p.position); // Distance from center
            const float schwarzschildRadius = 2.0; // Adjust based on black hole
            baseColor = GravitationalRedshift(baseColor, radius, schwarzschildRadius, redshiftStrength);
        }
    } else {
        // Standard temperature-based color (default)
        float t = saturate((p.temperature - 800.0) / 25200.0);

        if (t < 0.25) {
            float blend = t / 0.25;
            baseColor = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
        } else if (t < 0.5) {
            float blend = (t - 0.25) / 0.25;
            baseColor = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
        } else if (t < 0.75) {
            float blend = (t - 0.5) / 0.25;
            baseColor = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
        } else {
            float blend = (t - 0.75) / 0.25;
            baseColor = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
        }
    }

    // Generate billboard corner position in world space
    float2 cornerOffset;
    float2 texCoord;

    if (cornerIdx == 0) {  // Bottom-left
        cornerOffset = float2(-1, -1);
        texCoord = float2(0, 1);
    } else if (cornerIdx == 1) {  // Bottom-right
        cornerOffset = float2(1, -1);
        texCoord = float2(1, 1);
    } else if (cornerIdx == 2) {  // Top-left
        cornerOffset = float2(-1, 1);
        texCoord = float2(0, 0);
    } else {  // Top-right (cornerIdx == 3)
        cornerOffset = float2(1, 1);
        texCoord = float2(1, 0);
    }

    // CORRECT APPROACH: Calculate billboard basis PER-PARTICLE (like PlasmaDX mesh shader)
    // This ensures each particle faces the camera correctly based on its actual position
    float3 toCamera = cameraPos - p.position;
    float3 forward = normalize(toCamera);
    float3 right = normalize(cross(forward, float3(0, 1, 0)));  // Y-axis aligned
    float3 up = cross(right, forward);

    // Scale by particle radius
    right *= particleRadius;
    up *= particleRadius;

    // Build billboard corner in world space
    float3 worldPos = p.position + cornerOffset.x * right + cornerOffset.y * up;

    // Transform to clip space
    // DirectXMath is row-major, HLSL defaults to row-major
    // Use row-vector multiplication: v * M
    float4 clipPos = mul(float4(worldPos, 1.0), viewProj);

    // Calculate alpha based on density (for volumetric look)
    // BOOST alpha significantly - particles should be clearly visible
    float alpha = saturate(p.density * 5.0);  // 10x boost
    alpha = max(alpha, 0.8);  // Ensure minimum 80% opacity

    // Output
    PixelInput output;
    output.position = clipPos;
    output.texCoord = texCoord;
    output.color = float4(baseColor, 1.0);
    output.lighting = rtLight;  // Pass RT lighting to pixel shader
    output.alpha = alpha;

    return output;
}
