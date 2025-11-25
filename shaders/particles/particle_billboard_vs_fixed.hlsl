// Particle Billboard Vertex Shader - FIXED VERSION
// Corrects the vertex ordering issue causing diagonal rendering
// Reads particle data and generates camera-facing billboard quads
// Applies RT lighting from lighting buffer

cbuffer CameraConstants : register(b0)
{
    float4x4 viewProj;
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
    float3 padding3;
};

// Input: Particle data
// Phase 2: Extended to 64 bytes for lifetime/pyro support
struct Particle
{
    // === LEGACY FIELDS (32 bytes) ===
    float3 position;       // 12 bytes (offset 0)
    float temperature;     // 4 bytes  (offset 12)
    float3 velocity;       // 12 bytes (offset 16)
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

    // Temperature-based color mapping
    // Temperature: ~10000 / radius, ranges from ~33 (outer r=300) to ~1000 (inner r=10)
    float tempNorm = saturate((p.temperature - 33.0) / 967.0);

    // Enhanced color gradient: Black -> Red -> Orange -> Yellow -> White
    float3 baseColor;
    if (tempNorm < 0.25) {
        // Black to Red
        float t = tempNorm / 0.25;
        baseColor = lerp(float3(0.1, 0.0, 0.0), float3(1.0, 0.0, 0.0), t);
    } else if (tempNorm < 0.5) {
        // Red to Orange
        float t = (tempNorm - 0.25) / 0.25;
        baseColor = lerp(float3(1.0, 0.0, 0.0), float3(1.0, 0.5, 0.0), t);
    } else if (tempNorm < 0.75) {
        // Orange to Yellow
        float t = (tempNorm - 0.5) / 0.25;
        baseColor = lerp(float3(1.0, 0.5, 0.0), float3(1.0, 1.0, 0.0), t);
    } else {
        // Yellow to White
        float t = (tempNorm - 0.75) / 0.25;
        baseColor = lerp(float3(1.0, 1.0, 0.0), float3(1.0, 1.0, 1.0), t);
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

    // Scale by particle radius (with temperature-based size variation)
    float tempScale = 1.0 + tempNorm * 0.3;  // Hotter particles are slightly larger
    float scaledRadius = particleRadius * tempScale;
    cornerOffset *= scaledRadius;

    // Billboard world position (camera-facing)
    float3 worldPos = p.position + cornerOffset.x * cameraRight + cornerOffset.y * cameraUp;

    // Transform to clip space (matrix is transposed on CPU, so vector goes first)
    float4 clipPos = mul(float4(worldPos, 1.0), viewProj);

    // Calculate alpha based on density (for volumetric look)
    // Enhance alpha for better visibility
    float alpha = saturate(p.density * 0.8 + 0.2);  // Minimum 0.2 alpha

    // Output
    PixelInput output;
    output.position = clipPos;
    output.texCoord = texCoord;
    output.color = float4(baseColor, 1.0);
    output.lighting = rtLight;  // Pass RT lighting to pixel shader
    output.alpha = alpha;

    return output;
}