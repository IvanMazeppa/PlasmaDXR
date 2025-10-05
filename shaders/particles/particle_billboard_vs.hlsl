// Particle Billboard Vertex Shader
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
// Uses SV_VertexID to generate billboard vertices (6 vertices per particle = 2 triangles)
PixelInput main(uint vertexID : SV_VertexID)
{
    // Decode particle index and vertex index from vertex ID
    uint particleIdx = vertexID / 6;  // 6 vertices per particle (2 triangles)
    uint vertIdx = vertexID % 6;       // 0-5: vertex in quad

    // Map to corner: 0,1,2, 2,1,3 -> BL,BR,TL, TL,BR,TR
    uint cornerIdx;
    if (vertIdx == 0) cornerIdx = 0;      // BL
    else if (vertIdx == 1) cornerIdx = 1; // BR
    else if (vertIdx == 2) cornerIdx = 2; // TL
    else if (vertIdx == 3) cornerIdx = 2; // TL (second triangle)
    else if (vertIdx == 4) cornerIdx = 1; // BR (second triangle)
    else cornerIdx = 3;                    // TR

    // Read particle data
    Particle p = g_particles[particleIdx];

    // Read RT lighting for this particle
    float4 rtLight = g_rtLighting[particleIdx];

    // DEBUG: Output temperature as color to diagnose
    // Temperature: ~10000 / radius, ranges from ~33 (outer r=300) to ~1000 (inner r=10)
    // Map temperature 33-1000 to red->yellow gradient
    float tempNorm = saturate((p.temperature - 33.0) / 967.0);
    float3 baseColor = lerp(float3(1.0, 0.0, 0.0), float3(1.0, 1.0, 0.0), tempNorm);  // Red to Yellow

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

    // Scale by particle radius
    cornerOffset *= particleRadius;

    // Billboard world position (camera-facing)
    float3 worldPos = p.position + cornerOffset.x * cameraRight + cornerOffset.y * cameraUp;

    // Transform to clip space
    float4 clipPos = mul(viewProj, float4(worldPos, 1.0));

    // Calculate alpha based on density (for volumetric look)
    float alpha = saturate(p.density * 0.5);

    // Output
    PixelInput output;
    output.position = clipPos;
    output.texCoord = texCoord;
    output.color = float4(baseColor, 1.0);
    output.lighting = rtLight;  // Pass RT lighting to pixel shader
    output.alpha = alpha;

    return output;
}