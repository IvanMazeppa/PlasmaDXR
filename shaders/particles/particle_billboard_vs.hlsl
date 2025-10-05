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
// Uses SV_InstanceID for particle index and SV_VertexID for vertex within billboard
PixelInput main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    // DEBUG: First instance renders a test triangle to verify rendering works
    if (instanceID == 0 && vertexID < 3) {
        PixelInput output;
        if (vertexID == 0) output.position = float4(-0.5, -0.5, 0.5, 1.0);
        else if (vertexID == 1) output.position = float4(0.5, -0.5, 0.5, 1.0);
        else output.position = float4(0.0, 0.5, 0.5, 1.0);
        output.texCoord = float2(0, 0);
        output.color = float4(1.0, 0.0, 1.0, 1.0);  // MAGENTA test triangle
        output.lighting = float4(1, 1, 1, 1);
        output.alpha = 1.0;
        return output;
    }

    // Particle index comes from instance ID (one instance per particle)
    uint particleIdx = instanceID;

    // Vertex index is just the vertexID (0-5 for the 6 vertices of a quad)
    uint vertIdx = vertexID;

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

    // Transform to clip space (matrix is transposed on CPU, so vector goes first)
    float4 clipPos = mul(float4(worldPos, 1.0), viewProj);

    // Calculate alpha based on density (for volumetric look)
    float alpha = saturate(p.density * 0.5);

    // DEBUG: Force instance 0 (all 6 vertices) to render as visible quad
    if (instanceID == 0) {
        // Render a small quad at screen center (slightly offset from magenta triangle)
        float2 quadOffset = cornerOffset * 0.3;  // Smaller than test triangle
        clipPos = float4(quadOffset.x + 0.3, quadOffset.y, 0.5, 1.0);  // Offset right
        baseColor = float3(0, 1, 0);  // Bright green
        alpha = 1.0;  // Full opacity
    }

    // Output
    PixelInput output;
    output.position = clipPos;
    output.texCoord = texCoord;
    output.color = float4(baseColor, 1.0);
    output.lighting = rtLight;  // Pass RT lighting to pixel shader
    output.alpha = alpha;

    return output;
}