// Particle Billboard Vertex Shader - DEBUG VERSION
// Hardcoded positions to verify triangle formation
// This shader renders a single test quad at the center of the screen

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

struct Particle
{
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<float4> g_rtLighting : register(t1);

struct PixelInput
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
    float4 lighting : COLOR1;
    float alpha : COLOR2;
};

PixelInput main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    PixelInput output;

    // Only render first instance for debugging
    if (instanceID != 0) {
        output.position = float4(0, 0, 0, 0);  // Degenerate position (culled)
        output.texCoord = float2(0, 0);
        output.color = float4(0, 0, 0, 0);
        output.lighting = float4(0, 0, 0, 0);
        output.alpha = 0;
        return output;
    }

    // HARDCODED CLIP SPACE POSITIONS
    // These form a square in the center of the screen
    // Size: 0.5 in clip space (quarter of the screen)
    float4 positions[6] = {
        // Triangle 0: BL -> BR -> TR (CCW)
        float4(-0.25, -0.25, 0.5, 1.0),  // 0: BL
        float4( 0.25, -0.25, 0.5, 1.0),  // 1: BR
        float4( 0.25,  0.25, 0.5, 1.0),  // 2: TR

        // Triangle 1: BL -> TR -> TL (CCW)
        float4(-0.25, -0.25, 0.5, 1.0),  // 3: BL
        float4( 0.25,  0.25, 0.5, 1.0),  // 4: TR
        float4(-0.25,  0.25, 0.5, 1.0)   // 5: TL
    };

    // Texture coordinates for each vertex
    float2 texCoords[6] = {
        float2(0, 1),  // 0: BL
        float2(1, 1),  // 1: BR
        float2(1, 0),  // 2: TR
        float2(0, 1),  // 3: BL
        float2(1, 0),  // 4: TR
        float2(0, 0)   // 5: TL
    };

    // Different colors for each vertex to visualize triangle formation
    float4 colors[6] = {
        float4(1, 0, 0, 1),    // 0: Red (BL)
        float4(0, 1, 0, 1),    // 1: Green (BR)
        float4(0, 0, 1, 1),    // 2: Blue (TR)
        float4(1, 0, 0, 1),    // 3: Red (BL)
        float4(0, 0, 1, 1),    // 4: Blue (TR)
        float4(1, 1, 0, 1)     // 5: Yellow (TL)
    };

    // Output hardcoded values
    output.position = positions[vertexID];
    output.texCoord = texCoords[vertexID];
    output.color = colors[vertexID];
    output.lighting = float4(0, 0, 0, 0);
    output.alpha = 1.0;

    return output;
}