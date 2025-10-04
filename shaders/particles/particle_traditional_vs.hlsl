// Simple pass-through vertex shader
// Vertices are pre-transformed by compute shader

struct VertexInput {
    float4 position : POSITION;      // Already in clip space (from compute shader)
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;           // Color + lighting pre-multiplied
    float alpha : COLOR1;
};

struct PixelInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
    float alpha : COLOR1;
};

PixelInput main(VertexInput input) {
    PixelInput output;
    output.position = input.position;  // Pass-through (already transformed)
    output.texCoord = input.texCoord;
    output.color = input.color;
    output.alpha = input.alpha;
    return output;
}
