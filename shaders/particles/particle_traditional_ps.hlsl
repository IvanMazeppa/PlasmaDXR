// Simple pixel shader for compute-built particles
// Lighting already baked into vertex color

struct PixelInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;           // Base color + RT lighting (from compute)
    float alpha : COLOR1;
};

float4 main(PixelInput input) : SV_Target {
    // Circular particle shape with soft edges
    float2 centeredUV = input.texCoord * 2.0 - 1.0;
    float distance = length(centeredUV);
    float falloff = 1.0 - smoothstep(0.7, 1.0, distance);

    // Final color with circular falloff
    float finalAlpha = input.alpha * falloff;
    return float4(input.color.rgb, finalAlpha);
}
