// Particle Billboard Pixel Shader
// Renders circular billboard with RT lighting applied

// Input from vertex shader
struct PixelInput
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;        // Particle base color (temperature-based)
    float4 lighting : COLOR1;     // RT lighting contribution
    float alpha : COLOR2;
};

// Output
struct PixelOutput
{
    float4 color : SV_Target0;
};

PixelOutput main(PixelInput input)
{
    PixelOutput output;

    // Create circular shape (discard pixels outside circle)
    float2 coord = input.texCoord * 2.0 - 1.0;  // -1 to 1
    float distFromCenter = length(coord);

    if (distFromCenter > 1.0) {
        discard;  // Outside circle
    }

    // Soft edge falloff for antialiasing
    float edgeFalloff = 1.0 - smoothstep(0.8, 1.0, distFromCenter);

    // Combine base color with RT lighting
    // Base color is the particle's emission (temperature-based)
    // RT lighting is the light received from other particles
    float3 finalColor = input.color.rgb + input.lighting.rgb;

    // Apply alpha with edge falloff
    float finalAlpha = input.alpha * edgeFalloff;

    output.color = float4(finalColor, finalAlpha);

    return output;
}