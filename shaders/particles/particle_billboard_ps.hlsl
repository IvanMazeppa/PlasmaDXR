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

    // Combine base color with RT lighting
    float3 finalColor = input.color.rgb;

    // If we have RT lighting, use it (shader outputs GREEN if running but no hits)
    if (length(input.lighting.rgb) > 0.001) {
        finalColor = input.lighting.rgb;
    }

    output.color = float4(finalColor, 1.0);

    return output;
}