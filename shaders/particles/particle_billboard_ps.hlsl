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

    // Combine base temperature color with RT lighting
    // Base color = self-emission (temperature glow)
    // RT lighting = light received from nearby hot particles
    float3 baseColor = input.color.rgb * 0.3;  // Dim the base color (self-emission)
    float3 rtLighting = input.lighting.rgb;     // RT lighting from neighbors

    // Add them together - particles glow from their own heat + light from neighbors
    float3 finalColor = baseColor + rtLighting;

    output.color = float4(finalColor, 1.0);

    return output;
}