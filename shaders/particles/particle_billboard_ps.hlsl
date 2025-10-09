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

    // Create spherical particle shape using texture coordinates (from PlasmaDX)
    float2 center = input.texCoord - 0.5;
    float distFromCenter = length(center) * 2.0; // Scale to 0-1 range

    // Discard pixels outside circle for hard edge
    if (distFromCenter > 1.0) discard;

    // Simulate 3D sphere lighting with sqrt falloff (like a real sphere)
    float sphereZ = sqrt(max(0.0, 1.0 - distFromCenter * distFromCenter));

    // Smooth edge fadeout
    float edgeFade = 1.0 - smoothstep(0.8, 1.0, distFromCenter);

    // Combine sphere lighting with edge fade
    float intensity = sphereZ * edgeFade;

    // Base color from temperature
    float3 color = input.color.rgb;

    // Add bright center (hot core) - makes particle look 3D
    float hotSpot = pow(1.0 - distFromCenter, 3.0);
    color = lerp(color, color * 1.5, hotSpot * 0.5);

    // Apply sphere shading (makes it look round)
    color *= (0.6 + intensity * 0.8);

    // ADD RT lighting from neighbors
    color += input.lighting.rgb;

    // Alpha based on intensity and input alpha
    float alpha = intensity * input.alpha;

    output.color = float4(color, alpha);

    return output;
}