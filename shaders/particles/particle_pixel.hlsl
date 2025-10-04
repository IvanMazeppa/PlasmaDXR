// DirectX Pixel Shader for Particle Rendering
// Simple alpha-blended particles with circular falloff + Mode 9.2 particle-to-particle lighting

struct PixelInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float3 color : COLOR0;
    float alpha : COLOR1;
    float3 worldPos : TEXCOORD1;
    float temperature : TEXCOORD2;
    float3 lighting : TEXCOORD3;  // Mode 9.2 particle-to-particle lighting
};

struct PixelOutput {
    float4 color : SV_Target0;
    float4 emission : SV_Target1;  // Mode 9.2 emission buffer
};

PixelOutput main(PixelInput input) {
    PixelOutput output;

    // Create circular particle shape with soft edges
    float2 centeredUV = input.texCoord * 2.0 - 1.0; // Convert [0,1] to [-1,1]
    float distance = length(centeredUV);

    // Smooth circular falloff
    float falloff = 1.0 - smoothstep(0.7, 1.0, distance);

    // Base particle color from temperature (ambient illumination from blackbody radiation)
    float3 baseColor = input.color;

    // Mode 9.2: Add particle-to-particle lighting contribution
    // Lighting is pre-scaled by particle_lighting.hlsl, just add it to base color
    float3 finalColor = baseColor + input.lighting;

    // Apply alpha from mesh shader and circular falloff
    float finalAlpha = input.alpha * falloff;

    // Output final color
    output.color = float4(finalColor, finalAlpha);

    // Mode 9.2: Emission buffer (hot particles emit light for spatial grid)
    // Boost emission for hotter particles (>10000K threshold for visible light contribution)
    float emissionStrength = saturate((input.temperature - 10000.0) / 16000.0);
    emissionStrength = pow(emissionStrength, 1.5) * 3.0;  // Exponential falloff + scale boost
    output.emission = float4(baseColor * emissionStrength, emissionStrength);

    return output;
}
