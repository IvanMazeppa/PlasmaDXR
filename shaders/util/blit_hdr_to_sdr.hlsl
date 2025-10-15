// HDR to SDR Blit Shader
// Converts R16G16B16A16_FLOAT (HDR intermediate) to R8G8B8A8_UNORM (SDR swap chain)
// This shader runs as a fullscreen pass to copy Gaussian output to backbuffer

// Input: HDR texture from Gaussian renderer (already tone-mapped via ACES)
Texture2D<float4> g_hdrInput : register(t0);
SamplerState g_linearSampler : register(s0);

// Vertex shader output / Pixel shader input
struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// Fullscreen triangle vertex shader
// Generates 3 vertices covering entire screen without vertex buffer
VSOutput VSMain(uint vertexID : SV_VertexID) {
    VSOutput output;

    // Generate fullscreen triangle coordinates
    // vertexID 0: (-1, -1) -> (0, 1) UV
    // vertexID 1: (-1,  3) -> (0, -1) UV
    // vertexID 2: ( 3, -1) -> (2, 1) UV
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    output.uv = uv;

    return output;
}

// Pixel shader: Sample HDR and output to SDR
float4 PSMain(VSOutput input) : SV_Target {
    // Sample HDR texture (already tone-mapped by ACES in Gaussian shader)
    float4 hdrColor = g_hdrInput.SampleLevel(g_linearSampler, input.uv, 0);

    // HDR color is already in [0,1] range after ACES tone mapping
    // Just clamp to ensure no out-of-range values
    float3 sdrColor = saturate(hdrColor.rgb);

    // Output to 8-bit swap chain (alpha = 1.0)
    return float4(sdrColor, 1.0);
}
