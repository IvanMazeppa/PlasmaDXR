// Minimal Gaussian renderer test - just output solid color

cbuffer Constants : register(b0) {
    // Match the 60 DWORDs structure from RenderConstants
    float4x4 viewProj;          // 16 floats
    float4x4 invViewProj;        // 16 floats
    float3 cameraPos;            // 3 floats
    float particleRadius;        // 1 float
    float3 cameraRight;          // 3 floats
    float time;                  // 1 float
    float3 cameraUp;             // 3 floats
    uint screenWidth;            // 1 uint
    float3 cameraForward;        // 3 floats
    uint screenHeight;           // 1 uint
    float fovY;                  // 1 float
    float aspectRatio;           // 1 float
    uint particleCount;          // 1 uint
    float padding;               // 1 float

    // Physical emission toggles and strengths
    uint usePhysicalEmission;    // 1 uint
    float emissionStrength;      // 1 float
    uint useDopplerShift;        // 1 uint
    float dopplerStrength;       // 1 float
    uint useGravitationalRedshift; // 1 uint
    float redshiftStrength;      // 1 float
    float2 padding2;             // 2 floats
};

StructuredBuffer<float> dummy1 : register(t0);
Buffer<float> dummy2 : register(t1);
RaytracingAccelerationStructure dummy3 : register(t2);
RWTexture2D<float4> g_output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x >= screenWidth || dispatchThreadID.y >= screenHeight) {
        return;
    }

    // Just output red to verify pipeline works
    g_output[dispatchThreadID.xy] = float4(1, 0, 0, 1);
}
