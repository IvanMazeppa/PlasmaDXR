// ===========================================================================================
// Depth Buffer Clear Compute Shader - Phase 2: Screen-Space Shadow System
// ===========================================================================================
//
// PURPOSE:
// Clears the depth buffer to far plane (1.0) so empty pixels don't cause false occlusions.
// Runs before the depth pre-pass to initialize the buffer properly.
//
// ===========================================================================================

cbuffer ClearConstants : register(b0) {
    uint g_width;
    uint g_height;
    uint2 g_padding;
};

// Output depth buffer (R32_UINT - we store float depth as uint for atomic operations)
RWTexture2D<uint> g_depthBuffer : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixelPos = dispatchThreadID.xy;

    // Bounds check
    if (pixelPos.x >= g_width || pixelPos.y >= g_height) {
        return;
    }

    // Initialize to far plane (1.0 as uint)
    // asuint(1.0f) = 0x3F800000
    g_depthBuffer[pixelPos] = 0x3F800000;
}
