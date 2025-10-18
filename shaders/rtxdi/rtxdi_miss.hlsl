// RTXDI Miss Shader
// Milestone 3: DXR Pipeline Infrastructure
//
// Purpose: Handle rays that don't hit any geometry
//
// Note: For Milestone 3, we're not tracing rays to geometry yet
// (just sampling the light grid). This shader is here for completeness
// and will be used in future milestones for visibility testing.

// Ray payload (16 bytes)
struct RayPayload {
    float3 debugColor;  // Debug visualization
    uint selectedLight; // Index of selected light
};

[shader("miss")]
void Miss(inout RayPayload payload) {
    // No geometry hit - output black
    payload.debugColor = float3(0.0, 0.0, 0.0);
    payload.selectedLight = 0xFFFFFFFF; // Invalid light index
}
