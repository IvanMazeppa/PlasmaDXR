// DXR 1.2 Features HLSL Header
// Shader Execution Reordering (SER) and related DXR 1.2 features
// Requires Shader Model 6.6+ and DXR 1.2 support

#ifndef DXR12_FEATURES_HLSLI
#define DXR12_FEATURES_HLSLI

// Feature flags passed from CPU
cbuffer DXR12Features : register(b0) {
    uint g_EnableSER;              // 1 if SER is enabled, 0 otherwise
    uint g_SERCoherenceHint;       // Coherence hint for SER (0-3)
    uint g_EnableOMM;              // 1 if Opacity Micromaps enabled
    uint g_Reserved;               // Reserved for future features

    float g_SERPerformanceGain;    // Measured SER performance gain (for display)
    uint3 g_Padding;               // Align to 16-byte boundary
};

// SER (Shader Execution Reordering) helper functions
// These are placeholders - actual SER intrinsics will be available with SM 6.9

#if defined(DXR_1_2_AVAILABLE) && defined(SHADER_MODEL_6_9)

// ReorderThread intrinsic (SM 6.9 with DXR 1.2)
// This intrinsic allows shaders to provide coherence hints to the GPU
// The GPU can then reorder shader invocations to improve cache locality
void ReorderThreadWithHint(uint coherenceHint, uint hitIndex) {
    // Actual intrinsic call when available:
    // ReorderThread(coherenceHint, hitIndex);

    // For now, this is a no-op placeholder
    // When the real intrinsic is available, uncomment the line above
}

// SER coherence hint generation for volumetric rendering
uint GenerateVolumetricCoherenceHint(float3 rayOrigin, float3 rayDirection) {
    // Generate coherence hint based on ray spatial locality
    // This helps the GPU group rays that are likely to access similar data

    // Discretize ray origin into a coarse grid
    const float gridSize = 32.0f; // 32-unit grid cells
    int3 gridPos = int3(rayOrigin / gridSize);

    // Discretize ray direction into octants
    uint3 dirSign = uint3(rayDirection > 0.0f);
    uint octant = (dirSign.x << 2) | (dirSign.y << 1) | dirSign.z;

    // Combine spatial and directional coherence
    uint spatialHint = (gridPos.x & 0xFF) |
                      ((gridPos.y & 0xFF) << 8) |
                      ((gridPos.z & 0xFF) << 16);
    uint directionalHint = octant << 24;

    return spatialHint | directionalHint;
}

#else

// Fallback implementations for older hardware/SM versions
void ReorderThreadWithHint(uint coherenceHint, uint hitIndex) {
    // No-op for older hardware - SER is simply not applied
    // This ensures the shader compiles and runs without SER benefits
}

uint GenerateVolumetricCoherenceHint(float3 rayOrigin, float3 rayDirection) {
    return 0; // No coherence hint on older hardware
}

#endif // DXR_1_2_AVAILABLE && SHADER_MODEL_6_9

// HitObject helper functions (DXR 1.2 feature)
// HitObjects allow decoupling ray traversal from shading
// This enables better code reuse and more flexible ray tracing pipelines

#if defined(DXR_1_2_AVAILABLE)

// Placeholder HitObject structure
// This will be replaced with the actual HitObject type from SM 6.9
struct HitObject {
    uint packed[4]; // Placeholder - actual size TBD
};

// Trace ray and return HitObject for later shading
HitObject TraceRayToHitObject(RaytracingAccelerationStructure scene,
                             uint rayFlags,
                             uint instanceInclusionMask,
                             uint rayContributionToHitGroupIndex,
                             uint multiplierForGeometryContributionToShaderIndex,
                             uint missShaderIndex,
                             RayDesc ray) {
    HitObject hitObj = (HitObject)0;
    // TODO: Actual implementation when HitObject API is available
    return hitObj;
}

// Invoke shading on a HitObject
float4 InvokeHitObjectShading(HitObject hitObj, float3 rayDirection) {
    // TODO: Actual implementation when HitObject API is available
    return float4(0, 0, 0, 0);
}

#endif // DXR_1_2_AVAILABLE

// Performance measurement helpers
void RecordSERPerformanceStart() {
    // TODO: Add performance measurement when available
    // This could use timestamp queries or GPU timing
}

void RecordSERPerformanceEnd() {
    // TODO: Add performance measurement when available
}

// Utility macro to conditionally apply SER
#define APPLY_SER_IF_ENABLED(rayOrigin, rayDirection, hitIndex) \
    do { \
        if (g_EnableSER) { \
            uint coherenceHint = GenerateVolumetricCoherenceHint(rayOrigin, rayDirection); \
            ReorderThreadWithHint(coherenceHint, hitIndex); \
        } \
    } while(0)

// Volumetric rendering specific SER optimizations
namespace VolumetricSER {

    // SER hint for volume density sampling
    uint GenerateDensitySamplingHint(float3 samplePosition) {
        // Group rays sampling similar density regions
        const float voxelSize = 1.0f;
        int3 voxelPos = int3(samplePosition / voxelSize);
        return (voxelPos.x & 0x3FF) |
               ((voxelPos.y & 0x3FF) << 10) |
               ((voxelPos.z & 0x3FF) << 20);
    }

    // SER hint for light sampling
    uint GenerateLightSamplingHint(uint lightIndex, float3 lightPosition) {
        // Group rays interacting with the same light sources
        return (lightIndex & 0xFFFF) |
               (((uint)(lightPosition.x * 0.1f) & 0xFF) << 16) |
               (((uint)(lightPosition.y * 0.1f) & 0xFF) << 24);
    }

    // Combined hint for volumetric shading
    uint GenerateVolumetricShadingHint(float3 samplePos, uint lightIndex, float3 lightPos) {
        uint densityHint = GenerateDensitySamplingHint(samplePos);
        uint lightHint = GenerateLightSamplingHint(lightIndex, lightPos);

        // Combine hints with bias toward spatial coherence
        return (densityHint & 0xFFFFF) | ((lightHint & 0xFFF) << 20);
    }
}

#endif // DXR12_FEATURES_HLSLI