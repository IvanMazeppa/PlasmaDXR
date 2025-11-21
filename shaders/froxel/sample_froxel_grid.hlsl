// Froxel Grid Sampling - Pass 3
// Samples pre-lit froxel grid during ray marching for volumetric fog
// This replaces the expensive god ray loop with cheap 3D texture samples

// This file is included in particle_gaussian_raytrace.hlsl
// It provides the RayMarchFroxelGrid() function to replace RayMarchAtmosphericFog()

// === REQUIRED RESOURCES (must be bound before including this file) ===
// Texture3D<float4> g_froxelLightingGrid : register(t10);  // Pre-lit voxel grid
// SamplerState g_linearClampSampler : register(s0);       // Trilinear sampler

// === REQUIRED CONSTANTS (must be in constant buffer) ===
// float3 froxelGridMin;        // World-space grid minimum
// float3 froxelGridMax;        // World-space grid maximum
// uint3 froxelGridDimensions;  // Grid dimensions [x, y, z]

/**
 * Sample froxel grid at world position with trilinear interpolation
 *
 * @param worldPos World-space position to sample
 * @return float4(RGB light color, A density)
 */
float4 SampleFroxelGrid(float3 worldPos)
{
    // Convert world position to grid coordinates [0, gridDimensions]
    float3 gridCoord = (worldPos - froxelGridMin) / (froxelGridMax - froxelGridMin);

    // Normalize to texture coordinates [0, 1]
    float3 texCoord = gridCoord;

    // Sample with hardware trilinear interpolation
    // This is MUCH faster than manual 8-sample trilinear!
    return g_froxelLightingGrid.SampleLevel(g_linearClampSampler, texCoord, 0);
}

/**
 * Ray march through froxel grid for volumetric fog
 * Replaces the expensive RayMarchAtmosphericFog() function
 *
 * @param cameraPos Camera world position
 * @param rayDir Ray direction (normalized)
 * @param maxDistance Maximum ray march distance
 * @param fogDensityMultiplier Global fog density control (0-1)
 * @return Accumulated fog color (RGB)
 */
float3 RayMarchFroxelGrid(
    float3 cameraPos,
    float3 rayDir,
    float maxDistance,
    float fogDensityMultiplier)
{
    // Early exit if fog disabled
    if (fogDensityMultiplier < 0.001) {
        return float3(0, 0, 0);
    }

    // Ray marching configuration
    const uint NUM_STEPS = 32;              // Same as god rays for fair comparison
    const float stepSize = maxDistance / float(NUM_STEPS);

    float3 totalFogColor = float3(0, 0, 0);

    // === RAY MARCH LOOP ===
    // This is now MUCH cheaper than god rays:
    //   God Rays: 32 steps × 13 lights × RayQuery = 416 ops/pixel
    //   Froxels:  32 steps × 1 texture sample = 32 ops/pixel
    //   Speedup:  13× faster!

    for (uint step = 0; step < NUM_STEPS; step++) {
        // Sample position along ray (at step center)
        float t = (float(step) + 0.5) * stepSize;
        float3 samplePos = cameraPos + rayDir * t;

        // === SAMPLE PRE-LIT FROXEL GRID ===
        // This single texture sample replaces:
        //   - 13 light evaluations
        //   - 13 distance calculations
        //   - 13 attenuation calculations
        //   - 13 shadow rays
        // All pre-computed in Pass 2!

        float4 froxelData = SampleFroxelGrid(samplePos);

        // Extract lighting and density
        float3 lightColor = froxelData.rgb;  // Pre-multiplied by density in Pass 2
        float density = froxelData.a;

        // Early skip if no density (outside particle field)
        if (density < 0.001) {
            continue;
        }

        // Accumulate fog contribution (volumetric integral)
        // lightColor is already pre-multiplied by density in Pass 2
        // So we just need to apply the global fog multiplier and step size
        float3 fogContribution = lightColor * fogDensityMultiplier * stepSize;

        totalFogColor += fogContribution;
    }

    return totalFogColor;
}

/**
 * Debug visualization: Render froxel grid density as colored overlay
 * Useful for verifying density injection is working correctly
 *
 * @param worldPos World position to visualize
 * @return Debug color (red = high density, blue = low density)
 */
float3 DebugVisualizeFroxelDensity(float3 worldPos)
{
    float4 froxelData = SampleFroxelGrid(worldPos);
    float density = froxelData.a;

    // Color ramp: Blue (low) → Cyan → Green → Yellow → Red (high)
    float3 color = float3(0, 0, 0);

    if (density > 0.001) {
        // Normalize density to [0, 1] range (assuming max density ~2.0)
        float normalizedDensity = saturate(density / 2.0);

        // Simple heat map
        if (normalizedDensity < 0.25) {
            // Blue → Cyan
            float t = normalizedDensity / 0.25;
            color = lerp(float3(0, 0, 1), float3(0, 1, 1), t);
        }
        else if (normalizedDensity < 0.5) {
            // Cyan → Green
            float t = (normalizedDensity - 0.25) / 0.25;
            color = lerp(float3(0, 1, 1), float3(0, 1, 0), t);
        }
        else if (normalizedDensity < 0.75) {
            // Green → Yellow
            float t = (normalizedDensity - 0.5) / 0.25;
            color = lerp(float3(0, 1, 0), float3(1, 1, 0), t);
        }
        else {
            // Yellow → Red
            float t = (normalizedDensity - 0.75) / 0.25;
            color = lerp(float3(1, 1, 0), float3(1, 0, 0), t);
        }
    }

    return color;
}

/**
 * Debug visualization: Render froxel grid lighting as colored overlay
 * Useful for verifying voxel lighting is working correctly
 *
 * @param worldPos World position to visualize
 * @return Debug color (shows accumulated lighting)
 */
float3 DebugVisualizeFroxelLighting(float3 worldPos)
{
    float4 froxelData = SampleFroxelGrid(worldPos);
    float3 lightColor = froxelData.rgb;

    // Normalize for visibility (boost by 10× for debug)
    return lightColor * 10.0;
}
