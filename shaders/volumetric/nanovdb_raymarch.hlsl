// NanoVDB Volumetric Ray Marching Shader - SIMPLIFIED PROTOTYPE
// Renders sparse volumetric fog using procedural density for prototype testing
//
// NOTE: Full PNanoVDB.h integration deferred due to DXC compatibility issues.
// This prototype uses procedural density to validate the rendering pipeline.
// Once working, we'll integrate proper NanoVDB grid sampling.
//
// Based on:
// - NVIDIA NanoVDB (Museth, SIGGRAPH 2021)
// - Ray Tracing Gems II, Chapter 43

// ============================================================================
// CONSTANT BUFFER
// ============================================================================

cbuffer NanoVDBConstants : register(b0) {
    row_major float4x4 invViewProj;   // Inverse view-projection for ray generation
    float3 cameraPos;                  // Camera world position
    float densityScale;                // Global density multiplier

    float3 gridWorldMin;               // Grid AABB minimum
    float emissionStrength;            // Emission intensity

    float3 gridWorldMax;               // Grid AABB maximum
    float absorptionCoeff;             // Beer-Lambert absorption

    float3 sphereCenter;               // Procedural fog sphere center
    float scatteringCoeff;             // Henyey-Greenstein scattering

    float sphereRadius;                // Procedural fog sphere radius
    float maxRayDistance;              // Maximum ray march distance
    float stepSize;                    // World-space step size
    uint lightCount;                   // Number of active lights

    uint screenWidth;
    uint screenHeight;
    float time;                        // Animation time
    uint debugMode;                    // 0=normal, 1=debug solid color
};

// ============================================================================
// RESOURCES
// ============================================================================

// Light data (reuse existing light structure)
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
    // Additional fields padded to 64 bytes...
    float4 godRayParams1;
    float4 godRayParams2;
};
StructuredBuffer<Light> g_lights : register(t1);

// Output texture
RWTexture2D<float4> g_output : register(u0);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Generate view ray from pixel coordinates
void GenerateRay(uint2 pixelCoord, out float3 origin, out float3 direction) {
    // Normalized device coordinates [-1, 1]
    float2 ndc = float2(pixelCoord) / float2(screenWidth, screenHeight);
    ndc = ndc * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for DirectX

    // Unproject near and far plane points
    float4 nearPoint = mul(float4(ndc, 0.0, 1.0), invViewProj);
    float4 farPoint = mul(float4(ndc, 1.0, 1.0), invViewProj);

    nearPoint.xyz /= nearPoint.w;
    farPoint.xyz /= farPoint.w;

    origin = cameraPos;
    direction = normalize(farPoint.xyz - nearPoint.xyz);
}

// Ray-AABB intersection (for grid bounds)
bool RayAABBIntersection(float3 origin, float3 invDir, float3 boxMin, float3 boxMax,
                          out float tMin, out float tMax) {
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;

    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);

    tMin = max(max(tSmall.x, tSmall.y), tSmall.z);
    tMax = min(min(tLarge.x, tLarge.y), tLarge.z);

    tMin = max(tMin, 0.0);  // Start at camera
    return tMax >= tMin;
}

// Henyey-Greenstein phase function for anisotropic scattering
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Temperature to RGB (simplified blackbody)
float3 TemperatureToColor(float temp) {
    // Approximate blackbody radiation
    float t = saturate((temp - 1000.0) / 30000.0);
    float3 cool = float3(1.0, 0.3, 0.1);   // Red/orange (cool)
    float3 hot = float3(0.8, 0.9, 1.0);    // Blue-white (hot)
    return lerp(cool, hot, t);
}

// ============================================================================
// PROCEDURAL DENSITY (PROTOTYPE - replaces NanoVDB sampling)
// ============================================================================

// Procedural fog sphere density - simulates what NanoVDB would provide
// Uses constant buffer parameters for sphere center/radius
float SampleProceduralDensity(float3 worldPos) {
    // Sphere parameters from constant buffer (set by CreateFogSphere in C++)
    float falloffWidth = 3.0 * stepSize;  // halfWidth * voxelSize

    // Distance from center
    float dist = length(worldPos - sphereCenter);

    // Fog sphere: density = 1 at center, falls off to 0 at surface
    if (dist > sphereRadius + falloffWidth) {
        return 0.0;  // Outside sphere
    }

    // Smooth falloff using smoothstep
    float innerRadius = sphereRadius - falloffWidth;
    if (dist < innerRadius) {
        return 1.0;  // Full density at core
    }

    // Falloff region
    float t = (dist - innerRadius) / (falloffWidth * 2.0);
    return 1.0 - smoothstep(0.0, 1.0, t);
}

// ============================================================================
// LIGHTING
// ============================================================================

// Calculate in-scattering from lights at a sample point
float3 CalculateLighting(float3 samplePos, float3 viewDir, float density) {
    float3 totalLight = float3(0, 0, 0);

    // Phase function parameter (negative = backward scatter, positive = forward)
    float phaseG = 0.3;  // Slight forward scattering

    for (uint i = 0; i < lightCount && i < 16; i++) {
        Light light = g_lights[i];

        float3 lightDir = normalize(light.position - samplePos);
        float lightDist = length(light.position - samplePos);

        // Attenuation
        float attenuation = light.intensity / (1.0 + lightDist * lightDist * 0.0001);

        // Phase function (anisotropic scattering)
        float cosTheta = dot(-viewDir, lightDir);
        float phase = HenyeyGreenstein(cosTheta, phaseG);

        // Scattering contribution
        float3 scattering = light.color * attenuation * phase * scatteringCoeff;

        totalLight += scattering;
    }

    return totalLight * density;
}

// ============================================================================
// RAY MARCHING
// ============================================================================

// Main ray marching function
float4 RayMarchVolume(float3 rayOrigin, float3 rayDir) {
    // Check ray-AABB intersection with grid bounds
    float3 invDir = 1.0 / rayDir;
    float tMin, tMax;

    if (!RayAABBIntersection(rayOrigin, invDir, gridWorldMin, gridWorldMax, tMin, tMax)) {
        return float4(0, 0, 0, 0);  // Ray misses grid
    }

    // Clamp to max distance
    tMax = min(tMax, maxRayDistance);

    // Accumulated color and transmittance
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;

    // Ray marching loop
    float t = tMin;
    const float minTransmittance = 0.01;  // Early termination threshold

    [loop]
    while (t < tMax && transmittance > minTransmittance) {
        float3 samplePos = rayOrigin + rayDir * t;

        // Sample density (procedural for prototype)
        float density = SampleProceduralDensity(samplePos) * densityScale;

        if (density > 0.001) {
            // Beer-Lambert law for absorption
            float absorption = absorptionCoeff * density * stepSize;
            float sampleTransmittance = exp(-absorption);

            // Calculate lighting at this sample
            float3 lighting = CalculateLighting(samplePos, rayDir, density);

            // Emission (self-luminous gas)
            float3 emission = TemperatureToColor(density * 10000.0) * emissionStrength * density;

            // Accumulate color (front-to-back compositing)
            float3 sampleColor = (lighting + emission) * (1.0 - sampleTransmittance);
            accumulatedColor += transmittance * sampleColor;

            // Update transmittance
            transmittance *= sampleTransmittance;
        }

        t += stepSize;
    }

    // Final opacity = 1 - transmittance
    float opacity = 1.0 - transmittance;

    return float4(accumulatedColor, opacity);
}

// ============================================================================
// COMPUTE SHADER ENTRY POINT
// ============================================================================

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    // Bounds check
    if (DTid.x >= screenWidth || DTid.y >= screenHeight) {
        return;
    }

    // Generate view ray
    float3 rayOrigin, rayDir;
    GenerateRay(DTid.xy, rayOrigin, rayDir);

    // DEBUG MODE: Output solid magenta if ray hits grid AABB
    if (debugMode == 1) {
        float3 invDir = 1.0 / rayDir;
        float tMin, tMax;
        if (RayAABBIntersection(rayOrigin, invDir, gridWorldMin, gridWorldMax, tMin, tMax)) {
            // Ray hits grid bounds - check if there's any density
            float3 testPos = rayOrigin + rayDir * ((tMin + tMax) * 0.5);
            float testDensity = SampleProceduralDensity(testPos);
            if (testDensity > 0.001) {
                // BRIGHT MAGENTA = fog sphere is being sampled
                g_output[DTid.xy] = float4(1.0, 0.0, 1.0, 1.0);
            } else {
                // DARK CYAN = ray hits bounds but no density at midpoint
                g_output[DTid.xy] = float4(0.0, 0.5, 0.5, 1.0);
            }
        } else {
            // DARK RED = ray misses grid bounds entirely
            float4 existing = g_output[DTid.xy];
            g_output[DTid.xy] = float4(existing.rgb * 0.8 + float3(0.2, 0.0, 0.0), existing.a);
        }
        return;
    }

    // NORMAL MODE: Full ray marching
    float4 volumeColor = RayMarchVolume(rayOrigin, rayDir);

    // Read existing pixel color (for compositing)
    float4 existingColor = g_output[DTid.xy];

    // Composite volume over existing content (front-to-back)
    float3 finalColor = volumeColor.rgb + existingColor.rgb * (1.0 - volumeColor.a);
    float finalAlpha = volumeColor.a + existingColor.a * (1.0 - volumeColor.a);

    // Write output
    g_output[DTid.xy] = float4(finalColor, finalAlpha);
}
