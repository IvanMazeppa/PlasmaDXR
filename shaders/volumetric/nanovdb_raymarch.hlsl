// NanoVDB Volumetric Ray Marching Shader - PROTOTYPE with Procedural Gas
// Renders sparse volumetric fog with depth occlusion and procedural noise
//
// Features:
// - Depth-aware ray marching (respects existing geometry)
// - Procedural FBM noise for amorphous gas appearance
// - Animated turbulence
// - Beer-Lambert absorption with Henyey-Greenstein scattering
//
// Based on:
// - NVIDIA NanoVDB (Museth, SIGGRAPH 2021)
// - Ray Tracing Gems II, Chapter 43
// - Inigo Quilez noise functions

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
    float4 godRayParams1;
    float4 godRayParams2;
};
StructuredBuffer<Light> g_lights : register(t1);

// Depth buffer for occlusion (from Gaussian renderer)
Texture2D<float> g_depthBuffer : register(t2);

// Output texture
RWTexture2D<float4> g_output : register(u0);

// ============================================================================
// 3D NOISE FUNCTIONS (for amorphous gas appearance)
// Based on Inigo Quilez and Stefan Gustavson's implementations
// ============================================================================

// Hash function for pseudo-random values
float hash31(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return frac((p.x + p.y) * p.z);
}

float3 hash33(float3 p) {
    p = float3(dot(p, float3(127.1, 311.7, 74.7)),
               dot(p, float3(269.5, 183.3, 246.1)),
               dot(p, float3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
}

// Gradient noise (Perlin-like)
float gradientNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);

    // Quintic interpolation for smoother results
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // 8 corner gradients
    float n000 = dot(hash33(i + float3(0, 0, 0)), f - float3(0, 0, 0));
    float n100 = dot(hash33(i + float3(1, 0, 0)), f - float3(1, 0, 0));
    float n010 = dot(hash33(i + float3(0, 1, 0)), f - float3(0, 1, 0));
    float n110 = dot(hash33(i + float3(1, 1, 0)), f - float3(1, 1, 0));
    float n001 = dot(hash33(i + float3(0, 0, 1)), f - float3(0, 0, 1));
    float n101 = dot(hash33(i + float3(1, 0, 1)), f - float3(1, 0, 1));
    float n011 = dot(hash33(i + float3(0, 1, 1)), f - float3(0, 1, 1));
    float n111 = dot(hash33(i + float3(1, 1, 1)), f - float3(1, 1, 1));

    // Trilinear interpolation
    float n00 = lerp(n000, n100, u.x);
    float n01 = lerp(n001, n101, u.x);
    float n10 = lerp(n010, n110, u.x);
    float n11 = lerp(n011, n111, u.x);
    float n0 = lerp(n00, n10, u.y);
    float n1 = lerp(n01, n11, u.y);

    return lerp(n0, n1, u.z);
}

// Fractional Brownian Motion - layered noise for natural appearance
float fbm(float3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * gradientNoise(p * frequency);
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value / maxValue;  // Normalize to [-1, 1]
}

// Turbulence - absolute value FBM for wispy tendrils
float turbulence(float3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * abs(gradientNoise(p * frequency));
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value / maxValue;  // Normalize to [0, 1]
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Generate view ray from pixel coordinates
void GenerateRay(uint2 pixelCoord, out float3 origin, out float3 direction) {
    float2 ndc = float2(pixelCoord) / float2(screenWidth, screenHeight);
    ndc = ndc * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(float4(ndc, 1.0, 1.0), invViewProj);
    farPoint.xyz /= farPoint.w;

    origin = cameraPos;
    direction = normalize(farPoint.xyz - cameraPos);
}

// Ray-AABB intersection
bool RayAABBIntersection(float3 origin, float3 invDir, float3 boxMin, float3 boxMax,
                          out float tMin, out float tMax) {
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;
    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);
    tMin = max(max(tSmall.x, tSmall.y), tSmall.z);
    tMax = min(min(tLarge.x, tLarge.y), tLarge.z);
    tMin = max(tMin, 0.0);
    return tMax >= tMin;
}

// Henyey-Greenstein phase function
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Temperature to RGB (simplified blackbody)
float3 TemperatureToColor(float temp) {
    float t = saturate((temp - 1000.0) / 30000.0);
    float3 cool = float3(1.0, 0.3, 0.1);
    float3 hot = float3(0.8, 0.9, 1.0);
    return lerp(cool, hot, t);
}

// ============================================================================
// PROCEDURAL GAS DENSITY
// ============================================================================

float SampleProceduralDensity(float3 worldPos) {
    // Base sphere distance
    float dist = length(worldPos - sphereCenter);

    // Outside sphere - early out
    if (dist > sphereRadius * 1.2) {
        return 0.0;
    }

    // Base density from sphere with soft falloff
    float innerRadius = sphereRadius * 0.4;  // Dense core
    float baseDensity;
    if (dist < innerRadius) {
        baseDensity = 1.0;
    } else {
        float t = (dist - innerRadius) / (sphereRadius - innerRadius);
        baseDensity = 1.0 - smoothstep(0.0, 1.0, t);
    }

    // Add animated noise for gas-like turbulence
    float noiseScale = 0.015;  // Controls detail frequency
    float3 noisePos = worldPos * noiseScale;

    // Animate the noise (slow swirling)
    float animSpeed = 0.08;
    noisePos += float3(
        sin(time * animSpeed * 0.7),
        cos(time * animSpeed * 0.5),
        sin(time * animSpeed * 0.3)
    ) * 1.5;

    // Multi-octave FBM for large-scale structure
    float largeTurbulence = fbm(noisePos, 3) * 0.5 + 0.5;  // Remap to [0, 1]

    // Higher frequency turbulence for wispy details
    float detailNoise = turbulence(noisePos * 2.5, 2);

    // Combine base shape with noise
    float density = baseDensity * (0.5 + 0.5 * largeTurbulence);

    // Add wispy tendrils at the edges
    float edgeFactor = smoothstep(0.3, 0.9, dist / sphereRadius);
    density += detailNoise * edgeFactor * 0.25;

    // Add gentle swirling motion based on angle
    float3 toCenter = worldPos - sphereCenter;
    float swirl = sin(atan2(toCenter.z, toCenter.x) * 4.0 + time * 0.3) * 0.1;
    density += swirl * baseDensity * 0.5;

    return max(density, 0.0);
}

// ============================================================================
// LIGHTING
// ============================================================================

float3 CalculateLighting(float3 samplePos, float3 viewDir, float density) {
    float3 totalLight = float3(0, 0, 0);
    float phaseG = 0.3;

    for (uint i = 0; i < lightCount && i < 16; i++) {
        Light light = g_lights[i];
        float3 lightDir = normalize(light.position - samplePos);
        float lightDist = length(light.position - samplePos);
        float attenuation = light.intensity / (1.0 + lightDist * lightDist * 0.0001);
        float cosTheta = dot(-viewDir, lightDir);
        float phase = HenyeyGreenstein(cosTheta, phaseG);
        float3 scattering = light.color * attenuation * phase * scatteringCoeff;
        totalLight += scattering;
    }

    return totalLight * density;
}

// ============================================================================
// RAY MARCHING WITH DEPTH OCCLUSION
// ============================================================================

float4 RayMarchVolume(float3 rayOrigin, float3 rayDir, float sceneDepth) {
    float3 invDir = 1.0 / rayDir;
    float tMin, tMax;

    if (!RayAABBIntersection(rayOrigin, invDir, gridWorldMin, gridWorldMax, tMin, tMax)) {
        return float4(0, 0, 0, 0);
    }

    // Clamp ray march to scene depth (occlusion by geometry)
    // sceneDepth is in world units from camera
    tMax = min(tMax, sceneDepth);
    tMax = min(tMax, maxRayDistance);

    // If the closest geometry is before the volume starts, volume is occluded
    if (tMax < tMin) {
        return float4(0, 0, 0, 0);
    }

    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;
    float t = tMin;
    const float minTransmittance = 0.01;

    [loop]
    while (t < tMax && transmittance > minTransmittance) {
        float3 samplePos = rayOrigin + rayDir * t;
        float density = SampleProceduralDensity(samplePos) * densityScale;

        if (density > 0.001) {
            float absorption = absorptionCoeff * density * stepSize;
            float sampleTransmittance = exp(-absorption);
            float3 lighting = CalculateLighting(samplePos, rayDir, density);
            float3 emission = TemperatureToColor(density * 10000.0) * emissionStrength * density;
            float3 sampleColor = (lighting + emission) * (1.0 - sampleTransmittance);
            accumulatedColor += transmittance * sampleColor;
            transmittance *= sampleTransmittance;
        }

        t += stepSize;
    }

    float opacity = 1.0 - transmittance;
    return float4(accumulatedColor, opacity);
}

// ============================================================================
// COMPUTE SHADER ENTRY POINT
// ============================================================================

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= screenWidth || DTid.y >= screenHeight) {
        return;
    }

    float3 rayOrigin, rayDir;
    GenerateRay(DTid.xy, rayOrigin, rayDir);

    // Sample depth buffer to get scene depth
    // The depth buffer contains linear depth (distance from camera)
    float sceneDepth = g_depthBuffer[DTid.xy];

    // If depth is 0 or very small, assume far plane (no geometry)
    if (sceneDepth < 0.001) {
        sceneDepth = maxRayDistance;
    }

    // DEBUG MODE
    if (debugMode == 1) {
        float3 invDir = 1.0 / rayDir;
        float tMin, tMax;
        if (RayAABBIntersection(rayOrigin, invDir, gridWorldMin, gridWorldMax, tMin, tMax)) {
            float3 testPos = rayOrigin + rayDir * ((tMin + tMax) * 0.5);
            float testDensity = SampleProceduralDensity(testPos);
            if (testDensity > 0.001) {
                g_output[DTid.xy] = float4(1.0, 0.0, 1.0, 1.0);  // Magenta = density found
            } else {
                g_output[DTid.xy] = float4(0.0, 0.5, 0.5, 1.0);  // Cyan = in AABB but no density
            }
        } else {
            float4 existing = g_output[DTid.xy];
            g_output[DTid.xy] = float4(existing.rgb * 0.8 + float3(0.2, 0.0, 0.0), existing.a);
        }
        return;
    }

    // TEST 1: No depth (use max distance) + additive blend + noise enabled
    float effectiveDepth = maxRayDistance;  // NO DEPTH TEST

    // Ray march the volume
    float4 volumeColor = RayMarchVolume(rayOrigin, rayDir, effectiveDepth);

    // Read existing content
    float4 existingColor = g_output[DTid.xy];

    // ADDITIVE BLEND (known working)
    float3 finalColor = existingColor.rgb + volumeColor.rgb;
    float finalAlpha = max(existingColor.a, volumeColor.a);

    g_output[DTid.xy] = float4(finalColor, finalAlpha);
}
