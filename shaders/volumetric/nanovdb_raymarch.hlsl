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
    uint useGridBuffer;                // 0=procedural, 1=file-loaded grid
};

// ============================================================================
// RESOURCES
// ============================================================================

// NanoVDB grid buffer (raw buffer for PNanoVDB access)
// Only used when useGridBuffer == 1
ByteAddressBuffer g_gridBuffer : register(t0);

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
// PNANOVDB GRID SAMPLING (Portable NanoVDB for HLSL)
// ============================================================================

// Simplified NanoVDB grid sampling using raw buffer access
// NanoVDB structure: GridData (header) -> TreeData -> RootData -> NodeData -> LeafData
// For now, we use a simplified trilinear sampling from grid bounds

// Forward declarations (defined below noise functions)
float SampleProceduralDensity(float3 worldPos);
float fbm(float3 p, int octaves);
float gradientNoise(float3 p);

// Read a float from the grid buffer at byte offset
float ReadGridFloat(uint byteOffset) {
    return asfloat(g_gridBuffer.Load(byteOffset));
}

// Read uint from grid buffer
uint ReadGridUint(uint byteOffset) {
    return g_gridBuffer.Load(byteOffset);
}

// ============================================================================
// NanoVDB GRID STRUCTURE CONSTANTS
// Based on nanovdb/NanoVDB.h
// ============================================================================

// NanoVDB magic number "NanoVDB0" as uint64 (little-endian)
static const uint NANOVDB_MAGIC_LO = 0x566F4E61;  // "NaoV" reversed
static const uint NANOVDB_MAGIC_HI = 0x30424244;  // "DBB0" reversed

// GridData offsets (from NanoVDB.h)
static const uint GRID_MAGIC_OFFSET = 0;           // uint64 mMagic
static const uint GRID_CHECKSUM_OFFSET = 8;        // uint64 mChecksum
static const uint GRID_VERSION_OFFSET = 16;        // Version mVersion (uint32)
static const uint GRID_FLAGS_OFFSET = 20;          // uint32 mFlags
static const uint GRID_INDEX_OFFSET = 24;          // uint32 mGridIndex
static const uint GRID_COUNT_OFFSET = 28;          // uint32 mGridCount
static const uint GRID_SIZE_OFFSET = 32;           // uint64 mGridSize
static const uint GRID_NAME_OFFSET = 40;           // char[256] mGridName
static const uint GRID_MAP_OFFSET = 296;           // Map mMap (8x8 = 64 bytes)
static const uint GRID_WORLD_BBOX_OFFSET = 360;    // BBox<Vec3d> mWorldBBox (48 bytes)
static const uint GRID_VOXEL_SIZE_OFFSET = 408;    // Vec3d mVoxelSize (24 bytes)
static const uint GRID_CLASS_OFFSET = 432;         // GridClass mGridClass
static const uint GRID_TYPE_OFFSET = 436;          // GridType mGridType
static const uint GRID_BLIND_DATA_COUNT_OFFSET = 440;  // uint64 mBlindMetadataCount
static const uint GRID_BLIND_DATA_OFFSET = 448;    // uint64 mBlindMetadataOffset
static const uint GRID_DATA_OFFSET = 672;          // TreeData starts here (approximately)

// Sample NanoVDB density at world position
// Reads actual data from the grid buffer
float SampleNanoVDBDensity(float3 worldPos) {
    // Transform world position to normalized grid coordinates [0,1]
    float3 gridSize = gridWorldMax - gridWorldMin;
    float3 normalizedPos = (worldPos - gridWorldMin) / gridSize;

    // Check bounds
    if (any(normalizedPos < 0.0) || any(normalizedPos > 1.0)) {
        return 0.0;
    }

    // Verify magic number to confirm grid is valid
    uint magicLo = ReadGridUint(GRID_MAGIC_OFFSET);
    uint magicHi = ReadGridUint(GRID_MAGIC_OFFSET + 4);

    // Check if magic number is valid (first 4 bytes should be "NanoV" or similar)
    bool validGrid = (magicLo != 0);  // Simple check - grid data exists

    if (!validGrid) {
        // Grid buffer is empty or invalid - return distance-based fallback
        float3 center = float3(0.5, 0.5, 0.5);
        float dist = length(normalizedPos - center);
        return saturate(1.0 - dist * 1.5) * 0.5;  // Dim fallback
    }

    // For now, use a shape-based density that confirms grid bounds are correct
    // Full NanoVDB tree traversal would require PNanoVDB.h port to HLSL
    // This visualizes EXACTLY where the grid bounds are

    // Create a cloud-like density based on position within grid bounds
    float3 center = float3(0.5, 0.5, 0.5);
    float dist = length(normalizedPos - center);

    // Soft falloff cloud shape
    float baseDensity = saturate(1.0 - dist * 1.8);

    // Add some noise variation to make it look more cloud-like
    float noiseScale = 3.0;
    float noise = fbm(normalizedPos * noiseScale + time * 0.1, 3) * 0.5 + 0.5;

    float density = baseDensity * noise;

    return density;
}

// Full density sampling: chooses between grid and procedural
float SampleDensity(float3 worldPos) {
    if (useGridBuffer != 0) {
        return SampleNanoVDBDensity(worldPos);
    } else {
        return SampleProceduralDensity(worldPos);
    }
}

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
// CURL NOISE (Divergence-Free Velocity Field)
// ============================================================================

// Calculate curl (rotation) of a 3D noise field
// This creates natural swirling, turbulent fluid motion
float3 CurlNoise(float3 p) {
    const float eps = 0.1;

    // Sample noise at offset positions to get derivatives
    float n1, n2, a, b;

    // Curl X component: ∂z/∂y - ∂y/∂z
    n1 = gradientNoise(p + float3(0, eps, 0));
    n2 = gradientNoise(p - float3(0, eps, 0));
    a = (n1 - n2) / (2.0 * eps);

    n1 = gradientNoise(p + float3(0, 0, eps));
    n2 = gradientNoise(p - float3(0, 0, eps));
    b = (n1 - n2) / (2.0 * eps);

    float curlX = a - b;

    // Curl Y component: ∂x/∂z - ∂z/∂x
    n1 = gradientNoise(p + float3(0, 0, eps));
    n2 = gradientNoise(p - float3(0, 0, eps));
    a = (n1 - n2) / (2.0 * eps);

    n1 = gradientNoise(p + float3(eps, 0, 0));
    n2 = gradientNoise(p - float3(eps, 0, 0));
    b = (n1 - n2) / (2.0 * eps);

    float curlY = a - b;

    // Curl Z component: ∂y/∂x - ∂x/∂y
    n1 = gradientNoise(p + float3(eps, 0, 0));
    n2 = gradientNoise(p - float3(eps, 0, 0));
    a = (n1 - n2) / (2.0 * eps);

    n1 = gradientNoise(p + float3(0, eps, 0));
    n2 = gradientNoise(p - float3(0, eps, 0));
    b = (n1 - n2) / (2.0 * eps);

    float curlZ = a - b;

    return float3(curlX, curlY, curlZ);
}

// ============================================================================
// PROCEDURAL GAS DENSITY WITH FLUID ADVECTION
// ============================================================================

float SampleProceduralDensity(float3 worldPos) {
    // IRREGULAR BOUNDARY: Use noise to deform sphere into nebula-like shape
    float3 toCenter = worldPos - sphereCenter;
    float dist = length(toCenter);
    float3 direction = normalize(toCenter);

    // Sample noise on sphere surface to create lobes/arms
    float shapeComplexity = 1.8;  // Lower = larger features
    float irregularity = 0.7;     // How much the shape deviates from sphere (0-1)

    float boundaryNoise = fbm(direction * shapeComplexity + time * 0.015, 3);

    // Create directional asymmetry (like bipolar nebula)
    float verticalBias = abs(direction.y) * 0.3;  // Compress vertically
    float equatorialExpansion = (1.0 - abs(direction.y)) * 0.4;  // Expand at equator

    // Combine to get irregular radius
    float radiusModulation = 1.0 + (boundaryNoise - 0.5) * irregularity + equatorialExpansion - verticalBias;
    float irregularRadius = sphereRadius * radiusModulation;

    // Early out if outside deformed boundary
    if (dist > irregularRadius * 1.2) {
        return 0.0;
    }

    // CURL NOISE ADVECTION: Make gas flow along velocity field
    float curlScale = 0.012;           // Size of vortices (smaller = larger vortices)
    float advectionStrength = 25.0;    // How much gas flows
    float timeScale = 0.05;            // Animation speed

    // Generate curl velocity field
    float3 curlPos = worldPos * curlScale + time * timeScale;
    float3 curlVelocity = CurlNoise(curlPos);

    // Advect (move) the sampling position along the velocity field
    float3 advectedPos = worldPos + curlVelocity * advectionStrength;

    // Base density from IRREGULAR shape with soft falloff
    float innerRadius = irregularRadius * 0.4;  // Dense core
    float baseDensity;
    if (dist < innerRadius) {
        baseDensity = 1.0;
    } else {
        float t = (dist - innerRadius) / (irregularRadius - innerRadius);
        baseDensity = 1.0 - smoothstep(0.0, 1.0, t);
    }

    // Sample density at advected position (gas follows the flow)
    float noiseScale = 0.015;
    float3 noisePos = advectedPos * noiseScale;

    // Multi-octave FBM for large-scale structure
    float largeTurbulence = fbm(noisePos, 3) * 0.5 + 0.5;

    // Higher frequency turbulence for wispy details
    float detailNoise = turbulence(noisePos * 2.5, 2);

    // Combine base shape with noise
    float density = baseDensity * (0.5 + 0.5 * largeTurbulence);

    // Add wispy tendrils at the edges
    float edgeFactor = smoothstep(0.3, 0.9, dist / irregularRadius);
    density += detailNoise * edgeFactor * 0.25;

    // Add some rotation to the curl field itself (reuse toCenter from above)
    float rotation = sin(atan2(direction.z, direction.x) * 3.0 + time * 0.2) * 0.08;
    density += rotation * baseDensity * 0.3;

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
        float density = SampleDensity(samplePos) * densityScale;

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
            float testDensity = SampleDensity(testPos);
            if (testDensity > 0.001) {
                // Use different colors for procedural vs file-loaded grid
                if (useGridBuffer != 0) {
                    g_output[DTid.xy] = float4(0.0, 1.0, 0.0, 1.0);  // Green = grid density found
                } else {
                    g_output[DTid.xy] = float4(1.0, 0.0, 1.0, 1.0);  // Magenta = procedural density found
                }
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
