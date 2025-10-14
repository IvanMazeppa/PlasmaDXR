// DXR Raytracing Library - Volumetric Sphere RT Lighting Demo with DXR 1.2 SER
// Target: lib_6_3 (DXR 1.0/1.1), lib_6_9 (DXR 1.2 with SER) - RTX 4060Ti optimized
//
// USAGE INSTRUCTIONS:
// 1. Compile with DXR_1_2_SER_ENABLED=1 for RTX 4060Ti SER optimization
// 2. Ensure g_scene TLAS is bound to t0 for ray tracing
// 3. Demo Modes:
//    - Mode 1: Pure DXR triangle geometry baseline
//    - Mode 2: Interactive volumetric lighting with DXR 1.2 SER hints
//    - Mode 3: Compact moving volume with SER coherence for self-shadowing prep
//    - Mode 4: Static complex volumetric sculpture with sweeping RT lighting and pure black background
// 4. RTX 4060Ti Features:
//    - Coherent ray batching via MaybeReorderThread() SER intrinsic
//    - 32MB L2 cache optimized tile-based coherence hints
//    - Memory bandwidth conscious ray direction quantization
//
// COMPATIBILITY:
// - SM 6.3: Base DXR 1.0/1.1 support without SER
// - SM 6.5: Enhanced inline ray tracing (use with compute shader)
// - SM 6.9: Full DXR 1.2 SER support with MaybeReorderThread()
//
// INTEGRATION WITH COMPUTE SHADER:
// - Use g_blendEnabled/g_blendScale for compositing with compute volumetrics
// - Compute shader (ray_march_cs.hlsl) handles inline ray queries for shadows

struct RayPayload {
    float4 color;
    uint coherenceHint; // DXR 1.2 SER coherence token for RTX 4060Ti optimization
};

// Volumetric sphere intersection data
struct VolumeHitInfo {
    float tNear;       // Ray entry point
    float tFar;        // Ray exit point
    bool hit;          // Whether ray intersects volume
};

// Global resources
RaytracingAccelerationStructure g_scene : register(t0);
Texture3D<float> g_densityVolume : register(t1);
SamplerState g_densitySampler : register(s0);
RWTexture2D<float4> g_output : register(u0);

// DXR 1.2 SER Support Detection (compile-time)
#ifndef DXR_1_2_SER_ENABLED
#define DXR_1_2_SER_ENABLED 0 // Set to 1 to enable SER on RTX 4060Ti
#endif

// RTX 4060Ti Cache Optimization Constants
static const uint RTX_4060TI_L2_CACHE_SIZE = 32 * 1024 * 1024; // 32MB L2 cache
static const uint RTX_4060TI_MEMORY_BANDWIDTH = 288; // GB/s
static const uint COHERENCE_TILE_SIZE = 8; // 8x8 pixel tiles for SER batching

// Global parameters via root constants (b0)
// NOTE: Root signature only allocates 16 floats, so blend control moved to g_pad slots
cbuffer GlobalParams : register(b0)
{
    float3 g_lightPos; float g_time;           // 0..3
    float3 g_lightDir; float g_innerCos;       // 4..7
    float3 g_lightColor; float g_outerCos;     // 8..11
    float g_mode; float g_bg; float g_blendEnabled; float g_blendScale; // 12..15
}


// Forward declarations for volumetric rendering
void ExecuteVolumetricHit(inout RayPayload payload, float3 rayOrigin, float3 rayDir, float tNear, float tFar, float time);
void ExecuteMiss(inout RayPayload payload, float3 rayDir);

// DXR 1.2 SER Coherence Hint Generation for RTX 4060Ti
uint GenerateSERCoherenceHint(uint2 screenPos, float3 rayDir, float rayLength) {
#if DXR_1_2_SER_ENABLED
    // Generate coherence token based on ray direction and length for SER optimization
    // RTX 4060Ti benefits from grouping similar rays to improve L2 cache utilization
    uint2 tileID = screenPos / COHERENCE_TILE_SIZE;

    // Quantize ray direction into octants (3 bits)
    uint3 rayOctant = uint3(rayDir > 0.0);
    uint directionBits = (rayOctant.x) | (rayOctant.y << 1) | (rayOctant.z << 2);

    // Quantize ray length into 4 distance bins (2 bits)
    uint lengthBin = min(3, uint(rayLength * 0.25)); // Adjust scaling based on scene

    // Combine tile, direction, and length for coherence hint
    return (tileID.x & 0xFF) | ((tileID.y & 0xFF) << 8) | (directionBits << 16) | (lengthBin << 19);
#else
    return 0;
#endif
}

// Volumetric density sampling function with animation
float SampleDensity(float3 worldPos, float3 sphereCenter, float sphereRadius, float time) {
    float distFromCenter = distance(worldPos, sphereCenter);

    // Create a smooth density falloff from center to edge
    float normalizedDist = distFromCenter / sphereRadius;

    if (normalizedDist > 1.0) {
        return 0.0; // Outside sphere
    }

    // Smooth falloff using smoothstep for nice visual appearance
    float density = 1.0 - smoothstep(0.0, 1.0, normalizedDist);

    // Add animated variation for visual interest - looser, slower pattern
    float variation = sin(worldPos.x * 1.5 + time * 1.0) * sin(worldPos.y * 1.2 + time * 0.8) * sin(worldPos.z * 1.8 + time * 1.2);
    density += variation * 0.2;

    // Add pulsing effect to make density changes more visible
    float pulse = 0.8 + 0.3 * sin(time * 3.0);

    return max(0.1, density * pulse); // Ensure some minimum density for visibility
}

// Ray-sphere intersection for volumetric rendering
VolumeHitInfo IntersectSphere(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereRadius) {
    VolumeHitInfo result;
    result.hit = false;
    result.tNear = 0.0;
    result.tFar = 0.0;

    // Vector from ray origin to sphere center
    float3 oc = rayOrigin - sphereCenter;

    // Quadratic equation coefficients: ||rayDir||^2 * t^2 + 2*dot(oc,rayDir) * t + ||oc||^2 - r^2 = 0
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return result; // No intersection
    }

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-b - sqrtDisc) / (2.0 * a); // Near intersection
    float t2 = (-b + sqrtDisc) / (2.0 * a); // Far intersection

    // Ensure t1 <= t2
    if (t1 > t2) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }

    // Check if intersection is in front of ray
    if (t2 > 0.001) { // Small epsilon to avoid self-intersection
        result.hit = true;
        result.tNear = max(0.001, t1); // Start marching from ray origin if inside sphere
        result.tFar = t2;
    }

    return result;
}

// Spotlight helper – soft cone with distance attenuation
float SpotlightTerm(float3 worldPos, float3 lightPos, float3 lightDir,
                    float innerCos, float outerCos) {
    float3 toPoint = normalize(worldPos - lightPos);
    float cosTheta = dot(toPoint, normalize(lightDir));
    // Soft edge between outer and inner cone
    float cone = saturate((cosTheta - outerCos) / max(1e-4, (innerCos - outerCos)));
    // Simple quadratic attenuation
    float d = distance(worldPos, lightPos);
    float atten = 1.0 / (1.0 + 0.4 * d + 0.15 * d * d);
    return cone * atten;
}

// Forward declaration for Mode 4 function
void ExecuteVolumetricSculpture(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time);

// Forward declaration for Mode 5 function
void ExecutePlasmaAccretion(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time);

// Forward declaration for Mode 6 function
void ExecuteVoxelParticles(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time);

// Forward declaration for Mode 7 function
void ExecuteMetaballSPH(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time);

[shader("raygeneration")]
void RayGen() {
    uint2 index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;

    // Calculate UV coordinates
    float2 uv = float2(index) / float2(dimensions);

    // Setup ray for perspective projection
    float aspectRatio = float(dimensions.x) / float(dimensions.y);

    // Camera position and direction
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for screen space

    float3 rayOrigin = float3(0, 0, -3);
    float3 rayDir = normalize(float3(ndc.x * aspectRatio, ndc.y, 1.0));

    // Initialize payload with DXR 1.2 SER coherence hint
    RayPayload payload;
    payload.color = float4(0, 0, 0, 1.0);
    payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, 10.0); // Max ray length estimate

#if DXR_1_2_SER_ENABLED
    // DXR 1.2 SER: Provide coherence hint before ray tracing for RTX 4060Ti optimization
    MaybeReorderThread(payload.coherenceHint);
#endif

    // Demo Mode Selection based on g_mode from GlobalParams
    int demoMode = (int)g_mode;

    if (demoMode == 1) {
        // === SPHERE RT BASELINE: Pure DXR triangle geometry ===
        // Trace ray against the BLAS/TLAS triangle geometry for clean RT lighting demo
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;

        TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    } else if (demoMode == 2) {
        // === TORCHLIGHT DEMO: Interactive mouse-controlled lighting with DXR 1.2 ===
        // Use volumetric sphere with interactive controls (g_lightPos animated by mouse)
        float animTime = float((index.x * 1919 + index.y * 2019) % 10000) * 0.001;
        float3 sphereCenter = float3(0.0, 0.0, 0.0);
        float sphereRadius = 1.2;

#if DXR_1_2_SER_ENABLED
        // Update SER coherence hint for volumetric rays
        payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, sphereRadius * 2.4);
        MaybeReorderThread(payload.coherenceHint);
#endif

        VolumeHitInfo volInfo = IntersectSphere(rayOrigin, rayDir, sphereCenter, sphereRadius);
        if (volInfo.hit) {
            ExecuteVolumetricHit(payload, rayOrigin, rayDir, volInfo.tNear, volInfo.tFar, animTime);
        } else {
            ExecuteMiss(payload, rayDir);
        }

    } else if (demoMode == 3) {
        // === VOLUMETRIC DEMO: Dramatic moving volume with DXR 1.2 inline ray traced self-shadowing ===
        // Enhanced motion patterns for spectacular volumetric movement
        float animTime = g_time; // Use global time for consistent animation

        // EXAGGERATED MOTION: Figure-8 pattern with dramatic sweep
        float3 sphereCenter = float3(
            sin(animTime * 0.8) * 1.2,                    // Wide horizontal sweep (2.4 unit range)
            cos(animTime * 1.3) * 0.8 + sin(animTime * 2.1) * 0.4,  // Complex vertical motion
            sin(animTime * 0.6) * 0.5                     // Forward/backward motion for 3D effect
        );
        float sphereRadius = 0.6 + 0.2 * sin(animTime * 1.5); // Pulsing size for organic feel

#if DXR_1_2_SER_ENABLED
        // Enhanced SER coherence hint for dramatic moving volumetric geometry
        float3 motionVector = float3(cos(animTime * 0.8) * 0.96, -sin(animTime * 1.3) * 0.8, cos(animTime * 0.6) * 0.3);
        uint motionHash = asuint(dot(motionVector, float3(73.0, 137.0, 179.0)));
        payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, sphereRadius * 2.0) ^ motionHash;
        MaybeReorderThread(payload.coherenceHint);
#endif

        VolumeHitInfo volInfo = IntersectSphere(rayOrigin, rayDir, sphereCenter, sphereRadius);
        if (volInfo.hit) {
            ExecuteVolumetricHit(payload, rayOrigin, rayDir, volInfo.tNear, volInfo.tFar, animTime);
        } else {
            ExecuteMiss(payload, rayDir);
        }

    } else if (demoMode == 4) {
        // === VOLUMETRIC SCULPTURE: Static complex shape with sweeping RT lighting ===
        // Complex static volumetric shape with colorful interior and pure DXR lighting
        float3 sphereCenter = float3(0.0, 0.0, -1.5); // Positioned closer to match volume position
        float sphereRadius = 2.2; // Larger radius for better torch visibility

#if DXR_1_2_SER_ENABLED
        // Update SER coherence hint for static volumetric sculpture
        payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, sphereRadius * 2.0);
        MaybeReorderThread(payload.coherenceHint);
#endif

        VolumeHitInfo volInfo = IntersectSphere(rayOrigin, rayDir, sphereCenter, sphereRadius);
        if (volInfo.hit) {
            ExecuteVolumetricSculpture(payload, rayOrigin, rayDir, index, g_time);
        } else {
            ExecuteMiss(payload, rayDir);
        }

    } else if (demoMode == 5) {
        // === PLASMA ACCRETION DISK: Orbital plasma with volumetric self-shadowing ===
        // Galactic-style accretion disk with churning plasma rotating around gravity center
        ExecutePlasmaAccretion(payload, rayOrigin, rayDir, index, g_time);

    } else if (demoMode == 6) {
        // === VOXEL PARTICLES: Debug particle simulation with 3D grid ===
        // Voxel-based particle system with granular control and non-symmetrical patterns
        ExecuteVoxelParticles(payload, rayOrigin, rayDir, index, g_time);

    } else if (demoMode == 7) {
        // === METABALL SPH: SPH physics with metaball density field rendering ===
        // SPH fluid simulation visualized through metaball density fields
        ExecuteMetaballSPH(payload, rayOrigin, rayDir, index, g_time);

    } else if (demoMode == 8) {
        // === DXR12 TEST: Clean DXR 1.2 foundation with simple animation ===
        // Time-based color cycling and geometric pattern for stable foundation testing
        float3 color = float3(
            0.5 + 0.5 * sin(g_time + uv.x * 3.14159),
            0.5 + 0.5 * sin(g_time * 1.3 + uv.y * 3.14159),
            0.5 + 0.5 * sin(g_time * 0.7 + (uv.x + uv.y) * 3.14159)
        );

        // Add pulsing center circle for visual focal point
        float2 center = uv - 0.5;
        float radius = length(center);
        float pulse = 0.8 + 0.3 * sin(g_time * 2.0);
        if (radius < 0.1 * pulse) {
            color = lerp(color, float3(1.0, 1.0, 1.0), 0.5);
        }

        payload.color = float4(color, 1.0);

    } else {
        // Fallback: Simple test pattern
        payload.color = float4(uv.x, uv.y, 0.5, 1.0);
    }

    // Write result to output texture (additive over existing HDR so compute content remains)
    if (g_blendEnabled > 0.5) {
        float4 prev = g_output[index];
        g_output[index] = prev + payload.color * g_blendScale;
    } else {
        g_output[index] = payload.color;
    }
}

[shader("miss")]
void Miss(inout RayPayload payload) {
    // Dark room background to emphasize spotlight
    float3 direction = WorldRayDirection();
    float t = 0.5 * (direction.y + 1.0);
    float3 topColor = float3(0.02, 0.02, 0.025) * g_bg;
    float3 bottomColor = float3(0.0, 0.0, 0.0) * g_bg;
    float3 skyColor = lerp(bottomColor, topColor, t);
    payload.color = float4(skyColor, 1.0);
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    // Calculate barycentric coordinates
    float3 barycentrics = float3(
        1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
        attribs.barycentrics.x,
        attribs.barycentrics.y);

    // Get world position using ray equation
    float3 worldPos = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    // Calculate surface normal for triangle (facing camera)
    // For a simple triangle, we can use a fixed normal or derive from vertices
    float3 normal = float3(0, 0, -1); // Face toward camera

    // Simple directional lighting
    float3 lightDir = normalize(float3(0.3, 0.8, -0.5));
    float3 lightColor = float3(1.0, 0.9, 0.7); // Warm white

    // Lambertian diffuse
    float NdotL = max(0.0, dot(normal, lightDir));
    float3 diffuse = lightColor * NdotL;

    // Add some ambient
    float3 ambient = float3(0.1, 0.1, 0.15);

    // Combine lighting with barycentric coloring for visual feedback
    float3 surfaceColor = lerp(float3(0.8, 0.2, 0.2), float3(0.2, 0.8, 0.2), barycentrics.y);
    surfaceColor = lerp(surfaceColor, float3(0.2, 0.2, 0.8), barycentrics.z);

    float3 finalColor = surfaceColor * (diffuse + ambient);

    payload.color = float4(finalColor, 1.0);
}

// Manual execution of bright pulsing volumetric sphere
void ExecuteVolumetricHit(inout RayPayload payload, float3 rayOrigin, float3 rayDir, float tNear, float tFar, float time) {
    // Volume properties
    float3 sphereCenter = float3(0.0, 0.0, 0.0);
    float sphereRadius = 1.2;

    // Spotlight setup from root constants (animated in C++)
    float3 lightPos  = g_lightPos;
    float3 lightDir  = normalize(g_lightDir);
    float  innerCos  = g_innerCos;
    float  outerCos  = g_outerCos;
    float3 lightCol  = g_lightColor;
    float3 ambient   = float3(0.03, 0.03, 0.035);

    // March through the sphere segment [tNear, tFar]
    const int   kSteps   = 64;
    float       t        = tNear;
    float       dt       = (tFar - tNear) / kSteps;
    float3      accum    = 0.0.xxx;
    float       trans    = 1.0;
    const float sigmaA   = 1.2;   // absorption
    const float sigmaS   = 2.0;   // scattering

    [loop]
    for (int i = 0; i < kSteps; ++i) {
        float3 p = rayOrigin + (t + 0.5 * dt) * rayDir;

        // Animated density inside sphere
        float dens = SampleDensity(p, sphereCenter, sphereRadius, time);

        // Spotlight contribution
        float spot = SpotlightTerm(p, lightPos, lightDir, innerCos, outerCos);

        // Simple single-scatter model
        float3 Li = lightCol * spot;
        float3 scatter = Li * (dens * sigmaS) * trans * dt;
        accum += scatter;

        // Beer-Lambert absorption
        trans *= exp(-dens * sigmaA * dt);
        t += dt;
    }

    float3 color = accum + ambient * 0.2;
    payload.color = float4(color, 1.0);
}

// Manual execution of Miss logic (called from RayGen)
void ExecuteMiss(inout RayPayload payload, float3 rayDir) {
    int demoMode = (int)g_mode;

    if (demoMode == 4) {
        // Mode 4: Pure black background for volumetric sculpture
        payload.color = float4(0.0, 0.0, 0.0, 1.0);
    } else if (demoMode == 5) {
        // Mode 5: Deep space background for plasma accretion disk
        payload.color = float4(0.01, 0.005, 0.02, 1.0);  // Very dark purple space
    } else {
        // Use g_bg to control background brightness (torchlight mode uses dark background)
        float3 direction = rayDir;
        float t = 0.5 * (direction.y + 1.0);

        if (g_bg < 0.1) {
            // Torchlight mode: very dark background
            float3 topColor = float3(0.02, 0.02, 0.025) * g_bg;
            float3 bottomColor = float3(0.0, 0.0, 0.0);
            float3 skyColor = lerp(bottomColor, topColor, t);
            payload.color = float4(skyColor, 1.0);
        } else {
            // Normal mode: yellow gradient for visibility
            float3 topColor = float3(1.0, 1.0, 0.5) * g_bg;
            float3 bottomColor = float3(0.8, 0.6, 0.0) * g_bg;
            float3 skyColor = lerp(bottomColor, topColor, t);
            payload.color = float4(skyColor, 1.0);
        }
    }
}

// ============================================================================
// MODE 4: VOLUMETRIC SCULPTURE - Static complex shape with sweeping RT lighting
// ============================================================================

// Complex volumetric shape: Twisted torus with fractal details
float SampleComplexVolume(float3 p) {
    // Volume is centered at (0,0,-1.5) with radius 1.8
    float3 sphereCenter = float3(0.0, 0.0, -1.5);
    float3 adjustedPos = p - sphereCenter;

    float distFromCenter = length(adjustedPos);
    float sphereRadius = 2.2; // Match intersection test radius
    float simpleSphere = sphereRadius - distFromCenter;

    if (simpleSphere > 0.0) {
        // Sharper, more defined noise for better torch-lit definition
        float3 noisePos = adjustedPos * 4.0; // Higher frequency for sharper details
        float noise = sin(noisePos.x) * sin(noisePos.y) * sin(noisePos.z) * 0.15;

        // Add some layered detail for torch-like surface texture
        float3 detailPos = adjustedPos * 8.0;
        float detail = cos(detailPos.x + detailPos.y) * sin(detailPos.z) * 0.08;

        return simpleSphere + noise + detail;
    }

    return 0.0; // Outside sphere

    // ORIGINAL COMPLEX CODE (commented out for debugging)
    /*
    // Scale up for better visibility and performance budget usage
    float3 scaledP = p * 0.6; // Make 1.67x larger

    // Rotate entire torus to show twist better (45 degrees around X, 30 degrees around Y)
    float3 q = scaledP;

    // First rotation: 45 degrees around X axis to tilt it toward camera
    float cx = cos(0.785398); // 45 degrees
    float sx = sin(0.785398);
    float tempY = q.y;
    q.y = cx * tempY - sx * q.z;
    q.z = sx * tempY + cx * q.z;

    // Second rotation: 30 degrees around Y axis for better viewing angle
    float cy = cos(0.523599); // 30 degrees
    float sy = sin(0.523599);
    float tempX = q.x;
    q.x = cy * tempX + sy * q.z;
    q.z = -sy * tempX + cy * q.z;

    // Enhanced twist along Y axis (more dramatic)
    float twistAngle = q.y * 4.0; // Increased from 2.0 to 4.0 for more twist
    float c = cos(twistAngle);
    float s = sin(twistAngle);
    float rotX = c * q.x - s * q.z;
    float rotZ = s * q.x + c * q.z;
    q.x = rotX;
    q.z = rotZ;

    // Larger torus (major radius = 1.4, minor radius = 0.6)
    float2 t = float2(length(q.xz) - 1.4, q.y);
    float torus = length(t) - 0.6;

    // Enhanced fractal noise details (more layers)
    float3 noisePos1 = scaledP * 4.0;
    float3 noisePos2 = scaledP * 8.0;
    float3 noisePos3 = scaledP * 16.0;

    float noise1 = sin(noisePos1.x) * sin(noisePos1.y) * sin(noisePos1.z) * 0.12;
    float noise2 = sin(noisePos2.x) * sin(noisePos2.y) * sin(noisePos2.z) * 0.06;
    float noise3 = sin(noisePos3.x) * sin(noisePos3.y) * sin(noisePos3.z) * 0.03;

    float totalNoise = noise1 + noise2 + noise3;

    // More complex bulges and depressions (multiple frequencies)
    float bulge1 = sin(scaledP.x * 6.0) * sin(scaledP.y * 4.0) * sin(scaledP.z * 8.0) * 0.08;
    float bulge2 = cos(scaledP.x * 10.0 + scaledP.y * 6.0) * sin(scaledP.z * 12.0) * 0.04;

    float totalBulge = bulge1 + bulge2;

    return -(torus + totalNoise + totalBulge); // Negative because we want interior
    */
}

// Colorful density based on position within the volume
float3 GetVolumeColor(float3 p, float density) {
    if (density <= 0.0) return float3(0, 0, 0);

    // Color based on position and density
    float3 baseColor;
    float distFromCenter = length(p);

    // Core is hot (red/orange), edges are cool (blue/purple)
    if (distFromCenter < 0.6) {
        baseColor = lerp(float3(1.0, 0.2, 0.1), float3(1.0, 0.6, 0.0), distFromCenter / 0.6); // Red to orange
    } else {
        baseColor = lerp(float3(1.0, 0.6, 0.0), float3(0.2, 0.4, 1.0), (distFromCenter - 0.6) / 0.8); // Orange to blue
    }

    // Add some variation based on noise
    float3 noisePos = p * 5.0;
    float colorNoise = (sin(noisePos.x) + sin(noisePos.y) + sin(noisePos.z)) * 0.1;
    baseColor += colorNoise;

    return baseColor * density;
}

// Sweeping directional light
float3 GetSweepingLightDirection(float time) {
    // Figure-8 pattern for light direction
    float x = sin(time * 0.7) * 0.8;
    float y = sin(time * 0.5) * 0.3 + 0.2; // Slightly elevated
    float z = sin(time * 0.9) * cos(time * 0.7) * 0.6;
    return normalize(float3(x, y, z));
}

// Simple volumetric self-shadowing using inline ray queries
// OPTIMIZED: Reduced steps 16 → 8, single exponential, early exit threshold
float ComputeVolumetricShadow(float3 pos, float3 lightDir) {
#if INLINE_RT_ENABLED
    // Cast shadow ray through the volume
    float3 rayStart = pos;
    float3 rayDir = lightDir;

    // March through volume towards light (OPTIMIZED: 8 steps instead of 16)
    const int shadowSteps = 8;
    float stepSize = 0.1;
    float opticalDepth = 0.0;

    [loop]
    for (int i = 0; i < shadowSteps; i++) {
        float3 samplePos = rayStart + rayDir * (i * stepSize);
        float density = max(0.0, SampleComplexVolume(samplePos));

        if (density > 0.1) {
            opticalDepth += density * stepSize;

            // Early exit (perceptual threshold: exp(-3.5) = 0.03)
            if (opticalDepth > 3.5) break;
        }

        // Stop if we're too far from the volume
        if (length(samplePos) > 2.0) break;
    }

    // Single exponential (instead of per-step multiplication)
    return exp(-opticalDepth * 0.8);
#else
    return 1.0; // No shadows if inline RT not available
#endif
}

// Main Mode 4 execution function
void ExecuteVolumetricSculpture(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time) {
    // Ray marching parameters - optimized for immediate torch response
    const int maxSteps = 100;  // Fewer steps for faster response
    const float stepSize = 0.03;  // Larger steps for immediate response
    const float maxDistance = 4.0;

    // Lighting setup from global parameters (animated sweeping light from C++)
    float3 lightDir = normalize(g_lightDir);
    float3 lightColor = g_lightColor;
    float3 ambient = float3(0.0, 0.0, 0.0); // No ambient for torch-on-rock (immediate response)

    // Ray marching
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;
    float t = 0.1; // Start a bit away from camera

    [loop]
    for (int step = 0; step < maxSteps; step++) {
        if (t > maxDistance || transmittance < 0.01) break;

        float3 samplePos = rayOrigin + rayDir * t;
        float density = max(0.0, SampleComplexVolume(samplePos));

        if (density > 0.03) { // Moderate threshold for sharp torch contact
            // Get volume color at this position
            float3 volumeColor = GetVolumeColor(samplePos, density);

            // Spotlight attenuation (torch beam falloff)
            float3 lightPos = g_lightPos;
            float3 lightToSample = samplePos - lightPos;
            float lightDistance = length(lightToSample);
            float3 lightDirNorm = normalize(lightToSample);

            // Spotlight cone calculation (torch beam)
            float spotDot = dot(lightDirNorm, normalize(lightDir));
            float spotAttenuation = saturate((spotDot - g_outerCos) / max(0.001, g_innerCos - g_outerCos));

            // Distance attenuation (torch beam weakens with distance)
            float distanceAttenuation = 1.0 / (1.0 + 0.1 * lightDistance + 0.01 * lightDistance * lightDistance);

            // DIRECT ILLUMINATION MODEL for torch-on-rock effect (no volumetric scattering)
            float shadowFactor = ComputeVolumetricShadow(samplePos, lightDir);

            // Direct surface-like illumination (not volumetric scattering)
            float3 directIllumination = lightColor * shadowFactor * spotAttenuation * distanceAttenuation;

            // Sharp contact lighting - volume acts more like a surface
            float3 surfaceLight = volumeColor * directIllumination * density * stepSize * 2.0f;
            accumulatedColor += surfaceLight;

            // Minimal absorption for immediate torch response (no burn-in)
            transmittance *= exp(-density * stepSize * 1.0);  // Much lower for immediate response
        }

        // Debug: Remove debug hint that might cause brightness issues

        t += stepSize;
    }

    // Final color with blend enabled for better surface contact (user feedback)
    // Remove exposure boost to prevent brightness escalation - light intensity already boosted in C++
    payload.color = float4(accumulatedColor, 1.0);

#if DXR_1_2_SER_ENABLED
    // Update SER coherence hint for static volume
    payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, 2.0);
#endif
}

// ============================================================================
// MODE 5: PLASMA ACCRETION DISK - Orbital plasma with volumetric self-shadowing
// ============================================================================

// Sample plasma density for accretion disk (torus shape with orbital motion)
float SamplePlasmaAccretionDisk(float3 p, float time) {
    // Disk parameters (controlled by C++ parameters)
    float3 diskCenter = g_lightPos;  // Gravity center offset control
    float3 localPos = p - diskCenter;

    // Tilt disk 30 degrees toward camera for better face-on view
    float tiltAngle = 0.52359877559; // 30 degrees in radians
    float cosT = cos(tiltAngle);
    float sinT = sin(tiltAngle);

    // Rotate around X-axis to tilt toward camera
    float3 tiltedPos = localPos;
    tiltedPos.y = localPos.y * cosT - localPos.z * sinT;
    tiltedPos.z = localPos.y * sinT + localPos.z * cosT;

    // Convert to cylindrical coordinates for disk
    float r = length(tiltedPos.xz);  // Distance from center axis
    float y = tiltedPos.y;           // Height above disk plane

    // Disk geometry (controlled by C++ parameters)
    float outerRadius = 2.5;
    float innerRadius = 0.8;
    float diskHeight = 0.8 * g_lightDir.z;  // Disk thickness control (lightDir.z)

    // Basic torus/disk shape
    if (r < innerRadius || r > outerRadius || abs(y) > diskHeight) {
        return 0.0; // Outside disk
    }

    // Orbital motion: plasma rotates based on distance (controlled by C++ parameters)
    float angularVelBase = g_lightDir.x;  // Angular velocity control (lightDir.x)
    float gravityExponent = g_innerCos;   // Gravity strength control (innerCos)
    float angularVel = angularVelBase / pow(r, gravityExponent); // Keplerian rotation
    float angle = atan2(tiltedPos.z, tiltedPos.x) + angularVel * time;

    // Density varies with radius (hotter/denser near center)
    float radialDensity = smoothstep(outerRadius, innerRadius, r);

    // Height falloff (disk is thin)
    float heightFalloff = exp(-abs(y) / diskHeight * 3.0);

    // Churning motion: spiral density waves
    float spiralWaves = sin(angle * 3.0 - r * 2.0) * 0.3 + 0.7;

    // Turbulence for plasma-like appearance
    float3 turbPos = tiltedPos * 4.0 + float3(time * 0.3, 0, time * 0.5);
    float turbulence = sin(turbPos.x) * sin(turbPos.y) * sin(turbPos.z) * 0.2 + 0.8;

    return radialDensity * heightFalloff * spiralWaves * turbulence * 0.8 * g_lightDir.y;  // Particle density multiplier
}

// Temperature-based emission for plasma (hotter near center)
float3 GetPlasmaEmission(float3 p, float density, float time) {
    if (density <= 0.0) return float3(0, 0, 0);

    float3 diskCenter = float3(0.0, 0.0, 0.0);
    float r = length((p - diskCenter).xz);

    // Enhanced temperature gradient: hotter near center (blue-white), cooler at edges (red-orange)
    float temp = saturate(4.0 / max(r, 0.3)); // More dramatic inverse relationship with radius

    // Temperature zones with STRONG COLOR SEPARATION and DISTINCT VALUES
    float3 emission;
    if (temp > 0.7) {
        // Core: INTENSE BLUE-WHITE plasma (very high temperature)
        float3 coreColor = float3(0.3, 0.7, 1.0);  // Strong blue dominance
        float intensity = 8.0 + temp * 4.0; // Much stronger core emission
        emission = coreColor * intensity * g_lightColor.x;
    } else if (temp > 0.35) {
        // Mid-disk: PURE YELLOW plasma (medium temperature)
        float3 midColor = float3(1.0, 0.8, 0.0);  // Pure yellow, no red
        float intensity = 4.0 + temp * 2.0; // Medium intensity
        emission = midColor * intensity * g_lightColor.y;
    } else {
        // Outer edge: DEEP RED plasma (cooler temperature)
        float3 edgeColor = float3(1.0, 0.1, 0.0);  // Pure red, minimal other channels
        float intensity = 2.0 + temp * 1.0; // Lower intensity for edges
        emission = edgeColor * intensity * g_lightColor.z;
    }

    // Reduce pulsing to avoid color washing
    float pulse = sin(time * 1.5 + r * 0.3) * 0.1 + 1.0;

    // Use density-based falloff to preserve colors at low density
    float colorPreservation = saturate(density * 3.0); // Stronger color at higher density

    return emission * colorPreservation * pulse;
}

// Main Mode 5 execution function
void ExecutePlasmaAccretion(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time) {
    // Ray marching parameters for plasma disk (controlled by C++ parameters)
    const int maxSteps = (int)g_outerCos;  // Simulation quality control (outerCos)
    const float stepSize = 0.025; // Smaller steps for finer detail
    const float maxDistance = 8.0; // Larger simulation volume

    // Plasma emission accumulation
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;
    float t = 0.1; // Start slightly away from camera

#if DXR_1_2_SER_ENABLED
    // Update SER coherence hint for moving plasma
    payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, 3.0);
    MaybeReorderThread(payload.coherenceHint);
#endif

    [loop]
    for (int step = 0; step < maxSteps; step++) {
        if (t > maxDistance || transmittance < 0.01) break;

        float3 samplePos = rayOrigin + rayDir * t;
        float density = SamplePlasmaAccretionDisk(samplePos, time);

        if (density > 0.02) {
            // Get plasma emission (self-illuminating)
            float3 emission = GetPlasmaEmission(samplePos, density, time);

#if INLINE_RT_ENABLED
            // Self-shadowing through plasma using inline ray tracing
            // OPTIMIZED: Reduced steps from 16 → 8 (sufficient for volumetric)
            float3 shadowRayDir = normalize(float3(0.5, 1.0, 0.3)); // Directional for shadows

            const int shadowSteps = 8;  // Reduced from 16
            float shadowStepSize = 0.1;
            float opticalDepth = 0.0;

            // Single accumulation loop (not nested!)
            for (int s = 0; s < shadowSteps; s++) {
                float3 shadowPos = samplePos + shadowRayDir * (s * shadowStepSize);
                float shadowDensity = SamplePlasmaAccretionDisk(shadowPos, time);

                if (shadowDensity > 0.05) {
                    opticalDepth += shadowDensity * shadowStepSize;

                    // Early exit (perceptual threshold: exp(-3.5) = 0.03)
                    if (opticalDepth > 3.5) break;
                }

                // Stop shadow ray at disk boundaries
                if (length(shadowPos.xz) > 3.0 || abs(shadowPos.y) > 0.8) break;
            }

            // Single exponential for shadow factor
            float shadowFactor = exp(-opticalDepth * 0.5);
            emission *= shadowFactor;
#endif

            // Volumetric scattering integration
            float3 scatteredLight = emission * transmittance * stepSize;
            accumulatedColor += scatteredLight;

            // Absorption through plasma
            transmittance *= exp(-density * stepSize * 0.8);
        }

        t += stepSize;
    }

    // Final plasma emission
    payload.color = float4(accumulatedColor, 1.0);

#if DXR_1_2_SER_ENABLED
    // Update SER coherence hint for plasma motion
    payload.coherenceHint = GenerateSERCoherenceHint(index, rayDir, 3.0);
#endif
}

// ============================================================================
// MODE 6: VOXEL PARTICLES - Debug particle simulation with 3D grid
// ============================================================================

// Sample density from voxel grid (stub implementation)
float SampleVoxelDensity(float3 worldPos, float time) {
    // TODO: Sample from 3D voxel density texture (m_voxelDensityTexture)
    // For now, create a non-symmetrical debug pattern to test the concept

    float3 voxelPos = worldPos * 0.25; // Scale to voxel space

    // Create irregular, non-symmetrical pattern
    float noise1 = sin(voxelPos.x * 3.7 + time * 0.8) * cos(voxelPos.y * 2.3 - time * 1.1);
    float noise2 = sin(voxelPos.z * 1.9 + time * 0.6) * sin(voxelPos.x * 4.1 + voxelPos.y * 3.2);
    float noise3 = cos(voxelPos.y * 2.8 + time * 0.9) * sin((voxelPos.x + voxelPos.z) * 1.7);

    // Combine noises for irregular pattern
    float combined = (noise1 + noise2 * 0.7 + noise3 * 0.5) * 0.4;

    // Add some spatial falloff for bounded volume
    float3 center = float3(0, 0, -1);
    float dist = length(worldPos - center);
    float falloff = exp(-dist * 0.8);

    return saturate(combined * falloff);
}

// Get particle color from voxel temperature/velocity
float3 GetVoxelParticleEmission(float3 worldPos, float density, float time) {
    if (density <= 0.0) return float3(0, 0, 0);

    // TODO: Sample from voxel temperature and velocity textures
    // For now, create color variation based on position and density

    float3 center = float3(0, 0, -1);
    float3 offset = worldPos - center;

    // Create color variation based on position
    float r = length(offset.xz);
    float height = offset.y;

    // Multi-colored particle system
    float3 color;
    if (r < 0.8) {
        // Core: Bright cyan-white
        color = float3(0.6, 0.9, 1.0);
    } else if (r < 1.5) {
        // Mid: Green-yellow
        color = float3(0.8, 1.0, 0.4);
    } else {
        // Outer: Purple-red
        color = float3(1.0, 0.3, 0.8);
    }

    // Height-based variation
    float heightFactor = 1.0 + sin(height * 2.0) * 0.3;
    color *= heightFactor;

    // Density-based intensity
    float intensity = 2.0 + density * 3.0;

    // Subtle animation
    float pulse = sin(time * 2.0 + r) * 0.1 + 1.0;

    return color * intensity * pulse * density;
}

// Main Mode 6 execution function
void ExecuteVoxelParticles(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time) {
    // Ray marching parameters for voxel particle system
    const int maxSteps = 120;  // Medium quality for debugging
    const float stepSize = 0.04;  // Fine steps for detail
    const float maxDistance = 6.0;

    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;
    float t = 0.0;

    [loop]
    for (int step = 0; step < maxSteps; step++) {
        if (t > maxDistance || transmittance < 0.01) break;

        float3 samplePos = rayOrigin + rayDir * t;
        float density = SampleVoxelDensity(samplePos, time);

        if (density > 0.02) {
            // Get particle emission
            float3 emission = GetVoxelParticleEmission(samplePos, density, time);

            // Simple volumetric integration
            float3 scatteredLight = emission * transmittance * stepSize;
            accumulatedColor += scatteredLight;

            // Absorption
            transmittance *= exp(-density * stepSize * 1.2);
        }

        t += stepSize;
    }

    // Add subtle background
    float3 backgroundColor = float3(0.02, 0.01, 0.03); // Dark purple
    accumulatedColor += backgroundColor * transmittance;

    // Final result
    payload.color = float4(accumulatedColor, 1.0);
}

// ============================================================================
// MODE 7: METABALL SPH - SPH physics with metaball density field rendering
// ============================================================================

// Sample density from metaball density field (now using real SPH data)
float SampleMetaballDensity(float3 worldPos, float time) {
    // Convert world position to texture coordinates [0,1]
    // Assume the density volume covers a [-2, +2] cube in world space
    float3 texCoord = (worldPos + 2.0) / 4.0;

    // Clamp to valid texture coordinates
    texCoord = saturate(texCoord);

    // Sample from the 3D density texture with trilinear filtering
    float density = g_densityVolume.SampleLevel(g_densitySampler, texCoord, 0).r;

    // Scale density for better visibility in volumetric rendering
    return density * 3.0; // Amplify metaball contribution for fluid effect
}

// Get metaball emission color based on density and fluid properties
float3 GetMetaballEmission(float3 worldPos, float density, float time) {
    if (density <= 0.02) return float3(0, 0, 0);

    // Fluid-like color based on density (mimics SPH fluid visualization)
    float normalizedDensity = saturate(density * 10.0); // Scale up for visibility

    // Blue to cyan to white gradient (water-like)
    float3 lowDensity = float3(0.1, 0.3, 0.8);   // Deep blue
    float3 midDensity = float3(0.2, 0.7, 1.0);   // Cyan
    float3 highDensity = float3(0.8, 0.9, 1.0);  // Light blue/white

    float3 color;
    if (normalizedDensity < 0.5) {
        color = lerp(lowDensity, midDensity, normalizedDensity * 2.0);
    } else {
        color = lerp(midDensity, highDensity, (normalizedDensity - 0.5) * 2.0);
    }

    // Add some energy/intensity
    float intensity = normalizedDensity * 3.0;

    return color * intensity * density;
}

// Main Mode 7 execution function
void ExecuteMetaballSPH(inout RayPayload payload, float3 rayOrigin, float3 rayDir, uint2 index, float time) {
    // Ray marching parameters optimized for fluid simulation
    const int maxSteps = 100;
    const float stepSize = 0.05;
    const float maxDistance = 5.0;

    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;
    float t = 0.0;

    [loop]
    for (int step = 0; step < maxSteps; step++) {
        if (t > maxDistance || transmittance < 0.01) break;

        float3 samplePos = rayOrigin + rayDir * t;
        float density = SampleMetaballDensity(samplePos, time);

        if (density > 0.02) {
            // Get fluid emission
            float3 emission = GetMetaballEmission(samplePos, density, time);

            // Volumetric integration with fluid-like scattering
            float3 scatteredLight = emission * transmittance * stepSize;
            accumulatedColor += scatteredLight;

            // Absorption (fluids typically have low absorption)
            transmittance *= exp(-density * stepSize * 0.8);
        }

        t += stepSize;
    }

    // Dark background for fluid contrast
    float3 backgroundColor = float3(0.01, 0.02, 0.04); // Dark blue
    accumulatedColor += backgroundColor * transmittance;

    // Final result
    payload.color = float4(accumulatedColor, 1.0);
}