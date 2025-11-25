// 3D Gaussian Splatting Ray Tracing
// Replaces billboard rasterization with volumetric ray-traced Gaussians
// Proper depth sorting, transparency, and volumetric appearance

#include "gaussian_common.hlsl"
#include "plasma_emission.hlsl"
// NOTE: volumetric_shadows.hlsl inlined below after resource declarations (Phase 0.15.0)
// NOTE: god_rays.hlsl removed - atmospheric fog function defined inline below for Light struct access
// NOTE: sample_froxel_grid.hlsl included AFTER resource declarations (needs g_froxelLightingGrid, constant buffer params)

// Match C++ ParticleRenderer_Gaussian::RenderConstants structure
cbuffer GaussianConstants : register(b0)
{
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
    float3 cameraPos;
    float particleRadius;          // baseParticleRadius
    float3 cameraRight;
    float time;
    float3 cameraUp;
    uint screenWidth;
    float3 cameraForward;
    uint screenHeight;
    float fovY;
    float aspectRatio;
    uint particleCount;
    float padding;

    uint usePhysicalEmission;
    float emissionStrength;
    uint useDopplerShift;
    float dopplerStrength;
    uint useGravitationalRedshift;
    float redshiftStrength;
    float emissionBlendFactor;  // 0.0 = pure artistic, 1.0 = pure physical
    float padding2;

    uint useShadowRays;
    uint useInScattering;
    uint usePhaseFunction;
    float phaseStrength;
    float inScatterStrength;
    float rtLightingStrength;
    uint useAnisotropicGaussians;
    float anisotropyStrength;


    // Multi-light system
    uint lightCount;               // Number of active lights (0-16)
    float3 padding3;               // Padding for alignment

    // PCSS soft shadow system
    uint shadowRaysPerLight;       // 1 (performance), 4 (balanced), 8 (quality)
    uint enableTemporalFiltering;  // Temporal accumulation for soft shadows
    float temporalBlend;           // Blend factor for temporal filtering (0.0-1.0)
    uint useRTXDI;                 // 0=multi-light (13 lights), 1=RTXDI (1 sampled light)
    uint debugRTXDISelection;      // DEBUG: Visualize selected light index (0=off, 1=on)
    float3 debugPadding;           // Padding for alignment

    // God ray system (Phase 5 Milestone 5.3c) - DEPRECATED, replaced by froxel system
    float godRayDensity;           // Global god ray density (0.0-1.0) - UNUSED
    float godRayStepMultiplier;    // Ray march step multiplier (0.5-2.0) - UNUSED
    float2 godRayPadding;          // Padding for alignment

// Froxel volumetric fog system (Phase 5 - DEPRECATED)
    // Constants removed
    float froxelPaddingPlaceholder[12]; // Keep alignment if needed or remove entirely if C++ struct updated

    // Phase 1 Lighting Fix
    float rtMinAmbient;            // Global ambient term (0.0-0.2)
    float3 lightingPadding;        // Padding for alignment

    // Phase 1.5 Adaptive Particle Radius
    uint enableAdaptiveRadius;     // Toggle for density/distance-based radius scaling
    float adaptiveInnerZone;       // Distance threshold for inner shrinking (0-200 units)
    float adaptiveOuterZone;       // Distance threshold for outer expansion (200-600 units)
    float adaptiveInnerScale;      // Min scale for inner dense regions (0.1-1.0)
    float adaptiveOuterScale;      // Max scale for outer sparse regions (1.0-3.0)
    float densityScaleMin;         // Min density scale clamp (0.1-1.0)
    float densityScaleMax;         // Max density scale clamp (1.0-5.0)
    float adaptivePadding;         // Padding for alignment

    // Volumetric RT Lighting (Phase 3.9)
    uint volumetricRTSamples;      // Number of light rays per sample point (4-32)
    float volumetricRTDistance;    // Max distance to search for emitters (100-1000)
    float volumetricRTAttenuation; // Attenuation factor for distance falloff (0.00001-0.001)

    // Probe Grid System (Phase 0.13.1)
    uint useProbeGrid;             // Toggle probe grid lighting (replaces volumetric ReSTIR)
    float3 probeGridPadding2;      // Padding for alignment
    uint useVolumetricRT;          // Toggle: 0=legacy per-particle, 1=volumetric per-sample
    float volumetricRTIntensity;   // Intensity boost for particle emission (50-500)
    float3 volumetricRTPadding;    // Padding for GPU alignment

    // === Phase 2: Screen-Space Contact Shadows ===
    uint useScreenSpaceShadows;    // Toggle screen-space shadow system
    uint ssSteps;                  // Ray march steps (8=fast, 16=balanced, 32=quality)
    uint debugScreenSpaceShadows;  // Debug visualization (0=off, 1=show shadow coverage)
    float ssPadding;               // Padding for alignment

    // === Ground Plane (Reflective Surface Experiment) ===
    uint enableGroundPlane;        // Toggle ground plane rendering
    float3 groundPlaneAlbedo;      // Surface reflectance (RGB)
};

// ============================================================================
// SPRINT 1: MATERIAL SYSTEM (Phase 2)
// ============================================================================

// Material properties for each material type (matches C++ MaterialTypeProperties)
struct MaterialTypeProperties {
    float3 albedo;                  // 12 bytes - Surface/volume color (RGB)
    float opacity;                  // 4 bytes  - Opacity multiplier (0-1)
    float emissionMultiplier;       // 4 bytes  - Emission strength multiplier
    float scatteringCoefficient;    // 4 bytes  - Volumetric scattering coefficient
    float phaseG;                   // 4 bytes  - Henyey-Greenstein phase function parameter (-1 to 1)
    float padding[9];               // 36 bytes - Padding to 64 bytes
};  // Total: 64 bytes per material

// Material properties constant buffer (5 material types × 64 bytes = 320 bytes)
cbuffer MaterialProperties : register(b1)
{
    MaterialTypeProperties g_materials[5];  // PLASMA, STAR, GAS_CLOUD, ROCKY, ICY
};

// Light structure for multi-light system (64 bytes with god ray parameters)
struct Light {
    // Base properties (32 bytes)
    float3 position;               // 12 bytes
    float intensity;               // 4 bytes
    float3 color;                  // 12 bytes
    float radius;                  // 4 bytes (for soft shadows)

    // God ray parameters (32 bytes)
    float enableGodRays;          // 4 bytes (0.0=disabled, 1.0=enabled)
    float godRayIntensity;        // 4 bytes
    float godRayLength;           // 4 bytes
    float godRayFalloff;          // 4 bytes
    float3 godRayDirection;       // 12 bytes (normalized)
    float godRayConeAngle;        // 4 bytes (half-angle in radians)
    float godRayRotationSpeed;    // 4 bytes (rad/s)
    float _padding;               // 4 bytes (GPU alignment)
    // Total: 64 bytes
};

// Light array (after constant buffer to avoid size issues)
StructuredBuffer<Light> g_lights : register(t4);

// PCSS shadow buffers (temporal filtering for soft shadows)
Texture2D<float> g_prevShadow : register(t5);  // Previous frame shadow (read-only)

// RTXDI: Selected light indices per pixel (optional - only when RTXDI enabled)
// R channel: asfloat(lightIndex) - 0-15 or 0xFFFFFFFF if no lights
// G/B channels: debug data (cell index, light count)
Texture2D<float4> g_rtxdiOutput : register(t6);

// TEMPORAL COLOR ACCUMULATION (Priority 1 Fix - eliminates flashing)
// Previous frame color for temporal stability (read-only)
Texture2D<float4> g_prevColor : register(t9);

// ============================================================================
// FROXEL VOLUMETRIC FOG SYSTEM (Phase 5 - DEPRECATED)
// ============================================================================
// Texture3D<float4> g_froxelLightingGrid : register(t10); // REMOVED
SamplerState g_linearClampSampler : register(s0);  // Linear clamp for smooth interpolation

// ============================================================================
// PROBE GRID SYSTEM (Phase 0.13.1)
// ============================================================================
// Hybrid Probe Grid: Pre-computed lighting at sparse 32³ grid for zero atomic contention
// Particles interpolate between nearest 8 probes using trilinear sampling

// Probe structure (matches C++ ProbeGridSystem::Probe)
struct Probe {
    float3 position;              // World-space probe location (12 bytes)
    uint lastUpdateFrame;         // Frame when last updated (4 bytes)
    float3 irradiance[9];         // SH L2 irradiance (9 × 12 bytes = 108 bytes)
    uint padding[1];              // Align to 128 bytes
};

// Probe grid parameters (constant buffer)
cbuffer ProbeGridParams : register(b4)
{
    float3 gridMin;               // Grid world-space minimum [-1500, -1500, -1500]
    float gridSpacing;            // Distance between probes (93.75 units)
    uint gridSize;                // Grid dimension (32)
    uint totalProbes;             // Total probe count (32,768)
    uint2 probeGridPadding;       // Padding for alignment
};

// Probe buffer (structured buffer)
StructuredBuffer<Probe> g_probeGrid : register(t7);

// === Phase 2: Screen-Space Shadow Depth Buffer ===
// Depth buffer from pre-pass (R32_UINT storing float depth bits)
Texture2D<uint> g_shadowDepth : register(t8);

// Derived values
static const float2 resolution = float2(screenWidth, screenHeight);
static const float2 invResolution = 1.0 / resolution;
static const float baseParticleRadius = particleRadius;
static const float volumeStepSize = 0.1;         // Finer steps for better quality
static const float densityMultiplier = 2.0;      // Increased for more volumetric appearance
static const float shadowBias = 0.01;            // Bias for shadow ray origin

// Froxel volumetric fog sampling (Phase 5 - REMOVED)
// #include "../froxel/sample_froxel_grid.hlsl"

// Input: Particle buffer
StructuredBuffer<Particle> g_particles : register(t0);

// Input: RT lighting (from particle-to-particle lighting pass)
StructuredBuffer<float4> g_rtLighting : register(t1);

// Input: Ray tracing acceleration structure
RaytracingAccelerationStructure g_particleBVH : register(t2);

// Output: Final rendered image
RWTexture2D<float4> g_output : register(u0);

// PCSS: Current frame shadow buffer (write-only)
RWTexture2D<float> g_currShadow : register(u2);

// TEMPORAL COLOR ACCUMULATION: Current frame color (for next frame's temporal blend)
RWTexture2D<float4> g_currColor : register(u3);

// RT DEPTH BUFFER: Output hit distance for RTXDI temporal reprojection (Phase 4 M5 fix)
// Stores tNear (first hit distance) for depth-based world position reconstruction
RWTexture2D<float> g_rtDepth : register(u4);

// ==============================================================================
// VOLUMETRIC RAYTRACED SHADOW SYSTEM (Phase 0.15.0)
// ==============================================================================
// Replaces PCSS with proper volumetric self-shadowing via DXR 1.1 RayQuery
//
// This system provides physically accurate volumetric shadows for 3D Gaussian
// particles using inline ray tracing (RayQuery API). Unlike PCSS which treats
// particles as surfaces, this computes true volumetric attenuation using
// Beer-Lambert law as rays traverse through semi-transparent particle volumes.
//
// Key Features:
// - Volumetric attenuation via Beer-Lambert absorption
// - Soft shadows through area light sampling (Poisson disk)
// - Temporal accumulation for noise reduction (67ms convergence)
// - Early ray termination optimization
// - Temperature-based density for physically accurate occlusion
//
// Performance:
// - Performance (1 ray/light): ~115 FPS @ 10K particles
// - Balanced (4 rays/light):   ~92 FPS @ 10K particles
// - Quality (8 rays/light):    ~65 FPS @ 10K particles
// ==============================================================================

// Shadow bias to avoid self-intersection
static const float SHADOW_BIAS = 0.01;

// Maximum particle hits per shadow ray (performance vs quality)
static const uint MAX_SHADOW_HITS = 8;

// Opacity threshold for early ray termination
static const float SHADOW_OPACITY_THRESHOLD = 0.99;

//------------------------------------------------------------------------------
// Volumetric Shadow Occlusion (Single Ray)
// Casts one shadow ray and accumulates volumetric opacity through particles
//------------------------------------------------------------------------------
float ComputeVolumetricShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float3 lightDir,
    float  lightDistance)
{
    // Setup shadow ray
    RayDesc shadowRay;
    shadowRay.Origin = particlePos + lightDir * SHADOW_BIAS;  // Offset to avoid self-intersection
    shadowRay.Direction = lightDir;
    shadowRay.TMin = SHADOW_BIAS;
    shadowRay.TMax = lightDistance - SHADOW_BIAS;

    // Initialize ray query for inline raytracing
    // NOTE: Do NOT use RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES - we WANT to hit particles!
    RayQuery<RAY_FLAG_NONE> query;

    query.TraceRayInline(
        g_particleBVH,             // Reuse existing TLAS
        RAY_FLAG_NONE,
        0xFF,                      // Instance inclusion mask
        shadowRay
    );

    // Accumulate shadow opacity through volume
    float shadowOpacity = 0.0;
    uint hitCount = 0;

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint particleIndex = query.CandidatePrimitiveIndex();
            Particle occluder = g_particles[particleIndex];

            // Compute particle radius (same as rendering code)
            float3 scale = ComputeGaussianScale(
                occluder,
                baseParticleRadius,
                useAnisotropicGaussians != 0,
                anisotropyStrength,
                enableAdaptiveRadius != 0,
                adaptiveInnerZone,
                adaptiveOuterZone,
                adaptiveInnerScale,
                adaptiveOuterScale,
                densityScaleMin,
                densityScaleMax
            );
            float occluderRadius = max(scale.x, max(scale.y, scale.z));  // Use largest axis

            // Calculate intersection distance through particle volume
            float tHit = query.CandidateTriangleRayT();
            float distThroughParticle = min(occluderRadius * 2.0, shadowRay.TMax - tHit);

            // Beer-Lambert law: I = I0 * exp(-density * distance)
            // Use temperature as proxy for density (hotter particles = denser = darker shadows)
            float density = saturate(occluder.temperature / 26000.0);

            // Volumetric attenuation through this particle
            float attenuation = 1.0 - exp(-density * distThroughParticle * 0.5);

            // Accumulate opacity (blend with existing opacity)
            shadowOpacity += attenuation * (1.0 - shadowOpacity);

            // Commit this procedural primitive hit
            query.CommitProceduralPrimitiveHit(tHit);

            hitCount++;

            // Early out if fully occluded or hit limit reached
            if (shadowOpacity >= SHADOW_OPACITY_THRESHOLD || hitCount >= MAX_SHADOW_HITS)
            {
                query.Abort();
                break;
            }
        }
    }

    // Return shadow factor (0 = fully shadowed, 1 = fully lit)
    return 1.0 - saturate(shadowOpacity);
}

//------------------------------------------------------------------------------
// Soft Shadow Sampling (Multi-ray with Poisson Disk)
// Casts multiple rays with jittered offsets for soft shadow penumbra
//------------------------------------------------------------------------------
float ComputeSoftShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float  lightRadius,
    uint2  pixelCoord,
    uint   frameCount,
    uint   numRays)
{
    // Calculate light direction and distance
    float3 toLight = lightPos - particlePos;
    float lightDistance = length(toLight);
    float3 lightDir = toLight / lightDistance;

    // Single ray for performance mode
    if (numRays == 1)
    {
        return ComputeVolumetricShadowOcclusion(
            particlePos,
            lightPos,
            lightDir,
            lightDistance
        );
    }

    // Multi-ray soft shadows with Poisson disk sampling
    float shadowSum = 0.0;

    // Poisson disk offsets for area light sampling (16 samples)
    static const float2 poissonDisk[16] = {
        float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725),
        float2(-0.09418410, -0.92938870), float2(0.34495938,  0.29387760),
        float2(-0.91588581,  0.45771432), float2(-0.81544232, -0.87912464),
        float2(-0.38277543,  0.27676845), float2(0.97484398,  0.75648379),
        float2(0.44323325, -0.97511554), float2(0.53742981, -0.47373420),
        float2(-0.26496911, -0.41893023), float2(0.79197514,  0.19090188),
        float2(-0.24188840,  0.99706507), float2(-0.81409955,  0.91437590),
        float2(0.19984126,  0.78641367), float2(0.14383161, -0.14100790)
    };

    // Build tangent space for light direction
    float3 tangent = abs(lightDir.y) < 0.999 ?
                     normalize(cross(lightDir, float3(0, 1, 0))) :
                     normalize(cross(lightDir, float3(1, 0, 0)));
    float3 bitangent = cross(lightDir, tangent);

    // Temporal rotation for sample distribution (reduces noise over time)
    float rotation = (frameCount % 16) * (3.14159265 / 8.0);
    float cosRot = cos(rotation);
    float sinRot = sin(rotation);

    // Cast multiple shadow rays
    [unroll(8)]  // Maximum expected rays (Quality preset)
    for (uint i = 0; i < numRays; i++)
    {
        // Apply temporal rotation to Poisson disk sample
        float2 offset = poissonDisk[i % 16];
        float2 rotatedOffset = float2(
            offset.x * cosRot - offset.y * sinRot,
            offset.x * sinRot + offset.y * cosRot
        );

        // Jitter light position for area light effect
        float3 jitteredLightPos = lightPos +
                                   tangent * (rotatedOffset.x * lightRadius) +
                                   bitangent * (rotatedOffset.y * lightRadius);

        float3 toJitteredLight = jitteredLightPos - particlePos;
        float3 jitteredLightDir = normalize(toJitteredLight);
        float jitteredLightDist = length(toJitteredLight);

        // Cast shadow ray
        float shadowFactor = ComputeVolumetricShadowOcclusion(
            particlePos,
            jitteredLightPos,
            jitteredLightDir,
            jitteredLightDist
        );

        shadowSum += shadowFactor;
    }

    // Average shadow factor across samples
    return shadowSum / float(numRays);
}

//------------------------------------------------------------------------------
// Temporal Shadow Accumulation
// Blends current shadow with previous frame for temporal stability
// Uses ping-pong buffers (t5: prev, u2: curr)
//------------------------------------------------------------------------------
float ApplyTemporalAccumulation(
    uint2 pixelCoord,
    float currentShadow,
    float blendFactor)
{
    // Read previous frame shadow
    float prevShadow = g_prevShadow[pixelCoord];

    // Temporal blend (default 0.1 = 10% new sample, 90% history = 67ms convergence @ 120 FPS)
    float finalShadow = lerp(prevShadow, currentShadow, blendFactor);

    // Write to current shadow buffer (will be prev next frame)
    g_currShadow[pixelCoord] = finalShadow;

    return finalShadow;
}

//------------------------------------------------------------------------------
// Main Shadow API
// Computes volumetric shadow with soft shadows and temporal filtering
//------------------------------------------------------------------------------
float CastVolumetricShadowRay(
    float3 particlePos,
    float3 lightPos,
    float  lightRadius,
    uint2  pixelCoord,
    uint   numRays,
    uint   frameCount,
    bool   useTemporalFiltering,
    float  temporalBlendFactor)
{
    // Compute shadow occlusion (single or multi-ray)
    float shadowFactor = ComputeSoftShadowOcclusion(
        particlePos,
        lightPos,
        lightRadius,
        pixelCoord,
        frameCount,
        numRays
    );

    // Apply temporal accumulation if enabled
    if (useTemporalFiltering)
    {
        shadowFactor = ApplyTemporalAccumulation(
            pixelCoord,
            shadowFactor,
            temporalBlendFactor
        );
    }

    return shadowFactor;
}

// =============================================================================
// SPHERICAL HARMONICS L2 RECONSTRUCTION
// =============================================================================

/**
 * Evaluate SH L2 basis functions (same as update_probes.hlsl)
 */
void EvaluateSH_L2(float3 dir, out float sh[9]) {
    dir = normalize(dir);
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;

    sh[0] = 0.282095;                      // Band 0
    sh[1] = 0.488603 * y;                  // Band 1
    sh[2] = 0.488603 * z;
    sh[3] = 0.488603 * x;
    sh[4] = 1.092548 * x * y;              // Band 2
    sh[5] = 1.092548 * y * z;
    sh[6] = 0.315392 * (3.0 * z2 - 1.0);
    sh[7] = 1.092548 * x * z;
    sh[8] = 0.546274 * (x2 - y2);
}

/**
 * Reconstruct irradiance from SH coefficients for a given direction
 *
 * @param shCoeffs Array of 9 RGB SH coefficients from a probe
 * @param direction Normalized direction to evaluate lighting
 * @return Reconstructed RGB irradiance for that direction
 */
float3 ReconstructSH_L2(float3 shCoeffs[9], float3 direction) {
    // Evaluate basis functions for this direction
    float shBasis[9];
    EvaluateSH_L2(direction, shBasis);

    // Dot product: sum of (coefficient × basis) for all 9 terms
    float3 irradiance = float3(0, 0, 0);
    for (uint i = 0; i < 9; i++) {
        irradiance += shCoeffs[i] * shBasis[i];
    }

    return irradiance;
}

// =============================================================================
// PROBE GRID SAMPLING (Phase 0.13.1)
// =============================================================================
/**
 * Sample probe grid irradiance using trilinear interpolation with FULL SH L2 reconstruction
 *
 * Replaces Volumetric ReSTIR (which suffered from atomic contention at ≥2045 particles)
 * Zero atomic operations = zero contention = scales to 10K+ particles!
 *
 * Algorithm:
 * 1. Convert world position to grid coordinates
 * 2. Find 8 nearest probes (corner of grid cell)
 * 3. Reconstruct directional irradiance from SH L2 coefficients for each probe
 * 4. Trilinear interpolation between reconstructed values
 *
 * @param worldPos World-space position to sample lighting at
 * @param viewDir Normalized view direction (for directional scattering)
 * @return Irradiance (RGB color) at the given position
 */
float3 SampleProbeGrid(float3 worldPos, float3 viewDir) {
    // Convert world position to grid coordinates
    float3 gridCoord = (worldPos - gridMin) / gridSpacing;

    // Clamp to valid grid bounds [0, gridSize-1]
    gridCoord = clamp(gridCoord, float3(0, 0, 0), float3(gridSize - 1, gridSize - 1, gridSize - 1));

    // Base grid index (integer part)
    int3 gridIdx0 = int3(floor(gridCoord));
    int3 gridIdx1 = min(gridIdx0 + int3(1, 1, 1), int3(gridSize - 1, gridSize - 1, gridSize - 1));

    // Interpolation weights (fractional part)
    float3 t = frac(gridCoord);

    // Fetch 8 corner probes (trilinear cube)
    // Linear index formula: x + y*gridSize + z*gridSize²
    uint stride = gridSize;
    uint strideZ = gridSize * gridSize;

    uint idx000 = gridIdx0.x + gridIdx0.y * stride + gridIdx0.z * strideZ;
    uint idx001 = gridIdx0.x + gridIdx0.y * stride + gridIdx1.z * strideZ;
    uint idx010 = gridIdx0.x + gridIdx1.y * stride + gridIdx0.z * strideZ;
    uint idx011 = gridIdx0.x + gridIdx1.y * stride + gridIdx1.z * strideZ;
    uint idx100 = gridIdx1.x + gridIdx0.y * stride + gridIdx0.z * strideZ;
    uint idx101 = gridIdx1.x + gridIdx0.y * stride + gridIdx1.z * strideZ;
    uint idx110 = gridIdx1.x + gridIdx1.y * stride + gridIdx0.z * strideZ;
    uint idx111 = gridIdx1.x + gridIdx1.y * stride + gridIdx1.z * strideZ;

    // Bounds check (safety against out-of-bounds access)
    if (idx000 >= totalProbes || idx001 >= totalProbes ||
        idx010 >= totalProbes || idx011 >= totalProbes ||
        idx100 >= totalProbes || idx101 >= totalProbes ||
        idx110 >= totalProbes || idx111 >= totalProbes) {
        return float3(0, 0, 0);  // Out of bounds - return black
    }

    // FULL SH L2 RECONSTRUCTION for directional lighting!
    // For each of the 8 corner probes, reconstruct irradiance for the view direction
    // This enables proper Henyey-Greenstein phase function scattering!

    float3 c000 = ReconstructSH_L2(g_probeGrid[idx000].irradiance, viewDir);
    float3 c001 = ReconstructSH_L2(g_probeGrid[idx001].irradiance, viewDir);
    float3 c010 = ReconstructSH_L2(g_probeGrid[idx010].irradiance, viewDir);
    float3 c011 = ReconstructSH_L2(g_probeGrid[idx011].irradiance, viewDir);
    float3 c100 = ReconstructSH_L2(g_probeGrid[idx100].irradiance, viewDir);
    float3 c101 = ReconstructSH_L2(g_probeGrid[idx101].irradiance, viewDir);
    float3 c110 = ReconstructSH_L2(g_probeGrid[idx110].irradiance, viewDir);
    float3 c111 = ReconstructSH_L2(g_probeGrid[idx111].irradiance, viewDir);

    // Trilinear interpolation
    // First interpolate along X axis (4 lerps)
    float3 c00 = lerp(c000, c100, t.x);
    float3 c01 = lerp(c001, c101, t.x);
    float3 c10 = lerp(c010, c110, t.x);
    float3 c11 = lerp(c011, c111, t.x);

    // Interpolate along Y axis (2 lerps)
    float3 c0 = lerp(c00, c10, t.y);
    float3 c1 = lerp(c01, c11, t.y);

    // Final interpolation along Z axis (1 lerp)
    float3 finalIrradiance = lerp(c0, c1, t.z);

    return finalIrradiance;
}

// ============================================================================
// ATMOSPHERIC FOG RAY MARCHING - Volumetric God Rays
// ============================================================================
// Marches through UNIFORM ATMOSPHERIC FOG at regular intervals,
// independent of particle positions. This creates visible light shafts
// even in empty space, just like real fog/dust scattering sunlight.
float3 RayMarchAtmosphericFog(
    float3 cameraPos,
    float3 rayDir,
    float maxDistance,
    StructuredBuffer<Light> lights,
    uint lightCount,
    float totalTime,
    float godRayDensity,
    RaytracingAccelerationStructure accelStructure
) {
    // Early exit if god rays globally disabled
    if (godRayDensity < 0.001) {
        return float3(0, 0, 0);
    }

    // Configuration
    const uint NUM_STEPS = 32;  // 32 steps = good quality/performance balance
    const float stepSize = maxDistance / float(NUM_STEPS);

    float3 totalFogColor = float3(0, 0, 0);

    // Ray March Loop
    for (uint step = 0; step < NUM_STEPS; step++) {
        // Current position along ray (sample at step center for better accuracy)
        float t = (float(step) + 0.5) * stepSize;
        float3 samplePos = cameraPos + rayDir * t;

        // Sample all lights at this fog position
        for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
            Light light = lights[lightIdx];

            // Skip if this light has god rays disabled
            if (light.enableGodRays < 0.5) {
                continue;
            }

            // Calculate direction and distance to light
            float3 toLight = light.position - samplePos;
            float distToLight = length(toLight);

            // Skip if outside light's god ray range
            if (distToLight < 0.001 || distToLight > light.godRayLength) {
                continue;
            }

            float3 lightDir = toLight / distToLight;

            // Beam Direction (with optional rotation)
            float3 beamDir = light.godRayDirection;
            if (abs(light.godRayRotationSpeed) > 0.001) {
                float rotationAngle = light.godRayRotationSpeed * totalTime;
                // Rotate around Y-axis
                float c = cos(rotationAngle);
                float s = sin(rotationAngle);
                beamDir = float3(
                    beamDir.x * c - beamDir.z * s,
                    beamDir.y,
                    beamDir.x * s + beamDir.z * c
                );
            }

            // Cone Volume Test
            float alignment = dot(lightDir, beamDir);
            float coneThreshold = cos(light.godRayConeAngle);

            if (alignment < coneThreshold) {
                continue;  // Outside cone, skip this light
            }

            // Radial Falloff (Gaussian beam shape)
            float axisDistance = distToLight * sqrt(max(0.0, 1.0 - alignment * alignment));
            float radialFalloff = exp(-axisDistance * light.godRayFalloff);

            // Distance Attenuation
            float distanceFalloff = 1.0 / (1.0 + distToLight * distToLight * 0.0001);

            // Shadow Ray (particles attenuate fog based on opacity)
            // This allows god rays to penetrate through the particle cloud!
            RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

            RayDesc shadowRay;
            shadowRay.Origin = samplePos + lightDir * 0.1;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = distToLight - 0.1;

            q.TraceRayInline(accelStructure, RAY_FLAG_NONE, 0xFF, shadowRay);
            q.Proceed();

            // Calculate light transmission through particles
            // Instead of completely blocking, particles attenuate based on their density
            float transmission = 1.0; // Start with full light transmission

            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                // Read the particle that was hit by shadow ray
                uint hitParticleIdx = q.CommittedPrimitiveIndex();
                Particle hitParticle = g_particles[hitParticleIdx];

                // Calculate particle opacity based on density
                // Density ranges from ~0.01 to ~2.0 in accretion disk
                // Scale to reasonable opacity: low density = transparent, high density = opaque
                float particleOpacity = saturate(hitParticle.density * 0.3);

                // Transmission = how much light passes through
                // Low opacity → high transmission (light shines through!)
                // High opacity → low transmission (light dimmed but not blocked)
                transmission = 1.0 - particleOpacity;

                // CRITICAL: Add minimum transmission so god rays always penetrate somewhat
                // This ensures light shafts are visible even through dense regions
                transmission = max(transmission, 0.2); // At least 20% light gets through
            }

            // Calculate Scattering Contribution (now modulated by transmission)
            // Transmission of 1.0 = full brightness (empty space or transparent particle)
            // Transmission of 0.2 = 20% brightness (dense particle but still visible!)
            float scatteringStrength = light.godRayIntensity * radialFalloff * distanceFalloff * godRayDensity * transmission;
            float3 scatteringColor = light.color * scatteringStrength;

            // Accumulate fog color (volumetric integral)
            // Now god rays will be visible THROUGH the particle cloud!
            totalFogColor += scatteringColor * stepSize;
        }

        // === ADDED: Indirect/Ambient Volumetric Lighting from Probe Grid ===
        // If probe grid is enabled, add its contribution to the fog
        // This illuminates the "air" with bounced light, not just direct shafts
        if (useProbeGrid != 0) {
             // Sample probe grid for ambient/indirect lighting
             // We pass rayDir (view direction) to reconstruct directional irradiance
             float3 indirect = SampleProbeGrid(samplePos, rayDir); 
             
             // Scale by fog density and a tuning factor (0.05 = 5% ambient strength)
             // This ensures the fog isn't too bright but picks up the color of the environment
             float3 indirectScatter = indirect * godRayDensity * 0.05;
             
             totalFogColor += indirectScatter * stepSize;
        }
    }

    return totalFogColor;
}

// Hit record for batch processing
struct HitRecord {
    uint particleIdx;
    float tNear;
    float tFar;
    float sortKey; // For depth sorting
};

// Volumetric lighting parameters
struct VolumetricParams {
    float3 lightPos;       // Primary light position (e.g., black hole center)
    float3 lightColor;     // Light color/intensity
    float scatteringG;     // Henyey-Greenstein g parameter (-1 to 1, 0=isotropic)
    float extinction;      // Extinction coefficient for shadows
};

// Cast single shadow ray to check occlusion (returns transmittance 0-1)
float CastSingleShadowRay(float3 origin, float3 direction, float maxDist) {
    RayDesc shadowRay;
    shadowRay.Origin = origin + direction * shadowBias;
    shadowRay.Direction = direction;
    shadowRay.TMin = 0.001;
    shadowRay.TMax = maxDist;

    float transmittance = 1.0;

    // Use RayQuery to accumulate optical depth along shadow ray
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
    shadowQuery.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, shadowRay);

    while (shadowQuery.Proceed()) {
        if (shadowQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = shadowQuery.CandidatePrimitiveIndex();

            // Simple occlusion test - if we hit any particle, we're in shadow
            shadowQuery.CommitProceduralPrimitiveHit(0.5);
            transmittance *= 0.3; // Partial occlusion
        }
    }

    if (shadowQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        return transmittance;
    }

    return 1.0; // No occlusion
}

// ===========================================================================================
// Phase 2: Screen-Space Contact Shadow Ray March
// ===========================================================================================
// Samples depth buffer in screen space to detect occlusion between sample point and light.
// This is lighting-agnostic - works with probe grid, inline RQ, multi-light, RTXDI.
//
// INPUTS:
//   worldPos:  Current sample point in world space
//   lightDir:  Normalized direction to light source
//   maxDist:   Maximum ray march distance
//   numSteps:  Quality setting (8=performance, 16=balanced, 32=quality)
//
// RETURNS:
//   Shadow factor [0, 1]: 0=fully shadowed, 1=fully lit
//
// PERFORMANCE: ~0.3ms @ 16 steps, 13 lights, 1440p
float ScreenSpaceShadow(float3 worldPos, float3 lightDir, float maxDist, uint numSteps) {
    // 1. Transform world position to clip space
    float4 clipPos = mul(float4(worldPos, 1.0), viewProj);

    // Reject if behind camera
    if (clipPos.w <= 0.0) {
        return 1.0;  // No shadow (invalid)
    }

    // Perspective divide to get NDC
    float3 ndcPos = clipPos.xyz / clipPos.w;

    // 2. Transform light end point to clip space
    float3 lightEndWorld = worldPos + lightDir * maxDist;
    float4 clipLightEnd = mul(float4(lightEndWorld, 1.0), viewProj);

    // Reject if light end behind camera
    if (clipLightEnd.w <= 0.0) {
        return 1.0;  // No shadow (invalid)
    }

    float3 ndcLightEnd = clipLightEnd.xyz / clipLightEnd.w;

    // 3. Convert NDC to screen UV coordinates
    // NDC: [-1, 1] → UV: [0, 1]
    float2 screenUV = float2(
        (ndcPos.x + 1.0) * 0.5,
        (1.0 - ndcPos.y) * 0.5  // Flip Y (NDC +Y is up, screen +Y is down)
    );

    float2 screenUVEnd = float2(
        (ndcLightEnd.x + 1.0) * 0.5,
        (1.0 - ndcLightEnd.y) * 0.5
    );

    // 4. Ray march in screen space
    float2 rayDir = screenUVEnd - screenUV;
    float rayLength = length(rayDir);

    // Early out if ray is too short
    if (rayLength < 0.001) {
        return 1.0;  // No shadow (degenerate ray)
    }

    rayDir = normalize(rayDir);
    float stepSize = rayLength / float(numSteps);

    float occlusion = 0.0;
    float totalWeight = 0.0;

    for (uint i = 1; i <= numSteps; i++) {
        // Sample position along ray
        float t = float(i) / float(numSteps);
        float2 sampleUV = screenUV + rayDir * stepSize * float(i);

        // Bounds check
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
            sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            break;  // Ray left screen
        }

        // Convert UV to pixel coordinates
        int2 samplePixel = int2(sampleUV * float2(screenWidth, screenHeight));

        // Sample depth buffer (stored as uint, convert back to float)
        uint depthBits = g_shadowDepth[samplePixel];
        float sceneDepth = asfloat(depthBits);

        // Compute ray depth at this point
        float rayDepth = lerp(ndcPos.z, ndcLightEnd.z, t);

        // Compare depths with bias
        const float depthBias = 0.001;
        if (rayDepth > sceneDepth + depthBias) {
            // Ray is behind scene geometry = occluded
            // Weight by distance from start (closer occlusions matter more)
            float weight = 1.0 - t;
            occlusion += weight;
            totalWeight += weight;
        } else {
            totalWeight += 1.0 - t;
        }
    }

    // 5. Contact hardening: shadows sharper near contact
    // Compute contact factor based on ray length in screen space
    float contactFactor = saturate(rayLength / 0.1);  // 0.1 = full screen width threshold

    // Compute final shadow term
    float shadowTerm = 1.0;
    if (totalWeight > 0.0) {
        float occlusionRatio = occlusion / totalWeight;
        shadowTerm = 1.0 - (occlusionRatio * contactFactor);
    }

    return saturate(shadowTerm);
}

// ============================================================================
// VOLUMETRIC RAYTRACED SHADOW SYSTEM (Phase 0.15.0)
// Replaces PCSS with proper volumetric self-shadowing
// ============================================================================

// Wrapper function to maintain compatibility with existing code
// This calls the new volumetric shadow system from volumetric_shadows.hlsl
float CastPCSSShadowRay(float3 origin, float3 lightPos, float lightRadius, uint2 pixelPos, uint numSamples) {
    // Calculate frame count from time for temporal variation
    // Approximate frame count (assuming ~120 FPS avg)
    uint frameCount = uint(time * 120.0);

    // Call new volumetric shadow system
    return CastVolumetricShadowRay(
        origin,                      // Particle position
        lightPos,                    // Light position
        lightRadius,                 // Light radius for soft shadows
        pixelPos,                    // Pixel coordinate for temporal filtering
        numSamples,                  // Number of shadow rays (1/4/8)
        frameCount,                  // Frame count for temporal rotation
        enableTemporalFiltering != 0, // Enable temporal accumulation
        temporalBlend                // Blend factor (0.1 default)
    );
}

// Compute in-scattering from nearby particles (OPTIMIZED + RUNTIME CONTROLLED)
float3 ComputeInScattering(float3 pos, float3 viewDir, uint skipIdx) {
    float3 totalScattering = float3(0, 0, 0);

    // Adaptive sampling based on distance from camera
    float distFromCamera = length(pos - cameraPos);
    uint numSamples = distFromCamera < 100.0 ? 4 : 2;

    for (uint i = 0; i < numSamples; i++) {
        float phi = (i + 0.5) * 6.28318 / numSamples;
        float3 scatterDir = float3(cos(phi), 0.5, sin(phi));
        scatterDir = normalize(scatterDir);

        RayDesc scatterRay;
        scatterRay.Origin = pos;
        scatterRay.Direction = scatterDir;
        scatterRay.TMin = 0.01;
        scatterRay.TMax = 80.0;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> scatterQuery;
        scatterQuery.TraceRayInline(g_particleBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, scatterRay);
        scatterQuery.Proceed();

        if (scatterQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            uint idx = scatterQuery.CommittedPrimitiveIndex();
            if (idx != skipIdx) {
                Particle p = g_particles[idx];
                float dist = length(p.position - pos);
                float atten = 1.0 / (1.0 + dist * 0.02);
                float3 emission = TemperatureToEmission(p.temperature);
                float intensity = EmissionIntensity(p.temperature);
                float scatterStrength = p.density * 2.0;
                float phase = HenyeyGreenstein(dot(viewDir, scatterDir), 0.5);
                totalScattering += emission * intensity * phase * atten * scatterStrength;
            }
        }
    }
    return totalScattering / numSamples;
}

// ============================================================================
// VOLUMETRIC RT LIGHTING - Per-Sample-Point Evaluation
// ============================================================================
// Evaluates RT lighting at ANY point in space (not just particle centers).
// This creates smooth volumetric scattering exactly like the multi-light system.
// Replaces billboard-era per-particle lookup with continuous volumetric evaluation.
//
// Cost: 8-16 RayQuery traversals per sample (same as particle-to-particle,
//       but evaluated at sample point instead of particle center)
// Benefit: Eliminates discrete jumps, creates smooth volumetric glow
// Phase 3.9: Volumetric RT Lighting with Proper Scattering
// Treats neighbor particles as virtual lights using g_rtLighting[] as intensity
// Applies SAME volumetric math as multi-lights: distance attenuation + phase function + PCSS
float3 InterpolateRTLighting(float3 worldPos, uint skipIdx, float3 viewDir, float phaseG, uint2 pixelPos) {
    float3 totalLight = float3(0, 0, 0);

    // Number of interpolation samples (runtime configurable)
    // 4 = Fast (tetrahedral interpolation)
    // 8 = Balanced (cubic interpolation) - DEFAULT
    // 16 = Smooth (high quality)
    uint numSamples = volumetricRTSamples > 0 ? volumetricRTSamples : 8;

    // Maximum interpolation distance (runtime configurable)
    // Controls the "smoothness radius" - how far to search for neighbors
    // Larger = smoother gradients but blurrier
    // Smaller = sharper transitions but more discrete
    // Default: 200.0 (matches average particle spacing of ~139 units)
    float maxDistance = volumetricRTDistance > 0.0 ? volumetricRTDistance : 200.0;

    // Fibonacci sphere sampling for even spatial distribution
    const float PHI = 1.618033988749895; // Golden ratio

    for (uint i = 0; i < numSamples; i++) {
        // Generate evenly distributed direction (full sphere)
        float theta = 2.0 * 3.14159265359 * i / PHI;
        float phi = acos(1.0 - 2.0 * (i + 0.5) / numSamples);

        float sinPhi = sin(phi);
        float3 sampleDir = normalize(float3(
            cos(theta) * sinPhi,
            sin(theta) * sinPhi,
            cos(phi)
        ));

        // Cast ray to find nearest particle in this direction
        RayDesc probeRay;
        probeRay.Origin = worldPos;
        probeRay.Direction = sampleDir;
        probeRay.TMin = 0.01;  // Small bias to avoid self-intersection
        probeRay.TMax = maxDistance;

        // Inline ray tracing to find nearest particle
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(g_particleBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, probeRay);
        query.Proceed();

        // If we found a neighbor particle, treat it as a virtual light!
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            uint neighborIdx = query.CommittedPrimitiveIndex();

            // Skip self (though unlikely at sample point)
            if (neighborIdx == skipIdx) continue;

            // Get neighbor particle position for volumetric scattering math
            Particle neighbor = g_particles[neighborIdx];
            float3 neighborPos = neighbor.position;

            // Calculate light direction (from sample point TO neighbor)
            float3 lightDir = normalize(neighborPos - worldPos);
            float lightDist = length(neighborPos - worldPos);

            // Use g_rtLighting[] as the "light intensity/color" for this virtual light
            float3 lightColor = g_rtLighting[neighborIdx].rgb;

            // === VOLUMETRIC SCATTERING (same math as multi-lights!) ===

            // 1. Distance attenuation (quadratic falloff, line 844 in multi-light)
            float normalizedDist = lightDist / maxDistance;
            float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

            // 2. Phase function (Henyey-Greenstein for anisotropic scattering)
            float phase = 1.0;
            if (usePhaseFunction != 0) {
                float cosTheta = dot(-viewDir, lightDir);
                phase = HenyeyGreenstein(cosTheta, phaseG);
            }

            // 3. PCSS soft shadow (NEW: integrate with multi-light shadow system!)
            float shadowTerm = 1.0;
            if (useShadowRays != 0) {
                // Treat neighbor particle as light source with radius
                // Use particle radius (particleRadius constant) for soft shadow sampling
                shadowTerm = CastPCSSShadowRay(worldPos, neighborPos, particleRadius, pixelPos, shadowRaysPerLight);
            }

            // 4. Accumulate light contribution with shadow term (same as multi-lights!)
            float3 lightContribution = lightColor * attenuation * phase * shadowTerm;
            totalLight += lightContribution;
        }
    }

    // Return accumulated light (NO AVERAGING - same as multi-lights!)
    // Multi-lights loop and accumulate: totalLighting += lightContribution (line 874)
    // They never divide by light count - we shouldn't either!
    // This is physically correct: each neighbor is an independent light source
    return totalLight;
}

// Generate camera ray from pixel coordinates
RayDesc GenerateCameraRay(float2 pixelPos) {
    // NDC coordinates (-1 to 1)
    float2 ndc = (pixelPos + 0.5) * invResolution * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for D3D

    // Unproject to world space
    float4 nearPoint = mul(float4(ndc, 0.0, 1.0), invViewProj);
    float4 farPoint = mul(float4(ndc, 1.0, 1.0), invViewProj);

    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    RayDesc ray;
    ray.Origin = nearPoint.xyz;
    ray.Direction = normalize(farPoint.xyz - nearPoint.xyz);
    ray.TMin = 0.01;
    ray.TMax = 10000.0;

    return ray;
}

// Insert hit into sorted list (simple insertion sort for small batches)
void InsertHit(inout HitRecord hits[64], inout uint hitCount, uint particleIdx, float tNear, float tFar, uint maxHits) {
    if (hitCount >= maxHits) return;

    HitRecord newHit;
    newHit.particleIdx = particleIdx;
    newHit.tNear = tNear;
    newHit.tFar = tFar;
    newHit.sortKey = tNear; // Sort by entry distance

    // Insertion sort (simple for small batches)
    uint insertPos = hitCount;
    for (uint i = 0; i < hitCount; i++) {
        if (newHit.sortKey < hits[i].sortKey) {
            insertPos = i;
            break;
        }
    }

    // Shift elements
    for (uint i = hitCount; i > insertPos; i--) {
        hits[i] = hits[i - 1];
    }

    hits[insertPos] = newHit;
    hitCount++;
}


[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;

    // Early exit if outside render target
    if (any(pixelPos >= (uint2)resolution))
        return;

    // Generate primary ray
    RayDesc ray = GenerateCameraRay((float2)pixelPos);

    // Collect all Gaussian intersections via RayQuery
    const uint MAX_HITS = 64; // Batch size
    HitRecord hits[MAX_HITS];
    uint hitCount = 0;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    // Process all AABB candidates
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = query.CandidatePrimitiveIndex();

            // Read particle
            Particle p = g_particles[particleIdx];

            // Compute Gaussian parameters (with anisotropic control + adaptive radius)
            float3 scale = ComputeGaussianScale(
                p, baseParticleRadius,
                useAnisotropicGaussians != 0,
                anisotropyStrength,
                enableAdaptiveRadius != 0,
                adaptiveInnerZone,
                adaptiveOuterZone,
                adaptiveInnerScale,
                adaptiveOuterScale,
                densityScaleMin,
                densityScaleMax
            );
            float3x3 rotation = ComputeGaussianRotation(p.velocity);

            // Detailed Gaussian-ellipsoid intersection
            float2 t = RayGaussianIntersection(ray.Origin, ray.Direction, p.position, scale, rotation);

            // Valid intersection?
            if (t.x > ray.TMin && t.x < ray.TMax && t.y > t.x) {
                // Commit the AABB hit (required for procedural primitives)
                query.CommitProceduralPrimitiveHit(t.x);

                // Store in hit list
                InsertHit(hits, hitCount, particleIdx, t.x, t.y, MAX_HITS);
            }
        }
    }

    // =============================================================================
    // GROUND PLANE HIT DETECTION (Reflective Surface Experiment)
    // =============================================================================
    float groundPlaneT = 1e10;   // Distance to ground plane (very far = no hit)
    float3 groundPlaneColor = float3(0, 0, 0);
    bool groundPlaneHit = false;

    if (enableGroundPlane != 0 && query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Check if this is the ground plane (InstanceID == 2)
        if (query.CommittedInstanceID() == 2) {
            groundPlaneHit = true;
            groundPlaneT = query.CommittedRayT();

            // Ground plane hit position and normal (always pointing up)
            float3 hitPos = ray.Origin + ray.Direction * groundPlaneT;
            float3 normal = float3(0, 1, 0);  // Ground plane normal is +Y

            // Calculate diffuse lighting from all lights
            float3 diffuse = float3(0, 0, 0);
            for (uint l = 0; l < lightCount; l++) {
                Light light = g_lights[l];
                float3 toLight = light.position - hitPos;
                float dist = length(toLight);
                float3 L = toLight / dist;

                // Lambertian diffuse
                float NdotL = max(dot(normal, L), 0.0);

                // Distance attenuation
                float attenuation = 1.0 / (1.0 + 0.0001 * dist * dist);

                diffuse += light.color * light.intensity * NdotL * attenuation;
            }

            // Apply ground plane albedo
            groundPlaneColor = groundPlaneAlbedo * diffuse;
        }
    }

    // =============================================================================
    // Volume Rendering Loop
    // =============================================================================
    uint pixelIndex = pixelPos.y * screenWidth + pixelPos.x;

    // Initialize reservoir (always, even if ReSTIR is off)
    // Volume rendering: march through sorted Gaussians
    float3 accumulatedColor = float3(0, 0, 0);
    float logTransmittance = 0.0;  // Log-space accumulation for numerical stability

    // PCSS temporal filtering: Accumulate shadows across all volume march steps
    float currentShadowAccum = 0.0;
    float shadowSampleCount = 0.0;

    for (uint i = 0; i < hitCount; i++) {
        // Early exit if fully opaque (convert log-space to linear for check)
        float transmittance = exp(logTransmittance);
        if (transmittance < 0.001) break;

        HitRecord hit = hits[i];
        Particle p = g_particles[hit.particleIdx];

        // =========================================================================
        // PHASE 3: MATERIAL SYSTEM - Get per-particle material properties
        // =========================================================================
        MaterialTypeProperties mat = g_materials[p.materialType];

        // Material-specific volumetric lighting parameters
        float scatteringG = mat.phaseG;                  // Henyey-Greenstein phase function (-1 to 1)
        float extinction = mat.opacity;                   // Opacity/extinction coefficient
        float scatteringCoeff = mat.scatteringCoefficient; // Volumetric scattering strength

        // Gaussian parameters (with anisotropic control + adaptive radius)
        float3 scale = ComputeGaussianScale(
            p, baseParticleRadius,
            useAnisotropicGaussians != 0,
            anisotropyStrength,
            enableAdaptiveRadius != 0,
            adaptiveInnerZone,
            adaptiveOuterZone,
            adaptiveInnerScale,
            adaptiveOuterScale,
            densityScaleMin,
            densityScaleMax
        );
        float3x3 rotation = ComputeGaussianRotation(p.velocity);

        // Ray-march through this Gaussian with fixed step count for stability
        float tStart = max(hit.tNear, ray.TMin);
        float tEnd = min(hit.tFar, ray.TMax);

        // Fixed step count prevents flickering (was variable based on distance)
        const uint steps = 16; // Consistent sampling regardless of particle size
        float stepSize = (tEnd - tStart) / float(steps);

        for (uint step = 0; step < steps; step++) {
            // CRITICAL FIX: Sub-pixel jitter REMOVED - it caused temporal instability (shuddering)
            // When combined with numerical instability at large radii, jitter amplified the cube artifact
            // Fixed sampling at step centers provides stable, consistent ray marching
            float t = tStart + (step + 0.5) * stepSize;  // Sample at step center for stability
            float3 pos = ray.Origin + ray.Direction * t;

            // Evaluate Gaussian density at this point
            float density = EvaluateGaussianDensity(pos, p.position, scale, rotation, p.density);

            // CRITICAL FIX: Removed double exponential falloff!
            // EvaluateGaussianDensity() already applies exp(-0.5 * dist²) in gaussian_common.hlsl:220
            // The extra sphericalFalloff exp(-distFromCenter²) caused double exponential = over-darkening
            // Now density is correctly modulated only by the density multiplier
            density *= densityMultiplier;

            // CRITICAL FIX: Smooth density threshold prevents hard edges
            // Hard cutoff (density < 0.01) caused visible "banding" at particle edges
            // Smooth falloff via smoothstep creates gradual transparency
            float densityThreshold = 0.01;
            float smoothWidth = 0.02;  // Transition width
            float densityWeight = smoothstep(densityThreshold - smoothWidth,
                                              densityThreshold + smoothWidth,
                                              density);

            // Early skip for truly empty regions (still needed for performance)
            if (densityWeight < 0.001) continue;

            // Modulate density by smooth weight
            density *= densityWeight;

            // === HYBRID EMISSION MODEL: Blend artistic + physical + material albedo ===
            float3 emission;
            float intensity;

            if (usePhysicalEmission != 0) {
                // Calculate BOTH artistic and physical colors
                float3 artisticEmission = TemperatureToEmission(p.temperature);
                float3 physicalEmission = ComputePlasmaEmission(
                    p.position,
                    p.velocity,
                    p.temperature,
                    p.density,
                    cameraPos
                );

                // Apply emission strength to physical emission
                physicalEmission = lerp(float3(0.5, 0.5, 0.5), physicalEmission, emissionStrength);

                // Phase 3: Blend with material albedo for color diversity
                physicalEmission = lerp(physicalEmission, mat.albedo, 0.3);  // 30% material color blend

                // Optional Doppler shift (only on physical component)
                if (useDopplerShift != 0) {
                    float3 viewDir = normalize(cameraPos - p.position);
                    physicalEmission = DopplerShift(physicalEmission, p.velocity, viewDir, dopplerStrength);
                }

                // Optional gravitational redshift (only on physical component)
                if (useGravitationalRedshift != 0) {
                    float radius = length(p.position);
                    const float schwarzschildRadius = 2.0;
                    physicalEmission = GravitationalRedshift(physicalEmission, radius, schwarzschildRadius, redshiftStrength);
                }

                // Temperature-based auto-blend: Cool particles stay artistic, hot particles go physical
                // This prevents the whole disk from going blue when physical emission is enabled
                float tempBlend = saturate((p.temperature - 8000.0) / 10000.0);  // 0 below 8000K, 1 above 18000K

                // Combine manual blend factor with temperature-based blend
                float finalBlend = emissionBlendFactor * tempBlend;

                // Blend: 0.0 = pure artistic (warm colors), 1.0 = pure physical (accurate blues)
                emission = lerp(artisticEmission, physicalEmission, finalBlend);

                // Phase 3: Apply material-specific emission multiplier
                intensity = EmissionIntensity(p.temperature) * mat.emissionMultiplier;
            } else {
                // Standard temperature-based color (artistic) blended with material albedo
                float3 temperatureColor = TemperatureToEmission(p.temperature);
                emission = lerp(temperatureColor, mat.albedo, 0.5);  // 50% material albedo blend

                // Phase 3: Apply material-specific emission multiplier
                intensity = EmissionIntensity(p.temperature) * mat.emissionMultiplier;
            }

            // === RT LIGHTING: Probe Grid + Direct RT (ADDITIVE COMBINATION) ===
            // Phase 2 Fix: Enable BOTH probe grid AND inline RayQuery simultaneously
            // Probe grid provides volumetric ambient scattering in dense regions
            // Direct RT provides particle-to-particle illumination everywhere

            float3 probeGridLight = float3(0, 0, 0);
            float3 directRTLight = float3(0, 0, 0);

            // Sample probe grid if enabled (volumetric ambient scattering)
            if (useProbeGrid != 0) {
                // PROBE GRID MODE (Phase 0.13.1): Zero atomic contention!
                // Pre-computed lighting at sparse 48³ grid with trilinear interpolation
                // Scales to 10K+ particles without GPU hang
                // NOW WITH FULL SH L2 RECONSTRUCTION for directional scattering!
                probeGridLight = SampleProbeGrid(pos, ray.Direction);
            }

            // Sample direct RT lighting (always active when RT lighting enabled)
            if (useVolumetricRT != 0) {
                // VOLUMETRIC SCATTERING MODE: Treat neighbors as virtual lights
                // Applies same volumetric math as multi-lights (attenuation + phase function + PCSS)
                // This creates true volumetric glow with proper light scattering!
                directRTLight = InterpolateRTLighting(pos, hit.particleIdx, ray.Direction, scatteringG, pixelPos);
            } else {
                // LEGACY MODE: Per-particle lookup (billboard-era)
                // Fast but causes discrete brightness jumps
                // Phase 0.15.2: Add shadow support to legacy RT lighting
                directRTLight = g_rtLighting[hit.particleIdx].rgb;

                // Apply volumetric shadows if enabled (Phase 0.15.2)
                if (useShadowRays != 0 && length(directRTLight) > 0.001) {
                    // Get the RT-lit particle position (treat it as a virtual light source)
                    Particle rtParticle = g_particles[hit.particleIdx];
                    float3 particlePos = rtParticle.position;

                    // Cast shadow ray from current position to RT-lit particle
                    float shadowTerm = CastPCSSShadowRay(
                        pos,                  // Current sample point
                        particlePos,          // RT-lit particle position (virtual light)
                        particleRadius,       // Light radius for soft shadows
                        pixelPos,             // Pixel coordinate for temporal filtering
                        shadowRaysPerLight    // Shadow quality (1/4/8 rays)
                    );

                    // Modulate RT lighting by shadow term
                    directRTLight *= shadowTerm;
                }
            }

            // Combine both sources (additive for maximum flexibility)
            // rtLightingStrength will multiply the combined result
            float3 rtLight = probeGridLight + directRTLight;

            // === CRITICAL FIX: Physical emission is self-emitting, not lit ===
            // Physical emission (blackbody radiation) should NOT be modulated by external lighting
            // Non-physical emission (temperature color) CAN be modulated by external lighting

            float3 totalEmission;

            if (usePhysicalEmission != 0) {
                // Physical emission: Self-emitting blackbody radiation (INDEPENDENT of external light)
                totalEmission = emission * intensity;

                // Optional: Add in-scattering as separate contribution
                if (useInScattering != 0) {
                    float3 inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
                    totalEmission += inScatter * inScatterStrength;
                }
            } else {
                // Non-physical emission: Temperature-based color that CAN be lit by external sources

                // Base ambient level (allows particles to be visible even with no self-emission)
                // Phase 1 lighting fix: Use rtMinAmbient parameter (default 0.05, adjustable 0.0-0.2)
                float3 ambientBase = float3(rtMinAmbient, rtMinAmbient, rtMinAmbient);

                // RT lighting: Separate from multi-light to avoid scaling conflicts
                // Volumetric RT has large boost (50-500×) so clamp must be high
                // Legacy RT (per-particle) is pre-computed so clamp is lower
                float rtClampMax = useVolumetricRT != 0 ? 100.0 : 10.0;
                float3 rtContribution = clamp(rtLight * rtLightingStrength, 0.0, rtClampMax);

                // === MULTI-LIGHT SYSTEM: Accumulate lighting from all active lights ===
                float3 totalLighting = float3(0, 0, 0);

                if (useRTXDI != 0) {
                    // === RTXDI MODE: Use single RTXDI-selected light ===
                    // Read selected light index from RTXDI output buffer
                    float4 rtxdiData = g_rtxdiOutput[pixelPos];
                    uint selectedLightIndex = asuint(rtxdiData.r);

                    // DEBUG: Visualize selected light index
                    if (debugRTXDISelection != 0) {
                        if (selectedLightIndex == 0xFFFFFFFF) {
                            // No light: Black
                            totalLighting = float3(0, 0, 0);
                        } else if (selectedLightIndex < lightCount) {
                            // BOUNDS CHECK: Valid light index
                            // Color-code by light index (0-12 = rainbow colors)
                            float hue = float(selectedLightIndex) / max(float(lightCount), 1.0);  // 0.0-1.0
                            // Simple hue to RGB (red → green → blue)
                            totalLighting = float3(
                                saturate(abs(hue * 6.0 - 3.0) - 1.0),
                                saturate(2.0 - abs(hue * 6.0 - 2.0)),
                                saturate(2.0 - abs(hue * 6.0 - 4.0))
                            ) * 5.0;  // Boost for visibility
                        } else {
                            // OUT OF BOUNDS: Magenta warning (should never happen!)
                            totalLighting = float3(1, 0, 1) * 5.0;
                        }
                    } else {
                        // NORMAL MODE: RTXDI with soft spatial blending (Phase 4 M5+ fix)
                        // Instead of using ONLY the RTXDI-selected light (causes patchwork),
                        // blend the selected light (70%) with nearest lights (30%) for softer boundaries

                        const float RTXDI_PRIMARY_WEIGHT = 0.7;  // Weight for RTXDI-selected light
                        const float RTXDI_BLEND_WEIGHT = 0.3;    // Weight for distance-blended lights

                        float3 primaryLighting = float3(0, 0, 0);
                        float3 blendedLighting = float3(0, 0, 0);
                        float blendWeightSum = 0.0;

                        // === PRIMARY: RTXDI-selected light ===
                        if (selectedLightIndex != 0xFFFFFFFF && selectedLightIndex < lightCount) {
                            Light light = g_lights[selectedLightIndex];

                            float3 lightDir = normalize(light.position - pos);
                            float lightDist = length(light.position - pos);
                            float normalizedDist = lightDist / max(light.radius, 1.0);
                            float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

                            float shadowTerm = 1.0;
                            if (useShadowRays != 0) {
                                shadowTerm = CastPCSSShadowRay(pos, light.position, light.radius, pixelPos, shadowRaysPerLight);
                            }

                            float phase = 1.0;
                            if (usePhaseFunction != 0) {
                                float cosTheta = dot(-ray.Direction, lightDir);
                                phase = HenyeyGreenstein(cosTheta, scatteringG);
                            }

                            if (enableTemporalFiltering != 0) {
                                currentShadowAccum += shadowTerm;
                                shadowSampleCount += 1.0;
                            }

                            primaryLighting = light.color * light.intensity * attenuation * shadowTerm * phase;
                        }

                        // === SECONDARY: Blend nearest lights for soft spatial coherence ===
                        // Find top 3 nearest lights (excluding RTXDI selection) and blend by distance
                        for (uint blendIdx = 0; blendIdx < lightCount && blendIdx < 16; blendIdx++) {
                            if (blendIdx == selectedLightIndex) continue;  // Skip RTXDI selection

                            Light light = g_lights[blendIdx];
                            float3 lightDir = normalize(light.position - pos);
                            float lightDist = length(light.position - pos);

                            // Distance-based weight: closer lights contribute more
                            float distWeight = 1.0 / (1.0 + lightDist * lightDist * 0.0001);

                            // Only include lights with significant contribution (culling far lights)
                            if (distWeight < 0.01) continue;

                            float normalizedDist = lightDist / max(light.radius, 1.0);
                            float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

                            // Skip shadow rays for blend lights (performance)
                            // Skip phase function for blend lights (prevents compound darkening)
                            // Blended lights act as soft fill light for boundary smoothing
                            float3 lightContrib = light.color * light.intensity * attenuation;
                            blendedLighting += lightContrib * distWeight;
                            blendWeightSum += distWeight;
                        }

                        // Normalize blended lighting
                        if (blendWeightSum > 0.0) {
                            blendedLighting /= blendWeightSum;
                        }

                        // Combine primary (RTXDI) and blended (spatial) lighting
                        totalLighting = primaryLighting * RTXDI_PRIMARY_WEIGHT + blendedLighting * RTXDI_BLEND_WEIGHT;
                    }

                } else {
                    // === MULTI-LIGHT MODE: Loop all lights (original 13-light brute force) ===
                    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
                        Light light = g_lights[lightIdx];

                        // Direction and distance to this light
                        float3 lightDir = normalize(light.position - pos);
                        float lightDist = length(light.position - pos);

                        // Use light.radius for soft falloff (makes radius slider functional)
                        float normalizedDist = lightDist / max(light.radius, 1.0);  // Normalize by radius
                        float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);  // Quadratic for soft edge

                        // Cast shadow ray to this light (if enabled)
                        float shadowTerm = 1.0;

                        // Phase 2: Screen-space shadows (replaces PCSS when enabled)
                        if (useScreenSpaceShadows != 0) {
                            // Screen-space contact shadows - ray march through depth buffer
                            shadowTerm = ScreenSpaceShadow(pos, lightDir, lightDist, ssSteps);

                            // DEBUG VISUALIZATION: Override lighting with shadow debug colors
                            if (debugScreenSpaceShadows != 0 && lightIdx == 0) {
                                // Enhanced debug visualization with dramatic color gradient:
                                // shadowTerm: 1.0 = fully lit (bright green)
                                //           : 0.5-0.8 = partial shadow (yellow/orange gradient)
                                //           : 0.0 = fully shadowed (bright red)

                                float3 debugColor;
                                if (shadowTerm < 0.5) {
                                    // Heavy shadow: Red to orange (0.0-0.5)
                                    float t = shadowTerm / 0.5;
                                    debugColor = float3(1.0, t * 0.5, 0.0);  // Red → orange
                                } else if (shadowTerm < 0.8) {
                                    // Partial shadow: Orange to yellow (0.5-0.8)
                                    float t = (shadowTerm - 0.5) / 0.3;
                                    debugColor = float3(1.0, 0.5 + t * 0.5, 0.0);  // Orange → yellow
                                } else {
                                    // Mostly lit: Yellow to green (0.8-1.0)
                                    float t = (shadowTerm - 0.8) / 0.2;
                                    debugColor = float3(1.0 - t, 1.0, 0.0);  // Yellow → green
                                }

                                totalLighting = debugColor * 500.0;  // 10× boost for visibility
                                break;  // Only show first light
                            }
                        }
                        else if (useShadowRays != 0) {
                            // Legacy PCSS shadow rays (being replaced)
                            shadowTerm = CastPCSSShadowRay(pos, light.position, light.radius, pixelPos, shadowRaysPerLight);
                        }

                        // Apply phase function for view-dependent scattering (if enabled)
                        float phase = 1.0;
                        if (usePhaseFunction != 0) {
                            float cosTheta = dot(-ray.Direction, lightDir);
                            phase = HenyeyGreenstein(cosTheta, scatteringG);
                        }

                        // PCSS temporal filtering: Accumulate shadow values for temporal filter
                        if (enableTemporalFiltering != 0) {
                            currentShadowAccum += shadowTerm;
                            shadowSampleCount += 1.0;
                        }

                        // Accumulate this light's contribution
                        float3 lightContribution = light.color * light.intensity * attenuation * shadowTerm * phase;
                        totalLighting += lightContribution;
                    }
                }

                // === MULTI-LIGHT ILLUMINATION ===
                // Accumulate multi-light contribution with proper scaling
                float3 multiLightContribution = float3(0, 0, 0);

                if (useRTXDI != 0) {
                    // RTXDI: NO multiplier (1 importance-sampled light)
                    // Expected: Dimmer than multi-light (this is CORRECT)
                    multiLightContribution = totalLighting * 0.02;  // Apply 2% scaling for external light
                } else {
                    // Multi-light: 10× multiplier to match RT lighting strength
                    // Then apply 2% scaling for reasonable brightness
                    multiLightContribution = totalLighting * 10.0 * 0.02;  // = 0.2× total
                }

                // === RT LIGHTING CONTRIBUTION ===
                // Apply separate scaling for RT lighting (already boosted 50-500× for volumetric)
                // This needs MUCH less aggressive scaling than multi-light
                float3 rtExternalLight = rtContribution * 0.5;  // 50% scaling (vs 2% for multi-light)

                // Add in-scattering for volumetric depth (TOGGLEABLE)
                float3 inScatter = float3(0, 0, 0);
                if (useInScattering != 0) {
                    inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
                }

                // FIXED: Separate self-emission glow from external lighting
                // Self-emission: Particle's blackbody glow (Phase 3: uses material emission multiplier)
                float3 selfEmission = emission * intensity * emissionStrength;

                // Phase 3: Use material albedo for scattering color (not emission color)
                // This gives particles realistic scattering properties based on material type
                float3 particleAlbedo = lerp(float3(1, 1, 1), mat.albedo, scatteringCoeff * 0.3);  // Material-driven scattering

                // Combine all lighting sources
                float3 externalLight = (ambientBase + rtExternalLight + multiLightContribution) * particleAlbedo;

                // Final combination: glow + external lighting + in-scattering (all additive)
                totalEmission = selfEmission + externalLight + inScatter * inScatterStrength;
            }

            // NOTE: God ray contribution moved to separate atmospheric fog pass
            // See RayMarchAtmosphericFog() call after particle rendering

            // Volume rendering equation with proper absorption/emission
            float absorption = density * stepSize * extinction;

            // Log-space transmittance accumulation (eliminates precision loss)
            logTransmittance -= absorption;
            float transmittance = exp(logTransmittance);

            // Emission contribution with stable transmittance
            float3 emission_contribution = totalEmission * (1.0 - exp(-absorption));
            accumulatedColor += transmittance * emission_contribution;

            // Early exit
            if (transmittance < 0.001) break;
        }
    }

    // PCSS temporal filtering: Blend current and previous shadow values
    if (enableTemporalFiltering != 0 && shadowSampleCount > 0.0) {
        // Calculate average shadow value for this pixel
        float currentShadow = currentShadowAccum / shadowSampleCount;

        // Read previous frame's shadow value
        float prevShadow = g_prevShadow[pixelPos];

        // Temporal blend: low blend value = more history (smoother but slower convergence)
        float finalShadow = lerp(prevShadow, currentShadow, temporalBlend);

        // Write to current shadow buffer for next frame
        g_currShadow[pixelPos] = finalShadow;
    } else {
        // No temporal filtering - write current shadow directly
        if (shadowSampleCount > 0.0) {
            g_currShadow[pixelPos] = currentShadowAccum / shadowSampleCount;
        } else {
            g_currShadow[pixelPos] = 1.0; // Fully lit (no shadow data)
        }
    }

    // Background color (pure black space - or ground plane if hit)
    float3 backgroundColor = groundPlaneHit ? groundPlaneColor : float3(0.0, 0.0, 0.0);
    float finalTransmittance = exp(logTransmittance);
    float3 finalColor = accumulatedColor + finalTransmittance * backgroundColor;

    // =========================================================================
    // FROXEL VOLUMETRIC FOG SAMPLING (Phase 5 - DEPRECATED)
    // =========================================================================
    // REMOVED: Froxel system replaced by Gaussian Volume

    // DEPRECATED: Old god ray system (replaced by froxel system for 5× performance)
    /*
    if (godRayDensity > 0.001) {
        float3 atmosphericFog = RayMarchAtmosphericFog(...);
        finalColor += atmosphericFog * 0.1;
    }
    */

    // Enhanced tone mapping for HDR
    // Use ACES tone mapping for better color preservation
    float3 aces_input = finalColor;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    finalColor = saturate((aces_input * (a * aces_input + b)) /
                          (aces_input * (c * aces_input + d) + e));

    // Gamma correction
    finalColor = pow(finalColor, 1.0 / 2.2);

    // DEBUG: Visual indicators for active features (AFTER tone mapping so they're visible!)
    // Top-left corner: Shadow rays (red bar if ON)
    if (useShadowRays != 0 && pixelPos.x < 100 && pixelPos.y < 20) {
        finalColor = float3(1, 0, 0); // Solid red bar
    }
    // Top-right corner: In-scattering (green bar if ON)
    if (useInScattering != 0 && pixelPos.x > resolution.x - 100 && pixelPos.y < 20) {
        finalColor = float3(0, 1, 0); // Solid green bar
    }
    // Bottom-left corner: Phase function (blue bar if ON)
    if (usePhaseFunction != 0 && pixelPos.x < 100 && pixelPos.y > resolution.y - 20) {
        finalColor = float3(0, 0, 1); // Solid blue bar
    }

    // === RT DEPTH OUTPUT (Phase 4 M5 Fix) ===
    // Write first hit distance for RTXDI temporal reprojection
    // This replaces the planar Z=0 assumption with actual depth
    float outputDepth = 10000.0;  // Default: far plane (no hit)
    if (hitCount > 0) {
        outputDepth = hits[0].tNear;  // First (closest) hit distance
    }
    g_rtDepth[pixelPos] = outputDepth;

    // === PRIORITY 1 FIX: TEMPORAL COLOR ACCUMULATION ===
    // Eliminates flashing caused by per-frame random sampling in RayQuery lighting
    // Uses exponential moving average (EMA) for smooth temporal convergence
    if (enableTemporalFiltering != 0) {
        // Read previous frame's color
        float3 prevColor = g_prevColor[pixelPos].rgb;

        // Blend current frame with previous frame (10% new, 90% history)
        // This gives ~67ms convergence time @ 120 FPS (same as shadow temporal)
        float3 blendedColor = lerp(prevColor, finalColor, temporalBlend);

        // Write blended result to output
        g_output[pixelPos] = float4(blendedColor, 1.0);

        // Write current frame to temporal buffer (for next frame's blend)
        g_currColor[pixelPos] = float4(finalColor, 1.0);
    } else {
        // No temporal filtering - write current frame directly
        g_output[pixelPos] = float4(finalColor, 1.0);
        g_currColor[pixelPos] = float4(finalColor, 1.0);
    }
}