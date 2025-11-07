// 3D Gaussian Splatting Ray Tracing
// Replaces billboard rasterization with volumetric ray-traced Gaussians
// Proper depth sorting, transparency, and volumetric appearance

#include "gaussian_common.hlsl"
#include "plasma_emission.hlsl"
// NOTE: god_rays.hlsl removed - atmospheric fog function defined inline below for Light struct access

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

    // God ray system (Phase 5 Milestone 5.3c)
    float godRayDensity;           // Global god ray density (0.0-1.0)
    float godRayStepMultiplier;    // Ray march step multiplier (0.5-2.0)
    float2 godRayPadding;          // Padding for alignment

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

// Derived values
static const float2 resolution = float2(screenWidth, screenHeight);
static const float2 invResolution = 1.0 / resolution;
static const float baseParticleRadius = particleRadius;
static const float volumeStepSize = 0.1;         // Finer steps for better quality
static const float densityMultiplier = 2.0;      // Increased for more volumetric appearance
static const float shadowBias = 0.01;            // Bias for shadow ray origin

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

// PCSS soft shadow ray (multi-sample with Poisson disk)
float CastPCSSShadowRay(float3 origin, float3 lightPos, float lightRadius, uint2 pixelPos, uint numSamples) {
    float3 toLight = lightPos - origin;
    float lightDist = length(toLight);
    float3 lightDir = toLight / lightDist;

    // Single ray for performance mode
    if (numSamples == 1) {
        return CastSingleShadowRay(origin, lightDir, lightDist);
    }

    // Multi-ray PCSS for balanced/quality modes
    float shadowAccum = 0.0;
    float randomAngle = Hash12(float2(pixelPos)) * 6.28318; // Random rotation per pixel

    // Build tangent space for light disk sampling
    float3 tangent = abs(lightDir.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    tangent = normalize(cross(tangent, lightDir));
    float3 bitangent = cross(lightDir, tangent);

    for (uint i = 0; i < numSamples; i++) {
        // Get Poisson disk sample (rotate for temporal stability)
        float2 diskSample = Rotate2D(PoissonDisk16[i % 16], randomAngle);

        // Scale by light radius and map to 3D disk perpendicular to light direction
        float3 offset = (diskSample.x * tangent + diskSample.y * bitangent) * lightRadius;
        float3 sampleLightPos = lightPos + offset;

        // Cast shadow ray to this sample position
        float3 sampleDir = normalize(sampleLightPos - origin);
        float sampleDist = length(sampleLightPos - origin);
        shadowAccum += CastSingleShadowRay(origin, sampleDir, sampleDist);
    }

    return shadowAccum / float(numSamples);
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

// =============================================================================
// ReSTIR HELPER FUNCTIONS
// =============================================================================

// Simple hash function for pseudo-random numbers
float Hash(uint seed) {
    seed = seed * 747796405u + 2891336453u;
    seed = ((seed >> 16) ^ seed) * 747796405u;
    seed = ((seed >> 16) ^ seed);
    return float(seed) / 4294967295.0;
}

// =============================================================================
// PROBE GRID SAMPLING (Phase 0.13.1)
// =============================================================================
/**
 * Sample probe grid irradiance using trilinear interpolation
 *
 * Replaces Volumetric ReSTIR (which suffered from atomic contention at ≥2045 particles)
 * Zero atomic operations = zero contention = scales to 10K+ particles!
 *
 * Algorithm:
 * 1. Convert world position to grid coordinates
 * 2. Find 8 nearest probes (corner of grid cell)
 * 3. Trilinear interpolation between probes
 *
 * @param worldPos World-space position to sample lighting at
 * @return Irradiance (RGB color) at the given position
 */
float3 SampleProbeGrid(float3 worldPos) {
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

    // Sample irradiance from probes (SH L0 coefficient only for MVP)
    // Full SH L2 reconstruction can be added later for better directionality
    float3 c000 = g_probeGrid[idx000].irradiance[0];
    float3 c001 = g_probeGrid[idx001].irradiance[0];
    float3 c010 = g_probeGrid[idx010].irradiance[0];
    float3 c011 = g_probeGrid[idx011].irradiance[0];
    float3 c100 = g_probeGrid[idx100].irradiance[0];
    float3 c101 = g_probeGrid[idx101].irradiance[0];
    float3 c110 = g_probeGrid[idx110].irradiance[0];
    float3 c111 = g_probeGrid[idx111].irradiance[0];

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
    // ReSTIR: Temporal Resampling for Better Light Sampling
    // =============================================================================
    uint pixelIndex = pixelPos.y * screenWidth + pixelPos.x;

    // Initialize reservoir (always, even if ReSTIR is off)
    // Volume rendering: march through sorted Gaussians
    float3 accumulatedColor = float3(0, 0, 0);
    float logTransmittance = 0.0;  // Log-space accumulation for numerical stability

    // Volumetric lighting parameters (used in multi-light loop)
    const float scatteringG = 0.7;      // Henyey-Greenstein g parameter for phase function
    const float extinction = 1.0;       // Extinction coefficient for volume rendering

    // PCSS temporal filtering: Accumulate shadows across all volume march steps
    float currentShadowAccum = 0.0;
    float shadowSampleCount = 0.0;

    for (uint i = 0; i < hitCount; i++) {
        // Early exit if fully opaque (convert log-space to linear for check)
        float transmittance = exp(logTransmittance);
        if (transmittance < 0.001) break;

        HitRecord hit = hits[i];
        Particle p = g_particles[hit.particleIdx];

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
            // Add sub-pixel jitter to reduce temporal aliasing
            // Uses pixel position as seed for consistent per-pixel randomness
            float jitter = frac(sin(dot(float2(pixelPos), float2(12.9898, 78.233))) * 43758.5453);
            float t = tStart + (step + jitter) * stepSize;
            float3 pos = ray.Origin + ray.Direction * t;

            // Evaluate Gaussian density at this point
            float density = EvaluateGaussianDensity(pos, p.position, scale, rotation, p.density);

            // Enhanced spherical falloff for better 3D appearance
            float3 toCenter = pos - p.position;
            float distFromCenter = length(toCenter) / length(scale);
            float sphericalFalloff = exp(-distFromCenter * distFromCenter * 2.5); // Slightly softer (was 3.0)
            density *= sphericalFalloff * densityMultiplier;

            // Higher threshold to skip low-density regions that cause flickering
            if (density < 0.01) continue;

            // === HYBRID EMISSION MODEL: Blend artistic + physical ===
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
                intensity = EmissionIntensity(p.temperature);
            } else {
                // Standard temperature-based color (artistic)
                emission = TemperatureToEmission(p.temperature);
                // Intensity based on temperature (emissionStrength applied separately at final composition)
                intensity = EmissionIntensity(p.temperature);
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
                probeGridLight = SampleProbeGrid(pos);
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
                // Kept for comparison and fallback
                directRTLight = g_rtLighting[hit.particleIdx].rgb;
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
                        // NORMAL MODE: Validate light index (0xFFFFFFFF = no light in cell)
                        if (selectedLightIndex != 0xFFFFFFFF && selectedLightIndex < lightCount) {
                            // Use ONLY the RTXDI-selected light
                            Light light = g_lights[selectedLightIndex];

                            // Direction and distance to this light
                            float3 lightDir = normalize(light.position - pos);
                            float lightDist = length(light.position - pos);

                            // Use light.radius for soft falloff
                            float normalizedDist = lightDist / max(light.radius, 1.0);
                            float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

                            // Cast shadow ray to this light (if enabled)
                            float shadowTerm = 1.0;
                            if (useShadowRays != 0) {
                                shadowTerm = CastPCSSShadowRay(pos, light.position, light.radius, pixelPos, shadowRaysPerLight);
                            }

                            // Apply phase function for view-dependent scattering (if enabled)
                            float phase = 1.0;
                            if (usePhaseFunction != 0) {
                                float cosTheta = dot(-ray.Direction, lightDir);
                                phase = HenyeyGreenstein(cosTheta, scatteringG);
                            }

                            // PCSS temporal filtering: Accumulate shadow values
                            if (enableTemporalFiltering != 0) {
                                currentShadowAccum += shadowTerm;
                                shadowSampleCount += 1.0;
                            }

                            // RTXDI-selected light contribution
                            totalLighting = light.color * light.intensity * attenuation * shadowTerm * phase;
                        }
                        // else: No light selected for this pixel - use ambient only (totalLighting = 0)
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
                        if (useShadowRays != 0) {
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
                // Self-emission: Particle's blackbody glow (controlled by emissionStrength)
                float3 selfEmission = emission * intensity * emissionStrength;

                // Apply subtle particle color to external lighting
                // This prevents complete color wash-out while avoiding blown-out emission colors
                float3 particleAlbedo = lerp(float3(1, 1, 1), emission, 0.15);  // Only 15% emission tint

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

    // Background color (pure black space - no blue tint)
    float3 backgroundColor = float3(0.0, 0.0, 0.0);
    float finalTransmittance = exp(logTransmittance);
    float3 finalColor = accumulatedColor + finalTransmittance * backgroundColor;

    // =========================================================================
    // ATMOSPHERIC FOG RAY MARCHING - Volumetric God Rays
    // =========================================================================
    // March through uniform atmospheric fog independent of particle positions
    // This creates visible light shafts even in empty space!
    if (godRayDensity > 0.001) {
        float3 atmosphericFog = RayMarchAtmosphericFog(
            cameraPos,           // Camera position
            ray.Direction,       // Ray direction (from GenerateCameraRay)
            3000.0,              // Max ray march distance (covers entire scene)
            g_lights,            // Light array
            lightCount,          // Number of active lights
            time,                // Total elapsed time (for rotation)
            godRayDensity,       // Global fog density
            g_particleBVH        // TLAS for shadow rays
        );

        // Add atmospheric fog to final color (before tone mapping)
        // Scale by 0.1 for balanced contribution relative to particles
        finalColor += atmosphericFog * 0.1;
    }

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
    // Bottom-right corner: ReSTIR status (complex debug info)
    if (pixelPos.x > resolution.x - 200 && pixelPos.y > resolution.y - 40) {
        // Show different colors based on ReSTIR state
        // Show gray (legacy ReSTIR removed)
        finalColor = float3(0.3, 0.3, 0.3);
    }

    g_output[pixelPos] = float4(finalColor, 1.0);
}