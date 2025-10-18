// 3D Gaussian Splatting Ray Tracing
// Replaces billboard rasterization with volumetric ray-traced Gaussians
// Proper depth sorting, transparency, and volumetric appearance

#include "gaussian_common.hlsl"
#include "plasma_emission.hlsl"

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
    float padding4;                // Alignment
};

// Light structure for multi-light system
struct Light {
    float3 position;               // 12 bytes
    float intensity;               // 4 bytes
    float3 color;                  // 12 bytes
    float radius;                  // 4 bytes (for soft shadows)
};

// Light array (after constant buffer to avoid size issues)
StructuredBuffer<Light> g_lights : register(t4);

// PCSS shadow buffers (temporal filtering for soft shadows)
Texture2D<float> g_prevShadow : register(t5);  // Previous frame shadow (read-only)

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

            // Compute Gaussian parameters (with anisotropic control)
            float3 scale = ComputeGaussianScale(p, baseParticleRadius,
                                                useAnisotropicGaussians != 0,
                                                anisotropyStrength);
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

        // Gaussian parameters (with anisotropic control)
        float3 scale = ComputeGaussianScale(p, baseParticleRadius,
                                            useAnisotropicGaussians != 0,
                                            anisotropyStrength);
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
                intensity = EmissionIntensity(p.temperature);
            }

            // === FIXED: RT lighting as illumination, not replacement ===
            float3 rtLight;

            // Use pre-computed RT lighting
            rtLight = g_rtLighting[hit.particleIdx].rgb;

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
                float3 illumination = float3(1, 1, 1); // Base self-illumination

                // Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
                // Clamp to prevent over-brightness from extreme ReSTIR samples
                illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);

                // === MULTI-LIGHT SYSTEM: Accumulate lighting from all active lights ===
                float3 totalLighting = float3(0, 0, 0);

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

                // Apply multi-light illumination to external lighting
                // FIX: Removed weak lerp(0.1, 1.0, ...) that capped contribution too low
                // Multi-light should be comparable in strength to RT lighting (which is clamped to 10.0)
                illumination += totalLighting * 10.0;

                // Add in-scattering for volumetric depth (TOGGLEABLE)
                float3 inScatter = float3(0, 0, 0);
                if (useInScattering != 0) {
                    inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
                }

                // Combine emission with external illumination and in-scattering
                totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;
            }

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