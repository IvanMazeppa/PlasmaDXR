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

    uint useShadowRays;
    uint useInScattering;
    uint usePhaseFunction;
    float phaseStrength;
};

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

// Cast shadow ray to check occlusion (returns transmittance 0-1)
float CastShadowRay(float3 origin, float3 direction, float maxDist) {
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

// Compute in-scattering from nearby particles
float3 ComputeInScattering(float3 pos, float3 viewDir, uint skipIdx) {
    float3 totalScattering = float3(0, 0, 0);

    // Sample a few random directions for scattering
    const uint numSamples = 4;

    for (uint i = 0; i < numSamples; i++) {
        // Simple stratified sampling on hemisphere
        float phi = (i + 0.5) * 6.28318 / numSamples;
        float3 scatterDir = float3(cos(phi), 0.5, sin(phi));
        scatterDir = normalize(scatterDir);

        // Trace a short ray to gather light from nearby particles
        RayDesc scatterRay;
        scatterRay.Origin = pos;
        scatterRay.Direction = scatterDir;
        scatterRay.TMin = 0.01;
        scatterRay.TMax = 50.0; // Short range for local scattering

        RayQuery<RAY_FLAG_NONE> scatterQuery;
        scatterQuery.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, scatterRay);

        while (scatterQuery.Proceed()) {
            if (scatterQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint idx = scatterQuery.CandidatePrimitiveIndex();
                if (idx != skipIdx) {
                    Particle p = g_particles[idx];

                    // Simple distance-based attenuation
                    float dist = length(p.position - pos);
                    float atten = 1.0 / (1.0 + dist * 0.1);

                    // Add scattered light from this particle
                    float3 emission = TemperatureToEmission(p.temperature);
                    float intensity = EmissionIntensity(p.temperature);

                    // Phase function for scattering direction
                    float phase = HenyeyGreenstein(dot(viewDir, scatterDir), 0.3);

                    totalScattering += emission * intensity * phase * atten;
                }
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

            // Compute Gaussian parameters
            float3 scale = ComputeGaussianScale(p, baseParticleRadius);
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

    // Volume rendering: march through sorted Gaussians
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;

    // Setup volumetric lighting
    VolumetricParams volParams;
    volParams.lightPos = float3(0, 10, 0);  // Light above origin (black hole)
    volParams.lightColor = float3(2, 2, 2); // Bright white light
    volParams.scatteringG = 0.3;            // Forward scattering
    volParams.extinction = 0.5;             // Medium extinction

    for (uint i = 0; i < hitCount; i++) {
        // Early exit if fully opaque
        if (transmittance < 0.001) break;

        HitRecord hit = hits[i];
        Particle p = g_particles[hit.particleIdx];

        // Gaussian parameters
        float3 scale = ComputeGaussianScale(p, baseParticleRadius);
        float3x3 rotation = ComputeGaussianRotation(p.velocity);

        // Ray-march through this Gaussian
        float tStart = max(hit.tNear, ray.TMin);
        float tEnd = min(hit.tFar, ray.TMax);

        uint steps = max(1, (uint)((tEnd - tStart) / volumeStepSize));
        float stepSize = (tEnd - tStart) / steps;

        for (uint step = 0; step < steps; step++) {
            float t = tStart + (step + 0.5) * stepSize;
            float3 pos = ray.Origin + ray.Direction * t;

            // Evaluate Gaussian density at this point
            float density = EvaluateGaussianDensity(pos, p.position, scale, rotation, p.density);

            // Enhanced spherical falloff for better 3D appearance
            float3 toCenter = pos - p.position;
            float distFromCenter = length(toCenter) / length(scale);
            float sphericalFalloff = exp(-distFromCenter * distFromCenter * 3.0);
            density *= sphericalFalloff * densityMultiplier;

            // Skip if negligible
            if (density < 0.001) continue;

            // Compute emission color (with optional physical emission)
            float3 emission;
            float intensity;

            if (usePhysicalEmission != 0) {
                // Physical blackbody emission
                emission = ComputePlasmaEmission(
                    p.position,
                    p.velocity,
                    p.temperature,
                    p.density,
                    cameraPos
                );

                // Apply emission strength
                emission = lerp(float3(0.5, 0.5, 0.5), emission, emissionStrength);

                // Optional Doppler shift
                if (useDopplerShift != 0) {
                    float3 viewDir = normalize(cameraPos - p.position);
                    emission = DopplerShift(emission, p.velocity, viewDir, dopplerStrength);
                }

                // Optional gravitational redshift
                if (useGravitationalRedshift != 0) {
                    float radius = length(p.position);
                    const float schwarzschildRadius = 2.0;
                    emission = GravitationalRedshift(emission, radius, schwarzschildRadius, redshiftStrength);
                }

                intensity = EmissionIntensity(p.temperature);
            } else {
                // Standard temperature-based color
                emission = TemperatureToEmission(p.temperature);
                intensity = EmissionIntensity(p.temperature);
            }

            // === FIXED: RT lighting as illumination, not replacement ===
            float3 rtLight = g_rtLighting[hit.particleIdx].rgb;

            // RT light acts as external illumination on the particle volume
            // It modulates the emission based on received light
            float3 illumination = float3(1, 1, 1); // Base self-illumination

            // Add RT lighting as external contribution (like being lit by neighbors)
            illumination += rtLight * 0.5; // Reduced influence to preserve colors

            // === NEW: Cast shadow ray to primary light source (TOGGLEABLE) ===
            float shadowTerm = 1.0;
            if (useShadowRays != 0) {
                float3 toLightDir = normalize(volParams.lightPos - pos);
                float lightDist = length(volParams.lightPos - pos);
                shadowTerm = CastShadowRay(pos, toLightDir, lightDist);
            }

            // Apply shadow to external illumination
            illumination *= lerp(0.3, 1.0, shadowTerm); // Ambient + direct

            // === NEW: Add in-scattering for volumetric depth (TOGGLEABLE) ===
            float3 inScatter = float3(0, 0, 0);
            if (useInScattering != 0) {
                inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
            }

            // === FIXED: Combine emission with illumination properly ===
            // Emission is the particle's intrinsic color
            // Illumination modulates it based on external light
            float3 totalEmission = emission * intensity * illumination + inScatter * 0.3;

            // Apply phase function for view-dependent scattering (TOGGLEABLE + ADJUSTABLE)
            if (usePhaseFunction != 0) {
                float3 lightDir = normalize(volParams.lightPos - pos);
                float cosTheta = dot(-ray.Direction, lightDir); // Negative for forward scattering
                float phase = HenyeyGreenstein(cosTheta, volParams.scatteringG);

                // Boost the phase function effect dramatically
                totalEmission *= (1.0 + phase * phaseStrength);
            }

            // Volume rendering equation with proper absorption/emission
            float absorption = density * stepSize * volParams.extinction;
            float3 emission_contribution = totalEmission * (1.0 - exp(-absorption));

            accumulatedColor += transmittance * emission_contribution;
            transmittance *= exp(-absorption);

            // Early exit
            if (transmittance < 0.001) break;
        }
    }

    // Background color (pure black space - no blue tint)
    float3 backgroundColor = float3(0.0, 0.0, 0.0);
    float3 finalColor = accumulatedColor + transmittance * backgroundColor;

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

    g_output[pixelPos] = float4(finalColor, 1.0);
}