// Enhanced 3D Gaussian Volumetric Renderer with Dramatic RT Effects
// Fixes for visible shadows, halos, and forward scattering

#include "gaussian_common.hlsl"
#include "plasma_emission.hlsl"

// Match C++ ParticleRenderer_Gaussian::RenderConstants structure
cbuffer GaussianConstants : register(b0)
{
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
    float3 cameraPos;
    float particleRadius;
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

// Derived values - ENHANCED for visibility
static const float2 resolution = float2(screenWidth, screenHeight);
static const float2 invResolution = 1.0 / resolution;
static const float baseParticleRadius = particleRadius;
static const float volumeStepSize = 0.5;         // Larger steps for performance
static const float densityMultiplier = 5.0;      // INCREASED for stronger volume
static const float shadowBias = 0.1;             // Larger bias to prevent self-shadowing

// Buffers
StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<float4> g_rtLighting : register(t1);
RaytracingAccelerationStructure g_particleBVH : register(t2);
RWTexture2D<float4> g_output : register(u0);

// Hit record for batch processing
struct HitRecord {
    uint particleIdx;
    float tNear;
    float tFar;
    float sortKey;
};

// Enhanced volumetric lighting parameters
struct VolumetricParams {
    float3 lightPos;       // MOVED OUTSIDE particle disk
    float3 lightColor;     // BRIGHTER light
    float scatteringG;     // STRONGER forward scattering
    float extinction;      // HIGHER extinction for dramatic shadows
    float ambientLevel;    // Ambient light to prevent pure black
};

// ENHANCED shadow ray with variable occlusion based on density
float CastShadowRay(float3 origin, float3 direction, float maxDist) {
    RayDesc shadowRay;
    shadowRay.Origin = origin + direction * shadowBias;
    shadowRay.Direction = direction;
    shadowRay.TMin = 0.01;
    shadowRay.TMax = maxDist;

    float accumOpticalDepth = 0.0;

    // Use RayQuery to accumulate optical depth along shadow ray
    RayQuery<RAY_FLAG_NONE> shadowQuery;
    shadowQuery.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, shadowRay);

    uint hitCount = 0;
    const uint maxHits = 8; // Sample up to 8 particles for shadows

    while (shadowQuery.Proceed() && hitCount < maxHits) {
        if (shadowQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = shadowQuery.CandidatePrimitiveIndex();
            Particle p = g_particles[particleIdx];

            // Calculate density contribution to optical depth
            float3 hitPoint = origin + direction * shadowQuery.CandidateTriangleBarycentrics().x;
            float3 scale = ComputeGaussianScale(p, baseParticleRadius);
            float3x3 rotation = ComputeGaussianRotation(p.velocity);

            // Sample density at hit point
            float density = EvaluateGaussianDensity(hitPoint, p.position, scale, rotation, p.density);

            // Accumulate optical depth (more density = more shadow)
            accumOpticalDepth += density * 2.0; // Stronger shadow accumulation

            shadowQuery.CommitProceduralPrimitiveHit(shadowQuery.CandidateTriangleBarycentrics().x);
            hitCount++;
        }
    }

    // Beer-Lambert law for transmittance with DRAMATIC falloff
    float transmittance = exp(-accumOpticalDepth * 3.0); // Stronger extinction

    // Add minimum ambient to prevent pure black shadows
    return max(0.05, transmittance);
}

// ENHANCED in-scattering with more samples and longer range
float3 ComputeInScattering(float3 pos, float3 viewDir, uint skipIdx, VolumetricParams volParams) {
    float3 totalScattering = float3(0, 0, 0);

    // INCREASED samples for better quality
    const uint numSamples = 12; // More samples for visible halos
    const float scatterRange = 150.0; // Longer range for extended glow

    // Stratified sampling on hemisphere towards light
    float3 lightDir = normalize(volParams.lightPos - pos);

    for (uint i = 0; i < numSamples; i++) {
        // Generate sample direction biased toward light
        float phi = (i + 0.5) * 6.28318 / numSamples;
        float theta = 0.5 + 0.3 * sin(phi * 2.0); // Vary elevation

        // Create sample direction in light-oriented hemisphere
        float3 tangent = normalize(cross(lightDir, float3(0, 1, 0)));
        float3 bitangent = cross(lightDir, tangent);

        float3 scatterDir = lightDir * cos(theta) +
                           tangent * sin(theta) * cos(phi) +
                           bitangent * sin(theta) * sin(phi);
        scatterDir = normalize(scatterDir);

        // Trace scatter ray
        RayDesc scatterRay;
        scatterRay.Origin = pos;
        scatterRay.Direction = scatterDir;
        scatterRay.TMin = 0.1;
        scatterRay.TMax = scatterRange;

        RayQuery<RAY_FLAG_NONE> scatterQuery;
        scatterQuery.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, scatterRay);

        float accumScatter = 0.0;
        uint scatterHits = 0;

        while (scatterQuery.Proceed() && scatterHits < 3) {
            if (scatterQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint idx = scatterQuery.CandidatePrimitiveIndex();
                if (idx != skipIdx) {
                    Particle p = g_particles[idx];

                    float dist = length(p.position - pos);

                    // ENHANCED distance attenuation - longer falloff for halos
                    float atten = 1.0 / (1.0 + dist * 0.02 + dist * dist * 0.0001);

                    // Get particle emission
                    float3 emission = TemperatureToEmission(p.temperature);
                    float intensity = EmissionIntensity(p.temperature);

                    // ENHANCED phase function for strong forward scattering
                    float cosTheta = dot(viewDir, scatterDir);
                    float phase = HenyeyGreenstein(cosTheta, volParams.scatteringG);

                    // Check if this scattered light is shadowed
                    float3 toLight = normalize(volParams.lightPos - p.position);
                    float shadowTerm = CastShadowRay(p.position, toLight, length(volParams.lightPos - p.position));

                    // Accumulate scattered light with shadow modulation
                    accumScatter += intensity * phase * atten * shadowTerm * 2.0; // Amplified
                    totalScattering += emission * accumScatter;

                    scatterHits++;
                }
            }
        }
    }

    return totalScattering / numSamples * 3.0; // Amplify final scattering
}

// Generate camera ray
RayDesc GenerateCameraRay(float2 pixelPos) {
    float2 ndc = (pixelPos + 0.5) * invResolution * 2.0 - 1.0;
    ndc.y = -ndc.y;

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

// Insert hit into sorted list
void InsertHit(inout HitRecord hits[64], inout uint hitCount, uint particleIdx, float tNear, float tFar, uint maxHits) {
    if (hitCount >= maxHits) return;

    HitRecord newHit;
    newHit.particleIdx = particleIdx;
    newHit.tNear = tNear;
    newHit.tFar = tFar;
    newHit.sortKey = tNear;

    uint insertPos = hitCount;
    for (uint i = 0; i < hitCount; i++) {
        if (newHit.sortKey < hits[i].sortKey) {
            insertPos = i;
            break;
        }
    }

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

    if (any(pixelPos >= (uint2)resolution))
        return;

    // Generate primary ray
    RayDesc ray = GenerateCameraRay((float2)pixelPos);

    // Collect all Gaussian intersections
    const uint MAX_HITS = 64;
    HitRecord hits[MAX_HITS];
    uint hitCount = 0;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = query.CandidatePrimitiveIndex();
            Particle p = g_particles[particleIdx];

            float3 scale = ComputeGaussianScale(p, baseParticleRadius);
            float3x3 rotation = ComputeGaussianRotation(p.velocity);

            float2 t = RayGaussianIntersection(ray.Origin, ray.Direction, p.position, scale, rotation);

            if (t.x > ray.TMin && t.x < ray.TMax && t.y > t.x) {
                query.CommitProceduralPrimitiveHit(t.x);
                InsertHit(hits, hitCount, particleIdx, t.x, t.y, MAX_HITS);
            }
        }
    }

    // ENHANCED volumetric parameters for DRAMATIC effects
    VolumetricParams volParams;

    // FIX 1: Move light WELL OUTSIDE the particle disk
    // Place it at 500 units away (disk is 10-300 radius)
    volParams.lightPos = float3(0, 500, 200);  // High and forward for dramatic shadows

    // FIX 2: MUCH brighter light for visible effects
    volParams.lightColor = float3(10, 10, 10); // 5x brighter

    // FIX 3: Strong forward scattering for dramatic halos
    volParams.scatteringG = 0.7;  // Increased from 0.3 for stronger forward scatter

    // FIX 4: Higher extinction for more dramatic shadows
    volParams.extinction = 2.0;   // Increased from 0.5

    // Ambient light to prevent pure black
    volParams.ambientLevel = 0.1;

    // Volume rendering with ENHANCED accumulation
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;

    for (uint i = 0; i < hitCount; i++) {
        if (transmittance < 0.001) break;

        HitRecord hit = hits[i];
        Particle p = g_particles[hit.particleIdx];

        float3 scale = ComputeGaussianScale(p, baseParticleRadius);
        float3x3 rotation = ComputeGaussianRotation(p.velocity);

        float tStart = max(hit.tNear, ray.TMin);
        float tEnd = min(hit.tFar, ray.TMax);

        uint steps = max(1, (uint)((tEnd - tStart) / volumeStepSize));
        float stepSize = (tEnd - tStart) / steps;

        for (uint step = 0; step < steps; step++) {
            float t = tStart + (step + 0.5) * stepSize;
            float3 pos = ray.Origin + ray.Direction * t;

            // Evaluate Gaussian density
            float density = EvaluateGaussianDensity(pos, p.position, scale, rotation, p.density);

            // Enhanced spherical falloff
            float3 toCenter = pos - p.position;
            float distFromCenter = length(toCenter) / length(scale);
            float sphericalFalloff = exp(-distFromCenter * distFromCenter * 4.0); // Stronger falloff
            density *= sphericalFalloff * densityMultiplier;

            if (density < 0.001) continue;

            // Base emission color
            float3 emission;
            float intensity;

            if (usePhysicalEmission != 0) {
                emission = ComputePlasmaEmission(p.position, p.velocity, p.temperature, p.density, cameraPos);
                emission = lerp(float3(0.5, 0.5, 0.5), emission, emissionStrength);

                if (useDopplerShift != 0) {
                    float3 viewDir = normalize(cameraPos - p.position);
                    emission = DopplerShift(emission, p.velocity, viewDir, dopplerStrength);
                }

                if (useGravitationalRedshift != 0) {
                    float radius = length(p.position);
                    const float schwarzschildRadius = 2.0;
                    emission = GravitationalRedshift(emission, radius, schwarzschildRadius, redshiftStrength);
                }

                intensity = EmissionIntensity(p.temperature);
            } else {
                emission = TemperatureToEmission(p.temperature);
                intensity = EmissionIntensity(p.temperature);
            }

            // RT lighting contribution
            float3 rtLight = g_rtLighting[hit.particleIdx].rgb;

            // ENHANCED illumination model
            float3 illumination = float3(volParams.ambientLevel, volParams.ambientLevel, volParams.ambientLevel);

            // Add RT lighting
            illumination += rtLight * 2.0; // Stronger RT contribution

            // ENHANCED SHADOW RAYS - now with proper light position
            float shadowTerm = 1.0;
            if (useShadowRays != 0) {
                float3 toLightDir = normalize(volParams.lightPos - pos);
                float lightDist = length(volParams.lightPos - pos);
                shadowTerm = CastShadowRay(pos, toLightDir, lightDist);

                // Apply shadow with strong contrast
                illumination *= lerp(volParams.ambientLevel, 1.0, shadowTerm);

                // Add direct light contribution when not in shadow
                float3 directLight = volParams.lightColor * shadowTerm;
                illumination += directLight * 0.5;
            }

            // ENHANCED IN-SCATTERING for visible halos
            float3 inScatter = float3(0, 0, 0);
            if (useInScattering != 0) {
                inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx, volParams);
                // Boost in-scattering contribution significantly
                inScatter *= 2.0;
            }

            // Combine emission with enhanced illumination
            float3 totalEmission = emission * intensity * illumination + inScatter;

            // ENHANCED PHASE FUNCTION for dramatic forward scattering
            if (usePhaseFunction != 0) {
                float3 lightDir = normalize(volParams.lightPos - pos);
                float cosTheta = dot(-ray.Direction, lightDir);

                // Use stronger g parameter for more dramatic effect
                float phase = HenyeyGreenstein(cosTheta, volParams.scatteringG);

                // DRAMATICALLY boost phase function effect
                // Forward scattering (cosTheta > 0) gets huge boost
                float phaseBoost = 1.0 + phase * phaseStrength * 5.0; // 5x multiplier

                // Add rim lighting effect for halos
                float rimAngle = 1.0 - abs(cosTheta);
                float rimLight = pow(rimAngle, 2.0) * 0.5;

                totalEmission *= (phaseBoost + rimLight);
            }

            // Volume rendering equation with enhanced absorption
            float absorption = density * stepSize * volParams.extinction;
            float3 emission_contribution = totalEmission * (1.0 - exp(-absorption));

            accumulatedColor += transmittance * emission_contribution;
            transmittance *= exp(-absorption);

            if (transmittance < 0.001) break;
        }
    }

    // Dark space background with slight blue tint for contrast
    float3 backgroundColor = float3(0.01, 0.01, 0.02);
    float3 finalColor = accumulatedColor + transmittance * backgroundColor;

    // DEBUG: Visual indicators for active features with larger regions
    if (useShadowRays != 0 && pixelPos.x < 100 && pixelPos.y < 20) {
        // Red bar for shadows
        finalColor += float3(0.5, 0, 0);
    }
    if (useInScattering != 0 && pixelPos.x > resolution.x - 100 && pixelPos.x < resolution.x && pixelPos.y < 20) {
        // Green bar for in-scattering
        finalColor += float3(0, 0.5, 0);
    }
    if (usePhaseFunction != 0 && pixelPos.x < 100 && pixelPos.y > resolution.y - 20) {
        // Blue bar for phase function
        finalColor += float3(0, 0, 0.5);
    }

    // Exposure control for HDR content
    float exposure = 1.5; // Boost overall brightness
    finalColor *= exposure;

    // Enhanced tone mapping for high dynamic range
    // Reinhard extended tone mapping for better highlights
    float luminance = dot(finalColor, float3(0.2126, 0.7152, 0.0722));
    float white = 4.0; // White point
    float mappedLum = (luminance * (1.0 + luminance / (white * white))) / (1.0 + luminance);
    finalColor = finalColor * (mappedLum / luminance);

    // Subtle bloom simulation for glow
    float bloom = smoothstep(0.8, 1.0, luminance) * 0.3;
    finalColor += finalColor * bloom;

    // Gamma correction
    finalColor = pow(saturate(finalColor), 1.0 / 2.2);

    g_output[pixelPos] = float4(finalColor, 1.0);
}