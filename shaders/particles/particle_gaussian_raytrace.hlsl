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
    float inScatterStrength;
    float rtLightingStrength;
    uint useAnisotropicGaussians;
    float anisotropyStrength;

    // ReSTIR parameters
    uint useReSTIR;
    uint restirInitialCandidates;  // M = 16-32
    uint frameIndex;               // For temporal validation
    float restirTemporalWeight;    // 0-1, how much to trust previous frame
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

// ReSTIR: Reservoir structure (matches C++ side: 32 bytes for cache alignment)
struct Reservoir {
    float3 lightPos;       // 12 bytes - position of selected light source
    float weightSum;       // 4 bytes  - sum of weights (W)
    uint M;                // 4 bytes  - number of samples seen
    float W;               // 4 bytes  - final weight for this sample
    uint particleIdx;      // 4 bytes  - which particle is providing light
    float pad;             // 4 bytes  - padding to 32 bytes
};

// ReSTIR: Previous frame's reservoirs (read-only)
StructuredBuffer<Reservoir> g_prevReservoirs : register(t3);

// Output: Final rendered image
RWTexture2D<float4> g_output : register(u0);

// ReSTIR: Current frame's reservoirs (write-only)
RWStructuredBuffer<Reservoir> g_currentReservoirs : register(u1);

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

// Update reservoir with new sample using weighted reservoir sampling
void UpdateReservoir(inout Reservoir r, float3 lightPos, uint particleIdx, float weight, float random) {
    r.weightSum += weight;
    r.M += 1;

    // Probabilistically replace current sample
    float probability = weight / max(r.weightSum, 0.0001);
    if (random < probability) {
        r.lightPos = lightPos;
        r.particleIdx = particleIdx;
    }
}

// Validate previous frame's reservoir (check if light source still visible)
bool ValidateReservoir(Reservoir prevReservoir, float3 viewPos) {
    if (prevReservoir.M == 0) return false;

    // Cast shadow ray to previous light source
    float3 toLightDir = normalize(prevReservoir.lightPos - viewPos);
    float lightDist = length(prevReservoir.lightPos - viewPos);

    RayDesc ray;
    ray.Origin = viewPos + toLightDir * 0.01;
    ray.Direction = toLightDir;
    ray.TMin = 0.001;
    ray.TMax = lightDist - 0.01;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        uint hitIdx = query.CommittedPrimitiveIndex();
        // Valid if we hit the same particle (or within tolerance)
        return (hitIdx == prevReservoir.particleIdx);
    }

    // Valid if unoccluded
    return (query.CommittedStatus() == COMMITTED_NOTHING);
}

// Sample light particles using importance sampling
// Note: rayDirection is the CAMERA ray direction (not used for sampling, just for phase function later)
Reservoir SampleLightParticles(float3 rayOrigin, float3 rayDirection, uint pixelIndex, uint numCandidates) {
    Reservoir reservoir;
    reservoir.lightPos = float3(0, 0, 0);
    reservoir.weightSum = 0;
    reservoir.M = 0;
    reservoir.W = 0;
    reservoir.particleIdx = 0;
    reservoir.pad = 0;

    // DEBUG: Track how many rays we trace and how many hit
    uint raysTraced = 0;
    uint raysHit = 0;

    for (uint i = 0; i < numCandidates; i++) {
        raysTraced++;
        // Generate random direction (uniform sphere)
        float rand1 = Hash(pixelIndex * numCandidates + i + frameIndex * 1000);
        float rand2 = Hash(pixelIndex * numCandidates + i + frameIndex * 1000 + 1);

        // Uniform sphere sampling (not hemisphere - we want all directions)
        float cosTheta = 2.0 * rand1 - 1.0;  // -1 to 1
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi = 2.0 * 3.14159265 * rand2;

        float3 sampleDir;
        sampleDir.x = sinTheta * cos(phi);
        sampleDir.y = sinTheta * sin(phi);
        sampleDir.z = cosTheta;

        // Trace ray to find light source
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = normalize(sampleDir);
        ray.TMin = 0.01;
        ray.TMax = 500.0;  // Longer range to find more particles

        // Use RayQuery to find ANY procedural primitive hit
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

        // Process candidates - for procedural primitives we need to loop!
        while (query.Proceed()) {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                // Get the primitive index to find actual particle position
                uint candidateIdx = query.CandidatePrimitiveIndex();
                Particle candidateParticle = g_particles[candidateIdx];

                // FIX: Compute proper Gaussian parameters (same as main rendering loop)
                float3 scale = ComputeGaussianScale(candidateParticle, baseParticleRadius,
                                                    useAnisotropicGaussians != 0,
                                                    anisotropyStrength);
                float3x3 rotation = ComputeGaussianRotation(candidateParticle.velocity);

                // FIX: Compute actual ray-ellipsoid intersection (not just distance to center)
                float2 t = RayGaussianIntersection(ray.Origin, ray.Direction,
                                                   candidateParticle.position,
                                                   scale, rotation);

                // FIX: Proper validation (entry < exit, within ray range)
                if (t.x > ray.TMin && t.x < ray.TMax && t.y > t.x) {
                    // Commit the entry point (t.x), not the center distance
                    query.CommitProceduralPrimitiveHit(t.x);
                    // Let BVH traversal complete naturally (no break)
                }
            }
        }

        // Check if we got a hit
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            raysHit++;  // DEBUG: Count successful hits
            uint hitParticleIdx = query.CommittedPrimitiveIndex();
            Particle hitParticle = g_particles[hitParticleIdx];

            // Compute light contribution
            float3 emission = TemperatureToEmission(hitParticle.temperature);
            float intensity = EmissionIntensity(hitParticle.temperature);
            float dist = length(hitParticle.position - rayOrigin);

            // FIXED: Weaker attenuation for large scenes (accretion disk spans 10-300 radii)
            // Use linear + quadratic falloff with minimum to prevent division by zero
            float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

            // Weight = luminance of light contribution (importance)
            float weight = dot(emission * intensity * attenuation, float3(0.299, 0.587, 0.114));

            // DEBUG: For first pixel, store weight info in reservoir for debugging
            if (pixelIndex == 0 && raysHit == 1) {
                // Store debug info: temperature, intensity, weight in first hit
                reservoir.lightPos = float3(hitParticle.temperature, intensity, weight);
            }

            // FIXED: Lower threshold to capture low-temp particles at distance (10x more sensitive)
            // Agent analysis: 800K particles at 500+ units need this lower threshold
            if (weight > 0.000001) {
                // Random value for reservoir update
                float random = Hash(pixelIndex * numCandidates + i + frameIndex * 2000);

                // Update reservoir
                UpdateReservoir(reservoir, hitParticle.position, hitParticleIdx, weight, random);
            }
        }
    }

    // Compute final weight
    if (reservoir.M > 0) {
        reservoir.W = reservoir.weightSum / float(reservoir.M);
    }

    // DEBUG: Always encode hit stats for debugging
    if (reservoir.M == 0) {
        // No hits found - encode debug info to understand why
        // X = rays traced, Y = rays hit, Z = special marker
        reservoir.lightPos = float3(float(raysTraced), float(raysHit), 8888.0);
        reservoir.M = 88888;  // Special debug marker for "no hits"
    }

    return reservoir;
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
    Reservoir currentReservoir;
    currentReservoir.lightPos = float3(0, 0, 0);
    currentReservoir.weightSum = 0;
    currentReservoir.M = 0;
    currentReservoir.W = 0;
    currentReservoir.particleIdx = 0;
    currentReservoir.pad = 0;

    if (useReSTIR != 0) {
        // Load previous frame's reservoir
        Reservoir prevReservoir = g_prevReservoirs[pixelIndex];

        // Validate temporal sample (check if still visible from current CAMERA position)
        bool temporalValid = ValidateReservoir(prevReservoir, cameraPos);

        // Initialize current reservoir
        currentReservoir.lightPos = float3(0, 0, 0);
        currentReservoir.weightSum = 0;
        currentReservoir.M = 0;
        currentReservoir.W = 0;
        currentReservoir.particleIdx = 0;
        currentReservoir.pad = 0;

        // Reuse temporal sample if valid AND has non-zero weight (BUG FIX!)
        // CRITICAL: Without weightSum check, M persists while weightSum decays to 0
        if (temporalValid && prevReservoir.M > 0 && prevReservoir.weightSum > 0.000001) {
            // Decay M to prevent infinite accumulation
            float temporalM = prevReservoir.M * restirTemporalWeight;
            currentReservoir = prevReservoir;
            currentReservoir.M = max(1, uint(temporalM)); // Keep at least 1 sample

            // CRITICAL: Also decay weightSum proportionally to maintain W = weightSum/M balance
            currentReservoir.weightSum = prevReservoir.weightSum * restirTemporalWeight;
        }

        // Generate new candidate samples for this frame
        // Use CAMERA position, not near plane position!
        Reservoir newSamples = SampleLightParticles(cameraPos, ray.Direction,
                                                     pixelIndex, restirInitialCandidates);

        // Combine temporal + new samples using reservoir merging
        // IMPORTANT: Skip if M is our debug marker (88888) which means no real hits
        if (newSamples.M > 0 && newSamples.M != 88888) {
            // Update with new samples
            float combinedWeight = currentReservoir.weightSum + newSamples.weightSum;
            currentReservoir.M += newSamples.M;

            // Probabilistically select between temporal and new
            float random = Hash(pixelIndex + frameIndex * 3000);
            float newProbability = newSamples.weightSum / max(combinedWeight, 0.0001);

            if (random < newProbability) {
                currentReservoir.lightPos = newSamples.lightPos;
                currentReservoir.particleIdx = newSamples.particleIdx;
            }

            currentReservoir.weightSum = combinedWeight;
        }

        // Compute final weight
        if (currentReservoir.M > 0) {
            currentReservoir.W = currentReservoir.weightSum / float(currentReservoir.M);
        }

        // Store for next frame
        g_currentReservoirs[pixelIndex] = currentReservoir;
    } else {
        // DEBUG: Even when ReSTIR is OFF, write a test value to verify buffer binding works
        currentReservoir.lightPos = float3(pixelPos.x, pixelPos.y, 999.0);
        currentReservoir.M = 12345;  // Magic number to verify writes work
        g_currentReservoirs[pixelIndex] = currentReservoir;
    }

    // Volume rendering: march through sorted Gaussians
    float3 accumulatedColor = float3(0, 0, 0);
    float transmittance = 1.0;

    // Setup volumetric lighting
    VolumetricParams volParams;
    volParams.lightPos = float3(0, 500, 200);  // Light OUTSIDE the disk for proper shadows
    volParams.lightColor = float3(10, 10, 10); // Much brighter light for visible effects
    volParams.scatteringG = 0.7;            // Stronger forward scattering for halos
    volParams.extinction = 1.0;             // Stronger extinction for more dramatic shadows

    for (uint i = 0; i < hitCount; i++) {
        // Early exit if fully opaque
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
            float3 rtLight;

            if (useReSTIR != 0 && currentReservoir.M > 0 && currentReservoir.M != 88888) {
                // ReSTIR: Use the intelligently sampled light source
                Particle lightParticle = g_particles[currentReservoir.particleIdx];
                float3 lightEmission = TemperatureToEmission(lightParticle.temperature);
                float lightIntensity = EmissionIntensity(lightParticle.temperature);
                float dist = length(currentReservoir.lightPos - pos);

                // FIXED: Use same attenuation as sampling (must match for unbiased estimate!)
                float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

                // FIX: ReSTIR W already represents the average contribution
                // Don't scale by M - that causes over-brightness when M is high
                // W = weightSum / M is the unbiased estimator
                rtLight = lightEmission * lightIntensity * attenuation * currentReservoir.W;
            } else {
                // Fallback: Use pre-computed RT lighting
                rtLight = g_rtLighting[hit.particleIdx].rgb;
            }

            // RT light acts as external illumination on the particle volume
            // It modulates the emission based on received light
            float3 illumination = float3(1, 1, 1); // Base self-illumination

            // Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
            // Clamp to prevent over-brightness from extreme ReSTIR samples
            illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);

            // === NEW: Cast shadow ray to primary light source (TOGGLEABLE) ===
            float shadowTerm = 1.0;
            if (useShadowRays != 0) {
                float3 toLightDir = normalize(volParams.lightPos - pos);
                float lightDist = length(volParams.lightPos - pos);
                shadowTerm = CastShadowRay(pos, toLightDir, lightDist);
            }

            // Apply shadow to external illumination
            illumination *= lerp(0.1, 1.0, shadowTerm); // Deeper shadows for contrast

            // === NEW: Add in-scattering for volumetric depth (TOGGLEABLE) ===
            float3 inScatter = float3(0, 0, 0);
            if (useInScattering != 0) {
                inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
            }

            // === FIXED: Combine emission with illumination properly ===
            // Emission is the particle's intrinsic color
            // Illumination modulates it based on external light
            // In-scattering adds volumetric glow from nearby particles (RUNTIME ADJUSTABLE)
            float3 totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;

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
        if (useReSTIR != 0) {
            // ReSTIR is ON - show reservoir quality
            if (currentReservoir.M == 0) {
                // No samples found - RED alert!
                finalColor = float3(1, 0, 0);
            } else if (currentReservoir.M < restirInitialCandidates / 2) {
                // Few samples - Orange warning
                finalColor = float3(1, 0.5, 0);
            } else {
                // Good samples - Green success with brightness showing quality
                float quality = saturate(float(currentReservoir.M) / float(restirInitialCandidates));
                finalColor = float3(0, quality, 0);
            }
        } else {
            // ReSTIR is OFF - show gray
            finalColor = float3(0.3, 0.3, 0.3);
        }
    }

    g_output[pixelPos] = float4(finalColor, 1.0);
}