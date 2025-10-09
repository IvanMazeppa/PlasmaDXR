// DXR Particle-to-Particle Ray Traced Lighting - FIXED VERSION
// Improved to provide proper volumetric illumination with occlusion

cbuffer LightingConstants : register(b0)
{
    uint particleCount;
    uint raysPerParticle;      // 16 for best quality, 8 for high, 4 for medium
    float maxLightingDistance; // Ray TMax (e.g., 50.0 for larger range)
    float lightingIntensity;   // Global intensity multiplier (1.0-2.0 typical)
    float occlusionStrength;   // How much particles block light (0.5-1.0)
    float3 padding;
};

// Particle structure matching main shader
struct Particle
{
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

StructuredBuffer<Particle> g_particles : register(t0);
RaytracingAccelerationStructure g_particleBVH : register(t1);
RWStructuredBuffer<float4> g_particleLighting : register(u0);

// Improved hemisphere sampling using stratified Monte Carlo
float3 StratifiedHemisphereSample(uint sampleIndex, uint totalSamples, float3 normal)
{
    // Stratified sampling for better distribution
    uint sqrtSamples = (uint)sqrt((float)totalSamples);
    uint gridX = sampleIndex % sqrtSamples;
    uint gridY = sampleIndex / sqrtSamples;

    // Add jitter for anti-aliasing
    float u = (gridX + 0.5) / sqrtSamples;
    float v = (gridY + 0.5) / sqrtSamples;

    // Cosine-weighted hemisphere sampling for better importance sampling
    float phi = 2.0 * 3.14159265359 * u;
    float cosTheta = sqrt(1.0 - v); // Cosine weighting
    float sinTheta = sqrt(v);

    float3 sample = float3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    // Build tangent frame
    float3 up = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    // Transform to world space
    return tangent * sample.x + bitangent * sample.y + normal * sample.z;
}

// Physical blackbody emission with proper intensity
float3 BlackbodyEmission(float temperature)
{
    // Wien's displacement law for peak wavelength
    float peakWavelength = 2.898e6 / temperature; // nm

    // Approximate blackbody spectrum in RGB
    float t = saturate((temperature - 800.0) / 25200.0);

    float3 color;
    if (t < 0.2) {
        // Deep red to orange (cool plasma)
        color = lerp(float3(0.2, 0.02, 0.0), float3(1.0, 0.3, 0.05), t * 5.0);
    } else if (t < 0.4) {
        // Orange to yellow
        color = lerp(float3(1.0, 0.3, 0.05), float3(1.0, 0.7, 0.2), (t - 0.2) * 5.0);
    } else if (t < 0.6) {
        // Yellow to white-yellow
        color = lerp(float3(1.0, 0.7, 0.2), float3(1.0, 0.95, 0.8), (t - 0.4) * 5.0);
    } else if (t < 0.8) {
        // White-yellow to white
        color = lerp(float3(1.0, 0.95, 0.8), float3(1.0, 1.0, 0.95), (t - 0.6) * 5.0);
    } else {
        // White to blue-white (hot plasma)
        color = lerp(float3(1.0, 1.0, 0.95), float3(0.8, 0.85, 1.0), (t - 0.8) * 5.0);
    }

    // Add emission lines for hot plasma (H-alpha, etc.)
    if (temperature > 15000.0) {
        float lineStrength = (temperature - 15000.0) / 15000.0;
        color += float3(0.3, 0.0, 0.0) * lineStrength; // H-alpha boost
    }

    return color;
}

// Stefan-Boltzmann emission intensity
float EmissionIntensity(float temperature)
{
    // L ∝ T^4, but scale for visibility
    float normalized = saturate(temperature / 26000.0);
    return pow(normalized, 2.5); // Slightly higher than quadratic for drama
}

// Compute light transmission through participating media
float ComputeTransmission(RayDesc ray, uint skipIdx)
{
    float transmission = 1.0;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = query.CandidatePrimitiveIndex();
            if (particleIdx == skipIdx) continue;

            Particle p = g_particles[particleIdx];

            // Simplified sphere test for transmission
            float3 oc = ray.Origin - p.position;
            float b = dot(oc, ray.Direction);
            float radiusSq = 25.0; // Matches visual particle size better
            float c = dot(oc, oc) - radiusSq;
            float discriminant = b * b - c;

            if (discriminant >= 0.0) {
                float sqrtDisc = sqrt(discriminant);
                float t1 = -b - sqrtDisc;
                float t2 = -b + sqrtDisc;

                if (t1 >= ray.TMin && t1 <= ray.TMax) {
                    // Compute optical depth through particle
                    float pathLength = min(t2 - t1, 10.0); // Cap path length
                    float opticalDepth = p.density * pathLength * 0.1;
                    transmission *= exp(-opticalDepth);

                    if (transmission < 0.01) break; // Early exit for opaque paths
                }
            }
        }
    }

    return transmission;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIdx = dispatchThreadID.x;

    if (particleIdx >= particleCount)
        return;

    Particle receiver = g_particles[particleIdx];
    float3 receiverPos = receiver.position;

    // Accumulate direct illumination and shadowing
    float3 directLight = float3(0, 0, 0);
    float3 indirectLight = float3(0, 0, 0);
    float occlusion = 0.0;

    // Adaptive ray count based on particle importance
    uint adaptiveRays = raysPerParticle;
    if (receiver.temperature > 20000.0) {
        adaptiveRays = min(32, raysPerParticle * 2); // More rays for hot particles
    }

    // Cast rays in stratified hemisphere pattern
    for (uint rayIdx = 0; rayIdx < adaptiveRays; rayIdx++)
    {
        // Generate better distributed ray direction
        float3 normal = normalize(receiver.velocity);
        if (length(normal) < 0.1) normal = float3(0, 1, 0); // Default up if no velocity

        float3 rayDir = StratifiedHemisphereSample(rayIdx, adaptiveRays, normal);

        RayDesc ray;
        ray.Origin = receiverPos + rayDir * 0.1; // Larger bias for volumetric particles
        ray.Direction = rayDir;
        ray.TMin = 0.01;
        ray.TMax = maxLightingDistance;

        // Find closest hit using inline ray tracing
        float closestT = ray.TMax;
        uint closestIdx = 0xFFFFFFFF;

        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

        while (query.Proceed()) {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint candidateIdx = query.CandidatePrimitiveIndex();
                if (candidateIdx == particleIdx) continue;

                Particle candidate = g_particles[candidateIdx];

                // Accurate sphere intersection matching visual size
                float particleRadius = 3.0; // Match visual particle size
                float3 oc = ray.Origin - candidate.position;
                float b = dot(oc, ray.Direction);
                float c = dot(oc, oc) - (particleRadius * particleRadius);
                float discriminant = b * b - c;

                if (discriminant >= 0.0) {
                    float t = -b - sqrt(discriminant);
                    if (t >= ray.TMin && t < closestT) {
                        closestT = t;
                        closestIdx = candidateIdx;
                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }
        }

        // Process the hit
        if (closestIdx != 0xFFFFFFFF) {
            Particle emitter = g_particles[closestIdx];

            // Get emission properties
            float3 emissionColor = BlackbodyEmission(emitter.temperature);
            float intensity = EmissionIntensity(emitter.temperature);

            // === VOLUMETRIC IMPROVEMENTS ===

            // 1. Distance attenuation with atmospheric scattering
            float distance = closestT;
            float attenuation = 1.0 / (1.0 + distance * 0.05 + distance * distance * 0.002);

            // 2. Compute transmission through medium (creates shadows)
            RayDesc shadowRay;
            shadowRay.Origin = receiverPos;
            shadowRay.Direction = rayDir;
            shadowRay.TMin = 0.01;
            shadowRay.TMax = distance;

            float transmission = ComputeTransmission(shadowRay, particleIdx);

            // 3. View-dependent effects (limb darkening/brightening)
            float3 toEmitter = normalize(emitter.position - receiverPos);
            float viewDot = abs(dot(toEmitter, rayDir));
            float limbEffect = pow(viewDot, 0.5); // Subtle limb brightening

            // 4. Density-based emission boost (denser = brighter)
            float densityBoost = 1.0 + emitter.density * 0.5;

            // Combine all factors
            float3 contribution = emissionColor * intensity * attenuation *
                                 transmission * limbEffect * densityBoost;

            // Separate direct and indirect based on angle
            float directness = max(0.0, dot(rayDir, float3(0, 1, 0))); // Up = direct
            directLight += contribution * directness;
            indirectLight += contribution * (1.0 - directness);

            // Accumulate occlusion for ambient darkening
            occlusion += (1.0 - transmission) * occlusionStrength;
        }
        else {
            // No hit - contributes to ambient lighting
            indirectLight += float3(0.01, 0.01, 0.02) * 0.1; // Slight blue ambient
        }
    }

    // Normalize and combine lighting
    directLight /= float(adaptiveRays);
    indirectLight /= float(adaptiveRays);
    occlusion = saturate(occlusion / float(adaptiveRays));

    // Final lighting combines direct + indirect with occlusion
    float3 finalLight = directLight + indirectLight * 0.3;

    // Apply ambient occlusion to darken occluded areas
    finalLight *= (1.0 - occlusion * 0.5);

    // Global intensity control
    finalLight *= lightingIntensity;

    // Add subtle self-emission based on particle's own temperature
    // This ensures particles are never completely black
    float3 selfEmission = BlackbodyEmission(receiver.temperature) *
                          EmissionIntensity(receiver.temperature) * 0.1;
    finalLight += selfEmission;

    // Store with improved dynamic range
    g_particleLighting[particleIdx] = float4(finalLight, occlusion);
}