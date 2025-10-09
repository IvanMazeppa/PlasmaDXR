// DXR Particle-to-Particle Ray Traced Lighting
// Uses RayQuery to trace hemisphere rays from each particle to find nearby emitters
// This is GENUINE ray tracing using DXR 1.1 hardware acceleration

cbuffer LightingConstants : register(b0)
{
    uint particleCount;
    uint raysPerParticle;      // 8 for high quality, 4 for medium, 2 for low
    float maxLightingDistance; // Ray TMax (e.g., 20.0)
    float lightingIntensity;   // Global intensity multiplier
};

// Input: Particle buffer (read positions and temperatures)
struct Particle
{
    float3 position;    // Offset 0-11
    float temperature;  // Offset 12-15
    float3 velocity;    // Offset 16-27
    float density;      // Offset 28-31
};

StructuredBuffer<Particle> g_particles : register(t0);

// Input: Ray tracing acceleration structure (per-particle BLAS)
RaytracingAccelerationStructure g_particleBVH : register(t1);

// Output: Lighting contribution per particle (RGBA, alpha unused)
RWStructuredBuffer<float4> g_particleLighting : register(u0);

// Helper: Fibonacci hemisphere sampling for even ray distribution
float3 FibonacciHemisphere(uint sampleIndex, uint numSamples, float3 normal)
{
    const float PHI = 1.618033988749895; // Golden ratio

    float theta = 2.0 * 3.14159265359 * sampleIndex / PHI;
    float phi = acos(1.0 - 2.0 * (sampleIndex + 0.5) / numSamples);

    float sinPhi = sin(phi);
    float x = cos(theta) * sinPhi;
    float y = sin(theta) * sinPhi;
    float z = cos(phi);

    // For now, assume hemisphere is oriented upwards (can be normal-aligned later)
    return normalize(float3(x, y, z));
}

// Helper: Convert temperature to emission color (blackbody radiation)
float3 TemperatureToEmission(float temperature)
{
    // Normalize temperature to 0-1 range (800K to 26000K)
    float t = saturate((temperature - 800.0) / 25200.0);

    // Same color gradient as particle rendering
    float3 color;
    if (t < 0.25) {
        float blend = t / 0.25;
        color = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
    } else if (t < 0.5) {
        float blend = (t - 0.25) / 0.25;
        color = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
    } else if (t < 0.75) {
        float blend = (t - 0.5) / 0.25;
        color = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
    } else {
        float blend = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
    }
    return color;
}

// Helper: Calculate emission intensity from temperature
float EmissionIntensity(float temperature)
{
    // Stefan-Boltzmann law: L ∝ T^4 (simplified for real-time)
    float normalized = temperature / 26000.0; // Normalize to max temp
    return pow(normalized, 2.0); // Quadratic falloff for performance (instead of T^4)
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIdx = dispatchThreadID.x;

    // Early exit if outside particle count
    if (particleIdx >= particleCount)
        return;

    // Read receiver particle
    Particle receiver = g_particles[particleIdx];
    float3 receiverPos = receiver.position;

    // Accumulate lighting from hemisphere rays
    float3 accumulatedLight = float3(0, 0, 0);

    // Cast rays in hemisphere around particle
    for (uint rayIdx = 0; rayIdx < raysPerParticle; rayIdx++)
    {
        // Generate ray direction using Fibonacci hemisphere sampling
        float3 rayDir = FibonacciHemisphere(rayIdx, raysPerParticle, float3(0, 1, 0));

        // Setup ray descriptor
        RayDesc ray;
        ray.Origin = receiverPos + rayDir * 0.01; // Offset slightly to avoid self-intersection
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = maxLightingDistance;

        // Create RayQuery object (DXR 1.1 inline ray tracing)
        RayQuery<RAY_FLAG_NONE> query;  // Don't use ACCEPT_FIRST_HIT for procedural - we need to test manually
        query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

        // Process all candidates (for procedural primitives, we must manually test intersection)
        while (query.Proceed())
        {
            // Check if we hit an AABB (procedural primitive candidate)
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                // Get candidate particle index
                uint candidateIdx = query.CandidatePrimitiveIndex();

                // Skip self-intersection
                if (candidateIdx == particleIdx)
                    continue;

                // Read particle position
                Particle candidate = g_particles[candidateIdx];

                // Manual sphere-ray intersection test
                // Use VERY large radius - particles are 30-50 units apart in outer disk!
                const float rtLightingRadius = 25.0;
                float3 oc = ray.Origin - candidate.position;
                float b = dot(oc, ray.Direction);
                float c = dot(oc, oc) - (rtLightingRadius * rtLightingRadius);
                float discriminant = b * b - c;

                // If ray hits sphere, commit the hit
                if (discriminant >= 0.0)
                {
                    float t = -b - sqrt(discriminant);
                    if (t >= ray.TMin && t <= ray.TMax)
                    {
                        // Commit this procedural primitive hit
                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }
        }

        // After Proceed() loop completes, check if we got a committed hit
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            // Get hit particle index (PrimitiveIndex maps to particle index)
            uint hitParticleIdx = query.CommittedPrimitiveIndex();

            // Avoid self-illumination
            if (hitParticleIdx == particleIdx)
                continue;

            // Read emitter particle
            Particle emitter = g_particles[hitParticleIdx];

            // Calculate emission based on temperature
            float intensity = EmissionIntensity(emitter.temperature);
            float3 emissionColor = TemperatureToEmission(emitter.temperature);

            // Calculate distance falloff (LINEAR falloff for sparse particles!)
            // Particles are 30-50 units apart - inverse square kills lighting at this scale
            float distance = query.CommittedRayT();
            float attenuation = 1.0 / (1.0 + distance * 0.01);  // Very weak falloff

            // Accumulate lighting contribution
            accumulatedLight += emissionColor * intensity * attenuation;
        }
    }

    // Average lighting over all rays and apply global intensity
    // Boost intensity dramatically for sparse particles
    float3 finalLight = (accumulatedLight / float(raysPerParticle)) * lightingIntensity * 50.0;

    // DEBUG REMOVED: Show actual lighting (black=no hits, colored=RT lighting)
    // Particles with no neighbors will be black, center will be bright

    g_particleLighting[particleIdx] = float4(finalLight, 0.0);
}
