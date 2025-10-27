// Particle-to-Particle Indirect Lighting (First Bounce GI)
// Traces hemisphere rays and samples DIRECT lighting from hit particles
// Applies diffuse BRDF for physically correct indirect reflection
//
// Architecture:
// - Input: Direct lighting buffer from previous pass
// - Output: Indirect lighting contribution
// - Method: RayQuery inline ray tracing (DXR 1.1)
// - Performance: 8 rays/particle (half of direct lighting)

cbuffer IndirectLightingConstants : register(b0)
{
    uint particleCount;
    uint raysPerParticle;      // 8 for balanced quality
    float maxLightingDistance;  // Same as direct (100.0)
    float indirectIntensity;    // Global indirect multiplier (0.3-0.5)
};

struct Particle
{
    float3 position;    // Offset 0-11
    float temperature;  // Offset 12-15
    float3 velocity;    // Offset 16-27
    float density;      // Offset 28-31
};

StructuredBuffer<Particle> g_particles : register(t0);
RaytracingAccelerationStructure g_particleBVH : register(t1);

// INPUT: Direct lighting from previous pass
StructuredBuffer<float4> g_directLighting : register(t2);

// OUTPUT: Indirect lighting contribution
RWStructuredBuffer<float4> g_indirectLighting : register(u0);

// Fibonacci hemisphere sampling for even ray distribution
// Based on golden ratio (φ = 1.618...) for low-discrepancy sampling
// Reference: "Practical Hash-based Owen Scrambling" (JCGT 2020)
float3 FibonacciHemisphere(uint sampleIndex, uint numSamples, float3 normal)
{
    const float PHI = 1.618033988749895; // Golden ratio

    float theta = 2.0 * 3.14159265359 * sampleIndex / PHI;
    float phi = acos(1.0 - 2.0 * (sampleIndex + 0.5) / numSamples);

    float sinPhi = sin(phi);
    float x = cos(theta) * sinPhi;
    float y = sin(theta) * sinPhi;
    float z = cos(phi);

    // For now, assume hemisphere is oriented upwards
    // TODO: Align to particle velocity for anisotropic scattering
    return normalize(float3(x, y, z));
}

// Diffuse BRDF: Lambertian reflectance
// Formula: f_r(wi, wo) = albedo * (N·L) / π
//
// Reference: RTXPT BxDF.hlsli:60-72 (DiffuseReflectionLambert)
// Note: Simplified for volumetric particles (no full tangent space)
float3 DiffuseBRDF(float3 normal, float3 lightDir, float3 albedo)
{
    float NdotL = max(0.0, dot(normal, lightDir));
    return albedo * NdotL / 3.14159265359;
}

// Calculate particle albedo from temperature (color it reflects)
// Physics: Hot particles emit more, reflect less (energy conservation)
// Cool particles emit less, reflect more
float3 ParticleAlbedo(float temperature)
{
    // Normalize temperature to 0-1 range (800K to 26000K)
    float t = saturate((temperature - 800.0) / 25200.0);

    // Inverse relationship: hotter = darker albedo (more emission, less reflection)
    float reflectivity = 1.0 - (t * 0.7); // 30% min reflectivity at max temp

    // Slightly warm tint for reflected light (accretion disk characteristic)
    return float3(reflectivity * 0.9, reflectivity * 0.8, reflectivity * 0.7);
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

    // Simplified normal for volumetric particles
    // TODO: Use velocity direction for anisotropic scattering
    float3 receiverNormal = float3(0, 1, 0);

    // Calculate receiver albedo based on temperature
    float3 receiverAlbedo = ParticleAlbedo(receiver.temperature);

    // Accumulate indirect lighting from hemisphere rays
    float3 accumulatedIndirect = float3(0, 0, 0);

    // Cast indirect rays (HALF the count of direct for performance)
    for (uint rayIdx = 0; rayIdx < raysPerParticle; rayIdx++)
    {
        // Generate ray direction using Fibonacci hemisphere sampling
        float3 rayDir = FibonacciHemisphere(rayIdx, raysPerParticle, receiverNormal);

        // Setup ray descriptor
        RayDesc ray;
        ray.Origin = receiverPos + rayDir * 0.01; // Offset to avoid self-intersection
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = maxLightingDistance;

        // Create RayQuery object (DXR 1.1 inline ray tracing)
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

        // Process all candidates (procedural primitives require manual intersection testing)
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
                // Match visual particle size for accurate lighting
                const float rtLightingRadius = 5.0;  // Matches direct lighting radius
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
                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }
        }

        // After Proceed() loop completes, check if we got a committed hit
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            // Get hit particle index
            uint hitParticleIdx = query.CommittedPrimitiveIndex();

            // Avoid self-illumination (redundant check for safety)
            if (hitParticleIdx == particleIdx)
                continue;

            // KEY DIFFERENCE FROM DIRECT LIGHTING:
            // Read DIRECT lighting from the hit particle (not emission)
            // This is how we get second-bounce indirect illumination
            float3 hitDirectLight = g_directLighting[hitParticleIdx].rgb;

            // Calculate indirect contribution using diffuse BRDF
            float3 lightDir = normalize(g_particles[hitParticleIdx].position - receiverPos);
            float3 brdf = DiffuseBRDF(receiverNormal, lightDir, receiverAlbedo);

            // Distance attenuation (same as direct lighting)
            // Linear falloff: 1 / (1 + distance * k)
            float distance = query.CommittedRayT();
            float attenuation = 1.0 / (1.0 + distance * 0.01);

            // Accumulate: hitDirectLight * BRDF * attenuation
            // Formula: L_indirect = ∫ L_direct(hit) * BRDF * cos(θ) * attenuation dω
            accumulatedIndirect += hitDirectLight * brdf * attenuation;
        }
    }

    // Average lighting over all rays and apply global intensity
    float3 finalIndirect = (accumulatedIndirect / float(raysPerParticle)) * indirectIntensity;

    // Write to output buffer
    // Alpha channel unused (could store hit distance for future denoising)
    g_indirectLighting[particleIdx] = float4(finalIndirect, 0.0);
}
