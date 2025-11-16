/**
 * Probe Grid System - Probe Update Shader
 *
 * Updates lighting at sparse probe grid points by:
 * 1. Casting rays in Fibonacci sphere distribution (64 rays per probe)
 * 2. Using RayQuery to intersect particles (no atomic contention!)
 * 3. Accumulating lighting from all lights
 * 4. Storing simplified spherical harmonics (RGB irradiance for MVP)
 *
 * Architecture: Zero atomic operations = zero contention!
 * Each probe writes to ITS OWN memory location (no conflicts).
 *
 * Performance: ~0.5-1.0ms per frame (amortized over 4 frames)
 */

//=============================================================================
// Constants
//=============================================================================

cbuffer ProbeUpdateConstants : register(b0) {
    float3 g_gridMin;              // Grid world-space minimum
    float g_gridSpacing;           // Distance between probes (62.5 units for 48³)

    uint g_gridSize;               // Grid dimension (48)
    uint g_raysPerProbe;           // Rays to cast per probe (16)
    uint g_particleCount;          // Number of particles
    uint g_lightCount;             // Number of lights

    uint g_frameIndex;             // Frame counter for temporal amortization
    uint g_updateInterval;         // Frames between full grid updates (4)
    float g_probeIntensity;        // Runtime intensity multiplier (200-2000, default 800)
    uint g_padding1;
};

//=============================================================================
// Resources
//=============================================================================

// Probe data structure (matches C++ struct exactly)
struct Probe {
    float3 position;               // 12 bytes - world-space location
    uint lastUpdateFrame;          // 4 bytes - temporal tracking

    // Simplified spherical harmonics (RGB irradiance for MVP)
    // Future: Full SH L2 (9 coefficients × RGB)
    // For now: Direct RGB accumulation
    float3 irradiance[9];          // 108 bytes - SH coefficients (only [0] used for MVP)

    uint padding[1];               // 4 bytes - alignment to 128
};

RWStructuredBuffer<Probe> g_probes : register(u0);

// Particle data (same structure as Gaussian renderer)
struct Particle {
    float3 position;
    float radius;
    float3 velocity;
    float temperature;
    float3 ellipsoidAxis1;
    float padding0;
    float3 ellipsoidAxis2;
    float padding1;
    float3 ellipsoidAxis3;
    float padding2;
};

StructuredBuffer<Particle> g_particles : register(t0);

// Light data (16 lights max, same as Gaussian renderer)
struct Light {
    float3 position;
    float radius;
    float3 color;
    float intensity;
    uint enabled;
    float3 padding;
};

StructuredBuffer<Light> g_lights : register(t1);

// Particle acceleration structure (TLAS)
RaytracingAccelerationStructure g_particleTLAS : register(t2);

//=============================================================================
// Fibonacci Sphere - Evenly Distributed Sphere Sampling
//=============================================================================

/**
 * Generate evenly distributed points on a unit sphere using Fibonacci spiral
 *
 * Based on: "How to generate equidistributed points on the surface of a sphere"
 * (Hannay & Nye, 1999)
 *
 * @param index Ray index [0, totalRays)
 * @param totalRays Total number of rays
 * @return Unit direction vector
 */
float3 FibonacciSphere(uint index, uint totalRays) {
    float goldenRatio = 1.618033988749895; // (1 + sqrt(5)) / 2
    float angleIncrement = 2.0 * 3.14159265359 / goldenRatio;

    // Normalized index [0, 1]
    float i = float(index) + 0.5;
    float phi = acos(1.0 - 2.0 * i / float(totalRays));
    float theta = angleIncrement * float(index);

    // Spherical to Cartesian
    float sinPhi = sin(phi);
    return float3(
        cos(theta) * sinPhi,
        sin(theta) * sinPhi,
        cos(phi)
    );
}

//=============================================================================
// Temperature-Based Blackbody Emission (Same as Gaussian Renderer)
//=============================================================================

float3 BlackbodyColor(float temperature) {
    // Wien's displacement law approximation
    // Cool red (1000K) → Yellow (3000K) → White (6000K) → Hot blue (15000K+)

    float t = clamp(temperature, 800.0, 30000.0);
    float3 color;

    if (t < 3000.0) {
        // Cool red-orange (800-3000K)
        float blend = (t - 800.0) / 2200.0;
        color = lerp(float3(1.0, 0.2, 0.0), float3(1.0, 0.6, 0.3), blend);
    }
    else if (t < 6000.0) {
        // Yellow-orange to white (3000-6000K)
        float blend = (t - 3000.0) / 3000.0;
        color = lerp(float3(1.0, 0.6, 0.3), float3(1.0, 0.95, 0.9), blend);
    }
    else if (t < 15000.0) {
        // White to blue-white (6000-15000K)
        float blend = (t - 6000.0) / 9000.0;
        color = lerp(float3(1.0, 0.95, 0.9), float3(0.7, 0.85, 1.0), blend);
    }
    else {
        // Hot blue (15000K+)
        float blend = clamp((t - 15000.0) / 15000.0, 0.0, 1.0);
        color = lerp(float3(0.7, 0.85, 1.0), float3(0.5, 0.7, 1.0), blend);
    }

    return color;
}

/**
 * Stefan-Boltzmann law: Radiant exitance ∝ T⁴
 * Scaled for visual brightness in [0, 1] range
 */
float BlackbodyIntensity(float temperature) {
    float t = clamp(temperature, 800.0, 30000.0);
    // Normalize to [0, 1] assuming 26000K = max brightness
    return pow(t / 26000.0, 4.0);
}

//=============================================================================
// Ray-Particle Intersection (Sphere Approximation for Probe Lighting)
//=============================================================================

/**
 * Compute lighting contribution from a single particle
 *
 * For probe lighting, we treat particles as spherical emitters
 * (full volumetric ray marching is too expensive for 32K probes)
 *
 * @param probePos Probe world position
 * @param particlePos Particle world position
 * @param particleRadius Particle radius
 * @param temperature Particle temperature
 * @return RGB radiance contribution
 */
float3 ComputeParticleLighting(float3 probePos, float3 particlePos, float radius, float temperature) {
    float3 offset = probePos - particlePos;
    float distance = length(offset);

    // VOLUMETRIC ATTENUATION (NOT inverse-square!)
    // Problem: Inverse-square (r²/d²) is too aggressive for volumetric scattering
    // At 2000 units: (50²/2000²) × 2000 intensity = 1.25 - far too dim!
    //
    // Solution: Hybrid attenuation optimized for volumetric probe grid:
    // - Close range (<200 units): Inverse-square prevents over-brightness
    // - Mid range (200-800 units): Linear falloff maintains visibility
    // - Far range (>800 units): Constant floor for ambient contribution

    float attenuation;
    float normalizedRadius = radius / 50.0; // Normalize to default particle size

    if (distance < 200.0) {
        // Close range: Inverse-square with radius scaling
        attenuation = (radius * radius) / max(distance * distance, 0.01);
    }
    else if (distance < 800.0) {
        // Mid range: Linear falloff
        // At 200 units: matches inverse-square
        // At 800 units: smoothly transitions to far range
        float invSqAtTransition = (radius * radius) / (200.0 * 200.0); // ~0.0625 for r=50
        float linearFalloff = invSqAtTransition * (1.0 - (distance - 200.0) / 600.0);
        attenuation = max(linearFalloff, 0.01);
    }
    else {
        // Far range: Constant ambient floor (prevents complete darkness)
        // This ensures probes at grid edges still receive volumetric scattering
        attenuation = 0.01 * normalizedRadius;
    }

    // Blackbody emission
    float3 color = BlackbodyColor(temperature);
    float intensity = BlackbodyIntensity(temperature);

    // Runtime intensity multiplier (200-2000, default 800)
    // Now much more effective with volumetric attenuation!
    return color * intensity * attenuation * g_probeIntensity;
}

//=============================================================================
// Main Compute Shader
//=============================================================================

[numthreads(8, 8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    // Compute probe grid index from thread ID
    uint3 probeIdx = dispatchThreadID;

    // Bounds check
    if (any(probeIdx >= g_gridSize)) {
        return;
    }

    // Compute linear probe index
    uint probeLinearIdx = probeIdx.x + probeIdx.y * g_gridSize + probeIdx.z * g_gridSize * g_gridSize;

    // Temporal amortization: Only update 1/g_updateInterval probes per frame
    // Frame 0: probes 0, 4, 8, 12, ...
    // Frame 1: probes 1, 5, 9, 13, ...
    // Frame 2: probes 2, 6, 10, 14, ...
    // Frame 3: probes 3, 7, 11, 15, ...
    uint updateSlot = g_frameIndex % g_updateInterval;
    if ((probeLinearIdx % g_updateInterval) != updateSlot) {
        return; // Not this probe's turn to update
    }

    // Calculate probe world position
    float3 probePos = g_gridMin + float3(probeIdx) * g_gridSpacing;

    // Initialize irradiance accumulator (RGB only for MVP)
    float3 totalIrradiance = float3(0, 0, 0);

    // Cast rays in Fibonacci sphere distribution
    for (uint rayIdx = 0; rayIdx < g_raysPerProbe; rayIdx++) {
        float3 direction = FibonacciSphere(rayIdx, g_raysPerProbe);

        // Setup ray for RayQuery
        RayDesc ray;
        ray.Origin = probePos;
        ray.Direction = direction;
        ray.TMin = 0.01;
        ray.TMax = 2000.0; // Max influence distance - increased from 200 to 2000 to cover 3000-unit grid (BUG FIX!)

        // Trace ray using inline ray tracing (no atomic contention!)
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_particleTLAS, RAY_FLAG_NONE, 0xFF, ray);

        // Process all AABB candidates (procedural primitives require manual intersection testing)
        // CRITICAL FIX: Add iteration limit to prevent infinite loops causing TDR timeout
        uint iterationCount = 0;
        const uint MAX_ITERATIONS = 1000;

        while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
            iterationCount++;

            if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint particleIdx = q.CandidatePrimitiveIndex();

                if (particleIdx < g_particleCount) {
                    Particle particle = g_particles[particleIdx];

                    // Ray-ellipsoid intersection test (simplified sphere for probes)
                    float3 oc = ray.Origin - particle.position;
                    float radius = particle.radius;
                    float b = dot(oc, ray.Direction);
                    float c = dot(oc, oc) - radius * radius;
                    float discriminant = b * b - c;

                    if (discriminant >= 0.0) {
                        float t = -b - sqrt(discriminant);
                        if (t > ray.TMin && t < ray.TMax) {
                            // Valid intersection - commit this hit
                            q.CommitProceduralPrimitiveHit(t);
                        }
                    }
                }
            }
        }

        // Diagnostic: If we hit the iteration limit, mark this probe with red (timeout)
        if (iterationCount >= MAX_ITERATIONS) {
            totalIrradiance = float3(10.0, 0.0, 0.0); // Bright red = timeout detected
        }

        // Check for committed hit (procedural primitive, not triangle!)
        if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            // Get hit information
            uint particleIdx = q.CommittedPrimitiveIndex();

            if (particleIdx < g_particleCount) {
                Particle particle = g_particles[particleIdx];

                // Compute particle emission contribution
                float3 particleLight = ComputeParticleLighting(
                    probePos,
                    particle.position,
                    particle.radius,
                    particle.temperature
                );

                totalIrradiance += particleLight;
            }
        }

        // TODO (Phase 2): Add external light source contributions
        // For now, particles are self-emissive only
    }

    // Average irradiance over all rays
    totalIrradiance /= float(g_raysPerProbe);

    // Write to probe buffer (ZERO ATOMIC OPERATIONS - each probe owns its slot!)
    Probe probe;
    probe.position = probePos;
    probe.lastUpdateFrame = g_frameIndex;

    // Store irradiance in SH coefficient [0] (DC term)
    // Future: Project into full SH L2 (9 coefficients)
    probe.irradiance[0] = totalIrradiance;
    for (uint i = 1; i < 9; i++) {
        probe.irradiance[i] = float3(0, 0, 0); // Higher-order terms (future)
    }

    probe.padding[0] = 0;

    g_probes[probeLinearIdx] = probe;
}
