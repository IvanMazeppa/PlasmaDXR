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
    float g_particleRadius;        // Base particle radius for intersection tests
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

// Particle data (matches C++ ParticleSystem::Particle)
// Phase 2: Extended to 64 bytes for lifetime/pyro support
struct Particle {
    // === LEGACY FIELDS (32 bytes) ===
    float3 position;       // 12 bytes (offset 0)
    float temperature;     // 4 bytes  (offset 12)
    float3 velocity;       // 12 bytes (offset 16)
    float density;         // 4 bytes  (offset 28)

    // === MATERIAL FIELDS (16 bytes) ===
    float3 albedo;         // 12 bytes (offset 32)
    uint materialType;     // 4 bytes  (offset 44)

    // === LIFETIME FIELDS (16 bytes) ===
    float lifetime;        // 4 bytes  (offset 48)
    float maxLifetime;     // 4 bytes  (offset 52)
    float spawnTime;       // 4 bytes  (offset 56)
    uint flags;            // 4 bytes  (offset 60)
};  // Total: 64 bytes

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
// Spherical Harmonics L2 Basis Functions
//=============================================================================

/**
 * Evaluate all 9 SH L2 basis functions for a given direction
 *
 * Band 0 (DC): 1 coefficient - uniform ambient
 * Band 1: 3 coefficients - linear directional (main light direction)
 * Band 2: 5 coefficients - quadratic (complex directional features)
 *
 * @param dir Normalized direction vector
 * @param sh Output array of 9 SH basis values
 */
void EvaluateSH_L2(float3 dir, out float sh[9]) {
    // Normalize direction (should already be normalized, but ensure it)
    dir = normalize(dir);

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    // Band 0 (l=0, m=0) - DC term
    sh[0] = 0.282095;  // Y[0,0] = sqrt(1/(4π))

    // Band 1 (l=1) - Linear directional
    sh[1] = 0.488603 * y;  // Y[1,-1] = sqrt(3/(4π)) * y
    sh[2] = 0.488603 * z;  // Y[1,0] = sqrt(3/(4π)) * z
    sh[3] = 0.488603 * x;  // Y[1,1] = sqrt(3/(4π)) * x

    // Band 2 (l=2) - Quadratic
    sh[4] = 1.092548 * x * y;           // Y[2,-2] = sqrt(15/(4π)) * xy
    sh[5] = 1.092548 * y * z;           // Y[2,-1] = sqrt(15/(4π)) * yz
    sh[6] = 0.315392 * (3.0 * z2 - 1.0); // Y[2,0] = sqrt(5/(16π)) * (3z² - 1)
    sh[7] = 1.092548 * x * z;           // Y[2,1] = sqrt(15/(4π)) * xz
    sh[8] = 0.546274 * (x2 - y2);       // Y[2,2] = sqrt(15/(16π)) * (x² - y²)
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
    // Solution: Hybrid attenuation optimized for volumetric probe grid (3000-unit coverage):
    // - Close range (<200 units): Inverse-square prevents over-brightness
    // - Mid range (200-1500 units): Linear falloff maintains visibility
    // - Far range (>1500 units): Constant floor for ambient contribution

    float attenuation;
    float normalizedRadius = radius / 50.0; // Normalize to default particle size

    if (distance < 200.0) {
        // Close range: Inverse-square with radius scaling
        attenuation = (radius * radius) / max(distance * distance, 0.01);
    }
    else if (distance < 1500.0) {
        // Mid range: Linear falloff (EXTENDED to 1500 for 3000-unit grid)
        // At 200 units: matches inverse-square
        // At 1500 units: smoothly transitions to far range
        float invSqAtTransition = (radius * radius) / (200.0 * 200.0); // ~0.0625 for r=50
        float linearFalloff = invSqAtTransition * (1.0 - (distance - 200.0) / 1300.0);
        attenuation = max(linearFalloff, 0.2);  // Floor raised from 0.01 to 0.2
    }
    else {
        // Far range: Constant ambient floor for volumetric GI
        // BOOSTED: 0.01 → 0.5 for visible volumetric ambient at grid edges
        attenuation = 0.5 * normalizedRadius;
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

    // Initialize SH coefficient accumulators (9 coefficients × RGB)
    float3 shCoefficients[9];
    for (uint i = 0; i < 9; i++) {
        shCoefficients[i] = float3(0, 0, 0);
    }

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
                    float radius = g_particleRadius;  // Use constant radius instead of per-particle
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
            shCoefficients[0] += float3(10.0, 0.0, 0.0); // Bright red = timeout detected
        }

        // Evaluate SH basis functions for this ray direction
        float shBasis[9];
        EvaluateSH_L2(direction, shBasis);

        // Accumulator for this ray's total radiance
        float3 rayRadiance = float3(0, 0, 0);

        // Check for committed hit (procedural primitive, not triangle!)
        if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            // Get hit information
            uint particleIdx = q.CommittedPrimitiveIndex();
            float hitDistance = q.CommittedRayT();
            float3 hitPosition = ray.Origin + ray.Direction * hitDistance;

            if (particleIdx < g_particleCount) {
                Particle particle = g_particles[particleIdx];

                // Compute particle emission contribution (if physical emission enabled)
                float3 particleLight = ComputeParticleLighting(
                    probePos,
                    particle.position,
                    g_particleRadius,  // Use constant radius instead of per-particle
                    particle.temperature
                );

                rayRadiance += particleLight;

                // PHASE 2 IMPLEMENTATION: External light source contributions
                // For each enabled light, compute direct illumination at hit point
                for (uint lightIdx = 0; lightIdx < g_lightCount; lightIdx++) {
                    Light light = g_lights[lightIdx];

                    if (light.enabled == 0) continue;

                    // Vector from hit position to light
                    float3 toLight = light.position - hitPosition;
                    float lightDistance = length(toLight);
                    float3 lightDir = toLight / lightDistance;

                    // Simple shadow ray test (binary visibility)
                    RayDesc shadowRay;
                    shadowRay.Origin = hitPosition + lightDir * 0.1; // Offset to avoid self-intersection
                    shadowRay.Direction = lightDir;
                    shadowRay.TMin = 0.01;
                    shadowRay.TMax = lightDistance - 0.1;

                    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
                    shadowQuery.TraceRayInline(g_particleTLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
                    shadowQuery.Proceed();

                    // If shadow ray didn't hit anything, light is visible
                    if (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) {
                        // Inverse-square attenuation for lights
                        float attenuation = 1.0 / max(lightDistance * lightDistance, 1.0);

                        // Light contribution with intensity
                        float3 lightContribution = light.color * light.intensity * attenuation;

                        // Scale by light radius (larger radius = more ambient influence)
                        lightContribution *= (light.radius / 100.0);

                        rayRadiance += lightContribution;
                    }
                }
            }
        }

        // Project this ray's radiance into SH coefficients
        for (uint shIdx = 0; shIdx < 9; shIdx++) {
            shCoefficients[shIdx] += rayRadiance * shBasis[shIdx];
        }
    }

    // Average SH coefficients over all rays (Monte Carlo integration)
    for (uint shIdx = 0; shIdx < 9; shIdx++) {
        shCoefficients[shIdx] /= float(g_raysPerProbe);
    }

    // Write to probe buffer (ZERO ATOMIC OPERATIONS - each probe owns its slot!)
    Probe probe;
    probe.position = probePos;
    probe.lastUpdateFrame = g_frameIndex;

    // Store all 9 SH coefficients (FULL L2 IMPLEMENTATION)
    // Band 0: irradiance[0] - uniform ambient
    // Band 1: irradiance[1-3] - linear directional (enables Henyey-Greenstein scattering!)
    // Band 2: irradiance[4-8] - quadratic directional (complex lighting features)
    for (uint i = 0; i < 9; i++) {
        probe.irradiance[i] = shCoefficients[i];
    }

    probe.padding[0] = 0;

    g_probes[probeLinearIdx] = probe;
}
