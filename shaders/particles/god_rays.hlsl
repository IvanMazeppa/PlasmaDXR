#ifndef GOD_RAYS_HLSL
#define GOD_RAYS_HLSL

// God Ray System - Volumetric Light Shafts (Phase 5 Milestone 5.3c)
//
// Implements volumetric scattering from ambient medium, creating dramatic light beams
// that are static in world space while particles pass through them.

// Rotate vector around Y-axis (vertical)
// Used for rotating god ray beam direction (searchlight effect)
float3 RotateVectorY(float3 v, float angleRadians) {
    float c = cos(angleRadians);
    float s = sin(angleRadians);
    return float3(
        v.x * c - v.z * s,
        v.y,
        v.x * s + v.z * c
    );
}

// Calculate god ray contribution at a point in space
// Returns RGB color contribution from volumetric scattering in ambient medium
//
// Parameters:
//   rayPos            - Current position along camera ray (world space)
//   light             - Light structure with base properties + god ray parameters
//   totalTime         - Total elapsed time (for rotation animation)
//   godRayDensity     - Global god ray density multiplier (0.0-1.0)
//   accelStructure    - TLAS for shadow ray traversal (particle occlusion)
//
// Returns:
//   float3 - RGB color contribution from god ray scattering
float3 CalculateGodRayContribution(
    float3 rayPos,
    Light light,
    float totalTime,
    float godRayDensity,
    RaytracingAccelerationStructure accelStructure
) {
    // Early exit if god rays disabled
    if (light.enableGodRays < 0.5) {
        return float3(0, 0, 0);
    }

    // === Step 1: Calculate position relative to light ===
    float3 toLight = light.position - rayPos;
    float distToLight = length(toLight);

    // Early exit if too close or too far
    if (distToLight < 0.001 || distToLight > light.godRayLength) {
        return float3(0, 0, 0);
    }

    float3 lightDir = toLight / distToLight;

    // === Step 2: Get beam direction (with optional rotation) ===
    float3 beamDir = light.godRayDirection;
    if (abs(light.godRayRotationSpeed) > 0.001) {
        float rotationAngle = light.godRayRotationSpeed * totalTime;
        beamDir = RotateVectorY(beamDir, rotationAngle);
    }

    // === Step 3: Check if inside cone volume ===
    float alignment = dot(lightDir, beamDir);
    float coneThreshold = cos(light.godRayConeAngle);

    if (alignment < coneThreshold) {
        return float3(0, 0, 0);  // Outside cone
    }

    // === Step 4: Calculate radial distance from beam axis ===
    // Distance from ray position to beam centerline
    float axisDistance = distToLight * sqrt(max(0.0, 1.0 - alignment * alignment));

    // === Step 5: Apply radial falloff (Gaussian) ===
    // Exponential falloff from beam center, controlled by godRayFalloff
    // Higher falloff = sharper beam edges, lower = softer/wider beam
    float radialFalloff = exp(-axisDistance * light.godRayFalloff);

    // === Step 6: Apply distance falloff (inverse square law) ===
    // Attenuate intensity based on distance from light
    float distanceFalloff = 1.0 / (1.0 + distToLight * distToLight * 0.0001);

    // === Step 7: Cast shadow ray (particles occlude god rays) ===
    // This creates shadows on the beam itself, showing particle occlusion
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

    RayDesc shadowRay;
    shadowRay.Origin = rayPos + lightDir * 0.1;  // Slight offset to avoid self-intersection
    shadowRay.Direction = lightDir;
    shadowRay.TMin = 0.0;
    shadowRay.TMax = distToLight - 0.1;

    q.TraceRayInline(accelStructure, RAY_FLAG_NONE, 0xFF, shadowRay);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return float3(0, 0, 0);  // Occluded by particle (shadow on beam)
    }

    // === Step 8: Calculate final intensity ===
    // Combine all attenuation factors
    float intensity = light.godRayIntensity * radialFalloff * distanceFalloff * godRayDensity;

    // === Step 9: Return god ray contribution ===
    // Multiply light color by final intensity
    return light.color * intensity;
}

// ============================================================================
// ATMOSPHERIC FOG RAY MARCHING - Volumetric God Rays
// ============================================================================
//
// This is the CORRECT implementation of god rays!
// Marches through UNIFORM ATMOSPHERIC FOG at regular intervals,
// independent of particle positions. This creates visible light shafts
// even in empty space, just like real fog/dust scattering sunlight.
//
// Algorithm:
//   1. Divide camera ray into N steps (32 recommended for quality/performance balance)
//   2. At each step, calculate god ray contribution from all enabled lights
//   3. Cast shadow rays to check if particles occlude the light
//   4. Accumulate scattering contributions
//   5. Return total atmospheric fog color
//
// Parameters:
//   cameraPos         - Camera position in world space
//   rayDir            - Ray direction (normalized)
//   maxDistance       - Maximum ray march distance (stop early for performance)
//   lights            - Array of lights (structured buffer)
//   lightCount        - Number of active lights
//   totalTime         - Total elapsed time (for rotation animation)
//   godRayDensity     - Global fog density (0.0-1.0, 0=no fog)
//   accelStructure    - TLAS for shadow ray occlusion testing
//
// Returns:
//   float3 - RGB color contribution from atmospheric fog scattering
//
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

    // === Configuration ===
    const uint NUM_STEPS = 32;  // 32 steps = good quality/performance balance
    const float stepSize = maxDistance / float(NUM_STEPS);

    float3 totalFogColor = float3(0, 0, 0);

    // === Ray March Loop ===
    for (uint step = 0; step < NUM_STEPS; step++) {
        // Current position along ray (sample at step center for better accuracy)
        float t = (float(step) + 0.5) * stepSize;
        float3 samplePos = cameraPos + rayDir * t;

        // === Sample all lights at this fog position ===
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

            // === Beam Direction (with optional rotation) ===
            float3 beamDir = light.godRayDirection;
            if (abs(light.godRayRotationSpeed) > 0.001) {
                float rotationAngle = light.godRayRotationSpeed * totalTime;
                beamDir = RotateVectorY(beamDir, rotationAngle);
            }

            // === Cone Volume Test ===
            float alignment = dot(lightDir, beamDir);
            float coneThreshold = cos(light.godRayConeAngle);

            if (alignment < coneThreshold) {
                continue;  // Outside cone, skip this light
            }

            // === Radial Falloff (Gaussian beam shape) ===
            float axisDistance = distToLight * sqrt(max(0.0, 1.0 - alignment * alignment));
            float radialFalloff = exp(-axisDistance * light.godRayFalloff);

            // === Distance Attenuation ===
            float distanceFalloff = 1.0 / (1.0 + distToLight * distToLight * 0.0001);

            // === Shadow Ray (particles occlude fog) ===
            RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

            RayDesc shadowRay;
            shadowRay.Origin = samplePos + lightDir * 0.1;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = distToLight - 0.1;

            q.TraceRayInline(accelStructure, RAY_FLAG_NONE, 0xFF, shadowRay);
            q.Proceed();

            // Skip if occluded by particle
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                continue;
            }

            // === Calculate Scattering Contribution ===
            float scatteringStrength = light.godRayIntensity * radialFalloff * distanceFalloff * godRayDensity;
            float3 scatteringColor = light.color * scatteringStrength;

            // Accumulate fog color (volumetric integral)
            totalFogColor += scatteringColor * stepSize;
        }
    }

    return totalFogColor;
}

#endif // GOD_RAYS_HLSL
