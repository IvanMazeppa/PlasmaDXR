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

#endif // GOD_RAYS_HLSL
