// 3D Gaussian Splatting Common Functions
// Shared between AABB generation and ray tracing

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

// Compute Gaussian scale from particle properties (with anisotropy control)
float3 ComputeGaussianScale(Particle p, float baseRadius, bool useAnisotropic, float anisotropyStrength) {
    // Scale based on temperature (hotter = larger)
    float tempScale = 1.0 + (p.temperature - 800.0) / 25200.0; // 1.0-2.0 range

    // Scale based on density (denser = larger)
    float densityScale = sqrt(p.density); // sqrt to keep reasonable

    float radius = baseRadius * tempScale * densityScale;

    if (useAnisotropic) {
        // Ellipsoid: stretch along velocity direction for motion blur effect
        float speedFactor = length(p.velocity) / 100.0; // Normalize velocity
        speedFactor = 1.0 + (speedFactor - 1.0) * anisotropyStrength; // Apply strength
        speedFactor = clamp(speedFactor, 1.0, 3.0 * anisotropyStrength); // Limit stretch

        // Perpendicular radii (circular cross-section)
        float perpRadius = radius;

        // Parallel radius (stretched along velocity)
        float paraRadius = radius * speedFactor;

        return float3(perpRadius, perpRadius, paraRadius);
    } else {
        // Spherical (isotropic) Gaussians
        return float3(radius, radius, radius);
    }
}

// Get Gaussian orientation from velocity
// Returns rotation matrix (no quaternion needed - simpler)
float3x3 ComputeGaussianRotation(float3 velocity) {
    // If velocity too small, return identity
    if (length(velocity) < 0.01) {
        return float3x3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );
    }

    float3 forward = normalize(velocity); // Z-axis

    // Choose perpendicular vector (avoid parallel to forward)
    float3 temp = abs(forward.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(temp, forward)); // X-axis
    float3 up = cross(forward, right); // Y-axis

    // Build rotation matrix (columns are basis vectors)
    return float3x3(
        right.x, up.x, forward.x,
        right.y, up.y, forward.y,
        right.z, up.z, forward.z
    );
}

// Conservative AABB for Gaussian (3 standard deviations)
struct AABB {
    float3 minPoint;
    float3 maxPoint;
};

AABB ComputeGaussianAABB(Particle p, float baseRadius, bool useAnisotropic, float anisotropyStrength) {
    float3 scale = ComputeGaussianScale(p, baseRadius, useAnisotropic, anisotropyStrength);

    // Conservative bound: max of all axes (axis-aligned)
    float maxRadius = max(scale.x, max(scale.y, scale.z)) * 3.0; // 3 std devs

    AABB result;
    result.minPoint = p.position - maxRadius;
    result.maxPoint = p.position + maxRadius;
    return result;
}

// Ray-Gaussian intersection using analytic method
// Returns t values for entry and exit (or -1 if no hit)
float2 RayGaussianIntersection(
    float3 rayOrigin,
    float3 rayDir,
    float3 gaussianPos,
    float3 scale,
    float3x3 rotation)
{
    // Transform ray to Gaussian's local space (unit sphere)
    float3 localOrigin = mul(rayOrigin - gaussianPos, rotation);
    float3 localDir = mul(rayDir, rotation);

    // Scale by inverse of Gaussian scale (transforms ellipsoid to unit sphere)
    localOrigin /= scale;
    localDir /= scale;

    // Now intersect ray with unit sphere
    float a = dot(localDir, localDir);
    float b = 2.0 * dot(localOrigin, localDir);
    float c = dot(localOrigin, localOrigin) - 1.0;

    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) return float2(-1, -1);

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-b - sqrtDisc) / (2.0 * a);
    float t2 = (-b + sqrtDisc) / (2.0 * a);

    return float2(t1, t2);
}

// Evaluate Gaussian density at a point
float EvaluateGaussianDensity(
    in float3 worldPos,
    in float3 gaussianPos,
    in float3 scale,
    in float3x3 rotation,
    in float opacity)
{
    // Transform to local space
    float3 diff = worldPos - gaussianPos;
    float3 localDiff = mul(diff, rotation);

    // Scale
    localDiff /= scale;

    // Gaussian function: exp(-0.5 * |x|²)
    float dist2 = dot(localDiff, localDiff);
    float gaussian = exp(-0.5 * dist2);

    return gaussian * opacity;
}

// Henyey-Greenstein phase function for scattering
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(abs(denom), 1.5));
}

// Temperature to emission color (same as existing)
float3 TemperatureToEmission(float temperature) {
    float t = saturate((temperature - 800.0) / 25200.0);

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

// Emission intensity from temperature (Stefan-Boltzmann-inspired)
float EmissionIntensity(float temperature) {
    float normalized = temperature / 26000.0;
    return pow(normalized, 2.0);
}