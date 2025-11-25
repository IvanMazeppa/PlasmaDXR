// 3D Gaussian Splatting Common Functions
// Shared between AABB generation and ray tracing
// Phase 2: Extended with lifetime system for pyro/explosion effects

// Extended particle structure (64 bytes, 16-byte aligned)
// Phase 2: Pyro/Explosion support with lifetime fields
struct Particle {
    // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
    float3 position;       // 12 bytes (offset 0)
    float temperature;     // 4 bytes  (offset 12)
    float3 velocity;       // 12 bytes (offset 16)
    float density;         // 4 bytes  (offset 28)

    // === MATERIAL FIELDS (16 bytes) ===
    float3 albedo;         // 12 bytes (offset 32) - Surface/volume color
    uint materialType;     // 4 bytes  (offset 44) - 0=PLASMA...7=SHOCKWAVE

    // === LIFETIME FIELDS (16 bytes) - Phase 2 Pyro/Explosions ===
    float lifetime;        // 4 bytes  (offset 48) - Current age in seconds
    float maxLifetime;     // 4 bytes  (offset 52) - Total duration (0 = infinite)
    float spawnTime;       // 4 bytes  (offset 56) - Time when spawned
    uint flags;            // 4 bytes  (offset 60) - Particle flags bitmask
};  // Total: 64 bytes (16-byte aligned ✓)

// Particle flags
#define FLAG_NONE           0
#define FLAG_EXPLOSION      (1 << 0)
#define FLAG_FADING         (1 << 1)
#define FLAG_IMMORTAL       (1 << 2)
#define FLAG_EMISSIVE_ONLY  (1 << 3)

// Compute Gaussian scale from particle properties (with anisotropy control)
// Phase 1.5: Added adaptive radius parameters
float3 ComputeGaussianScale(
    Particle p,
    float baseRadius,
    bool useAnisotropic,
    float anisotropyStrength,
    bool enableAdaptive,
    float innerZone,
    float outerZone,
    float innerScale,
    float outerScale,
    float densityMin,
    float densityMax
) {
    // DEFENSIVE: Validate particle data to prevent NaN/Inf propagation
    // Invalid density or position can cause TDR crashes in BLAS building
    float density = p.density;
    if (isnan(density) || isinf(density) || density <= 1e-6) {
        density = 1.0;  // Neutral density fallback
    }

    // Validate position (check for NaN/Inf in any component)
    float3 position = p.position;
    if (any(isnan(position)) || any(isinf(position))) {
        position = float3(0, 0, 0);  // Origin fallback
    }

    // Scale based on temperature (hotter = larger)
    float tempScale = 1.0 + (p.temperature - 800.0) / 25200.0; // 1.0-2.0 range
    tempScale = clamp(tempScale, 0.5, 3.0);  // Safety clamp

    // FIXED: Inverse density scaling (denser = SMALLER to reduce overlap)
    // Dense inner particles shrink, sparse outer particles grow
    float densityScale = 1.0 / sqrt(density);
    densityScale = clamp(densityScale, densityMin, densityMax); // User-configurable bounds

    // Distance-based expansion for outer sparse regions
    float distFromCenter = length(position);
    if (isnan(distFromCenter) || isinf(distFromCenter)) {
        distFromCenter = 100.0;  // Neutral distance fallback
    }
    float distanceScale = 1.0;

    if (enableAdaptive) {
        // Inner region: shrink to reduce overlap
        if (distFromCenter < innerZone) {
            distanceScale = lerp(innerScale, 1.0, distFromCenter / innerZone);
        }
        // Outer region: expand to improve visibility
        else if (distFromCenter > outerZone) {
            float outerBlend = saturate((distFromCenter - outerZone) / 500.0);
            distanceScale = lerp(1.0, outerScale, outerBlend);
        }
    }

    float radius = baseRadius * tempScale * densityScale * distanceScale;

    // FINAL VALIDATION: Ensure radius is finite and positive
    // This prevents NaN/Inf from reaching AABB bounds (which causes TDR crashes)
    if (isnan(radius) || isinf(radius) || radius <= 0.0) {
        radius = baseRadius;  // Fallback to unscaled radius
    }
    // Safety clamp to prevent extreme values
    radius = clamp(radius, baseRadius * 0.1, baseRadius * 10.0);

    if (useAnisotropic) {
        // Ellipsoid: stretch along velocity direction for motion blur effect
        // Particle velocities range 0-20 units/sec, normalize to 0-1 range
        // CRITICAL FIX: Previous formula produced values < 1.0 which got clamped, preventing any stretch
        // Correct formula: 1.0 (no stretch) to 3.0 (max stretch) based on velocity
        float normalizedSpeed = length(p.velocity) / 20.0; // 0-1 range
        float speedFactor = 1.0 + normalizedSpeed * 2.0 * anisotropyStrength; // 1.0 to 3.0 range
        speedFactor = clamp(speedFactor, 1.0, 3.0); // Defensive clamp

        // Perpendicular radii (circular cross-section)
        float perpRadius = radius;

        // Parallel radius (stretched along velocity)
        float paraRadius = radius * speedFactor;

        // CRITICAL FIX: Clamp AFTER anisotropic stretching to prevent BLAS explosion
        // Without this, paraRadius can reach 360+ units with small baseRadius + high adaptive scaling
        // This causes VRAM overflow during BLAS build → TDR crash
        float maxAllowedRadius = 100.0; // Maximum radius for any axis (conservative)
        perpRadius = min(perpRadius, maxAllowedRadius);
        paraRadius = min(paraRadius, maxAllowedRadius);

        return float3(perpRadius, perpRadius, paraRadius);
    } else {
        // Spherical (isotropic) Gaussians
        float clampedRadius = min(radius, 100.0); // Apply same cap for consistency
        return float3(clampedRadius, clampedRadius, clampedRadius);
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

    // Build rotation matrix (COLUMN-MAJOR - basis vectors are COLUMNS)
    // CORRECTIVE FIX: REVERTED to original form - the row-major version was WRONG
    // Column-major is CORRECT for HLSL's mul() semantics with vector*matrix multiplication
    // This properly transforms rays FROM world space TO Gaussian local space
    return float3x3(
        right.x, up.x, forward.x,   // First column (x-axis in world space)
        right.y, up.y, forward.y,   // Second column (y-axis in world space)
        right.z, up.z, forward.z    // Third column (z-axis in world space)
    );
}

// Conservative AABB for Gaussian (3 standard deviations)
struct AABB {
    float3 minPoint;
    float3 maxPoint;
};

AABB ComputeGaussianAABB(
    Particle p,
    float baseRadius,
    bool useAnisotropic,
    float anisotropyStrength,
    bool enableAdaptive,
    float innerZone,
    float outerZone,
    float innerScale,
    float outerScale,
    float densityMin,
    float densityMax
) {
    float3 scale = ComputeGaussianScale(
        p, baseRadius, useAnisotropic, anisotropyStrength,
        enableAdaptive, innerZone, outerZone, innerScale, outerScale, densityMin, densityMax
    );

    // Conservative bound: max of all axes (axis-aligned)
    // CRITICAL FIX: 4σ padding for anisotropic Gaussians (3σ insufficient for ellipsoids)
    // 3σ = 99.7% for spherical, but anisotropic can stretch 3× → need 4σ = 99.99% coverage
    float maxRadius = max(scale.x, max(scale.y, scale.z)) * 4.0; // 4 std devs

    AABB result;
    result.minPoint = p.position - maxRadius;
    result.maxPoint = p.position + maxRadius;
    return result;
}

// Ray-Gaussian intersection using analytic method
// Returns t values for entry and exit (or -1 if no hit)
// CRITICAL FIX: Uses Kahan's numerically stable quadratic formula to prevent
// catastrophic cancellation at large radii (>150.0) which caused cube artifacts
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

    // Now intersect ray with unit sphere: a*t² + b*t + c = 0
    float a = dot(localDir, localDir);
    float b = 2.0 * dot(localOrigin, localDir);
    float c = dot(localOrigin, localOrigin) - 1.0;

    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) return float2(-1, -1);

    // KAHAN'S NUMERICALLY STABLE QUADRATIC FORMULA
    // Prevents catastrophic cancellation when |b| >> sqrt(discriminant)
    // Standard formula: t = (-b ± sqrt(discriminant)) / (2a) UNSTABLE!
    // Kahan's formula: Compute one root stably, derive second via Vieta's relation
    float sqrtDisc = sqrt(discriminant);

    // Compute the more numerically stable root first
    // If b > 0: use (-b - sqrt(disc)) to avoid subtraction of similar magnitudes
    // If b < 0: use (-b + sqrt(disc)) for same reason
    float q = (b > 0.0)
        ? -0.5 * (b + sqrtDisc)  // b > 0: avoid cancellation in (-b - sqrt)
        : -0.5 * (b - sqrtDisc); // b < 0: avoid cancellation in (-b + sqrt)

    // Two roots via Vieta's formulas: t1*t2 = c/a, t1+t2 = -b/a
    float t1 = q / a;        // First root (stable)
    float t2 = c / q;        // Second root via Vieta (c/a = t1*t2 => t2 = c/(a*t1) = c/q)

    // Ensure t1 <= t2 (near hit first)
    if (t1 > t2) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }

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

// =============================================================================
// PCSS (Percentage-Closer Soft Shadows) Helper Functions
// =============================================================================

// Poisson disk samples for PCSS shadow sampling (16 samples)
static const float2 PoissonDisk16[16] = {
    float2(-0.94201624, -0.39906216),
    float2(0.94558609, -0.76890725),
    float2(-0.094184101, -0.92938870),
    float2(0.34495938, 0.29387760),
    float2(-0.91588581, 0.45771432),
    float2(-0.81544232, -0.87912464),
    float2(-0.38277543, 0.27676845),
    float2(0.97484398, 0.75648379),
    float2(0.44323325, -0.97511554),
    float2(0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023),
    float2(0.79197514, 0.19090188),
    float2(-0.24188840, 0.99706507),
    float2(-0.81409955, 0.91437590),
    float2(0.19984126, 0.78641367),
    float2(0.14383161, -0.14100790)
};

// Simple hash function for random rotation per pixel
float Hash12(float2 p) {
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

// Rotate 2D vector by angle
float2 Rotate2D(float2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(c * v.x - s * v.y, s * v.x + c * v.y);
}