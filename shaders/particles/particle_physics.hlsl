// Advanced particle physics simulation ported from PlasmaVulkan
// Includes: gravity, turbulence, orbital mechanics, constraints
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

struct ParticleConstants {
    float deltaTime;
    float totalTime;
    float blackHoleMass;
    float gravityStrength;
    float3 blackHolePosition;
    float turbulenceStrength;
    float3 diskAxis;
    float dampingFactor;
    float innerRadius;
    float outerRadius;
    float diskThickness;
    float viscosity;
    float angularMomentumBoost;
    uint constraintShape;  // 0=NONE, 1=SPHERE, 2=DISC, 3=TORUS, 4=ACCRETION_DISK
    float constraintRadius;
    float constraintThickness;
    float particleCount;
};

RWStructuredBuffer<Particle> particles : register(u0);
ConstantBuffer<ParticleConstants> constants : register(b0);

// Apply shape constraints to particles
void ApplyConstraints(inout float3 position, inout float3 velocity) {
    if (constants.constraintShape == 1) { // SPHERE
        float distance = length(position);
        if (distance > constants.constraintRadius && distance > 0.001) {
            float3 normal = position / distance;
            position = normal * constants.constraintRadius * 0.99;
            velocity = reflect(velocity, -normal) * 0.8;
        }
    }
    else if (constants.constraintShape == 2) { // DISC
        float radiusXZ = length(position.xz);
        float thickness = constants.constraintThickness * 0.5;

        if (radiusXZ > constants.constraintRadius && radiusXZ > 0.001) {
            float2 normalXZ = position.xz / radiusXZ;
            position.xz = normalXZ * constants.constraintRadius * 0.99;
            velocity.xz = reflect(velocity.xz, -normalXZ) * 0.8;
        }

        if (abs(position.y) > thickness) {
            float normalY = sign(position.y);
            position.y = normalY * thickness * 0.99;
            velocity.y = -velocity.y * 0.6;
        }
    }
    else if (constants.constraintShape == 3) { // TORUS
        float radiusXZ = max(length(position.xz), 0.001);
        float majorRadius = constants.constraintRadius;
        float minorRadius = constants.constraintThickness;

        float2 dirXZ = position.xz / radiusXZ;
        float3 torusCenter = float3(dirXZ * majorRadius, 0.0);
        float3 toCenter = position - torusCenter;
        float distToTorus = length(toCenter);

        if (distToTorus > minorRadius && distToTorus > 0.001) {
            float3 normal = toCenter / distToTorus;
            position = torusCenter + normal * minorRadius * 0.99;
            velocity = reflect(velocity, -normal) * 0.7;
        }

        if (radiusXZ < majorRadius - minorRadius) {
            position.xz = dirXZ * (majorRadius - minorRadius) * 0.99;
            velocity.xz = reflect(velocity.xz, -dirXZ) * 0.8;
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint particleIndex = id.x;
    if (particleIndex >= uint(constants.particleCount)) return;

    Particle p = particles[particleIndex];

    // Initialize particles if this is the first frame
    if (constants.totalTime < 0.01) {
        // Initialize with randomization
        uint seed = particleIndex * 1103515245u + 12345u;
        uint seed2 = seed * 1664525u + 1013904223u;
        uint seed3 = seed2 * 22695477u + 1;

        float angleRand = float((seed2 >> 16) & 0x7fff) / 32767.0;
        float angle = angleRand * 6.28318530718;

        float radiusRand = float((seed3 >> 16) & 0x7fff) / 32767.0;
        float radius = lerp(constants.innerRadius, constants.outerRadius, pow(radiusRand, 3.0));

        float randZ = (float((seed >> 16) & 0x7fff) / 32767.0 - 0.5) * constants.diskThickness;

        p.position = float3(
            cos(angle) * radius,
            randZ,
            sin(angle) * radius
        );

        // Initialize with Keplerian orbital velocity
        // v = sqrt(GM/r) in tangent direction
        float3 toCenter = constants.blackHolePosition - p.position;
        float distance = max(length(toCenter), 0.1);
        float3 toCenterNorm = toCenter / distance;

        // Calculate orbital direction (perpendicular to both radius and disk axis)
        float3 orbitalDir = cross(toCenterNorm, constants.diskAxis);
        float orbitalDirLen = length(orbitalDir);
        if (orbitalDirLen > 0.01) {
            orbitalDir /= orbitalDirLen;
        } else {
            orbitalDir = float3(1, 0, 0);  // Fallback
        }

        // Keplerian speed: v = sqrt(GM/r) - NOW USES BLACK HOLE MASS!
        // Use gravityStrength as a scaling factor (NOT multiplied by mass - that causes extreme speeds!)
        float keplerianSpeed = sqrt(constants.blackHoleMass / (distance + 0.1)) * 0.01;
        uint seed4 = seed3 * 2654435761u;
        float speedVariation = (float((seed4 >> 16) & 0x7fff) / 32767.0 - 0.5) * 0.2;  // ±10%
        float initialSpeed = keplerianSpeed * (1.0 + speedVariation);

        p.velocity = orbitalDir * initialSpeed;

        // Add small random perturbation for variety
        uint seed5 = seed4 * 48271u;
        uint seed6 = seed5 * 1103515245u;
        float3 randomPerturbation = float3(
            (float((seed5 >> 16) & 0x7fff) / 32767.0 - 0.5),
            (float((seed6 >> 16) & 0x7fff) / 32767.0 - 0.5),
            (float((seed4 >> 8) & 0x7fff) / 32767.0 - 0.5)
        ) * 5.0;
        p.velocity += randomPerturbation;

        // Temperature falls off with distance (hotter near center, distance already calculated above)
        // Use dynamic radius range from constants (not hardcoded)
        float tempFactor = saturate(1.0 - (distance - constants.innerRadius) /
                                    (constants.outerRadius - constants.innerRadius));  // 0=outer, 1=inner
        p.temperature = 800.0 + 25200.0 * pow(tempFactor, 2.0);  // 800K-26000K range

        // Density varies with distance - denser near black hole (accretion disk physics)
        // Exponential falloff creates realistic density gradient
        p.density = 0.2 + 2.8 * pow(tempFactor, 1.5);  // 0.2-3.0 range (denser near center)

        // Sprint 1 MVP: Initialize new material system fields
        // All particles start as PLASMA (backward compatibility)
        // Albedo: Warm orange plasma color (will be overridden by material constant buffer in Phase 2)
        p.albedo = float3(1.0, 0.4, 0.1);  // Hot plasma orange/red
        p.materialType = 0;  // PLASMA (legacy default)

        // Phase 2: Initialize lifetime fields
        // Default accretion disk particles are immortal (maxLifetime = 0)
        p.lifetime = 0.0;
        p.maxLifetime = 0.0;  // 0 = infinite lifetime (immortal)
        p.spawnTime = constants.totalTime;
        p.flags = FLAG_IMMORTAL;  // Accretion disk particles never die
    } else {
        // Physics update for existing particles
        float3 position = p.position;
        float3 velocity = p.velocity;

        // Calculate gravity
        float3 toCenter = constants.blackHolePosition - position;
        float distance = max(length(toCenter), 0.1);

        // Safe normalize: avoid NaN when toCenter is zero
        float3 toCenterNorm = distance > 0.01 ? toCenter / distance : float3(0, 1, 0);

        float gravityMagnitude = constants.gravityStrength / (distance * distance + 1.0);
        float3 gravityForce = toCenterNorm * gravityMagnitude;

        // CURL NOISE TURBULENCE - creates vortices
        float3 curlPos = position * 0.08 + float3(constants.totalTime * 0.03, 0, 0);
        float epsilon = 0.05;

        // Sample potential field at 6 points
        float px1 = sin(curlPos.x + epsilon) * cos(curlPos.y * 1.7) * sin(curlPos.z * 2.3);
        float px2 = sin(curlPos.x - epsilon) * cos(curlPos.y * 1.7) * sin(curlPos.z * 2.3);
        float py1 = sin(curlPos.x) * cos((curlPos.y + epsilon) * 1.7) * sin(curlPos.z * 2.3);
        float py2 = sin(curlPos.x) * cos((curlPos.y - epsilon) * 1.7) * sin(curlPos.z * 2.3);
        float pz1 = sin(curlPos.x) * cos(curlPos.y * 1.7) * sin((curlPos.z + epsilon) * 2.3);
        float pz2 = sin(curlPos.x) * cos(curlPos.y * 1.7) * sin((curlPos.z - epsilon) * 2.3);

        // Calculate curl (rotating flow field)
        float3 curl;
        curl.x = (pz1 - pz2) / (2.0 * epsilon) - (py1 - py2) / (2.0 * epsilon);
        curl.y = (px1 - px2) / (2.0 * epsilon) - (pz1 - pz2) / (2.0 * epsilon);
        curl.z = (py1 - py2) / (2.0 * epsilon) - (px1 - px2) / (2.0 * epsilon);

        // Add smaller scale eddies
        float3 curlPos2 = position * 0.25 + float3(constants.totalTime * 0.08, 0, 0);
        curl += float3(
            sin(curlPos2.y * 5.1) * cos(curlPos2.z * 4.3),
            sin(curlPos2.z * 5.1) * cos(curlPos2.x * 4.3),
            sin(curlPos2.x * 5.1) * cos(curlPos2.y * 4.3)
        ) * 0.2;

        // Apply turbulence to velocity
        velocity += curl * constants.turbulenceStrength * constants.deltaTime;

        // Add per-particle random noise that varies over time
        float randomPhase = float(particleIndex) * 0.1 + constants.totalTime * 0.5;
        float3 randomNoise = float3(
            sin(randomPhase * 1.7),
            sin(randomPhase * 2.3),
            sin(randomPhase * 3.1)
        ) * 8.0;
        velocity += randomNoise * constants.deltaTime;

        // KEPLERIAN ORBITAL MECHANICS
        // For stable circular orbit: v = sqrt(GM/r)
        // Calculate orbital velocity perpendicular to radius
        float3 tangent = cross(toCenterNorm, constants.diskAxis);
        float tangentLen = length(tangent);
        if (tangentLen < 0.01) {
            // Fallback if particle is along disk axis
            tangent = cross(toCenterNorm, float3(1.0, 0.0, 0.0));
            tangentLen = length(tangent);
        }
        tangent = tangentLen > 0.01 ? tangent / tangentLen : float3(1, 0, 0);

        // Calculate Keplerian orbital speed for this radius: v = sqrt(GM/r)
        // Use 0.01 scaling factor to keep velocities reasonable with large mass values
        float keplerianSpeed = sqrt(constants.blackHoleMass / (distance + 0.1)) * 0.01 * constants.angularMomentumBoost;

        // Project current velocity onto orbital direction
        float currentOrbitalSpeed = dot(velocity, tangent);

        // Calculate correction needed to achieve Keplerian orbit
        float speedDifference = keplerianSpeed - currentOrbitalSpeed;
        float3 orbitalCorrection = tangent * speedDifference * 2.5;  // Strong correction for stable orbits

        // Apply forces
        float3 acceleration = gravityForce + orbitalCorrection;
        velocity += acceleration * constants.deltaTime;

        // NEW: Alpha viscosity - Shakura-Sunyaev accretion (inward spiral)
        // Creates gradual radial drift toward black hole
        if (constants.viscosity > 0.001) {
            // Radial drift: particles slowly spiral inward while maintaining angular momentum
            float3 radialDrift = -toCenterNorm * constants.viscosity * 0.01;
            velocity += radialDrift * constants.deltaTime;
        }

        // Update position
        position += velocity * constants.deltaTime;

        // Apply damping
        velocity *= constants.dampingFactor;

        // Keep particles within bounds
        if (distance > constants.outerRadius * 2.0) {
            float3 pushBack = -normalize(position) * 2.0;
            velocity += pushBack * constants.deltaTime;
        }

        // Update temperature based on distance (hotter near center)
        // Use dynamic radius range from constants (not hardcoded)
        float tempFactor = saturate(1.0 - (distance - constants.innerRadius) /
                                    (constants.outerRadius - constants.innerRadius));  // 0=outer, 1=inner
        float targetTemp = 800.0 + 25200.0 * pow(tempFactor, 2.0);  // 800K-26000K range

        // Apply exponential smoothing to prevent abrupt color changes (flashing/blinking)
        // 0.90 = 90% previous temperature, 10% new temperature = smooth transition over ~10 frames
        p.temperature = lerp(targetTemp, p.temperature, 0.90);

        // Update density to match temperature/distance
        p.density = 0.2 + 2.8 * pow(tempFactor, 1.5);  // 0.2-3.0 range (denser near center)

        p.position = position;
        p.velocity = velocity;

        // Apply constraints if enabled
        if (constants.constraintShape != 0) {
            ApplyConstraints(p.position, p.velocity);
        }

        // ============================================================================
        // Phase 2: LIFETIME SYSTEM
        // ============================================================================
        // Update lifetime for non-immortal particles
        if ((p.flags & FLAG_IMMORTAL) == 0) {
            p.lifetime += constants.deltaTime;

            // Check if particle has expired
            if (p.maxLifetime > 0.0 && p.lifetime >= p.maxLifetime) {
                // Particle died - move it far away and zero velocity
                // (Will be recycled by explosion spawner or respawn system)
                p.position = float3(99999.0, 99999.0, 99999.0);
                p.velocity = float3(0.0, 0.0, 0.0);
                p.flags |= FLAG_FADING;  // Mark as dead/fading
            }
        }
    }

    particles[particleIndex] = p;
}
