// Advanced particle physics simulation ported from PlasmaVulkan
// Includes: gravity, turbulence, orbital mechanics, constraints

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

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

        // Random initial velocity
        uint seed4 = seed3 * 2654435761u;
        uint seed5 = seed4 * 48271u;
        uint seed6 = seed5 * 1103515245u;

        float3 randomVel = float3(
            (float((seed4 >> 16) & 0x7fff) / 32767.0 - 0.5),
            (float((seed5 >> 16) & 0x7fff) / 32767.0 - 0.5),
            (float((seed6 >> 16) & 0x7fff) / 32767.0 - 0.5)
        );
        p.velocity = randomVel * 40.0;

        float distance = length(p.position - constants.blackHolePosition);
        float tempFactor = 1.0 / max(distance * distance * distance / 1000.0, 1.0);
        p.temperature = 800.0 + 25000.0 * tempFactor;
        p.density = 1.0;
    } else {
        // Physics update for existing particles
        float3 position = p.position;
        float3 velocity = p.velocity;

        // Calculate gravity
        float3 toCenter = constants.blackHolePosition - position;
        float distance = max(length(toCenter), 0.1);

        float gravityMagnitude = constants.gravityStrength / (distance * distance + 1.0);
        float3 gravityForce = normalize(toCenter) * gravityMagnitude;

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

        // Calculate orbital correction (perpendicular to radius)
        float3 tangent = cross(normalize(toCenter), float3(0.0, 1.0, 0.0));
        if (length(tangent) < 0.1) {
            tangent = cross(normalize(toCenter), float3(1.0, 0.0, 0.0));
        }
        tangent = normalize(tangent);

        float currentSpeed = length(velocity);
        float targetOrbitalSpeed = sqrt(constants.gravityStrength / (distance + 1.0));
        float3 orbitalCorrection = tangent * (targetOrbitalSpeed - currentSpeed) * 0.02;
        orbitalCorrection += tangent * constants.angularMomentumBoost * 0.08;

        // Apply forces
        float3 acceleration = gravityForce + orbitalCorrection;
        velocity += acceleration * constants.deltaTime;

        // Update position
        position += velocity * constants.deltaTime;

        // Apply damping
        velocity *= constants.dampingFactor;

        // Keep particles within bounds
        if (distance > constants.outerRadius * 2.0) {
            float3 pushBack = -normalize(position) * 2.0;
            velocity += pushBack * constants.deltaTime;
        }

        // Update temperature based on distance
        float tempFactor = 1.0 / max(distance * distance * distance / 1000.0, 1.0);
        p.temperature = 800.0 + 25000.0 * tempFactor;

        p.position = position;
        p.velocity = velocity;

        // Apply constraints if enabled
        if (constants.constraintShape != 0) {
            ApplyConstraints(p.position, p.velocity);
        }
    }

    particles[particleIndex] = p;
}
