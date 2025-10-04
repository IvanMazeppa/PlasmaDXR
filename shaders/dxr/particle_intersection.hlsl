// DXR Particle Intersection Shader
// Procedural sphere-ray intersection for AABB-based particles
// This shader runs when a ray hits a particle AABB and performs analytic intersection

// Input: Particle buffer (read particle positions and radii)
struct Particle
{
    float3 position;    // Offset 0-11
    float temperature;  // Offset 12-15
    float3 velocity;    // Offset 16-27
    float density;      // Offset 28-31
};

StructuredBuffer<Particle> g_particles : register(t0);

cbuffer IntersectionConstants : register(b0)
{
    float particleRadius; // Fixed radius for all particles
    float3 padding;
};

// Particle hit attributes (empty for now, can store hit normal later)
struct ParticleAttributes
{
    float2 unused;
};

[shader("intersection")]
void ParticleIntersection()
{
    // Get particle index from primitive index
    uint particleIdx = PrimitiveIndex();
    Particle p = g_particles[particleIdx];

    // Get ray in object space (AABB local coordinates)
    float3 rayOrigin = ObjectRayOrigin();
    float3 rayDir = ObjectRayDirection();

    // Analytical sphere-ray intersection
    // Ray: P(t) = O + tD
    // Sphere: |P - C|^2 = r^2
    // Solve: |O + tD - C|^2 = r^2

    float3 oc = rayOrigin - p.position;
    float b = dot(oc, rayDir);
    float c = dot(oc, oc) - particleRadius * particleRadius;
    float discriminant = b * b - c;

    // Check if ray intersects sphere
    if (discriminant >= 0.0)
    {
        // Calculate intersection distance (nearest hit)
        float t = -b - sqrt(discriminant);

        // Check if intersection is within valid ray range
        if (t > RayTMin() && t < RayTCurrent())
        {
            // Report hit to DXR runtime
            ParticleAttributes attr;
            attr.unused = float2(0, 0);
            ReportHit(t, 0, attr);
        }
    }
}
