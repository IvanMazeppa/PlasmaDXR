// Water Material Shader for RT Liquid Simulation Rendering
// Uses DXR 1.1 RayQuery for reflection and refraction
// Single-bounce optics: 1 reflection + 1 refraction ray

#ifndef WATER_MATERIAL_HLSL
#define WATER_MATERIAL_HLSL

// Instance IDs for TLAS instances (matches RTLightingSystem_RayQuery.cpp)
static const uint INSTANCE_ID_PROBE_GRID = 0;
static const uint INSTANCE_ID_DIRECT_RT = 1;
static const uint INSTANCE_ID_GROUND = 2;
static const uint INSTANCE_ID_WATER = 3;

// Water material constants (hardcoded for MVP, will be cbuffer later)
static const float WATER_IOR = 1.33;                           // Index of refraction
static const float3 WATER_COLOR = float3(0.8, 0.9, 1.0);      // Slight blue tint
static const float3 WATER_ABSORPTION = float3(0.45, 0.09, 0.06); // Beer-Lambert absorption

// Fresnel-Schlick approximation
float FresnelSchlick(float cosTheta, float ior) {
    float r0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    return r0 + (1.0 - r0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Snell's law refraction (returns reflection if total internal reflection)
float3 RefractOrReflect(float3 incident, float3 normal, float eta) {
    float cosI = dot(-incident, normal);
    float sin2T = eta * eta * (1.0 - cosI * cosI);

    if (sin2T > 1.0) {
        return reflect(incident, normal);  // Total internal reflection
    }

    float cosT = sqrt(1.0 - sin2T);
    return eta * incident + (eta * cosI - cosT) * normal;
}

// Get triangle normal from barycentrics and vertex buffer
// vertexBuffer: interleaved pos(3) + normal(3) = 6 floats per vertex
float3 GetWaterTriangleNormal(
    RayQuery<RAY_FLAG_NONE> q,
    StructuredBuffer<float> waterVertexBuffer,
    Buffer<uint> waterIndexBuffer
) {
    uint primitiveIndex = q.CommittedPrimitiveIndex();
    float2 barycentrics = q.CommittedTriangleBarycentrics();

    // Get triangle indices
    uint i0 = waterIndexBuffer[primitiveIndex * 3 + 0];
    uint i1 = waterIndexBuffer[primitiveIndex * 3 + 1];
    uint i2 = waterIndexBuffer[primitiveIndex * 3 + 2];

    // Get normals from vertex buffer (offset by 3 for normal after position)
    float3 n0 = float3(
        waterVertexBuffer[i0 * 6 + 3],
        waterVertexBuffer[i0 * 6 + 4],
        waterVertexBuffer[i0 * 6 + 5]
    );
    float3 n1 = float3(
        waterVertexBuffer[i1 * 6 + 3],
        waterVertexBuffer[i1 * 6 + 4],
        waterVertexBuffer[i1 * 6 + 5]
    );
    float3 n2 = float3(
        waterVertexBuffer[i2 * 6 + 3],
        waterVertexBuffer[i2 * 6 + 4],
        waterVertexBuffer[i2 * 6 + 5]
    );

    // Interpolate normal using barycentrics
    float3 normal = n0 * (1.0 - barycentrics.x - barycentrics.y) +
                    n1 * barycentrics.x +
                    n2 * barycentrics.y;

    return normalize(normal);
}

// Single-bounce reflection ray
float3 TraceReflection(
    float3 origin,
    float3 direction,
    RaytracingAccelerationStructure tlas
) {
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin + direction * 0.01;  // Offset to avoid self-hit
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 10000.0;

    // Exclude water from reflection (avoid infinite recursion)
    uint instanceMask = 0xFFu & ~(1u << INSTANCE_ID_WATER);
    q.TraceRayInline(tlas, RAY_FLAG_NONE, instanceMask, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Hit scene geometry - return simple shaded color
        // TODO: Integrate with existing lighting for accurate color
        uint instanceID = q.CommittedInstanceID();
        if (instanceID == INSTANCE_ID_GROUND) {
            return float3(0.3, 0.3, 0.35);  // Ground plane color
        }
        return float3(0.4, 0.4, 0.45);  // Other geometry
    }

    // Sky/environment gradient
    return lerp(float3(0.5, 0.7, 1.0), float3(0.1, 0.2, 0.4),
                saturate(direction.y * 0.5 + 0.5));
}

// Single-bounce refraction ray with Beer-Lambert absorption
float3 TraceRefraction(
    float3 origin,
    float3 direction,
    RaytracingAccelerationStructure tlas,
    float3 absorption
) {
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin + direction * 0.01;  // Offset to avoid self-hit
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 10000.0;

    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 hitColor = float3(0, 0, 0);
    float hitDistance = ray.TMax;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        hitDistance = q.CommittedRayT();
        uint instanceID = q.CommittedInstanceID();

        if (instanceID == INSTANCE_ID_WATER) {
            // Hit water surface again (exit point)
            // For MVP, just use simple color
            hitColor = float3(0.2, 0.25, 0.3);
        } else if (instanceID == INSTANCE_ID_GROUND) {
            hitColor = float3(0.2, 0.2, 0.2);  // Ground through water
        } else {
            hitColor = float3(0.3, 0.3, 0.3);  // Other geometry through water
        }
    } else {
        // No hit - look at environment through water
        hitColor = float3(0.1, 0.15, 0.2);
    }

    // Beer-Lambert absorption: I = I0 * exp(-absorption * distance)
    float3 transmittance = exp(-absorption * hitDistance * 0.01);
    return hitColor * transmittance * WATER_COLOR;
}

// Main water shading function
// Called when a primary ray hits the water mesh
float3 ShadeWater(
    float3 hitPos,
    float3 normal,
    float3 viewDir,
    RaytracingAccelerationStructure tlas,
    float ior,
    float3 absorption
) {
    // Ensure normal faces viewer
    float3 N = dot(normal, viewDir) > 0 ? normal : -normal;
    float NdotV = saturate(dot(N, viewDir));

    // Fresnel determines reflection vs refraction ratio
    float fresnel = FresnelSchlick(NdotV, ior);

    // Reflection (single bounce)
    float3 reflectDir = reflect(-viewDir, N);
    float3 reflection = TraceReflection(hitPos, reflectDir, tlas);

    // Refraction (single bounce)
    float eta = 1.0 / ior;  // Air to water
    float3 refractDir = RefractOrReflect(-viewDir, N, eta);
    float3 refraction = TraceRefraction(hitPos, refractDir, tlas, absorption);

    // Combine based on Fresnel
    return lerp(refraction, reflection, fresnel);
}

// Simplified version using default constants
float3 ShadeWaterSimple(
    float3 hitPos,
    float3 normal,
    float3 viewDir,
    RaytracingAccelerationStructure tlas
) {
    return ShadeWater(hitPos, normal, viewDir, tlas, WATER_IOR, WATER_ABSORPTION);
}

#endif // WATER_MATERIAL_HLSL
