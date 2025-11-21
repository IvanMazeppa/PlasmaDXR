// Froxel Voxel Lighting - Pass 2
// Calculates lighting at each voxel using multi-light system
// This decouples lighting from rendering for massive performance gains

// Particle structure (for potential use in advanced lighting)
struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
    float lifetime;
    uint materialType;
    float2 padding;
};

// Light structure (64 bytes, matches C++)
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
    float enableGodRays;
    float godRayIntensity;
    float godRayLength;
    float godRayFalloff;
    float3 godRayDirection;
    float godRayConeAngle;
    float godRayRotationSpeed;
    float _padding;
};

// Froxel grid parameters
cbuffer FroxelParams : register(b0)
{
    float3 gridMin;
    float padding0;
    float3 gridMax;
    float padding1;
    uint3 gridDimensions;
    uint lightCount;
    float3 voxelSize;
    float lightingMultiplier;      // Global lighting scale (default 1.0)
};

// Input: Density grid from Pass 1
Texture3D<float> g_densityGrid : register(t0);

// Input: Light buffer
StructuredBuffer<Light> g_lights : register(t1);

// Input: Particle BVH for shadow rays
RaytracingAccelerationStructure g_particleBVH : register(t2);

// Output: 3D lighting grid (R16G16B16A16_FLOAT)
// RGB = accumulated light color, A = density (for sampling)
RWTexture3D<float4> g_lightingGrid : register(u0);

//------------------------------------------------------------------------------
// Voxel Lighting Compute Shader
// Thread group: 8×8×8 = 512 voxels per group
//------------------------------------------------------------------------------
[numthreads(8, 8, 8)]
void main(uint3 voxelIdx : SV_DispatchThreadID)
{
    // Bounds check
    if (any(voxelIdx >= gridDimensions))
        return;

    // Sample density at this voxel (from Pass 1 density injection)
    float density = g_densityGrid[voxelIdx];

    // Early out if empty voxel (no particles here)
    if (density < 0.001) {
        g_lightingGrid[voxelIdx] = float4(0, 0, 0, 0);
        return;
    }

    // Convert voxel index to world position (center of voxel)
    float3 worldPos = gridMin + (float3(voxelIdx) + 0.5) * voxelSize;

    // === ACCUMULATE LIGHTING FROM ALL LIGHTS ===
    float3 totalLight = float3(0, 0, 0);

    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
        Light light = g_lights[lightIdx];

        // Calculate light direction and distance
        float3 toLight = light.position - worldPos;
        float lightDist = length(toLight);
        float3 lightDir = toLight / max(lightDist, 0.001);

        // Distance attenuation (quadratic falloff with light radius)
        float normalizedDist = lightDist / max(light.radius, 1.0);
        float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

        // === SHADOW RAY (DISABLED - Performance optimization) ===
        // TODO: Re-enable with tighter BVH bounds or distance-based culling
        // Shadow rays are VERY expensive for 921K voxels × 13 lights = 12M rays
        // With oversized BVH bounds, almost every ray hits → everything 80% shadowed

        // For now, disable shadows entirely to verify multi-light contribution
        float shadowTerm = 1.0;  // Fully lit (no shadows)

        // ORIGINAL CODE (commented out for performance):
        // RayDesc shadowRay;
        // shadowRay.Origin = worldPos + lightDir * 0.1;
        // shadowRay.Direction = lightDir;
        // shadowRay.TMin = 0.01;
        // shadowRay.TMax = lightDist - 0.1;
        // RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        // q.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
        // q.Proceed();
        // float shadowTerm = (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) ? 0.2 : 1.0;

        // Accumulate this light's contribution
        // NOTE: We don't use phase function here - that's applied during sampling in Pass 3
        float3 lightContribution = light.color * light.intensity * attenuation * shadowTerm;
        totalLight += lightContribution;
    }

    // Apply global lighting multiplier
    totalLight *= lightingMultiplier;

    // Store in lighting grid
    // RGB = accumulated light color (pre-multiplied by density)
    // A = density (for sampling in Pass 3 - determines fog visibility)
    g_lightingGrid[voxelIdx] = float4(totalLight * density, density);
}
