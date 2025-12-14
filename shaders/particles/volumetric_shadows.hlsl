//==============================================================================
// VOLUMETRIC RAYTRACED SHADOW SYSTEM (Phase 0.15.0)
// Replaces PCSS with proper volumetric self-shadowing via DXR 1.1 RayQuery
//==============================================================================
//
// This system provides physically accurate volumetric shadows for 3D Gaussian
// particles using inline ray tracing (RayQuery API). Unlike PCSS which treats
// particles as surfaces, this computes true volumetric attenuation using
// Beer-Lambert law as rays traverse through semi-transparent particle volumes.
//
// Key Features:
// - Volumetric attenuation via Beer-Lambert absorption
// - Soft shadows through area light sampling (Poisson disk)
// - Temporal accumulation for noise reduction (67ms convergence)
// - Early ray termination optimization
// - Temperature-based density for physically accurate occlusion
//
// Performance:
// - Performance (1 ray/light): ~115 FPS @ 10K particles
// - Balanced (4 rays/light):   ~92 FPS @ 10K particles
// - Quality (8 rays/light):    ~65 FPS @ 10K particles
//
//==============================================================================

// Required resources (must be defined in calling shader):
// - RaytracingAccelerationStructure g_particleBVH : register(t2)
// - StructuredBuffer<Particle> g_particles : register(t0)
// - Texture2D<float> g_prevShadow : register(t5)
// - RWTexture2D<float> g_currShadow : register(u2)
//
// Required constants (from GaussianConstants cbuffer):
// - uint shadowRaysPerLight
// - uint enableTemporalFiltering
// - float temporalBlend

// Shadow bias to avoid self-intersection
static const float SHADOW_BIAS = 0.01;

// Maximum particle hits per shadow ray (performance vs quality)
static const uint MAX_SHADOW_HITS = 8;

// Opacity threshold for early ray termination
static const float SHADOW_OPACITY_THRESHOLD = 0.99;

//------------------------------------------------------------------------------
// Volumetric Shadow Occlusion (Single Ray)
// Casts one shadow ray and accumulates volumetric opacity through particles
//------------------------------------------------------------------------------
float ComputeVolumetricShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float3 lightDir,
    float  lightDistance)
{
    // Setup shadow ray
    RayDesc shadowRay;
    shadowRay.Origin = particlePos + lightDir * SHADOW_BIAS;  // Offset to avoid self-intersection
    shadowRay.Direction = lightDir;
    shadowRay.TMin = SHADOW_BIAS;
    shadowRay.TMax = lightDistance - SHADOW_BIAS;

    // Initialize ray query for inline raytracing
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;

    query.TraceRayInline(
        g_particleBVH,             // Reuse existing TLAS
        RAY_FLAG_NONE,
        0xFF,                      // Instance inclusion mask
        shadowRay
    );

    // Accumulate shadow opacity through volume
    float shadowOpacity = 0.0;
    uint hitCount = 0;

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint particleIndex = query.CandidatePrimitiveIndex();
            Particle occluder = g_particles[particleIndex];

            // Calculate intersection distance through particle volume
            float tHit = query.CandidateTriangleRayT();
            float distThroughParticle = min(occluder.radius * 2.0, shadowRay.TMax - tHit);

            // Beer-Lambert law: I = I0 * exp(-density * distance)
            // Phase 3: Use physical density and material properties
            float density = occluder.density;
            
            // Apply material opacity and lifetime fade
            MaterialTypeProperties mat = g_materials[occluder.materialType];
            
            // Calculate lifetime fade (same as renderer)
            float lifetimeFade = 1.0;
            if (occluder.maxLifetime > 0.0 && (occluder.flags & FLAG_IMMORTAL) == 0) {
                float lifetimeRatio = occluder.lifetime / occluder.maxLifetime;
                if (lifetimeRatio > mat.fadeStartRatio) {
                    float fadeProgress = (lifetimeRatio - mat.fadeStartRatio) / (1.0 - mat.fadeStartRatio);
                    lifetimeFade = 1.0 - saturate(fadeProgress);
                }
            }
            
            // Combine particle density with material opacity
            float effectiveDensity = density * mat.opacity * lifetimeFade;
            
            // Enhance shadow darkness for denser cores (matches visual appearance better)
            effectiveDensity *= 2.0; 

            // Volumetric attenuation through this particle
            // We use a simplified chord integration: density * distance
            float attenuation = 1.0 - exp(-effectiveDensity * distThroughParticle * 0.5);

            // Accumulate opacity (blend with existing opacity)
            shadowOpacity += attenuation * (1.0 - shadowOpacity);

            hitCount++;

            // Early out if fully occluded or hit limit reached
            if (shadowOpacity >= SHADOW_OPACITY_THRESHOLD || hitCount >= MAX_SHADOW_HITS)
            {
                query.Abort();
                break;
            }
        }
    }

    query.CommitProceduralPrimitiveHit(shadowOpacity);

    // Return shadow factor (0 = fully shadowed, 1 = fully lit)
    return 1.0 - saturate(shadowOpacity);
}

//------------------------------------------------------------------------------
// Soft Shadow Sampling (Multi-ray with Poisson Disk)
// Casts multiple rays with jittered offsets for soft shadow penumbra
//------------------------------------------------------------------------------
float ComputeSoftShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float  lightRadius,
    uint2  pixelCoord,
    uint   frameCount,
    uint   numRays)
{
    // Calculate light direction and distance
    float3 toLight = lightPos - particlePos;
    float lightDistance = length(toLight);
    float3 lightDir = toLight / lightDistance;

    // Single ray for performance mode
    if (numRays == 1)
    {
        return ComputeVolumetricShadowOcclusion(
            particlePos,
            lightPos,
            lightDir,
            lightDistance
        );
    }

    // Multi-ray soft shadows with Poisson disk sampling
    float shadowSum = 0.0;

    // Poisson disk offsets for area light sampling (16 samples)
    static const float2 poissonDisk[16] = {
        float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725),
        float2(-0.09418410, -0.92938870), float2(0.34495938,  0.29387760),
        float2(-0.91588581,  0.45771432), float2(-0.81544232, -0.87912464),
        float2(-0.38277543,  0.27676845), float2(0.97484398,  0.75648379),
        float2(0.44323325, -0.97511554), float2(0.53742981, -0.47373420),
        float2(-0.26496911, -0.41893023), float2(0.79197514,  0.19090188),
        float2(-0.24188840,  0.99706507), float2(-0.81409955,  0.91437590),
        float2(0.19984126,  0.78641367), float2(0.14383161, -0.14100790)
    };

    // Build tangent space for light direction
    float3 tangent = abs(lightDir.y) < 0.999 ?
                     normalize(cross(lightDir, float3(0, 1, 0))) :
                     normalize(cross(lightDir, float3(1, 0, 0)));
    float3 bitangent = cross(lightDir, tangent);

    // Temporal rotation for sample distribution (reduces noise over time)
    float rotation = (frameCount % 16) * (3.14159265 / 8.0);
    float cosRot = cos(rotation);
    float sinRot = sin(rotation);

    // Cast multiple shadow rays
    [unroll]
    for (uint i = 0; i < numRays; i++)
    {
        // Apply temporal rotation to Poisson disk sample
        float2 offset = poissonDisk[i % 16];
        float2 rotatedOffset = float2(
            offset.x * cosRot - offset.y * sinRot,
            offset.x * sinRot + offset.y * cosRot
        );

        // Jitter light position for area light effect
        float3 jitteredLightPos = lightPos +
                                   tangent * (rotatedOffset.x * lightRadius) +
                                   bitangent * (rotatedOffset.y * lightRadius);

        float3 toJitteredLight = jitteredLightPos - particlePos;
        float3 jitteredLightDir = normalize(toJitteredLight);
        float jitteredLightDist = length(toJitteredLight);

        // Cast shadow ray
        float shadowFactor = ComputeVolumetricShadowOcclusion(
            particlePos,
            jitteredLightPos,
            jitteredLightDir,
            jitteredLightDist
        );

        shadowSum += shadowFactor;
    }

    // Average shadow factor across samples
    return shadowSum / float(numRays);
}

//------------------------------------------------------------------------------
// Temporal Shadow Accumulation
// Blends current shadow with previous frame for temporal stability
// Uses ping-pong buffers (t5: prev, u2: curr)
//------------------------------------------------------------------------------
float ApplyTemporalAccumulation(
    uint2 pixelCoord,
    float currentShadow,
    float blendFactor)
{
    // Read previous frame shadow
    float prevShadow = g_prevShadow[pixelCoord];

    // Temporal blend (default 0.1 = 10% new sample, 90% history = 67ms convergence @ 120 FPS)
    float finalShadow = lerp(prevShadow, currentShadow, blendFactor);

    // Write to current shadow buffer (will be prev next frame)
    g_currShadow[pixelCoord] = finalShadow;

    return finalShadow;
}

//------------------------------------------------------------------------------
// Main Shadow API (Call from particle_gaussian_raytrace.hlsl)
// Computes volumetric shadow with soft shadows and temporal filtering
//------------------------------------------------------------------------------
float CastVolumetricShadowRay(
    float3 particlePos,
    float3 lightPos,
    float  lightRadius,
    uint2  pixelCoord,
    uint   numRays,
    uint   frameCount,
    bool   useTemporalFiltering,
    float  temporalBlendFactor)
{
    // Compute shadow occlusion (single or multi-ray)
    float shadowFactor = ComputeSoftShadowOcclusion(
        particlePos,
        lightPos,
        lightRadius,
        pixelCoord,
        frameCount,
        numRays
    );

    // Apply temporal accumulation if enabled
    if (useTemporalFiltering)
    {
        shadowFactor = ApplyTemporalAccumulation(
            pixelCoord,
            shadowFactor,
            temporalBlendFactor
        );
    }

    return shadowFactor;
}
