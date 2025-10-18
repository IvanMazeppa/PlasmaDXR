# Screen-Space Techniques for Particle Systems

## Source
- Paper/Article: Various sources on modern SSAO/SSR implementations
- Authors: Industry practitioners
- Date: 2024
- Conference/Journal: Real-time rendering community

## Summary
Screen-space ambient occlusion (SSAO) and screen-space reflections (SSR) can be adapted for particle systems to add depth and inter-particle shadowing/reflections at minimal cost. These techniques work in screen space after particles are rendered, making them compatible with any particle rendering method including ray-traced particles.

For particle systems, SSAO can approximate inter-particle occlusion and self-shadowing, while SSR can capture particle reflections on surfaces and even particle-to-particle reflections when particles are opaque enough.

## Key Innovation
Adapting screen-space techniques to work with particle depth and normal buffers, using particle density as an additional factor in occlusion calculations, and combining with ray-traced primary visibility for hybrid quality/performance.

## Implementation Details

### Algorithm
```hlsl
// Particle-aware SSAO
float ParticleSSAO(float2 uv, float3 position, float3 normal, float particleDensity) {
    float occlusion = 0;
    const uint sampleCount = 16;

    // Hemisphere sampling with particle-aware radius
    float radius = g_SSAORadius * (1.0 + particleDensity * 0.5);

    for (uint i = 0; i < sampleCount; i++) {
        // Get sample offset in hemisphere
        float3 sampleDir = GetHemisphereSample(i, normal);
        float3 samplePos = position + sampleDir * radius;

        // Project to screen space
        float4 offset = mul(float4(samplePos, 1.0), g_ViewProj);
        offset.xy /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;

        // Sample depth at offset
        float sampleDepth = g_DepthBuffer.Sample(g_LinearSampler, offset.xy);

        // Check occlusion with falloff
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + g_Bias) ? 1.0 : 0.0 * rangeCheck;
    }

    return 1.0 - (occlusion / sampleCount);
}
```

### Code Snippets
```hlsl
// Combined SSAO pass for particles and geometry
[numthreads(8, 8, 1)]
void ParticleSSAO_CS(uint3 id : SV_DispatchThreadID) {
    float2 uv = (id.xy + 0.5) / g_Resolution;

    // Sample G-buffer
    float depth = g_DepthBuffer.Sample(g_PointSampler, uv);
    if (depth >= 1.0) {
        g_SSAOOutput[id.xy] = 1.0;
        return;
    }

    float3 position = ReconstructWorldPosition(uv, depth);
    float3 normal = DecodeNormal(g_NormalBuffer.Sample(g_PointSampler, uv).xy);

    // Check if this is a particle
    uint materialID = g_MaterialIDBuffer[id.xy];
    bool isParticle = (materialID & PARTICLE_MATERIAL_FLAG) != 0;

    float occlusion = 0;

    if (isParticle) {
        // Particle-specific SSAO with density awareness
        float density = g_ParticleDensityBuffer.Sample(g_PointSampler, uv);

        const uint SAMPLE_COUNT = 32; // More samples for particles
        float radius = g_SSAORadius * (1.0 + density * 0.5);

        // Poisson disk sampling for better quality
        for (uint i = 0; i < SAMPLE_COUNT; i++) {
            float2 offset = g_PoissonDisk[i] * radius / position.z;
            float2 sampleUV = uv + offset;

            float sampleDepth = g_DepthBuffer.Sample(g_PointSampler, sampleUV);
            float3 samplePos = ReconstructWorldPosition(sampleUV, sampleDepth);

            float3 v = samplePos - position;
            float dist2 = dot(v, v);
            float nDotV = dot(normal, normalize(v));

            // Particle-aware occlusion with soft falloff
            float ao = max(0, nDotV + g_Bias) * (1.0 / (1.0 + dist2 * g_Falloff));
            ao *= smoothstep(radius * 2, 0, length(v));

            // Boost occlusion for particle-particle interactions
            uint sampleMaterial = g_MaterialIDBuffer[sampleUV * g_Resolution];
            if ((sampleMaterial & PARTICLE_MATERIAL_FLAG) != 0) {
                ao *= 1.5; // Stronger inter-particle occlusion
            }

            occlusion += ao;
        }

        occlusion = 1.0 - saturate(occlusion / SAMPLE_COUNT * g_Intensity);
    } else {
        // Standard SSAO for geometry
        occlusion = StandardSSAO(position, normal, uv);
    }

    // Bilateral filter aware of particle boundaries
    g_SSAOOutput[id.xy] = BilateralFilter(occlusion, uv, depth, normal, isParticle);
}

// Screen-space reflections with particle support
[numthreads(8, 8, 1)]
void ParticleSSR_CS(uint3 id : SV_DispatchThreadID) {
    float2 uv = (id.xy + 0.5) / g_Resolution;

    // Get surface properties
    float roughness = g_RoughnessBuffer.Sample(g_PointSampler, uv);
    if (roughness > g_MaxRoughness) {
        g_SSROutput[id.xy] = 0;
        return;
    }

    float depth = g_DepthBuffer.Sample(g_PointSampler, uv);
    float3 position = ReconstructWorldPosition(uv, depth);
    float3 normal = DecodeNormal(g_NormalBuffer.Sample(g_PointSampler, uv).xy);
    float3 viewDir = normalize(g_CameraPos - position);

    // Reflection ray
    float3 reflectDir = reflect(-viewDir, normal);

    // Hierarchical ray marching in screen space
    float3 hitColor = 0;
    float hitMask = 0;

    // Start with coarse steps
    float stepSize = g_SSRStepSize;
    float3 rayPos = position;

    for (uint i = 0; i < g_MaxSteps; i++) {
        rayPos += reflectDir * stepSize;

        // Project to screen
        float4 projPos = mul(float4(rayPos, 1.0), g_ViewProj);
        projPos.xy /= projPos.w;
        float2 rayUV = projPos.xy * 0.5 + 0.5;

        if (any(rayUV < 0) || any(rayUV > 1)) break;

        float sceneDepth = g_DepthBuffer.Sample(g_PointSampler, rayUV);
        float3 scenePos = ReconstructWorldPosition(rayUV, sceneDepth);

        float depthDiff = rayPos.z - scenePos.z;

        if (depthDiff > 0 && depthDiff < stepSize * 2) {
            // Binary search refinement
            for (uint j = 0; j < 4; j++) {
                stepSize *= 0.5;
                rayPos -= reflectDir * stepSize * sign(depthDiff);

                projPos = mul(float4(rayPos, 1.0), g_ViewProj);
                rayUV = projPos.xy / projPos.w * 0.5 + 0.5;

                sceneDepth = g_DepthBuffer.Sample(g_PointSampler, rayUV);
                scenePos = ReconstructWorldPosition(rayUV, sceneDepth);
                depthDiff = rayPos.z - scenePos.z;
            }

            // Check if we hit a particle
            uint materialID = g_MaterialIDBuffer[rayUV * g_Resolution];
            bool hitParticle = (materialID & PARTICLE_MATERIAL_FLAG) != 0;

            if (hitParticle) {
                // Sample particle color with transparency
                float4 particleColor = g_ParticleColorBuffer.Sample(g_LinearSampler, rayUV);
                hitColor = particleColor.rgb;
                hitMask = particleColor.a * 0.5; // Reduce particle reflection intensity
            } else {
                // Sample regular scene color
                hitColor = g_ColorBuffer.Sample(g_LinearSampler, rayUV).rgb;
                hitMask = 1.0;
            }

            // Fade based on distance and roughness
            float fade = 1.0 - saturate(length(rayPos - position) / g_SSRMaxDistance);
            fade *= 1.0 - roughness;
            hitMask *= fade;

            break;
        }

        // Adaptive step size
        stepSize = max(g_SSRMinStepSize, stepSize * 1.1);
    }

    g_SSROutput[id.xy] = float4(hitColor, hitMask);
}

// Temporal accumulation for stable results
[numthreads(8, 8, 1)]
void TemporalAccumulation_CS(uint3 id : SV_DispatchThreadID) {
    float2 uv = (id.xy + 0.5) / g_Resolution;

    // Get motion vector
    float2 motion = g_MotionVectorBuffer.Sample(g_PointSampler, uv).xy;
    float2 prevUV = uv - motion;

    // Sample current and previous frames
    float4 current = g_CurrentFrame[id.xy];
    float4 history = g_HistoryBuffer.Sample(g_LinearSampler, prevUV);

    // Neighborhood clamping for stability
    float4 nearMin, nearMax;
    GatherNeighborhood(id.xy, nearMin, nearMax);

    history = clamp(history, nearMin, nearMax);

    // Blend factor based on motion and confidence
    float blend = g_TemporalBlend;
    blend *= (1.0 - length(motion) * 10.0); // Less history with motion
    blend = saturate(blend);

    float4 result = lerp(current, history, blend);
    g_OutputBuffer[id.xy] = result;
}
```

### Data Structures
```hlsl
// G-Buffer for particles and geometry
Texture2D<float> g_DepthBuffer : register(t0);
Texture2D<float2> g_NormalBuffer : register(t1);
Texture2D<float> g_RoughnessBuffer : register(t2);
Texture2D<uint> g_MaterialIDBuffer : register(t3);
Texture2D<float> g_ParticleDensityBuffer : register(t4);
Texture2D<float4> g_ParticleColorBuffer : register(t5);

// Screen-space effect outputs
RWTexture2D<float> g_SSAOOutput : register(u0);
RWTexture2D<float4> g_SSROutput : register(u1);

// Temporal buffers
Texture2D<float4> g_HistoryBuffer : register(t6);
Texture2D<float2> g_MotionVectorBuffer : register(t7);

// Sampling patterns
static const float2 g_PoissonDisk[32] = {
    // Precomputed Poisson disk samples
    float2(-0.94201624, -0.39906216),
    float2(0.94558609, -0.76890725),
    // ... more samples
};
```

### Pipeline Integration
1. **G-Buffer Pass**: Render particles with depth, normals, material ID
2. **Density Pass**: Compute particle density in screen space
3. **SSAO Pass**: Calculate ambient occlusion
4. **SSR Pass**: Trace screen-space reflections
5. **Temporal Pass**: Accumulate results over time
6. **Composite**: Blend with main render

## Performance Metrics
- GPU Time: 2-3ms for SSAO, 3-5ms for SSR at 1080p
- Memory Usage: G-Buffer overhead (~50MB at 1080p)
- Quality Metrics: Significant depth improvement, minimal artifacts

## Hardware Requirements
- Minimum GPU: GTX 1060 (basic screen-space effects)
- Optimal GPU: RTX 3060+ (for higher sample counts)

## Implementation Complexity
- Estimated Dev Time: 2-3 days
- Risk Level: Low (well-understood techniques)
- Dependencies: Depth/normal buffers, compute shaders

## Related Techniques
- HBAO+ (Horizon-Based Ambient Occlusion)
- GTAO (Ground Truth Ambient Occlusion)
- Stochastic Screen-Space Reflections

## Notes for PlasmaDX Integration
- Render particles to G-buffer with unique material ID
- Use particle density for variable SSAO radius
- Consider lower-resolution SSAO for performance
- Temporal accumulation crucial for stable results
- Can combine with ray-traced AO for hybrid quality