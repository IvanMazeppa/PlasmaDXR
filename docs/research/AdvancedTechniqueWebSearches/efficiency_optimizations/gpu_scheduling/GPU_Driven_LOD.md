# GPU-Driven Adaptive LOD for Particle Systems

## Source
- Paper/Article: GPU Work Graphs and GPU-Driven Rendering
- Authors: AMD/Microsoft collaboration
- Date: 2024
- Conference/Journal: Industry technical documentation

## Summary
GPU work graphs represent a paradigm shift in GPU programmability, allowing the GPU to schedule work autonomously without CPU intervention. For particle systems, this enables dynamic LOD selection, adaptive quality adjustments, and intelligent work distribution based on real-time performance metrics.

The technique allows particles to self-organize into quality tiers based on screen coverage, distance, and importance, with the GPU dynamically allocating compute resources to maintain target framerates.

## Key Innovation
GPU-autonomous scheduling of particle workloads using work graphs, enabling particles to spawn additional work for higher quality rendering when performance permits, or reduce quality when under pressure.

## Implementation Details

### Algorithm
```hlsl
// GPU-driven LOD selection for particles
struct ParticleLOD {
    uint level;          // 0 = highest quality, 3 = lowest
    uint raysPerParticle; // Ray budget for lighting
    float splatRadius;   // Screen-space size
    uint shadingRate;    // Variable rate shading
};

ParticleLOD ComputeParticleLOD(Particle p, float3 cameraPos, float2 screenSize) {
    ParticleLOD lod;

    // Distance-based LOD
    float distance = length(p.position - cameraPos);
    float screenCoverage = ProjectedParticleSize(p, cameraPos, screenSize);

    // Importance factors
    float luminance = dot(p.emission, float3(0.299, 0.587, 0.114));
    float velocityImportance = length(p.velocity) / MAX_VELOCITY;

    // Combined importance score
    float importance = luminance * 0.5 + velocityImportance * 0.3 + screenCoverage * 0.2;

    // Select LOD based on importance and distance
    if (distance < 10.0 && importance > 0.7) {
        lod.level = 0;
        lod.raysPerParticle = 16;
        lod.splatRadius = p.radius;
        lod.shadingRate = 1;
    } else if (distance < 50.0 && importance > 0.3) {
        lod.level = 1;
        lod.raysPerParticle = 8;
        lod.splatRadius = p.radius * 0.75;
        lod.shadingRate = 2;
    } else if (distance < 100.0) {
        lod.level = 2;
        lod.raysPerParticle = 4;
        lod.splatRadius = p.radius * 0.5;
        lod.shadingRate = 4;
    } else {
        lod.level = 3;
        lod.raysPerParticle = 0; // No ray tracing, just rasterization
        lod.splatRadius = p.radius * 0.25;
        lod.shadingRate = 8;
    }

    return lod;
}
```

### Code Snippets
```hlsl
// GPU Work Graph node for adaptive particle processing
[Shader("node")]
[NodeLaunch("thread")]
[NodeMaxDispatchGrid(1024, 1, 1)]
[NumThreads(32, 1, 1)]
void ParticleProcessingNode(
    DispatchNodeInputRecord<ParticleWorkItem> inputData,
    [MaxRecords(32)] NodeOutput<HighQualityWork> highQualityOutput,
    [MaxRecords(128)] NodeOutput<MediumQualityWork> mediumQualityOutput,
    [MaxRecords(256)] NodeOutput<LowQualityWork> lowQualityOutput,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIdx = inputData.Get().particleIndex;
    Particle p = g_Particles[particleIdx];

    // GPU-driven LOD decision
    ParticleLOD lod = ComputeParticleLOD(p, g_CameraPos, g_ScreenSize);

    // Check GPU performance metrics
    float gpuUtilization = ReadGPUCounter(GPU_UTILIZATION);
    float targetFrameTime = 16.67; // 60 FPS target
    float currentFrameTime = ReadGPUCounter(FRAME_TIME);

    // Adaptive quality adjustment
    if (currentFrameTime > targetFrameTime * 0.9) {
        // Downgrade quality if approaching frame time limit
        lod.level = min(lod.level + 1, 3);
        lod.raysPerParticle = max(lod.raysPerParticle / 2, 0);
    } else if (currentFrameTime < targetFrameTime * 0.5) {
        // Upgrade quality if we have headroom
        lod.level = max(lod.level - 1, 0);
        lod.raysPerParticle = min(lod.raysPerParticle * 2, 32);
    }

    // Dispatch to appropriate quality pipeline
    if (lod.level == 0) {
        ThreadNodeOutputRecords<HighQualityWork> outRec =
            highQualityOutput.GetThreadNodeOutputRecords(1);
        outRec.Get().particleIndex = particleIdx;
        outRec.Get().rayCount = lod.raysPerParticle;
        outRec.OutputComplete();
    } else if (lod.level <= 2) {
        ThreadNodeOutputRecords<MediumQualityWork> outRec =
            mediumQualityOutput.GetThreadNodeOutputRecords(1);
        outRec.Get().particleIndex = particleIdx;
        outRec.Get().rayCount = lod.raysPerParticle;
        outRec.OutputComplete();
    } else {
        ThreadNodeOutputRecords<LowQualityWork> outRec =
            lowQualityOutput.GetThreadNodeOutputRecords(1);
        outRec.Get().particleIndex = particleIdx;
        outRec.OutputComplete();
    }
}

// High quality particle rendering with ray tracing
[Shader("node")]
[NodeLaunch("thread")]
[NumThreads(32, 1, 1)]
void HighQualityParticleNode(
    DispatchNodeInputRecord<HighQualityWork> inputData,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIdx = inputData.Get().particleIndex;
    uint rayCount = inputData.Get().rayCount;
    Particle p = g_Particles[particleIdx];

    // Full ray-traced lighting with multiple samples
    float3 lighting = 0;

    for (uint i = 0; i < rayCount; i++) {
        // Generate stratified samples for better coverage
        float2 xi = Hammersley(i, rayCount);
        float3 dir = SampleHemisphere(p.normal, xi);

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        RayDesc ray;
        ray.Origin = p.position;
        ray.Direction = dir;
        ray.TMin = 0.001;
        ray.TMax = 100.0;

        q.TraceRayInline(g_AccelStruct, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            // Hit another particle or geometry
            uint hitInstance = q.CommittedInstanceID();
            if (IsParticle(hitInstance)) {
                Particle hitParticle = g_Particles[GetParticleIndex(hitInstance)];
                lighting += hitParticle.emission * dot(p.normal, dir);
            }
        } else {
            // Environment lighting
            lighting += SampleEnvironment(dir) * dot(p.normal, dir);
        }
    }

    lighting /= rayCount;

    // High-quality shading with subsurface scattering
    float3 color = ParticleSubsurfaceScattering(p, lighting);

    // Write to high-quality buffer
    WriteParticleOutput(particleIdx, color, 0); // LOD 0
}

// Hierarchical culling with GPU-driven dispatch
[numthreads(32, 1, 1)]
void HierarchicalCulling(uint3 id : SV_DispatchThreadID) {
    uint clusterIdx = id.x;
    if (clusterIdx >= g_NumClusters) return;

    ParticleCluster cluster = g_Clusters[clusterIdx];

    // Frustum culling at cluster level
    if (!IsInFrustum(cluster.bounds, g_Frustum)) {
        return; // Entire cluster culled
    }

    // Occlusion query using Hi-Z buffer
    float2 screenMin, screenMax;
    ProjectBounds(cluster.bounds, screenMin, screenMax);
    float maxDepth = SampleHiZ(screenMin, screenMax);

    if (cluster.bounds.minZ > maxDepth) {
        InterlockedAdd(g_OccludedClusters, 1);
        return; // Cluster occluded
    }

    // Screen-space size for LOD
    float screenSize = length(screenMax - screenMin);
    uint particlesPerPixel = cluster.particleCount / max(1, screenSize * screenSize);

    // Adaptive dispatch based on density
    if (particlesPerPixel > 10) {
        // Too dense - use simplified rendering
        DispatchIndirect(g_SimplifiedRenderDispatch, clusterIdx);
    } else {
        // Normal density - process individually
        DispatchIndirect(g_PerParticleDispatch, clusterIdx);
    }
}

// Dynamic resolution scaling based on GPU load
[numthreads(1, 1, 1)]
void AdaptiveResolutionControl(uint3 id : SV_DispatchThreadID) {
    // Read GPU performance counters
    float frameTime = g_FrameTimeHistory[g_FrameCount % HISTORY_SIZE];
    float avgFrameTime = 0;

    for (uint i = 0; i < HISTORY_SIZE; i++) {
        avgFrameTime += g_FrameTimeHistory[i];
    }
    avgFrameTime /= HISTORY_SIZE;

    // Compute new resolution scale
    float targetTime = 16.67; // 60 FPS
    float scale = g_CurrentResolutionScale;

    if (avgFrameTime > targetTime * 1.1) {
        scale *= 0.95; // Reduce resolution
    } else if (avgFrameTime < targetTime * 0.8) {
        scale *= 1.05; // Increase resolution
    }

    scale = clamp(scale, 0.5, 1.0);

    // Apply hysteresis to avoid flickering
    if (abs(scale - g_CurrentResolutionScale) > 0.05) {
        g_CurrentResolutionScale = scale;
        g_RenderResolution = uint2(g_DisplayResolution * scale);
    }

    // Adjust particle LOD bias based on resolution
    g_ParticleLODBias = lerp(0, 2, 1.0 - scale);
}
```

### Data Structures
```hlsl
// GPU Work Graph structures
struct ParticleWorkItem {
    uint particleIndex;
};

struct HighQualityWork {
    uint particleIndex;
    uint rayCount;
};

struct MediumQualityWork {
    uint particleIndex;
    uint rayCount;
};

struct LowQualityWork {
    uint particleIndex;
};

// Performance monitoring
struct GPUPerformanceCounters {
    float frameTime;
    float drawTime;
    float computeTime;
    float rayTraceTime;
    uint particlesRendered;
    uint raysTraced;
    float gpuUtilization;
    float memoryBandwidth;
};

// Hierarchical clustering for culling
struct ParticleCluster {
    AABB bounds;
    uint particleStart;
    uint particleCount;
    float avgLuminance;
    uint lodBias;
};

// Dynamic dispatch arguments
RWByteAddressBuffer g_SimplifiedRenderDispatch : register(u0);
RWByteAddressBuffer g_PerParticleDispatch : register(u1);
```

### Pipeline Integration
1. **Initial Dispatch**: CPU dispatches initial work graph
2. **GPU Scheduling**: GPU autonomously processes particles
3. **Adaptive Quality**: Adjust LOD based on performance
4. **Hierarchical Culling**: Cull at cluster level first
5. **Dynamic Resolution**: Scale rendering resolution as needed

## Performance Metrics
- GPU Time: 20-50% reduction vs. CPU-driven dispatch
- Memory Usage: Minimal overhead for work graph buffers
- Quality Metrics: Maintains target framerate with adaptive quality

## Hardware Requirements
- Minimum GPU: RDNA3/Ada Lovelace (work graph support)
- Optimal GPU: Future architectures with enhanced work graphs

## Implementation Complexity
- Estimated Dev Time: 5-7 days
- Risk Level: High (new programming model)
- Dependencies: Work graphs API, latest drivers

## Related Techniques
- Mesh shaders for GPU-driven geometry
- Variable rate shading (VRS)
- Sampler feedback streaming

## Notes for PlasmaDX Integration
- Start with simpler GPU-driven culling before full work graphs
- Use ExecuteIndirect for current hardware compatibility
- Implement performance counters for adaptive quality
- Consider hybrid CPU/GPU scheduling during transition
- Profile extensively as behavior varies across GPUs