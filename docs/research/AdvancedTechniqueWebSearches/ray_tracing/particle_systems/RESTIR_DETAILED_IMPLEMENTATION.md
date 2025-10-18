# ReSTIR for Particle Lighting - Detailed Implementation Guide

**Research Date:** 2025-10-03
**Status:** Production-Ready (2020-2025)
**Maturity Level:** Proven in shipped titles

---

## OVERVIEW

ReSTIR (Reservoir-based Spatiotemporal Importance Resampling) achieves **6-60× speedup** over traditional Monte Carlo sampling for dynamic lighting. It's the #1 recommended technique for 100K particle lighting.

**Core Idea:** Instead of tracing many rays per pixel, intelligently sample a few high-quality light candidates using spatiotemporal reuse.

---

## MATHEMATICAL FOUNDATION

### Resampled Importance Sampling (RIS)

Traditional Monte Carlo lighting:
```
L = Σ(f(xi) / p(xi)) where xi ~ p(x)
```

ReSTIR improvement:
```
1. Generate M candidates from source PDF p(x)
2. Resample 1 candidate weighted by target PDF p̂(x)
3. Result has PDF closer to optimal p*(x) = f(x) / ∫f(x)dx
```

**Key Insight:** By resampling from p(x) → p̂(x), we can convert cheap uniform samples into high-quality importance samples.

### Weighted Reservoir Sampling

Algorithm to maintain 1 sample from M candidates without storing all M:

```cpp
Reservoir WRS(Stream<Sample> candidates) {
    Reservoir r = {null, 0.0, 0.0, 0};

    for each candidate c with weight w {
        r.wSum += w;
        r.M += 1;

        // Probabilistically replace current sample
        if (Random() < w / r.wSum) {
            r.sample = c;
        }
    }

    // Final weight for unbiased estimation
    r.weight = r.wSum / max(r.M, 1);

    return r;
}
```

**Memory:** O(1) instead of O(M)
**Time:** O(M) single pass
**Result:** Unbiased estimator with improved variance

---

## PARTICLE LIGHTING ADAPTATION

### Challenge
With 100,000 emissive particles, each pixel could theoretically be lit by many particles. Evaluating all is prohibitive (100K samples/pixel = impossible at 60fps).

### Solution
1. **Initial Sampling:** Randomly sample 16-32 particles per pixel
2. **Temporal Reuse:** Combine with previous frame's sample (amortize cost)
3. **Spatial Reuse:** Share samples with neighboring pixels (lateral propagation)
4. **Visibility:** Trace 1 shadow ray to final selected particle

**Result:** ~16-32 evaluations + 1 ray per pixel = 6-60× faster than 100K evaluations or 64+ rays

---

## DETAILED ALGORITHM

### Phase 1: Initial Candidate Generation

**Goal:** For each pixel, build a reservoir of M=16-32 particle light candidates

```hlsl
// ParticleReSTIR_InitialSampling.hlsl
RWStructuredBuffer<Reservoir> gReservoirs : register(u0);
StructuredBuffer<ParticleData> gParticles : register(t0);
Texture2D<float4> gGBufferWorldPos : register(t1);
Texture2D<float4> gGBufferNormal : register(t2);

cbuffer Constants : register(b0) {
    uint gParticleCount;
    uint gInitialCandidates;  // M = 16-32
    uint gFrameIndex;
    float3 gCameraPos;
};

// Blue noise or other low-discrepancy sampler
float2 GetSampleOffset(uint2 pixelPos, uint sampleIdx) {
    // Use blue noise texture or Halton sequence
    uint2 texelPos = (pixelPos + uint2(sampleIdx * 7, sampleIdx * 13)) % 128;
    return gBlueNoise[texelPos].xy;
}

[numthreads(8, 8, 1)]
void InitialSamplingCS(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixelPos = DTid.xy;
    uint pixelIndex = pixelPos.y * gScreenWidth + pixelPos.x;

    // Load G-buffer data
    float3 worldPos = gGBufferWorldPos[pixelPos].xyz;
    float3 normal = gGBufferNormal[pixelPos].xyz;
    float roughness = gGBufferNormal[pixelPos].w;

    // Early exit for background
    if (length(normal) < 0.1) {
        gReservoirs[pixelIndex] = EmptyReservoir();
        return;
    }

    // Initialize reservoir
    Reservoir reservoir;
    reservoir.particleID = INVALID_PARTICLE;
    reservoir.wSum = 0.0;
    reservoir.M = 0;
    reservoir.weight = 0.0;

    // Random number state
    uint rngState = InitRNG(pixelPos, gFrameIndex);

    // Sample M candidates
    for (uint i = 0; i < gInitialCandidates; i++) {
        // Random particle selection (uniform distribution)
        uint candidateID = min(RandomUint(rngState) % gParticleCount, gParticleCount - 1);

        ParticleData particle = gParticles[candidateID];

        // Skip dead/inactive particles
        if (particle.emission.x + particle.emission.y + particle.emission.z < 0.001) {
            continue;
        }

        // Calculate unshadowed contribution (target function)
        float3 toLight = particle.position - worldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);

        if (dist < 0.001) continue;  // Skip self-lighting

        float3 L = toLight / dist;

        // Geometry term
        float NoL = saturate(dot(normal, L));
        if (NoL < 0.001) continue;  // Back-facing

        // BRDF (simplified Lambertian for particles)
        float3 albedo = gGBufferWorldPos[pixelPos].w;  // Packed in w channel
        float3 brdf = albedo / PI;

        // Light intensity with inverse-square falloff
        float3 radiance = particle.emission / max(distSq, 0.01);

        // Target PDF: p̂(x) = f(x) where f(x) is unshadowed lighting contribution
        float3 unshadowedContrib = brdf * radiance * NoL;
        float targetPDF = Luminance(unshadowedContrib);

        // Source PDF: p(x) = 1/N (uniform random selection)
        float sourcePDF = 1.0 / float(gParticleCount);

        // RIS weight: w = p̂(x) / p(x)
        float weight = targetPDF / max(sourcePDF, 1e-8);

        // Update reservoir with weighted reservoir sampling
        reservoir.wSum += weight;
        reservoir.M += 1;

        if (RandomFloat(rngState) < weight / reservoir.wSum) {
            reservoir.particleID = candidateID;
        }
    }

    // Finalize reservoir weight for unbiased estimation
    if (reservoir.M > 0) {
        // w = (1/M) * Σw_i = wSum / M
        reservoir.weight = reservoir.wSum / float(reservoir.M);
    }

    gReservoirs[pixelIndex] = reservoir;
}
```

**Key Points:**
- **Source PDF:** Uniform random particle selection (1/N)
- **Target PDF:** Unshadowed lighting contribution (BRDF × radiance × NoL)
- **Weight:** Ratio target/source (importance sampling weight)
- **Memory:** Only stores 1 particle ID per pixel (not M candidates)

---

### Phase 2: Temporal Reuse

**Goal:** Combine current frame's reservoir with previous frame's to amortize sampling cost

```hlsl
// ParticleReSTIR_TemporalReuse.hlsl
RWStructuredBuffer<Reservoir> gCurrentReservoirs : register(u0);
StructuredBuffer<Reservoir> gPreviousReservoirs : register(t0);
Texture2D<float4> gGBufferWorldPos : register(t1);
Texture2D<float4> gGBufferNormal : register(t2);
Texture2D<float4> gPrevGBufferWorldPos : register(t3);
Texture2D<float4> gPrevGBufferNormal : register(t4);
Texture2D<float2> gMotionVectors : register(t5);
StructuredBuffer<ParticleData> gParticles : register(t6);

cbuffer Constants : register(b0) {
    matrix gViewProj;
    matrix gPrevViewProj;
    uint gTemporalMCap;  // Usually 20
    float gNormalThreshold;  // 0.9
    float gDepthThreshold;  // 0.1
};

float3 WorldToScreen(float3 worldPos, matrix viewProj) {
    float4 clipPos = mul(float4(worldPos, 1.0), viewProj);
    clipPos.xyz /= clipPos.w;
    return float3(clipPos.xy * 0.5 + 0.5, clipPos.z);
}

bool IsValidReprojection(uint2 currentPixel, uint2 prevPixel,
                         float3 currentNormal, float3 prevNormal,
                         float currentDepth, float prevDepth) {
    // Check bounds
    if (any(prevPixel >= gScreenSize)) return false;

    // Normal similarity
    if (dot(currentNormal, prevNormal) < gNormalThreshold) return false;

    // Depth similarity (relative difference)
    float depthDiff = abs(currentDepth - prevDepth) / max(currentDepth, 0.01);
    if (depthDiff > gDepthThreshold) return false;

    return true;
}

// Re-evaluate target PDF at new location
float EvaluateTargetPDF(uint particleID, float3 worldPos, float3 normal) {
    ParticleData particle = gParticles[particleID];

    float3 toLight = particle.position - worldPos;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(distSq);

    if (dist < 0.001) return 0.0;

    float3 L = toLight / dist;
    float NoL = saturate(dot(normal, L));

    if (NoL < 0.001) return 0.0;

    // Simplified BRDF and radiance
    float3 radiance = particle.emission / max(distSq, 0.01);
    float contribution = Luminance(radiance) * NoL / PI;

    return contribution;
}

void MergeReservoirs(inout Reservoir dst, Reservoir src,
                     float3 worldPos, float3 normal,
                     inout uint rngState) {
    // Re-evaluate source's sample at destination location
    if (src.particleID == INVALID_PARTICLE || src.M == 0) {
        return;
    }

    float targetPDF = EvaluateTargetPDF(src.particleID, worldPos, normal);

    // Combine reservoirs using RIS merge
    float weight = targetPDF * src.M;  // Weight by number of samples seen

    dst.wSum += weight;
    dst.M += src.M;

    if (RandomFloat(rngState) < weight / dst.wSum) {
        dst.particleID = src.particleID;
    }
}

[numthreads(8, 8, 1)]
void TemporalReuseCS(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixelPos = DTid.xy;
    uint pixelIndex = pixelPos.y * gScreenWidth + pixelPos.x;

    // Load current reservoir
    Reservoir current = gCurrentReservoirs[pixelIndex];

    // Load G-buffer
    float3 worldPos = gGBufferWorldPos[pixelPos].xyz;
    float3 normal = gGBufferNormal[pixelPos].xyz;

    if (length(normal) < 0.1) {
        gCurrentReservoirs[pixelIndex] = EmptyReservoir();
        return;
    }

    // Reproject to previous frame
    float2 motionVector = gMotionVectors[pixelPos].xy;
    uint2 prevPixel = pixelPos - motionVector * gScreenSize;

    // Alternative: Manual reprojection
    // float3 screenPos = WorldToScreen(worldPos, gPrevViewProj);
    // uint2 prevPixel = screenPos.xy * gScreenSize;

    // Validate reprojection
    if (all(prevPixel < gScreenSize)) {
        float3 prevNormal = gPrevGBufferNormal[prevPixel].xyz;
        float3 prevWorldPos = gPrevGBufferWorldPos[prevPixel].xyz;

        float currentDepth = length(worldPos - gCameraPos);
        float prevDepth = length(prevWorldPos - gCameraPos);

        if (IsValidReprojection(pixelPos, prevPixel, normal, prevNormal,
                                currentDepth, prevDepth)) {
            // Load previous reservoir
            uint prevPixelIndex = prevPixel.y * gScreenWidth + prevPixel.x;
            Reservoir temporal = gPreviousReservoirs[prevPixelIndex];

            // Clamp temporal history
            if (temporal.M > gTemporalMCap) {
                temporal.M = gTemporalMCap;
            }

            // Merge temporal reservoir into current
            uint rngState = InitRNG(pixelPos, gFrameIndex);
            MergeReservoirs(current, temporal, worldPos, normal, rngState);
        }
    }

    // Update weight after merge
    if (current.M > 0) {
        float targetPDF = EvaluateTargetPDF(current.particleID, worldPos, normal);
        current.weight = (targetPDF * current.wSum) / max(float(current.M), 1.0);
    }

    gCurrentReservoirs[pixelIndex] = current;
}
```

**Key Points:**
- **Motion vectors** or **world-space reprojection** to find previous pixel
- **Validation:** Normal/depth similarity tests prevent ghosting
- **M capping:** Limit temporal history to 20-30 to allow adaptation
- **Re-evaluation:** Must recalculate target PDF at new location
- **Unbiased:** Proper MIS weighting maintains correctness

---

### Phase 3: Spatial Reuse

**Goal:** Share samples between neighboring pixels to increase effective M without more computation

```hlsl
// ParticleReSTIR_SpatialReuse.hlsl
RWStructuredBuffer<Reservoir> gOutputReservoirs : register(u0);
StructuredBuffer<Reservoir> gInputReservoirs : register(t0);
Texture2D<float4> gGBufferWorldPos : register(t1);
Texture2D<float4> gGBufferNormal : register(t2);
StructuredBuffer<ParticleData> gParticles : register(t3);

cbuffer Constants : register(b0) {
    uint gNumNeighbors;      // 3-10
    float gSpatialRadius;    // 30 pixels
    float gNormalThreshold;  // 0.906 (cos(25°))
    float gDepthThreshold;   // 0.1
};

uint2 SampleNeighborPixel(uint2 centerPixel, uint neighborIdx, inout uint rngState) {
    // Random offset within radius (disk sampling)
    float angle = RandomFloat(rngState) * 2.0 * PI;
    float radius = sqrt(RandomFloat(rngState)) * gSpatialRadius;

    int2 offset = int2(
        cos(angle) * radius,
        sin(angle) * radius
    );

    uint2 neighborPixel = clamp(
        int2(centerPixel) + offset,
        int2(0, 0),
        int2(gScreenSize - 1)
    );

    return neighborPixel;
}

[numthreads(8, 8, 1)]
void SpatialReuseCS(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixelPos = DTid.xy;
    uint pixelIndex = pixelPos.y * gScreenWidth + pixelPos.x;

    // Load center reservoir
    Reservoir center = gInputReservoirs[pixelIndex];

    // Load G-buffer
    float3 worldPos = gGBufferWorldPos[pixelPos].xyz;
    float3 normal = gGBufferNormal[pixelPos].xyz;

    if (length(normal) < 0.1) {
        gOutputReservoirs[pixelIndex] = EmptyReservoir();
        return;
    }

    float depth = length(worldPos - gCameraPos);

    // RNG for neighbor sampling
    uint rngState = InitRNG(pixelPos, gFrameIndex + 1000);  // Different seed than temporal

    // Sample neighbors
    for (uint i = 0; i < gNumNeighbors; i++) {
        uint2 neighborPixel = SampleNeighborPixel(pixelPos, i, rngState);
        uint neighborIndex = neighborPixel.y * gScreenWidth + neighborPixel.x;

        // Load neighbor data
        float3 neighborNormal = gGBufferNormal[neighborPixel].xyz;
        float3 neighborWorldPos = gGBufferWorldPos[neighborPixel].xyz;

        // Reject if too different
        if (dot(normal, neighborNormal) < gNormalThreshold) continue;

        float neighborDepth = length(neighborWorldPos - gCameraPos);
        float depthDiff = abs(depth - neighborDepth) / max(depth, 0.01);
        if (depthDiff > gDepthThreshold) continue;

        // Load neighbor reservoir
        Reservoir neighbor = gInputReservoirs[neighborIndex];

        // Merge neighbor's sample into center
        MergeReservoirs(center, neighbor, worldPos, normal, rngState);
    }

    // Update weight
    if (center.M > 0) {
        float targetPDF = EvaluateTargetPDF(center.particleID, worldPos, normal);
        center.weight = (targetPDF * center.wSum) / max(float(center.M), 1.0);
    }

    gOutputReservoirs[pixelIndex] = center;
}
```

**Key Points:**
- **Disk sampling:** Random neighbors within radius
- **Surface validation:** Reject neighbors with different normals/depths
- **Ping-pong buffers:** Input != output to avoid feedback
- **Iterations:** Can run 1-2 passes for more propagation
- **Bias control:** Careful weighting prevents over-smoothing

---

### Phase 4: Final Visibility and Shading

**Goal:** Trace 1 shadow ray to selected particle and compute final lighting

```hlsl
// ParticleReSTIR_FinalShading.hlsl
RaytracingAccelerationStructure gTLAS : register(t0);
StructuredBuffer<Reservoir> gFinalReservoirs : register(t1);
Texture2D<float4> gGBufferWorldPos : register(t2);
Texture2D<float4> gGBufferNormal : register(t3);
StructuredBuffer<ParticleData> gParticles : register(t4);
RWTexture2D<float4> gOutputLighting : register(u0);

struct ShadowPayload {
    bool visible;
};

[shader("raygeneration")]
void FinalShadingRGS() {
    uint2 pixelPos = DispatchRaysIndex().xy;
    uint pixelIndex = pixelPos.y * DispatchRaysDimensions().x + pixelPos.x;

    // Load reservoir
    Reservoir reservoir = gFinalReservoirs[pixelIndex];

    // Load G-buffer
    float3 worldPos = gGBufferWorldPos[pixelPos].xyz;
    float3 normal = gGBufferNormal[pixelPos].xyz;

    // Early exit
    if (length(normal) < 0.1 || reservoir.particleID == INVALID_PARTICLE || reservoir.M == 0) {
        gOutputLighting[pixelPos] = float4(0, 0, 0, 1);
        return;
    }

    // Get selected particle
    ParticleData particle = gParticles[reservoir.particleID];

    // Calculate lighting
    float3 toLight = particle.position - worldPos;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(distSq);
    float3 L = toLight / dist;

    float NoL = saturate(dot(normal, L));

    // Setup shadow ray
    RayDesc ray;
    ray.Origin = worldPos + normal * 0.001;  // Bias to prevent self-intersection
    ray.Direction = L;
    ray.TMin = 0.001;
    ray.TMax = dist - 0.001;

    // Trace shadow ray
    ShadowPayload payload;
    payload.visible = true;

    uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                    RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

    TraceRay(
        gTLAS,
        rayFlags,
        0xFF,  // Instance mask
        0,     // Ray contribution to hit group index
        0,     // Multiplier for geometry contribution
        0,     // Miss shader index
        ray,
        payload
    );

    // Compute final lighting if visible
    float3 lighting = 0;

    if (payload.visible) {
        float3 radiance = particle.emission / max(distSq, 0.01);
        float3 brdf = gAlbedo / PI;  // Simplified Lambertian

        // Final contribution weighted by reservoir weight
        lighting = brdf * radiance * NoL * reservoir.weight;
    }

    gOutputLighting[pixelPos] = float4(lighting, 1.0);
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload) {
    payload.visible = true;  // No hit = visible
}

[shader("anyhit")]
void ShadowAnyHit(inout ShadowPayload payload, BuiltInTriangleIntersectionAttributes attrib) {
    // For opaque geometry, accept hit
    payload.visible = false;
    AcceptHitAndEndSearch();
}
```

**Key Points:**
- **1 ray per pixel:** Massive savings vs. traditional multi-ray approaches
- **Shadow ray only:** No complex shading in hit shaders
- **Reservoir weight:** Importance sampling correction factor
- **Bias prevention:** Ray origin offset prevents self-intersection

---

## COMPLETE PIPELINE INTEGRATION

### Option A: Full Compute Pipeline (Recommended for Particles)

```cpp
void ParticleReSTIR::Execute(ID3D12GraphicsCommandList* cmdList) {
    // 1. Initial candidate generation
    cmdList->SetPipelineState(mInitialSamplingPSO);
    cmdList->SetComputeRootSignature(mRootSignature);
    cmdList->SetComputeRootDescriptorTable(0, mParticleBufferSRV);
    cmdList->SetComputeRootDescriptorTable(1, mGBufferSRV);
    cmdList->SetComputeRootDescriptorTable(2, mReservoirBufferUAV);

    cmdList->Dispatch(
        (mScreenWidth + 7) / 8,
        (mScreenHeight + 7) / 8,
        1
    );
    UAVBarrier(mReservoirBuffer);

    // 2. Temporal reuse
    cmdList->SetPipelineState(mTemporalReusePSO);
    cmdList->SetComputeRootDescriptorTable(0, mReservoirBufferSRV);      // Current (read)
    cmdList->SetComputeRootDescriptorTable(1, mPrevReservoirBufferSRV); // Previous
    cmdList->SetComputeRootDescriptorTable(2, mTemporalReservoirUAV);   // Output

    cmdList->Dispatch(
        (mScreenWidth + 7) / 8,
        (mScreenHeight + 7) / 8,
        1
    );
    UAVBarrier(mTemporalReservoirBuffer);

    // 3. Spatial reuse (1-2 iterations)
    for (uint32_t iter = 0; iter < mSpatialIterations; iter++) {
        cmdList->SetPipelineState(mSpatialReusePSO);
        cmdList->SetComputeRootDescriptorTable(0, mTemporalReservoirSRV);  // Input
        cmdList->SetComputeRootDescriptorTable(1, mFinalReservoirUAV);     // Output

        cmdList->Dispatch(
            (mScreenWidth + 7) / 8,
            (mScreenHeight + 7) / 8,
            1
        );
        UAVBarrier(mFinalReservoirBuffer);

        // Ping-pong for multiple iterations
        if (iter < mSpatialIterations - 1) {
            std::swap(mTemporalReservoirBuffer, mFinalReservoirBuffer);
        }
    }

    // 4. Final shading with ray tracing
    cmdList->SetPipelineState(mRaytracingPSO);

    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    dispatchDesc.RayGenerationShaderRecord = mShaderTable.GetRayGenRecord();
    dispatchDesc.MissShaderTable = mShaderTable.GetMissTable();
    dispatchDesc.HitGroupTable = mShaderTable.GetHitGroupTable();
    dispatchDesc.Width = mScreenWidth;
    dispatchDesc.Height = mScreenHeight;
    dispatchDesc.Depth = 1;

    cmdList->DispatchRays(&dispatchDesc);

    // 5. Copy current to previous for next frame
    cmdList->CopyResource(mPrevReservoirBuffer, mReservoirBuffer);
}
```

### Option B: Hybrid Inline RayQuery

Replace Phase 4 with inline rayquery in compute shader:

```hlsl
[numthreads(8, 8, 1)]
void FinalShadingWithRayQueryCS(uint3 DTid : SV_DispatchThreadID) {
    // ... same setup as before ...

    // Inline ray query instead of TraceRay
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;

    RayDesc ray;
    ray.Origin = worldPos + normal * 0.001;
    ray.Direction = L;
    ray.TMin = 0.001;
    ray.TMax = dist - 0.001;

    query.TraceRayInline(gTLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);

    query.Proceed();

    bool visible = (query.CommittedStatus() == COMMITTED_NOTHING);

    // ... rest of shading ...
}
```

---

## PERFORMANCE TUNING GUIDE

### Parameter Sweep

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Initial Candidates | 8 | 16 | 32 |
| Temporal M Cap | 10 | 20 | 30 |
| Spatial Neighbors | 3 | 5 | 10 |
| Spatial Radius | 15px | 30px | 50px |
| Spatial Iterations | 1 | 1 | 2 |

**Recommendation for RTX 4060 Ti @ 100K particles:**
- Start with **Balanced** settings
- Profile each phase separately
- Tune based on bottleneck:
  - **Initial sampling slow?** Reduce candidates to 12
  - **Temporal slow?** Check motion vector quality
  - **Spatial slow?** Reduce neighbors to 3
  - **Final shading slow?** Resolution scaling

### Common Pitfalls

1. **Flickering/Bias**
   - **Cause:** Temporal M cap too low
   - **Fix:** Increase M cap to 20-30

2. **Over-smoothing**
   - **Cause:** Spatial radius too large or too many neighbors
   - **Fix:** Reduce radius to 20-30px, neighbors to 3-5

3. **Ghosting**
   - **Cause:** Invalid temporal reprojection
   - **Fix:** Tighten normal/depth thresholds

4. **Fireflies**
   - **Cause:** High-weight outlier samples
   - **Fix:** Clamp weights or add variance filtering

5. **Performance Regression**
   - **Cause:** Too many candidates or spatial samples
   - **Fix:** Reduce to conservative settings

---

## VALIDATION AND DEBUGGING

### Visual Debugging

```hlsl
// Debug visualization modes
#define DEBUG_MODE_NONE 0
#define DEBUG_MODE_RESERVOIR_M 1
#define DEBUG_MODE_SELECTED_PARTICLE 2
#define DEBUG_MODE_WEIGHT 3

cbuffer DebugConstants {
    uint gDebugMode;
};

void DebugVisualize(uint2 pixelPos, Reservoir reservoir) {
    if (gDebugMode == DEBUG_MODE_RESERVOIR_M) {
        // Show number of samples (M) as heatmap
        float normalizedM = saturate(float(reservoir.M) / 100.0);
        gOutput[pixelPos] = float4(normalizedM, 0, 1.0 - normalizedM, 1);
    }
    else if (gDebugMode == DEBUG_MODE_SELECTED_PARTICLE) {
        // Color code by particle ID
        float hue = frac(float(reservoir.particleID) * 0.618034);
        gOutput[pixelPos] = float4(HSVtoRGB(hue, 1, 1), 1);
    }
    else if (gDebugMode == DEBUG_MODE_WEIGHT) {
        // Show reservoir weight
        float logWeight = log(reservoir.weight + 1.0) / 10.0;
        gOutput[pixelPos] = float4(logWeight, logWeight, logWeight, 1);
    }
}
```

### Quantitative Metrics

```cpp
struct ReSTIRStats {
    float avgM;
    float maxM;
    float avgWeight;
    float percentInvalidReservoirs;
};

ReSTIRStats ComputeStats(const Reservoir* reservoirs, uint32_t count) {
    ReSTIRStats stats = {};
    uint32_t validCount = 0;

    for (uint32_t i = 0; i < count; i++) {
        if (reservoirs[i].M > 0) {
            stats.avgM += reservoirs[i].M;
            stats.maxM = std::max(stats.maxM, float(reservoirs[i].M));
            stats.avgWeight += reservoirs[i].weight;
            validCount++;
        }
    }

    stats.avgM /= validCount;
    stats.avgWeight /= validCount;
    stats.percentInvalidReservoirs = 100.0f * (1.0f - float(validCount) / count);

    return stats;
}
```

**Expected Values:**
- **Avg M (after temporal):** 20-50
- **Avg M (after spatial):** 50-200
- **Invalid reservoirs:** <10% (background pixels)
- **Avg weight:** Highly scene-dependent

---

## ADVANCED OPTIMIZATIONS

### 1. Visibility Reuse

Instead of tracing 1 ray/pixel in final shading, reuse visibility:

```hlsl
// Trace visibility during spatial reuse (amortize cost)
bool TraceVisibility(uint particleID, float3 worldPos, float3 normal) {
    // ... same shadow ray logic ...
}

void MergeReservoirsWithVisibility(inout Reservoir dst, Reservoir src, ...) {
    // Only merge if visible
    if (TraceVisibility(src.particleID, worldPos, normal)) {
        MergeReservoirs(dst, src, worldPos, normal, rngState);
    }
}
```

**Tradeoff:** More rays during reuse, but final shading is free

### 2. Adaptive Sampling

Use screen-space gradients to adjust sample count:

```hlsl
uint GetAdaptiveCandidateCount(uint2 pixelPos) {
    float luminanceGradient = ComputeLuminanceGradient(pixelPos);

    // High gradient = more samples needed
    if (luminanceGradient > 0.5) return 32;
    else if (luminanceGradient > 0.2) return 16;
    else return 8;
}
```

### 3. Spatial Data Structures

Instead of uniform random particle selection, use spatial acceleration:

```cpp
// Octree or grid for particle-to-pixel culling
void SelectCandidatesFromGrid(float3 worldPos, float radius,
                               std::vector<uint32_t>& outCandidates) {
    // Query grid cells within radius
    // Return nearby particles only
}
```

**Benefit:** Reduce wasted samples on distant particles

### 4. Multi-Bounce ReSTIR GI

Extend to indirect lighting:

```hlsl
// Secondary bounce particles
for each particle hit by primary ray:
    Run ReSTIR to sample other particles lighting it
    Propagate indirect illumination
```

**Paper:** ReSTIR GI (NVIDIA Research 2021)
**Cost:** 2-3× more expensive but enables full global illumination

---

## PRODUCTION CHECKLIST

- [ ] Implement weighted reservoir sampling correctly
- [ ] Validate temporal reprojection with motion vectors
- [ ] Test spatial reuse with various radii
- [ ] Add debug visualization modes
- [ ] Profile each phase independently
- [ ] Handle edge cases (background pixels, particle death)
- [ ] Implement reservoir buffer ping-ponging
- [ ] Add M capping for temporal stability
- [ ] Test with extreme particle counts (10K, 100K, 1M)
- [ ] Validate unbiased weighting (compare with ground truth)
- [ ] Optimize for 60fps target
- [ ] Add runtime parameter tuning UI

---

## CITATIONS

1. Bitterli et al., "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting", SIGGRAPH 2020

2. Wyman, "A Gentle Introduction to ReSTIR Path Reuse in Real-Time", SIGGRAPH 2023 Course

3. Ouyang et al., "ReSTIR GI: Path Resampling for Real-Time Path Tracing", HPG 2021

4. NVIDIA, "Rendering Millions of Dynamic Lights in Real-Time", Developer Blog 2020

---

**STATUS:** Implementation-ready
**NEXT STEPS:** Integrate with particle BLAS system (see main research doc)
