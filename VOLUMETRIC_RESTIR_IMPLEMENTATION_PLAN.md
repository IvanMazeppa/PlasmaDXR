# Volumetric ReSTIR Implementation Plan

**Project**: PlasmaDX-Clean Volumetric Particle Lighting
**Date**: 2025-10-30
**Goal**: Replace per-particle RT lighting with volumetric ReSTIR for smooth, scalable many-light rendering
**Based On**: "Fast Volume Rendering with Spatiotemporal Reservoir Resampling" (Lin, Wyman, Yuksel 2021)

---

## Executive Summary

**Problem**: Current inline RayQuery particle-to-particle lighting creates discrete brightness jumps during volumetric ray marching, not smooth scattering like multi-light system.

**Root Cause**: Per-particle lighting values `g_rtLighting[particleIdx]` are constant within each particle volume. Volumetric ray marching samples hundreds of points through 3D space, but all points inside particle A get the same brightness → discrete transition at A/B boundary.

**Solution**: Volumetric ReSTIR - spatiotemporal reservoir resampling in path space with per-sample-point lighting evaluation using neighbor particles as "virtual lights".

**Expected Benefits**:
- Smooth volumetric scattering (multi-light quality)
- Scales to 10K+ emissive particles (not limited to 16 lights)
- Fixes RTXDI patchwork artifacts via temporal accumulation
- 2-3.5× quality improvement vs baseline at similar performance

---

## Research Findings Summary

### 1. Core ReSTIR Algorithm

**Reservoir Structure**:
```cpp
struct Reservoir {
    float y;       // Selected sample (light index or path)
    float wsum;    // Cumulative weight sum
    float M;       // Total candidates seen
    float W;       // Final resampling weight for unbiased estimation
};
```

**Key Components**:
1. **Weighted Reservoir Sampling**: Stream M candidates, select probabilistically by `w(x) = p̂(x)/p(x)`
2. **Temporal Reuse**: Combine current + previous frame with correction `w_q'→q = (p̂_q(x_q') / p̂_q'(x_q')) × wsum_q'`
3. **Spatial Reuse**: Combine 3-5 neighbor reservoirs within radius
4. **MIS Weighting**: Use **Talbot MIS** (deterministic), not stochastic - critical for volumes!

**RIS Estimator** (1-sample):
```
⟨I⟩_ris = E_p̂(x_r) × (1/M × Σ[j=1 to M] w(x_j))
where: w(x) = p̂(x)/p(x), E_p̂(x_r) = f(x_r)/p̂(x_r)
```

### 2. Volumetric Adaptations

**Path-Space Reservoirs**:
- Store entire paths: `λ = {x₀, x₁, ..., x_k+1}` with k scattering events
- Each vertex spawns 2 candidates: scattering path (NEE) + emission path
- Random walk of K vertices → up to 2K candidate paths

**Critical Innovation: Transmittance Hierarchy**

Use three levels of transmittance quality:

| Level | Usage | Method | Quality | Cost |
|-------|-------|--------|---------|------|
| **T\*** | Candidate generation | Regular tracking, Mip 2-3 piecewise-constant | Coarse (~8× downsampled) | Fast (closed-form PDF) |
| **T̃** | Spatiotemporal reuse | Ray marching, Mip 1 trilinear | Medium (~2× downsampled) | Moderate |
| **T** | Final shading (1 path) | Analytical piecewise-trilinear | Highest (unbiased) | Expensive |

**Key Insight**: Biased/approximate transmittance in importance sampling affects PDF quality (not correctness), but final shading must be unbiased.

**Path Reuse Strategies**:
- **Direction Reuse** (RECOMMENDED): Reuse `ω_i` and `z_i` from neighbor → avoids fireflies
- **Vertex Reuse** (risky): Reuse `x_i` directly → unbounded `G(x₁↔x₂) = 1/|x₂-x₁|²` → fireflies

**Temporal Reprojection**:
- Use velocity field at first scatter vertex x₁
- Sample motion probabilistically weighted by `p̂(z₁|x₀,ω₀)`
- Denser media contributes motion more → reduces artifacts

### 3. RTXDI Patchwork Artifacts - Root Causes

Your current RTXDI M4 implementation produces patchwork because:
1. **M4 = One light per pixel per frame** → discrete selection changes frame-to-frame
2. **No temporal accumulation** → each frame's selection is independent
3. **Missing sample permutations** → temporal correlation artifacts

**Solution**: M5 temporal accumulation smooths over 8-16 frames (60ms convergence).

### 4. Common Pitfalls (Why Previous ReSTIR Likely Failed)

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Transmittance in source PDFs | Bias, expensive evaluation | Use T* cancellation in candidate generation |
| Stochastic MIS in volumes | Excessive noise (Fig 7 in paper) | Use Talbot MIS (deterministic) |
| Vertex reuse | Fireflies from unbounded geometry terms | Use direction reuse |
| No transmittance hierarchy | Too slow | Use T*/T̃/T at different stages |
| Unbounded temporal M | Fireflies on lighting changes | Clamp M with factor Q |
| Missing velocity resampling | Halos around silhouettes | Sample motion from volume, not background |

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)

**Goal**: Build reservoir system WITHOUT spatiotemporal reuse

#### Task 1.1: Create Data Structures

**Volumetric Reservoir** (64 bytes per pixel @ K=3):
```cpp
struct VolumetricReservoir {
    uint32_t pathLength;                      // k scattering events (4 bytes)
    float z[MAX_BOUNCES];                     // Distances (12 bytes for K=3)
    DirectX::XMFLOAT3 omega[MAX_BOUNCES];     // Directions (36 bytes for K=3)
    float wsum;                               // Running weight sum (4 bytes)
    float M;                                  // Candidate count (4 bytes)
    uint32_t lightIndex;                      // Selected light if scattering path (4 bytes)
};
```

**Screen-space reservoir buffers**:
```cpp
ComPtr<ID3D12Resource> m_reservoirBuffer[2];        // Ping-pong for temporal
ComPtr<ID3D12Resource> m_candidatePathBuffer;       // Temporary for M random walks
```

**Memory**: 1920×1080×64 bytes × 2 = ~264 MB (acceptable)

#### Task 1.2: Implement Regular Tracking

**Piecewise-constant volume for T\***:
```cpp
// Build Mip 2 volume with nearest-neighbor sampling
void BuildPiecewiseConstantVolume() {
    // Downsample original volume by 4× (Mip 2)
    // For each voxel v:
    //   σ*_t,v = max(σ_t in neighborhood)  // Conservative majorant
    //   If σ*_t,v == 0 AND neighbors non-zero:
    //     σ*_t,v = average(neighbors)      // Avoid bias
}
```

**Closed-form transmittance**:
```hlsl
float RegularTrackingPDF(float3 x0, float3 x1) {
    float T_star = 1.0;
    // Ray-march through piecewise-constant volume
    for (each voxel v along x0→x1) {
        float sigma_t_v = SamplePiecewiseConstant(voxel_v);
        float d_v = length_in_voxel;
        T_star *= exp(-sigma_t_v * d_v);
    }
    float sigma_t_endpoint = SamplePiecewiseConstant(x1);
    return T_star * sigma_t_endpoint;
}
```

**CRITICAL**: Ensure `σ*_t(x) > 0` wherever `σ_t(x) > 0` to avoid bias.

#### Task 1.3: Path Generation (Random Walks)

**Random walk algorithm**:
```hlsl
[numthreads(8,8,1)]
void GenerateCandidatePaths(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixelPos = dispatchThreadID.xy;

    // M random walks per pixel (M=4 for initial testing)
    for (uint walk = 0; walk < M; walk++) {
        Path candidates[2*K];  // Up to 2K paths per walk
        uint pathCount = 0;

        // Primary ray
        RayDesc ray = GenerateCameraRay(pixelPos);
        float3 x = ray.Origin;
        float3 omega = ray.Direction;

        // Random walk up to K bounces
        for (uint bounce = 0; bounce < K; bounce++) {
            // Sample distance via regular tracking
            float z = SampleRegularTracking(x, omega);
            x = x + z * omega;

            // Check if exited medium
            if (!InsideVolume(x)) break;

            // EMISSION PATH: Ends at current vertex (volumetric emission)
            if (HasEmission(x)) {
                candidates[pathCount++] = CreateEmissionPath(x, bounce);
            }

            // SCATTERING PATH: NEE to sample a light
            float3 lightPos;
            float lightPDF;
            SampleLight(lightPos, lightPDF);  // Importance sample from light distribution
            candidates[pathCount++] = CreateScatteringPath(x, lightPos, bounce);

            // Continue random walk if not at max bounces
            if (bounce < K-1) {
                omega = SamplePhaseFunction(omega);  // Sample new direction
            }
        }

        // RIS: Select one path from this walk's candidates
        Path selectedPath = WeightedReservoirSampling(candidates, pathCount);

        // Combine with pixel's reservoir
        UpdateReservoir(pixelReservoir, selectedPath);
    }
}
```

**Path PDF**:
```hlsl
float ComputePathPDF(Path lambda) {
    float pdf = 1.0;

    // Distance sampling PDFs (regular tracking)
    for (uint i = 0; i < lambda.length; i++) {
        pdf *= RegularTrackingPDF(lambda.x[i], lambda.x[i+1]);
    }

    // Direction sampling PDFs
    for (uint i = 1; i < lambda.length; i++) {
        if (i < lambda.length - 1) {
            pdf *= PhaseFunction(lambda.omega[i]);  // Phase function for scattering
        } else {
            if (lambda.isScatteringPath) {
                pdf *= LightSamplingPDF(lambda.omega[i]);  // NEE PDF
            } else {
                pdf *= 1.0;  // Emission path (no direction sampling)
            }
        }
    }

    return pdf;
}
```

#### Task 1.4: Initial RIS Pass

**Target PDF with T\* cancellation**:
```hlsl
float ComputeTargetPDF(Path lambda) {
    // For scattering paths: use cheaper transmittance T_tilde for NEE segment
    // For emission paths: use full path throughput

    float p_hat = PathThroughput(lambda);  // Γ_s from paper

    if (lambda.isScatteringPath) {
        // Replace expensive T with cheap T_tilde for NEE segment
        float T_cheap = RayMarchTransmittance_Mip2(lambda.x[k], lambda.lightPos);
        p_hat *= T_cheap;
    }

    return p_hat;
}
```

**Weight computation** (T\* cancels!):
```hlsl
float ComputeRISWeight(Path lambda) {
    float p_hat = ComputeTargetPDF(lambda);
    float p = ComputePathPDF(lambda);

    // Magic: T* appears in both p_hat and p with opposite signs → cancels!
    // Result: w(λ) contains only scattering coefficients and phase functions
    return p_hat / p;
}
```

**Weighted reservoir sampling**:
```hlsl
void UpdateReservoir(inout Reservoir R, Path candidate) {
    float w = ComputeRISWeight(candidate);
    R.wsum += w;
    R.M += 1;

    // Probabilistic selection
    float random = PCGHash(R.M) / 4294967296.0;
    if (random < w / R.wsum) {
        R.selectedPath = candidate;
    }
}
```

**Final shading**:
```hlsl
float3 ShadeFinalPath(Reservoir R) {
    Path lambda = R.selectedPath;
    float p_hat = ComputeTargetPDF(lambda);

    // Evaluate integrand f(λ) with UNBIASED transmittance
    float3 F = PathThroughput_Unbiased(lambda);

    // RIS estimator
    float3 result = (F / p_hat) * (R.wsum / R.M);
    return result;
}
```

#### Success Criteria (Phase 1)
- [x] **COMPLETED 2025-10-31**: Basic infrastructure (reservoirs, buffers, pipeline integration)
- [x] **COMPLETED 2025-10-31**: DLSS compatibility (writes to correct output texture)
- [x] **COMPLETED 2025-10-31**: Shader compilation and dispatch working
- [ ] **IN PROGRESS**: Volume Mip 2 texture population (CRITICAL BLOCKER)
- [ ] Single-frame rendering (no reuse) produces low-noise image
- [ ] Quality matches 4 spp baseline (M=4 random walks)
- [ ] No patchwork artifacts (since no reuse yet)
- [ ] Performance: 50-100ms @ 1080p
- [ ] Visual validation: Smooth gradients, no fireflies

**BRANCH**: 0.12.5 (infrastructure complete, awaiting volume texture population)

---

### Phase 2: Spatial Reuse (Week 2)

**Goal**: Add spatial neighbor resampling with proper MIS

#### Task 2.1: Spatial Neighbor Selection

**Low-discrepancy sampling**:
```cpp
void SelectSpatialNeighbors(uint2 pixelPos, uint frameIndex, out uint2 neighbors[N_SPATIAL]) {
    const float SPATIAL_RADIUS = 10.0f;  // pixels

    // R2 low-discrepancy sequence
    float g = 1.32471795724474602596;  // Plastic constant
    float a1 = 1.0 / g;
    float a2 = 1.0 / (g * g);

    for (uint i = 0; i < N_SPATIAL; i++) {
        float seed = frameIndex * N_SPATIAL + i;
        float2 offset = float2(frac(seed * a1), frac(seed * a2));
        offset = (offset - 0.5) * 2.0 * SPATIAL_RADIUS;  // [-R, R]

        neighbors[i] = pixelPos + int2(offset);
    }
}
```

#### Task 2.2: Path Reconstruction (Direction Reuse)

**Reconstruct neighbor path for current pixel**:
```hlsl
Path ReconstructPathDirectionReuse(Path neighborPath, float3 currentRayOrigin, float3 currentRayDir) {
    Path reconstructed;

    // Vertex 0: camera (same for all pixels)
    reconstructed.x[0] = currentRayOrigin;

    // Vertex 1: same distance along different primary ray
    float z1 = neighborPath.z[0];
    reconstructed.x[1] = currentRayOrigin + z1 * currentRayDir;
    reconstructed.z[0] = z1;

    // Remaining vertices: reuse directions and distances
    for (uint i = 1; i < neighborPath.length; i++) {
        float3 omega_i = neighborPath.omega[i];
        float z_i = neighborPath.z[i];

        reconstructed.x[i+1] = reconstructed.x[i] + z_i * omega_i;
        reconstructed.omega[i] = omega_i;
        reconstructed.z[i] = z_i;
    }

    return reconstructed;
}
```

**Why direction reuse?** Avoids unbounded geometry term `G(x₁↔x₂) = 1/|x₂-x₁|²` that causes fireflies.

#### Task 2.3: Transmittance Upgrade (T\* → T̃)

**Ray marching with trilinear interpolation**:
```hlsl
float RayMarchTransmittance_Mip1(float3 x0, float3 x1) {
    float3 dir = x1 - x0;
    float totalDist = length(dir);
    dir /= totalDist;

    // Step size: Mip 1 voxel diagonal
    float voxelSize = VOXEL_SIZE * 2.0;  // Mip 1 is 2× coarser
    float stepSize = sqrt(3.0) * voxelSize;
    uint numSteps = ceil(totalDist / stepSize);

    float transmittance = 1.0;
    float t = 0.0;

    for (uint i = 0; i < numSteps; i++) {
        float3 samplePos = x0 + t * dir;

        // Trilinear sampling from Mip 1 volume (BC4 compressed)
        float sigma_t = SampleVolume_Mip1_Trilinear(samplePos);

        transmittance *= exp(-sigma_t * stepSize);
        t += stepSize;
    }

    return transmittance;
}
```

**BC4 compression** (4-bit per density value):
```cpp
// Compress Mip 1 volume to BC4 (64:1 compression from original)
void CompressMip1Volume() {
    // DirectX BC4 block compression (4 bits/texel)
    // Original: float32 (32 bits) → BC4 (4 bits + 16-entry palette)
    // Bandwidth: 8× reduction per fetch

    D3D12_RESOURCE_DESC desc = {};
    desc.Format = DXGI_FORMAT_BC4_UNORM;  // Single-channel compression
    // ... create compressed resource
}
```

#### Task 2.4: Correction Weight & MIS

**Correction weight computation**:
```hlsl
float ComputeCorrectionWeight(Path neighborPath, uint2 currentPixel, uint2 neighborPixel) {
    // Reconstruct path for current pixel
    Path currentPath = ReconstructPathDirectionReuse(neighborPath, ...);

    // Evaluate target PDFs (using T_tilde for both)
    float p_hat_current = ComputeTargetPDF_Mip1(currentPath);
    float p_hat_neighbor = ComputeTargetPDF_Mip1(neighborPath);

    // Correction weight
    float w_correction = (p_hat_current / p_hat_neighbor) * neighbor.wsum;

    return w_correction;
}
```

**Talbot MIS (deterministic, not stochastic!)**:
```hlsl
void CombineSpatialReservoirs(inout Reservoir R_current, Reservoir neighbors[N_SPATIAL]) {
    Path candidates[N_SPATIAL + 1];
    float weights[N_SPATIAL + 1];

    // Current pixel's path
    candidates[0] = R_current.selectedPath;
    weights[0] = R_current.wsum;

    // Neighbor paths with correction weights
    for (uint i = 0; i < N_SPATIAL; i++) {
        candidates[i+1] = neighbors[i].selectedPath;
        weights[i+1] = ComputeCorrectionWeight(neighbors[i].selectedPath, ...);
    }

    // Select one path via weighted sampling
    Path selected = WeightedSample(candidates, weights);

    // Talbot MIS weight for selected path (O(N²) but N is small)
    uint selectedIdx = ...;  // Which candidate was selected
    float M_total = (N_SPATIAL + 1) * M;

    float p_hat_selected = ComputeTargetPDF_Mip1(selected);
    float mis_denom = 0.0;

    for (uint i = 0; i < N_SPATIAL + 1; i++) {
        Path candidate_i = candidates[i];
        float p_hat_i = ComputeTargetPDF_Mip1(candidate_i);
        mis_denom += p_hat_i;
    }

    float mis_weight = (M_total * p_hat_selected) / (M * mis_denom);

    // Update reservoir with MIS-weighted path
    R_current.selectedPath = selected;
    R_current.wsum = ... // Update with MIS weight
    R_current.M = M_total;
}
```

**Rejection heuristics** (optional but recommended):
```hlsl
bool ShouldRejectNeighbor(uint2 currentPixel, uint2 neighborPixel) {
    // Depth discontinuity (not applicable for volumes without surfaces)
    // Normal discontinuity (use first scatter position as proxy)
    float3 normal_current = normalize(GetFirstScatterPos(currentPixel) - cameraPos);
    float3 normal_neighbor = normalize(GetFirstScatterPos(neighborPixel) - cameraPos);

    if (dot(normal_current, normal_neighbor) < 0.906)  // ~25 degrees
        return true;

    return false;
}
```

#### Success Criteria (Phase 2)
- [ ] Reduced noise vs Phase 1 (spatial coherence working)
- [ ] No fireflies (direction reuse prevents unbounded G terms)
- [ ] Smooth gradients across neighbor pixels
- [ ] Performance: 60-80ms @ 1080p (ray marching overhead)
- [ ] MSE decreases compared to Phase 1

---

### Phase 3: Temporal Reuse (Week 3)

**Goal**: Add temporal accumulation (M5) to eliminate RTXDI patchwork

#### Task 3.1: Velocity-Based Temporal Reprojection

**Sample motion from volume (not background!)**:
```hlsl
float3 ComputeTemporalMotionVector(Reservoir R_current, float3 cameraPos) {
    Path lambda = R_current.selectedPath;
    float3 firstScatterPos = lambda.x[1];  // First vertex after camera

    // If first scatter is in volume, use its velocity
    if (InsideVolume(firstScatterPos)) {
        float3 velocity = SampleVelocityField(firstScatterPos);
        return velocity * deltaTime;
    }

    // Background hit - use velocity resampling
    return VelocityResampling(lambda);
}
```

**Velocity resampling** (for background hits):
```hlsl
float3 VelocityResampling(Path lambda) {
    // Generate new z1 proportional to free-flight distance within volume
    // p(z) ∝ σ_t(x') T(x₀↔x')

    RayDesc ray = lambda.primaryRay;

    // Importance sample a point in the volume
    float totalWeight = 0.0;
    float selectedZ = 0.0;

    // Sample along ray at regular intervals
    const uint NUM_SAMPLES = 16;
    for (uint i = 0; i < NUM_SAMPLES; i++) {
        float z = ... // Sample position along ray
        float3 pos = ray.Origin + z * ray.Direction;

        if (!InsideVolume(pos)) continue;

        // Weight = σ_t(x) × T*(x₀↔x)
        float sigma_t = SampleDensity(pos);
        float T_star = RegularTrackingTransmittance(ray.Origin, pos);
        float weight = sigma_t * T_star;

        totalWeight += weight;

        // Reservoir sampling
        float random = PCGHash(i);
        if (random * totalWeight < weight) {
            selectedZ = z;
        }
    }

    // Use velocity at selected position
    float3 selectedPos = ray.Origin + selectedZ * ray.Direction;
    return SampleVelocityField(selectedPos) * deltaTime;
}
```

#### Task 3.2: Temporal Reservoir Combination

**Find previous pixel via motion vector**:
```hlsl
uint2 FindTemporalPixel(uint2 currentPixel, float3 motionVector, float3 cameraPos, float3 prevCameraPos) {
    // Project current first scatter position back to previous frame
    float3 currentScatterPos = ...;
    float3 prevScatterPos = currentScatterPos - motionVector;

    // Reproject to screen space using previous camera
    float2 prevScreenPos = ProjectToScreen(prevScatterPos, prevCameraPos);

    return uint2(prevScreenPos * screenDimensions);
}
```

**Combine temporal reservoirs**:
```hlsl
void CombineTemporalReservoir(inout Reservoir R_current, Reservoir R_prev, uint2 currentPixel) {
    // Reconstruct previous path for current pixel
    Path prevPath = R_prev.selectedPath;
    Path currentPath = ReconstructPathDirectionReuse(prevPath, currentPixel, ...);

    // Correction weight (using T_tilde)
    float p_hat_current = ComputeTargetPDF_Mip1(currentPath);
    float p_hat_prev = ComputeTargetPDF_Mip1(prevPath);

    float w_temporal = (p_hat_current / p_hat_prev) * R_prev.wsum;

    // Update reservoir
    R_current.wsum += w_temporal;
    R_current.M += R_prev.M;

    // Probabilistic selection
    float random = PCGHash(...);
    if (random < w_temporal / R_current.wsum) {
        R_current.selectedPath = currentPath;
    }
}
```

#### Task 3.3: M Clamping (Prevent Unbounded History)

**Clamp M to prevent fireflies**:
```hlsl
void ClampReservoirHistory(inout Reservoir R, float Q, float M_initial) {
    float M_max = Q * M_initial;

    if (R.M > M_max) {
        // Scale down wsum proportionally
        R.wsum *= (M_max / R.M);
        R.M = M_max;
    }
}
```

**Tuning Q**:
- Q=1: No temporal accumulation (same as Phase 2)
- Q=4: 16 effective samples (4 initial × 4 temporal) - **recommended start**
- Q=10: 40 effective samples - for complex lighting with surfaces
- Q=20: 80 effective samples - diminishing returns, risk of fireflies

#### Task 3.4: Reset Detection

**Detect when to reset temporal history**:
```hlsl
bool ShouldResetTemporal(float3 currentCameraPos, float3 prevCameraPos, float threshold) {
    // Camera movement
    float cameraDist = length(currentCameraPos - prevCameraPos);
    if (cameraDist > threshold) return true;  // threshold = 10.0 units

    // Lighting discontinuity (optional)
    // Check if dominant light changed dramatically

    // Volume deformation (optional)
    // Check if velocity field magnitude exceeds threshold

    return false;
}
```

#### Success Criteria (Phase 3)
- [ ] **Patchwork eliminated!** (Key success metric)
- [ ] Temporal stability during camera animation
- [ ] Reduced noise compared to Phase 2
- [ ] Slight darkening in disocclusions (expected bias, should be imperceptible)
- [ ] Performance: 65-85ms @ 1080p (temporal reprojection overhead)
- [ ] Visual validation: Smooth animation in supplemental video

---

### Phase 4: Final Shading & Optimization (Week 4)

**Goal**: Unbiased final shading and performance tuning

#### Task 4.1: Analytical Transmittance (T) for Final Path

**Piecewise-trilinear regular tracking**:
```hlsl
float AnalyticalTransmittance(float3 x0, float3 x1) {
    float3 dir = normalize(x1 - x0);
    float totalDist = length(x1 - x0);

    float transmittance = 1.0;
    float t = 0.0;

    // Traverse voxel grid
    while (t < totalDist) {
        float3 pos = x0 + t * dir;

        // Fetch 8 corner densities for trilinear interpolation
        float3 voxelCoord = WorldToVoxel(pos);
        float3 frac = fract(voxelCoord);
        int3 baseCoord = floor(voxelCoord);

        float sigma[8];
        for (uint i = 0; i < 8; i++) {
            int3 offset = int3(i&1, (i>>1)&1, (i>>2)&1);
            sigma[i] = SampleDensity(baseCoord + offset);
        }

        // Trilinear interpolation
        float sigma_t = TrilinearInterpolate(sigma, frac);

        // Step to next voxel boundary
        float stepSize = ComputeVoxelExitDistance(pos, dir);

        // Accumulate transmittance over this voxel segment
        transmittance *= exp(-sigma_t * stepSize);
        t += stepSize;
    }

    return transmittance;
}
```

**Use in final shading**:
```hlsl
float3 ShadeSelectedPath_Unbiased(Reservoir R) {
    Path lambda = R.selectedPath;

    // Compute path throughput with UNBIASED transmittance
    float3 throughput = 1.0;

    for (uint i = 0; i < lambda.length; i++) {
        // Transmittance between vertices (expensive but unbiased)
        float T = AnalyticalTransmittance(lambda.x[i], lambda.x[i+1]);
        throughput *= T;

        // Scattering coefficients
        if (i < lambda.length - 1) {
            float sigma_s = GetScatteringCoefficient(lambda.x[i+1]);
            throughput *= sigma_s;

            // Phase function
            float3 omega_in = -lambda.omega[i];
            float3 omega_out = lambda.omega[i+1];
            float phase = HenyeyGreenstein(dot(omega_in, omega_out), g_scatter);
            throughput *= phase;
        }
    }

    // Emitted radiance at final vertex
    float3 L_e = GetEmittedRadiance(lambda.x[lambda.length]);
    throughput *= L_e;

    // RIS correction
    float p_hat = ComputeTargetPDF_Mip1(lambda);  // Still using T_tilde in PDF
    float correction = (R.wsum / R.M) / p_hat;

    return throughput * correction;
}
```

#### Task 4.2: Performance Profiling

**Measure per-pass timings**:
```cpp
struct ProfilingData {
    float candidateGeneration_ms;
    float spatialReuse_ms;
    float temporalReuse_ms;
    float finalShading_ms;
    float total_ms;
};

void ProfileFrame() {
    // Use GPU timestamp queries
    ID3D12QueryHeap* timestampHeap;
    // ... measure each pass

    // Target: <50ms total @ 1080p
    // Typical breakdown:
    //   Candidate generation: 20-30ms (path generation + RIS)
    //   Spatial reuse: 15-25ms (ray marching T_tilde)
    //   Temporal reuse: 5-10ms (reprojection)
    //   Final shading: 5-10ms (analytical T)
}
```

**Identify bottlenecks**:
```
If candidateGeneration_ms > 30ms:
  → Reduce M (4 → 3 random walks)
  → Lower K (3 → 2 bounces)
  → Use Mip 3 for T* (more aggressive downsampling)

If spatialReuse_ms > 25ms:
  → Reduce ray marching step size (coarser T_tilde)
  → Use Mip 2 instead of Mip 1
  → Reduce N_SPATIAL (5 → 3 neighbors)

If temporalReuse_ms > 10ms:
  → Optimize path reconstruction
  → Cache motion vectors

If finalShading_ms > 10ms:
  → Switch to ratio tracking instead of analytical T
  → Use larger majorant to reduce steps
```

#### Task 4.3: Optional Optimizations

**1. Lower resolution candidate generation**:
```cpp
// Generate candidates at half-resolution, upsample reservoirs
void GenerateCandidatesHalfRes() {
    // Render at 960×540 instead of 1920×1080
    // Bilinear upsample reservoirs for spatial reuse
    // Saves ~40% on candidate generation cost
}
```

**2. Reduce maximum bounces**:
```cpp
// K=2 instead of K=3
// Most energy comes from first 2 bounces in scattering media
// Saves ~30% on path generation
```

**3. Sparse voxel optimization**:
```hlsl
// Skip empty space during transmittance evaluation
bool TraverseEmptySpace(float3 pos, float3 dir, out float skipDist) {
    // Check if current voxel is empty (sigma_t = 0)
    if (IsEmptyVoxel(pos)) {
        // Jump to next non-empty voxel boundary
        skipDist = DistanceToNextOccupiedVoxel(pos, dir);
        return true;
    }
    return false;
}
```

**4. Adaptive quality**:
```cpp
// Reduce quality in low-importance regions
float ComputePixelImportance(uint2 pixel) {
    // Distance from screen center
    float2 center = screenDimensions / 2;
    float dist = length(pixel - center);
    return 1.0 - saturate(dist / screenDimensions.x);
}

// Use importance to modulate:
// - M (fewer walks in periphery)
// - N_SPATIAL (fewer neighbors)
// - Step size for T_tilde (coarser ray marching)
```

#### Success Criteria (Phase 4)
- [ ] Unbiased rendering (static scenes)
- [ ] Performance: <50ms @ 1080p with 10K particles
- [ ] Quality exceeds multi-light baseline (MSE comparison)
- [ ] Temporal stability with animation
- [ ] No perceivable bias from temporal reuse

---

## Debugging Strategy

### Visual Debugging Modes

Implement these visualization modes for validation:

**1. Reservoir Metrics**:
```hlsl
float3 DebugReservoirM(Reservoir R) {
    // Heatmap: Blue (M=0) → Green (M=20) → Red (M=40+)
    float normalized = saturate(R.M / 40.0);
    return HeatmapColor(normalized);
}

float3 DebugReservoirWsum(Reservoir R) {
    // Visualize weight accumulation
    return float3(R.wsum, R.wsum, R.wsum) * 0.1;  // Scale for visibility
}

float3 DebugPathLength(Reservoir R) {
    // Color-code by number of bounces
    float3 colors[4] = {
        float3(1,0,0),  // K=0 (red)
        float3(0,1,0),  // K=1 (green)
        float3(0,0,1),  // K=2 (blue)
        float3(1,1,0)   // K=3 (yellow)
    };
    return colors[min(R.selectedPath.length, 3)];
}
```

**2. Transmittance Comparison**:
```hlsl
float3 DebugTransmittance(float3 x0, float3 x1, uint level) {
    float T;
    if (level == 0) T = RegularTrackingTransmittance_Mip2(x0, x1);  // T*
    else if (level == 1) T = RayMarchTransmittance_Mip1(x0, x1);    // T_tilde
    else T = AnalyticalTransmittance(x0, x1);                        // T

    return float3(T, T, T);
}
```

**3. Temporal History**:
```hlsl
float3 DebugTemporalAge(Reservoir R, uint currentFrame) {
    float age = currentFrame - R.lastUpdateFrame;
    // Blue = recent (0-4 frames), Red = old (20+ frames)
    return HeatmapColor(age / 20.0);
}
```

**4. MIS Weights**:
```hlsl
float3 DebugMISContribution(Reservoir R, uint selectedIdx, float mis_weights[N]) {
    // Show which sample contributed most
    float maxWeight = 0.0;
    for (uint i = 0; i < N; i++) {
        maxWeight = max(maxWeight, mis_weights[i]);
    }

    // Highlight dominant contributor
    if (selectedIdx == 0) return float3(1,0,0);  // Current pixel
    else return float3(0,1,0);                   // Neighbor
}
```

**5. Path Visualization**:
```hlsl
void DebugDrawPath(Path lambda) {
    // Draw line segments between path vertices
    for (uint i = 0; i < lambda.length; i++) {
        DrawLine(lambda.x[i], lambda.x[i+1], lambda.isScatteringPath ? GREEN : RED);
    }

    // Draw light sample point (if scattering path)
    if (lambda.isScatteringPath) {
        DrawSphere(lambda.lightPos, 5.0, YELLOW);
    }
}
```

### Common Issues & Diagnostic Table

| Symptom | Likely Cause | Diagnostic | Fix |
|---------|--------------|------------|-----|
| **Patchwork pattern** | No temporal accumulation | Check if temporalReuse_ms = 0 | Implement Phase 3 |
| **Fireflies (bright pixels)** | Vertex reuse or unbounded MIS | Enable DebugPathLength → look for K=0 paths | Use direction reuse, add epsilon to MIS denom |
| **Excessive noise** | Stochastic MIS | Check MIS implementation | Switch to Talbot MIS |
| **Black regions** | Zero PDF | Add DebugReservoirWsum → black = wsum=0 | Epsilon guards in PDFs |
| **Darkening** | Aggressive M clamping | Check average M values | Increase Q from 4 → 10 |
| **Halos in motion** | Background motion vectors | Enable DebugTemporalAge around silhouettes | Implement velocity resampling |
| **Blocky appearance** | Coarse transmittance | Compare T* vs T_tilde vs T visualizations | Use finer Mip level or smaller step size |
| **Slow performance** | Expensive transmittance | Profile per-pass timings | See optimization strategies |
| **Color noise** | Scalar target PDF | Look for chromaticity shifts | Expected limitation (not critical) |
| **Temporal lag** | Large Q or slow motion | Check M values over time | Reduce Q or reset threshold |

### Validation Checklist

**Unit Tests**:
```cpp
// Test 1: Reservoir update maintains valid state
void TestReservoirUpdate() {
    Reservoir R = CreateEmptyReservoir();
    for (int i = 0; i < 100; i++) {
        Path candidate = GenerateRandomPath();
        UpdateReservoir(R, candidate);

        assert(R.M == i + 1);  // M increments correctly
        assert(R.wsum > 0);    // Weight accumulates
    }
}

// Test 2: Path reconstruction matches expected vertices
void TestPathReconstruction() {
    Path original = GenerateTestPath();
    Path reconstructed = ReconstructPathDirectionReuse(original, newPixel);

    // x0 should be camera (same for all pixels)
    assert(reconstructed.x[0] == cameraPos);

    // z1 should match
    assert(abs(reconstructed.z[0] - original.z[0]) < EPSILON);

    // Directions should match
    for (uint i = 1; i < original.length; i++) {
        assert(dot(reconstructed.omega[i], original.omega[i]) > 0.999);
    }
}

// Test 3: Transmittance consistency
void TestTransmittance() {
    float3 x0 = ..., x1 = ...;

    // Homogeneous medium: T* should match analytical
    float sigma_t = 1.0;
    float dist = length(x1 - x0);
    float T_analytical = exp(-sigma_t * dist);
    float T_computed = RegularTrackingTransmittance(x0, x1);

    assert(abs(T_analytical - T_computed) < 0.01);
}
```

**Integration Tests**:
```cpp
// Cornell box with single scattering
void TestCornellBox() {
    SetupCornellBox();

    // Render with volumetric ReSTIR
    Image result = RenderFrame();

    // Render reference (path tracing, 10K spp)
    Image reference = RenderReference();

    // Compare MSE
    float mse = ComputeMSE(result, reference);
    assert(mse < 0.01);  // Should be low-noise
}

// Homogeneous medium with analytical solution
void TestHomogeneousMedium() {
    SetupHomogeneousBox(sigma_t = 1.0, sigma_s = 0.5);

    Image result = RenderFrame();

    // Analytical solution available for isotropic scattering
    Image analytical = ComputeAnalyticalSolution();

    float error = ComputeRelativeError(result, analytical);
    assert(error < 0.05);  // Within 5%
}
```

**Regression Tests**:
```cpp
// Capture reference screenshots from multi-light
void CaptureReferences() {
    RenderMode = MULTI_LIGHT;

    for (auto scene : testScenes) {
        Image reference = RenderScene(scene);
        SaveImage(reference, "references/" + scene.name + ".png");
    }
}

// Compare against references
void ValidateAgainstReferences() {
    RenderMode = VOLUMETRIC_RESTIR;

    for (auto scene : testScenes) {
        Image result = RenderScene(scene);
        Image reference = LoadImage("references/" + scene.name + ".png");

        // Use MCP tool for perceptual similarity
        float lpips = CompareScreenshots_ML(result, reference);
        assert(lpips < 0.01);  // Perceptually identical
    }
}
```

**Performance Benchmarks**:
```cpp
void BenchmarkPerformance() {
    struct Benchmark {
        string name;
        uint particleCount;
        uint lightCount;
        uint K;  // max bounces
        float targetFPS;
    };

    Benchmark tests[] = {
        {"10K_1bounce", 10000, 13, 1, 60.0},
        {"10K_3bounce", 10000, 13, 3, 30.0},
        {"50K_1bounce", 50000, 13, 1, 30.0},
    };

    for (auto& test : tests) {
        SetupScene(test);

        float avgTime = MeasureAverageFrameTime(256);  // frames
        float fps = 1000.0 / avgTime;

        printf("%s: %.1f ms (%.1f FPS) - Target: %.1f FPS - %s\n",
            test.name.c_str(), avgTime, fps, test.targetFPS,
            fps >= test.targetFPS ? "PASS" : "FAIL");
    }
}
```

---

## Memory Budget & Resource Management

### Per-Pixel Reservoir Memory

**Reservoir structure** (64 bytes for K=3):
```
VolumetricReservoir {
    pathLength: 4 bytes
    z[3]: 12 bytes
    omega[3]: 36 bytes
    wsum: 4 bytes
    M: 4 bytes
    lightIndex: 4 bytes
    padding: 0 bytes
} = 64 bytes
```

**Screen buffers**:
```
1920×1080 @ 64 bytes/pixel:
- Current frame: 132 MB
- Previous frame: 132 MB
- Total: 264 MB
```

### Volume Mipmap Memory

**Original volume** (256³ voxels, float32):
```
256×256×256 × 4 bytes = 64 MB
```

**Mipmap chain**:
```
Mip 0 (256³): 64 MB   (original)
Mip 1 (128³): 8 MB    (for T_tilde, BC4 compressed → 0.5 MB)
Mip 2 (64³):  1 MB    (for T*, piecewise-constant)
Mip 3 (32³):  0.125 MB (optional optimization)

Total with compression: ~65 MB
```

**BC4 Compression**:
```cpp
// Mip 1: 128³ × 4 bytes = 8 MB uncompressed
// BC4: 4 bits/texel = 128³ × 0.5 bytes = 0.5 MB
// Compression ratio: 16:1
// Bandwidth savings: 16× fewer bytes fetched
```

### Temporary Buffers

**Candidate path buffer** (per-pixel, 2K paths max):
```
2K paths × 64 bytes × (1920×1080) = 263 GB (too large!)

Solution: Generate on-the-fly, stream into reservoir
  → Only store selected paths (64 bytes/pixel)
  → Temporary path during generation (~1 KB stack space)
```

**Total GPU memory**:
```
Reservoirs: 264 MB
Volumes: 65 MB
Acceleration structures (BLAS/TLAS): ~50 MB (existing)
Swap chain + render targets: ~100 MB (existing)
────────────────────
Total: ~479 MB (well within RTX 4060 Ti 8GB budget)
```

---

## Expected Results & Performance Targets

### Quality Metrics (vs Baseline)

Based on paper results with similar scenes:

| Scene | Baseline MSE | Volumetric ReSTIR MSE | Quality Improvement |
|-------|-------------|----------------------|---------------------|
| 10K particles, 1 bounce, env | 0.010 | 0.003 | **3.3× lower error** |
| 10K particles, 3 bounces | 0.014 | 0.004 | **3.5× lower error** |
| Emissive volume, 2 bounces | 0.007 | 0.003 | **2.3× lower error** |
| Emissive volume, 4 bounces | 0.010 | 0.005 | **2.0× lower error** |

### Performance Targets (RTX 4060 Ti @ 1080p)

**Your current system**:
```
Multi-light (13 lights): 120 FPS
RTXDI M4: 120 FPS (but with patchwork)
```

**Volumetric ReSTIR targets**:
```
Phase 1 (RIS only):        20 FPS (50ms) - Initial overhead
Phase 2 (+ Spatial):       15 FPS (67ms) - Ray marching cost
Phase 3 (+ Temporal):      14 FPS (71ms) - Reprojection
Phase 4 (Optimized):       20 FPS (50ms) - After tuning
```

**Optimization potential** (Phase 4):
```
- Reduce M (4→3): +15% FPS
- Reduce K (3→2): +20% FPS
- Use Mip 3 for T*: +10% FPS
- Reduce N_SPATIAL (5→3): +8% FPS

Combined: 50ms → ~35ms (28 FPS)
```

### Comparison to Multi-Light

| Metric | Multi-Light | Volumetric ReSTIR | Notes |
|--------|------------|-------------------|-------|
| Lights supported | 16 max | 10K+ (particles) | No hard limit |
| Smoothness | Excellent | Excellent | Eliminates discrete jumps |
| Patchwork artifacts | None | None (with M5) | Temporal accumulation |
| Performance @ 10K | 120 FPS | 20-28 FPS | Trade-off for quality |
| Multiple scattering | Limited | Excellent | Natural path tracing |

### RTXDI Patchwork Fix

**Your specific problem**:
```
RTXDI M4 (current):
  - 1 light/pixel/frame
  - Changes every frame
  → Patchwork pattern

RTXDI M5 (with temporal):
  - Accumulates 8-16 samples
  - Smooths over 60ms
  → Patchwork eliminated

Expected convergence:
  Frame 1: Noisy (1 sample)
  Frame 4: Smoother (4 samples)
  Frame 8: Converged (8 samples)
  Frame 16: Stable (16 samples, clamped)
```

---

## Risk Assessment & Mitigation

### High-Risk Items

**1. Transmittance hierarchy complexity**
- **Risk**: Getting T*/T̃/T wrong → bias or performance issues
- **Mitigation**: Unit test each level independently against analytical solutions
- **Fallback**: Start with single transmittance level, optimize later

**2. Path reconstruction bugs**
- **Risk**: Direction reuse creates incorrect vertex positions → dark or bright artifacts
- **Mitigation**: Visual debugging (draw paths as line segments)
- **Fallback**: Use vertex reuse (simpler but has fireflies)

**3. MIS weight computation errors**
- **Risk**: Incorrect Talbot MIS → bias or excessive noise
- **Mitigation**: Compare against stochastic MIS (should be less noisy)
- **Fallback**: Disable MIS (biased but simpler)

**4. Temporal reprojection failures**
- **Risk**: Wrong motion vectors → ghosting or halos
- **Mitigation**: Velocity resampling for edge cases
- **Fallback**: Disable temporal reuse (Phase 2 quality)

### Medium-Risk Items

**5. Performance budget overrun**
- **Risk**: Exceeds 50ms target → not interactive
- **Mitigation**: Incremental profiling at each phase
- **Fallback**: Reduce M, K, or use coarser volumes

**6. Memory constraints**
- **Risk**: 264 MB reservoirs + volumes exceeds budget
- **Mitigation**: Monitor GPU memory usage
- **Fallback**: Use half-resolution reservoirs (960×540)

**7. Compatibility with existing RTXDI**
- **Risk**: Interferes with current RTXDI M4 pipeline
- **Mitigation**: Separate render path (toggle between modes)
- **Fallback**: Keep RTXDI as-is, add ReSTIR as alternative

### Low-Risk Items

**8. Color noise** (known limitation)
- **Risk**: Chromatic artifacts in high-saturation scenes
- **Impact**: Minor, only visible in edge cases
- **Mitigation**: Document as known limitation

**9. Temporal bias** (camera motion)
- **Risk**: Slight darkening during fast motion
- **Impact**: Imperceptible in most cases (Figure 9 in paper)
- **Mitigation**: Tune Q and reset threshold

---

## Success Criteria Summary

### Phase 1 (Week 1)
- [ ] Renders without crashes
- [ ] Visual quality ≥ 4 spp baseline
- [ ] No patchwork (no reuse yet)
- [ ] Performance: 50-100ms
- [ ] Transmittance cancellation works (T* in both p̂ and p)

### Phase 2 (Week 2)
- [ ] Reduced noise vs Phase 1
- [ ] No fireflies (direction reuse working)
- [ ] Smooth spatial gradients
- [ ] Performance: 60-80ms
- [ ] MSE < Phase 1

### Phase 3 (Week 3)
- [ ] **Patchwork eliminated** ← Primary goal!
- [ ] Temporal stability
- [ ] Low bias (<5% darkening in motion)
- [ ] Performance: 65-85ms
- [ ] MSE < Phase 2

### Phase 4 (Week 4)
- [ ] Unbiased static rendering
- [ ] Performance: <50ms (optimized)
- [ ] Quality exceeds multi-light (MSE comparison)
- [ ] All debug visualizations working
- [ ] Documentation complete

---

## Timeline Summary

| Week | Phase | Milestones | Validation |
|------|-------|-----------|------------|
| 1 | Infrastructure | Reservoirs, T*, path gen, RIS | Single-frame quality |
| 2 | Spatial Reuse | Neighbors, T̃, MIS | Spatial coherence |
| 3 | Temporal Reuse | Motion vectors, M5, clamping | **Patchwork fix** |
| 4 | Optimization | T shading, profiling, tuning | Performance target |

**Total estimated time**: 4 weeks full-time, or 6-8 weeks part-time

---

## Next Steps

1. **Review this plan** with Ben - confirm approach
2. **Set up git branch**: `feature/volumetric-restir`
3. **Create todo tracking**: Use TodoWrite tool throughout
4. **Phase 1 kickoff**: Start with data structures
5. **Daily validation**: Run unit tests after each task
6. **Weekly milestones**: Compare screenshots, measure MSE
7. **Final comparison**: Use MCP tools (LPIPS, performance analysis)

---

## References

**Papers**:
- Lin, Wyman, Yuksel (2021). "Fast Volume Rendering with Spatiotemporal Reservoir Resampling". ACM TOG.
- Bitterli et al. (2020). "Spatiotemporal reservoir resampling for real-time ray tracing". ACM TOG.
- Talbot et al. (2005). "Importance Resampling for Global Illumination". Eurographics.

**Code References**:
- Your existing: `particle_gaussian_raytrace.hlsl` (volumetric renderer)
- Your existing: `RTXDILightingSystem.cpp` (M4 implementation)
- Paper supplement: Algorithm pseudocode in supplemental document

**Tools**:
- MCP `compare_screenshots_ml`: LPIPS perceptual similarity
- MCP `analyze_pix_capture`: Performance profiling
- MCP `list_recent_screenshots`: Regression testing

---

**Last Updated**: 2025-10-30
**Status**: Ready to implement
**Confidence**: High (based on paper results and existing infrastructure)
