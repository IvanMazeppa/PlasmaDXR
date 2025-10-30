# Volumetric RT Scattering Solutions (RT-first)

Date: 2025-10-30  
Context: See `LIGHTING_PROBLEM_EXPLANATION.md` for the spatial sampling mismatch with inline RayQuery per-particle lighting.

Goal: Replicate multi-light smooth volumetric scattering for ~10K 3D Gaussian particles using ray tracing at scale.

---

## 1) Volumetric ReSTIR (RTXDI-style per-froxel reservoirs)
- Score: 10/10
- What it delivers: Multi-light smoothness at scale by sampling “virtual lights” (nearby particles) per froxel with spatiotemporal reuse. Reuses your RTXDI/grid foundations.

How it works:
- Partition the world into a 3D froxel grid (e.g., 64×64×64 over the active volume).
- For each froxel, maintain a small reservoir (1–4 entries) of nearby “virtual lights” drawn from particles (importance ∝ candidateIntensity × phase × 1/d²).
- Use temporal and spatial reuse (à la RTXDI M4/M5). During ray-march, each sample point fetches 1–2 lights from its froxel’s reservoir and evaluates single scattering (attenuation + Henyey–Greenstein).

Data layout (GPU):
```cpp
// One reservoir entry per candidate light
struct ReservoirLight {
    float3 position;     // particle position (acts as light position)
    float  intensity;    // scalar intensity (see notes below)
    float3 color;        // RGB color (derive from temperature, or white if RT-only)
    float  weight;       // reservoir weight (W)
    uint   particleIndex;// optional: for debug/validation
    uint   _pad0, _pad1, _pad2;
};

// Froxel atlas: packed linear buffer with per-froxel spans
struct FroxelSpan {
    uint  offset;  // start into g_froxelReservoirs
    uint  count;   // number of valid entries (0..K)
    uint  _pad0, _pad1;
};

RWStructuredBuffer<ReservoirLight> g_froxelReservoirs; // size = FroxelCount * K
RWStructuredBuffer<FroxelSpan>    g_froxelSpans;       // size = FroxelCount
```

Build pass (compute):
```hlsl
// Pseudocode: build reservoirs per froxel using reservoir sampling
[numthreads(8,8,8)]
void BuildFroxelReservoirs(uint3 froxelId : SV_DispatchThreadID) {
    uint froxelIndex = Flatten(froxelId);
    Reservoir res[K]; // small local array

    // Sample candidates: particles from spatial hash / grid cells near the froxel
    for (each candidate particle C in neighborCells) {
        float3 toC  = C.position - FroxelCenter(froxelId);
        float  dist = length(toC);
        float  phase= HG(dot(normalize(-viewDir), normalize(toC)), g); // or view-independent pre-pass

        // Intensity can be:
        //  - from RT external lighting estimate per particle (if available), OR
        //  - from a heuristic (temperature→luminance) if running RT-only exploration
        float  intensity = EstimateParticleIntensity(C);
        float  weight    = intensity * phase / (1.0 + dist*dist);

        ReservoirUpdate(res, C, weight); // standard ReSTIR update
    }

    // Write winners to global buffers
    uint base = froxelIndex * K;
    g_froxelSpans[froxelIndex] = (FroxelSpan) { base, K };
    [unroll]
    for (uint i=0;i<K;i++) g_froxelReservoirs[base+i] = ToReservoirLight(res[i]);
}
```

Shading glue (`shaders/particles/particle_gaussian_raytrace.hlsl`):
```hlsl
float3 ShadeFromFroxelReservoir(float3 samplePos, float3 viewDir) {
    uint froxelIndex = ComputeFroxelIndex(samplePos);
    FroxelSpan span  = g_froxelSpans[froxelIndex];

    float3 total = 0;
    [loop]
    for (uint i=0;i<span.count;i++) {
        ReservoirLight L = g_froxelReservoirs[span.offset + i];
        float3  ldir = normalize(L.position - samplePos);
        float   dist = length(L.position - samplePos);
        float   atten = 1.0 / (1.0 + (dist*dist));
        float   phase = HenyeyGreenstein(dot(-viewDir, ldir), g_scatter);
        float   shadow= 1.0; // optionally cast a single shadow ray per candidate
        total += L.color * (L.intensity * atten * phase * shadow);
    }
    return total / max(1u, span.count);
}
```

Notes:
- Intensity source if physical emission is disabled: use a per-particle external-light proxy (e.g., multi-light contribution cached at a small set of probe points per froxel or a lightweight prepass). A simple heuristic that scales with light proximity can suffice initially.
- Temporal accumulation: ping-pong reservoir buffers per frame; blend winners using standard ReSTIR temporal update.

Why it reproduces multi-light smoothness:
- Each sample shades with lights positioned in space (nearby particles), so contributions vary continuously as the ray moves.

References:
- ReSTIR DI (Bitterli et al.)
- RTXDI SDK (NVIDIA)
- Search: `ReSTIR DI RTXDI volumetric reservoirs froxel emissive particles`

---

## 2) Light Tree (Lightcuts) over particles with adaptive traversal
- Score: 9/10
- What it delivers: O(log N) many-light shading with error control; per-sample smooth gradients via clustered “virtual lights”.

How it works:
- Build a BVH/light tree over particle centers each frame (compute or CPU). Each node stores sum flux, centroid, and a tight bound (sphere/ellipsoid) to estimate max contribution.
- At each ray-march sample, traverse adaptively: split nodes whose bound exceeds a screening threshold; accept nodes whose bound is small and accumulate their representative; cast shadows only for the top contributors.

Data layout:
```cpp
struct LightNode {
    float3 center;  float radius;
    float3 fluxRGB; float errorBound; // conservative upper bound
    uint   left;    uint right;       // indices (or -1)
    uint   isLeaf;  uint first; uint count; uint _pad;
};
StructuredBuffer<LightNode> g_lightTree;
```

Shading glue:
```hlsl
float3 ShadeWithLightTree(float3 P, float3 viewDir) {
    float3 sum = 0;
    Stack stack; stack.push(rootIndex);
    while (!stack.empty()) {
        uint i = stack.pop();
        LightNode N = g_lightTree[i];
        if (N.errorBound < g_threshold) {
            // Accept node as clustered light
            float3 L = N.center - P; float dist = length(L);
            float atten = 1.0 / (1.0 + dist*dist);
            float phase = HenyeyGreenstein(dot(-viewDir, normalize(L)), g_scatter);
            sum += N.fluxRGB * (atten * phase);
        } else if (N.isLeaf != 0) {
            // Evaluate leaf particles individually (optionally shadow top K)
            [loop] for (uint k=0;k<N.count;k++) { /* accumulate */ }
        } else {
            stack.push(N.left); stack.push(N.right);
        }
    }
    return sum;
}
```

References:
- Lightcuts (Walter et al.)
- Bounded many-light sampling
- Search: `Lightcuts BVH many lights volumetric single-scattering real-time`

---

## 3) DDGI-for-Volumes (froxel irradiance cache, sparse RT updates)
- Score: 8.5/10
- What it delivers: Stable smooth lighting by caching directional irradiance per froxel and amortizing RT cost.

How it works:
- 3D cache storing low-order SH (e.g., 3 bands) per froxel. Each frame update a subset via RT samples toward important directions (guided by light/particle distributions). Apply HG phase at shading time.

Data layout:
```cpp
struct SH9 { float c[9*3]; }; // RGB
RWTexture3D<SH9> g_froxelIrradiance; // or StructuredBuffer
RWTexture3D<float> g_froxelValidity; // confidence/age
```

Update pass:
```hlsl
// Sample a few rays per froxel this frame; accumulate into SH with temporal filtering
```

Shading glue:
```hlsl
float3 ShadeFromIrradiance(float3 P, float3 viewDir) {
    SH9 sh = g_froxelIrradiance[ToFroxel(P)];
    float3 L = EvaluateSH(sh, /*direction proxy or integrate HG*/);
    return L; // modulate by density/transmittance in the ray-march
}
```

References:
- DDGI / RTXGI concepts adapted to volumes
- Search: `DDGI volumetric irradiance cache ray tracing participating media`

---

## 4) Unshadowed emission splat + RT occlusion correction
- Score: 8/10
- What it delivers: Smooth base field from analytic splatting, corrected with a few RT occlusion samples for depth/contrast.

How it works:
- Prepass: splat each particle’s contribution into a 3D irradiance volume analytically (gaussian integral per voxel).
- At shading: read base irradiance; cast 1–2 shadow rays from the sample toward dominant directions/near clusters; apply HG with an occlusion term.

Data layout:
```cpp
RWTexture3D<float3> g_emissionVolume;  // unshadowed irradiance
```

Shading glue:
```hlsl
float3 ShadeSplatPlusRT(float3 P, float3 viewDir) {
    float3 base = g_emissionVolume[ToVoxel(P)];
    float occl  = RT_OcclusionProbe(P); // 1–2 rays
    float phase = /* HG with best dir */;
    return base * occl * phase;
}
```

References:
- Voxel GI ideas (Crassin) adapted to volumes
- Deep shadow map concepts
- Search: `volumetric irradiance splatting voxel occlusion ray tracing`

---

## 5) Photon beams (RT-driven beam injection, sparse voxel gather)
- Score: 7.5/10
- What it delivers: High-quality volumetric shafts and highlights through forward energy injection.

How it works:
- Select top-K particles per frame (reservoir). From each, cast a few RT “photon beams”; deposit energy into a sparse voxel/hash grid along ray segments (with transmittance). During shading, gather from nearby beam voxels.

Data layout:
```cpp
struct BeamVoxel { float3 radiance; float transmittance; };
// Sparse hash grid (GPU) keyed by voxel coords → BeamVoxel accumulators
```

Shading glue:
```hlsl
float3 ShadeBeams(float3 P) {
    // Gather from beam voxels around P (few lookups)
    return GatherSparseBeams(P);
}
```

References:
- Progressive Photon Beams (Hachisuka)
- Beam Radiance Estimation (Jarosz)
- Search: `progressive photon beams volumetric real-time hash grid`

---

## Integration notes (your codebase)
- Shaders: integrate in `shaders/particles/particle_gaussian_raytrace.hlsl` by replacing per-particle lookups with per-sample shading from (1) froxel reservoirs, (2) light tree, or (3) irradiance volumes.
- Systems: extend `RTXDILightingSystem.cpp` to manage froxel grids/reservoir buffers and temporal passes. Keep inline RayQuery for shadow probes and neighbor queries.
- Memory budget (typical):
  - Froxel reservoirs: 64³ × K=2 → ~0.5–1.0 GB if naïve; use coarser grids (32³), compression, or sparse tiles; or store compact entries (position as 16-bit encodings, intensity FP16) to get ≤100–200 MB.
  - Light tree: ~2N nodes (~20K) × 48–64B ≈ ~1–1.5 MB.
  - Irradiance cache: 32³ × SH9 RGB (9×3×FP16) ≈ ~2.7 MB.

## Quick-start choice
- Fastest path leveraging what you have: (1) Volumetric ReSTIR on top of your RTXDI grid, with K=1–2 and a 32³ froxel grid.
- Deterministic bounds/quality: (2) Light Tree.
- Most stable visuals amortized over time: (3) DDGI-for-Volumes.

## Validation
- Compare against multi-light ground truth on a downscaled scene (fewer particles) and measure SSIM/LPIPS on the Gaussian output; adjust K/thresholds until match.
