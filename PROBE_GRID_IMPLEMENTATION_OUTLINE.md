# Hybrid Probe Grid for Volumetric Particle Scattering
**Goal:** Smooth light scattering for 10K particles from 50+ lights without ReSTIR complexity

---

## Architecture Overview

```
Grid Creation (once) → Probe Update (every 4 frames) → Particle Query (every frame)
```

**Core Idea:** Pre-compute lighting at sparse grid points, particles interpolate between them

---

## Phase 1: Data Structures (Day 1)

### Probe Grid
```cpp
struct Probe {
    float3 position;           // World-space probe location
    float3 irradiance[9];      // Spherical harmonics (L2 = 9 coefficients)
    uint32_t lastUpdateFrame;  // For temporal amortization
};

// Grid parameters
const uint32_t GRID_SIZE = 32;  // 32³ = 32,768 probes
const float GRID_SPACING = 93.75f;  // 3000 units / 32 = 93.75 per cell
const float3 GRID_MIN = float3(-1500, -1500, -1500);
```

### GPU Buffers
```cpp
// In ProbeGridSystem.h
ComPtr<ID3D12Resource> m_probeBuffer;        // 32,768 × 48 bytes = 1.5 MB
ComPtr<ID3D12Resource> m_particleBuffer;     // Your existing particle data
ComPtr<ID3D12Resource> m_lightBuffer;        // Your existing 16 lights
```

---

## Phase 2: Probe Update Shader (Day 2-3)

### Compute Shader: `update_probes.hlsl`
```hlsl
[numthreads(8, 8, 8)]  // 512 threads per group
void UpdateProbes(uint3 probeIdx : SV_DispatchThreadID) {
    // 1. Calculate probe world position
    float3 probePos = GridToWorld(probeIdx);

    // 2. Sample 64 rays in sphere (Fibonacci distribution)
    float3 irradiance = float3(0, 0, 0);
    for (int i = 0; i < 64; i++) {
        float3 direction = FibonacciSphere(i, 64);

        // 3. Trace ray to nearest particle (RayQuery)
        RayDesc ray;
        ray.Origin = probePos;
        ray.Direction = direction;
        ray.TMin = 0.01;
        ray.TMax = 200.0;  // Max influence distance

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_particleTLAS, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            // 4. Hit particle - accumulate lighting from all lights
            float3 hitPos = probePos + direction * q.CommittedRayT();
            float3 lighting = ComputeParticleLighting(hitPos);
            irradiance += lighting;
        }
    }

    // 5. Store spherical harmonics (simplified to RGB for MVP)
    g_probes[ProbeIndex(probeIdx)].irradiance = irradiance / 64.0;
}
```

**Dispatch:** `(32/8, 32/8, 32/8) = (4, 4, 4)` thread groups = 64 dispatches
**Update:** 8,192 probes per frame (full grid updated every 4 frames)

---

## Phase 3: Particle Query Shader (Day 4-5)

### Gaussian Shader Addition: `query_probe_grid.hlsl`
```hlsl
// Add to particle_gaussian_raytrace.hlsl

float3 SampleProbeGrid(float3 worldPos) {
    // 1. Convert world pos to grid coordinates
    float3 gridPos = (worldPos - GRID_MIN) / GRID_SPACING;
    int3 baseProbe = floor(gridPos);
    float3 blend = frac(gridPos);

    // 2. Trilinear interpolation between 8 nearest probes
    float3 irradiance = float3(0, 0, 0);
    for (int z = 0; z <= 1; z++) {
        for (int y = 0; y <= 1; y++) {
            for (int x = 0; x <= 1; x++) {
                int3 probeIdx = baseProbe + int3(x, y, z);
                float3 weight = lerp(1.0 - blend, blend, float3(x, y, z));
                float w = weight.x * weight.y * weight.z;

                // Bounds check
                if (all(probeIdx >= 0) && all(probeIdx < GRID_SIZE)) {
                    uint idx = probeIdx.x + probeIdx.y * GRID_SIZE + probeIdx.z * GRID_SIZE * GRID_SIZE;
                    irradiance += g_probes[idx].irradiance * w;
                }
            }
        }
    }

    return irradiance;
}

// In main rendering loop
float3 probeIrradiance = SampleProbeGrid(particleWorldPos);
float3 finalColor = emission + probeIrradiance * particleDensity;
```

---

## Phase 4: Integration (Day 6-7)

### Files to Create/Modify

**New files:**
- `src/lighting/ProbeGridSystem.h` (class declaration)
- `src/lighting/ProbeGridSystem.cpp` (implementation)
- `shaders/probe_grid/update_probes.hlsl` (probe update compute)
- `shaders/probe_grid/probe_common.hlsl` (shared utilities)

**Modified files:**
- `src/core/Application.cpp` (add ProbeGridSystem initialization)
- `shaders/particles/particle_gaussian_raytrace.hlsl` (add probe sampling)

### ProbeGridSystem Interface
```cpp
class ProbeGridSystem {
public:
    void Initialize(ID3D12Device* device, ResourceManager* resources);
    void UpdateProbes(ID3D12GraphicsCommandList* cmdList,
                      ID3D12Resource* particleTLAS,
                      ID3D12Resource* lightBuffer,
                      uint32_t frameIndex);  // Only update 1/4 per frame

    D3D12_GPU_DESCRIPTOR_HANDLE GetProbeBufferSRV() const;

private:
    ComPtr<ID3D12Resource> m_probeBuffer;
    ComPtr<ID3D12PipelineState> m_updateProbePSO;
    ComPtr<ID3D12RootSignature> m_updateProbeRS;
};
```

---

## Performance Estimates

**Probe Updates (every 4 frames):**
- 8,192 probes × 64 rays = 524,288 rays
- Amortized: 131,072 rays per frame
- Cost: ~0.5-1.0ms

**Particle Queries (every frame):**
- 10K particles × 8 probe reads = 80,000 reads
- Cost: ~0.2-0.3ms (cache-friendly)

**Total overhead:** ~0.7-1.3ms per frame
**Expected FPS:** 90-110 FPS @ 1440p (vs current 120 FPS)

---

## Advantages Over Volumetric ReSTIR

| Feature | Volumetric ReSTIR | Probe Grid |
|---------|-------------------|------------|
| Complexity | Extreme | Medium |
| Implementation time | Months (debugging) | 1-2 weeks |
| Debugging difficulty | Very high | Low |
| Stability | Crashes at 2045 | Stable |
| Many-light support | Yes (complex) | Yes (simple) |
| Quality | Excellent | Good |

---

## MVP Checklist (1 Week Sprint)

**Day 1:** Data structures + buffer creation
- [ ] Create ProbeGridSystem class
- [ ] Allocate 1.5 MB probe buffer
- [ ] Set up descriptor heap entries

**Day 2-3:** Probe update shader
- [ ] Write update_probes.hlsl
- [ ] Fibonacci sphere ray distribution
- [ ] RayQuery particle intersection
- [ ] Light accumulation logic

**Day 4-5:** Particle query integration
- [ ] Add SampleProbeGrid() to gaussian shader
- [ ] Trilinear interpolation code
- [ ] Replace/augment current lighting

**Day 6:** Testing & debugging
- [ ] Verify probe positions correct
- [ ] Check interpolation smoothness
- [ ] Profile performance

**Day 7:** Polish & optimization
- [ ] Temporal amortization (update 1/4 per frame)
- [ ] ImGui controls (grid density, update rate)
- [ ] Compare vs inline RayQuery quality

---

## Success Metrics

✅ **No more brightness flashing** - smooth interpolation between probes
✅ **Support 50+ lights** - pre-baked into probe grid
✅ **90+ FPS** at 10K particles
✅ **Stable** - no crashes or hangs
✅ **2 weeks max** implementation time

---

## Fallback Plan

If probe grid doesn't achieve desired quality:
1. Increase probe density (32³ → 48³)
2. Use spherical harmonics L2 (9 coefficients) instead of RGB
3. Add temporal filtering (blend with previous frame)

---

**Ready to implement!** Start with Day 1 tasks, build incrementally, test frequently.
