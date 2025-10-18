# Ray Traced Particle Lighting Research Index

**Research Session:** 2025-10-03
**Objective:** 100,000 particles with real-time ray traced lighting @ 60fps
**Target Hardware:** RTX 4060 Ti (Ada Lovelace, DXR 1.2)

---

## QUICK START

**Read these in order:**

1. **EXECUTIVE_SUMMARY_PARTICLE_RT.md** - Start here (15 min read)
   - Decision matrix: Is this feasible?
   - Performance projections
   - 4-week implementation roadmap

2. **RT_PARTICLE_LIGHTING_RESEARCH_2025.md** - Main research (45 min read)
   - All 4 techniques with detailed analysis
   - Performance metrics
   - Citations and links

3. **RESTIR_DETAILED_IMPLEMENTATION.md** - Algorithm deep-dive (60 min read)
   - Complete ReSTIR implementation
   - Phase-by-phase code
   - Tuning guide

4. **BLAS_PERFORMANCE_GUIDE.md** - Acceleration structure optimization (45 min read)
   - Clustering strategies
   - Memory pooling
   - Build/refit decisions

5. **ADA_LOVELACE_DXR12_FEATURES.md** - Hardware-specific optimizations (30 min read)
   - SER implementation
   - DXR 1.2 features
   - Ada Lovelace advantages

**Total reading time:** ~3 hours
**Implementation time:** 3-4 weeks

---

## DOCUMENT HIERARCHY

```
agent/AdvancedTechniqueWebSearches/
│
├── EXECUTIVE_SUMMARY_PARTICLE_RT.md          [START HERE]
│   └── High-level overview, go/no-go decision, roadmap
│
├── RESEARCH_INDEX.md                         [THIS FILE]
│   └── Navigation guide, bibliography, quick reference
│
├── ray_tracing/
│   └── particle_systems/
│       ├── RT_PARTICLE_LIGHTING_RESEARCH_2025.md
│       │   └── Main research document with all techniques
│       │
│       └── RESTIR_DETAILED_IMPLEMENTATION.md
│           └── Complete ReSTIR algorithm and code
│
└── efficiency_optimizations/
    ├── BLAS_PERFORMANCE_GUIDE.md
    │   └── Clustered BLAS, memory pooling, build strategies
    │
    └── ADA_LOVELACE_DXR12_FEATURES.md
        └── SER, OMM, DXR 1.2, Ada-specific optimizations
```

---

## TECHNIQUE SUMMARY TABLE

| Technique | Priority | Complexity | Dev Time | Speedup | Doc Reference |
|-----------|----------|------------|----------|---------|---------------|
| **ReSTIR Sampling** | MANDATORY | Medium | 40-60h | 6-60× | RESTIR_DETAILED_IMPLEMENTATION.md |
| **Clustered BLAS** | MANDATORY | Medium | 24-32h | N/A (enables RT) | BLAS_PERFORMANCE_GUIDE.md |
| **SER (Ada)** | Highly Recommended | Low | 8-12h | 1.5-2× | ADA_LOVELACE_DXR12_FEATURES.md |
| **Inline RayQuery** | Optional | Low | 16-24h | 1.1-1.2× | RT_PARTICLE_LIGHTING_RESEARCH_2025.md |
| **Memory Pooling** | Mandatory | Medium | Included in BLAS | 2× memory savings | BLAS_PERFORMANCE_GUIDE.md |
| **Compact Data** | Recommended | Low | 4-8h | 1.1-1.2× | ADA_LOVELACE_DXR12_FEATURES.md |

**Critical Path:**
1. Clustered BLAS (enables ray tracing)
2. ReSTIR sampling (enables lighting)
3. SER (performance optimization)

---

## PERFORMANCE QUICK REFERENCE

### Target Budget (60fps = 16.67ms frame)

| Stage | Optimized Time | % of Budget |
|-------|---------------|-------------|
| BLAS Rebuild (1000 clusters) | 8-10ms | 52% |
| TLAS Build | 0.5-1ms | 4% |
| ReSTIR Sampling | 3-4ms | 20% |
| Visibility Rays (with SER) | 4-6ms | 30% |
| **TOTAL (with optimizations)** | **12-15ms** | **78%** |

**Margin:** 1.67-4.67ms for other rendering

### Key Parameters

```cpp
// ReSTIR
const uint INITIAL_CANDIDATES = 16;      // 8-32 range
const uint TEMPORAL_M_CAP = 20;          // Limit history
const uint SPATIAL_NEIGHBORS = 5;        // 3-10 range
const uint SPATIAL_ITERATIONS = 1;       // 1-2 max

// BLAS
const uint PARTICLES_PER_CLUSTER = 100;  // 50-200 range
const uint NUM_CLUSTERS = 1000;          // For 100K particles
const float GRID_CELL_SIZE = 10.0f;      // World units

// Rendering
const float LIGHTING_RESOLUTION_SCALE = 0.75f;  // 0.5-1.0 range
const uint MAX_RAY_RECURSION = 1;               // Visibility only
```

---

## ALGORITHM PSEUDOCODE QUICK REF

### ReSTIR (Simplified)

```cpp
// Phase 1: Initial Sampling
for each pixel:
    reservoir = empty
    for i in 0..16:  // M candidates
        particle = random_particle()
        weight = unshadowed_contribution(particle) / (1/N)
        reservoir.update(particle, weight)

// Phase 2: Temporal Reuse
for each pixel:
    prev_reservoir = reproject_to_previous_frame(pixel)
    current_reservoir.merge(prev_reservoir)

// Phase 3: Spatial Reuse
for each pixel:
    for neighbor in sample_neighbors(5):
        neighbor_reservoir = load(neighbor)
        current_reservoir.merge(neighbor_reservoir)

// Phase 4: Visibility
for each pixel:
    particle = reservoir.selected_particle
    if trace_shadow_ray(particle):
        lighting = particle.emission * reservoir.weight
```

### Clustered BLAS

```cpp
// Clustering
clusters = []
grid = spatial_hash_grid(particles, cell_size=10.0)
for each grid_cell in grid:
    if cell.particle_count > MAX_PER_CLUSTER:
        split_into_multiple_clusters(cell)
    else:
        clusters.add(cell)

// BLAS Build
for each cluster in clusters:
    vertices, indices = generate_billboards(cluster.particles)
    upload_geometry(cluster, vertices, indices)
    build_blas(cluster, geometry_address, blas_pool_offset)

// TLAS Build
instances = []
for each cluster in clusters:
    instance.blas_address = cluster.blas_pool_offset
    instance.transform = identity
    instances.add(instance)
build_tlas(instances)
```

---

## CITATIONS & SOURCES

### Core Papers

1. **ReSTIR (SIGGRAPH 2020)**
   - Title: "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting"
   - Authors: Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz
   - Link: https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct
   - **Status:** Production-Ready
   - **Maturity:** Proven (used in shipped titles)

2. **ReSTIR GI (HPG 2021)**
   - Title: "ReSTIR GI: Path Resampling for Real-Time Path Tracing"
   - Authors: Ouyang, Liu, Peng, Guo, Cao, Yang, Wu, Yan
   - Link: https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
   - **Status:** Experimental
   - **Note:** Extension for global illumination (not needed for MVP)

3. **ReSTIR Course (SIGGRAPH 2023)**
   - Title: "A Gentle Introduction to ReSTIR Path Reuse in Real-Time"
   - Author: Chris Wyman
   - Link: https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
   - **Status:** Educational
   - **Note:** Best learning resource for ReSTIR

### Technical Articles

4. **NVIDIA RTX Best Practices (2023)**
   - Title: "Best Practices: Using NVIDIA RTX Ray Tracing"
   - Publisher: NVIDIA Developer Blog
   - Link: https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/
   - **Topics:** BLAS/TLAS construction, particle systems, memory management

5. **DXR Memory Management (2020)**
   - Title: "Managing Memory for Acceleration Structures in DirectX Raytracing"
   - Publisher: NVIDIA Developer Blog
   - Link: https://developer.nvidia.com/blog/managing-memory-for-acceleration-structures-in-dxr/
   - **Topics:** Memory pooling, TLB optimization, alignment

6. **Rendering Millions of Lights (2020)**
   - Title: "Rendering Millions of Dynamic Lights in Real-Time"
   - Publisher: NVIDIA Developer Blog
   - Link: https://developer.nvidia.com/blog/rendering-millions-of-dynamics-lights-in-realtime/
   - **Topics:** ReSTIR implementation, performance metrics

### Specifications

7. **DirectX Raytracing Spec**
   - Title: "DirectX Raytracing (DXR) Functional Spec"
   - Publisher: Microsoft
   - Link: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html
   - **Version:** DXR 1.2
   - **Topics:** API reference, inline raytracing, work graphs

8. **Opacity Micromaps Spec (2023)**
   - Title: "Opacity Micromaps Documentation"
   - Publisher: Microsoft
   - Link: https://microsoft.github.io/DirectX-Specs/d3d/Opacity-Micromaps.html
   - **Topics:** OMM for alpha-tested geometry

### Architecture Documents

9. **Ada Lovelace Whitepaper (2022)**
   - Title: "NVIDIA Ada Lovelace GPU Architecture"
   - Publisher: NVIDIA
   - Link: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
   - **Topics:** RT Core Gen 3, SER, performance improvements

10. **Shader Execution Reordering (2022)**
    - Title: "Improve Shader Performance with Shader Execution Reordering"
    - Publisher: NVIDIA Developer Blog
    - Link: https://developer.nvidia.com/blog/improve-shader-performance-and-in-game-frame-rates-with-shader-execution-reordering/
    - **Topics:** SER implementation, performance data

### Tutorials

11. **DXR Tutorial Part 1**
    - Title: "DX12 Raytracing Tutorial - Part 1"
    - Publisher: NVIDIA Developer
    - Link: https://developer.nvidia.com/rtx/raytracing/dxr/dx12-raytracing-tutorial-part-1
    - **Topics:** Basic DXR setup, BLAS/TLAS, shader tables

12. **BVH Building (2022)**
    - Title: "How to build a BVH – part 5: TLAS & BLAS"
    - Author: Jacco Bikker
    - Link: https://jacco.ompf2.com/2022/05/07/how-to-build-a-bvh-part-5-tlas-blas/
    - **Topics:** Acceleration structure theory, optimization

### Conference Presentations

13. **SIGGRAPH 2024 - Ray Tracing Roundup**
    - Link: https://s2024.siggraph.org/past-present-and-future-of-ray-tracing/
    - **Topics:** State of RT, future directions

14. **SIGGRAPH 2025 - Advances in Real-Time Rendering**
    - Link: https://www.advances.realtimerendering.com/s2025/index.html
    - **Topics:** Latest game rendering techniques (20th anniversary)

### Additional Resources

15. **Real-Time Rendering Resources**
    - Link: http://www.realtimerendering.com/
    - **Topics:** Bibliography, code samples, latest research

16. **Shader Toy (Community)**
    - Link: https://www.shadertoy.com/
    - **Topics:** Practical shader implementations (search for "ReSTIR")

---

## IMPLEMENTATION CHEAT SHEET

### File Structure (Recommended)

```cpp
// Header files
ParticleClusterer.h                  // Spatial clustering
BLASMemoryPool.h                     // Memory pooling
ParticleBLASBuilder.h                // BLAS construction
ParticleTLASBuilder.h                // TLAS construction
ReSTIRManager.h                      // ReSTIR pipeline

// Shader files
ParticleReSTIR_InitialSampling.hlsl  // Phase 1
ParticleReSTIR_TemporalReuse.hlsl    // Phase 2
ParticleReSTIR_SpatialReuse.hlsl     // Phase 3
ParticleReSTIR_FinalShading.hlsl     // Phase 4 (RayGen)
ParticleReSTIR_Common.hlsli          // Shared structures

// Optional: SER version
ParticleLighting_SER.hlsl            // Ada Lovelace optimized
```

### GPU Resources

```cpp
// Acceleration Structures
ID3D12Resource* blasMemoryPool;      // 500MB pooled BLAS
ID3D12Resource* tlasBuffer;          // ~2MB TLAS
ID3D12Resource* blasScratchBuffer;   // Temp for builds

// Particle Data
ID3D12Resource* particleBuffer;      // 100K × 16 bytes = 1.6MB

// ReSTIR Buffers
ID3D12Resource* reservoirBuffer[2];     // Ping-pong, ~10MB each
ID3D12Resource* temporalReservoirBuffer; // ~10MB
ID3D12Resource* finalReservoirBuffer;    // ~10MB

// G-Buffer (existing)
ID3D12Resource* gBufferWorldPos;
ID3D12Resource* gBufferNormal;
ID3D12Resource* gBufferMotionVectors;

// Output
ID3D12Resource* lightingOutput;      // Final lit image
```

### Shader Constants

```hlsl
cbuffer ReSTIRConstants : register(b0) {
    uint gParticleCount;             // 100,000
    uint gInitialCandidates;         // 16
    uint gTemporalMCap;              // 20
    uint gSpatialNeighbors;          // 5
    float gSpatialRadius;            // 30 pixels
    uint gFrameIndex;                // For RNG seed
    float3 gCameraPos;
    matrix gViewProj;
    matrix gPrevViewProj;
};

cbuffer BLASConstants : register(b1) {
    uint gNumClusters;               // 1000
    float gGridCellSize;             // 10.0
};
```

---

## DEBUGGING CHECKLIST

### Visual Validation

- [ ] Cluster ID visualization (color code by cluster)
- [ ] Reservoir M heatmap (number of samples)
- [ ] Selected particle visualization (which light chosen)
- [ ] ReSTIR weight heatmap (sample quality)
- [ ] Motion vector validation (temporal reprojection)
- [ ] BLAS bounds visualization (cluster AABBs)

### Performance Profiling

- [ ] GPU timestamps for each stage
- [ ] PIX/NSight capture analysis
- [ ] Memory bandwidth profiling
- [ ] RT core utilization metrics
- [ ] Cache hit rate monitoring

### Correctness Tests

- [ ] Compare ReSTIR output with ground truth (many samples)
- [ ] Validate unbiased weighting (E[ReSTIR] = E[ground truth])
- [ ] Check temporal stability (no flickering)
- [ ] Test edge cases (0 particles, 1 particle, 1M particles)

---

## COMMON PITFALLS & SOLUTIONS

### Problem: Flickering / Temporal Instability

**Causes:**
- Temporal M cap too low
- Motion vector errors
- Aggressive spatial reuse

**Solutions:**
- Increase M cap to 20-30
- Validate reprojection (normal/depth thresholds)
- Reduce spatial radius or neighbors

### Problem: Over-smoothing / Blurry Lighting

**Causes:**
- Too many spatial neighbors
- Spatial radius too large
- Incorrect weight calculations

**Solutions:**
- Reduce neighbors to 3-5
- Reduce radius to 20-30 pixels
- Verify target PDF evaluation

### Problem: Slow BLAS Build

**Causes:**
- Too many clusters
- Not using memory pooling
- Geometry upload bottleneck

**Solutions:**
- Increase particles per cluster (100→200)
- Implement memory pool (single allocation)
- Use persistent mapped buffers

### Problem: Low SER Speedup

**Causes:**
- Particles already coherent
- Not using ReorderThread()
- Driver version too old

**Solutions:**
- Verify scattered particle distribution
- Check HitObject API usage
- Update NVIDIA driver (526+)

---

## NEXT STEPS

### Week 1: Foundation
1. Read EXECUTIVE_SUMMARY_PARTICLE_RT.md
2. Read BLAS_PERFORMANCE_GUIDE.md
3. Implement particle clustering
4. Test with 10K particles first

### Week 2: ReSTIR
1. Read RESTIR_DETAILED_IMPLEMENTATION.md
2. Implement Phase 1: Initial sampling
3. Implement Phase 2: Temporal reuse
4. Implement Phase 3-4: Spatial + visibility

### Week 3: Optimization
1. Read ADA_LOVELACE_DXR12_FEATURES.md
2. Enable SER
3. Add temporal BLAS caching
4. Resolution scaling

### Week 4: Polish
1. Debug visualization
2. Performance tuning
3. Stress testing
4. Documentation

---

## SUPPORT & COMMUNITY

### Forums & Discussions
- NVIDIA Developer Forums: https://forums.developer.nvidia.com/
- Real-Time Rendering Slack: http://www.realtimerendering.com/blog/
- DirectX Discord: https://discord.gg/directx

### Sample Code Repositories
- NVIDIA Falcor: https://github.com/NVIDIAGameWorks/Falcor
- Microsoft DirectX-Graphics-Samples: https://github.com/microsoft/DirectX-Graphics-Samples
- DXR Tutorial Code: https://github.com/NVIDIAGameWorks/DxrTutorials

### Tools
- PIX (Microsoft): https://devblogs.microsoft.com/pix/
- NVIDIA NSight Graphics: https://developer.nvidia.com/nsight-graphics
- RenderDoc: https://renderdoc.org/

---

## REVISION HISTORY

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-03 | 1.0 | Initial research compilation |
|  |  | - 4 core techniques documented |
|  |  | - Performance projections |
|  |  | - Implementation roadmap |

---

## CONTACT & ATTRIBUTION

**Research Compiled By:** Claude (Graphics Research Agent)
**Date:** 2025-10-03
**Project:** PlasmaDX - Accretion Disk Particle Lighting
**Target Hardware:** RTX 4060 Ti (Ada Lovelace)

**License:** Research for internal project use
**Attribution:** Please cite original papers when publishing results

---

**END OF RESEARCH INDEX**

Navigate to specific documents for detailed implementation guidance.
