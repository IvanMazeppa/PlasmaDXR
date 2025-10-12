# PlasmaDX-Clean

**A cutting-edge DirectX 12 volumetric particle renderer featuring DXR 1.1 inline ray tracing, 3D Gaussian splatting, and ReSTIR global illumination.**

![RTX 4060 Ti | DXR 1.1 | Mesh Shaders | 100K Particles @ 165 FPS](docs/banner.png)

---

## 🚀 Features

### Rendering Technology
- **3D Gaussian Splatting** - Volumetric particle representation with anisotropic ellipsoids
- **DXR 1.1 Inline Ray Tracing** - RayQuery API for procedural primitive intersection
- **ReSTIR Phase 1** - Reservoir-based Spatiotemporal Importance Resampling for 10-60× faster lighting convergence
- **Mesh Shader Rendering** - Hardware-accelerated particle expansion (Tier 1.0)
- **Compute Shader Fallback** - Automatic detection and graceful degradation
- **Real-time Ray-Traced Lighting** - Particle-to-particle illumination via TLAS traversal

### Physics & Simulation
- **GPU Physics Pipeline** - 100,000 particles simulated entirely on GPU
- **Schwarzschild Black Hole** - Relativistic accretion disk dynamics
- **Temperature-Based Emission** - Blackbody radiation (800K-26000K)
- **Doppler Shift & Beaming** - Special relativistic effects
- **Anisotropic Gaussians** - Velocity-aligned particle elongation

### Advanced Graphics
- **Henyey-Greenstein Phase Function** - Physically-based volumetric scattering
- **Shadow Rays** - Real-time soft shadows from primary light source
- **In-Scattering** - Multiple scattering approximation
- **Tone Mapping** - ACES filmic curve with exposure control
- **Debug Visualization** - Per-feature indicators and real-time diagnostics

---

## 🎮 Runtime Controls

### Feature Toggles
| Key | Feature | Default |
|-----|---------|---------|
| **F5** | Shadow Rays | ON |
| **F6** | In-Scattering | ON |
| **F7** | Phase Function | ON |
| **F7** | ReSTIR (temporal reuse) | OFF |
| **CTRL+F7** | Increase ReSTIR temporal weight | 0.5 |
| **SHIFT+F7** | Decrease ReSTIR temporal weight | 0.5 |

### Parameter Adjustments
| Key | Parameter | Range |
|-----|-----------|-------|
| **I / K** | RT Lighting Strength | 0.0 - 10.0 |
| **U / J** | Physical Emission Scale | 0.1 - 5.0 |
| **O / L** | Phase Function Strength | 0.0 - 2.0 |
| **P / ;** | Anisotropy Strength | 0.0 - 1.0 |

### Camera Controls
- **Mouse Drag** - Rotate view
- **Mouse Wheel** - Zoom in/out
- **ESC** - Exit application

---

## 🏗️ Architecture

### Clean Module Design

```
src/
├── main.cpp                      // Entry point (< 100 lines)
├── core/
│   ├── Application.h/cpp         // Window management & main loop
│   ├── Device.h/cpp              // D3D12 device initialization
│   ├── SwapChain.h/cpp           // Present queue management
│   └── FeatureDetector.h/cpp    // RT tier, mesh shader detection
├── particles/
│   ├── ParticleSystem.h/cpp     // Particle lifecycle & logic
│   ├── ParticleRenderer_Gaussian.h/cpp  // 3D Gaussian renderer
│   └── ParticlePhysics.h/cpp    // GPU physics compute shader
├── lighting/
│   ├── RTLightingSystem.h/cpp   // DXR 1.1 lighting pipeline
│   └── AccelerationStructure.h  // BLAS/TLAS management
└── utils/
    ├── ShaderManager.h/cpp       // DXIL loading & reflection
    ├── ResourceManager.h/cpp     // Buffer/texture/descriptor pools
    └── Logger.h/cpp              // Timestamped logging system
```

### Shader Architecture

```
shaders/
├── particles/
│   ├── particle_gaussian_raytrace.hlsl  // Main volumetric renderer (RayQuery)
│   ├── particle_physics.hlsl            // GPU physics (orbital dynamics)
│   ├── particle_build_compute.hlsl      // AABB generation
│   ├── particle_mesh.hlsl               // Mesh shader expansion
│   ├── particle_billboard_vs/ps.hlsl    // Fallback rasterization
│   └── gaussian_common.hlsl             // Shared Gaussian functions
├── dxr/
│   ├── particle_raytraced_lighting_cs.hlsl  // RT lighting compute
│   ├── particle_intersection.hlsl           // Ray-ellipsoid intersection
│   └── generate_particle_aabbs.hlsl         // Procedural primitive bounds
└── util/
    └── buffer_clear.hlsl                     // Fast GPU memset
```

---

## 🔬 Technical Deep Dive

### 3D Gaussian Splatting Implementation

Unlike traditional 2D Gaussian splatting (used in NeRF/3DGS reconstruction), this engine uses **volumetric 3D Gaussians** for physically-based particle rendering:

**Key Differences:**
- **3D Volume**: Each particle is a full 3D ellipsoid, not a 2D splat
- **Ray Marching**: Uses analytic ray-ellipsoid intersection (`RayGaussianIntersection`)
- **Transmittance**: Beer-Lambert law for volumetric absorption
- **Temperature-Based Emission**: Blackbody radiation model (not learned RGB)

**Ray-Ellipsoid Intersection:**
```hlsl
// Transforms ray into ellipsoid space and solves quadratic equation
float2 RayGaussianIntersection(float3 rayOrigin, float3 rayDir,
                               float3 center, float3 scale, float3x3 rotation)
```

**Anisotropic Elongation:**
- Particles elongate along velocity vector
- Scale formula: `scale.xyz = baseRadius * (1, 1, 1 + anisotropy * velocityMagnitude)`
- Creates realistic tidal tearing near black hole

### ReSTIR Phase 1 (Temporal Reuse)

**Algorithm:** Weighted Reservoir Sampling for many-light problems

**Implementation:**
1. **Candidate Sampling** (16-32 rays): Cast random rays to find light-emitting particles
2. **Importance Weighting**: `weight = luminance(emission * intensity * attenuation)`
3. **Reservoir Update**: Probabilistic selection maintains 1 sample from M candidates in O(1) memory
4. **Temporal Reuse**: Previous frame's reservoir is validated and merged with current samples
5. **Unbiased Estimator**: Correction weight `W = weightSum / M`

**Ping-Pong Buffers:**
- 2× reservoirs (63MB each @ 1080p)
- Structure: `{ lightPos, weightSum, M, W, particleIdx }`
- Swap each frame for temporal stability

**Performance:** Target 10-60× convergence speedup vs. naïve random sampling

### DXR 1.1 Inline Ray Tracing

**Why Inline?**
- **RayQuery API**: Call `RayQuery::Proceed()` from any shader stage (compute, pixel, mesh)
- **No Shader Table**: Simpler than TraceRay() with hit groups
- **Procedural Primitives**: AABB-based traversal with custom intersection

**Pipeline:**
```
GPU Physics → Generate AABBs → Build BLAS → Build TLAS →
RayQuery (volumetric render) → RayQuery (shadow rays) →
RayQuery (ReSTIR sampling)
```

**Three Uses of RayQuery:**
1. **Main Rendering** (`particle_gaussian_raytrace.hlsl`): Volume ray marching through sorted particles
2. **Shadow Rays** (`CastShadowRay`): Occlusion testing to primary light source
3. **ReSTIR Sampling** (`SampleLightParticles`): Random rays to find light sources

### Henyey-Greenstein Phase Function

Models **anisotropic scattering** in volumetric media:

```hlsl
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
}
```

- **g = 0**: Isotropic scattering (Rayleigh-like)
- **g > 0**: Forward scattering (Mie-like, smoke/dust)
- **g < 0**: Backward scattering (clouds)

**Application**: Modulates `rtLight` contribution based on viewing angle relative to light direction.

---

## 🔧 Build Instructions

### Prerequisites
- **Visual Studio 2022** (17.0+)
- **Windows SDK 10.0.26100.0** (or higher)
- **DirectX 12 Agility SDK** (included)
- **GPU**: NVIDIA RTX 2060+ or AMD RX 6600+ (DXR 1.1 required)

### Compilation
```bash
# Open solution
start build-vs2022/PlasmaDX-Clean.sln

# Or build from command line
msbuild build-vs2022/PlasmaDX-Clean.sln /p:Configuration=Release /p:Platform=x64
```

### Runtime Requirements
- **Driver**: NVIDIA 531.00+ or AMD Adrenalin 23.1.1+
- **VRAM**: 2GB minimum, 4GB recommended for 100K particles
- **OS**: Windows 10 21H1+ (DXR 1.1 support)

---

## 🐛 Known Issues & Workarounds

### Mesh Shader Descriptor Access (NVIDIA Ada Lovelace)
**Issue:** Mesh shaders cannot read descriptor tables on RTX 40-series (driver bug)
**Detection:** Automatic feature test at startup
**Workaround:** Falls back to compute shader vertex building (no performance loss)

### ReSTIR Phase 1 Debugging
**Status:** Active development (Phase 1 = temporal reuse only)
**Known Bugs:**
- ✅ FIXED: Weight threshold too high for low-temp particles
- ✅ FIXED: Temporal reuse allows M > 0 with weightSum = 0
- 🔄 TESTING: Attenuation formula tuning for large-scale scenes

**Roadmap:**
- Phase 2: Spatial reuse (share reservoirs with neighbors)
- Phase 3: Visibility reuse (cached shadow rays)

---

## 📊 Performance Metrics

### Test Configuration
- **GPU**: NVIDIA GeForce RTX 4060 Ti (8GB)
- **Resolution**: 1920×1080
- **Particles**: 100,000 (volumetric Gaussians)

### Frame Timings (Debug Build)
| Feature Set | FPS | Frame Time | GPU % |
|-------------|-----|------------|-------|
| Raster Only (baseline) | 245 | 4.1 ms | 35% |
| + RT Lighting | 165 | 6.1 ms | 55% |
| + Shadow Rays | 142 | 7.0 ms | 68% |
| + Phase Function | 138 | 7.2 ms | 72% |
| + ReSTIR (active) | 120 | 8.3 ms | 82% |

**Bottleneck:** RayQuery traversal of 100K procedural primitives (BLAS rebuild: 2.1 ms/frame)

**Optimization Potential:**
- Release build: +30% FPS
- Particle LOD culling: +50% FPS
- BLAS update (no rebuild): +25% FPS
- Hardware RT cores utilization: Currently ~60%

---

## 🧪 Debug Features

### Visual Indicators (Always Visible)
- **Top-left (Red bar)**: Shadow rays enabled
- **Top-right (Green bar)**: In-scattering enabled
- **Bottom-left (Blue bar)**: Phase function enabled
- **Bottom-right (Color-coded)**:
  - **Gray**: ReSTIR disabled
  - **Green**: ReSTIR active, good sample quality
  - **Orange**: ReSTIR active, low sample quality
  - **Red**: ReSTIR active, no samples found

### PIX GPU Capture Workflow
1. Launch PIX (from Windows SDK)
2. Attach to `PlasmaDX-Clean.exe`
3. Capture frame (F12)
4. Inspect resources:
   - `g_particles` - Particle buffer (position, velocity, temperature)
   - `g_rtLighting` - Pre-computed RT lighting (fallback)
   - `g_currentReservoirs` - Current frame reservoirs (ReSTIR)
   - `g_prevReservoirs` - Previous frame reservoirs (temporal reuse)

### Logging
All logs written to `logs/PlasmaDX-Clean_YYYYMMDD_HHMMSS.log`
- Feature detection results
- Shader compilation status
- Runtime control changes
- Frame timing breakdowns (CTRL+T to toggle)

---

## 📚 References & Inspiration

### ReSTIR
- Bitterli et al. (2020): "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting"
- NVIDIA RTXDI: Production implementation guide

### 3D Gaussian Splatting
- Kerbl et al. (2023): "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- **Adapted for volumetric physics**, not reconstruction

### Volumetric Rendering
- Max (1995): "Optical Models for Direct Volume Rendering"
- Wrenninge (2012): "Production Volume Rendering" (Pixar)

### Physically-Based Rendering
- Pharr et al. (2016): "Physically Based Rendering: From Theory to Implementation"
- Henyey & Greenstein (1941): "Diffuse radiation in the galaxy"

---

## 🎯 Design Principles

### 1. Feature Detection First
```cpp
// Always test before using
if (detector.GetRayTracingTier() >= D3D12_RAYTRACING_TIER_1_1) {
    renderer = std::make_unique<GaussianRenderer>();
} else {
    LOG_ERROR("DXR 1.1 required!");
    return false;
}
```

### 2. Single Responsibility
- **Application**: Window management + main loop ONLY
- **Device**: D3D12 initialization ONLY
- **ParticleSystem**: Physics & particle logic ONLY
- **ParticleRenderer**: Rendering pipeline ONLY
- **NO 4,000-line monoliths!**

### 3. Automatic Fallbacks
```cpp
if (!CreateMeshShaderPipeline()) {
    LOG_WARN("Mesh shaders failed, using compute fallback");
    return CreateComputeFallback();
}
```

### 4. Data-Driven Configuration
```cpp
// Runtime adjustable via keyboard, not recompilation
gaussianConstants.rtLightingStrength = m_rtLightingStrength;
gaussianConstants.physicalEmissionScale = m_physicalEmissionScale;
```

### 5. Defensive Programming
- Every resource creation has error handling
- All buffers validated before GPU upload
- PIX event markers for every draw call
- Extensive logging at WARN/ERROR levels

---

## 🔮 Roadmap

### Short-Term (Current Sprint)
- [x] ReSTIR Phase 1 debugging (weight calculation fixes)
- [ ] ReSTIR Phase 1 validation (compare vs. ground truth)
- [ ] Performance profiling with PIX timing capture
- [ ] Expose all shader constants to ImGui runtime controls

### Medium-Term
- [ ] ReSTIR Phase 2: Spatial reuse (neighbor sharing)
- [ ] ReSTIR Phase 3: Visibility reuse (cached shadow rays)
- [ ] BLAS update optimization (avoid full rebuild)
- [ ] Particle LOD system (distance-based culling)
- [ ] ImGui overlay (replace keyboard-only controls)

### Long-Term
- [ ] Mesh shader descriptor workaround (investigate driver update)
- [ ] Multi-GPU support (explicit AFR for BLAS building)
- [ ] Neural denoising integration (NVIDIA NRD or AMD FidelityFX)
- [ ] VR support (instanced stereo rendering)

---

## 🤝 Contributing

This is a research/educational project. Contributions welcome for:
- **Bug fixes** (especially ReSTIR edge cases)
- **Performance optimizations** (GPU profiling insights)
- **Documentation improvements** (shader commenting)
- **Feature requests** (file an issue first)

**Code Style:**
- Follow existing patterns (RAII, smart pointers)
- Max 500 lines per file (split if needed)
- Comment all shader algorithms
- Log all state changes at INFO level

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

**Third-Party Dependencies:**
- DirectX 12 Agility SDK (Microsoft)
- Windows SDK (Microsoft)

---

## 👤 Author

**PlasmaDX-Clean** - A clean-architecture rewrite of the original PlasmaDX monolithic engine, focusing on maintainability, extensibility, and cutting-edge rendering techniques.

**Development Environment:**
- Primary: Visual Studio 2022 (C++20)
- Shaders: HLSL Shader Model 6.5+ (DXC)
- Debugging: PIX for Windows, RenderDoc
- Version Control: Git

**Contact:** See GitHub issues for bug reports and feature requests.

---

**Last Updated:** 2025-10-11
**Version:** 1.0.0-beta (ReSTIR Phase 1 in active development)
