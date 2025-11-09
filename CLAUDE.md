# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

The user is named Ben.

**PlasmaDX-Clean** is a DirectX 12 volumetric particle renderer featuring DXR 1.1 inline ray tracing, 3D Gaussian splatting, volumetric RT lighting, NVIDIA RTXDI integration, and ML-accelerated physics via Physics-Informed Neural Networks (PINNs). Simulates a black hole accretion disk achieving 20 FPS @ 1440p with 10K particles, 16 lights, and full RT lighting on RTX 4060 Ti hardware.

**Current Status (2025-11-09):**
- **RT Engine Breakthrough** - First working volumetric RT lighting system ✅ COMPLETE (Phase 2.6)
- **Physical Emission Hybrid System** - Artistic/physical blend mode ✅ COMPLETE (Phase 2.5)
- **Multi-Light System** - 13 lights with dynamic control ✅ COMPLETE (Phase 3.5)
- **PCSS Soft Shadows** - Temporal filtering at 115-120 FPS ✅ COMPLETE (Phase 3.6)
- **NVIDIA DLSS 3.7** - Super Resolution operational (Ray Reconstruction shelved) ✅ COMPLETE
- **Variable Refresh Rate** - Tearing mode support ✅ COMPLETE
- **Screen-Space Contact Shadows** - Depth pre-pass system ✅ COMPLETE (Phase 2)
- RTXDI M5 (Phase 2) - Temporal accumulation with ping-pong buffers 🔄 IN PROGRESS
- PINN ML Physics - Python training complete, C++ integration in progress 🔄 IN PROGRESS
- Adaptive Particle Radius (Phase 1.5) - Camera-distance adaptive sizing ✅ COMPLETE
- MCP server operational with 5 tools (performance, PIX analysis, ML comparison, screenshot listing, visual quality assessment)
- F2 screenshot capture system ✅ COMPLETE
- God rays system ⚠️ SHELVED (performance/quality issues)
- 30×30×30 spatial grid covering 3000×3000×3000 unit world space

**Core Technology Stack:**
- DirectX 12 with Agility SDK
- DXR 1.1 (RayQuery API for inline ray tracing)
- **NVIDIA DLSS 3.7** (Super Resolution + Ray Reconstruction)
- HLSL Shader Model 6.5+
- ImGui, PIX for Windows
- ONNX Runtime (ML inference, optional)
- PyTorch (PINN training)

---

## Build System

### Primary Build Method (Visual Studio)
```bash
# Open solution (primary workflow)
start PlasmaDX-Clean.sln

# Or build from command line
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Quick rebuild (most common during development)
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Rebuild

# Clean build from scratch
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Clean
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build
```

### Two Build Configurations

**Debug** - Daily development (zero PIX overhead):
- D3D12 debug layer enabled
- Output: `build/Debug/PlasmaDX-Clean.exe`

**DebugPIX** - PIX GPU debugging (instrumented):
- PIX capture support enabled
- Auto-capture at specified frame
- Output: `build/DebugPIX/PlasmaDX-Clean-PIX.exe`

### Shader Compilation

Shaders are compiled automatically during build via CMake custom commands. The build system uses `dxc.exe` to compile all `.hlsl` files to `.dxil` bytecode.

**Manual shader compilation (if needed):**
```bash
dxc.exe -T cs_6_5 -E main shaders/particles/particle_physics.hlsl -Fo particle_physics.dxil

# For DXR raytracing shaders (lib_6_3)
dxc.exe -T lib_6_3 shaders/rtxdi/rtxdi_raygen.hlsl -Fo shaders/rtxdi/rtxdi_raygen.dxil
```

### ONNX Runtime (ML Features) - Optional

**Detection:** CMake automatically detects ONNX Runtime presence
- If found: `ENABLE_ML_FEATURES=ON`, PINN available
- If missing: `ENABLE_ML_FEATURES=OFF`, warning shown, ML features disabled

**Setup (Optional):**
```bash
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to: external/onnxruntime/
# Required structure: external/onnxruntime/include/, lib/
```

---

## Configuration System

The application uses a JSON-based configuration system with hierarchical loading:

1. **Command-line:** `--config=<path>` (highest priority)
2. **Environment:** `PLASMADX_CONFIG=<path>`
3. **Build directory:** `./config.json`
4. **User default:** `configs/user/default.json`
5. **Hardcoded defaults** (fallback)

**Key config directories:**
- `configs/user/` - User/development configs
- `configs/builds/` - Build-specific defaults (Debug.json, DebugPIX.json)
- `configs/agents/` - AI agent configs (pix_agent.json for autonomous debugging)
- `configs/scenarios/` - Test scenarios (close_distance.json, far_distance.json, etc.)
- `configs/presets/` - Shadow quality presets (performance.json, balanced.json, quality.json)
- `ml/models/` - PINN trained models (pinn_accretion_disk.onnx)
- `ml/training_data/` - Physics trajectory datasets for PINN training

**Common workflow:**
```bash
# Default run
./build/Debug/PlasmaDX-Clean.exe

# Custom config
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/stress_test.json

# PINN training data collection
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
```

See `configs/README.md` for complete documentation.

---

## Architecture

### Clean Module Design

The codebase follows strict separation of concerns with single-responsibility modules:

**Core Systems** (`src/core/`):
- `Application.h/cpp` - Window management and main loop ONLY
- `Device.h/cpp` - D3D12 device initialization
- `SwapChain.h/cpp` - Present queue management
- `FeatureDetector.h/cpp` - RT tier and mesh shader capability detection

**Particle Systems** (`src/particles/`):
- `ParticleSystem.h/cpp` - Particle lifecycle, GPU physics compute shader, accretion disk simulation
- `ParticleRenderer_Gaussian.h/cpp` - 3D Gaussian splatting volumetric renderer (RayQuery-based)
- `ParticlePhysics.h/cpp` - Physics constants and initialization

**Lighting Systems** (`src/lighting/`):
- `RTLightingSystem_RayQuery.h/cpp` - DXR 1.1 inline ray tracing lighting pipeline
- `RTXDILightingSystem.h/cpp` - NVIDIA RTXDI weighted reservoir sampling
- Builds BLAS/TLAS acceleration structures
- Computes particle-to-particle illumination
- **IMPORTANT:** The Gaussian renderer reuses the TLAS from this system (no duplicate infrastructure)

**ML Systems** (`src/ml/`):
- `AdaptiveQualitySystem.h/cpp` - ONNX Runtime integration for PINN physics inference
- Hybrid mode: PINN for far particles, traditional physics for near ISCO
- Performance: 5-10× speedup at 100K particles

**Utilities** (`src/utils/`):
- `ShaderManager.h/cpp` - DXIL loading and reflection
- `ResourceManager.h/cpp` - Buffer/texture/descriptor pool management
- `Logger.h/cpp` - Timestamped logging to `logs/` directory

### Key Architecture Principles

1. **Feature Detection First** - Always test capabilities before using (RT tier, mesh shaders, ONNX Runtime)
2. **Single Responsibility** - No 4,000-line monoliths; max ~500 lines per file
3. **Automatic Fallbacks** - Mesh shader failure → compute shader fallback, ONNX missing → traditional physics
4. **Data-Driven Configuration** - Runtime adjustable via JSON/ImGui, not recompilation
5. **Defensive Programming** - Every resource creation has error handling and PIX event markers

---

## Shader Architecture

### Key Shaders

**Physics Simulation:**
- `shaders/particles/particle_physics.hlsl` - GPU physics compute shader (Schwarzschild black hole gravity, Keplerian orbital dynamics, temperature-based blackbody emission, anisotropic Gaussian elongation)

**Volumetric Rendering:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Main volumetric renderer (RayQuery API, analytic ray-ellipsoid intersection, Beer-Lambert law for volumetric absorption, Henyey-Greenstein phase function, hybrid emission blend system)

**Ray Traced Lighting:**
- `shaders/dxr/particle_raytraced_lighting_cs.hlsl` - RT lighting compute (particle-to-particle illumination via TLAS traversal, shadow rays for occlusion)

**RTXDI System:**
- `shaders/rtxdi/rtxdi_raygen.hlsl` - DXR raygen shader for weighted reservoir sampling
- `shaders/rtxdi/rtxdi_temporal_accumulate.hlsl` - M5 temporal accumulation (ping-pong)
- `shaders/rtxdi/rtxdi_miss.hlsl` - Miss shader for RTXDI pipeline

**Acceleration Structure:**
- `shaders/dxr/generate_particle_aabbs.hlsl` - Procedural primitive AABB generation
- `shaders/dxr/particle_intersection.hlsl` - Ray-ellipsoid intersection shader

**IMPORTANT:** Root constants are limited to 64 DWORDs. Large constant structures should use constant buffers instead.

---

## 3D Gaussian Splatting Implementation

Unlike traditional 2D Gaussian splatting (NeRF/3DGS reconstruction), this engine uses **volumetric 3D Gaussians** for physically-based particle rendering:

**Key differences from 2D splatting:**
- Full 3D ellipsoid volume, not 2D splat
- Ray marching with analytic ray-ellipsoid intersection
- Beer-Lambert law for volumetric absorption
- Temperature-based blackbody emission (not learned RGB)
- Anisotropic elongation along velocity vectors (tidal tearing effects)

**Ray-Ellipsoid Intersection:**
The core algorithm transforms rays into ellipsoid space and solves the quadratic equation. See `RayGaussianIntersection()` in `gaussian_common.hlsl`.

---

## RTXDI Implementation (Phase 4 - ACTIVE)

**Status:** M5 Phase 2 In Progress - Temporal Accumulation with Ping-Pong Buffers

**Implementation:**
- **DXR Pipeline:** Raygen shader performs weighted reservoir sampling
- **Light Grid:** 30×30×30 spatial acceleration structure (27,000 cells)
- **World Coverage:** -1500 to +1500 units per axis (3000-unit range)
- **Cell Size:** 100 units per cell (optimized for wide light distribution)
- **Output:** R32G32B32A32_FLOAT texture (selected light index per pixel)
- **Temporal Accumulation:** Ping-pong buffers for M5 temporal reuse

**Architecture:**
1. **Light Grid Building** (compute shader): Uploads 13-16 lights to GPU, computes spatial grid cells, assigns lights to cells, calculates importance weights
2. **Weighted Reservoir Sampling** (DXR raygen shader): Maps pixel to world position → grid cell, reads cell's light list and weights, performs weighted random selection (1 light per pixel per frame), uses PCG hash for temporal variation
3. **Temporal Accumulation (M5)** (compute shader): Ping-pong buffers for temporal reuse, accumulates 8-16 samples over 60ms, smooths patchwork pattern from M4 Phase 1
4. **Gaussian Renderer Integration**: Reads RTXDI output buffer (t6), uses temporally accumulated light selection, auto-disables RT particle-to-particle lighting in RTXDI mode

**Key Technical Details:**
- **Fibonacci Sphere Distribution** (RTXDI Sphere preset) - 13 lights @ 1200-unit radius
- **Dual-Ring Formation** (RTXDI Ring preset) - 16 lights @ 600-1000 unit radii
- **Cross Pattern** (RTXDI Sparse preset) - 5 lights for debugging grid behavior
- **Temporal Smoothing** - M5 accumulation reduces patchwork artifacts

**Migration from Custom ReSTIR:**
Original custom ReSTIR implementation (126 MB reservoir buffers) was deprecated in favor of lightweight RTXDI approach. Custom implementation removed, RTXDI built from scratch using NVIDIA ReSTIR paper principles (Bitterli et al. 2020).

**Current Status:**
- Phase 1 (M4): Weighted sampling ✅ COMPLETE
- Phase 2 (M5): Temporal accumulation 🔄 IN PROGRESS
- Phase 3 (M6): Spatial reuse ⏳ PLANNED

---

## Multi-Light System (Phase 3.5 - COMPLETE ✅)

**Status:** Implemented and working

First production-ready many-light system using RT volumetric rendering. 13 lights distributed across the accretion disk achieve realistic multi-directional shadowing, rim lighting, and atmospheric scattering effects.

**Architecture:**
- Light array stored in `m_lights` vector (Application.cpp:118)
- Uploaded to GPU via structured buffer (32 bytes/light)
- Gaussian raytrace shader reads all lights (particle_gaussian_raytrace.hlsl:726)
- Bulk color control system for all lights simultaneously

**ImGui Controls:**
- Position (X/Y/Z sliders), Color (RGB picker), Intensity (0.1 - 20.0), Radius (10.0 - 200.0)
- Per-light enable/disable toggles
- **Bulk Controls:** Apply color/intensity to all lights at once

**Presets:**
- **Stellar Ring** - 13 lights in circular formation (default)
- **Dual Binary** - 2 opposing lights (binary star system)
- **Trinary Dance** - 3 lights in triangular formation
- **Single Beacon** - 1 centered light (debugging/comparison)

**Performance:** 120+ FPS @ 10K particles with 13 lights (RTX 4060 Ti, 1080p)

---

## PCSS Soft Shadows (Phase 3.6 - COMPLETE ✅)

Full PCSS (Percentage-Closer Soft Shadows) implementation with temporal filtering achieving soft shadows at 115-120 FPS (Performance preset) on RTX 4060 Ti @ 1080p with 10K particles.

### Three Shadow Quality Presets:

1. **Performance** (Default - 1-ray + temporal filtering): 1 ray per light, temporal accumulation (67ms convergence to 8-ray quality), Target: 115-120 FPS
2. **Balanced** (4-ray PCSS): 4 rays per light (Poisson disk sampling), instant soft shadows, Target: 90-100 FPS
3. **Quality** (8-ray PCSS): 8 rays per light (Poisson disk sampling), highest quality, Target: 60-75 FPS

**Technical Implementation:**
- Shadow buffers: 2× R16_FLOAT (ping-pong, 4MB @ 1080p)
- Root signature: 10 parameters (was 8, +2 for shadow buffers)
- Shader resources: `t5: g_prevShadow`, `u2: g_currShadow`
- Temporal blend formula: `finalShadow = lerp(prevShadow, currentShadow, 0.1)`

**Preset configs:** `configs/presets/shadows_performance.json`, `shadows_balanced.json`, `shadows_quality.json`

**See:** `PCSS_IMPLEMENTATION_SUMMARY.md` for complete technical details

---

## Adaptive Particle Radius (Phase 1.5 - COMPLETE ✅)

**Status:** Fully functional with all bugs fixed

Dynamic particle sizing system that adjusts particle radii based on camera distance and local particle density to maintain visual quality at all viewing distances.

### Core Functionality:

**Camera-Distance Adaptive Sizing:**
- **Inner Zone** (close to camera): Particles shrink to reduce overlap and maintain detail
- **Transition Zone** (mid-distance): Linear interpolation between inner and outer scales
- **Outer Zone** (far from camera): Particles grow to remain visible at distance

**Density-Based Scaling:**
- Local particle density affects size scaling
- High-density regions: More aggressive shrinking to prevent overlap
- Low-density regions: Less aggressive scaling to maintain coverage
- Configurable density scale range (min/max multipliers)

### Technical Implementation:

**Shader Integration:**
- Computed in `particle_gaussian_raytrace.hlsl` and `generate_particle_aabbs.hlsl`
- Camera distance calculated per-particle every frame
- Smooth transitions using `smoothstep()` interpolation
- Applied before AABB generation for RT acceleration structure

**ImGui Controls:**
- Enable/disable toggle
- Inner/Outer zone thresholds (distance from camera)
- Inner/Outer scale multipliers (shrink/grow factors)
- Density scale min/max (density-based adjustment range)
- **Event-driven updates** - setters called only when values change (prevents freezing)

**Parameters:**
```cpp
bool m_enableAdaptiveRadius = true;       // Master toggle
float m_adaptiveInnerZone = 150.0f;       // Distance where shrinking starts
float m_adaptiveOuterZone = 800.0f;       // Distance where growing starts
float m_adaptiveInnerScale = 0.3f;        // Shrink to 30% in inner zone
float m_adaptiveOuterScale = 3.0f;        // Grow to 300% in outer zone
float m_densityScaleMin = 0.5f;           // Minimum density multiplier
float m_densityScaleMax = 1.5f;           // Maximum density multiplier
```

### Bug Fixes Applied:

1. **TDR Crash Fix** - Prevented GPU timeout by fixing infinite loop in AABB generation
2. **BLAS Rebuild Fix** - Ensured BLAS updates correctly when adaptive radius changes
3. **ImGui Freeze Fix** - Changed to event-driven updates (only call setters on value change)

**Key Lesson:** ImGui widgets return `bool` when values change. Always use this to avoid calling setters every frame:

```cpp
// WRONG (calls setter 60× per second)
ImGui::SliderFloat("Value", &value, 0, 1);
SetValue(value);

// CORRECT (calls setter only on change)
if (ImGui::SliderFloat("Value", &value, 0, 1)) {
    SetValue(value);
}
```

### Performance Impact:

- Minimal overhead (<0.1ms per frame)
- Improves visual quality at all distances
- Reduces overdraw in close-up views
- Maintains visibility at extreme distances

**See:** `ADAPTIVE_RADIUS_FIX.md`, `ADAPTIVE_RADIUS_TDR_FIX.md`, `ADAPTIVE_RADIUS_BLAS_FIX.md`, `ADAPTIVE_RADIUS_IMGUI_FREEZE_FIX.md` for complete implementation history

---

## Physics-Informed Neural Networks (Phase 5 - ACTIVE 🔄)

**Status:** Python training pipeline complete ✅, C++ integration in progress 🔄

Research-level PINN that learns accretion disk particle forces while respecting fundamental astrophysics:

### Physics Constraints Enforced:
1. ✅ **General Relativity** - Schwarzschild metric (V_eff with GR correction term)
2. ✅ **Keplerian Motion** - Ω = √(GM/r³) for circular orbits
3. ✅ **Angular Momentum Conservation** - L = r²Ω
4. ✅ **Shakura-Sunyaev Viscosity** - α-disk model (ν = α c_s H)
5. ✅ **Energy Conservation** - Total energy along trajectories

### Key Benefits:
- **5-10× faster** than full GPU physics shader (at 100K particles)
- **Scientifically accurate** - respects conservation laws & GR
- **Hybrid mode ready** - PINN for far particles, shader for close-up
- **Retrainable** - collect new data, improve model

### Network Architecture:

**Input:** `(r, θ, φ, v_r, v_θ, v_φ, t)` - 7D phase space + time
**Hidden:** 5 layers × 128 neurons (Tanh activation)
**Output:** `(F_r, F_θ, F_φ)` - 3D force vector in spherical coordinates
**Parameters:** ~50,000 trainable weights

**Loss Function:**
```
Loss = λ_data · MSE(F_pred, F_true) +
       λ_kepler · Physics_Keplerian +
       λ_L · Physics_AngularMomentum +
       λ_E · Physics_Energy +
       λ_GR · Physics_GeneralRelativity
```

### Quick Start (Python Training):

```bash
# 1. Install dependencies
cd ml
pip install -r requirements_pinn.txt

# 2. Collect training data from GPU
../build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
python collect_physics_data.py --input ../PIX/buffer_dumps

# 3. Train PINN (~20 minutes on GPU)
python pinn_accretion_disk.py

# 4. Test model
python test_pinn.py --model models/pinn_accretion_disk.onnx
```

### C++ Integration Status:

**Completed:**
- ✅ ONNX Runtime linked in CMakeLists.txt
- ✅ AdaptiveQualitySystem class created
- ✅ Automatic detection of ONNX Runtime presence

**In Progress:**
- 🔄 PINN model loading and inference
- 🔄 Hybrid mode (PINN + traditional physics)

**Pending:**
- ⏳ ImGui controls for PINN mode
- ⏳ Performance benchmarking vs traditional physics
- ⏳ Real-time retraining system

### Expected Performance:

| Particle Count | Traditional Physics | PINN Physics | Speedup |
|----------------|---------------------|--------------|---------|
| 10K | 120 FPS | 120 FPS | 1.0× (no benefit) |
| 50K | 45 FPS | 180 FPS | **4.0×** |
| 100K | 18 FPS | 110 FPS | **6.1×** |

**Why faster?** Traditional: O(N) particle updates + O(N·M) RT lighting (expensive), PINN: O(N) neural network inference (constant time per particle)

**See:** `ml/PINN_README.md` and `PINN_IMPLEMENTATION_SUMMARY.md` for complete documentation

---

## NVIDIA DLSS Integration (Phase 7 - PARTIAL ✅)

**Status:** DLSS Super Resolution operational, Ray Reconstruction shelved

NVIDIA Deep Learning Super Sampling 3.7 Super Resolution providing AI-powered upscaling for volumetric particle rendering.

**Implemented: Super Resolution ✅**
- Renders at lower internal resolution with AI upscaling (e.g., 720p → 1440p output)
- Quality modes: Performance, Balanced, Quality, Ultra Performance
- Motion vectors via `shaders/particles/compute_motion_vectors.hlsl` for temporal stability
- Dynamic resolution calculation based on quality preset

**Performance Impact:**
- Performance Mode: +40-60% FPS (720p → 1440p upscaling)
- Quality Mode: +20-30% FPS (960p → 1440p upscaling)
- Balanced Mode: +30-40% FPS (810p → 1440p upscaling)

**Technical Implementation:**
- CMake auto-detects DLSS SDK at `dlss/` directory
- `src/dlss/DLSSSystem.h/cpp` - NGX initialization and feature management
- `ParticleRenderer_Gaussian.cpp` - Integration point for upscaling pipeline
- Copies `nvngx_dlss.dll` to output directory

**Shelved: Ray Reconstruction ⚠️**

Ray Reconstruction (DLSS-RR) was investigated but **shelved** for architectural incompatibility:

**Why shelved:**
1. **G-Buffer Requirements**: DLSS-RR requires full G-buffer (normals, albedo, emissive albedo, roughness, metallic)
2. **Particle Simulation Mismatch**: Particles don't have traditional surface properties (normals, albedo) that make sense geometrically
3. **Fake Data Problem**: Filling G-buffer with synthetic/interpolated data of uncertain quality - DLSS-RR behavior unpredictable with "fake" inputs
4. **Overhead Negates Gains**: Building and maintaining G-buffer for 100K procedural particles would negate denoising performance benefits
5. **Volumetric vs Surface**: DLSS-RR designed for surface-based ray tracing (hard surfaces with clear normals), not volumetric rendering

**Alternative Denoising Approaches:**
- PCSS temporal accumulation (already implemented, works well)
- Custom temporal filtering using existing ping-pong buffers
- RTXDI M5 temporal reuse (in progress)
- AMD FidelityFX Denoiser (surface-agnostic, future consideration)

**Requirements:**
- NVIDIA RTX GPU (Tensor Cores required)
- Driver 531.00+ (DLSS 3.7 support)
- DLSS SDK files in `dlss/` directory

---

## Dynamic Emission System (Phase 3.8 - COMPLETE ✅)

**Status:** ✅ IMPLEMENTED - RT-driven dynamic star radiance

Transform static physical emission into RT-responsive dynamic emission using five key techniques:

### Core Techniques:

1. **RT Lighting Suppression** - Emission strength inversely proportional to RT lighting intensity
   - Well-lit particles (high RT lighting) → low emission visibility
   - Shadow particles (low RT lighting) → high emission (fills in darkness)
   - Default suppression: 70% (configurable via ImGui)

2. **Selective Emission (Temperature Threshold)** - Only particles >22000K emit significantly
   - Cool particles (80-90%) are purely RT-driven → maximum dynamicism
   - Hot particles create focal points with self-emission
   - Threshold: 22000K (configurable)

3. **Temporal Modulation** - Gentle pulsing/scintillation using frame counter
   - Each particle pulses at slightly different rate (70-100% brightness range)
   - Creates "breathing" stars effect
   - Rate: 0.03 (3% pulse frequency, configurable)

4. **Distance-Based LOD** - Adaptive emission based on camera distance
   - Close particles (<300 units): 50% emission → RT lighting dominates detail
   - Far particles (>1000 units): 100% emission → maintains visibility
   - Smooth falloff in between

5. **Improved Blackbody Colors** - Wien's law approximation for accurate star colors
   - Cool red-orange (1000-3000K) → Yellow-orange (3000-6000K) → White (6000-15000K) → Hot blue (15000-30000K)
   - Physically accurate temperature visualization

### Implementation Details:

**RT Lighting Constant Buffer:** Expanded from 4 → 14 DWORDs (56 bytes)
```cpp
struct LightingConstants {
    uint32_t particleCount;           // 1 DWORD
    uint32_t raysPerParticle;         // 1 DWORD
    float maxLightingDistance;        // 1 DWORD
    float lightingIntensity;          // 1 DWORD
    DirectX::XMFLOAT3 cameraPosition; // 3 DWORDs
    uint32_t frameCount;              // 1 DWORD
    float emissionStrength;           // 1 DWORD (0.25 default)
    float emissionThreshold;          // 1 DWORD (22000K default)
    float rtSuppression;              // 1 DWORD (0.7 default)
    float temporalRate;               // 1 DWORD (0.03 default)
};  // Total: 14 DWORDs
```

**CRITICAL:** When expanding constant buffers, always update:
1. Struct definition (header file)
2. Root signature (`InitAsConstants()` call)
3. Upload code (`SetComputeRoot32BitConstants()` call)
4. Shader cbuffer declaration (HLSL)
5. Manually recompile shader if CMake doesn't auto-rebuild

**ImGui Controls:**
- Four sliders: Emission Strength (0.0-1.0), Temp Threshold (18000-26000K), RT Suppression (0.0-1.0), Temporal Rate (0.01-0.1)
- Three presets: Max Dynamicism, Balanced, Star-Like
- Located in "Rendering Features" section (F1 to open)

**Performance Impact:** <0.1ms overhead @ 100K particles (zero additional rays, minimal math)

**See:** `DYNAMIC_EMISSION_IMPLEMENTATION.md` for complete technical details

---

## Volumetric God Rays (Phase 3.7 - SHELVED ⚠️)

**Status:** ⚠️ **SHELVED** - Marked for deactivation due to performance and quality issues. Revisit after RTXDI M6 complete.

---

## DXR 1.1 Inline Ray Tracing

**Why RayQuery API?**
- Call `RayQuery::Proceed()` from any shader stage (compute, pixel, mesh)
- No shader binding table (SBT) complexity
- Simpler than TraceRay() with hit groups
- Perfect for procedural primitives (AABB-based traversal)

**Pipeline:**
```
GPU Physics → Generate AABBs → Build BLAS → Build TLAS →
RayQuery (volumetric render) → RayQuery (shadow rays) → TraceRay (RTXDI sampling)
```

**Three uses of ray tracing:**
1. **Main Rendering** (RayQuery) - Volume ray marching through sorted particles
2. **Shadow Rays** (RayQuery) - Occlusion testing to lights
3. **RTXDI Sampling** (TraceRay) - Weighted reservoir sampling via raygen shader

**Acceleration Structure Reuse:**
The Gaussian renderer reuses the TLAS built by RTLightingSystem. Do NOT create duplicate BLAS/TLAS infrastructure.

---

## Physics Simulation

### Accretion Disk Physics

**Implemented:**
- Schwarzschild black hole gravity (Newtonian approximation)
- Keplerian orbital dynamics
- Temperature-based blackbody radiation (800K-26000K)
- Doppler shift and relativistic beaming
- Anisotropic Gaussian elongation

**In Progress (PINN ML Integration):**
- Physics-informed neural networks for force prediction
- Hybrid mode: PINN for far particles (r > 10×R_ISCO), traditional for near
- 5-10× performance improvement at 100K particles

**Planned (see PHYSICS_PORT_ANALYSIS.md):**
- Constraint shapes system (SPHERE, DISC, TORUS, ACCRETION_DISK)
- Black hole mass parameter (affects Keplerian velocity)
- Alpha viscosity (Shakura-Sunyaev accretion - inward spiral)
- Enhanced temperature models (velocity-based heating)

### Runtime Physics Controls

**Keyboard shortcuts:**
- Up/Down: Gravity strength
- Left/Right: Angular momentum
- Ctrl+Up/Down: Turbulence
- Shift+Up/Down: Damping
- [/]: Particle size

All physics parameters are exposed in the ImGui interface with real-time adjustments.

---

## PIX GPU Debugging Workflow

### Programmatic Capture System

**DebugPIX build** includes PIX integration for autonomous debugging:

1. Build DebugPIX configuration
2. Run with PIX config: `./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json`
3. Automatic capture at specified frame (default: frame 120)
4. Captures saved to `PIX/Captures/`

**Buffer Dumps:**
Add `--dump-buffers <frame>` to save GPU buffers to `PIX/buffer_dumps/`:
- `g_particles.bin` - Particle positions, velocities, temperatures (for PINN training)
- `g_currentReservoirs.bin` - Current frame ReSTIR reservoirs (deprecated)
- `g_prevReservoirs.bin` - Previous frame ReSTIR reservoirs (deprecated)
- `g_rtLighting.bin` - Pre-computed RT lighting

**PINN Training Data:**
```bash
# Collect physics training data
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120

# Process for PINN training
cd ml
python collect_physics_data.py --input ../PIX/buffer_dumps
```

All draw calls and compute dispatches are wrapped with PIX event markers. Use PIX timeline view to navigate the frame.

---

## MCP Server for RTXDI Quality Analysis

**Status:** Operational with 4 tools

**Location:** `agents/rtxdi-quality-analyzer/`

**Server:** `rtxdi_server.py` (flat structure with SDK 0.1.4)

### Available Tools

1. **`compare_performance`** - Compare RTXDI performance metrics between legacy renderer, RTXDI M4, and RTXDI M5
2. **`analyze_pix_capture`** - Analyze PIX GPU captures for RTXDI bottlenecks and performance issues
3. **`compare_screenshots_ml`** - ML-powered before/after screenshot comparison using LPIPS perceptual similarity (~92% correlation to human judgment)
4. **`list_recent_screenshots`** - List recent screenshots from `screenshots/` directory sorted by time (newest first)

### Key Implementation Details

**CRITICAL: Lazy Loading Pattern for ML Tools**

The server uses lazy loading for PyTorch and LPIPS to avoid MCP timeout (30-second limit). PyTorch + LPIPS = 528MB of weights. Loading at import time = 30+ seconds (exceeds MCP timeout), lazy loading = ~8 second startup, ML loads only when `compare_screenshots_ml` is called.

**Running the server:**
```bash
cd agents/rtxdi-quality-analyzer
./run_server.sh
```

### Screenshot Capture System

**In-Application Screenshot Capture:**

Press **F2** during rendering to capture the exact GPU framebuffer at full native resolution.

**Implementation:**
- F2 key binding in `Application.cpp:976-979`
- Direct GPU backbuffer capture via D3D12
- BGRA → RGB conversion with vertical flip for BMP format
- Saves to `screenshots/screenshot_YYYY-MM-DD_HH-MM-SS.bmp`
- Captures at native resolution (1440p) not monitor resolution

**Why BMP instead of PNG:** No external library dependency, lossless quality, easy to implement, ML comparison tool accepts both formats

**Complete workflow:**
```bash
# 1. Capture screenshots in PlasmaDX (F2 key)
# 2. List recent screenshots via MCP
list_recent_screenshots(limit=5)

# 3. Compare screenshots via MCP
compare_screenshots_ml(
    before_path="/path/to/screenshot1.bmp",
    after_path="/path/to/screenshot2.bmp",
    save_heatmap=True
)
# Returns LPIPS similarity score and saves difference heatmap to PIX/heatmaps/
```

---

## Autonomous Testing with v3 Agents

**Status:** Custom plugin system operational (plasmadx-testing-v3)

**Installation:** `~/.claude/plugins/plasmadx-testing-v3/`

PlasmaDX uses Claude Code's plugin system for autonomous multi-agent testing. Custom v3 agents provide specialized DirectX 12 / DXR expertise for buffer validation, debugging, stress testing, and performance analysis.

### Custom v3 Agents

**buffer-validator-v3** - Autonomous GPU buffer validation (validates binary buffer dumps, detects NaN/Inf, out-of-bounds values, statistical anomalies)

**pix-debugger-v3** - Root cause analysis for rendering issues (diagnoses black screens, color artifacts, missing effects, provides specific file:line fixes)

**stress-tester-v3** - Comprehensive stress testing (particle count scaling 100 → 100K, multi-light scaling 0 → 50 lights, camera distance testing)

**performance-analyzer-v3** - Performance profiling (identifies bottlenecks, compares against performance targets, generates optimization recommendations)

### Built-in Claude Code Agents

- **pix-debugging-agent** - Autonomous PIX capture analysis
- **dxr-graphics-debugging-engineer-v2** - DXR rendering diagnosis
- **dx12-mesh-shader-engineer-v2** - Mesh/Amplification shader work
- **physics-performance-agent-v2** - Physics simulation optimization

See `CLAUDE_CODE_PLUGINS_GUIDE.md` for complete plugin system documentation.

---

## Critical Implementation Details

### Descriptor Heap Management

**IMPORTANT:** The ResourceManager maintains a central descriptor heap for all SRVs/UAVs/CBVs. Always allocate descriptors through ResourceManager, never create ad-hoc descriptor heaps.

### Buffer Resource States

**Common transition sequence:**
```
UNORDERED_ACCESS (compute write) →
UAV Barrier →
NON_PIXEL_SHADER_RESOURCE (compute read) →
UNORDERED_ACCESS (next pass)
```

Always insert UAV barriers between dependent compute dispatches.

### Root Signature Limitations

- Root constants: 64 DWORD limit (256 bytes)
- Root descriptors: Direct buffer pointers (best performance)
- Descriptor tables: For large descriptor arrays or typed UAVs

**When to use each:**
- Small constants (<64 DWORDs): Root constants
- Structured buffers: Root descriptors
- Typed UAVs (R16G16B16A16_FLOAT): Descriptor tables (required)

### Acceleration Structure Rebuilds

**Current implementation:** Full BLAS/TLAS rebuild every frame (2.1ms @ 100K particles)

**Optimization potential:** BLAS update (no rebuild): +25% FPS, Instance culling: +50% FPS, Static BLAS with dynamic TLAS: Possible but requires careful particle management

Do NOT attempt BLAS updates without thorough testing - easy to introduce crashes.

---

## Known Issues and Workarounds

### Mesh Shader Descriptor Access (NVIDIA Ada Lovelace)
**Issue:** RTX 40-series driver bug prevents mesh shaders from reading descriptor tables
**Detection:** Automatic at startup via FeatureDetector
**Workaround:** Falls back to compute shader vertex building (no performance loss)

### RTXDI Temporal Accumulation (M5)
**Status:** In active development
**Known issues:**
- ✅ FIXED: Weight threshold too high for low-temp particles (M4)
- ✅ FIXED: Temporal reuse allows M > 0 with weightSum = 0 (M4)
- 🔄 TESTING: Ping-pong buffer accumulation (M5)
- 🔄 TESTING: Convergence rate tuning (M5)

### God Rays System
**Status:** SHELVED ⚠️
**Known issues:** Performance impact not acceptable, visual artifacts at certain angles, conflicts with RTXDI lighting
**Workaround:** Disable via ImGui (feature marked for deactivation)

### PINN ML Integration
**Status:** C++ integration in progress
**Known issues:**
- ⏳ ONNX Runtime model loading not complete
- ⏳ Hybrid mode switching not implemented
- ⏳ ImGui controls not exposed

### ImGui Integration
**Quirk:** ImGui requires a separate descriptor heap for fonts/textures. The Application class manages this automatically. Don't create additional ImGui descriptor heaps.

---

## Performance Targets

**Test Configuration:** RTX 4060 Ti, 1920×1080, 100K particles

| Feature Set | Target FPS | Current | Notes |
|-------------|------------|---------|-------|
| Raster Only | 245 | 245 ✅ | Baseline |
| + RT Lighting | 165 | 165 ✅ | TLAS rebuild bottleneck |
| + Shadow Rays | 142 | 142 ✅ | PCSS Performance preset |
| + Phase Function | 138 | 138 ✅ | Henyey-Greenstein |
| + RTXDI M4 (active) | 120 | 120 ✅ | Weighted sampling |
| + RTXDI M5 (active) | 115 | TBD 🔄 | Temporal accumulation |
| + DLSS Performance | 190 | 190 ✅ | 720p→1440p upscaling |
| + PINN Physics (100K) | 280+ | TBD 🔄 | PINN + DLSS combined |

**Bottleneck:** RayQuery traversal of 100K procedural primitives (BLAS rebuild: 2.1ms/frame)

**Optimization priorities:**
1. PINN ML physics: +50-100 FPS at 100K particles (ACTIVE)
2. RTXDI M5 optimization: +5-10 FPS (IN PROGRESS)
3. BLAS update (not rebuild): +25% FPS (PLANNED)
4. Particle LOD culling: +50% FPS (PLANNED)
5. Release build optimization: +30% FPS (PLANNED)

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

---

## File Naming Conventions

- Headers: `.h` (not `.hpp`)
- Implementation: `.cpp`
- Shaders: `.hlsl`
- Compiled shaders: `.dxil`
- Configs: `.json`
- ML models: `.onnx`
- Training data: `.npz`

**Naming style:**
- Classes: PascalCase (`ParticleSystem`, `RTLightingSystem_RayQuery`, `AdaptiveQualitySystem`)
- Functions: PascalCase (`Initialize`, `Render`)
- Variables: camelCase (`m_particleCount`, `particleBuffer`)
- Constants: UPPER_SNAKE_CASE (`BLACK_HOLE_MASS`, `R_ISCO`)

---

## Logging System

All logs are written to timestamped files in `logs/` directory:
```
logs/PlasmaDX-Clean_YYYYMMDD_HHMMSS.log
```

**Log levels:**
- `LOG_INFO` - General information, startup messages
- `LOG_WARN` - Non-critical issues, fallback activations
- `LOG_ERROR` - Recoverable errors, resource creation failures
- `LOG_CRITICAL` - Unrecoverable errors, immediate exit

---

## Dependencies and External Libraries

**Included in repository:**
- DirectX 12 Agility SDK (`external/D3D12/`)
- ImGui (`external/imgui/`)
- d3dx12.h helper library (`src/utils/d3dx12.h`)
- RTXDI Runtime SDK (`external/RTXDI-Runtime/`)

**Optional dependencies:**
- ONNX Runtime (`external/onnxruntime/`) - For PINN ML physics (optional)

**System dependencies (must be installed):**
- Visual Studio 2022 (C++17 required)
- Windows SDK 10.0.26100.0 or higher
- DXC shader compiler (part of Windows SDK)
- PIX for Windows (optional, for GPU debugging)

**Python dependencies (for PINN training):**
- PyTorch >= 2.0.0
- ONNX >= 1.14.0
- NumPy, Matplotlib, SciPy
- See `ml/requirements_pinn.txt`

**Driver requirements:**
- NVIDIA: 531.00+ (DXR 1.1 support)
- AMD: Adrenalin 23.1.1+ (DXR 1.1 support)

---

## Reference Documentation

**Critical development documentation:**
- `MASTER_ROADMAP_V2.md` - **AUTHORITATIVE** current roadmap and development status
- `PARTICLE_FLASHING_ROOT_CAUSE_ANALYSIS.md` - 14,000-word visual quality investigation
- `SHADOW_RTXDI_IMPLEMENTATION_ROADMAP.md` - RTXDI integration guide (890 lines)
- `BUILD_GUIDE.md` - Quick build reference for Debug vs DebugPIX configurations
- `PCSS_IMPLEMENTATION_SUMMARY.md` - PCSS soft shadows technical details

**In-repo documentation:**
- `README.md` - Project overview, features, controls
- `configs/README.md` - Configuration system reference
- `PHYSICS_PORT_ANALYSIS.md` - Physics feature porting plan
- `PIX/docs/QUICK_REFERENCE.md` - PIX capture system guide
- `ml/PINN_README.md` - PINN training and integration guide
- `PINN_IMPLEMENTATION_SUMMARY.md` - PINN implementation overview
- `DYNAMIC_EMISSION_IMPLEMENTATION.md` - Physical emission hybrid system details

**External references:**
- [DirectX 12 Programming Guide](https://docs.microsoft.com/en-us/windows/win32/direct3d12/)
- [DXR 1.1 Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)
- [ReSTIR Paper](https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling) - Bitterli et al. (2020)
- [RTXDI Documentation](https://github.com/NVIDIAGameWorks/RTXDI) - NVIDIA RTX Direct Illumination SDK
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al. (2023)
- [PINN Paper](https://doi.org/10.1016/j.jcp.2018.10.045) - Raissi et al. (2019)
- [ONNX Runtime](https://onnxruntime.ai/) - Microsoft ML inference runtime

---

## Immediate Next Steps for Development

**Recently completed (Phase 0-3.6):**
1. ✅ RT Engine Breakthrough - First working volumetric RT lighting (Phase 2.6)
2. ✅ Physical Emission Hybrid System - Artistic/physical blend mode (Phase 2.5)
3. ✅ Multi-Light System - 13 lights with dynamic control (Phase 3.5)
4. ✅ PCSS Soft Shadows - Temporal filtering @ 115-120 FPS (Phase 3.6)
5. ✅ NVIDIA DLSS 3.7 Super Resolution (Ray Reconstruction shelved)
6. ✅ MCP server with 5 tools (added visual quality assessment)
7. ✅ F2 screenshot capture system
8. ✅ Variable refresh rate support
9. ✅ Screen-Space Contact Shadows (Phase 2)

**Current sprint priorities:**
1. 🔄 RTXDI M5 temporal accumulation (IN PROGRESS - Phase 4.1)
2. 🔄 C++ ONNX Runtime integration for PINN (IN PROGRESS - Phase 5)
3. ⏳ RT-based star radiance enhancements (scintillation, coronas, spikes) - 1-2 weeks
4. ⏳ Hybrid physics mode (PINN + traditional) - 2-3 days

**Deferred (Low Priority):**
- Fix non-working features (in-scattering F6, Doppler shift R, gravitational redshift G)
- Physics controls UI improvements
- Particle add/remove system (useful for testing)
- God rays system (shelved indefinitely - performance/quality issues)

**Roadmap (see MASTER_ROADMAP_V2.md for authoritative details):**
- **Phase 3.5-3.6:** Multi-light + PCSS ✅ COMPLETE
- **Phase 4 (Current):** RTXDI M5 + Shadow Quality 🔄 IN PROGRESS
- **Phase 5 (Current):** PINN ML Integration (Python ✅, C++ 🔄)
- **Phase 6 (Next):** Custom Temporal Denoising (not DLSS-RR)
- **Phase 7 (Future):** Enhanced Star Radiance Effects
- **Phase 8 (Long-term):** Celestial Body System (heterogeneous particles, LOD, material-aware RT)
- **Phase 9 (Long-term):** VR/AR Support (instanced stereo rendering)
---

**Last Updated:** 2025-11-09
**Project Version:** 0.14.4 (Based on git branch)
**Documentation maintained by:** Claude Code sessions

**Note:** See `MASTER_ROADMAP_V2.md` for the most up-to-date development status and detailed technical implementation plans.
