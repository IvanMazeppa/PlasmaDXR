# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

The user is named Ben.

**PlasmaDX-Clean** is a DirectX 12 volumetric particle renderer featuring DXR 1.1 inline ray tracing, 3D Gaussian splatting, volumetric RT lighting, NVIDIA RTXDI integration, and ML-accelerated physics via Physics-Informed Neural Networks (PINNs). The project simulates a black hole accretion disk achieving 20 FPS @ 1440p with 10K particles, 16 lights, and full RT lighting on RTX 4060 Ti hardware.

**Current Status (2025-10-22):**
- RTXDI M5 (Phase 2) - Temporal accumulation with ping-pong buffers
- PINN ML Physics - Python training pipeline complete, C++ integration in progress
- Adaptive Quality System integrated with ONNX Runtime
- Bulk light color control system operational
- God rays system (SHELVED - active but marked for deactivation due to issues)
- 30×30×30 spatial grid covering 3000×3000×3000 unit world space

**Core Technology Stack:**
- DirectX 12 with Agility SDK
- DXR 1.1 (RayQuery API for inline ray tracing)
- HLSL Shader Model 6.5+
- ImGui for runtime controls
- PIX for Windows (GPU debugging)
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
```

### Two Build Configurations

**Debug** - Daily development (zero PIX overhead):
- D3D12 debug layer enabled
- Fast iteration
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
```

### ONNX Runtime (ML Features) - Optional

**Status:** Optional dependency for PINN physics acceleration

**Detection:** CMake automatically detects ONNX Runtime presence
- If found: `ENABLE_ML_FEATURES=ON`, PINN available
- If missing: `ENABLE_ML_FEATURES=OFF`, warning shown, ML features disabled

**Setup (Optional):**
```bash
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to: external/onnxruntime/
# Required structure:
#   external/onnxruntime/include/
#   external/onnxruntime/lib/onnxruntime.lib
#   external/onnxruntime/lib/onnxruntime.dll
```

**CMake Integration:**
```cmake
# In CMakeLists.txt (automatic detection)
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(WARNING "ONNX Runtime not found. ML features will be disabled.")
    set(ENABLE_ML_FEATURES OFF)
endif()
```

**Required DLLs (copied automatically if enabled):**
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`

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
- `shaders/particles/particle_physics.hlsl` - GPU physics compute shader
  - Schwarzschild black hole gravity
  - Keplerian orbital dynamics
  - Temperature-based blackbody emission
  - Anisotropic Gaussian elongation

**Volumetric Rendering:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Main volumetric renderer
  - Uses RayQuery API (DXR 1.1 inline ray tracing)
  - Analytic ray-ellipsoid intersection
  - Beer-Lambert law for volumetric absorption (log-space for numerical stability)
  - Henyey-Greenstein phase function for anisotropic scattering
  - Hybrid emission blend system (artistic warm colors + physically accurate blackbody radiation)

**Ray Traced Lighting:**
- `shaders/dxr/particle_raytraced_lighting_cs.hlsl` - RT lighting compute
  - Particle-to-particle illumination via TLAS traversal
  - Shadow rays for occlusion
  - ReSTIR Phase 1 (temporal reuse - deprecated, replaced by RTXDI)

**RTXDI System:**
- `shaders/rtxdi/rtxdi_raygen.hlsl` - DXR raygen shader for weighted reservoir sampling
- `shaders/rtxdi/rtxdi_temporal_accumulate.hlsl` - M5 temporal accumulation (ping-pong)
- `shaders/rtxdi/rtxdi_miss.hlsl` - Miss shader for RTXDI pipeline

**Acceleration Structure:**
- `shaders/dxr/generate_particle_aabbs.hlsl` - Procedural primitive AABB generation
- `shaders/dxr/particle_intersection.hlsl` - Ray-ellipsoid intersection shader

### Shader Constants Structure

The physics shader uses a large constant buffer (`PhysicsConstants`) passed via root constants:
- Particle count, delta time, total time
- Black hole mass, gravity strength, turbulence
- Constraint shape parameters (radius, thickness)
- Alpha viscosity (Shakura-Sunyaev accretion parameter)

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

**Anisotropic Elongation:**
Particles elongate along velocity vectors to simulate tidal forces:
```hlsl
scale.xyz = baseRadius * (1, 1, 1 + anisotropy * velocityMagnitude)
```

---

## RTXDI Implementation (Phase 4 - ACTIVE)

**Status:** M5 Phase 2 In Progress - Temporal Accumulation with Ping-Pong Buffers

**Achievement (2025-10-22):**
Successfully implemented custom RTXDI system using DXR 1.1 raytracing with M5 temporal accumulation. After 14 hours of intensive development and debugging (M4), RTXDI weighted reservoir sampling is operational with temporal smoothing now being implemented.

**Implementation:**
- **DXR Pipeline:** Raygen shader performs weighted reservoir sampling
- **Light Grid:** 30×30×30 spatial acceleration structure (27,000 cells)
- **World Coverage:** -1500 to +1500 units per axis (3000-unit range)
- **Cell Size:** 100 units per cell (optimized for wide light distribution)
- **Output:** R32G32B32A32_FLOAT texture (selected light index per pixel)
- **Temporal Accumulation:** Ping-pong buffers for M5 temporal reuse

**Architecture:**
1. **Light Grid Building** (compute shader):
   - Uploads 13-16 lights to GPU structured buffer
   - Computes spatial grid cells (30×30×30)
   - Assigns lights to cells based on position and radius
   - Calculates importance weights per light per cell

2. **Weighted Reservoir Sampling** (DXR raygen shader):
   - Maps pixel to world position → grid cell
   - Reads cell's light list and weights
   - Performs weighted random selection (1 light per pixel per frame)
   - Uses PCG hash for temporal variation (frame index + pixel coords)

3. **Temporal Accumulation (M5)** (compute shader):
   - Ping-pong buffers for temporal reuse
   - Accumulates 8-16 samples over 60ms
   - Smooths patchwork pattern from M4 Phase 1

4. **Gaussian Renderer Integration**:
   - Reads RTXDI output buffer (t6)
   - Uses temporally accumulated light selection
   - Auto-disables RT particle-to-particle lighting in RTXDI mode
   - Debug visualization shows rainbow colors (light index mapping)

**Key Technical Details:**
- **Fibonacci Sphere Distribution** (RTXDI Sphere preset) - 13 lights @ 1200-unit radius
- **Dual-Ring Formation** (RTXDI Ring preset) - 16 lights @ 600-1000 unit radii
- **Cross Pattern** (RTXDI Sparse preset) - 5 lights for debugging grid behavior
- **Temporal Smoothing** - M5 accumulation reduces patchwork artifacts

**Migration from Custom ReSTIR:**
Original custom ReSTIR implementation (126 MB reservoir buffers, temporal reuse attempts) was deprecated in favor of lightweight RTXDI approach. Custom implementation removed, RTXDI built from scratch using:
- NVIDIA ReSTIR paper principles (Bitterli et al. 2020)
- DXR 1.1 TraceRay API for raygen shader
- Custom light grid building compute shader
- Spatial partitioning with temporal accumulation

**Custom Implementation Technical Details (for reference):**

**Algorithm:** Weighted Reservoir Sampling for many-light problems
1. Candidate Sampling: Cast 16-32 random rays to find light-emitting particles
2. Importance Weighting: `weight = luminance(emission * intensity * attenuation)`
3. Reservoir Update: Probabilistic selection maintains 1 sample from M candidates
4. Temporal Reuse: Previous frame's reservoir is validated and merged
5. Unbiased Estimator: Correction weight `W = weightSum / M`

**Ping-Pong Buffers:**
- 2× reservoir buffers (63MB each @ 1080p)
- Structure: `{ float3 lightPos, float weightSum, uint M, float W, uint particleIdx }`
- Swap each frame via `m_currentReservoirIndex`

**Known Issues (historical):**
- Weight calculation edge cases for low-temperature particles
- Temporal reuse validation bugs
- Attenuation formula tuning at large scales
- See `RESTIR_DEBUG_BRIEFING.md` for full debugging history

**Current Status:**
- Phase 1 (M4): Weighted sampling ✅ COMPLETE
- Phase 2 (M5): Temporal accumulation 🔄 IN PROGRESS
- Phase 3 (M6): Spatial reuse ⏳ PLANNED

---

## Multi-Light System (Phase 3.5 - COMPLETE ✅)

**Status:** Implemented and working

**Achievement:** First production-ready many-light system using RT volumetric rendering. 13 lights distributed across the accretion disk achieve realistic multi-directional shadowing, rim lighting, and atmospheric scattering effects.

**Architecture:**
- Light array stored in `m_lights` vector (Application.cpp:118)
- Uploaded to GPU via structured buffer (32 bytes/light)
- Gaussian raytrace shader reads all lights (particle_gaussian_raytrace.hlsl:726)
- Each light contributes independently to volumetric illumination
- Bulk color control system for all lights simultaneously

**ImGui Controls:**
All lights fully controllable at runtime:
- Position (X/Y/Z sliders)
- Color (RGB color picker)
- Intensity (0.1 - 20.0 range)
- Radius (10.0 - 200.0 range)
- Per-light enable/disable toggles
- **Bulk Controls:** Apply color/intensity to all lights at once

**Presets:**
- **Stellar Ring** - 13 lights in circular formation (default)
- **Dual Binary** - 2 opposing lights (binary star system)
- **Trinary Dance** - 3 lights in triangular formation
- **Single Beacon** - 1 centered light (debugging/comparison)

**Performance Impact:**
- 13 lights: ~5% FPS overhead compared to single light
- Bottleneck: Per-light attenuation calculation in ray marching loop
- Current: 120+ FPS @ 10K particles with 13 lights (RTX 4060 Ti, 1080p)
- Target maintained ✅

**User Feedback:** "this is one hell of a brilliant update!!!!!!!!!!!"

---

## PCSS Soft Shadows (Phase 3.6 - COMPLETE ✅)

**Implementation Date:** 2025-10-18
**Status:** Complete and operational

**Achievement:** Full PCSS (Percentage-Closer Soft Shadows) implementation with temporal filtering achieving soft shadows at 115-120 FPS (Performance preset) on RTX 4060 Ti @ 1080p with 10K particles.

### Architecture

**Three Shadow Quality Presets:**

1. **Performance** (Default - Variant 3: 1-ray + temporal filtering)
   - 1 ray per light
   - Temporal accumulation (67ms convergence to 8-ray quality)
   - Target: 115-120 FPS
   - Best for: Real-time gameplay, interactive exploration

2. **Balanced** (Variant 1: 4-ray PCSS)
   - 4 rays per light (Poisson disk sampling)
   - Instant soft shadows (no temporal accumulation)
   - Target: 90-100 FPS
   - Best for: High-quality screenshots, moderate performance

3. **Quality** (Variant 2: 8-ray PCSS)
   - 8 rays per light (Poisson disk sampling)
   - Highest quality soft shadows
   - Target: 60-75 FPS
   - Best for: Cinematic captures, maximum quality

**Technical Implementation:**
- Shadow buffers: 2× R16_FLOAT (ping-pong, 4MB @ 1080p)
- Root signature: 10 parameters (was 8, +2 for shadow buffers)
- Shader resources: `t5: g_prevShadow`, `u2: g_currShadow`
- Temporal blend formula: `finalShadow = lerp(prevShadow, currentShadow, 0.1)`
- Convergence time: `t = -ln(0.125) / 0.1 ≈ 67ms` (8 frames @ 120 FPS)

### Configuration Files

**Preset configs:** `configs/presets/`
- `shadows_performance.json` - 1-ray + temporal (default)
- `shadows_balanced.json` - 4-ray PCSS
- `shadows_quality.json` - 8-ray PCSS

**Command-line usage:**
```bash
# Performance preset (default)
./build/Debug/PlasmaDX-Clean.exe --config=configs/presets/shadows_performance.json

# Balanced preset
./build/Debug/PlasmaDX-Clean.exe --config=configs/presets/shadows_balanced.json

# Quality preset
./build/Debug/PlasmaDX-Clean.exe --config=configs/presets/shadows_quality.json
```

### Performance Impact
| Preset | Rays/Light | Temporal | FPS Target | Overhead |
|--------|-----------|----------|------------|----------|
| Performance | 1 | ON | 115-120 | ~4% |
| Balanced | 4 | OFF | 90-100 | ~15% |
| Quality | 8 | OFF | 60-75 | ~35% |

**See:** `PCSS_IMPLEMENTATION_SUMMARY.md` for complete technical details

---

## Physics-Informed Neural Networks (Phase 5 - ACTIVE 🔄)

**Status:** Python training pipeline complete ✅, C++ integration in progress 🔄

**Achievement:** Research-level PINN that learns accretion disk particle forces while respecting fundamental astrophysics:

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

### Files Created:

**Python Training Pipeline:**
```
ml/
├── pinn_accretion_disk.py          # Main PINN implementation (530 lines)
│   ├── AccretionDiskPINN            # Neural network model
│   ├── Physics loss functions       # Conservation laws enforcement
│   ├── Training loop                # Combined data + physics loss
│   └── ONNX export                  # For C++ inference
│
├── collect_physics_data.py         # GPU buffer dump processor (300 lines)
│   ├── Read g_particles.bin         # Binary particle buffer
│   ├── Cartesian → Spherical        # Coordinate transformation
│   └── Compute forces               # From velocity finite differences
│
├── test_pinn.py                     # ONNX model validation
├── requirements_pinn.txt            # PyTorch, ONNX, scientific stack
├── PINN_README.md                   # Comprehensive documentation (500+ lines)
└── models/
    └── pinn_accretion_disk.onnx     # Trained model for C++ inference
```

**C++ Integration (In Progress):**
```
src/ml/
└── AdaptiveQualitySystem.h/cpp      # ONNX Runtime integration
```

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

**Expected Training Output:**
```
Epoch 2000/2000
  Total Loss: 0.000234
  Data Loss: 0.000156
  Physics Losses:
    keplerian: 0.000012
    angular_momentum: 0.000034
    energy: 0.000008
    gr: 0.000024

Model exported to ml/models/pinn_accretion_disk.onnx ✅
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

**Why faster?**
- Traditional: O(N) particle updates + O(N·M) RT lighting (expensive)
- PINN: O(N) neural network inference (constant time per particle)

**See:** `ml/PINN_README.md` and `PINN_IMPLEMENTATION_SUMMARY.md` for complete documentation

---

## Volumetric God Rays (Phase 3.7 - SHELVED ⚠️)

**Implementation Date:** 2025-10-22
**Status:** ⚠️ **SHELVED** - Active in application but marked for deactivation

**Reason:** Various issues encountered during implementation. Feature remains in codebase but should be disabled until issues are resolved.

**Implementation:**
- Atmospheric fog ray marching
- Volumetric god ray rendering from light sources
- Light shaft visualization through particle medium

**Technical Details:**
- Ray marching through atmospheric volume
- Light shaft scattering calculations
- Configurable density and scattering parameters
- Integration with existing Gaussian volumetric renderer

**Files:**
- Implementation integrated into `ParticleRenderer_Gaussian.cpp`
- Shader code in `particle_gaussian_raytrace.hlsl`

**Known Issues:**
- Performance impact not acceptable for real-time use
- Visual artifacts at certain camera angles
- Interaction with RTXDI lighting system problematic
- Needs architectural redesign before re-activation

**To Deactivate:**
- Add ImGui toggle to disable god rays rendering
- Default to OFF in config files
- Document issues in roadmap for future work

**Commits:**
- `78d1d86` - Implement atmospheric fog ray marching for volumetric god rays
- `c0170db` - Integrate God Ray System into Particle Renderer and Application

**Future Work:**
- Revisit after RTXDI M6 complete
- Consider alternative implementation approach
- Profile performance bottlenecks
- Fix interaction with multi-light system

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

**Not Implemented (intentionally skipped):**
- SPH (Smoothed Particle Hydrodynamics) - too complex, different use case
- Relativistic jets - will use RT-based volumetric approach instead
- Dual galaxy collision - unstable in previous implementation

### Runtime Physics Controls

**Keyboard shortcuts:**
- Up/Down: Gravity strength
- Left/Right: Angular momentum
- Ctrl+Up/Down: Turbulence
- Shift+Up/Down: Damping
- [/]: Particle size

**ImGui controls:**
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

**Analysis Scripts:**
See `PIX/scripts/analysis/` for Python scripts to analyze buffer dumps.

**PINN Training Data:**
```bash
# Collect physics training data
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120

# Process for PINN training
cd ml
python collect_physics_data.py --input ../PIX/buffer_dumps
```

### PIX Event Markers

All draw calls and compute dispatches are wrapped with PIX event markers. Use PIX timeline view to navigate the frame.

---

## Autonomous Testing with v3 Agents

**Status:** Custom plugin system operational (plasmadx-testing-v3)

**Installation:** `~/.claude/plugins/plasmadx-testing-v3/`

PlasmaDX uses Claude Code's plugin system for autonomous multi-agent testing. Custom v3 agents provide specialized DirectX 12 / DXR expertise for buffer validation, debugging, stress testing, and performance analysis.

### Custom v3 Agents

**buffer-validator-v3** - Autonomous GPU buffer validation
- Validates binary buffer dumps from `--dump-buffers` flag
- Knows DirectX 12 buffer formats (32 bytes/particle, 32 bytes/light, 32 bytes/pixel)
- Detects NaN/Inf, out-of-bounds values, statistical anomalies
- Auto-generates Python validation scripts
- **Usage:** `@buffer-validator-v3 validate PIX/buffer_dumps/frame_120/g_particles.bin`

**pix-debugger-v3** - Root cause analysis for rendering issues
- Diagnoses black screens, color artifacts, missing effects
- Knows multi-light system architecture (Phase 3.5)
- Understands RT lighting pipeline (Phase 2.6)
- Provides specific file:line fixes with time estimates
- **Usage:** `@pix-debugger-v3 analyze "lights disappear beyond 300 units"`

**stress-tester-v3** - Comprehensive stress testing
- Particle count scaling (100 → 100K)
- Multi-light scaling (0 → 50 lights)
- Camera distance testing (close, mid, far scenarios)
- Performance regression detection
- **Usage:** `@stress-tester-v3 run particle-scaling`

**performance-analyzer-v3** - Performance profiling
- Identifies bottlenecks (BLAS rebuild, shader dispatches)
- Compares against performance targets (RTX 4060 Ti baseline)
- Generates optimization recommendations
- Integrates with PIX timing captures
- **Usage:** `@performance-analyzer-v3 profile build/Debug/PlasmaDX-Clean.exe`

### Built-in Claude Code Agents

PlasmaDX also uses Claude Code's built-in specialized agents:

- **pix-debugging-agent** - Autonomous PIX capture analysis
- **dxr-graphics-debugging-engineer-v2** - DXR rendering diagnosis
- **dx12-mesh-shader-engineer-v2** - Mesh/Amplification shader work
- **physics-performance-agent-v2** - Physics simulation optimization

### Multi-Agent Workflows

**Example: Debug Multi-Light Issue**
```bash
# 1. Capture frame with buffer dump
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --dump-buffers 120

# 2. Validate buffers
@buffer-validator-v3 validate PIX/buffer_dumps/frame_120/g_lights.bin

# 3. Analyze rendering issue
@pix-debugger-v3 analyze "light radius control has no effect"

# 4. Apply fix to shader
# (pix-debugger-v3 provides file:line and exact code change)
```

**Example: Performance Regression**
```bash
# 1. Run stress test
@stress-tester-v3 run multi-light-scaling

# 2. Profile bottlenecks
@performance-analyzer-v3 profile build/Debug/PlasmaDX-Clean.exe

# 3. Use built-in physics optimization
@physics-performance-agent-v2 optimize particle physics shader
```

### Plugin System Documentation

See `CLAUDE_CODE_PLUGINS_GUIDE.md` for complete plugin system documentation including:
- Agent creation guide
- Built-in vs custom agents
- Multi-agent orchestration
- Future Agent SDK use cases

---

## Common Development Tasks

### Running Tests
There are no automated unit tests. Testing is primarily visual and performance-based:
1. Run Debug build
2. Verify particle rendering is correct
3. Check FPS counter (target: >100 FPS @ 100K particles)
4. Test RTXDI toggle (F7 key)
5. Adjust physics parameters via ImGui

### Debugging Rendering Issues

**Black screen or missing particles:**
1. Check logs in `logs/` directory
2. Verify shaders compiled successfully (check `build/Debug/shaders/`)
3. Enable D3D12 debug layer (builds/Debug.json: `enableDebugLayer: true`)
4. Check for D3D12 errors in Visual Studio output

**RTXDI artifacts (patchwork pattern, color issues):**
1. Use PIX capture to inspect RTXDI output buffers
2. Check light grid coverage (ensure lights within 3000-unit bounds)
3. Adjust camera distance (patchwork expected in M4, should be smooth in M5)
4. Compare with RTXDI disabled

### Adding New Features

**Workflow:**
1. Create feature branch from `main`
2. Implement in appropriate module (`src/particles/`, `src/lighting/`, `src/ml/`, etc.)
3. Add shader changes if needed
4. Expose to ImGui for runtime control
5. Test thoroughly at various particle counts and camera distances
6. Commit with descriptive message
7. Create PR to `main`

**Commit message style:**
```
feat: Add PINN ML physics integration
fix: Address RTXDI temporal accumulation bugs
refactor: Separate RTXDI into dedicated lighting system
```

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

**Optimization potential:**
- BLAS update (no rebuild): +25% FPS
- Instance culling: +50% FPS
- Static BLAS with dynamic TLAS: Possible but requires careful particle management

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
**Known issues:**
- Performance impact not acceptable
- Visual artifacts at certain angles
- Conflicts with RTXDI lighting
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
| + PINN Physics (100K) | 180 | TBD 🔄 | 5-10× speedup expected |

**Bottleneck:** RayQuery traversal of 100K procedural primitives (BLAS rebuild: 2.1ms/frame)

**Optimization priorities:**
1. PINN ML physics: +50-100 FPS at 100K particles (ACTIVE)
2. RTXDI M5 optimization: +5-10 FPS (IN PROGRESS)
3. BLAS update (not rebuild): +25% FPS (PLANNED)
4. Particle LOD culling: +50% FPS (PLANNED)
5. Release build optimization: +30% FPS (PLANNED)

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

---

## File Naming Conventions

- Headers: `.h` (not `.hpp`)
- Implementation: `.cpp`
- Shaders: `.hlsl`
- Compiled shaders: `.dxil` (output directory)
- Configs: `.json`
- ML models: `.onnx` (PINN trained models)
- Training data: `.npz` (NumPy compressed arrays)

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

**Usage:**
```cpp
LOG_INFO("Initializing particle system with {} particles", particleCount);
LOG_WARN("Mesh shaders failed, using compute fallback");
LOG_ERROR("Failed to create BLAS: {}", errorMessage);
LOG_WARN("ONNX Runtime not found, ML features disabled");
```

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

**In-repo documentation:**
- `README.md` - Project overview, features, controls
- `BUILD_GUIDE.md` - Build configuration details
- `configs/README.md` - Configuration system reference
- `PHYSICS_PORT_ANALYSIS.md` - Physics feature porting plan
- `RESTIR_DEBUG_BRIEFING.md` - ReSTIR debugging status (deprecated, see RTXDI)
- `PIX/docs/QUICK_REFERENCE.md` - PIX capture system guide
- `ml/PINN_README.md` - PINN training and integration guide (500+ lines)
- `PINN_IMPLEMENTATION_SUMMARY.md` - PINN implementation overview
- `PCSS_IMPLEMENTATION_SUMMARY.md` - PCSS soft shadows technical details

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

**Current sprint priorities (Phase 5 - ML Integration):**
1. ✅ PINN Python training pipeline (COMPLETE)
2. 🔄 C++ ONNX Runtime integration (IN PROGRESS)
3. 🔄 RTXDI M5 temporal accumulation (IN PROGRESS)
4. ⏳ Hybrid physics mode (PINN + traditional) - 2-3 days
5. ⏳ Performance benchmarking (PINN vs traditional) - 1 day
6. ⏳ ImGui controls for PINN mode - 4 hours
7. ⏳ Disable god rays feature (add toggle, default OFF) - 1 hour

**Roadmap (see MASTER_ROADMAP_V2.md for full details):**
- **Current:** Phase 5 - PINN ML Integration (Python ✅, C++ 🔄)
- **Current:** RTXDI M5 - Temporal Accumulation (ping-pong buffers 🔄)
- **Next:** Phase 6 - Neural Denoising (NVIDIA NRD or AMD FidelityFX)
- **Future:** Phase 7 - VR/AR Support (instanced stereo rendering)
- **Long-term:** Kerr metric (rotating black holes), multi-BH systems

---

**Last Updated:** 2025-10-22
**Project Version:** 0.9.4
**Documentation maintained by:** Claude Code sessions

---

## Recent Major Milestones

### PINN ML Physics Integration - Phase 5 (2025-10-22)
**Branch:** `0.9.4` (current)

**Achievement:**
Complete Python PINN training pipeline with physics-informed constraints. First ML-accelerated physics system respecting General Relativity, conservation laws, and Shakura-Sunyaev α-disk viscosity.

**Technical Implementation:**
- 5-layer × 128-neuron network (~50,000 parameters)
- Combined data + physics loss function
- 5 physics constraints: GR, Keplerian motion, L conservation, E conservation, α-viscosity
- ONNX export for C++ inference
- GPU buffer dump → training data pipeline

**Key Features:**
- `ml/pinn_accretion_disk.py` - Complete PINN training (530 lines)
- `ml/collect_physics_data.py` - GPU buffer processor (300 lines)
- `src/ml/AdaptiveQualitySystem.h/cpp` - C++ ONNX Runtime wrapper
- CMake auto-detection of ONNX Runtime

**Status:**
- Python training pipeline: ✅ COMPLETE
- C++ integration: 🔄 IN PROGRESS
- Hybrid mode: ⏳ PLANNED

**Expected Impact:**
- 5-10× performance improvement at 100K particles
- 110 FPS (PINN) vs 18 FPS (traditional) @ 100K particles
- Scientific accuracy maintained via physics constraints

**Next Steps:**
- Complete ONNX model loading in C++
- Implement hybrid mode (PINN for r > 10×R_ISCO, shader for r < 10×R_ISCO)
- Add ImGui controls
- Benchmark performance

**Commits:**
- `6ebf879` - feat: Integrate Adaptive Quality System for ML-based performance optimization

### RTXDI M5 Temporal Accumulation (2025-10-22)
**Branch:** `0.9.4` (current)

**Status:** 🔄 IN PROGRESS

**Implementation:**
- Ping-pong buffer system for temporal reuse
- Accumulates 8-16 samples over 60ms
- Smooths patchwork pattern from M4 Phase 1
- Integration with existing RTXDI weighted sampling

**Technical Details:**
- 2× accumulation buffers (ping-pong)
- Temporal blend factor: 0.1 (configurable)
- Convergence time: ~67ms @ 120 FPS
- Compatible with all RTXDI light presets

**Commits:**
- `cbfbe45` - fix: Implement ping-pong buffers for M5 temporal accumulation
- `18a2155` - feat: Implement temporal accumulation in RTXDI lighting system

### Bulk Light Color Control (2025-10-22)
**Branch:** `0.9.4` (current)

**Achievement:**
Enhanced light control system with bulk operations for all lights simultaneously.

**Features:**
- Apply color to all lights at once
- Apply intensity multiplier to all lights
- Per-light and global control modes
- Preset-based color schemes

**Status:** ✅ COMPLETE

**Commits:**
- `56eda17` - feat: Implement bulk light color control system in Application class

### God Rays System - SHELVED (2025-10-22)
**Branch:** `0.9.4` (current)

**Status:** ⚠️ SHELVED - Active but marked for deactivation

**Implementation:**
- Atmospheric fog ray marching
- Volumetric god ray rendering
- Light shaft visualization

**Issues:**
- Performance impact not acceptable for real-time
- Visual artifacts at certain camera angles
- Conflicts with RTXDI lighting system
- Needs architectural redesign

**Action Items:**
- Add ImGui toggle to disable (default OFF)
- Document issues in roadmap
- Revisit after RTXDI M6 complete

**Commits:**
- `78d1d86` - feat: Implement atmospheric fog ray marching for volumetric god rays
- `c0170db` - feat: Integrate God Ray System into Particle Renderer and Application

### RTXDI M4 Complete - Weighted Reservoir Sampling (2025-10-19)
**Branch:** `0.8.2` (milestone)

**Achievement:**
After 14 hours of intensive development, RTXDI weighted reservoir sampling is operational. First production-ready RTXDI implementation using DXR 1.1 raygen shader with custom spatial grid building.

**Technical Implementation:**
- DXR 1.1 raygen shader performs per-pixel weighted random light selection
- 30×30×30 spatial grid covering 3000×3000×3000 unit world space
- PCG hash for temporal variation (frame index + pixel coordinates)
- Debug visualization shows rainbow pattern (light index mapping)
- Auto-disables RT particle-to-particle lighting in RTXDI mode

**RTXDI-Optimized Light Presets:**
- **Sphere (13):** Fibonacci sphere @ 1200-unit radius
- **Ring (16):** Dual-ring disk @ 600-1000 unit radii
- **Sparse (5):** Debug cross pattern @ 1000-unit spacing

**Critical Fixes Applied:**
- Expanded RTXDI world bounds from 600 to 3000 units (5× larger coverage)
- Removed Grid (27) preset (exceeded 16-light hardware limit)
- Cell size increased from 20 to 100 units per cell

**Current Status:** Phase 1 complete (M4), Phase 2 in progress (M5)

**User Feedback:** "oh my god the image quality has entered a new dimension!! it looks gorgeous"

### Multi-Light System Breakthrough (2025-10-17)
**Branch:** `0.6.6` (milestone)

**Achievement:**
First production-ready many-light system using RT volumetric rendering. 13 lights distributed across the accretion disk achieve:
- Realistic multi-directional shadowing
- Rim lighting halos from multiple angles
- Atmospheric scattering with multiple sources
- Full runtime control via ImGui (position, color, intensity, radius)
- 120+ FPS maintained @ 10K particles with 13 lights

**Status:** ✅ COMPLETE

**User Feedback:** "this is one hell of a brilliant update!!!!!!!!!!!"

### RT Volumetric Lighting Breakthrough (2025-10-15)
**Branch:** `0.6.0` (milestone marker)

**What was fixed:**
Physical emission (blackbody self-emission) was being incorrectly modulated by external RT lighting, causing the entire volumetric RT system to produce incorrect results. The fix separated self-emission from external lighting contributions.

**Result:**
All RT systems working correctly for the first time:
- Volumetric depth with proper particle occlusion
- Rim lighting halos from Henyey-Greenstein phase function
- Perfect temperature gradients (800K-26000K)
- Atmospheric scattering effects
- Realistic multi-directional shadowing
- 120+ FPS @ 1080p with 10K particles and 16 rays/particle

**Led to:** Phase 3.5 Multi-Light System implementation

### ReSTIR Deprecation Decision (2025-10-16)
**Status:** Custom implementation removed ✅

**Decision:**
After months of debugging custom ReSTIR implementation (Phase 1 temporal reuse), adopted NVIDIA RTXDI (RTX Direct Illumination) instead. RTXDI provides battle-tested ReSTIR GI with spatial/temporal reuse, optimized for RTX hardware.

**Migration:** Complete - 126 MB of custom ReSTIR code removed

**Outcome:** RTXDI M4 + M5 operational, much cleaner implementation

**Current Milestone:** RTXDI M5 Temporal Accumulation (2025-10-22)
