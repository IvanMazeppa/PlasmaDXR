# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

The user is named Ben.

**PlasmaDX-Clean** is a DirectX 12 volumetric particle renderer featuring DXR 1.1 inline ray tracing, 3D Gaussian splatting, volumetric RT lighting, and NVIDIA RTXDI integration. The project simulates a black hole accretion disk achieving 20 FPS @ 1440p with 10K particles, 16 lights, and full RT lighting on RTX 4060 Ti hardware.

**Current Status (2025-10-19):**
- RTXDI M4 (Phase 1) Complete - Weighted reservoir sampling operational
- RTXDI-optimized light presets implemented (Sphere, Ring, Sparse)
- 30×30×30 spatial grid covering 3000×3000×3000 unit world space
- Screenshot automation tools for Windows/WSL workflow

**Core Technology Stack:**
- DirectX 12 with Agility SDK
- DXR 1.1 (RayQuery API for inline ray tracing)
- HLSL Shader Model 6.5+
- ImGui for runtime controls
- PIX for Windows (GPU debugging)

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

**Common workflow:**
```bash
# Default run
./build/Debug/PlasmaDX-Clean.exe

# Custom config
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/stress_test.json
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
- Builds BLAS/TLAS acceleration structures
- Computes particle-to-particle illumination
- **IMPORTANT:** The Gaussian renderer reuses the TLAS from this system (no duplicate infrastructure)

**Utilities** (`src/utils/`):
- `ShaderManager.h/cpp` - DXIL loading and reflection
- `ResourceManager.h/cpp` - Buffer/texture/descriptor pool management
- `Logger.h/cpp` - Timestamped logging to `logs/` directory

### Key Architecture Principles

1. **Feature Detection First** - Always test capabilities before using (RT tier, mesh shaders)
2. **Single Responsibility** - No 4,000-line monoliths; max ~500 lines per file
3. **Automatic Fallbacks** - Mesh shader failure → compute shader fallback
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
  - ReSTIR Phase 1 (temporal reuse - in active development)

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

**Status:** M4 Phase 1 Complete ✅ - Weighted Reservoir Sampling Operational

**Achievement (2025-10-19):**
Successfully implemented custom RTXDI system using DXR 1.1 raytracing. After 14 hours of intensive development and debugging, RTXDI weighted reservoir sampling is working correctly with visible patchwork pattern (expected Phase 1 behavior).

**Implementation:**
- **DXR Pipeline:** Raygen shader performs weighted reservoir sampling
- **Light Grid:** 30×30×30 spatial acceleration structure (27,000 cells)
- **World Coverage:** -1500 to +1500 units per axis (3000-unit range)
- **Cell Size:** 100 units per cell (optimized for wide light distribution)
- **Output:** R32G32B32A32_FLOAT texture (selected light index per pixel)

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

3. **Gaussian Renderer Integration**:
   - Reads RTXDI output buffer (t6)
   - Uses single selected light per pixel (Phase 1)
   - Auto-disables RT particle-to-particle lighting in RTXDI mode
   - Debug visualization shows rainbow colors (light index mapping)

**Key Technical Details:**
- **Fibonacci Sphere Distribution** (RTXDI Sphere preset) - 13 lights @ 1200-unit radius
- **Dual-Ring Formation** (RTXDI Ring preset) - 16 lights @ 600-1000 unit radii
- **Cross Pattern** (RTXDI Sparse preset) - 5 lights for debugging grid behavior
- **Patchwork Pattern** - Expected Phase 1 behavior (1 sample/pixel/frame), will smooth with M5

**Migration from Custom ReSTIR:**
Original custom ReSTIR implementation (126 MB reservoir buffers, temporal reuse attempts) was deprecated in favor of lightweight RTXDI approach. Custom implementation removed, RTXDI built from scratch using:
- NVIDIA ReSTIR paper principles (Bitterli et al. 2020)
- DXR 1.1 TraceRay API for raygen shader
- Custom light grid building compute shader
- Spatial partitioning (no temporal reuse yet)

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

**Migration Path:**
- Phase 4: Remove custom ReSTIR code
- Phase 4: Integrate NVIDIA RTXDI SDK
- Phase 4: Port existing lighting to RTXDI API

---

## Multi-Light System (Phase 3.5 - Current Priority)

**Status:** Implemented and working, 3 polish fixes needed

**Achievement:** First production-ready many-light system using RT volumetric rendering. 13 lights distributed across the accretion disk achieve realistic multi-directional shadowing, rim lighting, and atmospheric scattering effects.

**Architecture:**
- Light array stored in `m_lights` vector (Application.cpp:118)
- Uploaded to GPU via structured buffer (32 bytes/light)
- Gaussian raytrace shader reads all lights (particle_gaussian_raytrace.hlsl:726)
- Each light contributes independently to volumetric illumination

**ImGui Controls:**
All lights fully controllable at runtime:
- Position (X/Y/Z sliders)
- Color (RGB color picker)
- Intensity (0.1 - 20.0 range)
- Radius (10.0 - 200.0 range)
- Per-light enable/disable toggles

**Presets:**
- **Stellar Ring** - 13 lights in circular formation (default)
- **Dual Binary** - 2 opposing lights (binary star system)
- **Trinary Dance** - 3 lights in triangular formation
- **Single Beacon** - 1 centered light (debugging/comparison)

**Current Issues (see MULTI_LIGHT_FIXES_NEEDED.md):**

1. **Sphere Boundary Issue** (particle_gaussian_raytrace.hlsl:726)
   - **Symptom:** Light appears to vanish beyond ~300-400 units
   - **Root Cause:** Attenuation falloff formula too steep (`1.0 / (1.0 + lightDist * 0.01)`)
   - **Fix:** Reduce falloff constant from 0.01 to 0.001 (10x wider range)
   - **File:** shaders/particles/particle_gaussian_raytrace.hlsl:726
   - **Time:** 5 minutes

2. **Light Radius Has No Effect** (particle_gaussian_raytrace.hlsl:726)
   - **Symptom:** Adjusting light radius slider does nothing
   - **Root Cause:** Shader uploads `light.radius` but never uses it in attenuation calculation
   - **Fix:** Normalize distance by radius: `float normalizedDist = lightDist / max(light.radius, 1.0);`
   - **File:** shaders/particles/particle_gaussian_raytrace.hlsl:726
   - **Time:** 5 minutes

3. **Can't Fully Disable RT Lighting** (Application.cpp:1850, Application.h:118)
   - **Symptom:** RT particle-to-particle lighting (Phase 2.6) always active, only strength adjustable
   - **Root Cause:** No boolean toggle, only `m_rtLightingStrength` slider (0.0-5.0)
   - **Fix:** Add `bool m_enableRTLighting` checkbox in ImGui, apply in Update() loop
   - **Files:**
     - src/core/Application.h:118 (add bool member)
     - src/core/Application.cpp:1850 (add ImGui checkbox)
     - src/core/Application.cpp:340 (apply toggle before upload)
   - **Time:** 15 minutes

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

### ImGui Controls

**UI Layout:**
```
Rendering Features
└─ Shadow Rays (F5) [Checkbox]
   ├─ Shadow Quality
   ├─ Preset: [Dropdown: Performance|Balanced|Quality|Custom]
   ├─ Info: "1-ray + temporal (120 FPS target)"
   └─ Custom Controls (if Custom selected)
      ├─ Rays Per Light: [Slider: 1-16]
      ├─ Temporal Filtering: [Checkbox]
      └─ Temporal Blend: [Slider: 0.0-1.0] (with tooltip)
```

**Preset auto-apply:**
- Changing preset dropdown instantly updates all shadow parameters
- No restart required for runtime switching
- Visual feedback via color-coded FPS targets (green/yellow/red)

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

### Technical Details

**Temporal Filtering Algorithm:**
1. Accumulate shadow samples during volume ray march
2. Calculate average shadow value per pixel
3. Read previous frame's shadow value
4. Blend: `lerp(prevShadow, currentShadow, temporalBlend)`
5. Write to current shadow buffer for next frame

**PCSS Multi-Ray Sampling:**
- Poisson disk samples (16 precomputed samples)
- Per-pixel random rotation for temporal stability
- Tangent-space disk sampling perpendicular to light direction
- Light radius controls penumbra size (soft shadow spread)

**Performance Impact:**
| Preset | Rays/Light | Temporal | FPS Target | Overhead |
|--------|-----------|----------|------------|----------|
| Performance | 1 | ON | 115-120 | ~4% |
| Balanced | 4 | OFF | 90-100 | ~15% |
| Quality | 8 | OFF | 60-75 | ~35% |

### Files Modified

**C++ Headers:**
- `src/particles/ParticleRenderer_Gaussian.h` - Shadow buffer resources, RenderConstants
- `src/core/Application.h` - ShadowPreset enum, control variables

**C++ Implementation:**
- `src/particles/ParticleRenderer_Gaussian.cpp` - Buffer creation, root signature, bindings
- `src/core/Application.cpp` - ImGui controls, constant upload

**Shaders:**
- `shaders/particles/gaussian_common.hlsl` - Poisson disk, Hash12(), Rotate2D()
- `shaders/particles/particle_gaussian_raytrace.hlsl` - CastPCSSShadowRay(), temporal filtering

**Configs:**
- `configs/presets/shadows_performance.json`
- `configs/presets/shadows_balanced.json`
- `configs/presets/shadows_quality.json`

**Documentation:**
- `PCSS_IMPLEMENTATION_SUMMARY.md` - Complete implementation summary

### Known Limitations

1. **Temporal artifacts during fast camera movement**
   - Motion blur visible during rapid camera rotation
   - Future: Motion vector-based reprojection

2. **Light radius dependency**
   - Penumbra size tied to light.radius parameter
   - Requires per-light tuning for realistic soft shadows

3. **Convergence delay (Performance mode only)**
   - 67ms gradual shadow softening
   - Acceptable trade-off for 120 FPS performance

### Future Enhancements (Optional)

**Phase 2:**
- Motion vector-based reprojection (prevent blur)
- Adaptive sampling (more rays in penumbra)
- Variance-based convergence detection

**Phase 3:**
- Blocker distance estimation (true PCSS penumbra sizing)
- Contact-hardening shadows (distance-dependent softness)
- Blue noise sampling (better distribution)

### Integration Notes

**Compatibility:**
- ✅ Multi-light system (Phase 3.5) - All 13 lights support soft shadows
- ✅ ReSTIR reservoir system - Independent, no conflicts
- ✅ Physical emission modes - Works with all emission types
- ✅ Phase function scattering - Shadow rays respect phase function
- ✅ Anisotropic Gaussians - Compatible with anisotropic elongation

**See:** `PCSS_IMPLEMENTATION_SUMMARY.md` for complete technical details

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
RayQuery (volumetric render) → RayQuery (shadow rays) → RayQuery (ReSTIR sampling)
```

**Three uses of RayQuery:**
1. **Main Rendering** - Volume ray marching through sorted particles
2. **Shadow Rays** - Occlusion testing to primary light source
3. **ReSTIR Sampling** - Random rays to find light sources

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

**In Progress (see PHYSICS_PORT_ANALYSIS.md):**
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
- `g_particles.bin` - Particle positions, velocities, temperatures
- `g_currentReservoirs.bin` - Current frame ReSTIR reservoirs
- `g_prevReservoirs.bin` - Previous frame ReSTIR reservoirs
- `g_rtLighting.bin` - Pre-computed RT lighting

**Analysis Scripts:**
See `PIX/scripts/analysis/` for Python scripts to analyze buffer dumps.

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
4. Test ReSTIR toggle (F7 key)
5. Adjust physics parameters via ImGui

### Debugging Rendering Issues

**Black screen or missing particles:**
1. Check logs in `logs/` directory
2. Verify shaders compiled successfully (check `build/Debug/shaders/`)
3. Enable D3D12 debug layer (builds/Debug.json: `enableDebugLayer: true`)
4. Check for D3D12 errors in Visual Studio output

**ReSTIR artifacts (dots, color issues):**
1. Use PIX capture to inspect reservoir buffers
2. Check `RESTIR_DEBUG_BRIEFING.md` for known issues
3. Adjust camera distance (bugs appear at close range ~100-200 units)
4. Compare with ReSTIR disabled (F7 key)

### Adding New Features

**Workflow:**
1. Create feature branch from `main`
2. Implement in appropriate module (`src/particles/`, `src/lighting/`, etc.)
3. Add shader changes if needed
4. Expose to ImGui for runtime control
5. Test thoroughly at various particle counts and camera distances
6. Commit with descriptive message
7. Create PR to `main`

**Commit message style:**
```
feat: Add ReSTIR spatial reuse (Phase 2)
fix: Address buffer dump feature issues
refactor: Separate RTLightingSystem into RayQuery variant
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

### ReSTIR Phase 1 Debugging
**Status:** Active development
**Known bugs:**
- ✅ FIXED: Weight threshold too high for low-temp particles
- ✅ FIXED: Temporal reuse allows M > 0 with weightSum = 0
- 🔄 TESTING: Attenuation formula tuning

See `RESTIR_DEBUG_BRIEFING.md` for current status.

### ImGui Integration
**Quirk:** ImGui requires a separate descriptor heap for fonts/textures. The Application class manages this automatically. Don't create additional ImGui descriptor heaps.

---

## Performance Targets

**Test Configuration:** RTX 4060 Ti, 1920×1080, 100K particles

| Feature Set | Target FPS | Current |
|-------------|------------|---------|
| Raster Only | 245 | 245 ✅ |
| + RT Lighting | 165 | 165 ✅ |
| + Shadow Rays | 142 | 142 ✅ |
| + Phase Function | 138 | 138 ✅ |
| + ReSTIR (active) | 120 | 120 ✅ |

**Bottleneck:** RayQuery traversal of 100K procedural primitives (BLAS rebuild: 2.1ms/frame)

**Optimization priorities:**
1. BLAS update (not rebuild): +25% FPS
2. Particle LOD culling: +50% FPS
3. Release build optimization: +30% FPS

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

**Naming style:**
- Classes: PascalCase (`ParticleSystem`, `RTLightingSystem_RayQuery`)
- Functions: PascalCase (`Initialize`, `Render`)
- Variables: camelCase (`m_particleCount`, `particleBuffer`)
- Constants: UPPER_SNAKE_CASE (`BLACK_HOLE_MASS`)

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
```

---

## Dependencies and External Libraries

**Included in repository:**
- DirectX 12 Agility SDK (`external/D3D12/`)
- ImGui (`external/imgui/`)
- d3dx12.h helper library (`src/utils/d3dx12.h`)

**System dependencies (must be installed):**
- Visual Studio 2022 (C++17 required)
- Windows SDK 10.0.26100.0 or higher
- DXC shader compiler (part of Windows SDK)
- PIX for Windows (optional, for GPU debugging)

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
- `RESTIR_DEBUG_BRIEFING.md` - ReSTIR debugging status
- `PIX/docs/QUICK_REFERENCE.md` - PIX capture system guide

**External references:**
- [DirectX 12 Programming Guide](https://docs.microsoft.com/en-us/windows/win32/direct3d12/)
- [DXR 1.1 Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)
- [ReSTIR Paper](https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling) - Bitterli et al. (2020)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al. (2023)

---

## Immediate Next Steps for Development

**Current sprint priorities (Phase 3.5 - Multi-Light Polish):**
1. Fix light radius control (shader bug at particle_gaussian_raytrace.hlsl:726) - 5 minutes
2. Fix sphere boundary issue (attenuation falloff too steep) - 5 minutes
3. Add RT lighting toggle (Application.h/cpp changes) - 15 minutes
4. Test all 3 fixes with stress-tester-v3 agent

**Roadmap (see MASTER_ROADMAP_V2.md for full details):**
- **Current:** Phase 3.5 - Multi-Light System polish (3 fixes remaining)
- **Next:** Phase 4 - RTXDI Integration (replace custom ReSTIR with NVIDIA library)
- **Future:** Phase 5 - Celestial Bodies (planet rendering, stellar surface simulation)
- **Long-term:** Phase 6 - Neural Denoising, Phase 7 - VR/AR Support

---

**Last Updated:** 2025-10-19
**Project Version:** 0.8.2
**Documentation maintained by:** Claude Code sessions

---

## Recent Major Milestones

### RTXDI M4 Complete - Weighted Reservoir Sampling (2025-10-19)
**Branch:** `0.8.2` (current)

**Achievement:**
After 14 hours of intensive development, RTXDI weighted reservoir sampling is operational. First production-ready RTXDI implementation using DXR 1.1 raygen shader with custom spatial grid building.

**Technical Implementation:**
- DXR 1.1 raygen shader performs per-pixel weighted random light selection
- 30×30×30 spatial grid covering 3000×3000×3000 unit world space
- PCG hash for temporal variation (frame index + pixel coordinates)
- Debug visualization shows rainbow pattern (light index mapping)
- Auto-disables RT particle-to-particle lighting in RTXDI mode

**RTXDI-Optimized Light Presets:**
- **Sphere (13):** Fibonacci sphere @ 1200-unit radius (60-80% patchwork reduction)
- **Ring (16):** Dual-ring disk @ 600-1000 unit radii (accretion disk aesthetic)
- **Sparse (5):** Debug cross pattern @ 1000-unit spacing

**Critical Fixes Applied:**
- Expanded RTXDI world bounds from 600 to 3000 units (5× larger coverage)
- Removed Grid (27) preset (exceeded 16-light hardware limit)
- Cell size increased from 20 to 100 units per cell

**Current Status:** Phase 1 complete (weighted sampling operational), Phase 2 (temporal reuse) pending

**User Feedback:** "oh my god the image quality has entered a new dimension!! it looks gorgeous"

**Next Steps:** M5 Temporal Reuse (accumulate 8-16 samples over 60ms to smooth patchwork pattern)

### Multi-Light System Breakthrough (2025-10-17)
**Branch:** `0.6.6` (current)

**Achievement:**
First production-ready many-light system using RT volumetric rendering. 13 lights distributed across the accretion disk achieve:
- Realistic multi-directional shadowing
- Rim lighting halos from multiple angles
- Atmospheric scattering with multiple sources
- Full runtime control via ImGui (position, color, intensity, radius)
- 120+ FPS maintained @ 10K particles with 13 lights

**Current Status:** 3 polish fixes needed (total: 25 minutes)
1. Light radius control (shader doesn't use uploaded parameter)
2. Sphere boundary (attenuation falloff too steep beyond 300 units)
3. RT lighting toggle (can't fully disable particle-to-particle lighting)

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
**Status:** Custom implementation marked for deletion

**Decision:**
After months of debugging custom ReSTIR implementation (Phase 1 temporal reuse), decided to adopt NVIDIA RTXDI (RTX Direct Illumination) instead. RTXDI provides battle-tested ReSTIR GI with spatial/temporal reuse, optimized for RTX hardware.

**Migration:** Phase 4 will remove custom code and integrate RTXDI SDK

**Next Major Milestone:** Phase 4 - RTXDI Integration (Q1 2026)

