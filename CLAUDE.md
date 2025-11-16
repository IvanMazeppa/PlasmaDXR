# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

The user is named Ben, he is a novice programmer but has some experience with C++, Java and Python. The user has high-functioning autism and so tends to develop a very strong interest in a technical field, and so this has blossomed into a passion project for Ben. He has a very strong interest in AI, ML and LLMs, and is leveraging these tools to create thi experimental raytracing engine.

### Collaboration Preferences

**Communication Style:**
- **Be corrective when wrong** - If Ben has a misunderstanding (especially about SDK/API usage), correct it immediately but kindly
- **Explain the "why"** - Don't just say "this is wrong," explain why and what the correct approach is
- **Validate effort** - Acknowledge when Ben's approach was reasonable even if technically incorrect
- **Show what's salvageable** - When something needs fixing, emphasize what work can be reused

**When Ben is uncertain or frustrated:**
- Be supportive but honest
- Break down complex problems into manageable steps
- Provide concrete estimates ("this will take 2 hours" not "this might take a while")
- Show immediate progress when possible (build something that works quickly)

**Problem-solving approach:**
- Explain architectural patterns clearly
- Use comparisons to help understanding ("X is like Y, but...")
- Provide working code examples, not just descriptions
- Test ideas immediately rather than just planning

### Feedback Philosophy: Brutal Honesty

**CRITICAL:** When providing feedback, code reviews, or analysis (especially from autonomous agents), **brutal honesty is strongly preferred over sugar-coating**.

✅ **Good:** "ZERO LIGHTS ACTIVE - this is catastrophic, cannot assess visual quality"
❌ **Bad:** "Lighting could use some refinement to improve visual quality"

**Why:** Sugar-coated responses have caused problems by hiding critical issues. Direct, specific language accelerates debugging and saves development time.

---

**PlasmaDX-Clean** is a DirectX 12 volumetric particle renderer featuring DXR 1.1 inline ray tracing, 3D Gaussian splatting, volumetric RT lighting, NVIDIA RTXDI integration, and ML-accelerated physics via Physics-Informed Neural Networks (PINNs). Simulates a black hole accretion disk achieving 20 FPS @ 1440p with 10K particles, 16 lights, and full RT lighting on RTX 4060 Ti hardware. As this is an experiment into various RT technologies, RT lighting, shadowing (etc) should always be the first choice when deciding on a solution for an upgrade, but ONLY if RT is appropriate. RT should never be forced in just for the sake of using it, it should always benefit the project and impove image quality.

**Current Status (2025-11-12):**
- **Primary Renderer:** Gaussian volumetric RT lighting (particle_gaussian_raytrace.hlsl) ✅ ACTIVE
- **RTXDI System:** M5 temporal accumulation 🔄 WORK-IN-PROGRESS (not currently used as primary renderer due to quality issues; will switch once problems resolved)
- **Multi-Light System:** 13 lights with dynamic control ✅ COMPLETE
- **PCSS Soft Shadows:** Temporal filtering at 115-120 FPS ✅ COMPLETE
- **NVIDIA DLSS 3.7:** Super Resolution operational ✅ COMPLETE
- **PINN ML Physics:** Python training complete, C++ integration in progress 🔄
- **Adaptive Particle Radius:** Camera-distance adaptive sizing ✅ COMPLETE
- **MCP Server:** 5 tools operational (performance, PIX, ML comparison, screenshots, visual quality)
- **F2 Screenshot Capture** ✅ COMPLETE

**Core Technology Stack:**
- DirectX 12 with Agility SDK, DXR 1.1 (RayQuery API), NVIDIA DLSS 3.7, HLSL Shader Model 6.5+, ImGui, PIX for Windows, ONNX Runtime (ML, optional), PyTorch (PINN training)

---

## Build System

### Primary Build Method
```bash
# Open solution
start PlasmaDX-Clean.sln

# Command line build
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
```

**Debug** - Daily development (zero PIX overhead): `build/Debug/PlasmaDX-Clean.exe`
**DebugPIX** - PIX GPU debugging (instrumented): `build/DebugPIX/PlasmaDX-Clean-PIX.exe`

**Shaders:** Compiled automatically via CMake. Manual compilation:
```bash
dxc.exe -T cs_6_5 -E main shaders/particles/particle_physics.hlsl -Fo particle_physics.dxil
dxc.exe -T lib_6_3 shaders/rtxdi/rtxdi_raygen.hlsl -Fo shaders/rtxdi/rtxdi_raygen.dxil
```

**ONNX Runtime (Optional):** CMake auto-detects at `external/onnxruntime/`. If missing, ML features disabled with warning.

---

## Configuration System

Hierarchical JSON config loading:
1. Command-line: `--config=<path>` (highest priority)
2. Environment: `PLASMADX_CONFIG=<path>`
3. Build directory: `./config.json`
4. User default: `configs/user/default.json`
5. Hardcoded defaults (fallback)

**Key directories:**
- `configs/user/` - User/development configs
- `configs/scenarios/` - Test scenarios
- `configs/presets/` - Shadow quality presets
- `ml/models/` - PINN trained models
- `ml/training_data/` - Physics trajectory datasets

---

## Architecture

### Clean Module Design (Single-Responsibility Modules)

**Core Systems** (`src/core/`): Application, Device, SwapChain, FeatureDetector
**Particle Systems** (`src/particles/`): ParticleSystem (physics), ParticleRenderer_Gaussian (3D volumetric rendering), ParticlePhysics
**Lighting Systems** (`src/lighting/`): RTLightingSystem_RayQuery (DXR 1.1 inline ray tracing), RTXDILightingSystem (weighted reservoir sampling)
**ML Systems** (`src/ml/`): AdaptiveQualitySystem (ONNX Runtime, PINN physics inference)
**Utilities** (`src/utils/`): ShaderManager, ResourceManager, Logger

### Key Architecture Principles

1. **Feature Detection First** - Test capabilities before using (RT tier, mesh shaders, ONNX Runtime)
2. **Single Responsibility** - Max ~500 lines per file
3. **Automatic Fallbacks** - Mesh shader → compute shader, ONNX missing → traditional physics
4. **Data-Driven Configuration** - Runtime adjustable via JSON/ImGui
5. **Defensive Programming** - Error handling and PIX event markers everywhere

---

## Shader Architecture

**Key Shaders:**
- `particle_physics.hlsl` - GPU physics compute (black hole gravity, Keplerian dynamics, blackbody emission)
- `particle_gaussian_raytrace.hlsl` - **PRIMARY RENDERER** (RayQuery API, ray-ellipsoid intersection, Beer-Lambert law, Henyey-Greenstein phase)
- `particle_raytraced_lighting_cs.hlsl` - RT lighting compute (particle-to-particle illumination)
- `rtxdi_raygen.hlsl` - DXR raygen for weighted reservoir sampling (RTXDI, not currently active)
- `rtxdi_temporal_accumulate.hlsl` - M5 temporal accumulation (RTXDI, not currently active)
- `generate_particle_aabbs.hlsl` - Procedural primitive AABB generation
- `particle_intersection.hlsl` - Ray-ellipsoid intersection shader

**IMPORTANT:** Root constants limited to 64 DWORDs. Use constant buffers for large structures.

---

## 3D Gaussian Splatting Implementation

**Volumetric 3D Gaussians** (not traditional 2D splatting):
- Full 3D ellipsoid volume with ray marching
- Analytic ray-ellipsoid intersection
- Beer-Lambert law for volumetric absorption
- Temperature-based blackbody emission (not learned RGB)
- Anisotropic elongation along velocity vectors (tidal tearing)

**Core algorithm:** `RayGaussianIntersection()` in `gaussian_common.hlsl`

---

## RTXDI Implementation (Phase 4 - WORK-IN-PROGRESS)

**Status:** M5 Phase 2 In Progress - **NOT CURRENTLY USED AS PRIMARY RENDERER**

**Note:** RTXDI is currently disabled as the primary renderer due to quality issues (patchwork pattern, temporal instability). The Gaussian volumetric renderer (particle_gaussian_raytrace.hlsl) is the active primary renderer. RTXDI will be re-enabled once M5 temporal accumulation problems are resolved.

**Implementation:**
- Light Grid: 30×30×30 spatial acceleration (27,000 cells, 3000-unit world coverage)
- DXR raygen shader performs weighted reservoir sampling
- M5 temporal accumulation with ping-pong buffers (in progress)
- Output: R32G32B32A32_FLOAT texture (selected light index per pixel)

**Light Presets:**
- Fibonacci Sphere (13 lights @ 1200-unit radius)
- Dual-Ring Formation (16 lights @ 600-1000 unit radii)
- Cross Pattern (5 lights for debugging)

**Migration from Custom ReSTIR:** Original 126 MB reservoir buffers deprecated in favor of lightweight RTXDI.

**Current Status:**
- Phase 1 (M4): Weighted sampling ✅ COMPLETE
- Phase 2 (M5): Temporal accumulation 🔄 IN PROGRESS
- Phase 3 (M6): Spatial reuse ⏳ PLANNED

---

## Multi-Light System (Phase 3.5 - COMPLETE ✅)

13 lights distributed across accretion disk with realistic multi-directional shadowing, rim lighting, atmospheric scattering. Performance: 120+ FPS @ 10K particles (RTX 4060 Ti, 1080p).

**ImGui Controls:** Position, Color, Intensity (0.1-20.0), Radius (10.0-200.0), per-light enable/disable, bulk controls

**Presets:** Stellar Ring (13 lights, default), Dual Binary (2 lights), Trinary Dance (3 lights), Single Beacon (1 light)

---

## PCSS Soft Shadows (Phase 3.6 - COMPLETE ✅)

Percentage-Closer Soft Shadows with temporal filtering: 115-120 FPS (Performance preset) @ 1080p, 10K particles.

**Three Presets:**
1. **Performance** (1-ray + temporal, 67ms convergence): 115-120 FPS
2. **Balanced** (4-ray PCSS): 90-100 FPS
3. **Quality** (8-ray PCSS): 60-75 FPS

**Technical:** 2× R16_FLOAT ping-pong buffers (4MB @ 1080p), temporal blend: `lerp(prevShadow, currentShadow, 0.1)`

---

## Adaptive Particle Radius (Phase 1.5 - COMPLETE ✅)

Dynamic particle sizing based on camera distance and local density.

**Zones:**
- **Inner** (close): Particles shrink to reduce overlap
- **Transition** (mid): Linear interpolation
- **Outer** (far): Particles grow to maintain visibility

**Key Lesson (ImGui):** Always check return value to avoid calling setters every frame:
```cpp
// CORRECT
if (ImGui::SliderFloat("Value", &value, 0, 1)) {
    SetValue(value);  // Only on change
}
```

---

## Physics-Informed Neural Networks (Phase 5 - ACTIVE 🔄)

**Status:** Python training complete ✅, C++ integration in progress 🔄

PINN learns accretion disk forces while respecting astrophysics (GR, Keplerian motion, angular momentum conservation, Shakura-Sunyaev viscosity, energy conservation).

**Network:** 7D input (r,θ,φ,v_r,v_θ,v_φ,t) → 5×128 hidden (Tanh) → 3D force output (F_r,F_θ,F_φ), ~50K parameters

**Benefits:** 5-10× faster than GPU physics shader @ 100K particles, scientifically accurate, hybrid mode ready (PINN for far, shader for near ISCO)

**Quick Start:**
```bash
cd ml
pip install -r requirements_pinn.txt
../build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
python collect_physics_data.py --input ../PIX/buffer_dumps
python pinn_accretion_disk.py  # ~20 min training
python test_pinn.py --model models/pinn_accretion_disk.onnx
```

---

## NVIDIA DLSS Integration (Phase 7 - PARTIAL ✅)

**Super Resolution ✅ OPERATIONAL:** AI-powered upscaling (e.g., 720p → 1440p). Performance Mode: +40-60% FPS, Quality Mode: +20-30% FPS.

**Ray Reconstruction ⚠️ SHELVED:** Incompatible with volumetric rendering (requires full G-buffer, particles lack traditional surface properties). Alternative denoising: PCSS temporal accumulation, RTXDI M5 temporal reuse, AMD FidelityFX Denoiser (future).

---

## Dynamic Emission System (Phase 3.8 - COMPLETE ✅)

RT-driven dynamic star radiance:
1. **RT Lighting Suppression** - Emission inversely proportional to RT lighting (70% default)
2. **Selective Emission** - Only particles >22000K emit significantly
3. **Temporal Modulation** - Gentle pulsing/scintillation (0.03 rate)
4. **Distance-Based LOD** - Close: 50% emission, Far: 100% emission
5. **Improved Blackbody Colors** - Wien's law approximation

**RT Lighting Constant Buffer:** Expanded from 4 → 14 DWORDs (56 bytes)

**CRITICAL:** When expanding constant buffers, update: struct definition, root signature, upload code, shader cbuffer, manual shader recompile if needed.

---

## DXR 1.1 Inline Ray Tracing

**Why RayQuery API?** Call from any shader stage, no SBT complexity, perfect for procedural primitives.

**Pipeline:** GPU Physics → Generate AABBs → Build BLAS → Build TLAS → RayQuery (volumetric render) → RayQuery (shadow rays) → TraceRay (RTXDI sampling)

**Acceleration Structure Reuse:** Gaussian renderer reuses TLAS from RTLightingSystem. Do NOT create duplicate BLAS/TLAS.

---

## PIX GPU Debugging Workflow

**DebugPIX build:**
1. Build DebugPIX configuration
2. Run: `./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json`
3. Automatic capture at frame 120
4. Captures saved to `PIX/Captures/`

**Buffer Dumps:** `--dump-buffers <frame>` saves to `PIX/buffer_dumps/` (g_particles.bin for PINN training, g_rtLighting.bin)

---

## MCP Server for RTXDI Quality Analysis

**Status:** Operational with 5 tools (Location: `agents/dxr-image-quality-analyst/`)

**Tools:**
1. `compare_performance` - Compare legacy, RTXDI M4, M5 performance
2. `analyze_pix_capture` - Analyze PIX captures for bottlenecks
3. `compare_screenshots_ml` - ML-powered LPIPS perceptual similarity (~92% human correlation)
4. `list_recent_screenshots` - List recent screenshots (newest first)
5. `assess_visual_quality` - AI vision analysis for volumetric quality (7 dimensions)

**Screenshot Capture:** Press **F2** during rendering (captures GPU framebuffer at native resolution, saves to `screenshots/screenshot_YYYY-MM-DD_HH-MM-SS.bmp`)

**CRITICAL:** Lazy loading for PyTorch/LPIPS (528MB weights) to avoid MCP 30-second timeout.

---

## Critical Implementation Details

### Descriptor Heap Management
**IMPORTANT:** ResourceManager maintains central descriptor heap. Always allocate through ResourceManager, never create ad-hoc heaps.

### Buffer Resource States
**Common transition:** `UNORDERED_ACCESS (compute write) → UAV Barrier → NON_PIXEL_SHADER_RESOURCE (compute read) → UNORDERED_ACCESS (next pass)`

### Root Signature Limitations
- Root constants: 64 DWORD limit (256 bytes)
- Root descriptors: Direct buffer pointers (best performance)
- Descriptor tables: For large descriptor arrays or typed UAVs

### Acceleration Structure Rebuilds
**Current:** Full BLAS/TLAS rebuild every frame (2.1ms @ 100K particles). **Optimization potential:** BLAS update: +25% FPS, Instance culling: +50% FPS. Do NOT attempt without thorough testing.

---

## Known Issues and Workarounds

**Mesh Shader Descriptor Access (NVIDIA Ada Lovelace):** RTX 40-series driver bug. Auto-fallback to compute shader (no performance loss).

**RTXDI Temporal Accumulation (M5):** In active development. Patchwork pattern, temporal instability preventing use as primary renderer.

**God Rays System:** SHELVED ⚠️ (performance/quality issues, conflicts with RTXDI)

**PINN ML Integration:** C++ integration in progress (ONNX Runtime model loading, hybrid mode, ImGui controls pending).

---

## Performance Targets

**Test Configuration:** RTX 4060 Ti, 1920×1080, 100K particles

| Feature Set | Target FPS | Current | Notes |
|-------------|------------|---------|-------|
| Raster Only | 245 | 245 ✅ | Baseline |
| + RT Lighting | 165 | 165 ✅ | TLAS rebuild bottleneck |
| + Shadow Rays | 142 | 142 ✅ | PCSS Performance |
| + DLSS Performance | 190 | 190 ✅ | 720p→1440p |
| + PINN Physics (100K) | 280+ | TBD 🔄 | PINN + DLSS |

**Bottleneck:** RayQuery traversal of 100K procedural primitives (BLAS rebuild: 2.1ms/frame)

**Optimization priorities:** PINN ML physics (+50-100 FPS, ACTIVE), RTXDI M5 optimization (+5-10 FPS, IN PROGRESS), BLAS update (+25% FPS, PLANNED), Particle LOD culling (+50% FPS, PLANNED)

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation.

---

## File Naming Conventions

- Headers: `.h`, Implementation: `.cpp`, Shaders: `.hlsl`, Compiled shaders: `.dxil`, Configs: `.json`, ML models: `.onnx`, Training data: `.npz`
- Classes: PascalCase (`ParticleSystem`), Functions: PascalCase (`Initialize`), Variables: camelCase (`m_particleCount`), Constants: UPPER_SNAKE_CASE (`BLACK_HOLE_MASS`)

---

## Dependencies and External Libraries

**Included:** DirectX 12 Agility SDK, ImGui, d3dx12.h, RTXDI Runtime SDK
**Optional:** ONNX Runtime (PINN ML physics)
**System:** Visual Studio 2022 (C++17), Windows SDK 10.0.26100.0+, DXC shader compiler, PIX for Windows (optional)
**Python:** PyTorch >= 2.0.0, ONNX >= 1.14.0, NumPy, Matplotlib, SciPy
**Drivers:** NVIDIA 531.00+ (DXR 1.1), AMD Adrenalin 23.1.1+ (DXR 1.1)

---

## Reference Documentation

**Critical docs:** `MASTER_ROADMAP_V2.md` (AUTHORITATIVE roadmap), `PARTICLE_FLASHING_ROOT_CAUSE_ANALYSIS.md` (14K-word visual quality investigation), `SHADOW_RTXDI_IMPLEMENTATION_ROADMAP.md` (RTXDI integration guide), `BUILD_GUIDE.md`, `PCSS_IMPLEMENTATION_SUMMARY.md`

**In-repo docs:** `README.md`, `configs/README.md`, `PHYSICS_PORT_ANALYSIS.md`, `PIX/docs/QUICK_REFERENCE.md`, `ml/PINN_README.md`, `PINN_IMPLEMENTATION_SUMMARY.md`, `DYNAMIC_EMISSION_IMPLEMENTATION.md`

**External refs:** DirectX 12 Programming Guide, DXR 1.1 Spec, ReSTIR Paper (Bitterli et al. 2020), RTXDI Documentation, 3D Gaussian Splatting (Kerbl et al. 2023), PINN Paper (Raissi et al. 2019), ONNX Runtime

---

## Immediate Next Steps for Development

**Current sprint:**
1. 🔄 RTXDI M5 temporal accumulation (fix quality issues to enable as primary renderer)
2. 🔄 C++ ONNX Runtime integration for PINN
3. ⏳ RT-based star radiance enhancements (scintillation, coronas, spikes)
4. ⏳ Hybrid physics mode (PINN + traditional)

**Roadmap (see MASTER_ROADMAP_V2.md):**
- **Phase 3.5-3.6:** Multi-light + PCSS ✅ COMPLETE
- **Phase 4 (Current):** RTXDI M5 + Shadow Quality 🔄 IN PROGRESS
- **Phase 5 (Current):** PINN ML Integration (Python ✅, C++ 🔄)
- **Phase 6 (Next):** Custom Temporal Denoising
- **Phase 7 (Future):** Enhanced Star Radiance Effects
- **Phase 8 (Long-term):** Celestial Body System (heterogeneous particles, LOD, material-aware RT)

---

**Last Updated:** 2025-11-12
**Project Version:** 0.14.4
**Documentation maintained by:** Claude Code sessions

**Note:** See `MASTER_ROADMAP_V2.md` for the most up-to-date development status and detailed technical implementation plans.
