# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

The user is named Ben, a novice programmer with C++, Java, and Python experience. He has high-functioning autism with a strong passion for AI/ML/LLMs, leveraging these tools to create this experimental raytracing engine.

### Collaboration Preferences

- **Be corrective when wrong** - Correct misunderstandings immediately but kindly, explain the "why"
- **Validate effort** - Acknowledge reasonable approaches even if technically incorrect
- **Show what's salvageable** - Emphasize reusable work when fixing issues
- **Break down complex problems** - Manageable steps when Ben is uncertain
- **Test ideas immediately** - Working code examples over descriptions

### Feedback Philosophy: Brutal Honesty

**CRITICAL:** Brutal honesty is strongly preferred over sugar-coating.

✅ **Good:** "ZERO LIGHTS ACTIVE - this is catastrophic, cannot assess visual quality"
❌ **Bad:** "Lighting could use some refinement to improve visual quality"

Direct, specific language accelerates debugging and saves development time.

---

## What is PlasmaDX-Clean?

DirectX 12 volumetric particle renderer featuring:
- **DXR 1.1 inline ray tracing** (RayQuery API)
- **3D Gaussian splatting** (volumetric ellipsoids, not 2D splats)
- **NVIDIA RTXDI** for weighted reservoir sampling
- **ML-accelerated physics** via Physics-Informed Neural Networks (PINNs)
- **Black hole accretion disk simulation** achieving 20 FPS @ 1440p with 10K particles, 16 lights on RTX 4060 Ti

**RT Priority:** RT lighting/shadowing should be first choice for upgrades, but only when it benefits image quality - never force RT just for the sake of using it.

---

## Build Commands

```bash
# One-time setup: Generate Visual Studio solution
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64

# Build (from repo root)
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Release /p:Platform=x64

# Run
./build/bin/Debug/PlasmaDX-Clean.exe
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/user/default.json

# Manual shader recompilation (if .dxil is stale)
dxc.exe -T cs_6_5 -E main shaders/particles/particle_physics.hlsl -Fo build/bin/Debug/shaders/particles/particle_physics.dxil
dxc.exe -T cs_6_5 -E main shaders/particles/particle_gaussian_raytrace.hlsl -Fo build/bin/Debug/shaders/particles/particle_gaussian_raytrace.dxil
```

**Configurations:** Debug (daily dev), Release (performance), DebugPIX (GPU debugging)

**CRITICAL:** Stale .dxil files are the #1 cause of mysterious visual bugs. If you modify .hlsl and get unexpected visuals, rebuild or manually recompile.

---

## Configuration System

Hierarchical JSON loading (highest to lowest priority):
1. `--config=<path>` CLI argument
2. `PLASMADX_CONFIG` environment variable
3. `./config.json` in build directory
4. `configs/user/default.json`
5. Hardcoded defaults

**Key directories:** `configs/user/`, `configs/scenarios/`, `configs/presets/`, `ml/models/`, `ml/training_data/`

---

## Architecture Overview

### Core Systems (`src/`)

| Directory | Purpose |
|-----------|---------|
| `core/` | Application, Device, SwapChain, FeatureDetector |
| `particles/` | ParticleSystem (physics), ParticleRenderer_Gaussian (3D volumetric), ParticleRenderer_Billboard (fallback) |
| `lighting/` | RTLightingSystem_RayQuery (DXR 1.1), RTXDILightingSystem, VolumetricReSTIRSystem, ProbeGridSystem |
| `ml/` | AdaptiveQualitySystem (ONNX), PINNPhysicsSystem |
| `dlss/` | DLSSSystem (Super Resolution) |
| `debug/` | PIXCaptureHelper |
| `utils/` | ResourceManager, Logger |

### Key Shaders (`shaders/`)

| Shader | Purpose |
|--------|---------|
| `particles/particle_gaussian_raytrace.hlsl` | **PRIMARY RENDERER** - RayQuery API, ray-ellipsoid intersection, Beer-Lambert, Henyey-Greenstein |
| `particles/particle_physics.hlsl` | GPU physics - black hole gravity, Keplerian dynamics, blackbody emission |
| `gaussian_common.hlsl` | Core `RayGaussianIntersection()` algorithm |
| `dxr/generate_particle_aabbs.hlsl` | Procedural primitive AABB generation |
| `rtxdi/rtxdi_raygen.hlsl` | DXR raygen for weighted reservoir sampling |

### Architecture Principles

1. **Feature Detection First** - Test capabilities before using (RT tier, mesh shaders, ONNX)
2. **Single Responsibility** - Max ~500 lines per file
3. **Automatic Fallbacks** - Mesh shader → compute shader, ONNX missing → traditional physics
4. **Data-Driven Configuration** - Runtime adjustable via JSON/ImGui

---

## DXR 1.1 Pipeline

**Why RayQuery API?** Call from any shader stage, no SBT complexity, perfect for procedural primitives.

**Pipeline:** GPU Physics → Generate AABBs → Build BLAS → Build TLAS → RayQuery (volumetric render) → RayQuery (shadow rays) → TraceRay (RTXDI sampling)

**IMPORTANT:** Gaussian renderer reuses TLAS from RTLightingSystem. Do NOT create duplicate BLAS/TLAS.

---

## MCP Servers (Autonomous Agents)

PlasmaDX uses Model Context Protocol servers for specialized capabilities. All servers in `agents/`:

### DXR/Rendering Agents
- **dxr-image-quality-analyst** - LPIPS comparison, visual quality assessment, performance metrics
- **log-analysis-rag** - PIX capture analysis, buffer dumps, rendering issue diagnosis
- **path-and-probe** - Probe grid analysis, spherical harmonics debugging

### Blender VFX Generation System
Autonomous VFX asset generation using OpenAI Agents SDK:

- **blender-vfx-orchestrator** - Main orchestrator coordinating multi-agent VFX workflows
- **blender-executor** - Executes Blender Python scripts, parses errors
- **blender-manual** - Searches Blender documentation, tutorials, Python API
- **blender-librarian** - Budget tracking, playbook management, fix outcomes
- **script-generator** - Generates Blender scripts from templates, validates parameters
- **iteration-controller** - Manages iterative improvement loops, session state
- **asset-evaluator** - ML-powered render quality evaluation (LPIPS, CLIP, structural analysis)
- **experiment-tracker** - Records baselines, results, suggests experiments
- **mission-control** - Strategic orchestrator for project-wide decisions

### Using MCP Tools

```python
# Example: Evaluate render quality
mcp__asset-evaluator__evaluate_render(image_path="renders/explosion_v3.png")

# Example: Search Blender docs
mcp__blender-manual__search_nodes(query="volume scatter principled")

# Example: Generate VFX script
mcp__script-generator__generate_script(effect_type="explosion", parameters={...})
```

---

## Critical Implementation Details

### Root Signature Limitations
- Root constants: 64 DWORD limit (256 bytes)
- Use constant buffers for large structures

### Descriptor Heap Management
ResourceManager maintains central descriptor heap. Always allocate through ResourceManager, never create ad-hoc heaps.

### Buffer Resource States
Common transition: `UNORDERED_ACCESS (compute write) → UAV Barrier → NON_PIXEL_SHADER_RESOURCE (compute read) → UNORDERED_ACCESS (next pass)`

### When Expanding Constant Buffers
Update ALL of: struct definition, root signature, upload code, shader cbuffer, manual shader recompile if needed.

---

## Known Issues and Workarounds

| Issue | Workaround |
|-------|------------|
| **Mesh Shader Descriptor Access (RTX 40-series)** | Auto-fallback to compute shader (no performance loss) |
| **RTXDI M5 Temporal Accumulation** | Patchwork pattern, temporal instability - Gaussian renderer is primary |
| **Stale .dxil files** | Rebuild project or manually recompile with dxc |
| **Froxel System** | DEPRECATED - replaced by NanoVDB, code remains but unused |

---

## Debugging Tips

### Shader Issues
1. **Stale .dxil** - Check timestamps match .hlsl source
2. **Debug visualization left enabled** - Look for `DebugVisualize*()` calls in shaders
3. **Root signature mismatch** - Verify cbuffer layouts match C++ and HLSL exactly (causes device removal)
4. **PIX GPU captures** - Essential for DXR issues, use DebugPIX configuration

### Performance
1. Check frame timings in ImGui - individual pass times reveal bottlenecks
2. BLAS/TLAS rebuilds are expensive (2.1ms @ 100K particles)
3. Ray budget per pixel is critical - even 1 extra ray can cost 20% performance

### PIX Debugging Workflow
```bash
# Run with PIX config
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json

# Buffer dumps for ML training
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
# Saves to PIX/buffer_dumps/
```

**F2** captures screenshots to `screenshots/screenshot_YYYY-MM-DD_HH-MM-SS.bmp`

---

## PINN ML Physics

**Status:** Python training complete, C++ ONNX integration in progress

```bash
cd ml
pip install -r requirements_pinn.txt
../build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
python collect_physics_data.py --input ../PIX/buffer_dumps
python pinn_accretion_disk.py  # ~20 min training
python test_pinn.py --model models/pinn_accretion_disk.onnx
```

**Network:** 7D input (r,θ,φ,v_r,v_θ,v_φ,t) → 5×128 hidden (Tanh) → 3D force output

---

## Code Style

- Headers: `.h`, Implementation: `.cpp`, Shaders: `.hlsl`, Compiled: `.dxil`
- Classes/Functions: PascalCase, Variables: camelCase (`m_particleCount`), Constants: UPPER_SNAKE_CASE
- Max ~500 lines per file

---

## Reference Documentation

**Critical:** `MASTER_ROADMAP_V2.md` (authoritative roadmap), `PARTICLE_FLASHING_ROOT_CAUSE_ANALYSIS.md` (visual quality investigation), `BUILD_GUIDE.md`

**In-repo:** `README.md`, `configs/README.md`, `ml/PINN_README.md`, `PIX/docs/QUICK_REFERENCE.md`

**External:** DirectX 12 Programming Guide, DXR 1.1 Spec, ReSTIR Paper (Bitterli et al. 2020), RTXDI Documentation, 3D Gaussian Splatting (Kerbl et al. 2023)

**Always use context7** when you need code generation, setup steps, or library/API documentation.

---

## Dependencies

**Required:** DirectX 12 Agility SDK, RTXDI Runtime SDK, Visual Studio 2022 (C++17), Windows SDK 10.0.26100.0+, DXC shader compiler

**Optional:** ONNX Runtime (ML physics), DLSS SDK, PIX for Windows

**Python (ML):** PyTorch >= 2.0.0, ONNX >= 1.14.0, NumPy, Matplotlib, SciPy

**Drivers:** NVIDIA 531.00+ or AMD Adrenalin 23.1.1+ (DXR 1.1 required)
