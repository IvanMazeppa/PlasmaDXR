# Light Scattering Architecture Consultation for PlasmaDX-Clean

## Context and Goals

I'm developing **PlasmaDX-Clean**, an experimental DirectX 12 volumetric particle renderer simulating a black hole accretion disk. After successfully implementing VolumetricReSTIR (reservoir-based path tracing), I've discovered it doesn't provide the real-time light scattering effect I'm looking for. I need fresh perspectives on architectures and techniques that could achieve effective volumetric light scattering using modern NVIDIA Ada Lovelace / DXR 1.1+ capabilities.

**My Goal:** Create convincing volumetric light scattering where particles scatter and redirect light from multiple light sources in real-time, without the extreme computational cost of full path tracing.

**Current Issue:** VolumetricReSTIR only uses particle self-emission (blackbody radiation), completely ignoring my 13-light multi-light system. It's also slower than my existing Gaussian renderer (56 FPS vs 80 FPS) and requires hundreds of samples per pixel for smooth results.

---

## Current Architecture (What Works Well)

### Hardware Target
- **GPU:** NVIDIA RTX 4060 Ti (Ada Lovelace architecture)
- **Resolution:** 1440p (2560×1440)
- **Target FPS:** 90-120 FPS
- **Particle Count:** 10,000 particles

### Technology Stack
- **DirectX 12** with Agility SDK 1.614.0
- **DXR 1.1** inline ray tracing (RayQuery API)
- **NVIDIA DLSS 3.7** Super Resolution (operational)
- **Shader Model 6.5+** (HLSL)
- **Visual Studio 2022**, Windows 11, WSL2 Ubuntu

### Existing Rendering Systems

#### 1. Gaussian Volumetric Renderer (PRIMARY - WORKS WELL)
**Status:** ✅ Production-ready, 80 FPS @ 10K particles

**Features:**
- 3D Gaussian splatting with anisotropic ellipsoids
- Ray-ellipsoid intersection using RayQuery
- Beer-Lambert law volumetric absorption
- Henyey-Greenstein phase function scattering
- Temperature-based blackbody emission (Wien's law)
- Velocity-aligned tidal stretching

**Shaders:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Primary volumetric renderer
- `shaders/gaussian/gaussian_common.hlsl` - Ray-ellipsoid intersection math

**Architecture:**
```
GPU Physics → Generate AABBs → Build BLAS → Build TLAS →
RayQuery (Gaussian volumetric render) → Output texture
```

#### 2. Multi-Light RT System (WORKS WELL)
**Status:** ✅ Production-ready, 13 lights with RT shadows

**Features:**
- 13 dynamic point lights distributed around accretion disk
- Per-particle RT lighting using RayQuery
- Illuminates particles with configurable intensity/color/radius
- Integrated with Gaussian renderer

**Shaders:**
- `shaders/dxr/particle_raytraced_lighting_cs.hlsl` - Per-particle direct lighting
- `shaders/shadows/pcss_shadows.hlsl` - Soft shadow computation

**Performance:** 115-120 FPS with PCSS shadows (1-ray temporal)

#### 3. PCSS Soft Shadows (WORKS WELL)
**Status:** ✅ Production-ready, temporal accumulation

**Features:**
- Percentage-Closer Soft Shadows
- 2× R16_FLOAT ping-pong buffers for temporal filtering
- 3 quality presets (Performance/Balanced/Quality)
- 67ms convergence time at Performance preset

**Current Limitation:** Shadows only cast from lights to particles, no particle-to-particle shadowing

#### 4. Probe Grid System (PARTIALLY IMPLEMENTED)
**Status:** ⚠️ Exists but not fully utilized

**Current State:**
- Hybrid probe-based volumetric lighting from Phase 0.13.1
- Probes capture irradiance but particles don't sample from them yet

**Potential:** Could provide ambient volumetric scatter, needs particle sampling integration

#### 5. VolumetricReSTIR (WORKS BUT UNSUITABLE)
**Status:** ✅ Functional but not for general use

**What it does:**
- Reservoir-based volumetric path tracing
- Ray-sphere intersection with RayQuery
- Monte Carlo importance sampling
- Only uses particle self-emission (blackbody)

**Why it doesn't work for my goal:**
- ❌ Ignores all 13 external lights
- ❌ No integration with multi-light system
- ❌ No shadow system integration
- ❌ Slower than Gaussian (56 FPS vs 80 FPS)
- ❌ Needs 256+ samples/pixel for smooth results (currently 32)
- ❌ Physics-constrained colors (only blackbody silver/gold)
- ❌ Extreme flickering without M5 temporal accumulation

**Best use:** Specialized effects (explosions, fire) with additive compositing, NOT general light scattering

**Details:** `docs/VOLUMETRIC_RESTIR_IMPLEMENTATION_SUMMARY_20251120.md`

---

## What I'm Missing: Effective Light Scattering

### The Problem

**Current behavior:**
- Gaussian renderer: Particles receive direct lighting from 13 lights
- RT lighting: Shadow rays to lights, simple illumination
- **Missing:** Particles don't scatter/redirect light to other particles
- Result: Looks like "isolated spheres with shadows" not "volumetric medium with light transport"

**What I want:**
- Light from the 13 external lights enters the particle field
- Light scatters from particle to particle
- Creates volumetric glow, halos, caustic-like effects
- Maintains real-time performance (90+ FPS target)

**Why this matters:**
- Accretion disks should have luminous, volumetric appearance
- Need sense of depth and atmospheric scattering
- Current direct-lighting-only approach looks too "hard edged"

### What I've Tried

1. **VolumetricReSTIR (just finished):**
   - Result: Beautiful volumetric scattering BUT only from self-emission
   - Doesn't use external lights at all
   - Too slow for 10K particles (56 FPS)

2. **Multi-light RT with particle-to-particle rays:**
   - Too expensive computationally
   - User quote: "multi-light system is really expensive to run"

3. **Probe grid (partial):**
   - Exists but particles don't sample from it yet
   - Could provide ambient scatter but needs integration

---

## Technical Constraints and Preferences

### Must Have
- ✅ **Real-time performance:** 90+ FPS @ 1440p with 10K particles
- ✅ **DXR 1.1 inline ray tracing:** Use RayQuery API (already implemented)
- ✅ **Integrate with existing multi-light system:** Use the 13 configurable lights
- ✅ **Minimal shader changes:** Prefer enhancements over rewrites
- ✅ **Maintain Gaussian renderer:** Don't replace what works

### Nice to Have
- 🎯 **Leverage Ada Lovelace features:** RT cores, Tensor cores, Shader Execution Reordering
- 🎯 **Runtime tunable parameters:** ImGui controls for artistic control
- 🎯 **Composable architecture:** Layer effects without coupling
- 🎯 **Artistic control:** Physics-informed but tweakable for aesthetics

### Technical Assets Available
- ✅ Acceleration structure (BLAS/TLAS) rebuilt every frame
- ✅ Procedural primitive AABBs for all particles
- ✅ RayQuery inline ray tracing pipeline
- ✅ 13-light multi-light system with configurable properties
- ✅ Probe grid infrastructure (needs particle sampling)
- ✅ PCSS shadow system with temporal accumulation
- ✅ DLSS 3.7 for upscaling (can render at lower res if needed)
- ✅ ImGui runtime controls framework

---

## Suggested Approaches from Previous Analysis

From `docs/VOLUMETRIC_RESTIR_IMPLEMENTATION_SUMMARY_20251120.md`:

### 1. Screen-Space Scattering
- **Technique:** Post-process radial blur from bright particles
- **Cost:** ~2ms
- **Benefit:** Dramatic volumetric glow effect
- **Examples:** Control, Metro Exodus, many AAA games
- **Question:** How to make this convincing for multi-light interactions?

### 2. Probe Grid Enhancement
- **Technique:** Sample irradiance probes for ambient volumetric lighting
- **Cost:** ~1ms (probe sampling already implemented)
- **Benefit:** Ambient volumetric scatter without per-particle rays
- **Question:** Can this provide directional scatter from 13 lights?

### 3. Simplified Volumetric Fog
- **Technique:** Ray march from camera, sample lights at intervals
- **Cost:** ~3-5ms
- **Benefit:** True volumetric scattering from external lights
- **Question:** How to make this efficient for 10K particles + 13 lights?

### 4. Additive Compositing with VolumetricReSTIR
- **Technique:** Gaussian (9500 particles) + VolumetricReSTIR (500 special particles)
- **Cost:** ~13 FPS reduction (~67 FPS total)
- **Benefit:** Best of both worlds
- **Question:** Does this solve the light scattering problem or just add effects?

---

## What I Need From You

### Primary Questions

1. **What modern light scattering techniques exist for real-time volumetric rendering?**
   - Specifically for NVIDIA Ada Lovelace / DXR 1.1+
   - That can handle 10K particles + 13 lights @ 90+ FPS
   - That integrate with existing Gaussian splatting + RT lighting architecture

2. **Should I enhance/modify VolumetricReSTIR to use external lights?**
   - Estimated work: 2-3 weeks (add light sampling, shadows, temporal accumulation)
   - Question: Is this worth it vs alternatives?
   - Could I use RTXDI (weighted reservoir sampling) for light selection?

3. **Are there hybrid approaches I'm not considering?**
   - Combining techniques (screen-space + probe grid + selective path tracing?)
   - Using Ada Lovelace features I haven't explored (SER, Opacity Micromaps, DMM?)
   - Machine learning denoising to reduce sample counts?

4. **What about approximations that "look right" without full physics?**
   - Fake volumetric scatter with cheap tricks
   - Billboard-based light shafts
   - Depth-based fog with light accumulation
   - Artistic license > physical accuracy

### Specific Technical Interests

- **NVIDIA RTX features:** Shader Execution Reordering, RT Motion Blur, Opacity Micromaps
- **RTXDI SDK:** Already have runtime, could I use for light importance sampling?
- **Neural rendering:** DLSS Ray Reconstruction (shelved for volumetrics, but alternatives?)
- **Compute-based approaches:** Can I avoid ray tracing entirely with clever compute shaders?
- **Temporal techniques:** Already have temporal shadow accumulation, extend to scatter?

---

## Project Structure and Resources

### Where to Find Things

**Primary documentation:**
- `CLAUDE.md` - Project overview, architecture, current status
- `MASTER_ROADMAP_V2.md` - Development roadmap and phase status
- `docs/VOLUMETRIC_RESTIR_IMPLEMENTATION_SUMMARY_20251120.md` - VolumetricReSTIR journey

**Core code:**
- `src/core/` - Application, Device, SwapChain, FeatureDetector
- `src/particles/` - ParticleSystem, ParticleRenderer_Gaussian, physics
- `src/lighting/` - RTLightingSystem_RayQuery, VolumetricReSTIRSystem
- `src/utils/` - ShaderManager, ResourceManager, Logger

**Shaders:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Primary volumetric renderer
- `shaders/dxr/particle_raytraced_lighting_cs.hlsl` - Multi-light RT lighting
- `shaders/volumetric_restir/` - Path tracing shaders (path_generation, shading)
- `shaders/shadows/pcss_shadows.hlsl` - Soft shadow system

**Debug and tools:**
- `build/bin/Debug/` - Executable, shaders, screenshots, logs
- `build/bin/Debug/screenshots/` - F2 screenshot captures with metadata JSON
- `build/bin/Debug/PlasmaDX-Clean.log` - Runtime logs (initialization, performance, errors)
- `PIX/` - PIX GPU captures and buffer dumps
- `tools/convert_screenshots.py` - BMP → PNG converter (11MB → 100KB-1.5MB)
- `ml/` - PINN physics training (Python, PyTorch)

**Configuration:**
- `configs/user/default.json` - Default runtime settings
- `configs/scenarios/` - Test scenarios
- `configs/presets/` - Shadow quality presets

### How to Explore the Project

**Build the project:**
```bash
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
```

**Run with specific config:**
```bash
cd /build/bin/Debug/
./PlasmaDX-Clean.exe --config=configs/user/default.json
```

**Take screenshots:** Press F2 during rendering
- Outputs to `build/bin/Debug/screenshots/`
- Creates BMP + JSON metadata (renderer, settings, FPS, frame time)
- Convert with: `python3 tools/convert_screenshots.py`

**Check logs:**
- `build/bin/Debug/PlasmaDX-Clean.log` - Latest run
- Includes initialization, performance metrics, errors, warnings


**MCP Server for analysis:**
- 5 tools available: `dxr-image-quality-analyst`
- `compare_performance` - Performance analysis across renderers
- `analyze_pix_capture` - PIX capture bottleneck analysis
- `compare_screenshots_ml` - LPIPS perceptual similarity (~92% human correlation)
- `assess_visual_quality` - AI vision analysis (7 quality dimensions)

### Recent Screenshots Available

Latest screenshots (converted to PNG):
- `build/bin/Debug/screenshots/screenshot_2025-11-20_19-57-13.png` (1.5 MB)
- `build/bin/Debug/screenshots/screenshot_2025-11-19_21-52-*.png` (144-219 KB)
- `build/bin/Debug/screenshots/screenshot_2025-11-19_19-*.png` (92-475 KB)

Each has corresponding `.bmp.json` metadata file with:
- Active renderer
- Particle count
- Light configuration
- Shadow quality preset
- FPS and frame time
- Resolution and DLSS settings

---

## Development Workflow

### My Process
1. **Research:** Read papers, explore techniques, test with small prototypes
2. **Implement:** Write shaders and C++ integration
3. **Test:** Take F2 screenshots, check logs, use PIX for GPU debugging
4. **Iterate:** Adjust parameters via ImGui, recompile shaders if needed
5. **Document:** Create detailed summaries of learnings

### Collaboration with AI
- I work with Claude Code in WSL2 Ubuntu terminal
- Claude has full read/write access to codebase
- We build incrementally with frequent testing
- I'm comfortable with C++, HLSL, but learning DXR/RT techniques
- High-functioning autism: I prefer direct, honest technical feedback over sugar-coating

### Current Mental Block
After a month getting VolumetricReSTIR working, discovering it doesn't solve my light scattering problem has me questioning the approach. I need fresh architectural ideas that:
- Take advantage of modern RT hardware (Ada Lovelace)
- Don't require full path tracing convergence
- Integrate with what I already have working
- Provide the volumetric scatter "feel" I'm looking for

---

## Specific Questions for You

### Architecture
1. Given my Gaussian renderer + multi-light RT system, what's the **lowest-cost way to add convincing volumetric light scattering**?
2. Should I enhance the probe grid system for ambient scatter, or is that the wrong direction?
3. Are there screen-space techniques that can fake volumetric scatter convincingly?
4. Could I use a coarse 3D texture for light accumulation (voxel grid) and sample from it?

### NVIDIA Ada Lovelace Features
5. What Ada Lovelace / DXR 1.1+ features am I not leveraging that could help?
6. Should I investigate Shader Execution Reordering (SER) for better ray coherence?
7. Can Opacity Micromaps or Displaced Micro-Meshes help with volumetric rendering?
8. Are there Tensor Core operations (beyond DLSS) useful for light scattering?

### VolumetricReSTIR Direction
9. Is it worth 2-3 weeks to add external light support to VolumetricReSTIR?
10. Or should I treat it as a "special effects only" renderer and look elsewhere for general scatter?
11. Could I use RTXDI's light importance sampling without full path tracing?

### Hybrid / Novel Approaches
12. What about temporal upsampling? Render scatter at 1/4 res and temporally accumulate?
13. Could I compute a "light propagation volume" on GPU and sample from it?
14. Machine learning approaches? Train a network to predict scatter from sparse samples?
15. Are there "good enough" approximations that give 80% of the effect for 20% of the cost?

---

## What Success Looks Like

**Visual Goal:**
- Accretion disk with luminous volumetric appearance
- Light from 13 external lights scatters through particle field
- Particles glow with halos, not hard edges
- Sense of atmospheric depth and light transport
- Similar to cinematics in Interstellar, The Expanse, or high-end VFX

**Performance Goal:**
- 90+ FPS @ 1440p native (or 75+ FPS if using DLSS upscaling)
- 10,000 particles
- 13 dynamic lights
- Soft shadows maintained

**Workflow Goal:**
- Runtime tunable parameters via ImGui
- No 30-second shader recompile cycles for tweaks
- Quick iteration on visual appearance

---

## Your Turn

Please provide:
1. **Specific architectural recommendations** for achieving volumetric light scattering
2. **Prioritized list of approaches** (best to worst for my constraints)
3. **Implementation complexity estimates** (hours/days of work)
4. **Performance impact predictions** (FPS cost for each approach)
5. **Modern RT techniques** I should research (papers, GDC talks, NVIDIA blogs)
6. **Ada Lovelace features** I should investigate further
7. **Hybrid solutions** combining multiple techniques

**Be brutally honest** if VolumetricReSTIR is a dead-end for general light scattering. I value direct technical feedback over false encouragement.

**Focus on practical solutions** that work with NVIDIA RTX 4060 Ti today, not research papers requiring RTX 6090 Ti in 2027.

Thank you for taking the time to analyze this complex problem!

---

**Project:** PlasmaDX-Clean v0.18.1
**Date:** 2025-11-20
**Developer:** Ben (novice but enthusiastic)
**Repository Structure:** `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/`
**Build Output:** `build/bin/Debug/`
**Documentation:** `docs/`, `CLAUDE.md`, `MASTER_ROADMAP_V2.md`
