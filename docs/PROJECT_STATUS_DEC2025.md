# PlasmaDX-Clean Project Status (December 2025)

## Executive Summary

PlasmaDX-Clean is transitioning from a **procedural particle-based accretion disk simulation** to a **volumetric rendering engine** capable of loading pre-simulated volumetric data (NanoVDB). The goal is to achieve cinematic-quality volumetric effects (explosions, nebulae, pyro) with real-time RT lighting.

**Current State:** Hybrid system where:
- ✅ Procedural 3D Gaussian particles work with full RT lighting
- 🔄 NanoVDB volumetric loading works, but rendering integration incomplete
- ⚠️ Animation system loads but doesn't display (bounds/positioning issue)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PlasmaDX-Clean                                │
├─────────────────────────────────────────────────────────────────────┤
│  RENDERING SYSTEMS                                                   │
│  ├── ParticleRenderer_Gaussian  ✅ PRIMARY (3D volumetric Gaussians)│
│  ├── NanoVDBSystem              🔄 WIP (sparse volumetric grids)    │
│  ├── ParticleRenderer_Billboard ⏸️ FALLBACK (2D sprites)            │
│  └── FroxelSystem              ❌ DEPRECATED (replaced by NanoVDB)  │
├─────────────────────────────────────────────────────────────────────┤
│  LIGHTING SYSTEMS                                                    │
│  ├── RTLightingSystem_RayQuery  ✅ ACTIVE (DXR 1.1 inline RT)       │
│  ├── Multi-Light System         ✅ ACTIVE (29 lights)               │
│  ├── LuminousParticleSystem     ✅ ACTIVE (star particles = lights) │
│  ├── ProbeGridSystem            ✅ COMPLETE (SH indirect lighting)  │
│  └── RTXDILightingSystem        🔄 WIP (M5 temporal accumulation)   │
├─────────────────────────────────────────────────────────────────────┤
│  PHYSICS SYSTEMS                                                     │
│  ├── GPU Particle Physics       ✅ ACTIVE (compute shader)          │
│  └── PINN ML Physics            🔄 WIP (Python ✅, C++ integration) │
├─────────────────────────────────────────────────────────────────────┤
│  ML/QUALITY SYSTEMS                                                  │
│  ├── DLSS Super Resolution      ✅ COMPLETE (SDK not in WSL build)  │
│  └── Adaptive Quality System    ⏸️ PAUSED                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Two Rendering Paths

### Path A: Procedural 3D Gaussian Particles (WORKING ✅)

**How it works:**
1. GPU compute shader generates N particles with position, velocity, temperature
2. Particles are 3D ellipsoids (Gaussians) elongated along velocity vectors
3. DXR 1.1 builds acceleration structure over particle AABBs
4. Ray marching through each particle: Beer-Lambert absorption, blackbody emission
5. Multi-light system provides illumination with PCSS soft shadows

**Status:** Fully operational. 100-120 FPS @ 10K particles, 29 lights, RTX 4060 Ti

**Limitations:**
- Particles are procedurally generated, not artist-controlled
- Limited to accretion disk physics model
- Can't load external volumetric data (pyro sims, VDB files)

### Path B: NanoVDB Volumetric Rendering (IN PROGRESS 🔄)

**How it works:**
1. Load .nvdb file (converted from OpenVDB/Blender/EmberGen)
2. Upload sparse grid data to GPU as ByteAddressBuffer
3. Compute shader ray marches through grid bounds
4. Sample density at each step using NanoVDB accessor functions
5. Apply volumetric lighting, emission, scattering

**Current Status:**
- ✅ File loading works (single files and animation sequences)
- ✅ Bounds extraction from non-empty frames works
- ✅ GPU buffer upload works
- ⚠️ Rendering produces black screen (shader sampling issue)
- ⚠️ Grid positioning doesn't account for origin offset

**The Gap:** The raymarching shader (`nanovdb_raymarch.hlsl`) successfully dispatches but doesn't produce visible output. Likely causes:
1. Grid not centered at camera look-at point
2. Ray-grid intersection test failing
3. Density sampling returning 0
4. Step size too large for grid resolution

---

## Current Blockers

### 1. NanoVDB Animation Not Rendering

**Symptom:** Animation loads (131 frames, bounds extracted, frames cycle) but screen is black.

**Diagnosis:**
- Bounds: (0, 11, 0) to (256, 252, 328) ← Grid is in positive octant
- Camera: (800, 1200, 0) looking at (0, 0, 0) ← Looking away from grid
- The "Center at Origin" button should fix this but may not be working correctly

**Next Steps:**
1. Verify `CenterGridAtOrigin()` actually moves bounds to straddle origin
2. Add debug visualization (bounding box wireframe)
3. Add debug output to `nanovdb_raymarch.hlsl` to verify ray-grid intersection

### 2. 4-Minute Animation Load Time

**Symptom:** Loading 131 frames takes ~4 minutes, freezes UI

**Cause:** Synchronous loading on main thread, each frame requires:
- File read from disk
- GPU buffer creation
- SRV creation
- Device sync

**Solution (Future):** Async loading with progress callback, or pre-cached animation buffer

### 3. RTXDI M5 Quality Issues (Separate from NanoVDB)

**Symptom:** Patchwork lighting pattern, temporal instability

**Status:** Shelved in favor of getting NanoVDB working first

---

## File/Shader Inventory

### Core Volumetric Rendering Files

| File | Purpose | Status |
|------|---------|--------|
| `src/rendering/NanoVDBSystem.cpp` | Grid loading, GPU upload, dispatch | ✅ Works |
| `src/rendering/NanoVDBSystem.h` | Class definition, animation state | ✅ Works |
| `shaders/nanovdb/nanovdb_raymarch.hlsl` | GPU raymarching shader | ⚠️ Dispatches but no output |
| `external/nanovdb/nanovdb/NanoVDB.h` | NanoVDB header-only library | ✅ Integrated |

### Related Systems

| File | Purpose | Status |
|------|---------|--------|
| `shaders/particles/particle_gaussian_raytrace.hlsl` | 3D Gaussian particle rendering | ✅ Primary renderer |
| `src/particles/ParticleRenderer_Gaussian.cpp` | Gaussian dispatch, BLAS/TLAS build | ✅ Works |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | DXR 1.1 inline shadow rays | ✅ Works |

### Conversion Pipeline

| File | Purpose | Status |
|------|---------|--------|
| `scripts/convert_vdb_to_nvdb.py` | OpenVDB → NanoVDB conversion | ✅ Uses nanovdb_convert CLI |
| `scripts/blender_vdb_to_nvdb.py` | Blender batch export script | ✅ Updated for Linux |
| `assets/volumes/explosion/` | 131-frame gasoline explosion | ✅ Converted (3.1 GB) |
| `assets/volumes/clouds/` | 10 CloudPack volumes | ✅ Converted (331 MB) |

---

## What We've Learned

### NanoVDB is the Right Choice (vs OpenVDB)
- OpenVDB = CPU simulation library (no GPU support)
- NanoVDB = GPU-native format designed for raymarching
- Sparse data structure is critical for volumetrics (dense textures = massive memory)

### EmberGen VDB Quirks
- Frame 0 is often empty (0 voxels)
- worldBBox metadata is often invalid (inf/NaN)
- Must use indexBBox × voxelSize for bounds
- Grid is positioned in world space, not centered at origin

### The Froxel Experiment Failed
- Froxel (frustum-aligned voxels) was our first volumetric attempt
- Race conditions in density injection (non-atomic +=)
- Debug visualization accidentally left in shipped shaders
- **Deprecated** in favor of NanoVDB sparse grids

---

## Recommended Next Steps

### Immediate (Get NanoVDB Rendering)
1. **Debug the raymarcher** - Add `if (density > 0) output = float4(1,0,0,1)` to verify sampling
2. **Fix grid centering** - Ensure grid straddles camera look-at point
3. **Add bounding box visualization** - Draw wireframe box at grid bounds
4. **Test with static cloud first** - CloudPack volumes are simpler than animations

### Short-Term (Polish NanoVDB)
1. Async loading with progress bar
2. Material presets (fire, smoke, nebula)
3. RT lighting integration with NanoVDB (currently only Gaussians are lit)

### Medium-Term (Unify Systems)
1. Hybrid rendering: Gaussians + NanoVDB in same scene
2. NanoVDB volumes casting shadows on particles
3. Particle emission into NanoVDB grid (simulation feedback)

---

## Performance Reference

| Configuration | FPS | Notes |
|---------------|-----|-------|
| Gaussian particles only (10K) | 120+ | Primary renderer |
| + RT shadows (29 lights) | 100-120 | PCSS Performance preset |
| + DLSS Performance | 150+ | 720p → 1440p upscale |
| NanoVDB single volume | 60-90 | Depends on grid density |
| NanoVDB animation (131 frames) | 11 | Loading overhead, needs optimization |

Hardware: RTX 4060 Ti, 1920×1080 primary test resolution

---

## Glossary

- **3D Gaussian Splatting** - Representing volumes as 3D ellipsoids with density falloff
- **NanoVDB** - NVIDIA's GPU-native sparse volumetric data structure
- **DXR 1.1 RayQuery** - Inline ray tracing API (no separate raygen shader needed)
- **RTXDI** - ReSTIR Direct Illumination (weighted reservoir sampling for many lights)
- **PCSS** - Percentage-Closer Soft Shadows
- **Froxel** - Frustum + Voxel (deprecated volumetric fog approach)
- **PINN** - Physics-Informed Neural Network (ML-accelerated physics)

---

*Last Updated: 2025-12-22*
*Author: Claude Code Session*
