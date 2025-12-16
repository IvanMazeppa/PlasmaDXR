# NanoVDB Unified Roadmap V1

**Version:** 1.0
**Status:** Active Development
**Last Updated:** 2025-12-12
**Purpose:** Single authoritative document for NanoVDB + Blender pipeline development

---

## Executive Summary

This is a **Blender 5.0 → NanoVDB → PlasmaDX-Clean** volumetric rendering pipeline for stunning celestial bodies (nebulae, gas clouds, supernovae, accretion disks). The goal is creating production-quality volumetric assets that go straight from Blender into the real-time DX12 renderer.

### Current State: 70% Complete

| Subsystem | Status | Completion |
|-----------|--------|------------|
| **NanoVDB C++ Runtime** | Production | 100% |
| **Ray Marching Shader** | Production | 100% |
| **Animation System** | Production | 100% |
| **Procedural Fog** | Production | 100% |
| **VDB Conversion Scripts** | Working | 90% |
| **Blender Integration** | In Progress | 60% |
| **Recipe Library** | Started | 10% (1/14) |
| **Agent Ecosystem** | Production | 80% |

### Critical Known Issue

**File-loaded .nvdb grids may not render correctly.** Procedural fog works perfectly. Investigation needed:
- Grid bounds vs world transform mismatch?
- PNanoVDB tree traversal issue?
- Blender Z-up vs PlasmaDX Y-up coordinate system?
- Grid type validation failing?

**Debug approach:** Set `debugMode=1` in shader to see:
- 🟢 Green = grid density found (good)
- 🟣 Magenta = procedural density found
- 🔵 Cyan = inside AABB but no density (problem area)

---

## Part 1: Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRODUCTION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BLENDER 5.0 (Mantaflow Fluid Simulation)                                   │
│      │                                                                       │
│      ├── Domain Setup (Quick Smoke, adjust resolution)                       │
│      ├── Bake Simulation (frames 1-250)                                      │
│      └── Export OpenVDB (.vdb files)                                         │
│            │                                                                 │
│            ▼                                                                 │
│  CONVERSION (Python)                                                         │
│      │                                                                       │
│      ├── scripts/convert_vdb_to_nvdb.py                                     │
│      └── Output: .nvdb files (GPU-optimized)                                 │
│            │                                                                 │
│            ▼                                                                 │
│  PLASMADX-CLEAN (DX12 Real-Time Renderer)                                   │
│      │                                                                       │
│      ├── NanoVDBSystem::LoadFromFile()                                      │
│      ├── GPU Buffer Upload                                                   │
│      ├── nanovdb_raymarch.hlsl (PNanoVDB tree traversal)                    │
│      └── Additive blend with particle renderer                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Two-Worktree Development Model

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        GIT WORKTREE ARCHITECTURE                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PlasmaDX-Blender/                    PlasmaDX-NanoVDB/                      │
│  ├── Branch: feature/blender-         ├── Branch: feature/nanovdb-           │
│  │           integration              │           animated-assets            │
│  │                                    │                                      │
│  ├── Focus:                           ├── Focus:                             │
│  │   • Blender recipes                │   • NanoVDBSystem.cpp               │
│  │   • Python automation              │   • Ray marching shader              │
│  │   • Asset creation docs            │   • Animation system                 │
│  │   • VDB export settings            │   • Performance optimization         │
│  │                                    │                                      │
│  └── Outputs:                         └── Outputs:                           │
│      • .blend project files               • C++ implementation               │
│      • .vdb exported grids                • HLSL shaders                     │
│      • Recipe documentation               • ImGui controls                   │
│              │                                    │                          │
│              └────────────────┬───────────────────┘                          │
│                               │                                              │
│                               ▼                                              │
│                    VDBs/ (Shared Directory)                                  │
│                    ├── NanoVDB/ (.nvdb production files)                     │
│                    └── Blender_projects/ (.blend, .vdb source)               │
│                               │                                              │
│                               ▼                                              │
│                    PlasmaDX-Clean/ (main/0.23.x)                             │
│                    └── Integration + Production Builds                       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Technical Implementation

### NanoVDBSystem C++ API

```cpp
// Core API (src/rendering/NanoVDBSystem.h)

// File Loading
bool LoadFromFile(const std::string& nvdbPath);
bool LoadAnimationSequence(const std::string& directory, const std::string& prefix);

// Procedural Generation (no GPU memory, shader-based)
void CreateFogSphere(float3 center, float radius);

// Animation Control
void SetAnimationFrame(int frame);
void SetAnimationSpeed(float speed);
void SetAnimationPlaying(bool playing);

// Visual Parameters
void SetDensityScale(float scale);      // 0.1 - 10.0
void SetEmissionStrength(float str);    // 0.0 - 5.0
void SetAbsorptionCoeff(float coeff);   // 0.1 - 5.0
void SetScatteringCoeff(float coeff);   // 0.0 - 1.0

// Rendering
void Render(ID3D12GraphicsCommandList* cmdList, const RenderContext& ctx);
void SetEnabled(bool enabled);
```

### Shader Architecture (nanovdb_raymarch.hlsl)

**586 lines implementing:**

1. **PNanoVDB Integration** - HLSL-compatible NanoVDB tree traversal
2. **Trilinear Interpolation** - 8-corner sampling for smooth density
3. **Procedural FBM Noise** - Gradient noise with curl advection
4. **Beer-Lambert Absorption** - Physically-based light extinction
5. **Henyey-Greenstein Scattering** - Anisotropic phase function
6. **Multi-Light Support** - Up to 16 lights with attenuation
7. **Depth Occlusion** - Respects existing particle geometry
8. **Debug Mode** - Color-coded diagnostics

**Key Constant Buffer:**
```hlsl
cbuffer NanoVDBConstants : register(b0) {
    float4x4 invViewProj;      // Ray generation
    float3 cameraPos;
    float densityScale;
    float3 gridWorldMin;       // AABB bounds
    float emissionStrength;
    float3 gridWorldMax;
    float absorptionCoeff;
    float3 sphereCenter;       // Procedural fog center
    float scatteringCoeff;
    float sphereRadius;
    float maxRayDistance;
    float stepSize;
    uint lightCount;
    uint screenWidth;
    uint screenHeight;
    float time;                // Animation
    uint debugMode;            // 0=normal, 1=debug colors
    uint useGridBuffer;        // 0=procedural, 1=file-loaded
};
```

### VDB Conversion Scripts

**convert_vdb_to_nvdb.py:**
```bash
python scripts/convert_vdb_to_nvdb.py input.vdb [-o output.nvdb]
```

**inspect_vdb.py:**
```bash
python scripts/inspect_vdb.py file.vdb  # Shows: grids, bounds, voxel count, type
```

---

## Part 3: Blender 5.0 Integration

### ⚠️ CRITICAL: Claude Training Data Warning

**Claude's knowledge is 10+ months behind Blender 5.0!**

Before writing ANY Blender Python code:
1. **Use `blender-manual` MCP** to verify current API
2. **Known breaking changes:**
   - No BLOSC compression (use ZIP or NONE)
   - bpy.ops.geometry changes
   - New modifier stack behavior
3. See `docs/BLENDER_5_GUARDRAILS.md` for full policy

### Recipe Library

| Recipe | Category | Status | Complexity |
|--------|----------|--------|------------|
| hydrogen_cloud.md | Emission Nebulae | ✅ COMPLETE | Medium |
| emission_pillar.md | Emission Nebulae | 🔲 Planned | High |
| orion_style.md | Emission Nebulae | 🔲 Planned | High |
| dark_nebula.md | Dark Structures | 🔲 Planned | Medium |
| dust_lane.md | Dark Structures | 🔲 Planned | Low |
| supernova_remnant.md | Explosions | 🔲 Planned | High |
| stellar_flare.md | Explosions | 🔲 Planned | Medium |
| coronal_ejection.md | Explosions | 🔲 Planned | Medium |
| protoplanetary_disk.md | Stellar Phenomena | 🔲 Planned | High |
| accretion_corona.md | Stellar Phenomena | 🔲 Planned | High |
| planetary_nebula.md | Stellar Phenomena | 🔲 Planned | Medium |

### Property Mapping: Blender → PlasmaDX

| Blender Property | PlasmaDX Property | Conversion |
|------------------|-------------------|------------|
| Volume Density | `densityScale` | `densityScale = density * 0.4` |
| Anisotropy | `scatteringCoeff` | Direct map (-1 to +1) |
| Emission Strength | `emissionStrength` | `emission = strength * 0.25` |
| Volume Color | Procedural noise | Tone-mapped |
| Absorption | Inverted | `1 - color` |
| Temperature | `TemperatureToColor()` | Blackbody curve |

---

## Part 4: Agent Ecosystem

### Production Agents (No API Key Required)

| Agent | Type | Location | Purpose |
|-------|------|----------|---------|
| **blender-manual** | MCP Server | agents/blender-manual/ | 12 tools for Blender 5.0 API lookup |
| **blender-scripting** | Legacy Prompt | agents/blender-scripting/ | Generate bpy Python scripts |
| **celestial-body-curator** | Legacy Prompt | agents/celestial-body-curator/ | Author/maintain recipe library |
| **blender-diagnostics** | Legacy Prompt | agents/blender-diagnostics/ | Troubleshoot simulation/export |
| **gaussian-analyzer** | MCP Server | agents/gaussian-analyzer/ | Validate material properties |
| **materials-council** | MCP Server | agents/materials-council/ | Map Blender → PlasmaDX materials |
| **dxr-volumetric-pyro** | MCP Server | agents/dxr-volumetric-pyro/ | Design explosion effects |

### blender-manual MCP Tools

```python
# API Documentation
search_python_api("bpy.ops.fluid")
search_bpy_types("FluidModifier")
search_bpy_operators("fluid", "bake")

# VDB-Specific
search_vdb_workflow("export openvdb")
search_vdb_workflow("cache smoke")

# General
search_manual("volume rendering")
search_tutorials("volumetrics", "smoke")
read_page("physics/fluid/type/domain/cache.html")
```

### Agent Interaction Patterns

**Pattern 1: Recipe-Driven Asset Creation**
```
User: "Create a dark nebula"
       │
       ▼
celestial-body-curator
       │
       ├──► Provides dark_nebula recipe
       │
       └──► blender-scripting
                 │
                 ├──► Generates Python script
                 │
                 └──► blender-manual MCP
                           │
                           └──► API verification
```

**Pattern 2: Troubleshooting**
```
User: "VDB won't load in PlasmaDX"
       │
       ▼
blender-diagnostics
       │
       ├──► Check export settings
       │
       └──► vdb-converter (scripts/inspect_vdb.py)
                 │
                 └──► Validate grid contents
```

---

## Part 5: Development Phases

### Phase 1: Foundation ✅ COMPLETE

- [x] NanoVDBSystem C++ implementation
- [x] Ray marching shader with PNanoVDB
- [x] Procedural fog sphere (shader-based)
- [x] File loading (.nvdb)
- [x] Animation sequence loading
- [x] Runtime ImGui controls
- [x] Depth occlusion
- [x] Multi-light scattering

### Phase 2: Asset Pipeline 🔄 IN PROGRESS (60%)

- [x] VDB→NanoVDB conversion scripts
- [x] VDB inspection tool
- [x] Blender workflow documentation
- [x] blender-manual MCP (12 tools)
- [x] Agent prompts (scripting, curator, diagnostics)
- [x] Hydrogen cloud recipe
- [ ] **Debug file-loaded grid rendering issue**
- [ ] Complete recipe library (1/14 → 14/14)
- [ ] End-to-end pipeline test

### Phase 3: Performance Optimization ⏳ PLANNED

| Task | Priority | Notes |
|------|----------|-------|
| Step size auto-tuning | High | Adaptive based on density |
| LOD for large grids | High | Distance-based resolution |
| Brick caching for animation | Medium | Reduce frame-switch cost |
| DLSS integration | Medium | Already works, verify quality |
| Temporal reprojection | Low | Reduce noise |

**Performance Targets:**
- 10ms max for 256³ grid @ 1080p
- 30 FPS minimum for animated sequences
- <100MB GPU memory for typical assets

### Phase 4: Advanced Features ⏳ FUTURE

- Multi-volume compositing
- Hot-reload (watch directory)
- Geometry Nodes support
- Temperature-based emission (blackbody from temp grid)
- Velocity-based motion blur

---

## Part 6: Quick Reference

### Build Commands

```bash
# Build PlasmaDX-Clean
cd PlasmaDX-Clean
cmake --build build --target PlasmaDX-Clean

# Convert VDB
python scripts/convert_vdb_to_nvdb.py VDBs/Blender_projects/smoke.vdb

# Inspect VDB
python scripts/inspect_vdb.py VDBs/Blender_projects/smoke.vdb
```

### Runtime Controls (ImGui)

| Control | Range | Default |
|---------|-------|---------|
| Density Scale | 0.1 - 10.0 | 1.0 |
| Emission Strength | 0.0 - 5.0 | 0.5 |
| Absorption Coeff | 0.1 - 5.0 | 1.0 |
| Scattering Coeff | 0.0 - 1.0 | 0.3 |
| Step Size | 1.0 - 50.0 | 10.0 |
| Debug Mode | 0/1 | 0 |

### Key Files

| Category | File |
|----------|------|
| C++ System | `src/rendering/NanoVDBSystem.h/cpp` |
| Shader | `shaders/volumetric/nanovdb_raymarch.hlsl` |
| PNanoVDB Header | `shaders/volumetric/PNanoVDB.h` |
| Conversion | `scripts/convert_vdb_to_nvdb.py` |
| Inspection | `scripts/inspect_vdb.py` |
| Session Context | `docs/NanoVDB/NANOVDB_SESSION_CONTEXT.md` |

### Git Worktree Commands

```bash
# List worktrees
git worktree list

# Switch to Blender worktree
cd ../PlasmaDX-Blender

# Switch to NanoVDB worktree
cd ../PlasmaDX-NanoVDB

# Merge feature to main
cd ../PlasmaDX-Clean
git merge feature/nanovdb-animated-assets
```

---

## Part 7: Immediate Action Items

### Priority 1: Debug File-Loaded Grid Rendering

1. Build PlasmaDX-Clean in Debug
2. Load a known-good .nvdb file
3. Enable `debugMode=1` in ImGui
4. Check shader output colors:
   - Green = success (grid density found)
   - Cyan = problem (AABB hit but no density)
5. Compare `gridWorldMin/Max` with actual grid bounds
6. Verify coordinate system (Blender Z-up → PlasmaDX)

### Priority 2: End-to-End Pipeline Test

1. Open `VDBs/Blender_projects/hydrogen_cloud.blend`
2. Bake simulation (Mantaflow)
3. Export to VDB (Domain > Cache)
4. Convert: `python scripts/convert_vdb_to_nvdb.py ...`
5. Load in PlasmaDX: `nanoVDB->LoadFromFile(...)`
6. Verify visual output matches Blender preview

### Priority 3: Add More Recipes

Start with simpler recipes first:
1. **dark_nebula.md** - Low emission, high absorption
2. **stellar_flare.md** - Animated expansion

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [NANOVDB_SESSION_CONTEXT.md](./NANOVDB_SESSION_CONTEXT.md) | Session restore prompt |
| [NANOVDB_SYSTEM_OVERVIEW.md](./NANOVDB_SYSTEM_OVERVIEW.md) | Full C++ API details |
| [TWO_WORKTREE_WORKFLOW.md](./TWO_WORKTREE_WORKFLOW.md) | Git workflow |
| [VDB_PIPELINE_AGENTS.md](./VDB_PIPELINE_AGENTS.md) | Agent details |
| [BLENDER_5_GUARDRAILS.md](../BLENDER_5_GUARDRAILS.md) | MCP-first policy |
| [hydrogen_cloud.md](../blender_recipes/emission_nebulae/hydrogen_cloud.md) | First complete recipe |

---

*This document consolidates all NanoVDB pipeline documentation into a single source of truth.*
*Last generated: 2025-12-12*
