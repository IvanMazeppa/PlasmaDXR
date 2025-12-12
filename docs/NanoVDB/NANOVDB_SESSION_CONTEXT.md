# NanoVDB Session Context - Start Here

**Purpose:** Load this document at the start of any NanoVDB development session to restore full context.

**Last Updated:** 2025-12-12

---

## Quick Summary

You're working on a **Blender → NanoVDB → PlasmaDX** pipeline for volumetric celestial bodies. The system is 70% complete with known rendering issues in file-loaded grids.

---

## Project Goal

Create stunning volumetric celestial bodies (nebulae, gas clouds, explosions) using:
1. **Blender 5.0** - Create and bake volumetric simulations
2. **OpenVDB → NanoVDB** - Convert to GPU-friendly format
3. **PlasmaDX-Clean** - Real-time ray marching renderer (DX12)

---

## Current System Status

| Component | Status | Notes |
|-----------|--------|-------|
| NanoVDBSystem.cpp | Working | File loading, animation, ImGui controls |
| nanovdb_raymarch.hlsl | **ISSUE** | Procedural fog works, file-loaded grids may have problems |
| Blender recipes | 1/14 complete | hydrogen_cloud.md done |
| VDB conversion scripts | Working | convert_vdb_to_nvdb.py |
| Agent ecosystem | Production | blender-manual MCP, 4 agent prompts |

---

## Known Issue: VDB Rendering

**Symptom:** File-loaded .nvdb grids may not render correctly (need to verify exact symptoms from previous session)

**Possible causes (investigate):**
1. Grid bounds not matching world transform
2. PNanoVDB tree traversal issue
3. Coordinate system mismatch (Blender Z-up vs PlasmaDX)
4. Grid type validation failing

**Debug approach:**
```cpp
// In shader, debugMode=1 shows:
// Green = grid density found
// Magenta = procedural density found
// Cyan = in AABB but no density
```

---

## Key Files

### C++ Implementation
- `src/rendering/NanoVDBSystem.h` - API declaration
- `src/rendering/NanoVDBSystem.cpp` - GPU upload, animation, controls

### Shader
- `shaders/volumetric/nanovdb_raymarch.hlsl` - Ray marching with PNanoVDB

### Conversion
- `scripts/convert_vdb_to_nvdb.py` - OpenVDB → NanoVDB
- `scripts/inspect_vdb.py` - Debug VDB contents

### Assets
- `VDBs/NanoVDB/` - Production .nvdb files
- `VDBs/Blender_projects/` - Source .blend files

---

## Architecture

```
Blender 5.0 (Mantaflow)
    ↓ Bake to OpenVDB (.vdb)
convert_vdb_to_nvdb.py
    ↓ Convert to NanoVDB (.nvdb)
NanoVDBSystem::LoadFromFile()
    ↓ GPU buffer upload
nanovdb_raymarch.hlsl
    ↓ PNanoVDB tree traversal + ray marching
Screen output (additive blend with particles)
```

---

## Blender 5.0 Warning

**Claude's training data is 10+ months behind Blender 5.0!**

Before writing ANY Blender code:
1. Use `blender-manual` MCP to verify API
2. Known changes: No BLOSC compression, use ZIP
3. See `docs/BLENDER_5_GUARDRAILS.md`

---

## Two-Worktree Architecture

| Worktree | Branch | Focus |
|----------|--------|-------|
| PlasmaDX-Blender | feature/blender-integration | Recipes, scripts, docs |
| PlasmaDX-NanoVDB | feature/nanovdb-animated-assets | Shader, C++, performance |
| PlasmaDX-Clean | main/0.23.x | Integration, production |

---

## Immediate Priorities

1. **Diagnose VDB rendering issue** - Why do file-loaded grids not render correctly?
2. **Complete pipeline test** - Blender → VDB → NVDB → PlasmaDX working end-to-end
3. **Add more recipes** - dark_nebula, supernova_remnant next

---

## Available MCP Tools

### blender-manual (12 tools)
- `search_manual`, `read_page` - General docs
- `search_python_api`, `search_bpy_types` - Python API
- `search_vdb_workflow` - VDB export specifics

### gaussian-analyzer
- `simulate_material_properties` - Test material rendering
- `analyze_gaussian_parameters` - Particle structure

### materials-council
- Material property mapping between Blender and PlasmaDX

---

## Session Workflow

1. **Read this document** to restore context
2. **Check git status** in your worktree
3. **Build and run** to see current state
4. **Use debug mode** (debugMode=1) to diagnose issues
5. **Commit frequently** with descriptive messages
6. **Update this doc** with discoveries

---

## Related Documentation

- [NANOVDB_SYSTEM_OVERVIEW.md](./NANOVDB_SYSTEM_OVERVIEW.md) - Full technical details
- [TWO_WORKTREE_WORKFLOW.md](./TWO_WORKTREE_WORKFLOW.md) - Git workflow
- [VDB_PIPELINE_AGENTS.md](./VDB_PIPELINE_AGENTS.md) - Agent ecosystem
- [BLENDER_5_GUARDRAILS.md](../BLENDER_5_GUARDRAILS.md) - MCP-first policy

---

*This document replaces conversation context. Update it as you make discoveries.*
