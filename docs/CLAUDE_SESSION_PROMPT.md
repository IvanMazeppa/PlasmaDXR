# Claude Session Prompt for NanoVDB Development

**Copy and paste this prompt at the start of a new Claude Code session to restore full context.**

---

## Prompt

```
I'm working on the NanoVDB volumetric rendering pipeline for PlasmaDX-Clean, a DirectX 12 raytracing engine. Start by reading the session context document to understand the current state:

1. Read `docs/NanoVDB/NANOVDB_SESSION_CONTEXT.md` - Quick context restore
2. Read `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` - Full technical details
3. Check `git status` to see current branch and changes

## Project Summary

**Goal:** Blender 5.0 → NanoVDB → PlasmaDX pipeline for volumetric celestial bodies (nebulae, gas clouds, explosions).

**Current Status:**
- NanoVDB C++ runtime: PRODUCTION (100%)
- Ray marching shader: PRODUCTION (100%)
- Recipe library: 1/14 complete (hydrogen_cloud.md done)
- **KNOWN ISSUE:** File-loaded .nvdb grids may not render correctly (procedural fog works fine)

**Two-Worktree Architecture:**
- PlasmaDX-Blender (feature/blender-integration) - Recipes, scripts, docs
- PlasmaDX-NanoVDB (feature/nanovdb-animated-assets) - Shader, C++
- PlasmaDX-Clean (main/0.23.x) - Integration hub

**Blender 5.0 Warning:** Claude's training data is 10+ months behind Blender 5.0. ALWAYS use the `blender-manual` MCP tools to verify API before writing bpy code. Known changes: No BLOSC compression (use ZIP).

**MCP Tools Available:**
- blender-manual: search_python_api, search_vdb_workflow, search_bpy_types, read_page
- gaussian-analyzer: simulate_material_properties, analyze_gaussian_parameters
- materials-council: generate_material_shader, generate_particle_struct

**Key Files:**
- src/rendering/NanoVDBSystem.h/cpp - C++ implementation
- shaders/volumetric/nanovdb_raymarch.hlsl - 586-line ray marcher with PNanoVDB
- scripts/convert_vdb_to_nvdb.py - VDB conversion
- docs/blender_recipes/ - Recipe library

**Immediate Priorities:**
1. Debug file-loaded grid rendering (debugMode=1: green=good, cyan=problem)
2. Complete end-to-end pipeline test
3. Add dark_nebula.md and stellar_flare.md recipes

Please start by reading the context documents, then let me know what you'd like to work on.
```

---

## Quick Reference for Session

### Build Commands
```bash
cmake --build build --target PlasmaDX-Clean
python scripts/convert_vdb_to_nvdb.py input.vdb
```

### Debug Mode Colors (in shader)
- 🟢 Green = grid density found (working)
- 🟣 Magenta = procedural density found
- 🔵 Cyan = inside AABB but no density (investigate)
- 🔴 Red tint = ray missed AABB entirely

### MCP Quick Calls
```python
# Verify Blender API
mcp__blender-manual__search_python_api("bpy.ops.fluid")
mcp__blender-manual__search_vdb_workflow("export openvdb")

# Analyze materials
mcp__gaussian-analyzer__simulate_material_properties(material_type="GAS_CLOUD")
```

### Git Worktree Navigation
```bash
git worktree list
cd ../PlasmaDX-Blender   # Asset creation
cd ../PlasmaDX-NanoVDB   # Rendering dev
cd ../PlasmaDX-Clean     # Integration
```

---

## Alternative Short Prompt

If you need a shorter version:

```
I'm working on NanoVDB volumetric rendering for PlasmaDX-Clean (DX12 raytracing engine).

Read these to restore context:
- docs/NanoVDB/NANOVDB_SESSION_CONTEXT.md
- docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md

Key info: Blender→NanoVDB→PlasmaDX pipeline. 70% complete. Known issue: file-loaded .nvdb grids may not render (procedural works). Use blender-manual MCP for Blender 5.0 API verification.

What would you like to work on?
```

---

*This prompt restores full NanoVDB development context for new Claude Code sessions.*
