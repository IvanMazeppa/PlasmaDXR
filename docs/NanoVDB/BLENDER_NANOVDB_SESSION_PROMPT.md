# Blender + NanoVDB Session Recovery Prompt

**Purpose:** Copy this entire prompt at the start of a Claude Code session (in the Blender or NanoVDB worktree) to restore full context for volumetric asset development.

**Last Updated:** 2025-12-12

---

## The Prompt

```
I'm working on the Blender → NanoVDB → PlasmaDX volumetric rendering pipeline. This is a two-worktree architecture for creating stunning celestial bodies (nebulae, gas clouds, supernovae) in Blender 5.0 and rendering them in real-time with DXR 1.1.

## Restore Context

Start by reading these documents in order:

1. `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` - Full technical roadmap (AUTHORITATIVE)
2. `docs/BLENDER_SESSION_CONTEXT.md` - Blender-side context
3. `docs/NanoVDB/TWO_WORKTREE_WORKFLOW.md` - Git workflow between worktrees
4. `docs/NanoVDB/VDB_PIPELINE_AGENTS.md` - Available agents and tools

Then check `git status` and `git worktree list` to see current state.

---

## Project Summary

**Goal:** Production pipeline: Blender 5.0 Mantaflow → OpenVDB export → NanoVDB conversion → PlasmaDX real-time rendering

**Overall Completion:** ~70%

| Subsystem | Status | Notes |
|-----------|--------|-------|
| NanoVDB C++ Runtime | PRODUCTION (100%) | File loading, animation, procedural fog |
| Ray Marching Shader | PRODUCTION (100%) | 586 lines, PNanoVDB tree traversal |
| Blender Agent Ecosystem | PRODUCTION (80%) | blender-manual MCP, scripting, curator, diagnostics |
| Recipe Library | STARTED (10%) | 1/14 complete (hydrogen_cloud.md) |
| Automation Scripts | STARTED | hydrogen_cloud_v1.py created |

**KNOWN CRITICAL ISSUE:** File-loaded .nvdb grids may not render correctly (procedural fog works fine). Debug with `debugMode=1`:
- Green = grid density found (working)
- Cyan = inside AABB but no density (investigate)
- Magenta = procedural density

---

## Two-Worktree Architecture

| Worktree | Branch | Focus | tmux Session |
|----------|--------|-------|--------------|
| **PlasmaDX-Blender** | feature/blender-integration | Recipes, scripts, Blender docs | `tb` (claude-blender) |
| **PlasmaDX-NanoVDB** | feature/nanovdb-animated-assets | Shader, C++, rendering | `tn` (claude-nanovdb) |
| PlasmaDX-Clean | main/0.23.x | Integration hub | (main window) |

**tmux Quick Reference:**
- `tb` - attach to Blender worktree session
- `tn` - attach to NanoVDB worktree session
- `tls` - list all sessions
- `Ctrl+b d` - detach from session

---

## CRITICAL: Blender 5.0 API Warning

**Claude's training data is 10+ months behind Blender 5.0!**

Before writing ANY bpy Python code:
1. **ALWAYS use `blender-manual` MCP tools first** to verify API
2. **Known breaking changes:**
   - No BLOSC compression (use ZIP or NONE)
   - bpy.ops.geometry changes
   - New modifier stack behavior
3. See `docs/BLENDER_5_GUARDRAILS.md` for full policy

**MCP-First Workflow:**
```python
# WRONG - Don't guess at API
domain_settings.cache_compress_type = 'BLOSC'  # May not exist!

# RIGHT - Verify with MCP first
mcp__blender-manual__search_python_api("FluidDomainSettings")
mcp__blender-manual__search_bpy_types("FluidDomainSettings")
# Then write code based on verified API
```

---

## Available MCP Tools

### blender-manual (Blender 5.0 Documentation)
```python
# API verification (USE FIRST)
mcp__blender-manual__search_python_api("bpy.ops.fluid")
mcp__blender-manual__search_bpy_types("FluidDomainSettings")
mcp__blender-manual__search_bpy_operators("fluid", "bake")

# VDB-specific
mcp__blender-manual__search_vdb_workflow("export openvdb")
mcp__blender-manual__search_vdb_workflow("cache smoke")

# General
mcp__blender-manual__search_manual("volume rendering")
mcp__blender-manual__read_page("physics/fluid/type/domain/cache.html")
```

### gaussian-analyzer (Material Validation)
```python
mcp__gaussian-analyzer__simulate_material_properties(material_type="GAS_CLOUD")
mcp__gaussian-analyzer__analyze_gaussian_parameters()
mcp__gaussian-analyzer__estimate_performance_impact(particle_struct_bytes=48)
```

### materials-council (Property Mapping)
```python
mcp__material-system-engineer__generate_material_shader(material_type="GAS_CLOUD", properties={...})
mcp__material-system-engineer__search_codebase(pattern="NanoVDB")
```

### dxr-volumetric-pyro-specialist (Explosion Design)
```python
mcp__dxr-volumetric-pyro-specialist__design_explosion_effect(effect_type="supernova")
mcp__dxr-volumetric-pyro-specialist__estimate_pyro_performance(particle_count=10000)
```

---

## Key Files

### Blender Worktree (Asset Creation)
- `docs/blender_recipes/README.md` - Recipe library index
- `docs/blender_recipes/emission_nebulae/hydrogen_cloud.md` - First complete recipe
- `docs/blender_recipes/scripts/hydrogen_cloud_v1.py` - Automation script
- `agents/blender-scripting/AGENT_PROMPT.md` - Script generation expertise
- `agents/celestial-body-curator/AGENT_PROMPT.md` - Recipe authoring expertise

### NanoVDB Worktree (Rendering)
- `src/rendering/NanoVDBSystem.h/cpp` - C++ implementation
- `shaders/volumetric/nanovdb_raymarch.hlsl` - 586-line ray marcher
- `shaders/volumetric/PNanoVDB.h` - HLSL NanoVDB header
- `scripts/convert_vdb_to_nvdb.py` - VDB conversion
- `scripts/inspect_vdb.py` - VDB inspection

### Shared (Main Worktree)
- `VDBs/NanoVDB/` - Production .nvdb files
- `VDBs/Blender_projects/` - Source .blend files

---

## Recipe Library Status

### Complete (1)
- hydrogen_cloud.md - Emission nebula with Mantaflow + automation script

### High Priority (Add Next)
- dark_nebula.md - Absorption-only, simpler to create
- stellar_flare.md - Animated expansion effect
- supernova_remnant.md - Animated explosion

### Medium Priority
- emission_pillar.md - Towering gas column
- protoplanetary_disk.md - Rotating disk
- accretion_corona.md - Hot gas near star

---

## Blender → PlasmaDX Property Mapping

| Blender Property | PlasmaDX Property | Conversion |
|------------------|-------------------|------------|
| Volume Density | `densityScale` | `density * 0.4` |
| Anisotropy | `scatteringCoeff` | Direct (-1 to +1) |
| Emission Strength | `emissionStrength` | `strength * 0.25` |
| Temperature | `TemperatureToColor()` | Blackbody curve |

### Material Type Mapping
| Celestial Body | Recipe | PlasmaDX Material |
|----------------|--------|-------------------|
| Emission Nebula | hydrogen_cloud.md | GAS_CLOUD |
| Dark Nebula | dark_nebula.md | DUST |
| Supernova | supernova_remnant.md | PLASMA |

---

## Immediate Priorities

### Priority 1: Debug File-Loaded Grid Rendering
1. Build PlasmaDX-Clean in Debug
2. Load a known-good .nvdb file
3. Enable `debugMode=1` in ImGui
4. Check colors (green=good, cyan=problem)
5. Compare `gridWorldMin/Max` with actual grid bounds
6. Verify coordinate system (Blender Z-up → PlasmaDX Y-up)

### Priority 2: Test End-to-End Pipeline
1. Open `VDBs/Blender_projects/hydrogen_cloud.blend` (or create new)
2. Bake Mantaflow simulation
3. Export OpenVDB (Domain > Cache)
4. Convert: `python scripts/convert_vdb_to_nvdb.py input.vdb`
5. Load in PlasmaDX
6. Verify visual output

### Priority 3: Add More Recipes
Start with simpler recipes:
1. **dark_nebula.md** - Low emission, high absorption
2. **stellar_flare.md** - Animated expansion

---

## Build Commands

```bash
# Build PlasmaDX-Clean
cmake --build build --target PlasmaDX-Clean

# Convert VDB to NanoVDB
python scripts/convert_vdb_to_nvdb.py VDBs/Blender_projects/smoke.vdb

# Inspect VDB contents
python scripts/inspect_vdb.py VDBs/Blender_projects/smoke.vdb

# Switch worktrees
cd ../PlasmaDX-Blender   # Asset creation
cd ../PlasmaDX-NanoVDB   # Rendering dev
cd ../PlasmaDX-Clean     # Integration
```

---

## What would you like to work on?

Options:
1. Debug file-loaded grid rendering issue
2. Create/test a Blender recipe
3. Write automation scripts
4. Optimize NanoVDB shader performance
5. Add new recipe to the library
6. Something else

Please read the context documents first, then let me know!
```

---

## Short Version (If Needed)

```
I'm working on Blender → NanoVDB → PlasmaDX volumetric rendering pipeline.

Read these to restore context:
- docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md (authoritative roadmap)
- docs/BLENDER_SESSION_CONTEXT.md (Blender-side context)
- docs/NanoVDB/VDB_PIPELINE_AGENTS.md (available agents)

Key info:
- Two-worktree architecture: PlasmaDX-Blender (recipes) + PlasmaDX-NanoVDB (rendering)
- 70% complete overall
- KNOWN ISSUE: File-loaded .nvdb grids may not render (procedural works)
- CRITICAL: Use blender-manual MCP before ANY bpy code (Claude's data is 10+ months behind Blender 5.0)
- Recipe library: 1/14 complete (hydrogen_cloud.md)
- tmux sessions: tb (blender), tn (nanovdb)

What would you like to work on?
```

---

## Quick Reference Card

### tmux Sessions (B1 Main PC)
| Alias | Session | Worktree |
|-------|---------|----------|
| `tb` | claude-blender | PlasmaDX-Blender |
| `tn` | claude-nanovdb | PlasmaDX-NanoVDB |
| `tp` | claude-pinn | PlasmaDX-PINN-v4 |
| `tm` | claude-multi | PlasmaDX-MultiAgent |

### Debug Mode Colors (nanovdb_raymarch.hlsl)
| Color | Meaning |
|-------|---------|
| Green | Grid density found (working) |
| Cyan | Inside AABB but no density (problem) |
| Magenta | Procedural density found |
| Red tint | Ray missed AABB entirely |

### VDB Export Settings (Blender 5.0)
```python
settings.cache_data_format = 'OPENVDB'
settings.cache_directory = '//vdb_cache/'
settings.cache_precision = 'HALF'
# Use ZIP compression (BLOSC may not work in 5.0)
```

---

*This prompt restores full Blender + NanoVDB development context for new Claude Code sessions.*
