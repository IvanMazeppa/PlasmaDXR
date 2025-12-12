# Blender Session Context - Start Here

**Purpose:** Load this document at the start of any Blender worktree development session to restore full context.

**Last Updated:** 2025-12-12

---

## Quick Summary

You're working on the **Blender side of the VDB pipeline** - creating volumetric celestial assets in Blender 5.0 for export to PlasmaDX-Clean. The goal is production-ready recipes that output VDB files for real-time rendering.

---

## Project Goal

Create stunning volumetric celestial bodies using Blender 5.0:
1. **Mantaflow Simulations** - Smoke/fire physics for organic nebula shapes
2. **Geometry Nodes** - Procedural generation for parametric control
3. **VDB Export** - OpenVDB format for PlasmaDX consumption

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| blender-manual MCP | Production | 12 tools operational |
| blender-scripting agent | Production | Agent prompt complete |
| celestial-body-curator agent | Production | Agent prompt complete |
| Recipe Library | 1/14 complete | hydrogen_cloud.md done |
| Automation scripts | Planned | Not yet implemented |
| Blender 5.0 | Required | May need installation |

---

## ⚠️ CRITICAL: Blender 5.0 API Warning

**Claude's training data is 10+ months behind Blender 5.0!**

Before writing ANY Blender Python code:
1. **ALWAYS use `blender-manual` MCP** to verify current API
2. **Never assume** API is the same as older versions
3. **Known breaking changes:**
   - No BLOSC compression (use ZIP or NONE)
   - bpy.ops.geometry changes
   - New modifier stack behavior

**MCP-First Workflow:**
```python
# WRONG - Don't guess at API
domain_settings.openvdb_cache_compress_type = 'BLOSC'  # May not exist!

# RIGHT - Verify with MCP first
# 1. search_python_api("FluidDomainSettings")
# 2. read_page() the documentation
# 3. Then write code based on verified API
```

---

## Key Files

### Recipe Library
- `docs/blender_recipes/README.md` - Index and quick reference
- `docs/blender_recipes/emission_nebulae/hydrogen_cloud.md` - First complete recipe

### Agent Prompts
- `agents/blender-scripting/AGENT_PROMPT.md` - Script generation expertise
- `agents/celestial-body-curator/AGENT_PROMPT.md` - Recipe authoring expertise
- `agents/blender-diagnostics/AGENT_PROMPT.md` - Troubleshooting expertise

### Workflow Documentation
- `docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md` - Full pipeline specification
- `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` - NanoVDB rendering context

### Output Directories
- `VDBs/Blender_projects/` - Source .blend files
- `VDBs/NanoVDB/` - Converted .nvdb files (after conversion)

---

## Recipe Library Status

### Complete (1)
- ✅ `hydrogen_cloud.md` - Emission nebula with Mantaflow

### High Priority (Add Next)
- 🔲 `dark_nebula.md` - Absorption-only, simpler to create
- 🔲 `supernova_remnant.md` - Animated explosion effect
- 🔲 `stellar_flare.md` - Curved solar eruption

### Medium Priority
- 🔲 `emission_pillar.md` - Towering gas column
- 🔲 `protoplanetary_disk.md` - Rotating disk
- 🔲 `accretion_corona.md` - Hot gas near star

### Lower Priority
- 🔲 `orion_style.md` - Complex multi-region nebula
- 🔲 `coronal_ejection.md` - Animated CME
- 🔲 `planetary_nebula.md` - Shell structure
- 🔲 `dust_lane.md` - Dark absorption band

---

## MCP Tools Quick Reference

### API Verification (USE FIRST)
```python
# Search Python API
mcp__blender-manual__search_python_api("bpy.ops.fluid")
mcp__blender-manual__search_bpy_types("FluidDomainSettings")
mcp__blender-manual__search_bpy_operators("fluid", "bake")
```

### VDB-Specific
```python
# VDB export documentation
mcp__blender-manual__search_vdb_workflow("export openvdb")
mcp__blender-manual__search_vdb_workflow("cache smoke")
```

### General Documentation
```python
# Manual search
mcp__blender-manual__search_manual("volume rendering")
mcp__blender-manual__search_tutorials("volumetrics", "smoke")
mcp__blender-manual__read_page("physics/fluid/type/domain/cache.html")
```

---

## VDB Export Settings

### PlasmaDX-Compatible Configuration

```python
# Recommended settings for domain
settings = domain.modifiers['Fluid'].domain_settings
settings.cache_data_format = 'OPENVDB'
settings.cache_directory = '//vdb_cache/'
settings.cache_precision = 'HALF'              # 16-bit, good quality/size
# For compression: Use ZIP (BLOSC may not work in 5.0)
```

### Required Grids by Celestial Type

| Body Type | density | temperature | velocity |
|-----------|---------|-------------|----------|
| Emission Nebula | Required | Optional | No |
| Dark Nebula | Required | No | No |
| Supernova | Required | Required | Optional |
| Stellar Flare | Required | Required | No |
| Protoplanetary Disk | Required | Optional | Optional |

### Resolution Guidelines

| Level | Resolution | File Size/Frame | Use Case |
|-------|------------|-----------------|----------|
| Preview | 64³ | 2-5 MB | Testing |
| Standard | 128³ | 10-30 MB | Most use |
| High | 256³ | 50-150 MB | Hero shots |

---

## Blender → PlasmaDX Property Mapping

| Blender Property | PlasmaDX Property | Conversion |
|------------------|-------------------|------------|
| Volume Density | `densityScale` | `density * 0.4` |
| Anisotropy | `scatteringCoeff` | Direct (-1 to +1) |
| Emission Strength | `emissionStrength` | `strength * 0.25` |
| Volume Color | Procedural color | Tone-mapped |
| Absorption Color | Inverted | `1 - color` |
| Temperature | `TemperatureToColor()` | Blackbody curve |

### Material Type Mapping

| Celestial Body | Recipe | PlasmaDX Material |
|----------------|--------|-------------------|
| Emission Nebula | hydrogen_cloud.md | GAS_CLOUD |
| Dark Nebula | dark_nebula.md | DUST |
| Supernova | supernova_remnant.md | PLASMA |
| Stellar Corona | accretion_corona.md | PLASMA |

---

## Two-Worktree Architecture

| Worktree | Branch | Focus |
|----------|--------|-------|
| **PlasmaDX-Blender** | feature/blender-integration | Recipes, scripts, docs |
| PlasmaDX-NanoVDB | feature/nanovdb-animated-assets | Shader, C++ |
| PlasmaDX-Clean | main/0.23.x | Integration |

This session works in **PlasmaDX-Blender** - asset creation side.

---

## Session Workflow

1. **Read this document** to restore context
2. **Check git status** in your worktree
3. **Verify Blender 5.0** is available
4. **Use blender-manual MCP** before writing ANY bpy code
5. **Follow existing recipes** as templates for new ones
6. **Test manually** before creating automation scripts
7. **Update this doc** with discoveries

---

## Immediate Priorities

1. **Test hydrogen_cloud.md recipe** - Verify it works end-to-end
2. **Create dark_nebula.md recipe** - Simple absorption-only volume
3. **Create stellar_flare.md recipe** - Animated effect
4. **Document any Blender 5.0 API changes** found during testing

---

## Integration with NanoVDB Worktree

After creating VDB assets here:
1. Export OpenVDB from Blender (Domain > Cache)
2. Convert: `python scripts/convert_vdb_to_nvdb.py input.vdb`
3. Load in PlasmaDX: `nanoVDB->LoadFromFile("path/to/file.nvdb")`
4. Adjust visual parameters in ImGui

See `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` for rendering details.

---

## Related Documentation

- [NANOVDB_UNIFIED_ROADMAP_V1.md](./NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md) - Rendering pipeline
- [BLENDER_PLASMADX_WORKFLOW_SPEC.md](./BLENDER_PLASMADX_WORKFLOW_SPEC.md) - Full spec
- [Recipe Library](./blender_recipes/README.md) - Recipe index
- [Hydrogen Cloud Recipe](./blender_recipes/emission_nebulae/hydrogen_cloud.md) - Example

---

*This document replaces conversation context. Update it as you make discoveries.*
