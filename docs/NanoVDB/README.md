# NanoVDB Documentation

This directory contains documentation for the NanoVDB volumetric rendering system and its integration with the Blender asset creation pipeline.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [NANOVDB_ROADMAP.md](./NANOVDB_ROADMAP.md) | Development roadmap and status |
| [NANOVDB_SYSTEM_OVERVIEW.md](./NANOVDB_SYSTEM_OVERVIEW.md) | Technical system overview |
| [TWO_WORKTREE_WORKFLOW.md](./TWO_WORKTREE_WORKFLOW.md) | Development architecture |
| [VDB_PIPELINE_AGENTS.md](./VDB_PIPELINE_AGENTS.md) | AI agent ecosystem |

---

## System Summary

The NanoVDB system provides GPU-accelerated volumetric rendering for celestial bodies:

- **Procedural Fog:** Runtime-generated amorphous gas clouds
- **File Loading:** Load .nvdb grids from disk
- **Animation:** Multi-frame volumetric sequences
- **Runtime Control:** ImGui sliders for all parameters

---

## Development Architecture

```
PlasmaDX-Blender       →   VDBs/    →   PlasmaDX-NanoVDB   →   PlasmaDX-Clean
(Asset Creation)           (Shared)     (Rendering Dev)        (Integration)
```

Two git worktrees enable parallel development:
- **PlasmaDX-Blender** - Recipes, Blender scripts, documentation
- **PlasmaDX-NanoVDB** - Shader development, C++ implementation

---

## Getting Started

### Load an Existing VDB

```cpp
// In Application.cpp or similar
m_nanoVDB->LoadFromFile("VDBs/NanoVDB/cloud_01.nvdb");
m_nanoVDB->SetEnabled(true);
m_nanoVDB->SetDensityScale(1.0f);
m_nanoVDB->SetEmissionStrength(0.5f);
```

### Create New Asset

1. Follow a recipe in `docs/blender_recipes/`
2. Export from Blender as OpenVDB
3. Convert: `python scripts/convert_vdb_to_nvdb.py input.vdb`
4. Load in PlasmaDX

---

## Related Documentation

- [Blender Workflow Spec](../BLENDER_PLASMADX_WORKFLOW_SPEC.md)
- [Recipe Library](../blender_recipes/README.md)
- [CLAUDE.md](../../CLAUDE.md) - Project context

---

*Last Updated: 2025-12-11*
