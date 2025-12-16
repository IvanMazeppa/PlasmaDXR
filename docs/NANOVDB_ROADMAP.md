# NanoVDB Development Roadmap

**Version:** 1.0
**Status:** Active Development
**Last Updated:** 2025-12-11

---

## Executive Summary

This roadmap consolidates NanoVDB volumetric rendering development with the Blender asset creation pipeline. The system uses a **two-worktree architecture** to enable parallel development while maintaining integration.

### Current State

| Component | Status | Completion |
|-----------|--------|------------|
| NanoVDBSystem C++ | Production | 100% |
| Ray Marching Shader | Production | 100% |
| Animation Support | Production | 100% |
| VDB Conversion Scripts | Working | 90% |
| Blender Integration | In Progress | 60% |
| Recipe Library | Started | 10% |
| Agent Ecosystem | Production | 80% |

### Target State

- Seamless Blender → PlasmaDX workflow
- Rich recipe library for celestial bodies
- Automated conversion pipeline
- Performance-optimized rendering

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT ARCHITECTURE                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────┐        ┌──────────────────────────────────┐  │
│  │    BLENDER WORKTREE      │        │      NANOVDB WORKTREE            │  │
│  │   (Asset Creation)       │        │   (Rendering Development)        │  │
│  │                          │        │                                  │  │
│  │  Branch:                 │        │  Branch:                         │  │
│  │  feature/blender-        │        │  feature/nanovdb-animated-       │  │
│  │  integration             │        │  assets                          │  │
│  │                          │        │                                  │  │
│  │  Focus:                  │        │  Focus:                          │  │
│  │  - Recipes               │   ──►  │  - NanoVDBSystem.cpp             │  │
│  │  - Blender scripts       │  VDB   │  - Ray marching shader           │  │
│  │  - Documentation         │ files  │  - Animation system              │  │
│  │  - Learning guides       │        │  - Performance optimization      │  │
│  │                          │        │                                  │  │
│  └──────────────────────────┘        └──────────────────────────────────┘  │
│              │                                      │                       │
│              └──────────────────┬──────────────────┘                       │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │      MAIN WORKTREE      │                              │
│                    │   (Integration Hub)     │                              │
│                    │                         │                              │
│                    │   Branch: main/0.23.x   │                              │
│                    │                         │                              │
│                    │   - Production builds   │                              │
│                    │   - Feature merges      │                              │
│                    │   - Release management  │                              │
│                    └─────────────────────────┘                              │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Development Phases

### Phase 1: Foundation (COMPLETE)

**Status:** Production Ready

| Task | Status | Notes |
|------|--------|-------|
| NanoVDBSystem C++ implementation | DONE | Full API in place |
| Ray marching shader | DONE | PNanoVDB integration |
| Procedural fog sphere | DONE | Shader-based, no GPU memory |
| File loading (.nvdb) | DONE | Direct buffer upload |
| Animation sequence loading | DONE | Multi-frame support |
| Runtime controls (ImGui) | DONE | All parameters adjustable |
| Depth occlusion | DONE | Respects existing geometry |
| Multi-light scattering | DONE | Up to 16 lights |

**Key Files:**
- `src/rendering/NanoVDBSystem.h/cpp`
- `shaders/volumetric/nanovdb_raymarch.hlsl`

---

### Phase 2: Asset Pipeline (IN PROGRESS)

**Status:** 60% Complete

| Task | Status | Notes |
|------|--------|-------|
| VDB→NanoVDB conversion scripts | DONE | `scripts/convert_vdb_to_nvdb.py` |
| VDB inspection tool | DONE | `scripts/inspect_vdb.py` |
| Blender workflow documentation | DONE | `docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md` |
| blender-manual MCP server | DONE | 12 tools operational |
| blender-scripting agent | DONE | Agent spec complete |
| celestial-body-curator agent | DONE | Agent spec complete |
| Hydrogen cloud recipe | DONE | First complete recipe |
| Additional recipes | IN PROGRESS | 1/14 complete |
| Blender automation scripts | IN PROGRESS | Basic scripts exist |

**Key Files:**
- `docs/blender_recipes/`
- `agents/blender-*/`
- `scripts/convert_vdb_to_nvdb.py`

**Remaining Work:**
1. Complete recipe library (13 more recipes)
2. Automate batch conversion
3. Test full workflow end-to-end

---

### Phase 3: Performance Optimization (PLANNED)

**Status:** Not Started

| Task | Priority | Est. Effort |
|------|----------|-------------|
| Step size auto-tuning | High | 2 days |
| LOD system for large grids | High | 1 week |
| Brick caching for animation | Medium | 1 week |
| DLSS integration | Medium | 2 days |
| Temporal reprojection | Low | 2 weeks |

**Performance Targets:**
- 10ms max for 256³ grid @ 1080p
- 30 FPS minimum for animated sequences
- <100MB GPU memory for typical assets

---

### Phase 4: Advanced Features (FUTURE)

**Status:** Planning

| Feature | Description | Priority |
|---------|-------------|----------|
| Multi-volume compositing | Layer multiple VDB files | Medium |
| Hot-reload | Watch directory for VDB changes | Medium |
| Geometry Nodes support | Procedural VDB from GeoNodes | Low |
| Temperature-based emission | Blackbody from temperature grid | Low |
| Velocity-based motion blur | Use velocity grid for blur | Low |

---

## Recipe Library Status

### Complete Recipes

| Recipe | Category | Status |
|--------|----------|--------|
| hydrogen_cloud.md | Emission Nebulae | COMPLETE |

### Planned Recipes

| Recipe | Category | Priority |
|--------|----------|----------|
| emission_pillar.md | Emission Nebulae | High |
| dark_nebula.md | Dark Structures | High |
| supernova_remnant.md | Explosions | High |
| stellar_flare.md | Explosions | Medium |
| protoplanetary_disk.md | Stellar Phenomena | Medium |
| accretion_corona.md | Stellar Phenomena | Medium |
| orion_style.md | Emission Nebulae | Low |
| coronal_ejection.md | Explosions | Low |
| planetary_nebula.md | Stellar Phenomena | Low |
| dust_lane.md | Dark Structures | Low |

---

## Agent Ecosystem Status

### Production Agents

| Agent | Type | Status |
|-------|------|--------|
| blender-manual | MCP Server | PRODUCTION |
| blender-scripting | Agent Spec | PRODUCTION |
| celestial-body-curator | Agent Spec | PRODUCTION |
| blender-diagnostics | Agent Spec | PRODUCTION |
| gaussian-analyzer | MCP Server | PRODUCTION |
| materials-council | MCP Server | PRODUCTION |
| dxr-volumetric-pyro-specialist | MCP Server | PRODUCTION |

### Proposed Agents

| Agent | Type | Priority |
|-------|------|----------|
| vdb-converter-agent | MCP Server | High |
| blender-scene-validator | MCP Server | Medium |
| nanovdb-profiler | MCP Server | Low |
| vdb-pipeline-orchestrator | Agent Spec | Low |

---

## Integration Points

### With Existing PlasmaDX Systems

| System | Integration | Status |
|--------|-------------|--------|
| Gaussian Particle Renderer | Additive compositing | WORKING |
| Multi-Light System | Volumetric scattering | WORKING |
| DLSS | Render resolution handling | WORKING |
| RTXDI | Potential future integration | PLANNED |
| Probe Grid | Indirect lighting for volumes | PLANNED |

### With Blender Worktree

| Integration | Direction | Status |
|-------------|-----------|--------|
| Recipe library | Blender → Main | ACTIVE |
| VDB files | Blender → VDBs/ | ACTIVE |
| Agent specs | Blender → Main | ACTIVE |
| Conversion scripts | Main → All | SHARED |

---

## Quick Start Guide

### For Asset Creation (Blender)

```bash
# 1. Switch to Blender worktree
cd ../PlasmaDX-Blender

# 2. Open Blender project
# VDBs/Blender_projects/hydrogen_cloud.blend

# 3. Follow recipe
# docs/blender_recipes/emission_nebulae/hydrogen_cloud.md

# 4. Export VDB (in Blender)
# Domain > Cache > Format: OpenVDB > Bake

# 5. Convert to NanoVDB
cd ../PlasmaDX-Clean
python scripts/convert_vdb_to_nvdb.py VDBs/Blender_projects/vdb_cache/smoke_0050.vdb
```

### For Rendering Development (NanoVDB)

```bash
# 1. Switch to NanoVDB worktree
cd ../PlasmaDX-NanoVDB

# 2. Edit NanoVDB system
# src/rendering/NanoVDBSystem.cpp

# 3. Edit shader
# shaders/volumetric/nanovdb_raymarch.hlsl

# 4. Build and test
cmake --build build --target PlasmaDX-Clean

# 5. Merge to main when stable
cd ../PlasmaDX-Clean
git merge feature/nanovdb-animated-assets
```

---

## Metrics & Goals

### Performance Goals

| Metric | Current | Target |
|--------|---------|--------|
| Procedural fog FPS impact | -2-5 | <-5 |
| 128³ grid FPS impact | -3-5 | <-5 |
| 256³ grid FPS impact | -10-15 | <-10 |
| Animation frame switch | <1ms | <1ms |

### Quality Goals

| Metric | Current | Target |
|--------|---------|--------|
| Recipe completion | 1/14 | 14/14 |
| Conversion success rate | ~90% | 99% |
| Documentation coverage | 80% | 95% |
| Agent tool coverage | 80% | 95% |

---

## Documentation Index

### This Directory (docs/NanoVDB/)

| Document | Purpose |
|----------|---------|
| [NANOVDB_ROADMAP.md](./NANOVDB_ROADMAP.md) | This roadmap |
| [NANOVDB_SYSTEM_OVERVIEW.md](./NANOVDB_SYSTEM_OVERVIEW.md) | Technical overview |
| [TWO_WORKTREE_WORKFLOW.md](./TWO_WORKTREE_WORKFLOW.md) | Development architecture |
| [VDB_PIPELINE_AGENTS.md](./VDB_PIPELINE_AGENTS.md) | Agent ecosystem |

### Related Documentation

| Document | Location |
|----------|----------|
| Blender Workflow Spec | `docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md` |
| Recipe Library | `docs/blender_recipes/README.md` |
| CLAUDE.md (project context) | `CLAUDE.md` |
| Agent Specs | `agents/*/AGENT_PROMPT.md` |

---

## Next Steps

### Immediate (This Week)

1. Complete dark_nebula recipe (high priority, low complexity)
2. Test full pipeline end-to-end
3. Document any workflow pain points

### Short-Term (Next 2 Weeks)

1. Add 3-5 more high-priority recipes
2. Implement vdb-converter-agent
3. Performance profiling for existing grids

### Medium-Term (Next Month)

1. Complete recipe library (all 14 recipes)
2. LOD system for large grids
3. Hot-reload capability

---

*Document maintained by: Claude Code Agent Ecosystem*
