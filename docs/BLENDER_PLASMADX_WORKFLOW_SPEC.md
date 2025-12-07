# Blender-PlasmaDX VDB Workflow Specification

**Version:** 1.0.0
**Status:** Architecture Complete, Implementation In Progress
**Created:** 2025-12-07
**Author:** Claude Code Agent Ecosystem

---

## Executive Summary

This document specifies a complete workflow system for creating volumetric celestial bodies (nebulae, supernovae, gas clouds) in Blender 5.0 and rendering them in real-time in PlasmaDX-Clean via OpenVDB/NanoVDB export.

The system consists of:

- **3 AI Agents** for documentation, scripting, and recipe curation
- **1 MCP Server** providing 12 Blender documentation search tools
- **1 Recipe Library** with production-ready celestial body creation guides
- **Integration hooks** to existing PlasmaDX rendering agents

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Data Flow](#4-data-flow)
5. [Implementation Status](#5-implementation-status)
6. [Future Enhancements](#6-future-enhancements)
7. [Appendices](#7-appendices)

---

## 1. System Overview

### 1.1 Problem Statement

PlasmaDX-Clean's procedural particle generation excels at infinite variation but lacks artistic control for hero assets. Pre-authored volumetric content (nebulae, explosions) created in Blender offers precise artistic direction, but:

- Ben (the user) knows programming but not Blender's UI
- Blender's bpy API is powerful but complex with many hidden context requirements
- VDB export settings are non-obvious and easy to misconfigure
- No existing documentation bridges Blender volumetrics → PlasmaDX rendering

### 1.2 Solution

A multi-agent system that:

1. **Curates recipes** - Production-ready guides for specific celestial effects
2. **Generates scripts** - Python automation for Blender workflows
3. **Provides documentation** - Searchable Blender 5.0 manual + Python API
4. **Validates output** - Material property mapping to PlasmaDX renderer

### 1.3 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Try before building** | Manual workflow validation before automation |
| **Automate pain points** | Script only repetitive/error-prone tasks |
| **Single responsibility** | Each agent has one clear purpose |
| **Brutal honesty** | Direct feedback, no sugar-coating |
| **Incremental complexity** | Start simple, add features as proven needed |

---

## 2. Architecture

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER (Ben)                                      │
│                    "Create a nebula for my accretion disk"                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BLENDER WORKFLOW AGENTS                             │
│                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │ blender-manual  │   │blender-scripting│   │ celestial-body-curator  │   │
│  │   (MCP Server)  │   │    (Agent)      │   │       (Agent)           │   │
│  │                 │   │                 │   │                         │   │
│  │ 12 search tools │   │ bpy expertise   │   │ Recipe library          │   │
│  │ for Blender     │   │ Script gen      │   │ Celestial knowledge     │   │
│  │ documentation   │   │ Debug help      │   │ Standardized formats    │   │
│  └────────┬────────┘   └────────┬────────┘   └────────────┬────────────┘   │
│           │                     │                         │                 │
│           └─────────────────────┼─────────────────────────┘                 │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             BLENDER 5.0                                      │
│                                                                              │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────────────┐ │
│  │   Mantaflow   │   │ Geometry Nodes│   │     Volume Objects            │ │
│  │   (Fluids)    │   │ (Procedural)  │   │     (Static VDB)              │ │
│  └───────┬───────┘   └───────┬───────┘   └───────────────┬───────────────┘ │
│          │                   │                           │                  │
│          └───────────────────┼───────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │  OpenVDB Cache  │                                      │
│                    │  (.vdb files)   │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLASMADX INTEGRATION                                 │
│                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │gaussian-analyzer│   │ materials-      │   │dxr-volumetric-pyro-     │   │
│  │                 │   │ council         │   │specialist               │   │
│  │ Validate        │   │                 │   │                         │   │
│  │ properties      │   │ Map materials   │   │ Explosion design        │   │
│  └────────┬────────┘   └────────┬────────┘   └────────────┬────────────┘   │
│           │                     │                         │                 │
│           └─────────────────────┼─────────────────────────┘                 │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │   NanoVDB Loader        │                              │
│                    │   (C++ Runtime)         │                              │
│                    └────────────┬────────────┘                              │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │   DXR 1.1 Volumetric    │                              │
│                    │   Renderer              │                              │
│                    └─────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Interaction Model

```
┌────────────────────────────────────────────────────────────────┐
│                     QUERY ROUTING                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "How do I export VDB?"                                        │
│      → blender-manual (search_vdb_workflow)                    │
│      → blender-scripting (if script needed)                    │
│                                                                 │
│  "Create a supernova effect"                                   │
│      → celestial-body-curator (provides recipe)                │
│      → blender-scripting (generates script)                    │
│      → dxr-volumetric-pyro-specialist (explosion dynamics)     │
│                                                                 │
│  "My script throws an error"                                   │
│      → blender-scripting (debug help)                          │
│      → blender-manual (API documentation)                      │
│                                                                 │
│  "Will this render correctly in PlasmaDX?"                     │
│      → gaussian-analyzer (material validation)                 │
│      → materials-council (property mapping)                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 blender-manual MCP Server

**Location:** `agents/blender-manual/`
**Status:** ✅ Complete and Operational
**Type:** MCP Server (Model Context Protocol)

#### Purpose

Provides searchable access to Blender 5.0 documentation and Python API reference.

#### Tools (12 Total)

| Tool | Purpose | Example Query |
|------|---------|---------------|
| `search_manual` | General keyword search | "volume rendering" |
| `search_tutorials` | Learning resources | "fluid simulation" |
| `browse_hierarchy` | Navigate doc structure | "physics/fluid" |
| `read_page` | Full page content | "physics/fluid/cache.html" |
| `search_vdb_workflow` | VDB-specific search | "export openvdb" |
| `search_python_api` | bpy.ops/types/data | "bpy.ops.fluid.bake" |
| `search_nodes` | Shader/geometry nodes | "Principled Volume" |
| `search_modifiers` | Modifier documentation | "Fluid" |
| `search_semantic` | AI similarity search | "realistic smoke" |
| `list_api_modules` | List Python modules | "bpy" |
| `search_bpy_operators` | Search operators | "fluid", "bake" |
| `search_bpy_types` | Search types | "FluidDomainSettings" |

#### Technical Details

- **Index Size:** ~4,200 pages (2,200 manual + 2,000 API)
- **Cache:** `manual_index.json` (~50MB)
- **Embeddings:** `embeddings.npy` (semantic search, lazy-loaded)
- **Startup:** <2 seconds (cached), 60-90 seconds (fresh build)

#### Configuration

```json
{
  "mcpServers": {
    "blender-manual": {
      "command": "python",
      "args": ["agents/blender-manual/blender_server.py"]
    }
  }
}
```

---

### 3.2 blender-scripting Agent

**Location:** `agents/blender-scripting/AGENT_PROMPT.md`
**Status:** ✅ Specification Complete
**Type:** Claude Agent (prompt-based)

#### Purpose

Writes, debugs, and explains Python scripts that automate Blender workflows. Teaches Blender-specific patterns to programmers unfamiliar with the software.

#### Capabilities

| Capability | Description |
|------------|-------------|
| **Script Generation** | Create bpy scripts from natural language requests |
| **Debug Assistance** | Diagnose and fix script errors |
| **Pattern Teaching** | Explain context, modes, data paths |
| **API Lookup** | Use blender-manual tools to find correct API calls |

#### Key Knowledge Areas

1. **Context Requirements**
   - Active object must be set for many operators
   - Mode (Object/Edit/Sculpt) affects available operations
   - Selection state matters for batch operations

2. **bpy.data vs bpy.ops**
   - `bpy.data`: Direct access (fast, always works)
   - `bpy.ops`: Operator calls (context-dependent)

3. **FluidDomainSettings**
   - Cache configuration for VDB export
   - Resolution, compression, precision settings
   - Bake operators and their requirements

4. **Common Pitfalls**
   - Path escaping on Windows
   - Modifier name assumptions
   - Unsaved file before baking

#### Template Scripts Provided

| Script | Purpose |
|--------|---------|
| VDB Export Automation | Configure and bake fluid simulations |
| Quick Smoke Setup | Create domain + emitter ready for baking |

---

### 3.3 celestial-body-curator Agent

**Location:** `agents/celestial-body-curator/AGENT_PROMPT.md`
**Status:** ✅ Specification Complete
**Type:** Claude Agent (prompt-based)

#### Purpose

Authors and maintains a curated library of production-ready recipes for creating volumetric celestial phenomena. Bridges astrophysical accuracy with practical Blender workflows.

#### Responsibilities

| Responsibility | Description |
|----------------|-------------|
| **Recipe Authoring** | Create new celestial body recipes |
| **Recipe Maintenance** | Update recipes for new Blender versions |
| **Library Curation** | Organize, categorize, cross-reference |
| **Quality Assurance** | Ensure recipes produce working VDB output |

#### Recipe Format

Each recipe includes:

1. **Visual Reference** - Real-world examples, target appearance
2. **Astrophysical Properties** - Temperature, density, composition
3. **Blender Workflow** - Step-by-step with screenshots
4. **Python Automation** - One-click script
5. **Export Settings** - VDB configuration
6. **PlasmaDX Integration** - Material type mapping
7. **Troubleshooting** - Common issues and fixes

#### Collaboration

Works with:

- `blender-scripting`: Debug/optimize recipe scripts
- `gaussian-analyzer`: Validate material properties
- `dxr-volumetric-pyro-specialist`: Explosion effect design

---

### 3.4 Recipe Library

**Location:** `docs/blender_recipes/`
**Status:** 🔄 Structure Complete, Content In Progress

#### Directory Structure

```
docs/blender_recipes/
├── README.md                     # ✅ Complete - Index and reference
├── emission_nebulae/
│   ├── hydrogen_cloud.md         # ✅ Complete - Example recipe
│   ├── emission_pillar.md        # 📋 Planned
│   └── orion_style.md            # 📋 Planned
├── explosions/
│   ├── supernova_remnant.md      # 📋 Planned
│   ├── stellar_flare.md          # 📋 Planned
│   └── coronal_ejection.md       # 📋 Planned
├── stellar_phenomena/
│   ├── protoplanetary_disk.md    # 📋 Planned
│   ├── accretion_corona.md       # 📋 Planned
│   └── planetary_nebula.md       # 📋 Planned
├── dark_structures/
│   ├── dark_nebula.md            # 📋 Planned
│   └── dust_lane.md              # 📋 Planned
└── scripts/
    ├── quick_smoke_setup.py      # 📋 Planned
    ├── vdb_export_batch.py       # 📋 Planned
    └── celestial_presets.py      # 📋 Planned
```

#### Recipe Status Summary

| Category | Total | Complete | In Progress | Planned |
|----------|-------|----------|-------------|---------|
| Emission Nebulae | 3 | 1 | 0 | 2 |
| Explosions | 3 | 0 | 0 | 3 |
| Stellar Phenomena | 3 | 0 | 0 | 3 |
| Dark Structures | 2 | 0 | 0 | 2 |
| Scripts | 3 | 0 | 0 | 3 |
| **Total** | **14** | **1** | **0** | **13** |

---

### 3.5 Integration with Existing PlasmaDX Agents

#### gaussian-analyzer

**Role:** Validate material properties for real-time rendering

**Integration Point:** After VDB export, before PlasmaDX loading

**Tools Used:**

- `simulate_material_properties` - Test how Blender settings translate
- `estimate_performance_impact` - FPS impact of volume complexity

#### materials-council

**Role:** Map Blender volume properties to PlasmaDX particle materials

**Integration Point:** Recipe development, material type selection

**Property Mapping:**

| Blender Property | PlasmaDX Property | Notes |
|------------------|-------------------|-------|
| Density | opacity | Scale by 0.4 |
| Anisotropy | phase_function_g | Direct (-1 to +1) |
| Emission Strength | emission_multiplier | Scale by 0.25 |
| Color | albedo_rgb | Direct |
| Absorption Color | Inverted albedo | Invert for absorption |

#### dxr-volumetric-pyro-specialist

**Role:** Design explosion/fire effect parameters

**Integration Point:** Supernova, flare, CME recipe development

**Tools Used:**

- `design_explosion_effect` - Supernova dynamics
- `design_fire_effect` - Stellar fire parameters
- `estimate_pyro_performance` - FPS impact

---

## 4. Data Flow

### 4.1 Complete Workflow Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: RECIPE SELECTION                                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Request: "I need a gas cloud for my scene"                 │
│       │                                                           │
│       ▼                                                           │
│  celestial-body-curator                                          │
│       │                                                           │
│       ├──► Check recipe library                                  │
│       │    └──► docs/blender_recipes/emission_nebulae/           │
│       │                                                           │
│       ├──► Select appropriate recipe                             │
│       │    └──► hydrogen_cloud.md                                │
│       │                                                           │
│       └──► Provide recipe + script                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: BLENDER EXECUTION                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Option A: Manual Workflow                                       │
│       │                                                           │
│       ├──► Follow step-by-step recipe in Blender UI             │
│       ├──► Configure domain, emitter, materials                  │
│       └──► Bake simulation manually                              │
│                                                                   │
│  Option B: Python Automation                                     │
│       │                                                           │
│       ├──► Copy script from recipe                               │
│       ├──► Run in Blender's Scripting workspace                  │
│       └──► Script creates and bakes automatically                │
│                                                                   │
│  If errors occur:                                                │
│       │                                                           │
│       ├──► blender-scripting agent diagnoses                     │
│       └──► blender-manual provides API docs                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3: VDB EXPORT                                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Blender Cache Settings:                                         │
│       │                                                           │
│       ├──► Format: OpenVDB                                       │
│       ├──► Compression: BLOSC                                    │
│       ├──► Precision: Half (16-bit)                              │
│       └──► Output: //vdb_cache/fluid_data_####.vdb               │
│                                                                   │
│  Generated Files:                                                │
│       │                                                           │
│       ├──► density grid (required)                               │
│       ├──► temperature grid (optional, for emission color)       │
│       └──► velocity grid (optional, for motion blur)             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: PLASMADX INTEGRATION                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Coordinate Conversion:                                          │
│       │                                                           │
│       └──► Blender Z-up → PlasmaDX Y-up (rotate -90° X)          │
│                                                                   │
│  Material Mapping:                                               │
│       │                                                           │
│       ├──► gaussian-analyzer validates properties                │
│       ├──► materials-council maps to particle type               │
│       └──► Recipe provides recommended values                    │
│                                                                   │
│  NanoVDB Loading:                                                │
│       │                                                           │
│       ├──► Convert OpenVDB → NanoVDB (if needed)                 │
│       ├──► Load into GPU buffer                                  │
│       └──► Render via DXR 1.1 volumetric pipeline                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 VDB Grid Mapping

| Blender Grid | OpenVDB Name | PlasmaDX Usage |
|--------------|--------------|----------------|
| Smoke Density | `density` | Volume opacity/absorption |
| Temperature | `temperature` | Blackbody emission color |
| Flame | `flame` | Fire intensity (if present) |
| Velocity | `velocity` | Motion blur, animation |
| Color | `color` | Direct albedo (rare) |

### 4.3 Material Type Recommendations

| Celestial Body | PlasmaDX Material | Key Settings |
|----------------|-------------------|--------------|
| Emission Nebula | `GAS_CLOUD` | Low opacity, backward scatter, emission |
| Dark Nebula | `DUST` | High opacity, forward scatter, no emission |
| Supernova | `PLASMA` | High emission, outward velocity |
| Stellar Corona | `PLASMA` | Very high emission, low density |
| Protoplanetary Disk | `GAS_CLOUD` | Gradient density, rotation |

---

## 5. Implementation Status

### 5.1 Completion Summary

| Component | Status | Completion |
|-----------|--------|------------|
| blender-manual MCP Server | ✅ Complete | 100% |
| blender-scripting Agent Spec | ✅ Complete | 100% |
| celestial-body-curator Agent Spec | ✅ Complete | 100% |
| Recipe Library Structure | ✅ Complete | 100% |
| Recipe Library README | ✅ Complete | 100% |
| Hydrogen Cloud Recipe | ✅ Complete | 100% |
| Other Recipes | 📋 Planned | 0% |
| Automation Scripts | 📋 Planned | 0% |
| NanoVDB Loader (C++) | ⏳ Not Started | 0% |
| Workflow Plan Document | ✅ Updated | 100% |

### 5.2 What's Working Now

1. **Documentation Search** - All 12 blender-manual tools operational
2. **Agent Prompts** - Both new agents have complete specifications
3. **Example Recipe** - Hydrogen cloud recipe can be followed manually
4. **Recipe Format** - Standardized template for future recipes

### 5.3 What Needs Testing (Phase 0)

Before building more:

1. [ ] Install Blender 5.0
2. [ ] Follow hydrogen_cloud.md recipe manually
3. [ ] Run the Python automation script
4. [ ] Verify VDB files are created correctly
5. [ ] Document pain points in `BLENDER_HANDS_ON_NOTES.md`

### 5.4 Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| Blender 5.0 | Required | Not yet installed by user |
| OpenVDB | Included | Built into Blender |
| NanoVDB | Optional | For C++ runtime loading |
| sentence-transformers | Optional | For semantic search |

---

## 6. Future Enhancements

### 6.1 Short-Term (After Phase 0 Validation)

#### More Recipes

| Recipe | Priority | Complexity | Notes |
|--------|----------|------------|-------|
| Supernova Remnant | High | Medium | Uses pyro-specialist |
| Dark Nebula | High | Low | Absorption-only, simpler |
| Stellar Flare | Medium | Medium | Curved geometry |
| Protoplanetary Disk | Medium | High | Rotation dynamics |

#### Automation Scripts

| Script | Purpose |
|--------|---------|
| `batch_export.py` | Export multiple simulations |
| `resolution_ladder.py` | Create LOD versions |
| `material_presets.py` | Standard celestial materials |

### 6.2 Medium-Term

#### Blender Add-on

Convert scripts into proper Blender add-on with UI:

```
PlasmaDX Export Panel
├── Celestial Body Type [dropdown]
├── Quality Preset [Low/Medium/High]
├── Frame Range [start/end]
├── Export Path [browser]
└── [Bake & Export] button
```

Benefits:

- No scripting knowledge required
- Consistent settings
- Validation before bake
- Progress feedback

#### Hot-Reload in PlasmaDX

Watch VDB directory and reload on change:

```cpp
// Pseudo-code
FileWatcher watcher("vdb_cache/");
watcher.OnChange([](const Path& file) {
    if (file.extension() == ".vdb") {
        ReloadVolume(file);
    }
});
```

Benefits:

- Rapid iteration
- See changes immediately
- No restart required

### 6.3 Long-Term

#### Geometry Nodes Workflow

For procedural (non-simulated) volumes:

- Noise-based nebula shapes
- Parametric dust lanes
- Fractal star-forming regions

Benefits:

- Infinite variation from parameters
- No bake time
- Smaller file sizes

#### ONNX Volume Generation

Train ML model to generate VDB from text prompts:

```
Input: "wispy blue nebula with bright core"
Output: density + temperature grids
```

Very long-term, but interesting research direction.

#### Multi-Volume Compositing

Layer multiple VDB files in PlasmaDX:

```
Scene
├── Background Nebula (low-res, large)
├── Foreground Cloud (high-res, small)
└── Hero Explosion (animated)
```

Requires:

- Multi-volume renderer
- Blending/compositing logic
- Memory management

### 6.4 Tooling Improvements

#### Recipe Validator

Automated testing of recipes:

```python
def validate_recipe(recipe_path):
    # Parse recipe markdown
    # Extract Python script
    # Run in headless Blender
    # Check VDB output exists
    # Verify grid contents
    return ValidationReport
```

#### VDB Inspector Tool

Analyze VDB files before loading:

```
vdb_inspect fluid_data_0050.vdb

Grid: density
  Type: float
  Size: 128x128x128
  Min: 0.0, Max: 0.89
  Memory: 8.2 MB

Grid: temperature
  Type: float
  Size: 128x128x128
  Min: 0.0, Max: 2.1
  Memory: 8.2 MB
```

#### Performance Estimator

Predict FPS impact before loading:

```python
def estimate_fps(vdb_path, current_fps=120):
    grids = openvdb.read(vdb_path)
    voxels = grids['density'].activeVoxelCount()

    # Based on empirical measurements
    fps_cost = voxels * 0.00001  # ~10 FPS per million voxels

    return current_fps - fps_cost
```

---

## 7. Appendices

### 7.1 File Locations

| File | Purpose |
|------|---------|
| `docs/BLENDER_VDB_WORKFLOW_PLAN.md` | Original planning document |
| `docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md` | This specification |
| `docs/blender_recipes/README.md` | Recipe library index |
| `docs/blender_recipes/emission_nebulae/hydrogen_cloud.md` | Example recipe |
| `agents/blender-manual/blender_server.py` | MCP server |
| `agents/blender-scripting/AGENT_PROMPT.md` | Scripting agent spec |
| `agents/celestial-body-curator/AGENT_PROMPT.md` | Curator agent spec |

### 7.2 Quick Reference: VDB Export Settings

```python
# Minimum viable VDB export configuration
settings = domain.modifiers['Fluid'].domain_settings
settings.cache_data_format = 'OPENVDB'
settings.openvdb_cache_compress_type = 'BLOSC'
settings.cache_precision = 'HALF'
settings.cache_directory = '//vdb_cache/'
```

### 7.3 Quick Reference: Property Mapping

| Blender → PlasmaDX | Formula |
|--------------------|---------|
| Density → Opacity | `opacity = density * 0.4` |
| Anisotropy → Phase G | `phase_g = anisotropy` (direct) |
| Emission Strength → Emission | `emission = strength * 0.25` |
| Temperature → Temperature | `temp_k = temp * 10000` (if normalized) |

### 7.4 Glossary

| Term | Definition |
|------|------------|
| **bpy** | Blender Python API module |
| **Domain** | Volume where simulation occurs |
| **Flow** | Object that emits/absorbs fluid |
| **Mantaflow** | Blender's fluid simulation engine |
| **MCP** | Model Context Protocol (agent communication) |
| **NanoVDB** | GPU-optimized VDB format |
| **OpenVDB** | Industry-standard volumetric format |
| **Phase Function G** | Scattering direction (-1=back, +1=forward) |
| **VDB** | Volumetric Data Buffer format |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-07 | Initial specification |

---

*Specification maintained by: Claude Code Agent Ecosystem*
*Last Updated: 2025-12-07*
