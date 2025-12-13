# Blender Session Prompt - Start Here

**Copy this entire prompt into a new Claude session (Blender worktree) to begin working immediately.**

---

## The Prompt

```
I'm working on creating volumetric celestial assets (nebulae, gas clouds, explosions) in Blender 5.0 for real-time rendering in PlasmaDX-Clean, a DirectX 12 raytracing engine.

## What We're Doing

We use **Blender's Python scripting (bpy)** to automate the creation of volumetric simulations, then export them as **OpenVDB files** which get converted to **NanoVDB** for GPU-optimized rendering in PlasmaDX.

**The Pipeline:**
```
Blender 5.0 (Mantaflow simulation)
    ↓ Python script creates scene + bakes simulation
    ↓ Export as OpenVDB (.vdb files)
    ↓
Convert to NanoVDB (.nvdb)
    ↓ python scripts/convert_vdb_to_nvdb.py input.vdb
    ↓
PlasmaDX-Clean (DXR 1.1 ray marching)
    ↓ NanoVDBSystem loads and renders in real-time
    ↓
Stunning volumetric celestial bodies @ 60+ FPS
```

## Why Blender Scripting?

Ben (the user) knows programming but not Blender's UI. We write Python scripts that:
1. Create simulation domains and emitters
2. Configure fluid physics (Mantaflow)
3. Set up materials (Principled Volume shader)
4. Configure VDB export settings
5. Automate the entire workflow

This means: **one script = one-click nebula creation**.

---

## CRITICAL: Blender 5.0 API Warning

**Claude's training data is 10+ months behind Blender 5.0!**

Before writing ANY bpy code:
1. **ALWAYS use `blender-manual` MCP tools** to verify the current API
2. **Known breaking changes:**
   - No BLOSC compression (use ZIP or NONE)
   - Some bpy.ops calls have changed
3. **Never assume** - verify first!

**Example - the RIGHT way:**
```python
# Step 1: Verify API with MCP
mcp__blender-manual__search_bpy_types("FluidDomainSettings")

# Step 2: Check specific property
mcp__blender-manual__search_python_api("cache_data_format")

# Step 3: Write code based on verified API
settings.cache_data_format = 'OPENVDB'
settings.openvdb_cache_compress_type = 'ZIP'  # NOT 'BLOSC'!
```

---

## Key Technical Concepts

### Mantaflow (Blender's Fluid Simulator)
- **Domain**: The bounding box where simulation happens
- **Flow**: Objects that emit/absorb smoke/fire
- **Resolution**: Voxel grid size (64³ = fast preview, 128³ = production)
- **Cache**: Where VDB files are saved

### Principled Volume Shader
Maps to PlasmaDX material properties:

| Blender Setting | PlasmaDX Property | Conversion |
|-----------------|-------------------|------------|
| Color | albedo | Direct |
| Density | densityScale | × 0.4 |
| Anisotropy | scatteringCoeff | Direct (-1 to +1) |
| Emission Strength | emissionStrength | × 0.25 |
| Temperature | Temperature | Blackbody curve |

### VDB Export Settings (Blender 5.0)
```python
settings.cache_data_format = 'OPENVDB'
settings.openvdb_cache_compress_type = 'ZIP'  # NOT BLOSC!
settings.cache_precision = 'HALF'  # 16-bit, good balance
settings.cache_directory = '//vdb_cache/'
```

---

## Available MCP Tools

### blender-manual (USE FIRST for API verification)
```python
mcp__blender-manual__search_python_api("bpy.ops.fluid")
mcp__blender-manual__search_bpy_types("FluidDomainSettings")
mcp__blender-manual__search_bpy_operators("fluid", "bake")
mcp__blender-manual__search_vdb_workflow("export openvdb")
mcp__blender-manual__read_page("physics/fluid/type/domain/cache.html")
```

### gaussian-analyzer (Material validation)
```python
mcp__gaussian-analyzer__simulate_material_properties(material_type="GAS_CLOUD")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `docs/blender_recipes/README.md` | Recipe library index |
| `docs/blender_recipes/emission_nebulae/hydrogen_cloud.md` | Complete recipe with Python script |
| `docs/blender_recipes/scripts/hydrogen_cloud_v1.py` | Working automation script |
| `scripts/convert_vdb_to_nvdb.py` | VDB → NanoVDB conversion |
| `scripts/inspect_vdb.py` | Inspect VDB file contents |

---

## Recipe Library Status

| Recipe | Status | Notes |
|--------|--------|-------|
| hydrogen_cloud.md | COMPLETE | Working script, tested |
| dark_nebula.md | PLANNED | Absorption-only, simpler |
| stellar_flare.md | PLANNED | Animated effect |
| supernova_remnant.md | PLANNED | Explosion |

---

## Immediate Priorities

1. **Test hydrogen_cloud_v1.py** in Blender 5.0
2. **Create dark_nebula recipe** (simpler than emission nebula)
3. **Debug VDB loading** in PlasmaDX if issues arise

---

## Quick Start Workflow

### To create a nebula:
1. Open Blender 5.0
2. Go to Scripting workspace (tab at top)
3. Open `docs/blender_recipes/scripts/hydrogen_cloud_v1.py`
4. Run script (Alt+P)
5. Save .blend file (Ctrl+S)
6. Select domain → Physics → Fluid → Cache → Bake All
7. VDB files appear in `vdb_cache/` folder

### To convert for PlasmaDX:
```bash
python scripts/convert_vdb_to_nvdb.py vdb_cache/fluid_data_0050.vdb
```

---

## tmux Session Switching

- **F9** = Switch to Blender session
- **F10** = Switch to NanoVDB session
- **q** = Snap back to prompt after scrolling

---

What would you like to work on?
1. Test existing hydrogen_cloud script
2. Create a new recipe (dark_nebula, stellar_flare, etc.)
3. Debug VDB export/import issues
4. Something else
```

---

## Short Version

```
I'm creating volumetric celestial assets in Blender 5.0 for PlasmaDX-Clean.

Key docs to read:
- docs/blender_recipes/emission_nebulae/hydrogen_cloud.md (complete recipe)
- docs/blender_recipes/scripts/hydrogen_cloud_v1.py (working script)

Pipeline: Blender (Mantaflow) → VDB export → NanoVDB conversion → PlasmaDX rendering

CRITICAL: Use blender-manual MCP before ANY bpy code - Claude's data is 10+ months behind Blender 5.0! No BLOSC compression (use ZIP).

What would you like to work on?
```

---

*This prompt gets you working immediately on Blender asset creation.*
