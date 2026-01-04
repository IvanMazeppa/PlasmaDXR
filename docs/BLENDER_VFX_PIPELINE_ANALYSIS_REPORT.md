# Blender VFX Multi-Agent Pipeline Analysis Report

**Date:** 2026-01-04
**Analyst:** Claude Code
**Test Case:** Glass pane explosion scene (aerial_burst technique, 30 frames)

---

## Executive Summary

End-to-end testing of the Blender VFX orchestrator pipeline revealed **one critical bug** (now fixed) and **several quality/usability issues**. The pipeline successfully generates and executes Blender scripts, but visual quality of generated VFX is consistently poor, requiring manual iteration.

**Key Findings:**
1. **CRITICAL BUG FIXED:** 250-frame bake guardrail issue - Mantaflow ignored scene frame range
2. **Visual Quality:** Generated explosions score 53-54/100 (FAIL threshold is 60)
3. **MCP Server Reload:** Changes to server.py require MCP restart to take effect
4. **Volumetric-Only:** Pipeline cannot generate mesh-based effects (soft body, cloth, rigid body)

---

## Issue #1: 250-Frame Bake Guardrail Bug (CRITICAL - FIXED)

### Symptom
When requesting a 30-frame animation, Mantaflow baked 250 frames (`fluid_data_0001.vdb` through `fluid_data_0250.vdb`), causing:
- **5x longer bake time** (127 seconds vs 27 seconds)
- **8x more disk space** (250 VDB files vs 30)
- **User frustration** from excessive wait times

### Root Cause
The script-generator templates set `scene.frame_end` but did NOT set the Mantaflow cache frame range:
- `domain_settings.cache_frame_start`
- `domain_settings.cache_frame_end`
- `domain_settings.cache_frame_offset`

Mantaflow uses its own cache frame range (default 1-250) **independently** of the scene frame range.

### Fix Applied
Added cache frame settings to both PYRO_DOMAIN_TEMPLATE and LIQUID_DOMAIN_TEMPLATE in `agents/script-generator/server.py`:

```python
# Cache settings
settings.cache_type = 'ALL'
settings.cache_directory = Config.OUTPUT_DIR
settings.cache_data_format = 'OPENVDB'
# CRITICAL: Match cache frame range to scene frame range (fixes 250-frame default bug)
settings.cache_frame_start = Config.FRAME_START
settings.cache_frame_end = Config.FRAME_END
settings.cache_frame_offset = 0
```

### Verification
- Before fix: 250 VDB files, 127 seconds
- After fix: 30 VDB files, 27 seconds (5x improvement)

### Status
**FIXED** in `agents/script-generator/server.py` (lines 446-449 for pyro, lines 740-743 for liquid)

---

## Issue #2: Poor Visual Quality of Generated Effects

### Symptom
Generated explosion renders consistently score 53-54/100 on VFX quality evaluation, failing the 60-point pass threshold.

**Evaluation Results (glass_explosion_test_v2):**
```
composite_score: 53/100 - FAIL
dimension_scores:
  brightness: 5/25 (TOO DARK: mean brightness 23 << target 80)
  dynamic_range: 15/25
  color: 18/25 (Needs more warm color: warm_ratio 1.15)
  coverage: 10/15 (Effect is small: 10.0% coverage)
  structure: 5/10 (NO STRUCTURE: edge density 0.010 < 0.05)
```

### Root Cause Analysis
1. **Material Settings Too Conservative:** Principled Volume shader settings produce faint output
   - `Density: 10.0` - may need 20-50 for visible smoke
   - `Blackbody Intensity: 8.0` - may need 15-25 for visible flames
   - `Temperature: 2500K` - correct for orange flames

2. **Simulation Parameters:** Aerial burst technique produces spherical expansion but lacks dramatic fire visuals
   - `flame_smoke: 0.32` - very low, produces minimal smoke
   - `burning_rate: 2.95` - high but short-lived

3. **Camera Position:** Camera may be too far from the effect (6 units away for 6-unit domain)

4. **Transparent Background:** `film_transparent = True` may affect brightness perception

### Recommendations
1. **Increase Material Density/Emission:**
   ```python
   volume.inputs['Density'].default_value = 25.0  # Was 10.0
   volume.inputs['Blackbody Intensity'].default_value = 15.0  # Was 8.0
   ```

2. **Increase Flame Smoke Ratio:**
   ```python
   settings.flame_smoke = 1.5  # Was 0.32
   ```

3. **Add Background for Renders:**
   ```python
   scene.render.film_transparent = False  # Show environment
   ```

4. **Consider Closer Camera:**
   ```python
   bpy.ops.object.camera_add(location=(4, -4, 3))  # Was (6, -6, 4)
   ```

### Status
**OPEN** - Requires tuning of default parameters in script-generator

---

## Issue #3: MCP Server Does Not Hot-Reload Code Changes

### Symptom
After editing `agents/script-generator/server.py`, newly generated scripts did not include the fix. The running MCP server continued using cached code.

### Root Cause
MCP servers are Python processes that load code at startup. Changes to `server.py` require process restart.

### Workaround
1. Restart Claude Code session, OR
2. Manually edit generated scripts, OR
3. Use MCP server restart command (if available)

### Recommendation
Add documentation to SKILL.md noting that code changes require MCP server restart.

### Status
**DOCUMENTED** - Expected behavior, no code fix needed

---

## Issue #4: Blender 6.0 Deprecation Warning

### Symptom
```
DeprecationWarning: 'Material.use_nodes' is expected to be removed in Blender 6.0
```

### Root Cause
The script uses `mat.use_nodes = True` which is deprecated in Blender 5.0+ and will be removed in 6.0.

### Recommendation
Update script-generator templates to use Blender 5.0+ material node API without `use_nodes`:
```python
# Blender 5.0+ automatically enables nodes when using mat.node_tree
mat = bpy.data.materials.new(name="FireSmokeMaterial")
# mat.use_nodes = True  # REMOVE - deprecated in Blender 5.0
nodes = mat.node_tree.nodes  # Works without use_nodes in Blender 5.0+
```

### Status
**OPEN** - Minor, non-blocking

---

## Issue #5: Volumetric-Only Pipeline (As Documented)

### Context
The Gemini feedback analysis (GEMINI_FEEDBACK_ANALYSIS_AND_PLAN_AMENDMENTS.md) correctly identified that the pipeline only supports volumetric effects (Fluid/MantaFlow -> OpenVDB workflows).

### Impact
Requests for:
- Soft body physics (jelly, bounce)
- Cloth simulation (fabric, flags)
- Rigid body destruction
- Glass shattering (mesh-based)

...cannot be fulfilled by the current pipeline.

### Status
**DOCUMENTED** - Phase 2.5 in MULTI_AGENT_IMPROVEMENT_PLAN_V2.md addresses this with effect type registry and mesh simulation support.

---

## Issue #6: Glass Shattering Not Implemented

### Symptom
The test case requested "explosive detonates next to a clear glass pane, shattering it" but the generated script only creates the explosion - no glass pane or shattering effect.

### Root Cause
1. The script-generator only creates pyro/fluid simulations
2. Glass shattering requires:
   - Glass mesh geometry
   - Cell Fracture addon or rigid body simulation
   - Collision with explosion force field

### Recommendation
For glass shattering support, add to Phase 2.5:
1. Create glass pane mesh geometry
2. Apply Cell Fracture addon
3. Add rigid body physics to fragments
4. Create explosion force field effector
5. Bake rigid body simulation

### Status
**OPEN** - Requires Phase 2.5 implementation

---

## Summary of Changes Made

| File | Change | Status |
|------|--------|--------|
| `agents/script-generator/server.py` (line 446-449) | Added cache_frame_start/end/offset to pyro template | DONE |
| `agents/script-generator/server.py` (line 740-743) | Added cache_frame_start/end/offset to liquid template | DONE |

---

## Recommendations for Next Sprint

### Priority 1: Frame Count Guardrail (COMPLETE)
- [x] Fix 250-frame bake bug in script-generator templates

### Priority 2: Visual Quality Improvement
- [ ] Increase default material density/emission values
- [ ] Add "quality presets" (draft/preview/final) to script-generator
- [ ] Tune technique parameters for more dramatic visuals

### Priority 3: Blender API Compatibility
- [ ] Remove deprecated `mat.use_nodes = True`
- [ ] Test with Blender 5.0 API changes

### Priority 4: Effect Type Extensibility (Phase 2.5)
- [ ] Implement effect type registry
- [ ] Add mesh-based simulation support
- [ ] Add live render execution pattern for physics simulations

---

## Test Artifacts

| Artifact | Path |
|----------|------|
| Generated script (v1) | `assets/blender_scripts/generated/glass_explosion_test.py` |
| Generated script (v2) | `assets/blender_scripts/generated/glass_explosion_test_v2.py` |
| VDB output (v1 - 250 frames) | `build/vdb_output/glass_explosion_test/` |
| VDB output (v2 - 30 frames) | `build/vdb_output/glass_explosion_test_v2/` |
| Render preview | `build/vdb_output/glass_explosion_test_v2/render_0015.png` |
| Blend file | `build/vdb_output/glass_explosion_test_v2/glass_explosion_test_v2.blend` |

---

**Report compiled by Claude Code during end-to-end pipeline analysis.**
