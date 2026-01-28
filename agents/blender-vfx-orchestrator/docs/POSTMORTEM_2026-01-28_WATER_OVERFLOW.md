# Postmortem: Water Overflow Build Session (2026-01-28)

**Status:** System failures documented, root cause analysis pending
**Date:** 2026-01-28
**Session Duration:** ~4 hours

---

## Executive Summary

A session focused on fixing three pipeline issues (representative frame rendering, cache type, API spec validation) resulted in **dramatic image quality improvements** but **complete fluid simulation failure**. The cure became worse than the disease as successive patches introduced new path resolution bugs, leaving cache folders scattered across the filesystem and empty bake data.

---

## What Worked: Image Quality Breakthrough

The Code Writer Agent produced scripts with **exceptional visual quality**:

- Cinematic moody lighting with dramatic shadows
- Physically accurate glass refraction and caustics
- Beautiful depth-of-field bokeh
- Proper material setups (water IOR 1.333, glass transmission, metallic pipes)
- Professional-grade scene composition

**Evidence:** Screenshots show renders comparable to professional VFX work, despite zero fluid simulation data.

---

## What Failed: Fluid Simulation

### Symptom
- Bake operations report "complete" in seconds
- Cache files created but essentially empty (23 bytes to 1.4KB each)
- Green FLIP debug particles visible instead of fluid mesh
- No actual simulation data in any cache location

### Cache Folder Proliferation

Cache directories appeared in at least 5 locations:

| Location | Size | Status |
|----------|------|--------|
| `agents/blender-vfx-orchestrator/cache_mantaflow_water/` | Empty | Created by script |
| `assets/blender_scripts/generated/cache_mantaflow_water/` | 788KB (empty VDBs) | Created by API fixer patch |
| `build/vdb_output/water_overflow_v1/cache/` | 788KB (empty VDBs) | Unknown origin |
| `PlasmaDXR/cache_mantaflow_water/` | Empty | Created by script |
| `C:\Users\...\AppData\Local\Temp\...` | Unknown | Windows temp locations |

**Root Cause:** Multiple competing path resolution strategies:
1. Original script: `//cache_mantaflow_water` (Blender-relative, fails headless)
2. API Fixer patch: `os.path.dirname(os.path.abspath(__file__))` (script directory)
3. Script also calls: `ensure_dir(os.path.join(project_root, 'cache_mantaflow_water'))`

The `cache_directory` property on the domain gets set to one path while `ensure_dir()` creates a different path.

---

## Timeline of Changes

### Initial State (Before Session)
- Pipeline producing 500+ line scripts
- Scripts rendered all 60 frames for quality evaluation
- Cache type defaulted to REPLAY (insufficient for full bake)
- API Spec Agent rejecting valid page-level doc URLs

### Plan Implementation (3 Fixes)

**Fix 1: Representative Frame Rendering**
```python
# Added regex to BLENDER_50_FIXES
# Pattern: range(FRAME_START, FRAME_END + 1)
# Replace: [min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]
```
- **Intended:** Render 3 frames instead of 48-60
- **Actual:** Initial regex didn't match `scene.frame_start/frame_end` variable names
- **Patch:** Added additional regex patterns for common variations
- **Result:** Eventually worked, renders now output 3 frames

**Fix 2: Cache Type REPLAY → ALL**
```python
# Added regex to BLENDER_50_FIXES
# Pattern: cache_type = 'REPLAY'
# Replace: cache_type = 'ALL'
```
- **Intended:** Ensure full bake data stored
- **Actual:** Scripts use `cache_type = 'MODULAR'`, not REPLAY
- **Result:** Fix never triggers, no effect

**Fix 3: API Spec doc_ref Validation**
```python
# Added fallback: accept if object_type (class name) in doc_ref URL
```
- **Intended:** Accept page-level URLs like `FluidDomainSettings.html`
- **Actual:** Working as intended
- **Result:** API Spec Agent validation passes

### Cascade of Additional Fixes

| Fix | Trigger | Side Effect |
|-----|---------|-------------|
| BSDF Specular → Specular IOR Level | KeyError in render | None |
| BSDF Clearcoat → Coat Weight | KeyError in render | None |
| use_auto_smooth removal | AttributeError | None |
| Blender-relative path → absolute | Empty cache files | **Created competing paths** |
| Bake-before-render injection | Missing bake call | **Never triggered (bake exists)** |
| Animation→stills injection | Headless render issue | **Regex pattern mismatch** |

### Unintended Consequences

1. **Idempotency Failure:** The `use_nodes` deprecation fix appended comments repeatedly:
   ```python
   mat.use_nodes = True  # Deprecated...  # Deprecated...  # Deprecated...
   ```

2. **Path Confusion:** Three different path strategies in one script:
   - `bpy.path.abspath('//')` → Empty in headless/unsaved
   - `os.path.dirname(os.path.abspath(__file__))` → Script directory
   - Hardcoded `//cache_mantaflow_water` → Blender-relative

3. **Bake Injection Not Triggered:** The `_inject_bake_before_render()` function checks for existing bake calls and skips injection. Scripts already have `bpy.ops.fluid.bake_data()`, so no injection occurs—but those existing calls silently fail due to path issues.

---

## Root Cause Analysis

### Primary Issue: Path Resolution in Headless Mode

Blender's `//` relative path notation requires a saved .blend file to resolve against. In headless mode:

```python
bpy.path.abspath('//')  # Returns '' or current working directory
cache_dir = '//cache'   # Blender can't resolve, writes to wrong location
```

The simulation "runs" but writes to an unresolvable path, producing empty placeholder files.

### Secondary Issue: Patch Layering

Each fix addressed a symptom without understanding the system:

```
Original Bug → Patch A → New Bug → Patch B → Conflicts with Patch A → Patch C → ...
```

The API fixer now has **30+ regex patterns** plus **5 injection functions**, making behavior increasingly unpredictable.

### Tertiary Issue: Training Data vs Reality

The Code Writer Agent uses Blender patterns from training data (pre-5.0) that:
- Use deprecated `use_nodes = True` syntax
- Use `//` relative paths (common in tutorials)
- Reference removed shader inputs (Specular, Clearcoat)
- Assume GUI context for bake operations

The API Fixer catches some of these, but not all, and the patches sometimes conflict with each other.

---

## Artifacts Produced

### Working
- Scene geometry (table, wall, pipe, glass)
- Materials (water, glass, metal, wood)
- Lighting (3-point cinematic setup)
- Camera with depth-of-field
- Render settings (Cycles GPU, 256 samples)

### Not Working
- Fluid domain cache path resolution
- Actual fluid simulation data
- Bake operations (complete but empty)
- Headless execution path

---

## Recommendations

### Immediate (Get Fluid Working)

1. **Hardcode absolute Windows path for GUI testing:**
   ```python
   dset.cache_directory = r"C:\temp\mantaflow_cache"
   ```

2. **Test bake in GUI with known-good path** to confirm simulation setup is correct

3. **If GUI bake works:** Problem is purely path resolution
4. **If GUI bake fails:** Problem is simulation setup (domain size, emitter position, etc.)

### Short-Term (Stabilize Pipeline)

1. **Audit all path operations** in generated scripts
2. **Remove competing path strategies** - pick ONE approach
3. **Test API fixer patterns in isolation** before combining
4. **Add integration tests** for bake → render → evaluate flow

### Medium-Term (Architecture)

1. **Separate concerns:**
   - Scene building (geometry, materials, lights)
   - Simulation setup (domain, flows, effectors)
   - Baking (cache paths, frame ranges)
   - Rendering (output paths, frame selection)

2. **Template-based generation** for known-good simulation setups instead of LLM generation for everything

3. **Path abstraction layer** that handles headless vs GUI automatically

### Long-Term (Learning System)

1. **Document Blender 5.0 breaking changes** in knowledge base
2. **Add negative examples** to prevent training-data patterns
3. **Implement automated regression tests** for each fix
4. **Consider Blender Python linting** before execution

---

## Files Modified This Session

| File | Changes | Risk |
|------|---------|------|
| `tools/blender_api_fixer.py` | +8 regex patterns, +2 injection functions | High - complexity explosion |
| `models/api_spec.py` | +1 validation fallback | Low |
| `specialized_agents/code_writer_agent.py` | Instructions rewrite | Medium |
| `data/code_patterns/patterns_index.json` | Removed poison pattern | Low |
| `test_water_overflow.py` | Prompt restructuring | Low |

---

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| Script line count | 134 (broken) → 526 | 420-520 |
| Frames rendered | All 48-60 | 3 representative |
| Bake data produced | Unknown | 0 (empty files) |
| Image quality | Good | Excellent |
| System stability | Working | Broken |
| API fixer patterns | ~22 | ~30 |

---

## Conclusion

This session demonstrates the danger of **reactive patching** without **systemic understanding**. The image quality improvements prove the Code Writer can produce excellent work, but the pipeline infrastructure for headless execution has accumulated too much technical debt.

**Recommended next step:** Before adding more patches, create a minimal working example that:
1. Builds scene in GUI
2. Bakes with hardcoded absolute path
3. Confirms fluid simulation works
4. Then systematically identify which path causes the break

The dramatic quality improvement should not be discarded—the goal is to preserve that while fixing the execution infrastructure.

---

## Appendix: Cache File Evidence

```
# Empty VDB files (should be megabytes, not bytes)
-rw-r--r-- 1 maz3ppa maz3ppa 1395 Jan 28 21:48 fluid_data_0001.vdb
-rw-r--r-- 1 maz3ppa maz3ppa 1395 Jan 28 21:48 fluid_data_0002.vdb
...

# Empty mesh files
-rw-r--r-- 1 maz3ppa maz3ppa   23 Jan 28 21:30 fluid_mesh_0001.bobj.gz
-rw-r--r-- 1 maz3ppa maz3ppa   23 Jan 28 21:30 fluid_mesh_0002.bobj.gz
...
```

A working 48-frame liquid simulation at resolution 96 should produce **50-200MB** of cache data, not 788KB.
