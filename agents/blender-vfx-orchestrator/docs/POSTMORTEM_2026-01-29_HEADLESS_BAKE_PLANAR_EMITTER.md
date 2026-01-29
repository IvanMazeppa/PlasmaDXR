# Postmortem: Headless Bake Failure + Render-Loop Regression (2026-01-29)

**Status:** Root causes confirmed, fixes validated headlessly
**Scope:** Mantaflow liquid (water overflow) pipeline
**Context:** 0.31.2 bug-hunting (backed up), baseline 0.31.1

---

## Executive Summary

Two issues appeared together:

1) **Render regression** — scripts began rendering every frame instead of 1–3 representative frames.
2) **Simulation failure** — bake reported “complete” but caches were empty/tiny; only green FLIP debug particles appeared.

Both were reproduced and resolved in headless runs. The simulation failure was **not** a pathing issue once output roots were unified; it was a **planar inflow emitter** missing `use_plane_init=True`. Enabling that property immediately produced large cache data (tens of MB) and real fluid data. The render regression was due to **new loop patterns** not covered by the API fixer regexes.

---

## Symptoms

- Cache folders existed but files were tiny (e.g., data VDBs ~1.3 KB, mesh bobj ~23 B).
- Bakes completed quickly with no visible fluid mesh; only FLIP debug spheres.
- Some scripts rendered **all frames** (slow, huge overhead).
- Path proliferation (multiple cache roots) made debugging confusing.

---

## Root Causes

### A) Planar liquid emitter not initialized
- Liquid inflow emitters were created as planes.
- Blender 5.0 expects **`use_plane_init=True`** for planar liquid emitters.
- Without it, bakes can “complete” but emit no fluid volume.

### B) Render frame limiter no longer matched new loop styles
- The API fixer rewrote only a few loop patterns.
- New generator variants (e.g., `Config.FRAME_*`, `list(range(...))`, `return list(range(...))`) bypassed the regex, causing full‑frame renders.

### C) Path contract drift (made diagnosis harder)
- Scripts mixed `bpy.path.abspath('//')`, script dir paths, and output dirs.
- This produced caches in multiple places and muddied the signal, even after baking.
- Once env‑driven output/cache roots were enforced, the simulation failure persisted—indicating root cause A.

---

## What Changed During the Fix

### 1) Path contract stabilization
- `BLENDER_OUTPUT_DIR` and `BLENDER_CACHE_DIR` now provide a single root for headless runs.
- `bpy.path.abspath('//')` is now overridden (env‑aware) when unset.

### 2) Render loop normalization
- Added regex support for:
  - `Config.FRAME_START/END` loops
  - `list(range(...))` variants
  - `return list(range(...))`
- Result: 3 representative frames again.

### 3) Planar inflow fix
- Auto‑enable `use_plane_init=True` for **LIQUID** flows when a plane emitter exists.
- This change alone increased cache size from KB to ~23MB and produced real data.

---

## Validation Evidence (Headless)

**Before plane init:**
- `cache/data/*.vdb` ~1.3 KB
- `cache/mesh/*.bobj.gz` ~23 B

**After plane init:**
- `cache/data/*.vdb` grows from ~30 KB to 175+ KB over frames
- `cache` size ~23 MB for 48‑frame run
- 3 representative renders produced

---

## Files/Components Involved

- `agents/blender-vfx-orchestrator/tools/blender_api_fixer.py`
  - Render loop regex expansion
  - Env‑aware cache/output root handling
  - New planar liquid emitter fix (`use_plane_init=True`)
- `agents/blender-vfx-orchestrator/tools/blender_executor_tools.py`
  - Export `BLENDER_OUTPUT_DIR` + `BLENDER_CACHE_DIR`

---

## Lessons / Preventive Actions

1) **Emitter type matters** — planar emitters must use `use_plane_init` for liquid.
2) **Regex fixes drift** — render loop normalization should be a deterministic function, not scattered regex.
3) **Single output root** — cache/render roots must be uniform in headless mode; otherwise debugging becomes ambiguous.

---

## Next Steps (Recommended)

- Bake `use_plane_init` into **Script Writer instructions** and/or a guardrail (not just fixer).
- Add a lightweight **post‑bake cache sanity check** (size threshold) to flag empty sims.
- Consider a **single rendering helper** to eliminate loop‑pattern drift.

---

## Artifacts

- Headless run logs and outputs under:
  - `build/blender_cli_logs/20260129_014336_water_overflow_v1_plane_fix/`
  - `build/vfx_output/manual_water_overflow_plane_fix/`

