# Blender 5 → OpenVDB → NanoVDB Pipeline Findings (2025-12-17)

This document captures **high-signal findings** and **likely root causes** for the current issue:

> **Procedural NanoVDB volumes render, but Blender 5-created file-loaded `.nvdb` assets often appear invisible** in PlasmaDX-Clean.

These notes are intended to be used for multi-agent planning (e.g., `compound:engineering`) and for future debugging sessions.

## Scope / Constraints

- **Project**: `PlasmaDX-Clean` only.
- **No changes to other worktrees** (they’re intentionally left untouched for easy revert).
- This document is evidence-based: **shader + C++ code behavior is authoritative**. Blender behavior is referenced via the Blender 5 manual/API (via `blender-manual` MCP).

## TL;DR (Most likely causes)

1. **Grid type mismatch**: `nanovdb_raymarch.hlsl` currently supports only `PNANOVDB_GRID_TYPE_FLOAT` (type `1`).  
   Blender can write OpenVDB caches at **Half (16-bit)** or **Mini** precision, which can lead to NanoVDB grids that are **HALF/FP16/etc** instead of FLOAT → shader returns density 0 everywhere → **invisible volume**.
2. **Wrong grid selected**: `NanoVDBSystem::LoadFromFile()` loads **grid index 0** only. If the file contains multiple grids, **index 0 may not be `density`** (could be temperature/velocity/etc), and/or may not be float.
3. **Scale/transform issues** (secondary but common): Blender units and axis conventions (Z-up) can yield **very small world bounds** (e.g., “~6 units”), requiring scaling/repositioning in-engine. The UI already has scale/center controls, but these don’t fix grid-type/selection problems.

## Evidence & Key Observations

### A) Blender 5 OpenVDB Cache precision options can change the grid type

Blender 5 manual: `physics/fluid/type/domain/cache.html`

- **OpenVDB** cache writes one `.vdb` per frame.
- **Precision** choices include:
  - **Full**: 32-bit floats
  - **Half**: 16-bit floats
  - **Mini**: 8-bit where possible (falls back to 16-bit where not possible)

**Implication**: If Blender outputs Half/Mini precision, the converted `.nvdb` can contain **HALF/FP16** value grids, not FLOAT.

### B) PlasmaDX shader currently hard-rejects non-float grids

File: `shaders/volumetric/nanovdb_raymarch.hlsl`

- Shader reads `gridType = pnanovdb_grid_get_grid_type(...)`
- It currently accepts only:
  - `PNANOVDB_GRID_TYPE_FLOAT` (value `1`)
- For any other type, it returns density 0.

Supporting detail: `shaders/nanovdb/PNanoVDB.h` includes grid types:

- `PNANOVDB_GRID_TYPE_FLOAT = 1`
- `PNANOVDB_GRID_TYPE_HALF = 9`
- `PNANOVDB_GRID_TYPE_FP16 = 15`
- …and many others, plus helper readers (e.g., `pnanovdb_read_half`)

**Implication**: Even if bounds are correct and rays hit the AABB, the shader may still read **no density**.

### C) Loader always reads grid index 0 (may not be density)

File: `src/rendering/NanoVDBSystem.cpp`

`LoadFromFile()` uses:

- `nanovdb::io::readGrid<HostBuffer>(filepath, 0, 1)` → **grid index 0**

The code then tries `handle.grid<float>()`:

- If index 0 is not a float density grid, it falls back to generic bounds extraction and still uploads the buffer.

**Implication**: A multi-grid `.nvdb` can load “successfully” but still render as empty because:

- the wrong grid was selected, and/or
- the selected grid is not supported by the shader.

### D) Blender volume grids and sequence conventions matter

Blender 5 manual:

- `modeling/volumes/properties.html`:
  - Lists grids in a VDB file (name + type)
  - Describes VDB sequence naming (numeric suffixes)
- `modeling/volumes/introduction.html`:
  - Principled Volume expects grids named `density`, `color`, `temperature` by default

**Implication**: In practice, Blender-created assets often contain multiple grids (density/temperature/velocity), and sequences rely on consistent suffix naming.

## Symptoms & Debug Signals

From the internal roadmap/docs (`docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md`):

- **Procedural fog works** (shader path that does not sample PNanoVDB buffer).
- **Debug mode colors**:
  - “Inside AABB but no density” (often **cyan**)
  - “Grid density found” (often **green**)

This pattern is consistent with **grid-type/selection mismatch** rather than pure AABB miss.

## Concrete Hypotheses (Testable)

### H1 — Blender export precision produces HALF/FP16 grids

**If** Blender fluid cache “Volumetric Data → Precision” is **Half** or **Mini**  
**then** the `.vdb` → `.nvdb` conversion can preserve a non-float value type  
**and** the PlasmaDX shader will render density=0 for all samples.

**Fast test**:

- Load a failing `.nvdb` and log its `gridType` + grid name.
- If gridType is `HALF (9)` or `FP16 (15)`, H1 is confirmed.

### H2 — Grid index 0 isn’t the `density` grid

**If** the `.nvdb` contains multiple grids  
**then** `readGrid(..., index=0)` can select a non-density grid.

**Fast test**:

- Enumerate grids / names / types and confirm whether `density` is index 0.
- Alternatively, update loader to choose `density` by name.

### H3 — Axis/scale mismatch prevents the camera from intersecting the volume

This is usually a **secondary** issue (it produces “miss AABB” rather than “inside bounds but no density”), but it still matters:

- Blender is **Z-up**; many engines are **Y-up**.
- Blender simulation domains can have **small world bounds** (single-digit units) compared to PlasmaDX scene scale (hundreds/thousands).
- PlasmaDX already exposes runtime transforms:
  - view `gridWorldMin/Max`
  - **ScaleGridBounds(scale)**
  - **SetGridCenter(center)**

**Fast test**:

- Enable NanoVDB debug mode and confirm whether you are getting:
  - consistent AABB hits (not red-tinted miss), and
  - expected scaling after applying 50× / 100× / 200×.

## Quick Debug Checklist (Recommended order)

### 1) Confirm which grid you loaded (name + value type)

- In `NanoVDBSystem::LoadFromFile`, log:
  - **grid name** (if available)
  - **grid value type** (FLOAT vs HALF/FP16/etc)
  - bounds + voxel size + active voxel count

Why: this immediately confirms or rules out H1/H2.

### 2) Verify density is non-zero in the shader (independent of lighting)

Use the existing debug visualization:

- If you see **inside bounds but “no density”** colors, you likely have **grid type mismatch** or **wrong grid selected**.
- If you see “density found” colors but the volume is still visually weak, tune:
  - density scale
  - absorption/scattering coefficients
  - step size

### 3) Validate the Blender side quickly (no engine needed)

In Blender:

- Import the VDB as a volume object and use **Volume Properties → Grids** to view:
  - grid names (`density`, `temperature`, `velocity`, …)
  - their data types

Manual reference: `modeling/volumes/properties.html`

### 4) Validate conversion choices

The repo’s conversion script already encodes a key assumption:

- `scripts/convert_vdb_to_nvdb.py` defaults to selecting:
  1) `--grid <name>` if provided
  2) grid named **`density`**
  3) first float-ish grid

It also warns that the **shader only supports FLOAT grids** (as of this finding).

## Recommended Fix Strategy (Engineering Options)

### Option A — Add shader support for HALF/FP16 grids (most robust)

Implement in `shaders/volumetric/nanovdb_raymarch.hlsl`:

- Accept `PNANOVDB_GRID_TYPE_HALF (9)` and/or `PNANOVDB_GRID_TYPE_FP16 (15)` in addition to FLOAT.
- Read values via PNanoVDB helper(s) and convert to float for accumulation.

Pros:

- Works with Blender’s Half/Mini precision caches.
- Better memory/perf potential for large assets.

Cons:

- Requires careful sampling correctness and potentially different normalization.

### Option B — Enforce FLOAT grids end-to-end (simplest)

Standardize the pipeline:

- Blender: set OpenVDB cache precision to **Full (32-bit)**.
- Conversion: ensure output `.nvdb` is float grid (convert if needed).
- Engine: hard-error with a friendly message if a non-float grid is loaded.

Pros:

- Minimal shader complexity.
- Easier initial correctness.

Cons:

- Larger assets; may cost memory/bandwidth.

### Option C — Fix grid selection (required either way)

Regardless of A/B:

- Stop hardcoding **grid index 0** in `NanoVDBSystem::LoadFromFile`.
- Add selection by:
  - grid name (prefer `density`)
  - or an ImGui dropdown listing available grids (name + type)

## Documentation / Tooling Gaps Noted

### “inspect_vdb.py” is Blender-side (not a CLI inspector)

File: `scripts/inspect_vdb.py`

- This script uses `bpy` and `bpy.ops.object.volume_import`.
- Some docs imply a CLI inspector exists. In PlasmaDX-Clean, it does not.

Recommendation:

- Rename/document it explicitly as a Blender helper, or add a separate CLI tool if desired.

### Some referenced docs are missing or moved

Multiple docs reference files like:

- `docs/BLENDER_5_GUARDRAILS.md`
- `docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md`

In this worktree, the workflow spec lives at:

- `docs/blender_recipes/explosions/BLENDER_PLASMADX_WORKFLOW_SPEC.md`

Recommendation:

- Update references to point at the correct path (or create the missing top-level files).

## Immediate Next Actions (High Leverage)

1. **Log + UI**: Surface grid name + grid type + selected grid index in ImGui.
2. **Selection**: Load `density` by name, not index 0.
3. **Compatibility**: Either support HALF/FP16 grids in shader **or** enforce FLOAT conversion + Blender “Full precision”.
4. **Multi-volume world-space system** (separate feature): allow multiple NanoVDB instances with per-instance transforms and animation controls.

## References (Blender 5 manual paths)

- Cache (OpenVDB compression + precision): `physics/fluid/type/domain/cache.html`
- Volume properties (grid list + sequence naming): `modeling/volumes/properties.html`
- Volumes intro (grid naming conventions for Principled Volume): `modeling/volumes/introduction.html`
