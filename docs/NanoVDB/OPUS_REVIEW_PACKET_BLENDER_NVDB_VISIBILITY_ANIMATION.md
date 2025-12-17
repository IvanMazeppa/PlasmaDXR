# Opus Review Packet — Blender→NanoVDB visibility + animation failures (PlasmaDX-NanoVDB)

**Goal:** Provide a *code-grounded*, testable set of hypotheses for why **procedural fog renders**, but **Blender-derived `.nvdb` assets** (including **animations**) often render as “nothing” or don’t animate.

This document is intentionally **documentation-only** (no code changes). It’s meant to be shared with **Claude Opus 4.5** for review before we implement fixes.

---

## Scope / what I reviewed

You asked to review the NanoVDB pipeline docs and the implementation in the worktree/repo:

- Repo: `PlasmaDX-NanoVDB`

### Docs reviewed (as requested)

- `PlasmaDX-NanoVDB/docs/BLENDER_SESSION_PROMPT.md`
- `PlasmaDX-NanoVDB/docs/CLAUDE_SESSION_PROMPT.md`
- `PlasmaDX-NanoVDB/docs/NANOVDB_SESSION_CONTEXT.md`
- `PlasmaDX-NanoVDB/docs/NANOVDB_SYSTEM_OVERVIEW.md`
- `PlasmaDX-NanoVDB/docs/NANOVDB_UNIFIED_ROADMAP_V1.md`
- `PlasmaDX-NanoVDB/docs/README.md`
- `PlasmaDX-NanoVDB/docs/TROUBLESHOOTING_ASSET_VISIBILITY_AND_ANIMATION.md`

### Code reviewed (high-signal)

- Loader + constants: `PlasmaDX-NanoVDB/src/rendering/NanoVDBSystem.cpp` / `.h`
- Runtime integration + animation tick: `PlasmaDX-NanoVDB/src/core/Application.cpp`
- Shader sampling + debug colors: `PlasmaDX-NanoVDB/shaders/volumetric/nanovdb_raymarch.hlsl`
- Grid type enum truth: `PlasmaDX-NanoVDB/shaders/nanovdb/PNanoVDB.h`
- Conversion: `PlasmaDX-NanoVDB/scripts/convert_vdb_to_nvdb.py`

---

## What the current implementation *actually requires* (pipeline invariants)

These are the invariants implied by the **current shader + loader**:

- **Invariant: shader samples only grid type = FLOAT**

- In `nanovdb_raymarch.hlsl`, both nearest and trilinear paths check:
  - `gridType == 1u` (commented as “PNANOVDB_GRID_TYPE_FLOAT = 1”).
  - If not, they return **0 density** → volume becomes invisible.

- **Invariant: `LoadFromFile()` reads grid index 0**

- `NanoVDBSystem::LoadFromFile()` calls:
  - `nanovdb::io::readGrid(filepath, 0, 1)` → **grid index 0**.
- It *attempts* `handle.grid<float>()` for float; otherwise logs non-float and uploads anyway.
- But shader still returns 0 for non-float grids → “loads successfully” can still render nothing.

- **Invariant: file grid sampling uses NanoVDB’s own world->index transform**

- The shader calls `pnanovdb_grid_world_to_indexf()` on the buffer.
- So **world coordinates must be consistent** with the NanoVDB file’s embedded transform.

- **Invariant: animation advances only if playing + frames exist**

- `Application.cpp` calls `m_nanoVDBSystem->UpdateAnimation(m_deltaTime)` every frame when enabled.
- But `UpdateAnimation()` early-outs unless:
  - `m_animPlaying == true`
  - `m_animFrames` non-empty
  - `m_animFPS > 0`

---

## Highest-probability root cause (new key insight): HALF grids from Blender

### Why this is very likely

PNanoVDB explicitly defines:

- `PNANOVDB_GRID_TYPE_FLOAT = 1`
- `PNANOVDB_GRID_TYPE_HALF  = 9`
- `PNANOVDB_GRID_TYPE_VEC3F = 6`
…and many others.

Your Blender workflow docs (and earlier scripts) frequently set:

- `settings.cache_precision = 'HALF'`

If Blender exports Mantaflow caches as **half precision**, then:

### Why this matches the symptoms

- The `.vdb` density grid may be **half** (not float).
- Conversion to `.nvdb` will likely preserve the grid value type.
- The engine loads the `.nvdb` just fine.
- The shader sees `gridType == 9` (HALF), fails `gridType != 1`, returns 0 density → **invisible**.

This root cause produces a *very specific* failure signature:

- **Load looks “successful”** (file reads, GPU upload succeeds, AABB prints).
- **Debug mode shows mostly Cyan** when rays enter the AABB:
  - because the shader calls `SampleDensity(testPos)` and that returns **0** when `gridType != FLOAT`.
- **You never get Green** for file grids, even when the AABB is correct.

### How to confirm in < 5 minutes (no code changes)

- **Step 1 — Load a Blender-derived `.nvdb`** in PlasmaDX-NanoVDB and turn on:
  - NanoVDB enabled
  - `debugMode = 1` (ImGui)

- **Step 2 — Look at the debug color**
  - **Green**: file grid is returning density (good)
  - **Cyan**: inside AABB but sampled density ~0 (grid type/transform/scale issues)

- **Step 3 — Check the loader logs**
  - If you see: `Grid type: Float (density field)` → this root cause is less likely.
  - If you see warnings like: `Grid is not a float type` and/or an integer grid type printed from `GridData->mGridType`, then:
    - **6** likely means Vec3F (velocity) → wrong grid
    - **9** likely means HALF → unsupported in shader as written

---

## Root cause #2: “Wrong grid” (velocity Vec3F as grid 0)

### Why it happens in this codebase

- The loader reads **grid index 0**.
- Many VDB/NanoVDB assets contain multiple grids: `density`, `velocity`, `temperature`, etc.
- If conversion/export writes **velocity first**, grid 0 becomes `Vec3F` (type 6), and the shader returns 0 density.

### How to confirm

- Loader logs show:
  - `Grid type: Float` **absent**
  - `Grid type: 6` in the generic path (Vec3F)
- Debug mode: **Cyan** (AABB hit but density ~0)

### Non-code mitigation options (pick one)

- **Export/convert density-only** so grid 0 is density.
- **Write one `.nvdb` per grid** and ensure you load `*_density.nvdb`.

> Note: `PlasmaDX-NanoVDB/scripts/convert_vdb_to_nvdb.py` already *tries* to select density/float, but if the density grid is HALF (not FLOAT) its heuristics may still pick a non-float or fallback.

---

## Root cause #3: HALF density grid is real, but shader only supports FLOAT

This is related to root cause #1, but it’s important enough to call out separately because it suggests the “correct” fix direction:

### Options (design decision for Opus)

- **Option A — Force FLOAT at asset time**
  - Make Blender export **FULL** precision (not HALF).
  - Ensure conversion produces FLOAT NanoVDB.
  - Pros: simplest shader.
  - Cons: larger files + more bandwidth.

- **Option B — Add shader support for HALF (grid type 9)**
  - PNanoVDB provides `pnanovdb_read_half()` for HLSL (uses `f16tof32()`).
  - Pros: keeps assets small; matches typical VDB pipelines.
  - Cons: requires shader changes (and trilinear path needs half reads too).

- **Option C — Convert HALF→FLOAT during conversion**
  - If `pyopenvdb` exposes a way to cast grids to float (or if a tool exists), do it once at build time.
  - Pros: engine stays simple.
  - Cons: conversion step complexity; bigger output than HALF.

**My ranking:** Option B is the most robust long-term, because HALF is a very natural representation for volumetric density fields.

---

## Root cause #4: “Scaling” in the UI only scales the AABB, not the sampled grid

### Why this matters

In `NanoVDBSystem`, `ScaleGridBounds(scale)`:

- scales **only** `gridWorldMin/gridWorldMax` (AABB used for ray-box intersection)
- does **not** scale the NanoVDB transform used by `pnanovdb_grid_world_to_indexf()`

So:

- The *culling box* changes
- The *density field in space* does not

### How this can make a real volume look invisible

- If the actual volume is small (Blender-scale, e.g. bounds ~6 units) and:
  - `stepSize` is large (e.g. 10+), the marcher can “step over” thin features.
- If you enlarge the AABB, debug mode samples **the midpoint** of the (now huge) AABB and may show **Cyan**, even if density exists elsewhere.

### What to do during debugging (no code)

- Don’t touch AABB scaling/centering while diagnosing *type* issues.
- First get **Green** with the raw bounds and a tiny step size.

---

## Root cause #5: Animation is loaded but not actually “playing”

The animation system is present and `UpdateAnimation(dt)` is called every frame, but animation only advances if:

- `m_animPlaying == true` (ImGui Play button)
- `m_animFPS > 0` (speed slider)
- frames loaded successfully

### What to check

- After loading animation, check that **frameCount > 0** in the UI.
- Press Play; scrub frame slider to confirm buffer switching.

### When this still doesn’t show anything

If you can scrub frames but never see density (still cyan), the cause is likely **type (HALF/Vec3)**, not animation.

---

## Root cause #6: Per-frame bounds may differ, but bounds are extracted only from frame 0

In `LoadAnimationSequence()` the bounds are extracted from **filepaths[0]** only.

If later frames have different transforms/bounds, the AABB may no longer match, leading to:

- rays missing the true density region
- misleading debug colors

This is lower probability for Mantaflow-style sequences where bounds are stable, but it’s worth keeping in mind for some datasets.

---

## Minimal test plan (recommended to run before any code changes)

### Test 0 — Prove the pipeline can render a file grid at all

- Load the repo’s known `.nvdb` sample (e.g. `cloud_01.nvdb` referenced in UI defaults).
- Enable debug mode.
- Expect: **Green** somewhere on screen when viewing the volume.

If **never green**, focus on grid type support and grid selection first.

### Test 1 — Diagnose Blender-derived asset type immediately

- Load a Blender-derived `.nvdb`.
- Look for loader logs:
  - `Grid type: Float (density field)` (good)
  - or non-float warning and integer `mGridType` (bad)
- If grid type is **9**, treat it as the smoking gun for “HALF grids”.

### Test 2 — Confirm whether “HALF grids” is the core blocker

Do one of:

- Produce one VDB/NVDB using **FULL precision** (not HALF) and convert to `.nvdb`, then load.
- OR keep the same asset but swap to a known float-only `.nvdb`.

If float renders (green) and half does not, the fix direction is confirmed.

---

## Questions for Opus 4.5 (so we converge quickly)

- **Half support**: Given typical volumetric workflows, should we **support PNANOVDB_GRID_TYPE_HALF (9)** in shader?
- **Trilinear support**: If we support HALF, should we:
  - support only nearest sampling first, then trilinear
  - or implement both immediately
- **Grid selection**: Should the loader remain “grid index 0”, or should it select:
  - prefer grid named `density`
  - else first scalar grid of type FLOAT/HALF
- **Scaling semantics**: Do we want to treat **AABB scaling** as a *debug-only* tool and add a proper **sampling scale** parameter later?

---

## Suggested “first fix” direction (no implementation yet)

If Opus agrees, the smallest high-impact fix set is:

- Shader: support **FLOAT + HALF** grid types (1 and 9).
- Loader: prefer scalar density grids (name/type) over “grid 0”.
- Conversion: ensure output `.nvdb` is density-only or clearly named per-grid outputs.

This should unlock:

- Blender assets showing up reliably
- Animation sequences showing up (assuming frame loading works)
