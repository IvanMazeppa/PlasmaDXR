# Troubleshooting: NanoVDB assets not showing / not animating in PlasmaDX-Clean

This document targets the common failure mode you described: **procedural fog renders**, but **file-loaded `.nvdb` volumes appear invisible**, clipped, or do not animate.

It is written for the current PlasmaDX-Clean implementation:

- Loader: `src/rendering/NanoVDBSystem.cpp`
- Shader: `shaders/volumetric/nanovdb_raymarch.hlsl` (PNanoVDB sampling)
- Conversion: `scripts/convert_vdb_to_nvdb.py` and/or Blender export

---

## Fast triage checklist (5 minutes)

### 1) Confirm the system is actually rendering *something*

- **Enable the NanoVDB system** in your app/ImGui.
- Turn on shader **debug mode** (see “Debug colors” below).

Expected:
- **Magenta** means *procedural density path* is working.
- **Green** means *file grid density path* is working.
- **Cyan** means “ray is inside AABB but sampled density is ~0” (usually a grid/type/transform mismatch).

### 2) Confirm the `.nvdb` you’re loading contains a FLOAT grid at index 0

PlasmaDX currently loads **grid index 0**:

- In `NanoVDBSystem::LoadFromFile()` it reads `readGrid(filepath, 0, 1)`.
- In the shader, `SampleNanoVDBDensity()` explicitly rejects non-float grids:
  - `gridType != 1u` → returns 0 (invisible).

If your `.nvdb` file contains a **Vec3 velocity** grid (or any non-float grid) as the stored grid, it will render as “nothing”.

**Action:** In logs, look for:
- `[NanoVDB] Grid type: Float (density field)` (good)
- or warnings about “Grid is not a float type” (bad for current shader)

---

## Root cause #1 (very likely): converting the wrong grid (velocity) instead of density

### Why this happens

Blender/Mantaflow `.vdb` often contains multiple grids, e.g.:
- `density` (float) ✅
- `velocity` (vec3) ❌ for current shader
- `heat` / `temperature` / `flame` (float) (maybe)

If your converter picks **the last grid** or the **first grid that isn’t density**, your `.nvdb` becomes non-float or just the wrong scalar field.

### Confirm it

Run:

```bash
python scripts/convert_vdb_to_nvdb.py --list /path/to/frame_0100.vdb
```

You want to see a grid named **`density`** and some indication it’s a float-ish type.

### Fix it (recommended)

Use the updated converter with an explicit grid:

```bash
python scripts/convert_vdb_to_nvdb.py /path/to/frame_0100.vdb /path/to/out.nvdb --grid density
```

Or write one `.nvdb` per grid for inspection:

```bash
python scripts/convert_vdb_to_nvdb.py /path/to/frame_0100.vdb /path/to/out_dir --all-grids
```

Then load the `*_density.nvdb` file in PlasmaDX.

### Note on the old behavior (important)

If you previously used the older `scripts/convert_vdb_to_nvdb.py`, it wrote **multiple grids to the same output path**, so the last write “won”. This is exactly how you end up with a velocity grid in a file named like it’s density.

---

## Root cause #2: `.nvdb` file contains multiple grids, but PlasmaDX loads grid index 0 only

Even if the file contains `density`, PlasmaDX loads **index 0** only. If your exporter writes `velocity` first, you lose.

### Fix options

- **Conversion-side fix** (best right now): generate `.nvdb` that contains *only* `density` as grid 0 (the updated converter does this).
- **Engine-side improvement** (future): change loader to pick grid by type/name:
  - Prefer float grid named `density`
  - Else first float grid
  - Else fall back to procedural

---

## Root cause #3: AABB/transform mismatch (inside box but sampling empty)

Symptoms:
- Debug shows **Cyan** a lot (inside AABB but no density).

What this usually means:
- `gridWorldMin/gridWorldMax` don’t match the grid’s actual transform used by PNanoVDB, or
- Your volume is moved in engine with AABB updated but `gridOffset`/sampling transform doesn’t match expectations, or
- Coordinate system mismatch (Z-up vs Y-up) moves the density away from where you think it is.

### Actions

- **Do not reposition first.** Load at origin and verify Green shows up.
- In engine, temporarily **scale bounds up** (ImGui “Grid Scale”) to ensure your rays pass through the true density.
- Temporarily set **stepSize larger** and **densityScale higher** to brute-force visibility.

Also re-check the roadmap’s suspected mismatch:
- `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` mentions file-loaded grids not rendering and calls out bounds/transform/axis issues.

---

## Root cause #4: density is there but too small (thresholding + absorption)

In the shader, there’s an early threshold:
- it only accumulates when density > ~0.001 (and then applies Beer-Lambert).

If the density field is extremely low after conversion, you’ll see almost nothing.

### Actions

- Raise **densityScale** in-game aggressively (e.g. 10–100) just to see something.
- Reduce **absorptionCoeff**.
- Increase **emissionStrength** (even for “nebula” style).

---

## Root cause #5: animation is loaded but not advanced

Animation in `NanoVDBSystem` advances only if:
- `SetAnimationPlaying(true)` AND
- `UpdateAnimation(deltaTime)` is called each frame AND
- `m_animFPS > 0`

If any of these aren’t true, animation will “load” but stay on frame 0.

### Actions

- Log `GetAnimationFrame()` each second.
- Ensure the main loop calls `m_nanoVDB->UpdateAnimation(dt)`.
- Confirm that you are actually calling `LoadAnimationFromDirectory()` (or `LoadAnimationSequence()`).

---

## Debug colors (what they mean)

From `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md`:

- **Green**: grid density found (success path)
- **Magenta**: procedural density found
- **Cyan**: inside AABB but no density found (sampling mismatch)

If you never get Green on file grids, start with **Root cause #1** and **#2**.

---

## Practical “known-good” workflow (minimal variables)

1. In Blender, generate a smoke sim (only needs density).
2. Convert **one frame** explicitly:

```bash
python scripts/convert_vdb_to_nvdb.py /abs/path/frame_0100.vdb /abs/path/frame_0100_density.nvdb --grid density
```

3. Load that one file with `LoadFromFile()` (not animation).
4. Turn on debug mode and verify you can get **Green**.
5. Only then:
   - Convert a whole sequence
   - Load as animation
   - Add repositioning / scaling / axis conversions

---

## Where the “truth” is in this repo (high-signal files)

- `src/rendering/NanoVDBSystem.cpp`
  - `LoadFromFile()` (grid index 0)
  - `LoadAnimationSequence()` and `UpdateAnimation()`
- `shaders/volumetric/nanovdb_raymarch.hlsl`
  - Rejects non-float grids in `SampleNanoVDBDensity()`
  - Debug output colors
- `scripts/convert_vdb_to_nvdb.py`
  - Grid selection (use `--grid density`)
- `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md`
  - Documents the known “file grid may not render” issue and debug approach


