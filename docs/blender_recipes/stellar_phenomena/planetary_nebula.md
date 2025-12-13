# Planetary Nebula (Bipolar) — Blender OpenVDB Recipe

**Difficulty:** Intermediate  
**Method:** Mantaflow (Gas / Smoke) + 3 emitters (ring + bipolar jets)  
**Blender Version:** 5.0+  
**Export Format:** OpenVDB (density + temperature)  
**Primary Output:** `vdb_cache/` OpenVDB sequence + a `.blend` + optional renders  
**Script:** `docs/blender_recipes/scripts/blender_bipolar_planetary_nebula.py`

---

## What this creates (visual + intent)

This recipe generates a **bipolar planetary nebula**: a bright equatorial “ring” and two opposite “lobes/jets” expanding along a symmetry axis.

- **Ring**: Dense equatorial flow (torus emitter) that reads as a waist / disk.
- **Jets**: Two inflow emitters that push material along ±Z to form lobes.
- **Motion**: Slow precession / rotation in the emitters so the nebula doesn’t look perfectly symmetric.
- **“Space-like” behavior**: The domain gravity/buoyancy is set to zero so the shape is driven by emission velocity and turbulence rather than rising smoke.

---

## Quick start (headless, one command)

Run from repo root:

```bash
blender -b -P docs/blender_recipes/scripts/blender_bipolar_planetary_nebula.py -- \
  --output_dir "/abs/path/to/out/BipolarNebulaAsset" \
  --name "BipolarNebula" \
  --resolution 128 \
  --domain_size 6.0 \
  --frame_end 160 \
  --bake 1 \
  --render_still 1 \
  --still_frame 100 \
  --render_anim 0 \
  --render_res 1920 1080
```

**Outputs (inside `--output_dir`)**
- **Blend file**: `BipolarNebula.blend` (saved automatically before baking)
- **VDB cache directory**: `vdb_cache/` (Blender may place files inside subfolders depending on cache layout)
- **Renders**: `renders/BipolarNebula_still.png` (and/or animation frames if enabled)

Then convert a frame to NanoVDB:

```bash
python scripts/convert_vdb_to_nvdb.py "/abs/path/to/out/BipolarNebulaAsset/vdb_cache/<some_frame>.vdb"
```

---

## The knobs you should change (before/when running)

The script is designed to be “runs with defaults”, but **quality and look** depend heavily on a few parameters.

### Quality / performance

- **`--resolution`** (default 96)
  - **64**: fast iteration / test the pipeline
  - **96–128**: good “standard”
  - **192–256**: hero shots (bake time and cache size grow quickly)
- **`--frame_end`**
  - More frames = more evolution and more disk usage.
- **`--domain_size`**
  - Bigger domain gives more room for lobes to expand, but can dilute detail at fixed resolution.
  - If you increase domain size, also increase resolution to keep voxel detail.

### Shape / composition

- **Ring thickness and size**: controlled inside the script by torus parameters derived from `domain_size`
  - `major_radius = domain_size * 0.22`
  - `minor_radius = domain_size * 0.035`
- **Jet “punch”**: the main driver of bipolar lobes
  - `velocity_normal` and `density` on each jet flow
- **Turbulence detail**:
  - `dset.vorticity`, `dset.use_noise`, `dset.noise_strength`, `dset.noise_time_anim`

### Render output

- **`--render_still` / `--render_anim`**
  - `--render_still 1` is the easiest sanity check.
  - `--render_anim 1` will render the full frame range.
- **`--still_frame`**
  - Pick a frame where the shape is developed (often ~60–70% of frame_end).
- **`--render_res W H`**
  - Affects only the render output, not the VDB bake.

---

## What the script does (step-by-step tutorial)

This section mirrors the script structure so you (and Opus) can evolve it into higher-quality recipes.

### 1) Scene reset

The script deletes default objects and attempts to remove orphan datablocks. This keeps runs deterministic.

### 2) Create the Mantaflow GAS domain

It creates a cube domain and configures:

- **Gas domain**: `domain_settings.domain_type = 'GAS'`
- **Resolution**: `domain_settings.resolution_max = --resolution`
- **Adaptive domain**: enabled for memory savings and to keep the active region tight.
- **Space-like motion**:
  - `gravity = (0,0,0)`
  - `alpha = 0` (density buoyancy)
  - `beta = 0` (heat buoyancy)
  - The nebula expands because emitters inject velocity + the domain has turbulence/noise.

### 3) Cache/export settings (OpenVDB)

This is the most important part for the Blender → NanoVDB pipeline:

- `cache_type = 'ALL'` so we can use one bake (`bpy.ops.fluid.bake_all()`)
- `cache_data_format = 'OPENVDB'`
- `cache_directory = <output>/vdb_cache`
- `cache_frame_start/end` from args

#### Compression: why the script does a “safe fallback”

Blender 5’s manual mentions **Zip / Blosc / None** for OpenVDB compression, but the **Blender 5.0 Python API** for `FluidDomainSettings.openvdb_cache_compress_type` only exposes:

- `ZIP`
- `NONE`

So the script tries `"BLOSC"` and falls back to `"ZIP"` if it’s not in the enum list (via `_safe_enum_set`).

### 4) Emitters: ring + jets

The nebula shape comes mostly from emitter geometry:

- **Ring**: a torus emitter centered in the domain.
  - Inflow smoke with moderate density and velocity
  - Adds the equatorial waist and overall “donut” brightness.
- **Jets**: two cone emitters (one flipped) close to the center.
  - Higher `velocity_normal` and slightly higher density/temperature
  - Produces two lobes in opposite directions (bipolar).

### 5) Animation (symmetry breaking)

Planetary nebulae look artificial if perfectly axis-symmetric. The script:

- Rotates the ring over the shot.
- Adds small rotations to the jet emitters over the shot.

This creates a subtle precession that turns “CG perfect” into “plausibly organic”.

### 6) Material (for Blender preview + render sanity checks)

The domain gets a **Principled Volume** material:

- A base cyan/blue scattering color
- Emission enabled (so the nebula glows)
- Slight backward anisotropy (wispy look)

This material **does not affect the VDB export** — it’s for Blender visualization and the “bonus render” outputs.

### 7) Save the `.blend`, bake, render

To avoid the classic “bake into nowhere / relative path confusion” problem:

- The script saves the `.blend` to the output directory **before baking**.
- Then (optionally) bakes the simulation with:
  - `bpy.ops.fluid.bake_all()`
- Then (optionally) renders still/animation with:
  - `bpy.ops.render.render(write_still=True)` or `animation=True`

---

## Troubleshooting (most common failure modes)

### Bake runs but no `.vdb` files appear
- **Check output dir**: make sure `--output_dir` is an absolute path and writable.
- **Cache layout**: Blender may put VDBs in subfolders under `vdb_cache/` (domain may create per-type directories). Search recursively for `*.vdb`.
- **Frame range mismatch**: ensure `--frame_start/--frame_end` are what you expect.

### Script errors on compression/precision settings
- This is why `_safe_enum_set()` exists.
- If your Blender build exposes additional values, the script will use them; otherwise it falls back safely.

### Render is black / nothing visible
- Render is a convenience output; it’s possible to bake valid VDBs even if render settings are poor.
- Try `--still_frame` later in the range (e.g., 0.7× frame_end).
- Try increasing Principled Volume Density/Emission in the script material.

### PlasmaDX doesn’t show the grid
- Validate the VDB with:
  - `python scripts/inspect_vdb.py <frame>.vdb`
- Convert to NanoVDB with:
  - `python scripts/convert_vdb_to_nvdb.py <frame>.vdb`
- If file-loaded NVDB still doesn’t render, use your shader debug mode (`debugMode=1`) and check bounds/axis conversion (Blender Z-up vs DX Y-up).

---

## “Why” behind the major decisions (so you can evolve quality)

### Why Mantaflow for planetary nebulae?
- Mantaflow gives you “free” **turbulent, filamentary structure** that reads like astrophysical gas.
- It outputs OpenVDB sequences easily, which matches your NanoVDB pipeline.

### Why a ring + jets?
It’s the simplest controllable generator for the archetypal bipolar shape:
- Ring provides an equatorial density enhancement.
- Jets carve/drive lobes.
- It’s easy to parameterize (radii, velocities, densities).

### Why gravity/buoyancy = 0?
Smoke “rising” is a dead giveaway for Earth-bound sims. Zeroing gravity and buoyancy:
- Removes the default upward drift
- Leaves a shape driven by emitter velocity and turbulence
- Feels more “space plausible” even though it’s still smoke under the hood.

### Why “ALL” cache type?
Because it enables `bpy.ops.fluid.bake_all()`, which is the most automation-friendly, one-button bake route.

### Why the enum-safe setter?
Blender 5 has real API churn + some UI/manual vs RNA enum differences. `_safe_enum_set()` makes the script resilient:
- It can try “best” options first (like BLOSC)
- It won’t crash if that option isn’t present.

---

## Blender 5 sources used (MCP + where to find them)

These are the exact documentation sources I used to avoid relying on stale training data. You can open the same pages via the Blender manual MCP server.

### Manual (UI behavior and conceptual docs)
- **Fluid Domain Cache panel (OpenVDB format/compression/precision)**  
  - MCP path: `physics/fluid/type/domain/cache.html`

### Python API (authoritative operator/property names and enums)
- **Fluid bake operators** (`bpy.ops.fluid.bake_all`, etc.)  
  - MCP path: `bpy.ops.fluid.html`
- **Domain settings and enum values** (e.g., `cache_data_format`, `cache_type`, `openvdb_cache_compress_type`)  
  - MCP path: `bpy.types.FluidDomainSettings.html`
- **Flow settings** (e.g., `flow_type`, `flow_behavior`, `density`, `temperature`, `velocity_normal`)  
  - MCP path: `bpy.types.FluidFlowSettings.html`
- **Render operator** (`bpy.ops.render.render`)  
  - MCP path: `bpy.ops.render.html`
- **Saving `.blend`** (`bpy.ops.wm.save_as_mainfile`)  
  - MCP path: `bpy.ops.wm.html`

If you want to reproduce this workflow in a fresh session:
- Use MCP search tools like `search_bpy_types("FluidDomainSettings")` and `read_page("bpy.types.FluidDomainSettings.html")`.

---

## Next improvements (high-leverage for higher quality assets)

If you want Opus/Gemini/this agent to push quality up quickly, these are the most impactful upgrades:

- **Add temperature-driven emission**:
  - Drive the material with Blackbody and ensure the temperature grid is meaningful.
  - (Your NanoVDB shader can then map temperature→color consistently.)
- **Guiding / vector fields**:
  - Use a guiding domain or animated force fields to art-direct lobes.
- **Cache hygiene**:
  - Add a “free cache” step and an option to bake only selected sub-stages.
- **Determinism knobs**:
  - Add random seeds and controlled noise textures to get repeatable “variants”.
- **Multi-resolution / tiling**:
  - Bake a high-res close-up wedge plus a low-res full volume and blend in-engine.

---

## File locations in this repo

- **Script**: `docs/blender_recipes/scripts/blender_bipolar_planetary_nebula.py`
- **This recipe**: `docs/blender_recipes/stellar_phenomena/planetary_nebula.md`
- **VDB conversion**: `scripts/convert_vdb_to_nvdb.py`
- **VDB inspection**: `scripts/inspect_vdb.py`


