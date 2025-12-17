# GPT-5.2 — Supergiant Star — Blender OpenVDB Recipe

**Difficulty:** Intermediate  
**Method:** Mantaflow (Gas / Smoke) + 2 emitters (core + hotspot)  
**Blender Version:** 5.0+  
**Export Format:** OpenVDB (`density` is the primary grid)  
**Primary Output:** `vdb_cache/` OpenVDB sequence + `.blend` + optional renders  
**Script:** `assets/blender_scripts/GPT-5.2/blender_supergiant_star.py`

---

## What this creates

This recipe generates a **single supergiant star volume**: a large, glowing turbulent sphere with evolving interior “convection” patterns.

It is designed to produce:

- A **working OpenVDB cache** that can be converted to NanoVDB
- A **working Cycles render** (still by default, optional animation)

---

## Quick start (headless, one command)

Run from repo root:

```bash
blender -b -P assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
  --output_dir "/abs/path/to/out/SupergiantStarAsset" \
  --name "GPT-5-2_SupergiantStar" \
  --resolution 128 \
  --domain_size 10.0 \
  --star_radius 3.0 \
  --frame_end 96 \
  --bake 1 \
  --render_still 0 \
  --still_frame 60 \
  --render_anim 0 \
  --render_res 1920 1080
```

**Outputs (inside `--output_dir`)**

- **Blend file**: `SupergiantStar.blend`
- **VDB cache directory**: `vdb_cache/`
- **Renders**: `renders/SupergiantStar_still.png` (and/or animation frames if enabled)

---

## Converting to NanoVDB for PlasmaDX (important)

PlasmaDX’s current NanoVDB shader path only supports **FLOAT grids**. Mantaflow caches can include multiple grids (including vector velocity), so you should convert **explicitly**:

```bash
python scripts/convert_vdb_to_nvdb.py "/abs/path/to/out/SupergiantStarAsset/vdb_cache/<frame>.vdb" \
  "/abs/path/to/out/SupergiantStarAsset/SupergiantStar_density.nvdb" \
  --grid density
```

If your volume renders “invisible” in-game, read:

- `docs/NanoVDB/TROUBLESHOOTING_ASSET_VISIBILITY_AND_ANIMATION.md`

---

## The knobs you should change (to get higher quality)

### Quality / performance

- **`--resolution`**
  - 64: fast preview
  - 96–128: good “real asset” baseline
  - 192–256: high detail (larger files, slower bakes)

- **`--frame_end`**
  - 48–96 is a good starting range for motion
  - Longer sequences bake slower and consume more disk

### Shape & scale

- **`--domain_size`**
  - Must comfortably contain the star and the turbulent “halo”
  - If the star clips, increase domain_size or reduce star_radius

- **`--star_radius`**
  - Controls the size of the core emitter (primary density source)

### Render outputs

- **`--render_still` / `--still_frame`**
  - Use a later frame (e.g. 60+) so the star has developed structure

- **`--render_anim`**
  - Set to `1` if you want a rendered animation sequence for review

---

## Why the script is built this way (major decisions)

### “Supergiant star” from smoke (not fire)

The script uses Mantaflow **SMOKE inflows** because:

- PlasmaDX currently treats the NanoVDB as a **scalar density field**
- Converting/transporting “temperature” reliably isn’t the first priority for getting a working asset

### Two emitters (core + hotspot)

One emitter tends to look too uniform. The hotspot emitter:

- Adds evolving structure without requiring complex force-field setups
- Is animated to orbit slowly, generating more interesting turbulence

### “Space-like” physics

The domain is configured with:

- **gravity = 0**
- buoyancy disabled (`alpha=0`, `beta=0`)

This avoids “smoke rising” behavior and keeps the look closer to a volumetric plasma ball.

---

## Blender 5 API sources (MCP paths)

The script relies on Blender 5–verified operators/properties from the Blender Manual MCP:

- `bpy.ops.fluid.html` (bake operators, including `bake_all`)
- `bpy.types.FluidDomainSettings.html` (cache fields, OpenVDB enums)
- `bpy.types.FluidFlowSettings.html` (inflow settings)
- `physics/fluid/type/domain/cache.html` (OpenVDB caching UI behavior)

---

## Troubleshooting

### “VDB bakes but PlasmaDX shows nothing”

- Convert with `--grid density` (see above)
- In PlasmaDX, turn on shader debug mode; you should see **Green** for file grids.
- See: `docs/NanoVDB/TROUBLESHOOTING_ASSET_VISIBILITY_AND_ANIMATION.md`

### “Bake is too slow”

- Drop to `--resolution 64` for iteration
- Shorten `--frame_end`
- Keep adaptive domain enabled (script does)

### “Render shows a big white ball”

That’s almost always the **emitter mesh** being rendered (not the volume).

- This script **always hides emitters from renders**.
- By default, emitters stay visible in the viewport as **wireframes**.
  - To hide them in the viewport too: run with `--show_emitters 0`.

### “My GPU driver resets (TDR) when I try to render”

See:
- `assets/blender_docs/GPT-5.2/TDR_SAFE_WORKFLOW.md`
- Optional helper: `assets/blender_scripts/GPT-5.2/blender_tdr_safe_config.py`


