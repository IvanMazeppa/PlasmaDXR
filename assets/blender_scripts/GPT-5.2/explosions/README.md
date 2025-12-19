# GPT-5.2 Explosion Library (Blender 5.0+)

This folder contains **3 explosion assets** spanning from grenade-scale to supernova-scale.

Each explosion has **two scripts**:
- **`*_bake.py`**: sets up Mantaflow GAS + exports **OpenVDB** caches (`vdb_cache/`)
- **`*_render.py`**: imports a VDB sequence and renders an **MP4** (FFmpeg/H.264)

## 0) Pipeline Test (minimal)

- **Setup/Bake**: `blender_explosion_pipeline_test_bake.py`
- Writes a helper `convert_density_to_nvdb_nozip.sh` next to the output `.blend`.

## 1) Grenade Explosion

- **Bake**: `blender_explosion_grenade_bake.py`
- **Render**: `blender_explosion_grenade_render.py`

## 2) Stellar Burst Explosion (mid-scale nova)

- **Bake**: `blender_explosion_stellar_burst_bake.py`
- **Render**: `blender_explosion_stellar_burst_render.py`

## 3) Supernova Explosion (large-scale)

- **Bake**: `blender_explosion_supernova_bake.py`
- **Render**: `blender_explosion_supernova_render.py`

## Typical workflow (Blender UI)

1. Run a **bake** script (Alt+P) to generate a `.blend` and configure caching.
2. Bake:
   - Select Domain → Physics → Fluid → Cache → **Bake All**
3. Convert one representative VDB frame to NanoVDB for PlasmaDX:

```bash
nanovdb_convert -f -g density /path/to/fluid_data_0040.vdb /path/to/fluid_data_0040_density.nvdb
```

Important: your current engine build logs indicate **ZIP codec is disabled**, so prefer **codec NONE** (`nanovdb_convert` without `-z`) unless you enable ZIP in the engine build.

4. Run the **render** script and point it at the first VDB frame (it will treat it as a sequence when possible) and render MP4.

## Notes / knobs

- **Quality**: `--resolution` (domain voxel resolution). Higher = slower.
- **Duration**: `--frame_end`.
- **Look**: `--turbulence`, `--vortex` (where available), plus Principled Volume parameters in render scripts.

