# GPT-5.2 — Avoiding GPU TDRs while generating/rendering volumetric assets (Blender 5.0)

If you’re seeing **Windows TDR (Timeout Detection and Recovery)** events while “testing” volumetric scripts, it’s almost always caused by **long-running GPU work** (Cycles GPU/OptiX volume rendering is a common trigger).

This doc explains the safest workflow for your “mini-library” scripts.

---

## What Blender’s docs say (authoritative)

From the Blender 5.0 Manual:

- **Cycles render Device (CPU vs GPU Compute)**:  
  `render/cycles/render_settings/index.html`
- **Cycles volumes have “Step Rate” and “Max Steps” specifically to cap long renders**:  
  `render/cycles/render_settings/volumes.html`
- **GPU Rendering warns about Windows display driver timeouts** and suggests:
  - reduce tile size (performance panel) and/or
  - increase the time-out (system-level)  
  `render/cycles/gpu_rendering.html`

---

## Recommended workflow (TDR-safe)

### 1) Bake VDBs headless (no viewport, no GPU render)

For asset generation scripts, the safest baseline is:
- **Bake**: yes
- **Render**: no (disable `--render_still` / `--render_anim`)

Example (Supergiant):

```bash
blender -b -P assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
  --output_dir "/abs/path/to/out/SupergiantStarAsset" \
  --name "GPT-5-2_SupergiantStar" \
  --resolution 96 \
  --frame_end 96 \
  --bake 1 \
  --render_still 0 \
  --render_anim 0
```

### 2) If you *must* render, use CPU device first

GPU volume path tracing is the highest risk for TDR. If you only need a sanity-check image:
- render on **CPU** first (slower, but won’t trip TDR).

In the supergiant script this is now the default when using Cycles.

### 3) Clamp volume complexity

Cycles volume settings:
- increase **Step Rate** (bigger step size → fewer steps)
- lower **Max Steps** (hard cap)

These exist specifically to avoid extreme render times (`render/cycles/render_settings/volumes.html`).

---

## “One-file config” helper you can run first

Run:
- `assets/blender_scripts/GPT-5.2/blender_tdr_safe_config.py`

It:
- sets scene render engine to EEVEE (reduces accidental Cycles viewport)
- forces Cycles device to CPU (scene-level)
- applies conservative cycles samples/bounces/volume step clamps when those RNA props exist

---

## Optional system-level mitigations (last resort)

If you insist on **Cycles GPU volume rendering** on Windows and still hit TDR:
- Blender docs explicitly mention increasing the time-out (system-level).
- This is a Windows registry change; it’s powerful but can make the UI hang longer during real GPU stalls.

I can write a short “do you really want this?” checklist + exact registry keys if you want, but I’d recommend exhausting:
- CPU renders for preview
- higher volume step rate + lower max steps
- lower render resolution/samples
first.


