# Noise
DocType: manual-rewritten
DocPath: physics/fluid/type/domain/gas/noise.html
DocVersion: 5.0.1
Model: gpt-5-mini
OriginalWords: 756
RewrittenWords: 451
CompressionRatio: 0.60
---

One-line summary: Controls and parameters for adding Wavelet Turbulence (noise) to gas domains to add fine-scale detail without altering base fluid motion.

## Overview
- Noise implements Wavelet Turbulence for Fluid Simulation to add fine-scale details (vortices) on top of the base gas simulation.
- Enabling Noise signals the cache to read noise simulation data; if noise data is absent the domain will appear empty.
- Enabling/disabling the Noise checkbox does not reset the cache and can be used to switch between base-resolution and noise views.

## Parameters
- `Noise` (bool, range not specified, default not specified) — Enables noise (wavelet turbulence). Signals cache to read noise data; does not clear existing cache.
- `Upres Factor` (numeric, range not specified, default not specified) — Factor by which noise resolution is enhanced. Higher → more detailed noise. Scaling is coupled to `Resolution Divisions` but the two are not equivalent.
  - Higher `Upres Factor` → higher noise grid resolution → more detail but increased memory/compute.
- `Strength` (numeric, range not specified, default not specified) — Strength of the noise. Higher → more turbulent vortices.
- `Scale` (numeric, range not specified, default not specified) — Spatial scale of the noise. Larger → larger vortices.
- `Time` / `Animation Time` (numeric, range not specified, default not specified) — Time offset where the noise field is evaluated. Acts as a seed: different values produce visually different noise patterns while base fluid motion remains the same. Example values shown in documentation: 0.1, 1.0, 2.5, 10.0 (examples only).

## Bake / Cache controls (Modular cache only)
- `Bake Noise` (action, available only with Modular cache) — Starts baking the noise simulation. Progress is shown in the status bar. Baking can be paused or resumed.
- `Free Noise` (action, available only with Modular cache) — Deletes baked noise cache after baking completes.
- Runtime behavior:
  - Progress displayed in status bar.
  - Pressing Esc during bake → pauses the bake process.
  - Baking can be paused and resumed.

## Relationships, tradeoffs, and techniques
- Enabling noise adds high-frequency detail without changing base fluid motion.
- `Upres Factor` is coupled to `Resolution Divisions` → combinations of these two produce different visual styles; they are not interchangeable.
  - TECHNIQUE: Use lower base `Resolution Divisions` with larger `Upres Factor` to get visually smaller-scale features (can emulate pyroclastic plumes).
- `Strength` vs `Scale`:
  - Increase `Strength` → increases turbulence intensity (more vortices).
  - Increase `Scale` → increases vortex size.
- `Time` as seed:
  - Small changes in `Time` → different noise field evaluations; useful to vary appearance between otherwise identical domains.
- NOTE: If Noise is enabled but no noise cache/data exists, the domain will display empty noise—ensure noise is baked or available.