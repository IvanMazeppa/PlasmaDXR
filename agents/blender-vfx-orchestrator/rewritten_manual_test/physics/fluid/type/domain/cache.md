# Cache
DocType: manual-rewritten
DocPath: physics/fluid/type/domain/cache.html
DocVersion: 5.0.1
Model: gpt-5-mini
OriginalWords: 1751
RewrittenWords: 723
CompressionRatio: 0.41
---

Summary: Cache controls baking and storage of fluid domain simulation results (file formats, frame range, bake modes, resumability, mesh/export options).

## Overview
- Cache stores baked simulation results so they do not need recalculation. Baking is computationally expensive → allocate sufficient time and storage.
- NOTE: Fluid simulations use their own cache system; other physics use General Baking operators.
- TECHNIQUE: If domain or obstacle meshes have modifiers, the render settings are used when exporting mesh to the fluid solver. High-cost setups (e.g., moving mesh + Subdivision Surface as obstacle) → exponential increase in computation time and memory. To reduce sim time: disable modifiers or lower subdivision during testing; when rig/setup is correct, increase quality for realism.

## Cache parameters
- `cache_directory` (path, range not specified, default: not specified) — Directory where baked simulation files are stored. Inside this directory each simulation type (mesh, particles, noise, etc.) has its own subdirectory containing the simulation data.
- `frame_start` (int, range not specified, default: not specified) — First frame the simulation starts and the first one baked.
- `end` (int, range not specified, default: not specified) — Last frame the simulation ends and the last one baked. NOTE: Simulation is only calculated for positive frames between `frame_start` and `end`. To simulate longer than the default frame range, increase `end`.
- `offset` (int, range not specified, default: not specified) — Frame offset used when loading simulation from cache. Not considered when baking; only affects playback/loading.
- `type` (enum {Replay, Modular, All}, default: not specified) — Controls bake workflow:
  - Replay: cache is baked as the simulation is played in the viewport.
  - Modular: cache baked step-by-step; bake operators are distributed across domain-related panels (e.g., mesh bake in Mesh panel).
  - All: single bake tool bakes all selected settings at once; uses the Cache panel bake operator.
  - IMPORTANT: Replay only works when Playback Sync mode = "Play Every Frame". If using "Frame Dropping" or "Sync to Audio", use Modular or All.
- `resumable` (bool, range not specified, default: not specified) — Save extra data allowing baking to be paused and resumed. Extra data → more disk writes → avoid enabling at high resolutions.
- `Bake All` / `Free All` (operator, availability: only when using the Final cache type) — Bake All runs the entire simulation considering all parameters (equivalent to running all Modular steps at once). Progress is displayed in the status bar; pressing Esc aborts the bake. After baking, cache can be deleted with Free All. NOTE: Bake All cannot be paused or resumed because only the most essential cache files are stored on disk.

## Volumetric data (grids & particles)
- `format` (enum {Uni Cache, OpenVDB}, range not specified, default: not specified) — File format for volume-based simulation data:
  - Uni Cache: Blender’s own cache format with some compression. Each simulation object is stored in its own .uni cache file.
  - OpenVDB: advanced, efficient format. All simulation objects (grids, particles) are stored in a single .vdb file per frame.
- `compression` (enum {Zip, Blosc, None}, scope: OpenVDB only, range not specified, default: not specified) — Compression method used when writing OpenVDB files:
  - Zip: effective compression but slower than Blosc.
  - Blosc: multithreaded compression, similar size/quality to Zip, faster.
  - None: no compression.
- `precision` (enum {Full, Half, Mini}, scope: OpenVDB only, range not specified, default: not specified) — Precision level when writing OpenVDB:
  - Full: full precision (32-bit floats).
  - Half: 16-bit floats.
  - Mini: 8-bit mini-float where possible; where not possible, 16-bit floats are used instead.

## Mesh cache (Liquids only)
- `meshes_format` (enum {Binary Object, Object}, range not specified, default: not specified) — File format for mesh cache:
  - Binary Object: mesh data files with some compression.
  - Object: simple, standard mesh data format.

## Export
- `export_mantaflow_script` (bool, range not specified, default: not specified) — Export the simulation as a standalone Mantaflow script when baking (export occurs on Bake Data). Intended for developers/advanced users who can use the Mantaflow GUI. NOTE: Enable via Debug Value 3001.

## Behavior & UX notes
- Baking writes per-frame files/subdirectories under `cache_directory` for different simulation types.
- Increasing cache detail, mesh resolution, modifiers, or using OpenVDB high precision → higher disk usage and compute time.
- TECHNIQUE: Iterate at low resolution/modifier levels for setup and animation validation; switch to higher settings for final bakes.