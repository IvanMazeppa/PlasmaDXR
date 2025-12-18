# `compound:engineering` Prompt — Blender 5 → NanoVDB + World-Space Volumetrics (2025-12-17)

This document is a **ready-to-paste prompt** for the Claude Agent SDK plugin **`compound:engineering`** (or any other multi-agent planning tool).  
It focuses on **fixing Blender 5-created NanoVDB assets not rendering** and implementing a **world-space multi-volume + menu control system**.

> Copy everything inside the fenced block below into the tool/session.

---

```
You are compound:engineering running a 20+ agent engineering swarm. Your job is to plan and implement fixes + features in a DX12/DXR project.

#### Hard constraints
- ONLY modify code inside: /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
- DO NOT touch any other worktrees/branches/directories (they are intentionally left untouched for easy revert).
- Prefer small, reviewable commits and keep changes easy to revert.
- Treat code as authoritative over docs when they disagree, but resolve doc contradictions as part of the work.

---

### Context (what exists today)
Project: PlasmaDX-Clean (DX12 + DXR) supports real-time volumetrics via NanoVDB.

Key implementation:
- src/rendering/NanoVDBSystem.{h,cpp}: loads a single .nvdb grid file and uploads to GPU.
- shaders/volumetric/nanovdb_raymarch.hlsl: raymarches volume; has debug visualization.
- shaders/nanovdb/PNanoVDB.h: portable NanoVDB HLSL header; supports many grid types.
- src/core/Application.cpp: ImGui control panel already has a “NanoVDB Volumetric System” section.

The key problem:
- Procedural NanoVDB objects render (procedural fog sphere).
- Blender 5-created assets (converted to .nvdb) often do not render.
- Debug mode indicates cases like “inside AABB but no density”.

---

### What we’ve learned (high-signal findings)

1) Blender OpenVDB cache precision affects grid type
- Blender manual: physics/fluid/type/domain/cache.html
- Precision options: Full (32-bit), Half (16-bit), Mini (8-bit where possible; otherwise 16-bit).
Implication: Blender exports can become HALF/FP16 grids after conversion.

2) PlasmaDX shader currently rejects non-float grids
- In shaders/volumetric/nanovdb_raymarch.hlsl, sampling early-outs unless gridType == PNANOVDB_GRID_TYPE_FLOAT (1).
- PNanoVDB supports HALF (9), FP16 (15), etc, and provides helper readers (e.g. pnanovdb_read_half).
Implication: Half/Mini precision assets can render as density=0 everywhere.

3) NanoVDB loader always loads grid index 0
- NanoVDBSystem::LoadFromFile reads grid index 0 only.
Implication: If a file contains multiple grids (density/temperature/velocity), index 0 may not be density.

4) Scale/transform issues are secondary but common
- Blender Z-up vs engine Y-up, plus small bounds, can require scaling.
- PlasmaDX already has UI controls for scaling/centering bounds.

5) Tooling/docs inconsistencies exist
- scripts/inspect_vdb.py in PlasmaDX-Clean is a Blender-side bpy helper, not a CLI inspector.
- Some docs reference missing/moved files (e.g., docs/BLENDER_PLASMADX_WORKFLOW_SPEC.md actually exists under docs/blender_recipes/explosions/).

Save findings reference doc:
- docs/NanoVDB/BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md

---

### Goals (definition of done)

Primary goal A — Fix Blender 5 assets not rendering
- Make at least one Blender 5-exported volumetric asset (static and animated sequence) render reliably in PlasmaDX-Clean with correct density.

Primary goal B — World-space population + menu controls
Implement a world-space volumetric celestial body system that supports:
- Multiple NanoVDB volumes simultaneously (not just one global grid)
- Per-volume transforms (position/scale; rotation optional)
- Per-volume render parameters
- Per-volume animation sequences
- A menu/control scheme (ImGui acceptable) to manage volumes

---

### Key files to read first (authoritative)
- docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md
- docs/NanoVDB/VDB_PIPELINE_AGENTS.md
- docs/NanoVDB/BLENDER_NANOVDB_SESSION_PROMPT.md
- src/rendering/NanoVDBSystem.cpp
- shaders/volumetric/nanovdb_raymarch.hlsl
- shaders/nanovdb/PNanoVDB.h
- scripts/convert_vdb_to_nvdb.py
- src/core/Application.cpp (NanoVDB ImGui UI)

---

### Workstreams (parallelize aggressively)

Workstream 1 — Repro + diagnostics (highest priority)
- Establish 3 test cases:
  1) Procedural fog sphere renders
  2) Known repo .nvdb renders (e.g., VDBs/NanoVDB/cloud_01.nvdb, chimney_smoke/)
  3) Blender 5-produced .nvdb fails (current bug)
- For case (3), capture:
  - Loader logs: grid name, grid type, bounds, voxel size, active voxel count
  - Shader debug output (cyan/green/magenta)
- Produce a root-cause report with evidence.

Workstream 2 — Fix file-loaded grids rendering reliably
Implement fixes in PlasmaDX-Clean:
- Grid selection: stop hardcoding index 0; select by name (prefer density) and/or expose UI dropdown
- Grid type support: either
  - Add shader support for HALF/FP16 grids using PNanoVDB helpers, OR
  - Enforce conversion to FLOAT + hard error if not FLOAT
- Improve error messaging and ImGui readouts (grid name/type)

Acceptance criteria:
- A Blender-exported VDB with Precision=Half can be converted and still renders (by shader support or enforced conversion).
- A multi-grid .nvdb renders the density grid correctly.

Workstream 3 — Blender pipeline stabilization
- Use blender-manual MCP to confirm authoritative Blender 5 settings:
  - OpenVDB cache format, compression options, precision behavior
- Produce a “compatibility matrix”: Blender precision → resulting NanoVDB grid type → engine handling
- Ensure WSL/Windows headless automation is reproducible via assets/blender_scripts/GPT-5.2/run_blender_cli.sh
- Update docs inside PlasmaDX-Clean to match reality.

Workstream 4 — World-space multi-volume system + menu controls
- Implement NanoVDBVolumeInstance (per-volume parameters, transform, animation state)
- Implement NanoVDBVolumeManager (list of instances, render all enabled)
- Extend ImGui: add/remove instances, load file/sequence, per-instance params + playback

Acceptance criteria:
- At least 3 volumes can be placed simultaneously with independent controls.
- At least 1 volume plays an animation while others remain static.

---

### Explicit hypotheses to validate early
1) Blender precision Half/Mini → HALF/FP16 grids → shader rejects (gridType != FLOAT).
2) Multi-grid file → density not at index 0.
3) Axis/scale mismatch (secondary; should show as AABB miss or tiny volume).

---

### Output format
1) Phased plan with milestones and time estimates
2) Proposed code changes (PlasmaDX-Clean only)
3) Commit strategy (small, revert-friendly)
4) Validation checklist (steps a human can run)
5) Updated docs reflecting final pipeline

```


