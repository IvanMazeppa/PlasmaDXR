# Worklog: API Fixer Render/Bake Stability (2026-01-29)

## Context
Recent regressions in the Blender pipeline were traced to two main issues:
1) Full-frame render loops were not being constrained, causing long headless runs.
2) Liquid inflow bakes were producing near-empty caches unless `use_plane_init` was explicitly enabled.

This work focused on confirming the root causes in headless Blender 5.0 and updating the API fixer to prevent recurrence.

## What We Tested (Headless)
We ran multiple headless tests using a script derived from:
`build/vfx_output/manual_waver_overflow_user_test/water_overflow_v1_iter2_pattern_errfix.py`

Key tests:
- **Bake injection without context**: failed with `bpy.ops.fluid.bake_all.poll()` due to missing active object.
- **Bake injection + context override**: succeeded but produced small cache output (~0.8MB).
- **Bake injection + context override + forced `use_plane_init=True`**: succeeded with larger cache output (~4.4MB), proving `use_plane_init` materially increases bake data for liquid flows.

These results matched user observations from Blender UI tests (Is Planar being the only required change to get a functional liquid bake).

## API Fixer Changes
The API fixer now proactively injects `use_plane_init=True` for **any** liquid flow emitter.

### Why
- Blender 5.0’s Mantaflow liquid inflows can silently bake near-empty caches if `use_plane_init` is left at its default (False).
- The original logic only injected this setting when a plane primitive was detected (`primitive_plane_add`).
- In practice, many scripts use mesh emitters (e.g., nozzle cylinders), so the guard prevented injection.

### What Changed
- **Removed plane-only guard** so any `flow_type == 'LIQUID'` gets `use_plane_init=True` injected.
- **Updated the fix description text** to reflect the broader applicability.

### Location
- `agents/blender-vfx-orchestrator/tools/blender_api_fixer.py`
  - `_inject_plane_init_for_liquid_flow` now applies to all liquid flows, not only plane emitters.

## Current Status
- Render-loop limiting is already robust in the API fixer (covers common for-loop and range patterns).
- `use_plane_init` injection is now generalized, preventing silent empty-bake failures in headless runs.

## Notes / Follow-ups
- Cache sizes in headless will still vary with domain resolution, frame count, mesh output toggles, etc.
- If needed, bake operator context handling can be further hardened (ensure domain is active + object mode).

