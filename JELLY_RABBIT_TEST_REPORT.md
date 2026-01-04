# Jelly Rabbit Workflow Test Report

**Date:** 2026-01-03
**Asset:** Jelly Rabbit (Soft Body Mesh)
**Status:** Manual Success (Iteration 5) / Automated Failure

## Executive Summary
The test successfully produced a rendering of a jelly rabbit using Soft Body physics, but required significant manual intervention (5 iterations). The autonomous pipeline (`create_asset`) is currently non-functional for this use case. The system is heavily over-optimized for volumetric/pyro effects, lacking flexibility for mesh-based simulations.

## Key Findings

### 1. Autonomous Loop Failure
The `iteration-controller__create_asset` tool does not execute the pipeline. It runs for 0.0 seconds and returns a set of "recommendations" describing what the agent *should* do, rather than performing the actions.
- **Impact:** The "fully autonomous" mode described in documentation is effectively a prompt generator, not an agent runtime.
- **Fix Required:** The loop logic needs to be moved from the Skill definition into the actual Python code of `iteration-controller` or `blender-orchestrator` to be executable via tool call.

### 2. Script Generator Limitations
The `script-generator` ignored explicit instructions ("Use Soft Body physics... NOT a fluid simulation") and defaulted to a template based on the `effect_type="liquid"` parameter.
- **Behavior:** Generated a standard Mantaflow fluid script with a primitive cube/sphere.
- **Root Cause:** Appears to prioritize `effect_type` template selection over LLM-based code generation from the description.
- **Critique:** The system lacks a "Generic" or "Custom" mode for effects that don't fit the pre-defined pyro/nebula categories.

### 3. Evaluation Toolkit Bias
The `asset-evaluator` is strictly tuned for volumetrics.
- **Invalid Type:** `effect_type="liquid"` caused a crash in `evaluate_vfx_quality` (valid types: supernova, smoke, fire, etc.).
- **Scoring Mismatch:** When forced to use `nebula`, the mesh render scored 56/100 (FAIL) with "NO STRUCTURE" and "LOW CONTRAST". The tool penalizes the clean lines/smooth shading of a mesh, expecting high-frequency noise typical of smoke/fire.
- **Bug:** `extract_vfx_diagnostics` crashed with `Object of type bool is not JSON serializable` (likely a numpy type issue).

### 4. Blender API Gaps
The generated/manual scripts initially failed due to API mismatches with Blender 5.0 (e.g., `SoftBodySettings.stiffness` vs `goal_spring`).
- **Positive:** The `blender-executor` correctly identified the error and suggested checking the manual.
- **Positive:** The `blender-manual` tool (via my manual usage) correctly identified the new property names.

### 5. Final Success (Iteration 5)
After several attempts, a robust script (`jelly_rabbit_v5.py`) was created that solved all simulation issues:
- **Procedural Rabbit:** Created using joined geometric primitives (spheres) and Voxel Remesh to create a single manifold skin, replacing the Suzanne monkey.
- **Simulation Stability:**
    - **Fixed "Disappearing Mesh":** Abandoned `bpy.ops.ptcache.bake_all` (unreliable headless) in favor of a "Live Render" loop that explicitly calls `bpy.context.view_layer.update()` per frame.
    - **Fixed "Explosion/NaN":** Increased Solver Steps (Min 20, Max 100) and Damping (2.0) to stabilize the soft body calculation.
    - **Fixed "Floor Penetration":** Adjusted Collision margin and initial height.
- **Verification:** Added debug logging of Vertex Z positions to stdout, confirming stable motion (2.8m -> 0.09m) without NaN values.

## Successful Output
A manual override script was created to achieve the visual goal:
- **Technique:** Soft Body Modifier on procedural mesh.
- **Material:** Translucent Red (Transmission 0.85, IOR 1.33).
- **Physics:** Goal Spring 0.3, Damping 2.0, Linear Stepping.
- **Result:** Successfully rendered 50 frames of animation where the rabbit falls, hits the floor, and jiggles stably.

## Recommendations

1.  **Implement Generic Script Generation:** Add a `custom` or `general` effect type that bypasses templates and relies 100% on the LLM to write the script from the description.
2.  **Fix JSON Serialization:** Patch `asset-evaluator/server.py` to handle numpy booleans/floats before JSON serialization.
3.  **Expand Evaluation Types:** Add a `mesh` or `general` mode to `evaluate_vfx_quality` that doesn't penalize smooth gradients or lack of noise.
4.  **Operationalize the Loop:** Convert the `SKILL.md` logic into a Python-based state machine in `iteration-controller` so the `create_asset` tool actually runs the loop.
5.  **Robust Simulation Pattern:** Update the script generator to use the "Live Render with explicit `view_layer.update()`" pattern for physics simulations, as headless baking is too fragile.
