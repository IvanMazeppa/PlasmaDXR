# Gemini Feedback Analysis & Plan Amendments

**Date:** 2026-01-03
**Source:** Gemini 3 Pro (Jelly Rabbit Task)
**Target:** System Architects / Agent Developers

This document formalizes the bugs, architectural gaps, and improvement recommendations discovered during the manual execution of the "Jelly Rabbit" Soft Body simulation task.

---

## 1. Critical Bug Reports

### 1.1 Asset Evaluator JSON Serialization Crash
**Severity:** High
**Component:** `agents/asset-evaluator/server.py`
**Issue:** The tool `extract_vfx_diagnostics` crashed with `TypeError: Object of type bool is not JSON serializable`.
**Root Cause:** The image processing library (likely NumPy or OpenCV) returns `numpy.bool_` or `numpy.float32` types, which the standard Python `json` library cannot serialize.
**Fix:** Implement a custom JSON encoder or a recursive type conversion utility in the MCP server to convert NumPy types to native Python types (`bool`, `float`, `int`) before returning results.

### 1.2 Headless Baking Context Failure
**Severity:** High
**Component:** `blender-executor` / `script-generator`
**Issue:** Scripts using `bpy.ops.ptcache.bake_all(bake=True)` fail silently or exit with errors when run in headless mode (`--background`).
**Root Cause:** Many Blender operators depend on an active UI context (3D Viewport, Properties Panel) which does not exist in background mode.
**Fix:** 
1.  **Short Term:** Use `bpy.context.temp_override` (Blender 3.2+) to mock the context, though this is still fragile for physics.
2.  **Long Term (Recommended):** Adopt the **"Live Render" Pattern** (see below) which bypasses operators entirely.

---

## 2. Architectural Gaps

### 2.1 The "Orchestrator" Execution Gap
**Issue:** The `iteration-controller__create_asset` tool acts as a *planner*, returning a list of recommendations, but does not *execute* the iteration loop.
**Impact:** The system is not truly autonomous; it relies on the user (or chat agent) to manually read the plan and call the subsequent tools (`generate`, `execute`, `evaluate`).
**Recommendation:** 
- Move the loop logic from the text-based `SKILL.md` into a Python-based state machine within `agents/blender-orchestrator/orchestrator.py`.
- The `create_asset` tool should block (or use async polling) while it internally manages the generate-execute-evaluate loop, returning only the final result or a stream of updates.

### 2.2 Volumetric Tunnel Vision
**Issue:** The pipeline is over-optimized for Fluid/Smoke simulations.
- `script-generator` templates force `FluidDomain` setup.
- `asset-evaluator` metrics (Edge Density, Noise) penalize clean mesh geometry.
**Recommendation:**
- Introduce a **Simulation Type Registry**:
    - `VOLUMETRIC`: Pyro, Clouds (Current pipeline)
    - `MESH_PHYSICS`: Soft Body, Cloth, Rigid Body (New pipeline)
- Update `generate_script` to select templates based on this registry.

---

## 3. The "Live Render" Execution Pattern

To robustly support Soft Body, Cloth, and Particle simulations in headless mode, the `blender-executor` and `script-generator` must support a new execution pattern.

**Current Pattern (Bake & Export):**
1. Setup Scene
2. `bpy.ops.fluid.bake_all()`
3. Export `.vdb`
4. Render from cache

**Proposed Pattern (Live Render):**
Used for Mesh Physics where VDB caching is not applicable or reliable.
1. Setup Scene
2. **Force Linear Stepping:** Iterate frames 1 to End.
3. **Update Dependency Graph:** Call `bpy.context.view_layer.update()` at each frame.
4. **Render Immediately:** Save the frame while the physics state is valid in memory.
5. **No Disk Cache:** Explicitly disable disk caching to prevent read/write conflicts.

**Implementation Plan:**
- Modify `script-generator` to inject this "Live Render" loop code for non-volumetric templates.
- Update `blender-executor` to recognize that "Live Render" scripts might not produce `.vdb` files but valid `.png` sequences.

---

## 4. Evaluation Improvements for Meshes

The current `asset-evaluator` gives low scores to good mesh renders because it expects high-frequency noise (smoke).

**New Metrics Needed:**
1.  **Silhouette Stability:** For checking flickering or exploding meshes.
2.  **Surface Smoothness:** Penalize jagged artifacts (unless requested).
3.  **Motion Consistency:** Check that center-of-mass moves smoothly (no teleporting/NaNs).

**Action:** Update `asset-evaluator` to accept an `asset_category` parameter ("volumetric" vs "mesh") and apply appropriate scoring weights.