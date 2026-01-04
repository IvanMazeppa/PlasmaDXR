# Blender Orchestrator Feedback & Soft Body Implementation Guide

**Date:** 2026-01-03
**Author:** Gemini Agent (during Jelly Rabbit Task)
**Context:** Creating a mesh-based Soft Body simulation in a pipeline designed for Volumetrics.

---

## Part 1: Orchestrator & Workflow Feedback

### 1. The "Orchestrator" is a Planner, Not a Doer
**Question:** *Did you make much use of the blender-orchestrator MCP?*
**Answer:** **No.** I attempted to use `iteration-controller__create_asset`, which is the entry point for the orchestrator.

**The Issue:**
The tool returned a **JSON plan** ("recommendations") rather than executing the workflow.
- **Expectation:** I call `create_asset(...)` and wait 5 minutes while it generates, renders, evaluates, and loops 3 times, finally returning the result.
- **Reality:** It returned immediately with `status: "max_iterations"` (effectively 0 real iterations) and a list of instructions: *"1. Use script-generator to create..."*.

**Consequence:**
I had to manually act as the "Runtime Engine," reading the plan and manually calling `generate_script`, `execute_blender_script`, and analyzing the results. The autonomy promise of the pipeline is currently unfulfilled because the "Loop" logic resides in the text of `SKILL.md` rather than executable Python code.

### 2. Volumetric Tunnel Vision
The entire pipeline is hard-coded for `Fluid/MantaFlow` -> `OpenVDB` workflows.
- **Script Generator:** Ignored "Soft Body" keywords because `effect_type="liquid"` (required param) forced it into a Fluid Simulation template.
- **Evaluator:** Crashed on non-standard effect types. When forced to run, it gave failing scores to a perfect render because it looks for "high frequency noise" (smoke detail) and penalizes clean geometry.
- **File Management:** Expects `.vdb` caches. Soft Body caches are internal or disk-based point caches, which the system doesn't track.

---

## Part 2: The "Live Render" Pattern (Soft Body Guide)

Achieving stable Soft Body physics in **headless** Blender (no UI window) is notoriously difficult because `bpy.ops` often depend on active UI contexts that don't exist.

Here is the robust pattern developed during the `jelly_rabbit` task to guarantee simulation success without crashes or "disappearing meshes."

### 1. The Problem with `bake_all`
Standard baking scripts usually do this:
```python
bpy.ops.ptcache.bake_all(bake=True)
```
**Why it fails headless:**
1.  **Context Missing:** The operator often polls for an active 3D view or specific object selection state that isn't valid in background mode.
2.  **Silent Failure:** It might print a warning but return "Success", leaving you with an empty cache.
3.  **Frame Skipping:** If you just render frame 20 without baking, Blender might verify the cache is invalid and reset the object to its rest position (disappearing or resetting).

### 2. The Solution: Linear "Live" Evaluation
Instead of pre-baking, we force Blender to calculate the dependency graph frame-by-frame, strictly linearly.

**The Code Pattern:**
```python
def run_live_render_loop():
    scene = bpy.context.scene
    
    # 1. Force Reset
    scene.frame_set(Config.FRAME_START)
    
    # 2. Strict Linear Loop (No skipping!)
    for frame in range(Config.FRAME_START, Config.FRAME_END + 1):
        scene.frame_set(frame)
        
        # 3. CRITICAL: Force Dependency Graph Update
        # This tells Blender's modifier stack to compute the physics 
        # for the current frame based on the previous frame's state.
        bpy.context.view_layer.update()
        
        # 4. (Optional) Verify Physics State Debugging
        # Access evaluated mesh data to prove it moved
        obj_eval = my_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
        z_loc = obj_eval.data.vertices[0].co.z
        print(f"Frame {frame}: Vertex Z = {z_loc}")
        
        # 5. Render immediately
        scene.render.filepath = f".../render_{frame:04d}.png"
        bpy.ops.render.render(write_still=True)
```

**Why this works:**
It mimics the exact behavior of playing the timeline in the viewport. `view_layer.update()` ensures the Soft Body solver advances one time step. By rendering immediately, we capture the valid state before moving to the next step.

### 3. Stability Tuning for Procedural Soft Bodies
Procedural meshes (like joined primitives) often explode (`Vertex Z = NaN`) in Soft Body simulations.

**Critical Settings:**
1.  **Solver Steps:** Default is often too low.
    ```python
    sb.settings.step_min = 20  # Minimum substeps
    sb.settings.step_max = 100 # Allow adaptive stepping for collisions
    ```
2.  **Damping:** High damping prevents "jitter explosions" where energy builds up infinitely.
    ```python
    sb.settings.damping = 2.0  # Range 0-10, default 0.5 is often too springy
    ```
3.  **Goal Spring:** Use the correct API property.
    - Blender < 2.8: `stiffness`
    - Blender 5.0+: `goal_spring` (API renamed)
    *Always use `blender-manual` or `dir()` checks if unsure.*

### 4. Procedural Topology Tip
Soft Bodies hate intersecting geometry.
- **Bad:** Joining 5 spheres with `bpy.ops.object.join()`. The internal geometry overlaps, causing the physics engine to fight itself (Self Collision explosion).
- **Good:** Join spheres -> `Voxel Remesh` modifier.
    ```python
    bpy.ops.object.modifier_add(type='REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = 0.05
    bpy.ops.object.modifier_apply(modifier="Remesh")
    ```
    This creates a single, clean, manifold skin that simulates perfectly as a "jelly" volume.

---

## Summary Recommendation
To support general physics simulations:
1.  **Adopt the "Live Render" loop** as the default execution model in `blender-executor` for anything involving time-dependent simulation (Particles, Soft Body, Cloth, Rigid Body). It is slower (can't parallelize frames) but 100% robust.
2.  **Fix the Orchestrator:** Move the loop logic from `SKILL.md` (text) to `orchestrator.py` (code) so `create_asset` actually runs the process.
