# Flow
DocType: manual-rewritten
DocPath: physics/fluid/type/flow.html
DocVersion: 5.0.1
Model: gpt-5-mini
OriginalWords: 2387
RewrittenWords: 919
CompressionRatio: 0.39
---

One-line summary: Definitions and dense reference for Flow objects that add/remove/convert fluid inside a domain bounding box (Smoke/Fire/Liquid), their parameters, emission methods, and velocity inheritance.

## General
- Flow objects must be contained within the domain Bounding Box to affect the domain.
- Multiple Flow objects may be placed inside a domain.
- Geometry flow uses the object’s actual mesh geometry as fluid (unlike domain objects).
- NOTE: For Geometry flow, surface normals must point outwards or simulation will be incorrect.

## Settings
- `Flow Type` (enum, range not specified, default: not specified) — Smoke | Fire + Smoke | Fire | Liquid. Smoke/Fire options: domain may automatically create smoke from burnt fuel. Liquid emits liquid.
- `Flow Behavior` (enum, range not specified, default: not specified) — Inflow | Outflow | Geometry.
  - Inflow → object emits fluid into the domain (e.g., faucet or fire base).
  - Outflow → any fluid entering the object's Bounding Box is removed from the domain (useful as drain or to prevent full-domain fill). Outflow can be animated; removal area follows the object.
  - Geometry → mesh regions inside the domain bounding box become fluid (mesh geometry used directly).
- `Use Flow / Enables or disables the flow` (boolean, range not specified, default: not specified) — enable/disable emission; useful for animating when fluid is added/removed.
- `Sampling Substeps` (int, range not specified, default: not specified) — number of sub-steps per simulation step to reduce gaps when sources move fast. Higher → fewer emission gaps but more computation.
  - NOTE: Sub-steps occur at every simulation step, not per frame. Simulation step count is controlled by adaptive time stepping.
- `Smoke Color` (color, range not specified, default: not specified) — emitted smoke color; mixing differently colored smoke → blended resultant color.
- `Absolute Density` (boolean, range not specified, default: not specified) — when enabled, emitter produces additional smoke/fire only if there is space in the emitter region; when disabled, emission always produces and accumulates.
- `Initial Temperature` (float, range not specified, default: not specified) — temperature difference between emitted smoke and domain ambient. Effect on smoke depends on domain Heat Buoyancy.
- `Density` (float, range not specified, default: not specified) — amount of smoke emitted at once. Larger → more density produced.
- `Fuel` (float, range not specified, default: not specified) — amount of "fuel" burned per second. Larger → larger flames; smaller → smaller flames. (Example values shown in source: 0.5 and 1.0.)
- `Vertex Group` (identifier, range not specified, default: none) — when set, emission is controlled by the specified vertex group.

## Flow Source (emission methods)
- `Flow Source` (enum, range not specified, default: not specified) — Mesh | Particle System | (others not specified).
- `Mesh` (method) — emit fluid directly from the object mesh.
  - `Is Planar` (boolean, range not specified, default: not specified) — marks effector as a single-dimension object (plane) or the mesh is non-manifold; informs simulator to treat these meshes for more accurate results.
  - `Surface Emission` (float in voxels, range not specified, default: not specified) — maximum distance in voxels from mesh surface where fluid is emitted. Uses domain voxels → results vary with domain resolution.
  - `Volume Emission` (float, range [0, 1], default: not specified) — Fire/Smoke only: fraction of fluid emitted inside the emitter mesh where 0 = none, 1 = full. NOTE: Volume-based emission can be unpredictable for non-manifold meshes.
- `Particle System` (method, Fire/Smoke only) — create smoke/fire from a particle system on the flow object.
  - `Particle System` (Data ID, range not specified, default: not specified) — select an Emitter-type particle system to emit smoke/fire. NOTE: Only Emitter-type particle systems can add smoke.
  - `Set Size` (boolean, range not specified, default: not specified) — when enabled, the `Size` parameter defines the maximum distance in voxels at which particles can emit smoke (similar to Surface Emission); when disabled, particles fill the nearest voxel with smoke.
  - `Size` (float in voxels, range not specified, default: not specified) — maximum distance in voxels for particle emission when `Set Size` is enabled.

## Initial Velocity (inheritance)
- `Initial Velocity` (boolean, range not specified, default: not specified) — when enabled, emitted fluid inherits momentum from the flow source.
- `Source Factor` (float, range not specified, default: not specified) — multiplier for inherited velocity. Value = 1 → emitted fluid moves at the same speed as the source. Higher → proportionally faster inherited motion.
- `Normal` (parameter name appears; description truncated in source) — range not specified, default not specified.
  - NOTE: source text truncates description for `Normal`; source-provided behavior beyond this point is not available.

## TECHNIQUE
- For animated emission toggling, animate the Flow enable/disable property rather than moving domain parameters.
- Use Outflow paired with Inflow to prevent domain overfill.
- Use Sampling Substeps > 0 for fast-moving inflows to avoid gaps; tradeoff: more substeps → more compute.
- For planar or non-manifold meshes, enable `Is Planar` so the simulator treats them appropriately.
- Prefer Surface Emission for mesh sources when you need emission offset from geometry; be aware results scale with domain resolution (voxels).
- For particle-driven smoke/fire, enable `Set Size` when particles represent volumetric emitters; disable to confine emission to nearest voxel.

## NOTES / GOTCHAS
- Flow objects must lie inside the domain Bounding Box to be effective.
- Geometry flow uses mesh geometry directly; ensure outward normals.
- Volume Emission and non-manifold geometry → unpredictable emission behavior.
- Sub-steps operate per simulation step; overall simulation steps depend on adaptive time stepping.