#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with mixed/combined physics system knowledge.

Creates LLM-optimized markdown documents describing how to combine multiple Blender 5.0
physics systems (rigid body, fluid, cloth, particles, force fields) in a single scene.
Uploads them to the rewritten manual vector store. Also seeds the code pattern memory
with proven multi-physics snippets.

The pipeline struggles when scenes require multiple physics systems working together.
These docs teach the Research Agent and Script Writer how to coordinate bake order,
frame synchronization, and cross-system interactions.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_mixed_physics_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_mixed_physics_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_mixed_physics_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_mixed_physics_techniques.py --docs-only
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ORCHESTRATOR_ROOT))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================
# TECHNIQUE DOCUMENTS — LLM-optimized, dense, factual
# ============================================================

TECHNIQUE_DOCS = {
    "mixed_physics_overview.md": """# Combining Multiple Physics Systems in Blender 5.0
DocType: technique-guide
DocPath: techniques/mixed_physics/overview
DocVersion: 5.0.1
PhysicsDomain: mixed
---

## Overview

Blender 5.0 supports running multiple physics systems simultaneously in one scene.
Each system has its own solver, cache, and evaluation order. Combining them requires
understanding dependencies, bake order, and frame synchronization.

## Physics Systems in Blender 5.0

| System | Module | Per-Object | Domain Required |
|--------|--------|------------|-----------------|
| Rigid Body | `bpy.types.RigidBodyObject` | Yes | RigidBodyWorld (scene-level) |
| Mantaflow Fluid (gas/liquid) | `bpy.types.FluidDomainSettings` | Yes | Fluid Domain object |
| Cloth | `bpy.types.ClothModifier` | Yes | No |
| Particles | `bpy.types.ParticleSystem` | Yes | No |
| Soft Body | `bpy.types.SoftBodyModifier` | Yes | No |
| Force Fields | `bpy.types.FieldSettings` | Yes (effector) | No |

## Coexistence Rules

**Same object:**
- Rigid body + particle emitter: YES (particles emit from rigid body surface)
- Cloth + collision: NO (cloth IS a simulation, collision is for other sims to interact with it)
- Fluid domain + rigid body: NO (domain is the bounding box, not a physics object)
- Fluid flow + rigid body: NO (flow emits fluid, rigid body moves it — conflicting)
- Force field + rigid body: YES (object acts as both effector and rigid body)
- Particle emitter + cloth: NO (use separate objects)

**Separate objects (always safe):**
- Any combination works on separate objects as long as bake order is respected.

## Bake Order (CRITICAL)

Bake physics systems in dependency order. Later systems can reference earlier bakes
but NOT vice versa.

```
1. Rigid Body     — bpy.ops.ptcache.bake_all(bake=True) or per-object
2. Fluid (liquid/gas) — bpy.ops.fluid.bake_all()
3. Particles      — bpy.ops.ptcache.bake_all(bake=True) (after rigid body)
4. Cloth          — bpy.ops.ptcache.bake_all(bake=True) (after colliders settle)
```

If fluid needs to interact with rigid body obstacles, bake rigid body FIRST, then
set rigid body objects as fluid effectors and bake fluid.

## Frame Synchronization

All physics systems share `scene.frame_start` and `scene.frame_end`. Keep them consistent:

```python
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# Rigid body world uses scene frame range automatically
# Fluid domain has its own cache_frame_start/cache_frame_end — MATCH THEM
if fluid_domain:
    dset = fluid_domain.modifiers["Fluid"].domain_settings
    dset.cache_frame_start = 1
    dset.cache_frame_end = 120

# Particles use frame_start/frame_end per-system
# Cloth uses point cache frame range
```

## Performance Implications

| Combination | Performance Impact | Recommendation |
|-------------|-------------------|----------------|
| Rigid body + particles | LOW | Safe for most scenes |
| Rigid body + Mantaflow gas | MEDIUM | Keep fluid resolution_max <= 128 |
| Rigid body + Mantaflow liquid | HIGH | Keep resolution_max <= 96, use MESH cache |
| Cloth + wind forces | LOW | Safe, increase cloth quality_steps for stability |
| Fluid + cloth | VERY HIGH | Avoid unless essential; bake separately |
| 3+ systems combined | HIGH | Bake sequentially, monitor cache sizes |

## Decision Guide: Common Multi-Physics Scenarios

| Scene | Systems Needed | Approach |
|-------|---------------|----------|
| Explosion with debris | Rigid body + Mantaflow gas + particles | RB for shards, gas for fire/smoke, particles for sparks |
| Waterfall over rocks | Mantaflow liquid + rigid body effectors | RB rocks as PASSIVE effectors in fluid domain |
| Flag in storm | Cloth + force fields + particles | Cloth for flag, wind+turbulence forces, particles for rain |
| Car crash | Rigid body + particles + Mantaflow gas | RB for body panels, particles for glass, gas for engine fire |
| Underwater scene | Mantaflow liquid + rigid body + particles | Liquid domain, RB floating objects, particles for bubbles |

NOTE: `bpy.ops.object.effector_add()` NOT `forcefield_add` (removed in Blender 5.0).
NOTE: `substeps_per_frame` NOT `steps_per_second` for RigidBodyWorld.
NOTE: `resolution_max` NOT `resolution_divisions` for fluid domains.
NOTE: `use_dissolve_smoke` NOT `use_dissolve` for gas domains.
""",

    "technique_rigid_body_plus_particles.md": """# Rigid Body + Particle Debris Combination
DocType: technique-guide
DocPath: techniques/mixed_physics/rigid_body_particles
DocVersion: 5.0.1
PhysicsDomain: mixed
---

## Concept

Rigid body handles primary large objects (boulders, shards, vehicles). Particle systems
handle secondary effects: sparks, dust clouds, small debris chips. This is the most
common multi-physics combination and the easiest to set up.

## Architecture

```
Rigid Body World
  ├── Primary objects (ACTIVE rigid bodies)
  ├── Ground/walls (PASSIVE rigid bodies with collision)
  └── Particle emitters (can be rigid body objects OR separate empties)

Particle Systems
  ├── Debris emitter → small chip instances
  ├── Spark emitter → halo/line particles
  └── Dust emitter → volumetric or billboard particles
```

## Emitting Particles from Collision Events

Blender does not natively trigger particle emission from rigid body collisions.
Workaround: place particle emitters at known impact locations and time emission
to match the rigid body impact frame.

```python
import bpy
import random
from mathutils import Vector

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 90

# ====== RIGID BODY WORLD ======
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 8
rbw.solver_iterations = 15

# ====== GROUND PLANE ======
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
bpy.ops.rigidbody.object_add(type='PASSIVE')
ground.rigid_body.collision_shape = 'MESH'
ground.rigid_body.friction = 0.8

# Add collision modifier for particles
ground.modifiers.new("Collision", type='COLLISION')
ground.modifiers["Collision"].settings.damping = 0.6
ground.modifiers["Collision"].settings.friction = 0.5

# ====== FALLING BOULDERS ======
boulders = []
for i in range(5):
    x = random.uniform(-3, 3)
    y = random.uniform(-3, 3)
    z = random.uniform(8, 15)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5, location=(x, y, z))
    boulder = bpy.context.active_object
    boulder.name = f"Boulder_{i}"
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    boulder.rigid_body.mass = 50.0
    boulder.rigid_body.collision_shape = 'CONVEX_HULL'
    boulder.rigid_body.friction = 0.7
    boulder.rigid_body.restitution = 0.2
    boulder.rigid_body.linear_damping = 0.04
    boulder.rigid_body.angular_damping = 0.1
    boulders.append(boulder)

# ====== SPARK PARTICLES (metal-on-metal collision) ======
# Emitter at expected impact zone
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0.05))
spark_emitter = bpy.context.active_object
spark_emitter.name = "SparkEmitter"
spark_emitter.hide_render = True

ps_mod = spark_emitter.modifiers.new("Sparks", type='PARTICLE_SYSTEM')
spark_settings = ps_mod.particle_system.settings
spark_settings.count = 500
spark_settings.frame_start = 10       # Approximate first impact frame
spark_settings.frame_end = 50         # Cover impact window
spark_settings.lifetime = 15
spark_settings.lifetime_random = 0.5
spark_settings.physics_type = 'NEWTON'
spark_settings.mass = 0.001
spark_settings.normal_factor = 12.0   # Fast upward burst
spark_settings.factor_random = 5.0
spark_settings.tangent_factor = 4.0
spark_settings.damping = 0.05         # Low damping — sparks travel far
spark_settings.particle_size = 0.002
spark_settings.render_type = 'HALO'   # Glowing point particles
spark_settings.use_die_on_collision_with = True

# ====== DUST CLOUD PARTICLES ======
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0.02))
dust_emitter = bpy.context.active_object
dust_emitter.name = "DustEmitter"
dust_emitter.hide_render = True

ps_mod2 = dust_emitter.modifiers.new("Dust", type='PARTICLE_SYSTEM')
dust_settings = ps_mod2.particle_system.settings
dust_settings.count = 300
dust_settings.frame_start = 10
dust_settings.frame_end = 55
dust_settings.lifetime = 60
dust_settings.lifetime_random = 0.3
dust_settings.physics_type = 'NEWTON'
dust_settings.mass = 0.0001           # Very light dust
dust_settings.normal_factor = 3.0     # Slow upward drift
dust_settings.factor_random = 2.0
dust_settings.damping = 0.5           # High damping — dust hangs in air
dust_settings.particle_size = 0.05
dust_settings.size_random = 0.8
dust_settings.render_type = 'HALO'

# ====== BAKE ORDER ======
# 1. Bake rigid body first
bpy.ops.ptcache.bake_all(bake=True)
# Particles bake automatically with ptcache.bake_all
```

## Sparks from Metal-on-Metal

For sparks, use HALO render type with emission shader:

```python
# Spark material — emissive orange/yellow
mat = bpy.data.materials.new("M_Spark")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (1.0, 0.6, 0.1, 1)  # Hot orange
emit.inputs["Strength"].default_value = 50.0
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
```

## Common Pitfalls

NOTE: Particles do NOT automatically detect rigid body collisions. Time emission manually.
NOTE: `bpy.ops.ptcache.bake_all()` bakes BOTH rigid body and particle caches.
NOTE: Ground needs BOTH `rigid_body` (for boulders) AND `collision` modifier (for particles).
NOTE: `use_die_on_collision_with` requires collision modifier on target objects.

TECHNIQUE: For impact-synchronized sparks, estimate the impact frame from the drop height
using `t = sqrt(2h/g)` where g=9.81 and h is drop height in meters. Set particle
`frame_start` to `1 + int(t * fps)`.
""",

    "technique_fluid_plus_rigid_body.md": """# Fluid + Rigid Body Interaction
DocType: technique-guide
DocPath: techniques/mixed_physics/fluid_rigid_body
DocVersion: 5.0.1
PhysicsDomain: mixed
---

## Concept

Rigid body objects act as obstacles (effectors) inside a Mantaflow fluid domain. The
fluid flows around, over, and collides with rigid body geometry. This creates waterfalls
over rocks, water around pillars, waves hitting walls, etc.

## Architecture

```
Fluid Domain (cube encompassing the scene)
  ├── Flow source (inflow emitter)
  ├── Effector objects (rigid body rocks/obstacles)
  └── Outflow (optional drain)

Rigid Body World
  ├── PASSIVE rigid bodies = static obstacles (rocks, walls)
  └── ACTIVE rigid bodies = floating objects pushed by fluid (advanced)
```

## Key Constraint: Rigid Body as Fluid Effector

A rigid body object can ALSO be a fluid effector. Set the fluid modifier to EFFECTOR
type on the same object that has rigid body physics:

```python
import bpy
from mathutils import Vector

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# ====== FLUID DOMAIN ======
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1.5))
domain_obj = bpy.context.active_object
domain_obj.name = "FluidDomain"
domain_obj.scale = (4, 3, 3)
bpy.ops.object.transform_apply(scale=True)

mod = domain_obj.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
dset = mod.domain_settings
dset.domain_type = 'LIQUID'
dset.resolution_max = 96              # Balance quality vs bake time
dset.use_mesh = True                  # Generate mesh surface
dset.cache_frame_start = 1
dset.cache_frame_end = 120
dset.cache_type = 'ALL'               # Full bake (NOT REPLAY for multi-physics)
dset.use_adaptive_timesteps = True
dset.timesteps_max = 8
dset.use_flip_particles = True
dset.particle_scale = 1

# ====== WATER INFLOW ======
bpy.ops.mesh.primitive_plane_add(size=1.5, location=(-3, 0, 2.5))
inflow_obj = bpy.context.active_object
inflow_obj.name = "WaterInflow"

mod_flow = inflow_obj.modifiers.new("Fluid", type='FLUID')
mod_flow.fluid_type = 'FLOW'
fset = mod_flow.flow_settings
fset.flow_type = 'LIQUID'
fset.flow_behavior = 'INFLOW'
fset.use_inflow = True
fset.velocity_normal = 2.0            # Flow speed

# ====== RIGID BODY WORLD ======
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 5
rbw.solver_iterations = 10

# ====== ROCKS (Rigid Body + Fluid Effector) ======
rock_positions = [
    (-1, 0, 0.3), (0.5, -0.8, 0.5), (1.5, 0.5, 0.2),
    (0, 1.0, 0.4), (-0.5, -0.5, 0.6),
]
for i, pos in enumerate(rock_positions):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.4, location=pos)
    rock = bpy.context.active_object
    rock.name = f"Rock_{i}"

    # Deform into irregular rock shape
    for v in rock.data.vertices:
        v.co.x += (hash((i, v.index, 0)) % 100 - 50) * 0.002
        v.co.y += (hash((i, v.index, 1)) % 100 - 50) * 0.002
        v.co.z += (hash((i, v.index, 2)) % 100 - 50) * 0.002

    # Rigid body — PASSIVE (static obstacle)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    rock.rigid_body.collision_shape = 'MESH'
    rock.rigid_body.friction = 0.8

    # Fluid effector — makes fluid flow around this rock
    mod_eff = rock.modifiers.new("Fluid", type='FLUID')
    mod_eff.fluid_type = 'EFFECTOR'
    eset = mod_eff.effector_settings
    eset.effector_type = 'COLLISION'
    eset.surface_distance = 0.1

# ====== GROUND (catches pooling water) ======
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
bpy.ops.rigidbody.object_add(type='PASSIVE')
ground.rigid_body.collision_shape = 'MESH'

mod_ground = ground.modifiers.new("Fluid", type='FLUID')
mod_ground.fluid_type = 'EFFECTOR'
mod_ground.effector_settings.effector_type = 'COLLISION'

# ====== BAKE ORDER ======
# 1. Rigid body first (static rocks don't need baking, but if animated, bake first)
# 2. Fluid second — it reads rigid body positions
# For PASSIVE rigid bodies, no separate bake needed — they're static.
# Bake fluid:
bpy.ops.fluid.bake_all()
```

## Frame Range and Bake Coordination

Fluid and rigid body must share the same frame range:

```python
# Synchronize frame ranges
scene.frame_start = 1
scene.frame_end = 120

# Fluid domain cache MUST match scene range
dset.cache_frame_start = scene.frame_start
dset.cache_frame_end = scene.frame_end

# Rigid body world automatically uses scene frame range
```

## Animated Rigid Body Obstacles

For obstacles that move DURING fluid simulation (e.g., a gate opening):

```python
# Animated gate — PASSIVE rigid body with keyframed position
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1))
gate = bpy.context.active_object
gate.name = "Gate"
gate.scale = (0.1, 2, 1.5)
bpy.ops.object.transform_apply(scale=True)

bpy.ops.rigidbody.object_add(type='PASSIVE')
gate.rigid_body.kinematic = True       # Animated passive rigid body

# Keyframe gate lifting
gate.location = (0, 0, 1)
gate.keyframe_insert(data_path="location", frame=1)
gate.location = (0, 0, 3.5)           # Lifted up
gate.keyframe_insert(data_path="location", frame=40)

# Also make it a fluid effector
mod_gate = gate.modifiers.new("Fluid", type='FLUID')
mod_gate.fluid_type = 'EFFECTOR'
mod_gate.effector_settings.effector_type = 'COLLISION'
mod_gate.effector_settings.use_effector = True
```

## Common Pitfalls

NOTE: `resolution_max` NOT `resolution_divisions` for fluid domains.
NOTE: `use_adaptive_timesteps` NOT `use_adaptive_time_steps`.
NOTE: `cache_type = 'ALL'` required for reliable multi-physics bakes. `REPLAY` does NOT produce stable data for post-bake reads.
NOTE: Fluid effector objects MUST be inside the fluid domain bounding box to affect the simulation.
NOTE: `substeps_per_frame` NOT `steps_per_second` for RigidBodyWorld.
NOTE: Bake fluid AFTER rigid body. If rigid body positions change, fluid cache is invalid — rebake.

TECHNIQUE: For waterfall scenes, place the inflow above the rocks and let gravity + effector collisions create natural water paths. Set `velocity_normal` on the inflow to control water speed.
""",

    "technique_cloth_plus_forces.md": """# Cloth + Wind, Turbulence, and Collisions
DocType: technique-guide
DocPath: techniques/mixed_physics/cloth_forces
DocVersion: 5.0.1
PhysicsDomain: mixed
---

## Concept

Cloth simulation with multiple force fields creates realistic fabric behavior:
flags flapping in wind, banners in storms, curtains blowing, sails billowing.
Add collision objects for cloth to drape over or wrap around geometry.

## Force Field Types for Cloth

| Force Field | Effect on Cloth | Use Case |
|-------------|----------------|----------|
| Wind | Constant directional push | Flag blowing, sail filling |
| Turbulence | Random chaotic forces | Storm flapping, turbulent air |
| Vortex | Rotational force | Tornado, spinning debris |
| Force | Radial push/pull | Explosion blast on fabric |
| Drag | Air resistance | Parachute, slow settling |

## Creating Force Fields

```python
import bpy
from mathutils import Vector

# Wind force field
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 2))
wind = bpy.context.active_object
wind.name = "Wind"
wind.field.strength = 15.0           # Wind intensity
wind.field.flow = 0.5               # Airflow — adds inertia
wind.field.noise = 2.0              # Gustiness
wind.field.seed = 42                # Noise seed
wind.rotation_euler = (0, 1.5708, 0)  # Point along -X axis

# Turbulence force field
bpy.ops.object.effector_add(type='TURBULENCE', location=(3, 0, 2))
turb = bpy.context.active_object
turb.name = "Turbulence"
turb.field.strength = 8.0
turb.field.size = 1.5               # Scale of turbulent eddies
turb.field.noise = 3.0
turb.field.flow = 0.3
```

NOTE: Use `bpy.ops.object.effector_add()` NOT `forcefield_add` (does not exist in Blender 5.0).

## Cloth Setup for Flag/Banner

```python
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 250

# ====== FLAG MESH ======
bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 3))
flag = bpy.context.active_object
flag.name = "Flag"
flag.scale = (1.5, 1.0, 1.0)
bpy.ops.object.transform_apply(scale=True)

# Subdivide for cloth deformation
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=20)
bpy.ops.object.mode_set(mode='OBJECT')

# ====== CLOTH MODIFIER ======
cloth_mod = flag.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings

# Fabric properties — lightweight flag material
cs.mass = 0.15                       # Light fabric (kg/m^2)
cs.air_damping = 1.0                 # Air resistance

# Stiffness
cs.tension_stiffness = 15.0          # Resistance to stretching
cs.compression_stiffness = 15.0
cs.shear_stiffness = 5.0             # Resistance to shearing
cs.bending_stiffness = 0.5           # Low = flexible fabric

# Quality
cs.quality = 8                       # Simulation substeps (higher = more stable)
cs.time_scale = 1.0

# ====== PIN GROUP (fixed edge) ======
# Create vertex group for pinned vertices (left edge of flag)
vg = flag.vertex_groups.new(name="Pin")
for v in flag.data.vertices:
    if v.co.x < -1.4:               # Left edge vertices
        vg.add([v.index], 1.0, 'REPLACE')

cs.vertex_group_mass = "Pin"         # Pin these vertices

# ====== FLAGPOLE (collision object) ======
bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=6, location=(-1.5, 0, 3))
pole = bpy.context.active_object
pole.name = "Flagpole"
pole.modifiers.new("Collision", type='COLLISION')
pole.modifiers["Collision"].settings.thickness_outer = 0.02
pole.modifiers["Collision"].settings.damping = 0.5

# ====== WIND + TURBULENCE ======
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 3))
wind = bpy.context.active_object
wind.name = "Wind_Main"
wind.field.strength = 20.0
wind.field.flow = 0.5
wind.field.noise = 3.0
wind.field.seed = 1
wind.rotation_euler = (0, 1.5708, 0)

# Animate wind strength for gusts
wind.field.strength = 10.0
wind.field.keyframe_insert(data_path="strength", frame=1)
wind.field.strength = 30.0
wind.field.keyframe_insert(data_path="strength", frame=60)
wind.field.strength = 12.0
wind.field.keyframe_insert(data_path="strength", frame=90)
wind.field.strength = 35.0
wind.field.keyframe_insert(data_path="strength", frame=130)

bpy.ops.object.effector_add(type='TURBULENCE', location=(3, 0, 3))
turb = bpy.context.active_object
turb.name = "Turbulence_Main"
turb.field.strength = 10.0
turb.field.size = 2.0
turb.field.noise = 5.0
turb.field.flow = 0.3

# ====== RAIN PARTICLES (storm scene) ======
bpy.ops.mesh.primitive_plane_add(size=15, location=(0, 0, 10))
rain_emitter = bpy.context.active_object
rain_emitter.name = "RainEmitter"
rain_emitter.hide_render = True

ps_mod = rain_emitter.modifiers.new("Rain", type='PARTICLE_SYSTEM')
rain = ps_mod.particle_system.settings
rain.count = 3000
rain.frame_start = 1
rain.frame_end = 250
rain.lifetime = 30
rain.physics_type = 'NEWTON'
rain.mass = 0.001
rain.normal_factor = -0.5            # Downward bias
rain.factor_random = 0.3
rain.damping = 0.05
rain.particle_size = 0.01
rain.render_type = 'HALO'

# Rain affected by wind too
rain.effector_weights.wind = 0.3     # Rain drifts in wind

# ====== BAKE ======
bpy.ops.ptcache.bake_all(bake=True)
```

## Cloth Wrapping Around Rigid Body Objects

For cloth that drapes over or wraps around geometry (tablecloth, tarp, curtain):

```python
# Object to drape cloth over
bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 1))
table = bpy.context.active_object
table.name = "Table"
table.modifiers.new("Collision", type='COLLISION')
table.modifiers["Collision"].settings.thickness_outer = 0.01
table.modifiers["Collision"].settings.damping = 0.5
table.modifiers["Collision"].settings.friction = 0.5

# Cloth starts ABOVE the object and falls onto it
bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 3))
cloth = bpy.context.active_object
cloth.name = "Tablecloth"
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=25)
bpy.ops.object.mode_set(mode='OBJECT')

cloth_mod = cloth.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.mass = 0.3
cs.tension_stiffness = 20.0
cs.bending_stiffness = 1.0           # Slightly stiffer for tablecloth
cs.quality = 8

# Self-collision for folds
cloth_mod.collision_settings.use_self_collision = True
cloth_mod.collision_settings.self_friction = 5.0
cloth_mod.collision_settings.self_distance_min = 0.005
```

## Common Pitfalls

NOTE: Cloth `quality` is substeps per frame (higher = more stable but slower). Use 5-12.
NOTE: Collision modifier is on the OBSTACLE, not on the cloth object.
NOTE: `vertex_group_mass` pins vertices with weight 1.0 (fully pinned) and allows weight 0.0 (free).
NOTE: Wind direction is the object's local -Z axis by default. Rotate the wind empty to change direction.
NOTE: Particle `effector_weights` control how much each force field type affects the particle system.

TECHNIQUE: For realistic flag flapping, combine Wind (base direction) + Turbulence (chaotic variation) + animated Wind strength (gusts). This creates far more natural motion than Wind alone.
""",

    "technique_scene_physics_setup.md": """# Complete Scene Physics Orchestration
DocType: technique-guide
DocPath: techniques/mixed_physics/scene_orchestration
DocVersion: 5.0.1
PhysicsDomain: mixed
---

## Physics Evaluation Order in Blender

Blender evaluates physics modifiers in this order per frame:

```
1. Force Fields (computed first, available to all solvers)
2. Rigid Body World (scene-level, all rigid body objects)
3. Particle Systems (per-object, reads rigid body state)
4. Cloth (per-object, reads collision objects)
5. Soft Body (per-object)
6. Fluid (Mantaflow, per-domain)
```

This means force fields affect everything, rigid body runs before particles,
and fluid reads the final state of all other objects.

## Cache Management for Multi-Physics Scenes

Each physics system has its own cache. When editing one system, others may need rebaking.

```python
import bpy

scene = bpy.context.scene

# Free all caches before rebaking
bpy.ops.ptcache.free_bake_all()

# For fluid specifically
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'domain_settings'):
            if mod.domain_settings:
                bpy.ops.fluid.free_all()
                break
```

## Baking Strategy

### Sequential Bake (Safest for Multi-Physics)

```python
# Step 1: Set up all physics objects (no baking yet)
# Step 2: Bake in dependency order

# 2a. Rigid body — bake first (other systems may depend on positions)
bpy.ops.ptcache.bake_all(bake=True)

# 2b. Fluid — bake after rigid body effectors are settled
# Select fluid domain first
bpy.context.view_layer.objects.active = fluid_domain
bpy.ops.fluid.bake_all()

# Note: ptcache.bake_all() covers rigid body + particles + cloth
# Fluid has its own bake command
```

### Free Bake and Rebake Pattern

```python
def rebake_all_physics(scene):
    \"\"\"Free all caches and rebake in correct order.\"\"\"
    # Free everything
    bpy.ops.ptcache.free_bake_all()

    # Free fluid caches
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and mod.domain_settings:
                bpy.context.view_layer.objects.active = obj
                try:
                    bpy.ops.fluid.free_all()
                except:
                    pass

    # Bake rigid body + particles + cloth
    bpy.ops.ptcache.bake_all(bake=True)

    # Bake fluid
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and mod.domain_settings:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.fluid.bake_all()
```

## Timeline Setup

```python
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 150

# Substeps for accuracy across all systems
if scene.rigidbody_world:
    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 20

# Fluid cache must match scene range
# (set per fluid domain — see fluid docs)

# Render FPS affects physics timing
scene.render.fps = 24
# Higher FPS = slower physics (more frames per second of sim time)
```

## Scene Template 1: Explosion (Rigid Body + Mantaflow Gas + Particle Debris)

```python
import bpy
import random
from mathutils import Vector

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# ====== RIGID BODY WORLD ======
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20

# ====== SHARDS (from pre-fractured object) ======
# Assume shards[] created by fracture step
for shard in shards:
    bpy.context.view_layer.objects.active = shard
    shard.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    shard.rigid_body.mass = 0.5 + random.uniform(0, 1.0)
    shard.rigid_body.collision_shape = 'CONVEX_HULL'
    shard.select_set(False)

# ====== EXPLOSION FORCE ======
bpy.ops.object.effector_add(type='FORCE', location=(0, 0, 0))
explosion_force = bpy.context.active_object
explosion_force.name = "ExplosionForce"
explosion_force.field.strength = 500.0   # Strong initial push
explosion_force.field.falloff_power = 2  # Inverse square falloff
# Animate strength: strong burst then zero
explosion_force.field.strength = 500.0
explosion_force.field.keyframe_insert(data_path="strength", frame=1)
explosion_force.field.strength = 0.0
explosion_force.field.keyframe_insert(data_path="strength", frame=5)

# ====== MANTAFLOW FIRE/SMOKE DOMAIN ======
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 2))
fire_domain = bpy.context.active_object
fire_domain.name = "FireDomain"
fire_domain.scale = (5, 5, 5)
bpy.ops.object.transform_apply(scale=True)

mod = fire_domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
dset = mod.domain_settings
dset.domain_type = 'GAS'
dset.resolution_max = 96
dset.use_noise = True
dset.noise_scale = 2
dset.use_dissolve_smoke = True
dset.dissolve_speed = 40
dset.burning_rate = 0.8
dset.flame_smoke = 2.0
dset.flame_vorticity = 0.5
dset.cache_frame_start = 1
dset.cache_frame_end = 120
dset.cache_type = 'ALL'

# ====== FIRE FLOW SOURCE ======
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(0, 0, 0))
flow_obj = bpy.context.active_object
flow_obj.name = "FireSource"

mod_flow = flow_obj.modifiers.new("Fluid", type='FLUID')
mod_flow.fluid_type = 'FLOW'
fset = mod_flow.flow_settings
fset.flow_type = 'FIRE'
fset.flow_behavior = 'INFLOW'
fset.fuel_amount = 3.0
fset.temperature = 2.0

# ====== PARTICLE DEBRIS ======
bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0.1))
debris_emitter = bpy.context.active_object
debris_emitter.name = "DebrisEmitter"
debris_emitter.hide_render = True

ps_mod = debris_emitter.modifiers.new("Debris", type='PARTICLE_SYSTEM')
deb = ps_mod.particle_system.settings
deb.count = 400
deb.frame_start = 1
deb.frame_end = 5
deb.lifetime = 80
deb.physics_type = 'NEWTON'
deb.mass = 0.01
deb.normal_factor = 15.0
deb.factor_random = 8.0
deb.tangent_factor = 5.0
deb.damping = 0.1
deb.particle_size = 0.01
deb.render_type = 'HALO'

# ====== BAKE ORDER ======
# 1. Rigid body + particles
bpy.ops.ptcache.bake_all(bake=True)
# 2. Fluid (reads rigid body positions if effectors exist)
bpy.context.view_layer.objects.active = fire_domain
bpy.ops.fluid.bake_all()
```

## Scene Template 2: Storm (Cloth + Particles + Wind)

Core setup: see `technique_cloth_plus_forces.md` for complete code.
Key elements: cloth flag with pin group, wind + turbulence forces, rain particles.

## Scene Template 3: Water Scene (Liquid + Rigid Body + Particles)

Core setup: see `technique_fluid_plus_rigid_body.md` for complete code.
Key elements: liquid domain, rigid body rock effectors, spray particle system.

## Common Pitfalls

NOTE: `bpy.ops.ptcache.bake_all()` does NOT bake fluid. Use `bpy.ops.fluid.bake_all()` separately.
NOTE: `bpy.ops.fluid.bake_all()` requires the fluid domain to be the active object.
NOTE: `cache_type = 'ALL'` for fluid, not `REPLAY` — replay does not produce stable multi-physics data.
NOTE: Force field `strength` can be keyframed for time-varying effects (explosions, gusts).
NOTE: `bpy.ops.object.effector_add()` NOT `forcefield_add`.
NOTE: `use_dissolve_smoke` NOT `use_dissolve`.
NOTE: `resolution_max` NOT `resolution_divisions`.
NOTE: `substeps_per_frame` NOT `steps_per_second` for RigidBodyWorld.

TECHNIQUE: For explosion scenes, animate the Force field strength from high to zero over 3-5 frames.
This creates a blast wave effect: strong initial push, then gravity takes over.
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "rigid_body_particle_debris_combo",
        "issue": "scene only has large rigid body objects but lacks secondary debris — looks too clean",
        "code_snippet": """# Add particle debris emitters alongside rigid body simulation
# Ground needs BOTH rigid_body (for RB collisions) AND collision modifier (for particles)
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
bpy.ops.rigidbody.object_add(type='PASSIVE')
ground.rigid_body.collision_shape = 'MESH'
ground.modifiers.new("Collision", type='COLLISION')
ground.modifiers["Collision"].settings.damping = 0.6

# Spark emitter at impact zone
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0.05))
emitter = bpy.context.active_object
emitter.name = "SparkEmitter"
emitter.hide_render = True
ps_mod = emitter.modifiers.new("Sparks", type='PARTICLE_SYSTEM')
s = ps_mod.particle_system.settings
s.count = 500
s.frame_start = impact_frame
s.frame_end = impact_frame + 10
s.lifetime = 15
s.physics_type = 'NEWTON'
s.mass = 0.001
s.normal_factor = 12.0
s.factor_random = 5.0
s.damping = 0.05
s.render_type = 'HALO'

# Bake everything together
bpy.ops.ptcache.bake_all(bake=True)""",
        "effect_type": "mixed_physics",
        "improvement": 35.0,
        "experiment_id": "seed_mixed_physics_v1",
        "context_before": "Rigid body scene with only large objects — no secondary debris, sparks, or dust",
        "context_after": "Particle emitters at impact zones add sparks and dust. Ground has both rigid_body and collision modifier for dual-system interaction.",
    },
    {
        "name": "cloth_wind_turbulence_setup",
        "issue": "cloth flag hangs limp or moves unnaturally with only gravity — needs wind forces",
        "code_snippet": """# Wind + Turbulence for realistic cloth flapping
# NOTE: use effector_add NOT forcefield_add (removed in Blender 5.0)

# Primary wind direction
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 3))
wind = bpy.context.active_object
wind.name = "Wind_Main"
wind.field.strength = 20.0
wind.field.flow = 0.5
wind.field.noise = 3.0
wind.rotation_euler = (0, 1.5708, 0)  # Point along -X

# Animate gusts
wind.field.strength = 10.0
wind.field.keyframe_insert(data_path="strength", frame=1)
wind.field.strength = 30.0
wind.field.keyframe_insert(data_path="strength", frame=60)
wind.field.strength = 12.0
wind.field.keyframe_insert(data_path="strength", frame=90)

# Turbulence for chaotic variation
bpy.ops.object.effector_add(type='TURBULENCE', location=(3, 0, 3))
turb = bpy.context.active_object
turb.name = "Turbulence"
turb.field.strength = 10.0
turb.field.size = 2.0
turb.field.noise = 5.0

# Cloth settings for flag fabric
cloth_mod = flag.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.mass = 0.15
cs.tension_stiffness = 15.0
cs.bending_stiffness = 0.5
cs.quality = 8
cs.vertex_group_mass = "Pin"  # Pin left edge""",
        "effect_type": "mixed_physics",
        "improvement": 40.0,
        "experiment_id": "seed_mixed_physics_v1",
        "context_before": "Cloth simulation with only gravity — flag hangs limp, no wind motion",
        "context_after": "Wind + Turbulence forces create realistic flapping. Animated wind strength adds gusts. Pin group holds one edge fixed.",
    },
    {
        "name": "multi_physics_bake_coordination",
        "issue": "multi-physics scene has incorrect simulation because physics systems baked in wrong order",
        "code_snippet": """# Multi-physics bake order: rigid body first, then fluid, then verify
import bpy

scene = bpy.context.scene

# Step 1: Free all existing caches
bpy.ops.ptcache.free_bake_all()
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'domain_settings'):
            if mod.domain_settings:
                bpy.context.view_layer.objects.active = obj
                try:
                    bpy.ops.fluid.free_all()
                except:
                    pass

# Step 2: Bake rigid body + particles + cloth (ptcache systems)
bpy.ops.ptcache.bake_all(bake=True)

# Step 3: Bake fluid domains (reads rigid body effector positions)
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'domain_settings'):
            if mod.domain_settings:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.fluid.bake_all()

# NOTE: ptcache.bake_all does NOT bake fluid — separate command required
# NOTE: Fluid must bake AFTER rigid body if effectors are rigid body objects""",
        "effect_type": "mixed_physics",
        "improvement": 30.0,
        "experiment_id": "seed_mixed_physics_v1",
        "context_before": "Multi-physics scene baked in wrong order — fluid ignores rigid body obstacles, particles clip through objects",
        "context_after": "Sequential bake: free all caches, bake ptcache (rigid body + particles + cloth), then bake fluid. Ensures dependency order.",
    },
    {
        "name": "force_field_configuration_multi_system",
        "issue": "force fields affect all physics systems equally — need per-system control",
        "code_snippet": """# Force fields with per-system effector weight control
# NOTE: use effector_add NOT forcefield_add

# Create explosion force
bpy.ops.object.effector_add(type='FORCE', location=(0, 0, 0))
force = bpy.context.active_object
force.name = "ExplosionForce"
force.field.strength = 500.0
force.field.falloff_power = 2

# Animate blast: strong burst then zero
force.field.strength = 500.0
force.field.keyframe_insert(data_path="strength", frame=1)
force.field.strength = 0.0
force.field.keyframe_insert(data_path="strength", frame=5)

# Per-system effector weights — control which systems respond to which forces
# On particle system:
particle_settings = emitter.particle_systems[0].settings
particle_settings.effector_weights.force = 1.0     # Full explosion effect
particle_settings.effector_weights.wind = 0.3       # Light wind drift

# On cloth:
cloth_mod = cloth_obj.modifiers["Cloth"]
cloth_mod.settings.effector_weights.force = 0.5     # Half explosion effect
cloth_mod.settings.effector_weights.wind = 1.0       # Full wind effect
cloth_mod.settings.effector_weights.turbulence = 0.8 # Strong turbulence""",
        "effect_type": "mixed_physics",
        "improvement": 25.0,
        "experiment_id": "seed_mixed_physics_v1",
        "context_before": "All physics systems respond identically to force fields — cloth blows away in explosion, particles unaffected by wind",
        "context_after": "Per-system effector_weights let you tune how much each force field type affects each physics system independently.",
    },
]


# ============================================================
# UPLOAD TO VECTOR STORE
# ============================================================

def upload_technique_docs(dry_run: bool = False) -> dict:
    """Upload technique documents to the rewritten manual vector store."""
    store_id = os.getenv("BLENDER_REWRITTEN_MANUAL_STORE_ID")
    if not store_id:
        print("ERROR: BLENDER_REWRITTEN_MANUAL_STORE_ID not set", file=sys.stderr)
        print("Set it to the vector store ID from upload_rewritten_manual.py", file=sys.stderr)
        return {"success": False, "error": "No store ID"}

    if not OPENAI_AVAILABLE:
        print("ERROR: openai package required", file=sys.stderr)
        return {"success": False, "error": "No openai"}

    client = OpenAI()

    # Verify store exists
    try:
        vs = client.vector_stores.retrieve(store_id)
        print(f"Target store: {vs.name} ({store_id})")
        print(f"  Current files: {vs.file_counts.completed}")
    except Exception as e:
        print(f"ERROR: Cannot access store {store_id}: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

    if dry_run:
        print(f"\nDRY RUN — would upload {len(TECHNIQUE_DOCS)} files:")
        for name, content in TECHNIQUE_DOCS.items():
            print(f"  {name} ({len(content)} chars)")
        return {"success": True, "dry_run": True, "files": len(TECHNIQUE_DOCS)}

    # Upload files
    print(f"\nUploading {len(TECHNIQUE_DOCS)} technique documents...")
    success = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, content in TECHNIQUE_DOCS.items():
            filepath = Path(tmpdir) / name
            filepath.write_text(content)

            try:
                with open(filepath, "rb") as f:
                    file_response = client.files.create(file=f, purpose="assistants")
                client.vector_stores.files.create(
                    vector_store_id=store_id,
                    file_id=file_response.id,
                )
                success += 1
                print(f"  OK: {name} ({file_response.id})")
            except Exception as e:
                failed += 1
                print(f"  FAILED: {name}: {e}")

            time.sleep(0.1)  # Rate limit

    # Poll until processed
    if success > 0:
        print("\nWaiting for processing...")
        for _ in range(60):
            vs = client.vector_stores.retrieve(store_id)
            if vs.file_counts.in_progress == 0:
                break
            time.sleep(2)
        print(f"  Final: {vs.file_counts.completed} completed, {vs.file_counts.total} total")

    # Test query
    print("\nTest query: 'combining rigid body fluid cloth multiple physics systems'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="combining rigid body fluid cloth multiple physics systems bake order",
            max_num_results=3,
        )
        for i, result in enumerate(response.data):
            filename = result.filename or "unknown"
            score = result.score or 0
            preview = result.content[0].text[:100] if result.content else ""
            print(f"  [{i+1}] {filename} (score: {score:.3f})")
            print(f"      {preview}...")
    except Exception as e:
        print(f"  Test query error: {e}")

    return {"success": True, "uploaded": success, "failed": failed}


# ============================================================
# SEED CODE PATTERNS
# ============================================================

def seed_code_patterns(dry_run: bool = False) -> dict:
    """Seed code pattern memory with mixed physics techniques."""
    from utils.code_pattern_memory import get_pattern_memory

    memory = get_pattern_memory()
    seeded = []

    for pattern in CODE_PATTERNS:
        if dry_run:
            print(f"  Would seed: {pattern['name']}")
            seeded.append(pattern["name"])
            continue

        try:
            pid = memory.record_successful_pattern(
                issue=pattern["issue"],
                code_snippet=pattern["code_snippet"],
                effect_type=pattern["effect_type"],
                improvement=pattern["improvement"],
                experiment_id=pattern["experiment_id"],
                name=pattern["name"],
                context_before=pattern["context_before"],
                context_after=pattern["context_after"],
            )
            # Mark as applicable to all mixed-physics-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["mixed_physics", "explosion", "destruction", "storm", "water", "cloth"]
            memory._save_pattern(p)
            seeded.append(pattern["name"])
            print(f"  Seeded: {pattern['name']} ({pid})")
        except Exception as e:
            print(f"  FAILED: {pattern['name']}: {e}")

    return {
        "seeded": len(seeded),
        "total_patterns": len(memory.patterns),
        "names": seeded,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seed mixed physics technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED MIXED PHYSICS TECHNIQUES")
    print("=" * 60)

    # Upload technique docs to vector store
    if not args.patterns_only:
        print("\n--- TECHNIQUE DOCUMENTS → VECTOR STORE ---")
        docs_result = upload_technique_docs(dry_run=args.dry_run)
        print(f"Result: {docs_result}")

    # Seed code patterns
    if not args.docs_only:
        print("\n--- CODE PATTERNS → PATTERN MEMORY ---")
        patterns_result = seed_code_patterns(dry_run=args.dry_run)
        print(f"Result: {patterns_result}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
