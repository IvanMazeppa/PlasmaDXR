#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with particle system technique knowledge.

Creates LLM-optimized markdown documents describing diverse particle system
approaches in Blender 5.0, then uploads them to the rewritten manual vector store.
Also seeds the code pattern memory with proven particle snippets.

This directly addresses the technique diversity problem: the Research Agent can
only discover what's in the vector store. Without these docs, it defaults to
basic emitter particles every time.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_particle_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_particle_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_particle_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_particle_techniques.py --docs-only
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
    "particle_technique_overview.md": """# Particle System Techniques in Blender 5.0
DocType: technique-guide
DocPath: techniques/particles/overview
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Overview of Particle System Approaches

Blender 5.0 particle systems create point-based simulations for effects that involve
many small independent objects: sparks, rain, snow, debris, confetti, fur, grass.
The main settings type is `bpy.types.ParticleSettings`.

## Two Particle Modes

- **Emitter** — particles spawn over time, move with physics, die after lifetime.
  Use for: sparks, rain, snow, debris, confetti, fireflies, dust motes.
- **Hair** — particles grow as strands from surface at frame 1, no physics by default.
  Use for: fur, grass, hair, feathers, static fiber effects.

## Particle Physics Types (ParticleSettings.physics_type)

| Type | Behavior | Best For |
|------|----------|----------|
| `NEWTON` | Gravity + forces + damping | Sparks, rain, debris, confetti |
| `KEYED` | Follow path between keyed targets | Magic trails, guided projectiles |
| `BOIDS` | Flocking/swarming AI behavior | Birds, fish, insects, crowd sim |
| `FLUID` | SPH fluid-like interaction | Small-scale liquid splashes |
| `NO` | No physics, only initial velocity | Static placement, emitter-only |

## Force Fields for Particles

Force fields affect particle motion. Add via `bpy.ops.object.effector_add(type=...)`:

| Force Field | Type String | Effect |
|-------------|-------------|--------|
| Wind | `'WIND'` | Constant directional push |
| Turbulence | `'TURBULENCE'` | Chaotic noise-based displacement |
| Vortex | `'VORTEX'` | Spiral/rotational motion |
| Drag | `'DRAG'` | Velocity-dependent damping |
| Harmonic | `'HARMONIC'` | Spring-like attraction to point |
| Force | `'FORCE'` | Radial attraction/repulsion |
| Charge | `'CHARGE'` | Attract opposite, repel same |
| Magnetic | `'MAGNETIC'` | Lorentz force (velocity-dependent) |

IMPORTANT: Use `bpy.ops.object.effector_add()` NOT `bpy.ops.object.forcefield_add()`.
The `forcefield_add` operator does NOT exist in Blender 5.0.

## Particle Render Types (ParticleSettings.render_type)

| Type | Result | Use Case |
|------|--------|----------|
| `'HALO'` | Billboard point sprites | Sparks, dust, magical particles |
| `'OBJECT'` | Single mesh instance per particle | Debris, confetti, leaves |
| `'COLLECTION'` | Random mesh from collection | Varied debris, mixed objects |
| `'PATH'` | Strand/curve rendering | Hair, fur, grass, fibers |
| `'NONE'` | Invisible (physics only) | Collision testing, abstract sim |

## Decision Guide: Particles vs Mantaflow vs Geometry Nodes

| Need | Use |
|------|-----|
| Many independent small objects (sparks, rain, debris) | **Particles** |
| Continuous fluid/smoke/fire volume | **Mantaflow** |
| Procedural scattering on surface (grass, rocks) | **Geometry Nodes** |
| Flocking/swarming behavior | **Particles (Boids)** |
| Strand-based rendering (fur, hair) | **Particles (Hair)** or **GN** |
| Physics interaction with rigid bodies | **Particles + Collision modifier** |
| Fine dust/powder cloud | **Mantaflow** (volumetric) |

## Technique Comparison for Common Effects

| Effect | Primary Technique | Secondary | Notes |
|--------|-------------------|-----------|-------|
| Sparks | Emitter + Newton + gravity | Halo render | Short lifetime, high velocity |
| Rain | Emitter + Newton + collision | Line/path render | Long vertical strokes |
| Snow | Emitter + Newton + turbulence | Halo or object | Slow, random drift |
| Debris | Emitter + Newton + object instance | Collection render | Random rotation |
| Confetti | Emitter + Newton + drag | Collection render | Air damping, tumble |
| Fireflies | Emitter + Boids | Halo render + emission | Flocking glow points |
| Leaves in wind | Emitter + Wind + Turbulence | Object instance | Force field combo |

NOTE: For large-count particle effects (>10000), consider Geometry Nodes instancing
instead — it handles high counts more efficiently than the legacy particle system.

TECHNIQUE: Always combine at least 2 force fields for natural motion. Pure gravity
looks mechanical. Wind + Turbulence + Drag creates convincing atmospheric particles.
""",

    "technique_emitter_effects.md": """# Emitter Particle Effects in Blender 5.0
DocType: technique-guide
DocPath: techniques/particles/emitter_effects
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Emitter Particle System — Core API

```python
# Add particle system to emitter object
emitter = bpy.data.objects["Emitter"]
ps_mod = emitter.modifiers.new("ParticleSystem", type='PARTICLE_SYSTEM')
ps = ps_mod.particle_system
settings = ps.settings

# Core emission settings
settings.type = 'EMITTER'           # Not 'HAIR'
settings.count = 500                # Total particles over emission period
settings.frame_start = 1            # Start emitting
settings.frame_end = 50             # Stop emitting
settings.lifetime = 30              # Frames each particle lives
settings.lifetime_random = 0.3      # 0-1, randomize lifetime

# Physics
settings.physics_type = 'NEWTON'
settings.mass = 0.1                 # Particle mass (kg)
settings.particle_size = 0.01       # Display/collision size
settings.size_random = 0.5          # Size variation

# Velocity
settings.normal_factor = 5.0        # Speed along emitter normal
settings.tangent_factor = 0.0       # Speed along emitter surface
settings.factor_random = 1.0        # Random velocity component
settings.object_align_factor = (0, 0, 0)  # Object-axis velocity

# Damping
settings.damping = 0.05             # Velocity decay per frame (0=none, 1=full)
settings.drag_factor = 0.0          # Air drag
```

## Spark Fountain — Complete Script

Sparks from welding/fire: short lifetime, high velocity, small bright particles.

```python
import bpy
import random
from mathutils import Vector

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# Emitter — point source
bpy.ops.mesh.primitive_plane_add(size=0.1, location=(0, 0, 0.5))
emitter = bpy.context.active_object
emitter.name = "SparkEmitter"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Sparks", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 2000
settings.frame_start = 1
settings.frame_end = 100
settings.lifetime = 15
settings.lifetime_random = 0.5
settings.emit_from = 'FACE'

# Physics — fast upward with spread
settings.physics_type = 'NEWTON'
settings.mass = 0.001
settings.normal_factor = 6.0        # Strong upward burst
settings.tangent_factor = 3.0       # Horizontal spread
settings.factor_random = 2.0        # Variation
settings.damping = 0.02             # Minimal air resistance

# Size — tiny bright points
settings.particle_size = 0.002
settings.size_random = 0.4
settings.render_type = 'HALO'

# Gravity pulls sparks in arcs
settings.effector_weights.gravity = 1.0

# Emission material — bright orange/yellow
mat = bpy.data.materials.new("M_Spark")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (1.0, 0.6, 0.1, 1)  # Orange
emit.inputs["Strength"].default_value = 50.0               # Very bright
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
emitter.data.materials.append(mat)
```

## Rain with Ground Splash — Complete Script

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 250

# Large emitter plane above scene
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 15))
rain_emitter = bpy.context.active_object
rain_emitter.name = "RainEmitter"
rain_emitter.hide_render = True

ps_mod = rain_emitter.modifiers.new("Rain", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 5000
settings.frame_start = 1
settings.frame_end = 200
settings.lifetime = 60
settings.lifetime_random = 0.1
settings.emit_from = 'FACE'
settings.distribution = 'RAND'

# Physics — fast downward
settings.physics_type = 'NEWTON'
settings.mass = 0.0001
settings.normal_factor = -0.5       # Slight downward push from face normal
settings.object_align_factor = (0, 0, -12.0)  # Strong downward velocity
settings.factor_random = 0.3
settings.damping = 0.0

# Render as stretched lines (motion-blurred drops)
settings.particle_size = 0.005
settings.render_type = 'HALO'

# Ground collision
bpy.ops.mesh.primitive_plane_add(size=30.0, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.modifiers.new("Collision", type='COLLISION')
col = ground.modifiers["Collision"]
col.settings.damping = 0.8          # Rain splats, doesn't bounce much
col.settings.friction = 0.5
col.settings.stickiness = 0.2       # Slight stick on impact

# Wind for angled rain
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 10))
wind = bpy.context.active_object
wind.name = "RainWind"
wind.field.strength = 3.0
wind.field.noise = 0.5              # Gusty wind
wind.rotation_euler = (1.2, 0, 0)   # Angled
```

## Parameter Quick Reference by Effect Type

| Effect | count | lifetime | normal_factor | damping | particle_size | render_type |
|--------|-------|----------|---------------|---------|---------------|-------------|
| Sparks | 1000-3000 | 10-20 | 5-10 | 0.01-0.05 | 0.001-0.005 | HALO |
| Rain | 3000-10000 | 40-80 | 0.5 | 0.0 | 0.003-0.008 | HALO/PATH |
| Snow | 2000-5000 | 100-200 | 0.3-1.0 | 0.1-0.2 | 0.005-0.02 | HALO/OBJECT |
| Debris | 100-500 | 30-60 | 3-8 | 0.05-0.1 | 0.01-0.05 | OBJECT |
| Confetti | 200-1000 | 60-120 | 3-6 | 0.15-0.3 | 0.01-0.03 | COLLECTION |
| Dust | 500-2000 | 50-100 | 1-3 | 0.1-0.2 | 0.001-0.003 | HALO |

NOTE: `emit_from` options: `'VERT'`, `'FACE'`, `'VOLUME'`. Use `'FACE'` for surface emission,
`'VOLUME'` for volumetric spawn (explosions, bursts).

NOTE: `distribution` options: `'JIT'` (jittered uniform), `'RAND'` (random). Use `'RAND'`
for natural effects like rain.

TECHNIQUE: For sparks, set `lifetime_random = 0.5` and `factor_random = 2.0` to break
the uniform-spray look. Real sparks have wildly varying speeds and lifetimes.
""",

    "technique_particle_forces.md": """# Force Fields with Particles in Blender 5.0
DocType: technique-guide
DocPath: techniques/particles/force_fields
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Force Field API

Force fields are added as empty objects with field physics. They affect ALL particle
systems in range (unless excluded via effector weights or layers).

IMPORTANT: Use `bpy.ops.object.effector_add(type=...)` — NOT `forcefield_add`.

```python
# Add a wind force field
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 2))
wind = bpy.context.active_object
wind.name = "Wind"
wind.field.strength = 5.0           # Force magnitude
wind.field.noise = 1.0              # Temporal noise on strength
wind.field.seed = 42                # Noise seed
wind.field.flow = 0.5               # Adds flow-like behavior

# Rotation determines wind direction (Z-axis of empty = wind direction)
wind.rotation_euler = (1.57, 0, 0)  # Wind blowing along +Y
```

## Force Field Types — Detailed Reference

### Wind — Constant Directional Force
```python
bpy.ops.object.effector_add(type='WIND', location=(0, -5, 3))
ff = bpy.context.active_object
ff.field.strength = 8.0             # Newtons (strong breeze)
ff.field.noise = 2.0                # Gusts — higher = more chaotic
ff.field.flow = 0.3                 # Smooths force application
# Direction: local +Z axis of the empty object
```

### Turbulence — Chaotic Noise-Based Force
```python
bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 3))
ff = bpy.context.active_object
ff.field.strength = 3.0             # Turbulence intensity
ff.field.size = 1.5                 # Scale of noise pattern (larger = bigger swirls)
ff.field.noise = 0.5                # Additional temporal randomness
ff.field.flow = 0.5                 # Flow smoothing
```

### Vortex — Spiral Motion
```python
bpy.ops.object.effector_add(type='VORTEX', location=(0, 0, 0))
ff = bpy.context.active_object
ff.field.strength = 5.0             # Rotation speed
# Axis of rotation: local Z axis of empty
# Positive = counterclockwise, Negative = clockwise
```

### Drag — Velocity-Dependent Damping
```python
bpy.ops.object.effector_add(type='DRAG', location=(0, 0, 2))
ff = bpy.context.active_object
ff.field.linear_drag = 0.3          # Constant drag (like friction)
ff.field.quadratic_drag = 0.1       # Speed-dependent drag (like air resistance)
# Drag field slows particles proportional to velocity
```

### Harmonic — Spring-Like Attraction
```python
bpy.ops.object.effector_add(type='HARMONIC', location=(0, 0, 2))
ff = bpy.context.active_object
ff.field.strength = 5.0             # Spring stiffness
ff.field.damping = 0.1              # Oscillation damping
# Particles oscillate around the empty's position
```

## Force Field Falloff Settings

All force fields have falloff (how strength decreases with distance):

```python
ff = bpy.context.active_object
ff.field.falloff_type = 'SPHERE'    # 'SPHERE', 'TUBE', 'CONE'
ff.field.use_min_distance = True
ff.field.distance_min = 0.5         # Full strength within this radius
ff.field.use_max_distance = True
ff.field.distance_max = 10.0        # Zero force beyond this radius
ff.field.falloff_power = 2.0        # Exponent: 1=linear, 2=inverse-square
```

## Combining Multiple Force Fields

Real atmospheric effects need multiple forces working together:

```python
# Leaves in wind: Wind (main direction) + Turbulence (swirling) + Drag (air resistance)

# 1. Main wind direction
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 3))
wind = bpy.context.active_object
wind.name = "MainWind"
wind.field.strength = 4.0
wind.field.noise = 1.5              # Gusty
wind.rotation_euler = (1.57, 0, 0.3)  # Angled wind

# 2. Turbulence for chaotic swirling
bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
turb = bpy.context.active_object
turb.name = "WindTurbulence"
turb.field.strength = 2.0
turb.field.size = 2.0               # Large swirls
turb.field.noise = 1.0

# 3. Drag for air resistance (prevents infinite acceleration)
bpy.ops.object.effector_add(type='DRAG', location=(0, 0, 2))
drag = bpy.context.active_object
drag.name = "AirDrag"
drag.field.linear_drag = 0.1
drag.field.quadratic_drag = 0.05
```

## Leaves in Wind — Complete Script

```python
import bpy
import random

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 200

# Leaf mesh (simple flat quad)
bpy.ops.mesh.primitive_plane_add(size=0.05, location=(0, 0, 0))
leaf_template = bpy.context.active_object
leaf_template.name = "LeafTemplate"
leaf_template.hide_set(True)

# Leaf material
mat = bpy.data.materials.new("M_Leaf")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.2, 0.5, 0.05, 1)
bsdf.inputs["Roughness"].default_value = 0.8
leaf_template.data.materials.append(mat)

# Emitter — spawn area
bpy.ops.mesh.primitive_plane_add(size=6.0, location=(-3, 0, 4))
emitter = bpy.context.active_object
emitter.name = "LeafSpawner"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Leaves", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 300
settings.frame_start = 1
settings.frame_end = 150
settings.lifetime = 120
settings.lifetime_random = 0.3
settings.emit_from = 'FACE'

# Physics
settings.physics_type = 'NEWTON'
settings.mass = 0.0005              # Very light leaves
settings.normal_factor = 0.5
settings.factor_random = 1.0
settings.damping = 0.1

# Render as instanced leaf mesh
settings.render_type = 'OBJECT'
settings.instance_object = leaf_template
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0  # Random tumble
settings.particle_size = 1.0
settings.size_random = 0.4

# Force fields
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 3))
wind = bpy.context.active_object
wind.field.strength = 4.0
wind.field.noise = 2.0
wind.rotation_euler = (1.57, 0, 0)

bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
turb = bpy.context.active_object
turb.field.strength = 2.5
turb.field.size = 1.5

bpy.ops.object.effector_add(type='DRAG', location=(0, 0, 2))
drag = bpy.context.active_object
drag.field.linear_drag = 0.15
drag.field.quadratic_drag = 0.05

# Ground collision
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.modifiers.new("Collision", type='COLLISION')
ground.modifiers["Collision"].settings.damping = 0.9
ground.modifiers["Collision"].settings.friction = 0.8
```

TECHNIQUE: Always add Drag when using Wind — without it, particles accelerate indefinitely
and fly off-screen. Drag creates realistic terminal velocity behavior.

NOTE: `effector_weights` on the particle system can scale per-force-field influence:
`settings.effector_weights.gravity = 0.5` halves gravity effect on these particles.
""",

    "technique_particle_instancing.md": """# Particle Instancing — Rendering Objects as Particles in Blender 5.0
DocType: technique-guide
DocPath: techniques/particles/instancing
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Particle Render Types for Object Instancing

Instead of rendering particles as points/halos, instance real mesh objects at each
particle position. This is how you create debris showers, confetti, scattered rocks, etc.

## render_type = 'OBJECT' — Single Mesh

Every particle becomes a copy of one mesh object:

```python
settings = ps_mod.particle_system.settings
settings.render_type = 'OBJECT'
settings.instance_object = bpy.data.objects["DebrisMesh"]
settings.use_rotation_instance = True       # Particle rotation affects instance
settings.rotation_factor_random = 1.0       # Full random rotation
settings.use_scale_instance = True          # Particle size affects instance
settings.particle_size = 0.5               # Base scale multiplier
settings.size_random = 0.6                 # Scale variation (0-1)
```

## render_type = 'COLLECTION' — Random from Collection

Each particle randomly picks a mesh from a collection — gives visual variety:

```python
settings.render_type = 'COLLECTION'
settings.instance_collection = bpy.data.collections["DebrisVariants"]
settings.use_collection_pick_random = True  # Random selection
settings.use_collection_count = False       # Equal probability
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0
settings.particle_size = 1.0
settings.size_random = 0.5
```

## render_type = 'PATH' — Strand Rendering

For hair, fur, grass — renders as curves/strands:

```python
settings.render_type = 'PATH'
settings.path_start = 0.0                  # Strand start (0-1)
settings.path_end = 1.0                    # Strand end (0-1)
# Strand shape controlled by children and hair dynamics
```

## Rotation Control

```python
# Initial rotation
settings.rotation_mode = 'OB_Z'           # Align Z to velocity direction
# Options: 'NONE', 'NOR' (normal), 'NOR_TAN', 'VEL', 'OB_X', 'OB_Y', 'OB_Z', 'GLOB_X', etc.

# Angular velocity (tumbling)
settings.angular_velocity_mode = 'RAND'   # Random spin
settings.angular_velocity_factor = 2.0    # Spin speed multiplier
# Options: 'NONE', 'SPIN', 'RAND'

# Per-particle random rotation
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0     # 0=no random, 1=fully random
settings.phase_factor = 0.0               # Phase offset
settings.phase_factor_random = 1.0        # Random phase offset
```

## Confetti Burst — Complete Script

Random colored flat planes bursting upward with air damping and tumble:

```python
import bpy
import random

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 150

# Create confetti pieces — flat colored squares
confetti_col = bpy.data.collections.new("ConfettiPieces")
bpy.context.scene.collection.children.link(confetti_col)

colors = [
    (1.0, 0.1, 0.1, 1),   # Red
    (0.1, 0.5, 1.0, 1),   # Blue
    (1.0, 0.9, 0.1, 1),   # Yellow
    (0.1, 0.9, 0.2, 1),   # Green
    (1.0, 0.4, 0.7, 1),   # Pink
    (0.9, 0.5, 0.0, 1),   # Orange
]

for i, color in enumerate(colors):
    bpy.ops.mesh.primitive_plane_add(size=0.02)
    piece = bpy.context.active_object
    piece.name = f"Confetti_{i}"
    piece.scale = (1, 0.6, 1)  # Slightly rectangular
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new(f"M_Confetti_{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    piece.data.materials.append(mat)

    # Move to collection, unlink from scene
    confetti_col.objects.link(piece)
    bpy.context.scene.collection.objects.unlink(piece)
    piece.hide_set(True)

# Emitter — upward-facing point
bpy.ops.mesh.primitive_plane_add(size=0.3, location=(0, 0, 0.5))
emitter = bpy.context.active_object
emitter.name = "ConfettiCannon"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Confetti", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 500
settings.frame_start = 10
settings.frame_end = 15              # Short burst
settings.lifetime = 120
settings.lifetime_random = 0.2
settings.emit_from = 'FACE'

# Physics — burst up then float down
settings.physics_type = 'NEWTON'
settings.mass = 0.0002
settings.normal_factor = 8.0        # Strong upward burst
settings.tangent_factor = 3.0       # Spread sideways
settings.factor_random = 2.5
settings.damping = 0.15             # Air resistance slows confetti

# Instance from collection
settings.render_type = 'COLLECTION'
settings.instance_collection = confetti_col
settings.use_collection_pick_random = True
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0
settings.angular_velocity_mode = 'RAND'
settings.angular_velocity_factor = 3.0  # Fast tumble
settings.particle_size = 1.0
settings.size_random = 0.3

# Gravity + slight turbulence for flutter
settings.effector_weights.gravity = 0.3  # Reduced gravity — confetti is light

bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 3))
turb = bpy.context.active_object
turb.field.strength = 1.5
turb.field.size = 1.0
```

## Rock Debris Shower — Complete Script

```python
import bpy
import random
from mathutils import Vector

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 100

# Create rock variants
rock_col = bpy.data.collections.new("RockDebris")
bpy.context.scene.collection.children.link(rock_col)

for i in range(4):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.03)
    rock = bpy.context.active_object
    rock.name = f"Rock_{i}"
    # Deform to look rocky
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(rock.data)
    for v in bm.verts:
        v.co += Vector([random.uniform(-0.008, 0.008) for _ in range(3)])
    bm.to_mesh(rock.data)
    bm.free()

    mat = bpy.data.materials.new(f"M_Rock_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    gray = 0.15 + random.uniform(0, 0.15)
    bsdf.inputs["Base Color"].default_value = (gray, gray * 0.9, gray * 0.85, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    rock.data.materials.append(mat)

    rock_col.objects.link(rock)
    bpy.context.scene.collection.objects.unlink(rock)
    rock.hide_set(True)

# Emitter at explosion point
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 2))
emitter = bpy.context.active_object
emitter.name = "DebrisSource"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Debris", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 300
settings.frame_start = 5
settings.frame_end = 8               # Explosive burst
settings.lifetime = 60
settings.lifetime_random = 0.3
settings.emit_from = 'VOLUME'

settings.physics_type = 'NEWTON'
settings.mass = 0.5
settings.normal_factor = 12.0       # High-speed outward
settings.factor_random = 4.0
settings.damping = 0.02

settings.render_type = 'COLLECTION'
settings.instance_collection = rock_col
settings.use_collection_pick_random = True
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0
settings.angular_velocity_mode = 'RAND'
settings.angular_velocity_factor = 5.0
settings.particle_size = 1.0
settings.size_random = 0.7          # Wide size variation

# Ground collision
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.modifiers.new("Collision", type='COLLISION')
ground.modifiers["Collision"].settings.damping = 0.6
ground.modifiers["Collision"].settings.friction = 0.7
```

TECHNIQUE: For debris, always set `emit_from = 'VOLUME'` with a sphere emitter — this
gives an omnidirectional burst. For directional debris, use a hemisphere or cone emitter.

NOTE: `instance_object` / `instance_collection` must reference valid objects with geometry.
Create the instance meshes BEFORE configuring the particle system.
""",

    "technique_particle_collision.md": """# Particle Collision and Interaction in Blender 5.0
DocType: technique-guide
DocPath: techniques/particles/collision
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Collision Modifier — Making Surfaces Interact with Particles

Any mesh can become a collision surface for particles by adding a Collision modifier.
Particles bounce off, stick to, or die on collision surfaces.

```python
# Add collision to any mesh
surface = bpy.data.objects["Floor"]
bpy.context.view_layer.objects.active = surface
surface.modifiers.new("Collision", type='COLLISION')
col = surface.modifiers["Collision"]

# Collision settings (bpy.types.CollisionSettings)
col.settings.damping = 0.5          # Energy loss on bounce (0=elastic, 1=dead stop)
col.settings.damping_random = 0.1   # Randomize damping
col.settings.friction = 0.3         # Surface friction (0=ice, 1=sandpaper)
col.settings.friction_random = 0.1  # Randomize friction
col.settings.thickness_outer = 0.02 # Collision detection distance (outward)
col.settings.thickness_inner = 0.02 # Collision detection distance (inward)
col.settings.stickiness = 0.0       # Particle adhesion (0=none, 1=full stick)
col.settings.use_particle_kill = False  # True = particles die on contact
```

## Collision Response Types

| Setting | Value | Effect |
|---------|-------|--------|
| `damping` close to 0 | 0.0-0.2 | Bouncy (rubber ball, sparks) |
| `damping` close to 1 | 0.8-1.0 | Dead stop (mud, sticky surface) |
| `stickiness` > 0 | 0.1-1.0 | Particles adhere to surface |
| `use_particle_kill` | True | Particles die on contact (puddle splash) |
| `friction` high | 0.7-1.0 | Particles slow along surface (gravel) |

## Particles Bouncing Off Surface — Complete Script

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 150

# Angled bounce surface
bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 1))
surface = bpy.context.active_object
surface.name = "BounceSurface"
surface.rotation_euler = (0.3, 0, 0)  # Slight angle

# Add collision — bouncy with some friction
surface.modifiers.new("Collision", type='COLLISION')
col = surface.modifiers["Collision"]
col.settings.damping = 0.2           # 80% energy retained = bouncy
col.settings.friction = 0.2
col.settings.thickness_outer = 0.02

# Ball emitter above surface
bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, -1, 4))
emitter = bpy.context.active_object
emitter.name = "BallDropper"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Balls", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 200
settings.frame_start = 1
settings.frame_end = 80
settings.lifetime = 100
settings.emit_from = 'FACE'

settings.physics_type = 'NEWTON'
settings.mass = 0.01
settings.normal_factor = 0.0        # Drop straight down
settings.factor_random = 0.5        # Slight spread
settings.damping = 0.01

# Render as small spheres
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015, location=(10, 10, 10))
ball_mesh = bpy.context.active_object
ball_mesh.name = "BallInstance"
ball_mesh.hide_set(True)

settings.render_type = 'OBJECT'
settings.instance_object = ball_mesh
settings.particle_size = 1.0
settings.size_random = 0.3

# Floor collision to catch bounced particles
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "Floor"
floor.modifiers.new("Collision", type='COLLISION')
floor.modifiers["Collision"].settings.damping = 0.7
```

## Rain on Window (Particles Sticking to Surface) — Complete Script

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 250

# Window pane (vertical surface)
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 2))
window = bpy.context.active_object
window.name = "WindowPane"
window.rotation_euler = (1.5708, 0, 0)  # Vertical

# Glass material
mat = bpy.data.materials.new("M_WindowGlass")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Transmission Weight"].default_value = 0.9
bsdf.inputs["Roughness"].default_value = 0.05
bsdf.inputs["IOR"].default_value = 1.52
window.data.materials.append(mat)

# Collision — sticky for rain adhesion
window.modifiers.new("Collision", type='COLLISION')
col = window.modifiers["Collision"]
col.settings.damping = 0.95          # Nearly full energy absorption
col.settings.friction = 0.8          # High friction — drops crawl slowly
col.settings.stickiness = 0.7        # Strong adhesion
col.settings.thickness_outer = 0.005

# Rain emitter facing window
bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, -3, 2))
emitter = bpy.context.active_object
emitter.name = "RainSource"
emitter.rotation_euler = (1.5708, 0, 0)  # Face toward window
emitter.hide_render = True

ps_mod = emitter.modifiers.new("RainDrops", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 1000
settings.frame_start = 1
settings.frame_end = 200
settings.lifetime = 200
settings.emit_from = 'FACE'
settings.distribution = 'RAND'

settings.physics_type = 'NEWTON'
settings.mass = 0.00005
settings.object_align_factor = (0, 5.0, -0.5)  # Toward window + slight downward
settings.factor_random = 0.8
settings.damping = 0.0

# Small droplets
settings.particle_size = 0.003
settings.size_random = 0.5
settings.render_type = 'HALO'

# Slight wind for natural approach angle
bpy.ops.object.effector_add(type='WIND', location=(0, -5, 3))
wind = bpy.context.active_object
wind.field.strength = 2.0
wind.field.noise = 1.0
wind.rotation_euler = (1.57, 0, 0)  # Push toward window
```

## Keyed Particles — Following Paths Between Targets

Keyed particles move between target objects in sequence:

```python
settings.physics_type = 'KEYED'

# Add keyed targets (other objects with particle systems)
# Keyed particles interpolate position between target particle systems
# Useful for: magic trails, energy beams, guided projectiles
# Requires at least 2 keyed targets
```

## Self-Interaction Between Particle Systems

Particles from different systems can interact via:
1. **Collision** — both systems' particles bounce off collision surfaces
2. **Force fields** — both affected by same force fields
3. **Boids mutual avoidance** — Boids physics supports predator/prey relationships

```python
# Two particle systems on same emitter — independent but share physics
ps1_mod = emitter.modifiers.new("SystemA", type='PARTICLE_SYSTEM')
ps2_mod = emitter.modifiers.new("SystemB", type='PARTICLE_SYSTEM')
# Each has its own settings but both react to same collision surfaces and force fields
```

NOTE: Particles do NOT collide with each other within the same system or between systems.
Use Collision modifier on mesh surfaces to create interaction surfaces.

NOTE: `stickiness` values above 0.5 will make most particles permanently stick. Use lower
values (0.1-0.3) for partial adhesion where some particles bounce and others stick.

TECHNIQUE: For rain on window, combine high `stickiness` (0.7) with high `friction` (0.8)
and high `damping` (0.95). This makes drops hit and crawl slowly downward, mimicking real
water behavior on glass.
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "spark_emitter_with_gravity",
        "issue": "spark particles look too uniform — all same speed, direction, lifetime",
        "code_snippet": """# Spark emitter: high velocity, short lifetime, randomized
ps_mod = emitter.modifiers.new("Sparks", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 2000
settings.frame_start = 1
settings.frame_end = 100
settings.lifetime = 15
settings.lifetime_random = 0.5       # Varied lifetimes break uniformity
settings.physics_type = 'NEWTON'
settings.mass = 0.001
settings.normal_factor = 6.0        # Strong upward burst
settings.tangent_factor = 3.0       # Horizontal spread
settings.factor_random = 2.0        # Speed variation
settings.damping = 0.02
settings.particle_size = 0.002
settings.render_type = 'HALO'
settings.effector_weights.gravity = 1.0  # Arcing trajectories""",
        "effect_type": "particles",
        "improvement": 35.0,
        "experiment_id": "seed_particles_v1",
        "context_before": "Default emitter creates uniform spray — all particles same speed and lifetime, looks mechanical",
        "context_after": "High factor_random (2.0) + lifetime_random (0.5) + tangent_factor (3.0) creates natural spark scatter with varying arcs",
    },
    {
        "name": "rain_with_collision",
        "issue": "rain particles pass through ground or bounce unrealistically",
        "code_snippet": """# Rain with proper ground collision
# Emitter high above scene
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 15))
emitter = bpy.context.active_object
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Rain", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 5000
settings.lifetime = 60
settings.object_align_factor = (0, 0, -12.0)  # Fast downward
settings.physics_type = 'NEWTON'
settings.damping = 0.0
settings.render_type = 'HALO'

# Ground with collision — high damping so rain splats
ground = bpy.data.objects["Ground"]
ground.modifiers.new("Collision", type='COLLISION')
col = ground.modifiers["Collision"]
col.settings.damping = 0.8          # Rain doesn't bounce
col.settings.friction = 0.5
col.settings.stickiness = 0.2""",
        "effect_type": "particles",
        "improvement": 30.0,
        "experiment_id": "seed_particles_v1",
        "context_before": "Rain passes through ground plane — missing Collision modifier, or damping too low causing bouncing",
        "context_after": "Collision modifier on ground with damping=0.8 and stickiness=0.2 makes rain splat and settle realistically",
    },
    {
        "name": "force_field_wind_turbulence_combo",
        "issue": "particles move in straight lines or pure gravity arcs — no atmospheric feel",
        "code_snippet": """# Combined force fields for atmospheric particle motion
# IMPORTANT: use effector_add, NOT forcefield_add

# Wind — main directional push
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 3))
wind = bpy.context.active_object
wind.field.strength = 4.0
wind.field.noise = 2.0              # Gusty variation
wind.rotation_euler = (1.57, 0, 0)  # Direction

# Turbulence — chaotic swirling
bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
turb = bpy.context.active_object
turb.field.strength = 2.5
turb.field.size = 1.5               # Swirl scale

# Drag — prevents infinite acceleration, creates terminal velocity
bpy.ops.object.effector_add(type='DRAG', location=(0, 0, 2))
drag = bpy.context.active_object
drag.field.linear_drag = 0.15
drag.field.quadratic_drag = 0.05""",
        "effect_type": "particles",
        "improvement": 40.0,
        "experiment_id": "seed_particles_v1",
        "context_before": "Particles follow pure gravity or single-direction wind — motion looks mechanical and predictable",
        "context_after": "Wind + Turbulence + Drag combo creates natural atmospheric motion with gusts, swirls, and terminal velocity",
    },
    {
        "name": "object_instancing_for_debris",
        "issue": "debris particles render as dots or halos instead of solid objects",
        "code_snippet": """# Instance collection of varied debris meshes on particles
# Create debris collection with varied shapes
debris_col = bpy.data.collections.new("DebrisVariants")
bpy.context.scene.collection.children.link(debris_col)

for i in range(4):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.03)
    rock = bpy.context.active_object
    rock.name = f"Debris_{i}"
    # Deform for irregular shapes
    import bmesh
    from mathutils import Vector
    bm = bmesh.new()
    bm.from_mesh(rock.data)
    for v in bm.verts:
        v.co += Vector([random.uniform(-0.008, 0.008) for _ in range(3)])
    bm.to_mesh(rock.data)
    bm.free()
    debris_col.objects.link(rock)
    bpy.context.scene.collection.objects.unlink(rock)
    rock.hide_set(True)

# Configure particle render
settings.render_type = 'COLLECTION'
settings.instance_collection = debris_col
settings.use_collection_pick_random = True
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0
settings.angular_velocity_mode = 'RAND'
settings.angular_velocity_factor = 5.0
settings.particle_size = 1.0
settings.size_random = 0.7""",
        "effect_type": "particles",
        "improvement": 45.0,
        "experiment_id": "seed_particles_v1",
        "context_before": "Debris renders as point sprites (HALO) — looks like dots instead of solid chunks",
        "context_after": "Collection instancing with varied deformed ico_spheres + random rotation/size creates convincing debris shower",
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
    print("\nTest query: 'particle emitter sparks force field wind turbulence'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="particle emitter sparks force field wind turbulence",
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
    """Seed code pattern memory with particle techniques."""
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
            # Mark as applicable to all particle-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["particles", "emitter", "debris", "rain", "sparks", "confetti"]
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
        description="Seed particle technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED PARTICLE TECHNIQUES")
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
