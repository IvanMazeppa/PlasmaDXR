#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with cloth simulation technique knowledge.

Creates LLM-optimized markdown documents describing diverse cloth simulation
approaches in Blender 5.0, then uploads them to the rewritten manual vector store.
Also seeds the code pattern memory with proven cloth snippets.

This directly addresses the technique diversity problem: the Research Agent can
only discover what's in the vector store. Without these docs, it defaults to
basic cloth modifier every time.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_cloth_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_cloth_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_cloth_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_cloth_techniques.py --docs-only
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
    "cloth_technique_overview.md": """# Cloth Simulation Techniques in Blender 5.0
DocType: technique-guide
DocPath: techniques/cloth/overview
DocVersion: 5.0.1
PhysicsDomain: cloth
---

## Overview of Cloth Simulation Approaches

Blender 5.0 cloth simulation uses the `ClothModifier` on mesh objects. The modifier
exposes `ClothSettings` (simulation parameters) and `ClothCollisionSettings` (collision
behavior). All cloth simulations share this base — technique differences come from
mesh topology, pinning, force fields, and collision setup.

## Technique Comparison Table

| Technique | Realism | Complexity | Best For |
|-----------|---------|------------|----------|
| Basic Cloth (gravity drape) | MEDIUM | LOW | Tablecloths, curtains falling under gravity |
| Cloth + Pinning (vertex groups) | HIGH | LOW | Flags, banners, curtains on rods, held fabric |
| Cloth + Wind Force Field | HIGH | MEDIUM | Flags flapping, fabric in wind, outdoor scenes |
| Cloth + Collision Objects | HIGH | MEDIUM | Tablecloth on table, fabric draped over objects |
| Cloth + Rigid Body Interaction | VERY HIGH | HIGH | Fabric catching objects, character interaction |
| Cloth + Self-Collision | HIGH | MEDIUM | Folding, bunching, layered fabric |
| Cloth + Multiple Force Fields | VERY HIGH | HIGH | Complex outdoor scenes, turbulent environments |

## Decision Guide

- **Flag on a pole**: Cloth + Pinning (vertex group on pole edge) + Wind force field. Pin one edge,
  let the rest flap. Wind force field with noise for realistic flutter.
- **Tablecloth on table**: Cloth + Collision. Place cloth above table, add Collision modifier to table,
  let gravity drape the cloth. Enable self-collision for realistic folds.
- **Curtain on rod**: Cloth + Pinning (vertex group along top edge). Pin top row of vertices,
  let fabric hang. Optional wind for billowing curtain effect.
- **Fabric catching falling object**: Cloth + Collision + Rigid Body on the object. Cloth acts as
  a net/hammock, rigid body object falls into it.
- **Clothing on character**: Cloth + Pinning (shoulder/waist attachment) + Collision (character body
  as collision object). Most complex — requires good topology.
- **Sail or tarp**: Cloth + Pinning (corner/edge attachment) + Wind. Similar to flag but with
  different stiffness parameters for heavier material.

## Key API Reference

```python
# Add cloth modifier
cloth_mod = obj.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings              # bpy.types.ClothSettings
cc = cloth_mod.collision_settings    # bpy.types.ClothCollisionSettings

# Core simulation parameters
cs.quality = 5           # Simulation quality steps (1-80, default 5)
cs.mass = 0.3            # Mass per vertex in kg (default 0.3)
cs.air_damping = 1.0     # Air drag coefficient (0-10, default 1)
cs.time_scale = 1.0      # Simulation speed multiplier

# Stiffness (resistance to deformation)
cs.tension_stiffness = 15.0       # Stretch resistance (0-10000)
cs.compression_stiffness = 15.0   # Compression resistance (0-10000)
cs.bending_stiffness = 0.5        # Bend resistance (0-10000)
cs.shear_stiffness = 5.0          # Shear resistance (0-10000)

# Damping (energy loss during deformation)
cs.tension_damping = 5.0          # Stretch damping (0-50)
cs.compression_damping = 5.0      # Compression damping (0-50)
cs.bending_damping = 0.5          # Bend damping (0-50)
cs.shear_damping = 5.0            # Shear damping (0-50)

# Collision
cc.collision_quality = 2          # Collision detection quality (1-20)
cc.distance_min = 0.015           # Min collision distance
cc.use_self_collision = False     # Self-collision toggle
cc.self_distance_min = 0.015     # Min self-collision distance
cc.self_friction = 5.0            # Self-collision friction

# Pinning
cs.use_pin_cloth = True
cs.vertex_group_mass = "PinGroup"  # Vertex group name — pinned vertices don't move
cs.pin_stiffness = 1.0             # How strongly pinned (0-50)
```

## Mesh Topology Requirements

Cloth simulation quality depends heavily on mesh density:
- **Too few faces** (<100): Cloth looks blocky, cannot fold properly
- **Good density** (500-2000 faces): Smooth draping, realistic folds
- **High density** (2000-5000): Very detailed folds, slower simulation
- Use `bpy.ops.mesh.primitive_plane_add()` + subdivide for flat cloth
- Ensure QUADS (not triangles) for best cloth behavior
- Apply scale before adding cloth: `bpy.ops.object.transform_apply(scale=True)`

TECHNIQUE: Always subdivide the mesh BEFORE adding the cloth modifier. A 10-subdivision plane
(~10000 faces) gives excellent draping. For flags, 6-8 subdivisions is sufficient.

NOTE: `bpy.ops.object.effector_add(type='WIND')` for wind — NOT `forcefield_add` (does not exist).
NOTE: `Sheen Weight` not `Sheen` for fabric material shimmer in Principled BSDF.
NOTE: `Transmission Weight` not `Transmission` for translucent fabric.
""",

    "technique_cloth_flags_banners.md": """# Flag and Banner Cloth Simulation in Blender 5.0
DocType: technique-guide
DocPath: techniques/cloth/flags_banners
DocVersion: 5.0.1
PhysicsDomain: cloth
---

## Flag/Banner Simulation — Cloth + Pinning + Wind

Flags and banners are the most common cloth simulation. The key components:
1. Subdivided plane mesh (flag shape)
2. Vertex group pinning one edge (flagpole attachment)
3. Wind force field for flapping motion
4. Cloth settings tuned for fabric type

## Step 1: Create Flag Mesh with Proper Topology

```python
import bpy

# Create subdivided plane for flag
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 2.0))
flag = bpy.context.active_object
flag.name = "Flag"

# Subdivide for smooth cloth behavior
# 6 cuts = 49 faces per axis = ~2400 faces total — good for flags
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=6)
bpy.ops.object.mode_set(mode='OBJECT')

# Scale to flag proportions (wider than tall)
flag.scale = (1.5, 1.0, 1.0)
bpy.ops.object.transform_apply(scale=True)

# Rotate to hang vertically from Y axis
flag.rotation_euler = (1.5708, 0, 0)  # 90 degrees on X
bpy.ops.object.transform_apply(rotation=True)
```

## Step 2: Vertex Group for Pinning (Flagpole Edge)

```python
import bmesh

# Create vertex group for pinned edge (left edge = flagpole)
vg = flag.vertex_groups.new(name="PinGroup")

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(flag.data)
bm.verts.ensure_lookup_table()

# Find leftmost vertices (flagpole edge)
min_x = min(v.co.x for v in bm.verts)
threshold = 0.01  # Small tolerance

pin_indices = []
for v in bm.verts:
    if abs(v.co.x - min_x) < threshold:
        pin_indices.append(v.index)

bpy.ops.object.mode_set(mode='OBJECT')

# Assign weight 1.0 to pinned vertices
vg.add(pin_indices, 1.0, 'REPLACE')
```

## Step 3: Cloth Modifier with Flag Settings

```python
cloth_mod = flag.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings

# Fabric type parameters
cs.quality = 8               # Higher quality for flapping detail
cs.mass = 0.3                # Light fabric (kg per vertex)
cs.air_damping = 1.0         # Air resistance

# Stiffness — flags need moderate stiffness
cs.tension_stiffness = 15.0
cs.compression_stiffness = 15.0
cs.bending_stiffness = 0.5   # Low bending = more flexible flapping
cs.shear_stiffness = 5.0

# Damping — moderate for natural motion
cs.tension_damping = 5.0
cs.compression_damping = 5.0
cs.bending_damping = 0.5
cs.shear_damping = 5.0

# Pinning — use the vertex group
cs.use_pin_cloth = True
cs.vertex_group_mass = "PinGroup"
cs.pin_stiffness = 1.0       # Full pin strength
```

## Step 4: Wind Force Field

```python
# Add wind force field
bpy.ops.object.effector_add(type='WIND', location=(3, 0, 2.0))
wind = bpy.context.active_object
wind.name = "WindForce"

# Point wind at the flag
wind.rotation_euler = (0, 0, 3.14159)  # Blow in -X direction toward flag

# Wind settings
ff = wind.field
ff.strength = 25.0          # Wind force (tune: 10=light breeze, 30=strong wind, 60=gale)
ff.noise = 5.0              # Turbulence — CRITICAL for realistic flapping
ff.flow = 0.5               # How much wind follows surface
ff.seed = 42                # Random seed for noise pattern

# Optional: animate wind strength for gusts
wind.field.strength = 15.0
wind.field.keyframe_insert(data_path="strength", frame=1)
wind.field.strength = 40.0
wind.field.keyframe_insert(data_path="strength", frame=30)
wind.field.strength = 20.0
wind.field.keyframe_insert(data_path="strength", frame=60)
```

## Fabric Type Parameter Reference

| Parameter | Silk | Cotton | Canvas | Heavy Wool | Nylon |
|-----------|------|--------|--------|------------|-------|
| `mass` | 0.1 | 0.3 | 0.5 | 0.8 | 0.2 |
| `tension_stiffness` | 5.0 | 15.0 | 40.0 | 25.0 | 10.0 |
| `compression_stiffness` | 5.0 | 15.0 | 40.0 | 25.0 | 10.0 |
| `bending_stiffness` | 0.05 | 0.5 | 5.0 | 10.0 | 0.1 |
| `shear_stiffness` | 2.0 | 5.0 | 10.0 | 8.0 | 3.0 |
| `air_damping` | 2.0 | 1.0 | 0.5 | 0.3 | 1.5 |
| Visual character | Flows, ripples | Natural drape | Stiff, holds shape | Heavy folds | Light, snappy |

## Step 5: Flagpole (Optional Visual)

```python
# Create a simple cylinder flagpole
bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=4.0, location=(-1.5, 0, 2.0))
pole = bpy.context.active_object
pole.name = "Flagpole"
```

## Common Pitfalls

- **No wind** → Flag hangs limp. ALWAYS add `effector_add(type='WIND')` with `noise > 0`.
- **Wrong pin group** → All vertices pinned or none pinned. Verify vertex group name matches
  `cs.vertex_group_mass`. Use `flag.vertex_groups["PinGroup"]` to confirm it exists.
- **Too stiff** → Flag doesn't move. Lower `bending_stiffness` to 0.1-0.5 for flags.
- **No subdivision** → Blocky, unrealistic motion. Need 6+ subdivisions for smooth cloth.
- **Scale not applied** → Physics behaves incorrectly. Always `transform_apply(scale=True)` first.
- **Using `forcefield_add`** → Does NOT exist in Blender 5.0. Use `effector_add`.
- **Wind noise = 0** → Flag flaps in perfectly uniform pattern. Set noise >= 3.0 for realism.
- **Pin stiffness too low** → Pinned edge stretches. Keep `pin_stiffness >= 0.8`.

TECHNIQUE: For a national flag (e.g., Ireland tricolor), create the flag mesh, UV unwrap it,
and apply an image texture with the flag design. UV coordinates are preserved through cloth
simulation, so the texture stays correctly mapped as the flag flaps.

NOTE: Wind `strength` of 25-35 with `noise` of 5 gives natural outdoor flag motion. Increase
`noise` for gusty/turbulent conditions.
""",

    "technique_cloth_fabric_draping.md": """# Fabric Draping — Tablecloths, Curtains, Draped Fabric
DocType: technique-guide
DocPath: techniques/cloth/fabric_draping
DocVersion: 5.0.1
PhysicsDomain: cloth
---

## Gravity-Driven Draping Over Objects

Draping cloth (tablecloths, curtains, fabric over furniture) relies on:
1. Cloth object positioned ABOVE the collision surface
2. Collision modifier on the surface (table, rod, mannequin)
3. Self-collision enabled for realistic folds
4. Gravity pulls cloth down; collision prevents penetration

## Tablecloth Draping Over a Table

### Step 1: Create Table (Collision Object)

```python
import bpy

# Simple table top
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.0))
table = bpy.context.active_object
table.name = "Table"
table.scale = (1.0, 0.6, 0.03)  # Flat tabletop
bpy.ops.object.transform_apply(scale=True)

# Add collision modifier to table
col_mod = table.modifiers.new("Collision", type='COLLISION')
col_settings = col_mod.settings
col_settings.thickness_outer = 0.02   # Collision shell thickness
col_settings.thickness_inner = 0.01
col_settings.damping = 0.5            # Energy absorbed on impact
col_settings.friction = 0.5           # Surface friction
```

### Step 2: Create Cloth (Tablecloth)

```python
# Create tablecloth — larger than table so edges drape over
bpy.ops.mesh.primitive_plane_add(size=2.5, location=(0, 0, 1.5))
cloth_obj = bpy.context.active_object
cloth_obj.name = "Tablecloth"

# Subdivide for smooth draping — 10 cuts minimum for tablecloth
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=10)
bpy.ops.object.mode_set(mode='OBJECT')

# Position above table (gravity will drape it down)
cloth_obj.location.z = 1.2  # Slightly above table surface
```

### Step 3: Cloth Settings for Heavy Draping Fabric

```python
cloth_mod = cloth_obj.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings

# Heavy tablecloth fabric
cs.quality = 8
cs.mass = 0.5                # Heavier fabric for tablecloth
cs.air_damping = 1.0

# Moderate stiffness for natural draping
cs.tension_stiffness = 20.0
cs.compression_stiffness = 20.0
cs.bending_stiffness = 1.0   # Moderate — allows folding but not too floppy
cs.shear_stiffness = 5.0

cs.tension_damping = 5.0
cs.compression_damping = 5.0
cs.bending_damping = 0.5
cs.shear_damping = 5.0

# Enable self-collision for realistic fold overlaps
cc = cloth_mod.collision_settings
cc.use_self_collision = True
cc.self_distance_min = 0.015
cc.self_friction = 5.0
cc.collision_quality = 5      # Higher quality for table edge interaction
cc.distance_min = 0.015
```

### Step 4: Bake and Render a Settled Frame

```python
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 80    # Enough frames for cloth to settle
scene.frame_set(1)

# Bake cloth simulation
bpy.ops.ptcache.bake_all(bake=True)

# Render the settled frame (cloth has draped and stopped moving)
scene.frame_set(60)     # Frame where cloth is settled
```

## Curtain Draping (Pinned Top Edge)

Curtains combine draping with pinning — the top edge is fixed to a curtain rod.

```python
# Create curtain mesh
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 3.0))
curtain = bpy.context.active_object
curtain.name = "Curtain"

# Subdivide
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=8)
bpy.ops.object.mode_set(mode='OBJECT')

# Scale to curtain proportions (tall, not too wide)
curtain.scale = (0.8, 1.0, 1.5)
bpy.ops.object.transform_apply(scale=True)

# Rotate to hang vertically
curtain.rotation_euler = (1.5708, 0, 0)
bpy.ops.object.transform_apply(rotation=True)

# Create pin group for top edge
import bmesh
vg = curtain.vertex_groups.new(name="CurtainRod")
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(curtain.data)
bm.verts.ensure_lookup_table()

max_z = max(v.co.z for v in bm.verts)
pin_indices = [v.index for v in bm.verts if abs(v.co.z - max_z) < 0.01]
bpy.ops.object.mode_set(mode='OBJECT')
vg.add(pin_indices, 1.0, 'REPLACE')

# Cloth modifier
cloth_mod = curtain.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.quality = 6
cs.mass = 0.4
cs.bending_stiffness = 0.8
cs.tension_stiffness = 15.0
cs.use_pin_cloth = True
cs.vertex_group_mass = "CurtainRod"
cs.pin_stiffness = 1.0

# Self-collision for curtain folds
cc = cloth_mod.collision_settings
cc.use_self_collision = True
cc.self_distance_min = 0.015
```

## Light vs Heavy Fabric Draping Parameters

| Parameter | Light Silk | Cotton Tablecloth | Heavy Velvet | Canvas Tarp |
|-----------|-----------|-------------------|-------------|-------------|
| `mass` | 0.1 | 0.4 | 0.8 | 0.6 |
| `tension_stiffness` | 5.0 | 20.0 | 30.0 | 40.0 |
| `bending_stiffness` | 0.05 | 1.0 | 5.0 | 8.0 |
| `air_damping` | 2.0 | 1.0 | 0.5 | 0.3 |
| Drape character | Flows, pools | Natural folds | Heavy, deep folds | Stiff, angular |
| Self-collision | Optional | Recommended | Required | Optional |
| Settle time (frames) | 40 | 60 | 80 | 50 |

## Common Pitfalls

- **Cloth starts INSIDE collision object** → Explodes on frame 1. Always position cloth ABOVE surface.
- **No collision modifier on table** → Cloth falls through. Table needs `type='COLLISION'` modifier.
- **Self-collision disabled** → Cloth folds pass through each other, looks unnatural for tablecloth.
- **Too few subdivisions** → Blocky draping. 10+ cuts for tablecloth, 8+ for curtain.
- **Not enough frames to settle** → Cloth still moving at render time. Give 60-80 frames for draping.
- **Collision quality too low** → Cloth clips through table edges. Set `collision_quality >= 4`.

TECHNIQUE: For a tablecloth that hangs evenly on all sides, center the cloth plane precisely over
the table center and make the cloth 1.5-2x larger than the table surface.

NOTE: `collision_settings.distance_min` controls how far cloth stays from collision objects. Too small
= penetration, too large = visible gap. 0.01-0.02 is usually correct.
""",

    "technique_cloth_interaction.md": """# Cloth Interaction — Forces, Collision, Tearing
DocType: technique-guide
DocPath: techniques/cloth/interaction
DocVersion: 5.0.1
PhysicsDomain: cloth
---

## Cloth with Force Fields

Cloth responds to all Blender force field types. Common combinations:

### Wind Force Field

```python
# Add wind for outdoor cloth scenes
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 2))
wind = bpy.context.active_object
wind.name = "Wind"
wind.rotation_euler = (0, 0, 3.14159)  # Blow in -X

ff = wind.field
ff.strength = 30.0     # Force magnitude
ff.noise = 5.0         # Turbulence — essential for realism
ff.flow = 0.5          # Surface follow
ff.seed = 1
```

### Turbulence Force Field

```python
# Add turbulence for chaotic cloth motion (storm, explosion aftermath)
bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
turb = bpy.context.active_object
turb.name = "Turbulence"

ff = turb.field
ff.strength = 10.0     # Force magnitude
ff.size = 1.0          # Turbulence scale (larger = bigger swirls)
ff.noise = 2.0         # Additional noise
ff.flow = 0.5
```

### Vortex Force Field

```python
# Add vortex for spinning/swirling cloth (tornado, whirlpool)
bpy.ops.object.effector_add(type='VORTEX', location=(0, 0, 1))
vortex = bpy.context.active_object
vortex.name = "Vortex"

ff = vortex.field
ff.strength = 15.0     # Rotational force
ff.flow = 1.0          # Follow surface strongly
```

### Multiple Force Fields (Outdoor Scene)

```python
# Realistic outdoor environment: wind + turbulence
bpy.ops.object.effector_add(type='WIND', location=(5, 0, 2))
wind = bpy.context.active_object
wind.field.strength = 20.0
wind.field.noise = 3.0
wind.rotation_euler = (0, 0, 3.14159)

bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
turb = bpy.context.active_object
turb.field.strength = 5.0
turb.field.size = 2.0
```

## Cloth with Collision Objects

Any mesh can be a collision object for cloth. The Collision modifier defines the
interaction surface.

### Setting Up Collision Objects

```python
# Add collision to any mesh object
collision_obj = bpy.data.objects["MyObject"]
bpy.context.view_layer.objects.active = collision_obj
collision_obj.select_set(True)

col_mod = collision_obj.modifiers.new("Collision", type='COLLISION')
col = col_mod.settings
col.thickness_outer = 0.02   # Outward collision shell
col.thickness_inner = 0.01   # Inward collision shell
col.damping = 0.5            # Energy absorption (0 = bouncy, 1 = absorbs all)
col.friction = 0.5           # Surface friction (affects cloth sliding)
```

### Animated Collision Objects

```python
# Collision objects can be animated — cloth reacts dynamically
# Example: a ball pushing through a hanging cloth
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, -2, 2))
ball = bpy.context.active_object
ball.name = "Ball"

# Add collision
ball.modifiers.new("Collision", type='COLLISION')

# Animate ball moving through cloth
ball.location = (0, -2, 2)
ball.keyframe_insert(data_path="location", frame=1)
ball.location = (0, 2, 2)
ball.keyframe_insert(data_path="location", frame=50)
```

## Cloth Catching a Falling Object

Cloth acts as a net or hammock catching a rigid body object.

```python
import bpy
import bmesh
from mathutils import Vector

scene = bpy.context.scene

# === NET/HAMMOCK CLOTH ===
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 2.0))
net = bpy.context.active_object
net.name = "CatchNet"

# High subdivision for deformation
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=10)
bpy.ops.object.mode_set(mode='OBJECT')

# Pin four corners
vg = net.vertex_groups.new(name="Corners")
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(net.data)
bm.verts.ensure_lookup_table()

# Find corner vertices (extremes of X and Y)
corners = []
for v in bm.verts:
    if (abs(abs(v.co.x) - 1.5) < 0.01 and abs(abs(v.co.y) - 1.5) < 0.01):
        corners.append(v.index)

bpy.ops.object.mode_set(mode='OBJECT')
vg.add(corners, 1.0, 'REPLACE')

# Cloth modifier — stretchy net
cloth_mod = net.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.quality = 8
cs.mass = 0.3
cs.tension_stiffness = 10.0    # Some stretch for catching
cs.compression_stiffness = 10.0
cs.bending_stiffness = 0.2     # Flexible
cs.use_pin_cloth = True
cs.vertex_group_mass = "Corners"
cs.pin_stiffness = 1.0

cc = cloth_mod.collision_settings
cc.collision_quality = 5
cc.distance_min = 0.01

# === FALLING OBJECT ===
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 5.0))
ball = bpy.context.active_object
ball.name = "FallingBall"

# Rigid body — active, falls under gravity
bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = ball.rigid_body
rb.mass = 1.0
rb.collision_shape = 'SPHERE'
rb.friction = 0.5
rb.restitution = 0.2

# The collision between rigid body and cloth is automatic —
# Blender's cloth solver detects collision with rigid body objects

# Rigid body world
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20

# Bake
scene.frame_start = 1
scene.frame_end = 120
bpy.ops.ptcache.bake_all(bake=True)
```

## Cloth Tearing (Sewing Springs)

Blender 5.0 supports cloth tearing through `use_sewing_springs` and strain limits.
When strain exceeds the threshold, the cloth mesh splits at that point.

```python
cloth_mod = cloth_obj.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings

# Enable sewing springs (required for tearing behavior)
cs.use_sewing_springs = True
cs.sewing_force_max = 0.0    # 0 = no sewing force, allows tearing

# The cloth will tear when internal forces exceed the stiffness parameters
# Lower stiffness = tears more easily
cs.tension_stiffness = 5.0    # Low tension = tears under stretch
```

## Collision Quality Tuning

| `collision_quality` | Use Case | Performance |
|--------------------|----------|-------------|
| 1 | Fast preview | Low — some penetration |
| 2-3 | Default | Medium — good for most cases |
| 5 | Table edges, sharp collision objects | Higher — better edge handling |
| 8-10 | Complex collision with many objects | Slow — only for hero shots |

NOTE: `bpy.ops.object.effector_add(type='WIND')` — NOT `forcefield_add`.
NOTE: Collision modifier is separate from Rigid Body — they serve different purposes.
Collision is for soft body/cloth/fluid interaction. Rigid Body is for physics simulation.
NOTE: For cloth-rigid body interaction, the cloth uses Collision detection and the
falling object uses Rigid Body. Both systems run simultaneously.
""",

    "technique_cloth_materials.md": """# Cloth Materials and Rendering in Blender 5.0
DocType: technique-guide
DocPath: techniques/cloth/materials
DocVersion: 5.0.1
PhysicsDomain: cloth
---

## Principled BSDF for Fabric Materials

Fabric materials in Blender 5.0 use Principled BSDF with specific parameter ranges
to achieve realistic cloth appearance. Key properties: Sheen Weight for fabric shimmer,
Roughness for surface texture, Subsurface for translucency.

## Material Setup Patterns

### Cotton/Linen (Matte, Natural)

```python
import bpy

mat = bpy.data.materials.new("M_Cotton")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

bsdf.inputs["Base Color"].default_value = (0.8, 0.75, 0.7, 1)  # Natural off-white
bsdf.inputs["Roughness"].default_value = 0.9       # Very rough — matte fabric
bsdf.inputs["Sheen Weight"].default_value = 0.3     # Subtle fabric sheen
bsdf.inputs["Sheen Tint"].default_value = 0.5       # Warm-tinted sheen
bsdf.inputs["Specular IOR Level"].default_value = 0.2  # Low specular for fabric

nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
```

### Silk (Glossy, Shimmering)

```python
mat = bpy.data.materials.new("M_Silk")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

bsdf.inputs["Base Color"].default_value = (0.6, 0.1, 0.15, 1)  # Deep red silk
bsdf.inputs["Roughness"].default_value = 0.3        # Smooth — silky shine
bsdf.inputs["Sheen Weight"].default_value = 0.8      # Strong fabric shimmer
bsdf.inputs["Sheen Tint"].default_value = 0.8        # Color-tinted sheen
bsdf.inputs["Specular IOR Level"].default_value = 0.5
bsdf.inputs["Anisotropic"].default_value = 0.3       # Directional highlights

nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
```

### Velvet (Soft, Color-Shifting)

```python
mat = bpy.data.materials.new("M_Velvet")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

bsdf.inputs["Base Color"].default_value = (0.15, 0.0, 0.25, 1)  # Deep purple
bsdf.inputs["Roughness"].default_value = 0.8        # Soft matte base
bsdf.inputs["Sheen Weight"].default_value = 1.0      # Maximum sheen — velvet signature
bsdf.inputs["Sheen Tint"].default_value = 1.0        # Full color tint
bsdf.inputs["Sheen Roughness"].default_value = 0.4   # Sheen spread
bsdf.inputs["Specular IOR Level"].default_value = 0.1  # Minimal specular

nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
```

### Translucent Fabric (Sheer Curtain, Thin Silk)

```python
mat = bpy.data.materials.new("M_Sheer")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

bsdf.inputs["Base Color"].default_value = (0.95, 0.93, 0.9, 1)  # White/cream
bsdf.inputs["Roughness"].default_value = 0.6
bsdf.inputs["Transmission Weight"].default_value = 0.3  # Partially see-through
bsdf.inputs["Sheen Weight"].default_value = 0.2
bsdf.inputs["Alpha"].default_value = 0.8                # Slightly transparent

nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# Enable alpha blending for transparency
mat.blend_method = 'BLEND'    # EEVEE setting
mat.shadow_method = 'CLIP'
```

## Fabric Material Parameter Reference

| Parameter | Cotton | Silk | Velvet | Canvas | Wool | Satin |
|-----------|--------|------|--------|--------|------|-------|
| `Roughness` | 0.85 | 0.3 | 0.8 | 0.95 | 0.9 | 0.2 |
| `Sheen Weight` | 0.3 | 0.8 | 1.0 | 0.1 | 0.4 | 0.7 |
| `Sheen Tint` | 0.3 | 0.8 | 1.0 | 0.2 | 0.5 | 0.6 |
| `Specular IOR Level` | 0.2 | 0.5 | 0.1 | 0.15 | 0.2 | 0.5 |
| `Anisotropic` | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 | 0.5 |
| `Subsurface Weight` | 0.0 | 0.05 | 0.0 | 0.0 | 0.02 | 0.0 |

## Texture Mapping on Cloth

UV coordinates are preserved through cloth simulation. Apply textures BEFORE running
the cloth sim — the UVs will deform naturally with the mesh.

```python
# UV unwrap the cloth mesh before adding cloth modifier
bpy.context.view_layer.objects.active = cloth_obj
cloth_obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.unwrap(method='ANGLE_BASED')
bpy.ops.object.mode_set(mode='OBJECT')

# Add image texture
tex_node = nt.nodes.new("ShaderNodeTexImage")
tex_node.image = bpy.data.images.load("/path/to/fabric_texture.png")
nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

# Add texture coordinate and mapping nodes for control
tc = nt.nodes.new("ShaderNodeTexCoord")
mapping = nt.nodes.new("ShaderNodeMapping")
mapping.inputs["Scale"].default_value = (2.0, 2.0, 2.0)  # Tile the texture
nt.links.new(tc.outputs["UV"], mapping.inputs["Vector"])
nt.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
```

## Procedural Fabric Texture (No Image Required)

```python
# Noise texture for woven fabric look
noise = nt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 50.0   # Fine detail
noise.inputs["Detail"].default_value = 10.0
noise.inputs["Roughness"].default_value = 0.7

# Color ramp to map noise to fabric colors
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.7, 0.65, 0.6, 1)  # Light thread
ramp.color_ramp.elements[1].color = (0.5, 0.45, 0.4, 1)  # Dark thread

nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
```

## Blender 5.0 Material Input Names

| Correct (5.0) | WRONG (old name) |
|----------------|-------------------|
| `Sheen Weight` | `Sheen` |
| `Sheen Roughness` | `Sheen Roughness` (unchanged) |
| `Sheen Tint` | `Sheen Tint` (unchanged) |
| `Transmission Weight` | `Transmission` |
| `Specular IOR Level` | `Specular` |
| `Coat Weight` | `Coat` |

NOTE: Using old input names causes `KeyError`. Always use Blender 5.0 names.
NOTE: `ShaderNodeMix` replaces `ShaderNodeMixRGB`. Use `data_type='RGBA'` for color mixing.
NOTE: `ShaderNodeSeparateColor` replaces `ShaderNodeSeparateRGB`.

TECHNIQUE: For the most realistic fabric, combine Principled BSDF sheen with a subtle
bump/normal map. Even a procedural noise bump at 0.01-0.05 strength adds micro-surface
detail that catches light like real woven fabric.
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "flag_with_wind_complete",
        "issue": "flag cloth simulation hangs limp with no motion; needs wind force field for realistic flapping",
        "code_snippet": """# Complete flag with wind setup
# 1. Create subdivided plane for flag mesh
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 2.0))
flag = bpy.context.active_object
flag.name = "Flag"
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=6)
bpy.ops.object.mode_set(mode='OBJECT')
flag.scale = (1.5, 1.0, 1.0)
bpy.ops.object.transform_apply(scale=True)

# 2. Pin left edge (flagpole attachment)
vg = flag.vertex_groups.new(name="PinGroup")
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(flag.data)
bm.verts.ensure_lookup_table()
min_x = min(v.co.x for v in bm.verts)
pin_indices = [v.index for v in bm.verts if abs(v.co.x - min_x) < 0.01]
bpy.ops.object.mode_set(mode='OBJECT')
vg.add(pin_indices, 1.0, 'REPLACE')

# 3. Cloth modifier
cloth_mod = flag.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.quality = 8
cs.mass = 0.3
cs.tension_stiffness = 15.0
cs.compression_stiffness = 15.0
cs.bending_stiffness = 0.5
cs.air_damping = 1.0
cs.use_pin_cloth = True
cs.vertex_group_mass = "PinGroup"
cs.pin_stiffness = 1.0

# 4. Wind force field — essential for flapping
bpy.ops.object.effector_add(type='WIND', location=(3, 0, 2))
wind = bpy.context.active_object
wind.rotation_euler = (0, 0, 3.14159)
wind.field.strength = 25.0
wind.field.noise = 5.0    # Critical for realistic flutter""",
        "effect_type": "cloth",
        "improvement": 45.0,
        "experiment_id": "seed_cloth_v1",
        "context_before": "Flag cloth hangs limp under gravity with no wind — looks dead, no motion",
        "context_after": "Wind force field with noise=5 creates natural flapping; pinned edge stays attached to pole",
    },
    {
        "name": "tablecloth_draping_collision",
        "issue": "tablecloth falls through table or doesn't drape properly over edges",
        "code_snippet": """# Tablecloth draping with collision
# 1. Table with collision modifier
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.0))
table = bpy.context.active_object
table.name = "Table"
table.scale = (1.0, 0.6, 0.03)
bpy.ops.object.transform_apply(scale=True)
col_mod = table.modifiers.new("Collision", type='COLLISION')
col_mod.settings.thickness_outer = 0.02
col_mod.settings.damping = 0.5

# 2. Cloth above table — must start ABOVE collision surface
bpy.ops.mesh.primitive_plane_add(size=2.5, location=(0, 0, 1.2))
cloth = bpy.context.active_object
cloth.name = "Tablecloth"
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=10)
bpy.ops.object.mode_set(mode='OBJECT')

# 3. Cloth settings with self-collision for folds
cloth_mod = cloth.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.quality = 8
cs.mass = 0.5
cs.tension_stiffness = 20.0
cs.bending_stiffness = 1.0
cc = cloth_mod.collision_settings
cc.use_self_collision = True
cc.collision_quality = 5
cc.distance_min = 0.015""",
        "effect_type": "cloth",
        "improvement": 40.0,
        "experiment_id": "seed_cloth_v1",
        "context_before": "Cloth falls through table — missing Collision modifier on table surface",
        "context_after": "Collision modifier on table + cloth starting above surface = proper draping with edge folds",
    },
    {
        "name": "cloth_pinning_vertex_group",
        "issue": "cloth simulation pins all vertices or no vertices; vertex group setup incorrect",
        "code_snippet": """# Correct vertex group pinning for cloth
import bmesh

# Create vertex group on cloth object
vg = cloth_obj.vertex_groups.new(name="PinGroup")

# Select vertices to pin (example: top edge)
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(cloth_obj.data)
bm.verts.ensure_lookup_table()

# Find edge vertices by position
max_z = max(v.co.z for v in bm.verts)
pin_indices = [v.index for v in bm.verts if abs(v.co.z - max_z) < 0.01]

bpy.ops.object.mode_set(mode='OBJECT')

# Assign weight 1.0 to pinned vertices
vg.add(pin_indices, 1.0, 'REPLACE')

# Configure cloth to use pin group
cloth_mod = cloth_obj.modifiers.new("Cloth", type='CLOTH')
cs = cloth_mod.settings
cs.use_pin_cloth = True
cs.vertex_group_mass = "PinGroup"  # Must match vertex group name exactly
cs.pin_stiffness = 1.0             # 1.0 = fully rigid pin""",
        "effect_type": "cloth",
        "improvement": 35.0,
        "experiment_id": "seed_cloth_v1",
        "context_before": "Pinning fails — either all verts pinned (cloth frozen) or none pinned (cloth falls)",
        "context_after": "Correct workflow: create vertex group, find verts by position, assign weight, set vertex_group_mass to group name",
    },
    {
        "name": "cloth_collision_setup",
        "issue": "cloth passes through objects or collision detection is unreliable",
        "code_snippet": """# Proper collision setup for cloth interaction
# Step 1: Add Collision modifier to obstacle (NOT Rigid Body)
obstacle = bpy.data.objects["Obstacle"]
bpy.context.view_layer.objects.active = obstacle
obstacle.select_set(True)
col_mod = obstacle.modifiers.new("Collision", type='COLLISION')
col = col_mod.settings
col.thickness_outer = 0.02    # Collision shell outward
col.thickness_inner = 0.01    # Collision shell inward
col.damping = 0.5             # Energy absorption
col.friction = 0.5            # Surface friction

# Step 2: Tune cloth collision settings
cloth_mod = cloth_obj.modifiers["Cloth"]
cc = cloth_mod.collision_settings
cc.collision_quality = 5       # Higher = better detection (1-20)
cc.distance_min = 0.015        # Min distance from collision surface
cc.use_self_collision = True   # Prevent self-intersection
cc.self_distance_min = 0.015
cc.self_friction = 5.0

# NOTE: Collision modifier is for cloth/softbody/fluid interaction
# Rigid Body is for rigid body physics — different system entirely""",
        "effect_type": "cloth",
        "improvement": 30.0,
        "experiment_id": "seed_cloth_v1",
        "context_before": "Cloth passes through objects — using Rigid Body instead of Collision modifier, or collision_quality too low",
        "context_after": "Collision modifier on obstacle with thickness_outer=0.02, cloth collision_quality=5, distance_min=0.015 gives reliable interaction",
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
    print("\nTest query: 'cloth simulation flag wind pinning vertex group'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="cloth simulation flag wind pinning vertex group",
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
    """Seed code pattern memory with cloth techniques."""
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
            # Mark as applicable to all cloth-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["cloth", "fabric", "flag", "curtain", "draping"]
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
        description="Seed cloth technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED CLOTH TECHNIQUES")
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
