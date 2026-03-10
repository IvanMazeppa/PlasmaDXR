#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with destruction/shatter technique knowledge.

Creates LLM-optimized markdown documents describing diverse glass shattering
approaches in Blender 5.0, then uploads them to the rewritten manual vector store.
Also seeds the code pattern memory with proven destruction snippets.

This directly addresses the technique diversity problem: the Research Agent can
only discover what's in the vector store. Without these docs, it defaults to
basic prefracture rigid body every time.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_destruction_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_destruction_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_destruction_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_destruction_techniques.py --docs-only
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
    "destruction_technique_overview.md": """# Glass Destruction Techniques in Blender 5.0
DocType: technique-guide
DocPath: techniques/destruction/overview
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Overview of Glass Shattering Approaches

Blender 5.0 offers multiple approaches for glass destruction, each with different
realism/complexity tradeoffs. Do NOT default to the simplest approach — choose
based on the scene requirements.

## Technique Comparison Table

| Technique | Realism | Complexity | Best For |
|-----------|---------|------------|----------|
| Cell Fracture + Constraints (RECOMMENDED) | VERY HIGH | MEDIUM | Impact-driven glass breaks — brick through window, bullet hit |
| Cell Fracture Addon + Rigid Body | HIGH | LOW | Quick shatter with Voronoi fracture patterns |
| Manual Prefractured Rigid Body | LOW | MEDIUM | Only when Cell Fracture addon unavailable |
| Geometry Nodes Voronoi Fracture | HIGH | HIGH | Procedural fracture with fine control over shard shapes |
| Particle System Debris | MEDIUM | LOW | Secondary debris, sparks, small fragments |

IMPORTANT: Always use `bpy.ops.object.add_fracture_cell_objects()` (Cell Fracture addon) to create
shard geometry. Do NOT manually subdivide/cut meshes — Cell Fracture produces far more realistic
Voronoi fracture patterns and is the standard industry approach.

## Decision Guide

- **Impact-driven shattering** (brick through window, bullet hit): Cell Fracture addon to create shards + FIXED constraints with `use_breaking=True` between adjacent shards. Distance-based `breaking_threshold` (weaker near impact) creates realistic radial crack propagation.
- **Explosion/blast shattering**: Cell Fracture with high source_limit (100+) + ALL rigid bodies active from frame 1 + force field. Add particle system for fine debris.
- **Slow/artistic break**: Cell Fracture + Geometry Nodes for precise shard control. Animate constraint breaking thresholds over time for slow-motion effect.
- **Background/secondary breaks**: Cell Fracture addon with fewer shards (20-50) for distant objects.
- **DO NOT** manually cut meshes with Boolean or bmesh to create shards — use Cell Fracture addon.

TECHNIQUE: For maximum realism, ALWAYS combine primary shard simulation (rigid body) with secondary debris (particle system). Real glass produces both large shards and fine powder/chips.

NOTE: Cell Fracture is an ADDON — must be enabled via `bpy.ops.preferences.addon_enable(module='object_cell_fracture')` before use. In Blender 5.0, use `bpy.ops.object.add_fracture_cell_objects()` after enabling.
""",

    "technique_constraint_breaking.md": """# Constraint-Based Glass Shattering (Recommended for Realism)
DocType: technique-guide
DocPath: techniques/destruction/constraint_breaking
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Concept

Pre-fracture a glass mesh into shards, connect adjacent shards with FIXED rigid body constraints that have `use_breaking=True`. When impact force exceeds `breaking_threshold`, constraints break and shards separate — creating realistic progressive fracture propagation.

## Why This Is Better Than Basic Prefracture

Basic prefracture: all shards are independent rigid bodies from frame 1 → they ALL fall simultaneously → looks like a pre-broken object collapsing, not glass shattering.

Constraint-based: shards start CONNECTED → only break where force is applied → radial crack propagation from impact point → physically accurate.

## Step-by-Step Workflow

1. **Create glass mesh** — Plane or box scaled to window dimensions
2. **Fracture into shards** — Cell Fracture or Voronoi Texture + Boolean
3. **Set all shards as Active rigid body** — `obj.rigid_body.type = 'ACTIVE'`
4. **Connect adjacent shards with FIXED constraints** — Each pair of neighboring shards gets a constraint empty
5. **Configure breaking** — `constraint.rigid_body_constraint.use_breaking = True`, `breaking_threshold` varies by distance from impact
6. **Add projectile** — Kinematic rigid body animated to hit the glass
7. **Bake rigid body simulation**

## Key API — bpy.types.RigidBodyConstraint

```python
# Create constraint empty between two shards
bpy.ops.object.empty_add(type='PLAIN_AXES', location=midpoint)
empty = bpy.context.active_object
empty.name = f"Constraint_{i}"

# Add rigid body constraint
bpy.ops.rigidbody.constraint_add()
rbc = empty.rigid_body_constraint
rbc.type = 'FIXED'                    # Glue shards together
rbc.object1 = shard_a                 # First shard
rbc.object2 = shard_b                 # Second shard
rbc.use_breaking = True               # Enable breaking!
rbc.breaking_threshold = 25.0         # Impulse threshold (tune per scene)
rbc.disable_collisions = False        # Allow shards to collide after break
```

## Breaking Threshold Tuning

- `breaking_threshold` is impulse magnitude (force × time) that breaks the constraint
- Higher → stronger bond → needs more force to break
- **Glass window**: 15-40 (relatively brittle)
- **Tempered glass**: 50-100 (stronger but shatters completely when broken)
- **Thick glass block**: 80-150
- **Variable threshold for realism**: Shards near impact point should have LOWER threshold (break first), shards at edges should have HIGHER threshold (break later or not at all)

```python
# Variable breaking threshold based on distance from impact
impact_point = Vector((0, 0, 0.5))  # Where projectile hits
for empty in constraint_empties:
    rbc = empty.rigid_body_constraint
    dist = (empty.location - impact_point).length
    # Closer to impact = lower threshold = breaks easier
    rbc.breaking_threshold = 10.0 + dist * 40.0
```

## Rigid Body World Settings for Glass

```python
scene = bpy.context.scene
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10   # Higher = more accurate collision detection
rbw.solver_iterations = 20    # Higher = more stable stacking/resting
# NOTE: use substeps_per_frame, NOT steps_per_second (removed in Blender 5.0)
```

## Common Pitfalls

NOTE: `RigidBodyWorld.steps_per_second` does NOT exist in Blender 5.0. Use `substeps_per_frame` instead.
NOTE: Constraint empties must be in the rigid body collection to work.
NOTE: `bpy.ops.rigidbody.constraint_add()` requires the empty to be the active object in OBJECT mode.
NOTE: Too many constraints (>500) will slow simulation significantly. For a window, 100-200 constraints between major shards is sufficient.

TECHNIQUE: For radial crack patterns, create a Voronoi-based fracture centered on the impact point with higher cell density near impact and lower density at edges.
""",

    "technique_cell_fracture_workflow.md": """# Cell Fracture Addon Workflow for Glass (RECOMMENDED APPROACH)
DocType: technique-guide
DocPath: techniques/destruction/cell_fracture
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Cell Fracture Addon in Blender 5.0 — PRIMARY Fracture Method

The Cell Fracture addon (`object_cell_fracture`) creates Voronoi-based fracture patterns
from mesh objects. This is the RECOMMENDED way to create shattered glass in Blender 5.0.

IMPORTANT: Always use Cell Fracture to generate shard geometry. Do NOT manually subdivide
or Boolean-cut meshes — Cell Fracture produces physically-plausible Voronoi fracture
patterns that look realistic. After creating shards with Cell Fracture, add rigid body
physics and optionally FIXED constraints with `use_breaking=True` for impact-driven breaks.

## Enabling Cell Fracture (Required in Headless/CLI Mode)

The Cell Fracture addon is installed as a Blender extension but may not be enabled by
default in headless mode. Always enable it at the start of your script:

```python
# Enable the Cell Fracture extension (REQUIRED in headless mode)
import addon_utils
addon_utils.enable('object_cell_fracture', default_set=True, persistent=True)

# Select the glass object
glass_obj = bpy.data.objects["GlassPane"]
bpy.context.view_layer.objects.active = glass_obj
glass_obj.select_set(True)

# Run cell fracture
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},    # Source points from object's own particles
    source_limit=50,             # Number of fracture points → number of shards
    source_noise=0.05,           # Randomness in fracture pattern
    cell_scale=(1, 1, 1),        # Scale of individual cells
    margin=0.001,                # Gap between shards (tiny for glass)
    use_smooth_faces=False,      # Glass has sharp fracture edges
    use_data_match=True,         # Copy materials from original
    use_island_split=True,       # Split disconnected pieces
    recursion=0,                 # 0 = no recursive fracture
    recursion_chance=0.5,        # Chance of recursive split (if recursion > 0)
)
```

## Post-Fracture Setup

After Cell Fracture creates shard objects, they need rigid body physics:

```python
# Get all shard objects (Cell Fracture names them with _cell suffix)
shards = [obj for obj in bpy.data.objects if obj.name.startswith("GlassPane_cell")]

for shard in shards:
    bpy.context.view_layer.objects.active = shard
    shard.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    shard.rigid_body.mass = 0.1           # Light glass shards
    shard.rigid_body.friction = 0.4       # Glass friction
    shard.rigid_body.restitution = 0.3    # Some bounce
    shard.rigid_body.collision_shape = 'CONVEX_HULL'  # Fast + good for shards
    shard.rigid_body.linear_damping = 0.04
    shard.rigid_body.angular_damping = 0.1
    shard.select_set(False)
```

## Fracture Source Point Control

For impact-driven fracture, concentrate source points at the impact location:

```python
# Add a particle system to control fracture point distribution
# More particles near impact = more/smaller shards near impact
ps = glass_obj.modifiers.new("FracturePoints", type='PARTICLE_SYSTEM')
psettings = ps.particle_system.settings
psettings.count = 80
psettings.frame_start = 1
psettings.frame_end = 1
psettings.lifetime = 1
psettings.emit_from = 'FACE'
# Use vertex group weight to concentrate near impact
# (requires vertex group with weight gradient from impact point)
```

## Glass Material for Shards

```python
# Principled BSDF glass material
mat = bpy.data.materials.new("M_Glass")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.95, 0.97, 0.95, 1)  # Slight green tint
bsdf.inputs["Roughness"].default_value = 0.02
bsdf.inputs["Transmission Weight"].default_value = 1.0
bsdf.inputs["IOR"].default_value = 1.52  # Float glass IOR
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
```

NOTE: Use `Transmission Weight` not `Transmission` (Blender 5.0 rename).
NOTE: Use `Specular IOR Level` not `Specular` (Blender 5.0 rename).
NOTE: Cell Fracture with recursion > 1 creates exponentially many objects — use with caution.

TECHNIQUE: For realistic glass, use recursion=1 with recursion_chance=0.3 to create some smaller sub-shards near the fracture edges, mimicking real glass breakage patterns.
""",

    "technique_cell_fracture_constraints_combined.md": """# Cell Fracture + Constraints: Complete Hero Destruction Workflow (BEST APPROACH)
DocType: technique-guide
DocPath: techniques/destruction/cell_fracture_constraints_combined
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## The Recommended Approach for Impact-Driven Glass Destruction

This technique combines Cell Fracture addon (for realistic shard geometry) with rigid body
FIXED constraints (for impact-driven progressive breaking). This is the HIGHEST REALISM
approach for scenes where a projectile strikes glass.

IMPORTANT: This technique requires the Cell Fracture addon. Enable it with:
`addon_utils.enable('object_cell_fracture', default_set=True, persistent=True)`

## Complete Workflow

### Step 1: Create Glass Pane
```python
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 1.5))
glass = bpy.context.active_object
glass.name = "GlassPane"
glass.scale = (1.0, 0.003, 0.75)  # Thin for window glass
bpy.ops.object.transform_apply(scale=True)
```

### Step 2: Cell Fracture — Create Shards
```python
import addon_utils
addon_utils.enable('object_cell_fracture', default_set=True, persistent=True)

bpy.context.view_layer.objects.active = glass
glass.select_set(True)
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},
    source_limit=60,          # 40-80 for hero glass
    source_noise=0.05,
    cell_scale=(1, 1, 1),
    margin=0.0005,            # Tiny gap between shards
    use_smooth_faces=False,   # Sharp glass edges
    use_data_match=True,
    use_island_split=True,
    use_interior_vgroup=True,
)
```

### Step 3: Rigid Body Physics on Shards
```python
shards = [o for o in bpy.data.objects if o.name.startswith("GlassPane_cell")]
for shard in shards:
    bpy.context.view_layer.objects.active = shard
    shard.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    rb = shard.rigid_body
    rb.mass = 0.05 + random.uniform(0, 0.15)
    rb.collision_shape = 'CONVEX_HULL'
    rb.friction = 0.4
    rb.restitution = 0.3
    rb.linear_damping = 0.04
    rb.angular_damping = 0.1
    shard.select_set(False)
```

### Step 4: FIXED Constraints Between Adjacent Shards
This is what makes the break look realistic — shards are held together until impact force
exceeds the breaking threshold. Shards near impact break first (lower threshold).

```python
from mathutils import Vector
impact_point = Vector((0, 0, 1.5))  # Where projectile hits

# Find adjacent pairs (shards whose centers are close)
max_neighbor_dist = 0.3  # Adjust based on shard size
pairs = []
for i, a in enumerate(shards):
    for b in shards[i+1:]:
        if (a.location - b.location).length < max_neighbor_dist:
            pairs.append((a, b))

for i, (a, b) in enumerate(pairs):
    midpoint = (a.location + b.location) / 2
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=midpoint)
    empty = bpy.context.active_object
    empty.name = f"RBC_{i:03d}"
    bpy.ops.rigidbody.constraint_add()
    rbc = empty.rigid_body_constraint
    rbc.type = 'FIXED'
    rbc.object1 = a
    rbc.object2 = b
    rbc.use_breaking = True
    dist = (midpoint - impact_point).length
    rbc.breaking_threshold = 8.0 + dist * 35.0  # Weak near impact, strong at edges
```

### Step 5: Kinematic Projectile
```python
bpy.ops.mesh.primitive_cube_add(size=0.25, location=(-3, 0, 1.5))
brick = bpy.context.active_object
brick.name = "Brick"
bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = brick.rigid_body
rb.mass = 2.5
rb.collision_shape = 'BOX'
rb.kinematic = True

# Animated approach — kinematic then active at impact
brick.location = (-3, 0, 1.5)
brick.keyframe_insert(data_path="location", frame=1)
rb.keyframe_insert(data_path="kinematic", frame=1)

brick.location = (-0.1, 0, 1.5)  # Just before glass
brick.keyframe_insert(data_path="location", frame=4)

rb.kinematic = False  # Switch to physics at impact
rb.keyframe_insert(data_path="kinematic", frame=5)
```

### Step 6: Rigid Body World Settings
```python
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20
```

## Why This Works
- Cell Fracture creates physically-plausible Voronoi fracture geometry
- FIXED constraints hold shards together like real glass
- Distance-based breaking threshold creates radial crack propagation from impact
- Kinematic→active projectile gives controlled approach + realistic physics transfer
- Combined: shards near impact explode outward, distant shards crack and collapse

## Common Mistakes
- DO NOT manually subdivide/cut mesh — use Cell Fracture addon
- DO NOT use basic prefracture without constraints — all shards fall at once
- DO NOT forget `addon_utils.enable('object_cell_fracture')` in headless mode
- DO NOT set all rigid bodies to active without constraints — no cohesion
""",

    "technique_geometry_nodes_fracture.md": """# Geometry Nodes Voronoi Fracture
DocType: technique-guide
DocPath: techniques/destruction/geometry_nodes_fracture
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Geometry Nodes Procedural Fracture

Geometry Nodes provides the most control over fracture patterns. Use Voronoi Texture
to define fracture cell boundaries, then Boolean operations to cut the mesh.

## Advantages Over Cell Fracture

- **Controllable density gradient** — more shards at impact, fewer at edges
- **Repeatable** — same node tree produces same result (deterministic with same seed)
- **Animated** — fracture pattern can change over time
- **Custom shard shapes** — not limited to Voronoi cells

## Basic Voronoi Fracture Setup

```python
# Create Geometry Nodes modifier on glass object
glass_obj = bpy.data.objects["GlassPane"]
mod = glass_obj.modifiers.new("GeoFracture", type='NODES')

# Create node tree
tree = bpy.data.node_groups.new("VoronoiFracture", 'GeometryNodeTree')
mod.node_group = tree

# Key nodes for Voronoi fracture:
# 1. Voronoi Texture (3D) — defines cell boundaries
# 2. Distribute Points on Faces — generates fracture source points
# 3. Voronoi Texture Distance field — used to separate cells
# 4. Separate Geometry — splits mesh into individual shards
```

## Manual Voronoi-Based Cutting (More Practical Approach)

Since Geometry Nodes can't directly split meshes into separate objects for rigid body,
the practical workflow uses GN to generate cut planes, then Python Boolean operations:

```python
import bmesh
from mathutils import Vector
import random

def voronoi_fracture_manual(obj, num_points=40, impact_point=None, seed=42):
    \"\"\"Fracture mesh using Voronoi-inspired cutting planes.\"\"\"
    random.seed(seed)
    bbox = [Vector(v[:]) for v in obj.bound_box]
    center = sum(bbox, Vector()) / 8
    size = max((bbox[6] - bbox[0]).length, 0.01)

    # Generate fracture points — denser near impact
    points = []
    for i in range(num_points):
        if impact_point and random.random() < 0.6:
            # 60% of points near impact
            offset = Vector([random.gauss(0, size * 0.15) for _ in range(3)])
            points.append(impact_point + offset)
        else:
            # 40% distributed across object
            offset = Vector([random.uniform(-0.5, 0.5) * size for _ in range(3)])
            points.append(center + offset)

    # For each point, create a cutting plane using bisect
    shards = [obj]
    for pt in points:
        new_shards = []
        for shard in shards:
            # Random cut plane normal
            normal = Vector([random.gauss(0, 1) for _ in range(3)]).normalized()
            # Bisect creates two halves
            result = bisect_mesh(shard, pt, normal)
            if result:
                new_shards.extend(result)
            else:
                new_shards.append(shard)
        shards = new_shards

    return shards
```

## Instancing Approach for Many Small Shards

For secondary micro-shards (glass chips, powder), use Geometry Nodes instancing
instead of individual rigid body objects:

```python
# Instance small shard meshes on points — much more efficient than
# individual rigid body objects for hundreds of tiny fragments
# Use Distribute Points on Faces + Instance on Points
# Animate with simple gravity + noise rather than full rigid body sim
```

TECHNIQUE: Combine Geometry Nodes for secondary micro-debris with rigid body for primary shards. The GN debris is visual-only (no physics) but adds realism at zero sim cost.

NOTE: Geometry Nodes fracture requires more Python code but produces the most controllable results. Good for hero shots where fracture pattern matters.
""",

    "technique_particle_debris.md": """# Particle System Debris for Glass Destruction
DocType: technique-guide
DocPath: techniques/destruction/particle_debris
DocVersion: 5.0.1
PhysicsDomain: particles
---

## Why Add Particle Debris

Real glass destruction produces:
1. **Primary shards** — large pieces (rigid body simulation)
2. **Secondary chips** — small sharp fragments (particle system)
3. **Glass dust/powder** — fine particles catching light (particle system or volumetrics)

Most VFX simulations only include #1, which looks "too clean." Adding #2 and #3
dramatically increases realism with minimal computational cost.

## Emitter Setup for Glass Chips

```python
# Create emitter object at impact point
bpy.ops.mesh.primitive_plane_add(size=0.3, location=impact_point)
emitter = bpy.context.active_object
emitter.name = "GlassDebrisEmitter"
emitter.hide_render = True  # Emitter invisible, only particles visible

# Add particle system
ps_mod = emitter.modifiers.new("DebrisParticles", type='PARTICLE_SYSTEM')
ps = ps_mod.particle_system
settings = ps.settings

settings.count = 200                  # Number of debris particles
settings.frame_start = impact_frame   # Emit at impact
settings.frame_end = impact_frame + 3 # Short burst
settings.lifetime = 40                # Particles live until settling
settings.lifetime_random = 0.3

# Physics
settings.physics_type = 'NEWTON'
settings.mass = 0.005                 # Very light glass chips
settings.normal_factor = 8.0          # Explode outward from emitter normal
settings.factor_random = 3.0          # Random velocity variation
settings.tangent_factor = 2.0         # Spread tangentially
settings.damping = 0.3                # Air resistance
settings.use_size_deflect = True      # Collide based on particle size

# Size
settings.particle_size = 0.003        # Tiny glass chips
settings.size_random = 0.5            # Size variation

# Render as small objects
settings.render_type = 'OBJECT'
settings.instance_object = chip_mesh  # Small glass shard mesh
settings.use_rotation_instance = True
settings.rotation_factor_random = 1.0 # Tumble randomly
```

## Chip Mesh for Instancing

```python
# Create a small angular glass chip shape for instancing
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.003)
chip = bpy.context.active_object
chip.name = "GlassChip"

# Deform into irregular shard shape
bm = bmesh.new()
bm.from_mesh(chip.data)
for v in bm.verts:
    v.co += Vector([random.uniform(-0.001, 0.001) for _ in range(3)])
bm.to_mesh(chip.data)
bm.free()

# Apply glass material
chip.data.materials.append(glass_material)
# Hide from viewport but keep for particle instancing
chip.hide_set(True)
```

## Floor Collision for Settling

```python
# Add collision to floor for particles to land on
floor = bpy.data.objects["Floor"]
bpy.context.view_layer.objects.active = floor
floor.select_set(True)
floor.modifiers.new("Collision", type='COLLISION')
col = floor.modifiers["Collision"]
col.settings.damping = 0.7       # High damping for glass on concrete
col.settings.friction = 0.5
```

## Glass Dust (Volumetric Approach)

For ultra-fine glass powder catching light:

```python
# Quick Smoke domain around impact area
# Very small domain, short lifetime, catches rim lighting beautifully
bpy.ops.mesh.primitive_cube_add(size=1.0, location=impact_point)
dust_domain = bpy.context.active_object
dust_domain.name = "GlassDustDomain"

mod = dust_domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
dset = mod.domain_settings
dset.domain_type = 'GAS'
dset.resolution_max = 48       # Low res — just a suggestion of dust
dset.use_dissolve_smoke = True
dset.dissolve_speed = 20       # Fast dissolve
dset.alpha = 0.02              # Very subtle
```

TECHNIQUE: Layer particle debris ON TOP of rigid body simulation. Emit from impact point at moment of impact. 200 particles with Newton physics adds negligible sim time but huge visual impact.

NOTE: Particle render_type='OBJECT' requires the instance_object to exist and have geometry. Make sure to create the chip mesh BEFORE configuring the particle system.
""",

    "technique_projectile_kinematic.md": """# Projectile Setup for Impact-Driven Shattering
DocType: technique-guide
DocPath: techniques/destruction/projectile_kinematic
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Kinematic → Active Projectile Pattern

For impact-driven destruction (brick through window, bullet hit), the projectile
needs to be KINEMATIC (animated) during approach and switch to ACTIVE (simulated)
after impact. This prevents the rigid body solver from interfering with the
planned trajectory.

## Pattern: Animated Kinematic Projectile

```python
# Create projectile (brick)
bpy.ops.mesh.primitive_cube_add(size=1, location=(-3, 0, 1.0))
brick = bpy.context.active_object
brick.name = "Brick"
brick.scale = (0.215/2, 0.1025/2, 0.065/2)  # Standard brick dimensions

# Add rigid body
bpy.context.view_layer.objects.active = brick
bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = brick.rigid_body
rb.mass = 2.5                          # Brick mass in kg
rb.collision_shape = 'BOX'             # Fast collision for simple shape
rb.friction = 0.6
rb.restitution = 0.2
rb.kinematic = True                    # Start as kinematic (animated)

# Keyframe kinematic → active transition
# Frame 1: Kinematic ON, at start position
brick.location = (-3, 0, 1.0)
brick.keyframe_insert(data_path="location", frame=1)
rb.keyframe_insert(data_path="kinematic", frame=1)

# Frame 4: Just before impact, still kinematic
brick.location = (-0.1, 0, 1.0)       # Near glass surface
brick.keyframe_insert(data_path="location", frame=4)
rb.kinematic = True
rb.keyframe_insert(data_path="kinematic", frame=4)

# Frame 5: Impact! Switch to active (physics takes over)
rb.kinematic = False
rb.keyframe_insert(data_path="kinematic", frame=5)
# Blender carries forward the velocity from animated motion
```

## Key Rigid Body Properties for Projectile

```python
rb = projectile.rigid_body
rb.mass = 2.5                  # kg — determines impact force
rb.collision_shape = 'BOX'     # or 'CONVEX_HULL' for complex shapes
rb.friction = 0.6              # Surface friction
rb.restitution = 0.2           # Bounciness (low for brick/rock)
rb.linear_damping = 0.04       # Air resistance
rb.angular_damping = 0.1       # Rotational drag
rb.collision_margin = 0.001    # Collision detection margin
```

## Velocity Control

The projectile's velocity at the kinematic→active transition determines impact force.
Higher velocity = more force on constraints = more shards break.

- **Brick at 15 m/s**: Moderate impact, clean hole in window
- **Bullet at 400 m/s**: Massive impact, window disintegrates
- **Rock at 5 m/s**: Light impact, star crack pattern, partial break

```python
# Calculate keyframe spacing for desired velocity
# velocity = distance / (frames * (1/fps))
# For 15 m/s at 24fps over 3m distance:
# frames = distance / (velocity / fps) = 3.0 / (15/24) = 4.8 frames
```

NOTE: `rigid_body.kinematic` must be keyframed on BOTH the True and False frames for the transition to work. Just keyframing False on frame 5 is not enough — Blender interpolates between keyframes.

NOTE: The keyframed kinematic→active transition preserves the object's velocity at the switch point. This is how the projectile transfers momentum to the glass.

TECHNIQUE: For the projectile to carry realistic momentum through the glass, set its mass high enough relative to the glass shards. A 2.5kg brick hitting 0.1kg shards will push through convincingly.
""",

    "technique_combined_hero_destruction.md": """# Combined Hero Destruction Setup (Maximum Realism)
DocType: technique-guide
DocPath: techniques/destruction/combined_hero
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## The Full Pipeline for Cinematic Glass Destruction

For maximum realism, combine ALL techniques:

1. **Voronoi fracture** — Pre-fracture glass with impact-centered density
2. **Constraint network** — FIXED constraints with variable breaking thresholds
3. **Kinematic projectile** — Animated brick/bullet with kinematic→active switch
4. **Particle debris** — Secondary glass chips from impact point
5. **Rim lighting** — Hard lighting to catch glass shards in the air

## Complete Scene Setup Order

```python
import bpy
import bmesh
import random
from mathutils import Vector

scene = bpy.context.scene

# ====== 1. GLASS PANE ======
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 1.5))
glass = bpy.context.active_object
glass.name = "GlassPane"
glass.scale = (1.0, 0.005, 0.75)  # 2m × 1.5m × 6mm thick
bpy.ops.object.transform_apply(scale=True)

# ====== 2. FRACTURE (Cell Fracture or manual Voronoi) ======
# Impact point on the glass surface
impact_uv = Vector((0.0, 0.0, 1.5))

# Create shards (simplified — use cell_fracture or manual bisect)
# For demonstration, assume shards[] is populated by fracture step
shards = create_voronoi_shards(glass, num_shards=60, impact_point=impact_uv)

# ====== 3. RIGID BODY WORLD ======
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20
scene.frame_start = 1
scene.frame_end = 60

# ====== 4. RIGID BODY ON EACH SHARD ======
for shard in shards:
    bpy.context.view_layer.objects.active = shard
    shard.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    rb = shard.rigid_body
    rb.mass = 0.05 + random.uniform(0, 0.15)  # Variable mass
    rb.collision_shape = 'CONVEX_HULL'
    rb.friction = 0.4
    rb.restitution = 0.3
    rb.linear_damping = 0.04
    rb.angular_damping = 0.1
    shard.select_set(False)

# ====== 5. CONSTRAINT NETWORK ======
constraint_empties = []
for i, (shard_a, shard_b) in enumerate(find_adjacent_pairs(shards)):
    midpoint = (shard_a.location + shard_b.location) / 2
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=midpoint)
    empty = bpy.context.active_object
    empty.name = f"RBC_{i:03d}"

    bpy.ops.rigidbody.constraint_add()
    rbc = empty.rigid_body_constraint
    rbc.type = 'FIXED'
    rbc.object1 = shard_a
    rbc.object2 = shard_b
    rbc.use_breaking = True

    # Variable threshold: weaker near impact
    dist = (midpoint - impact_uv).length
    rbc.breaking_threshold = 8.0 + dist * 35.0

    constraint_empties.append(empty)

# ====== 6. PROJECTILE ======
bpy.ops.mesh.primitive_cube_add(size=1, location=(-3, 0, 1.5))
brick = bpy.context.active_object
brick.name = "Brick"
brick.scale = (0.215/2, 0.1025/2, 0.065/2)
bpy.ops.object.transform_apply(scale=True)

bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = brick.rigid_body
rb.mass = 2.5
rb.collision_shape = 'BOX'
rb.kinematic = True

# Keyframe approach
brick.location = (-3, 0, 1.5)
brick.keyframe_insert(data_path="location", frame=1)
rb.keyframe_insert(data_path="kinematic", frame=1)
brick.location = (-0.05, 0, 1.5)
brick.keyframe_insert(data_path="location", frame=4)
rb.kinematic = False
rb.keyframe_insert(data_path="kinematic", frame=5)

# ====== 7. BAKE ======
bpy.ops.ptcache.bake_all(bake=True)
```

## Timing and Frame Reference

| Frame | Event |
|-------|-------|
| 1 | Scene start, glass intact, brick approaching |
| 4 | Brick near glass, still kinematic |
| 5 | Impact! Brick goes active, constraints start breaking |
| 5-15 | Main shattering — radial crack propagation |
| 15-30 | Secondary collapse, fragments bouncing on floor |
| 30-60 | Settling, glass dust dissipating |
| 12 | Recommended render frame (peak action) |

TECHNIQUE: For cinematic slow-motion, increase frame_end and substeps_per_frame proportionally. 120 frames at 10 substeps = same physics as 60 frames at 5 substeps but 2× slower.

NOTE: The `find_adjacent_pairs()` function should use a k-d tree or distance threshold to find shard pairs whose faces are within a small distance (e.g. 0.01m) of each other. This determines the constraint network topology.
""",

    "technique_glass_material_shards.md": """# Glass Material Setup for Shards
DocType: technique-guide
DocPath: techniques/destruction/glass_material
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Realistic Glass Material for Shards

Glass shards need proper transmission, IOR, and edge effects to catch light
realistically during shattering.

## Principled BSDF Glass (Blender 5.0)

```python
mat = bpy.data.materials.new("M_GlassShard")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

# Glass properties
bsdf.inputs["Base Color"].default_value = (0.95, 0.97, 0.95, 1)  # Slight green tint (float glass)
bsdf.inputs["Roughness"].default_value = 0.01        # Very smooth — glass
bsdf.inputs["Transmission Weight"].default_value = 1.0  # Fully transparent
bsdf.inputs["IOR"].default_value = 1.52               # Float glass IOR
bsdf.inputs["Specular IOR Level"].default_value = 0.5  # Default specular

nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
```

## Key Blender 5.0 Input Names

| Blender 5.0 Name | Old Name (WRONG) |
|-------------------|-------------------|
| `Transmission Weight` | `Transmission` |
| `Specular IOR Level` | `Specular` |
| `Sheen Weight` | `Sheen` |
| `Coat Weight` | `Coat` |

NOTE: Using old input names will cause `KeyError` at runtime. Always use the Blender 5.0 names above.

## Edge Highlighting for Shard Edges

Real glass shards have visible edges where light refracts differently. Add a
Fresnel or Layer Weight node to brighten edges:

```python
# Add edge brightness effect
lw = nt.nodes.new("ShaderNodeLayerWeight")
lw.inputs["Blend"].default_value = 0.3

mix = nt.nodes.new("ShaderNodeMix")
mix.data_type = 'RGBA'
mix.inputs[6].default_value = (0.95, 0.97, 0.95, 1)  # Glass color
mix.inputs[7].default_value = (1.0, 1.0, 1.0, 1)      # Bright edge

nt.links.new(lw.outputs["Facing"], mix.inputs[0])      # Factor
nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"]) # Result → Base Color
```

NOTE: In Blender 5.0, `ShaderNodeMix` replaces `ShaderNodeMixRGB`. Use `data_type='RGBA'` for color mixing.

TECHNIQUE: For cinematic glass, add a subtle Volume Absorption to the material to give thick shards a green tint at edges (like real float glass).
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "constraint_breaking_glass_shatter",
        "issue": "glass shards all fall at once instead of shattering progressively from impact point",
        "code_snippet": """# Connect adjacent shards with FIXED constraints that break on impact
# Variable threshold: weaker near impact point for radial crack propagation
impact_point = Vector((0, 0, 1.5))
for i, (shard_a, shard_b) in enumerate(adjacent_pairs):
    midpoint = (shard_a.location + shard_b.location) / 2
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=midpoint)
    empty = bpy.context.active_object
    empty.name = f"RBC_{i:03d}"
    bpy.ops.rigidbody.constraint_add()
    rbc = empty.rigid_body_constraint
    rbc.type = 'FIXED'
    rbc.object1 = shard_a
    rbc.object2 = shard_b
    rbc.use_breaking = True
    dist = (midpoint - impact_point).length
    rbc.breaking_threshold = 8.0 + dist * 35.0  # Weaker near impact""",
        "effect_type": "shatter",
        "improvement": 40.0,
        "experiment_id": "seed_destruction_v1",
        "context_before": "Prefractured rigid bodies all separate simultaneously — looks like pre-broken object collapsing",
        "context_after": "FIXED constraints with use_breaking=True and distance-based threshold create realistic radial crack propagation from impact point",
    },
    {
        "name": "kinematic_to_active_projectile",
        "issue": "projectile doesn't hit glass properly; either flies through or bounces off unrealistically",
        "code_snippet": """# Projectile: kinematic during approach, active at impact
# Blender preserves velocity at transition point
bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = projectile.rigid_body
rb.mass = 2.5
rb.collision_shape = 'BOX'
rb.kinematic = True

# Keyframe approach trajectory
projectile.location = start_pos
projectile.keyframe_insert(data_path="location", frame=1)
rb.keyframe_insert(data_path="kinematic", frame=1)

projectile.location = near_glass_pos  # Just before contact
projectile.keyframe_insert(data_path="location", frame=impact_frame - 1)

# Switch to active physics at impact
rb.kinematic = False
rb.keyframe_insert(data_path="kinematic", frame=impact_frame)""",
        "effect_type": "shatter",
        "improvement": 30.0,
        "experiment_id": "seed_destruction_v1",
        "context_before": "Rigid body projectile either affected by gravity during approach or passes through glass",
        "context_after": "Kinematic→active transition at impact frame gives controlled approach with realistic physics transfer",
    },
    {
        "name": "rigid_body_world_glass_settings",
        "issue": "rigid body simulation is unstable; shards jitter, pass through each other, or explode",
        "code_snippet": """# Rigid body world settings tuned for glass destruction
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 10    # Higher = better collision detection for fast shards
rbw.solver_iterations = 20     # Higher = more stable constraint solving
# NOTE: use substeps_per_frame NOT steps_per_second (removed in Blender 5.0)

# Per-shard settings for glass
for shard in glass_shards:
    rb = shard.rigid_body
    rb.collision_shape = 'CONVEX_HULL'  # Good balance speed vs accuracy
    rb.mass = 0.05 + random.uniform(0, 0.15)  # Variable mass
    rb.friction = 0.4
    rb.restitution = 0.3    # Some bounce (glass is somewhat elastic)
    rb.linear_damping = 0.04
    rb.angular_damping = 0.1
    rb.collision_margin = 0.001""",
        "effect_type": "shatter",
        "improvement": 25.0,
        "experiment_id": "seed_destruction_v1",
        "context_before": "Default rigid body settings cause shards to pass through each other or jitter uncontrollably",
        "context_after": "substeps_per_frame=10 and solver_iterations=20 with CONVEX_HULL collision gives stable glass shard simulation",
    },
    {
        "name": "particle_debris_glass_chips",
        "issue": "glass destruction looks too clean; missing small fragments and dust",
        "code_snippet": """# Add particle debris for secondary glass chips at impact
bpy.ops.mesh.primitive_plane_add(size=0.3, location=impact_point)
emitter = bpy.context.active_object
emitter.name = "DebrisEmitter"
emitter.hide_render = True

ps_mod = emitter.modifiers.new("Debris", type='PARTICLE_SYSTEM')
settings = ps_mod.particle_system.settings
settings.count = 200
settings.frame_start = impact_frame
settings.frame_end = impact_frame + 3
settings.lifetime = 40
settings.physics_type = 'NEWTON'
settings.mass = 0.005
settings.normal_factor = 8.0    # Explode outward
settings.factor_random = 3.0
settings.tangent_factor = 2.0
settings.damping = 0.3
settings.particle_size = 0.003
settings.size_random = 0.5
settings.render_type = 'OBJECT'
settings.instance_object = chip_mesh_obj""",
        "effect_type": "shatter",
        "improvement": 20.0,
        "experiment_id": "seed_destruction_v1",
        "context_before": "Glass break only shows large shards — missing secondary debris that makes destruction look real",
        "context_after": "200 particle chips emitted at impact frame with Newton physics adds convincing secondary debris",
    },
    {
        "name": "glass_material_transmission_b50",
        "issue": "glass shards look opaque or have wrong material; not transparent",
        "code_snippet": """# Blender 5.0 glass material — correct input names
mat = bpy.data.materials.new("M_GlassShard")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.95, 0.97, 0.95, 1)
bsdf.inputs["Roughness"].default_value = 0.01
bsdf.inputs["Transmission Weight"].default_value = 1.0  # NOT "Transmission"
bsdf.inputs["IOR"].default_value = 1.52
bsdf.inputs["Specular IOR Level"].default_value = 0.5   # NOT "Specular"
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])""",
        "effect_type": "shatter",
        "improvement": 15.0,
        "experiment_id": "seed_destruction_v1",
        "context_before": "Glass material fails with KeyError on 'Transmission' or looks opaque",
        "context_after": "Correct Blender 5.0 input names: Transmission Weight, Specular IOR Level, IOR=1.52 for float glass",
    },
    {
        "name": "cell_fracture_complete_glass_shatter",
        "issue": "glass shatter uses manual mesh cutting or basic prefracture instead of Cell Fracture addon — results look unrealistic",
        "code_snippet": """# COMPLETE Cell Fracture glass shatter workflow — RECOMMENDED approach
# Step 1: Enable Cell Fracture addon (required in headless mode)
import addon_utils
addon_utils.enable('object_cell_fracture', default_set=True, persistent=True)

# Step 2: Create glass pane geometry
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 1.5))
glass = bpy.context.active_object
glass.name = "GlassPane"
glass.scale = (1.0, 0.003, 0.75)  # Thin pane shape
bpy.ops.object.transform_apply(scale=True)

# Step 3: Run Cell Fracture to create shards
bpy.context.view_layer.objects.active = glass
glass.select_set(True)
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},
    source_limit=60,         # 40-80 shards for hero glass
    source_noise=0.05,       # Slight randomness
    cell_scale=(1, 1, 1),
    margin=0.0005,           # Tiny gap between shards
    use_smooth_faces=False,  # Sharp fracture edges for glass
    use_data_match=True,     # Copy materials from original
    use_island_split=True,
    use_interior_vgroup=True,  # Marks interior faces
)

# Step 4: Collect shard objects and add rigid body physics
shards = [o for o in bpy.data.objects if o.name.startswith("GlassPane_cell")]
for shard in shards:
    bpy.context.view_layer.objects.active = shard
    shard.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    shard.rigid_body.mass = 0.05 + random.uniform(0, 0.15)
    shard.rigid_body.collision_shape = 'CONVEX_HULL'
    shard.rigid_body.friction = 0.4
    shard.rigid_body.restitution = 0.3
    shard.rigid_body.linear_damping = 0.04
    shard.rigid_body.angular_damping = 0.1
    shard.select_set(False)

# Step 5: Add FIXED constraints between adjacent shards (for impact-driven breaking)
impact_point = Vector((0, 0, 1.5))
for i, (a, b) in enumerate(adjacent_shard_pairs):
    midpoint = (a.location + b.location) / 2
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=midpoint)
    empty = bpy.context.active_object
    empty.name = f"RBC_{i:03d}"
    bpy.ops.rigidbody.constraint_add()
    rbc = empty.rigid_body_constraint
    rbc.type = 'FIXED'
    rbc.object1 = a
    rbc.object2 = b
    rbc.use_breaking = True
    dist = (midpoint - impact_point).length
    rbc.breaking_threshold = 8.0 + dist * 35.0  # Weaker near impact""",
        "effect_type": "shatter",
        "improvement": 55.0,
        "experiment_id": "seed_destruction_v2",
        "context_before": "Manual mesh cutting produces unrealistic fracture patterns; prefractured rigid body looks fake",
        "context_after": "Cell Fracture addon creates realistic Voronoi shards, FIXED constraints with distance-based breaking threshold give impact-driven radial crack propagation",
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
    print("\nTest query: 'glass shattering constraint breaking threshold'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="glass shattering constraint breaking threshold rigid body",
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
    """Seed code pattern memory with destruction techniques."""
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
            # Mark as applicable to all destruction-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["shatter", "destruction", "rigid_body", "explosion"]
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
        description="Seed destruction technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED DESTRUCTION TECHNIQUES")
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
