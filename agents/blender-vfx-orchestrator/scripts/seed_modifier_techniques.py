#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with modifier technique knowledge.

Creates LLM-optimized markdown documents describing modifier-based modeling
approaches in Blender 5.0, then uploads them to the rewritten manual vector store.
Also seeds the code pattern memory with proven modifier snippets.

This directly addresses the geometry quality problem: without modifier docs,
the agent creates flat/crude geometry instead of smooth wine glasses, beveled
furniture, or properly subdivided organic shapes.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_modifier_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_modifier_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_modifier_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_modifier_techniques.py --docs-only
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
    "modifier_technique_overview.md": """# Modifier-Based Modeling for VFX Scenes in Blender 5.0
DocType: technique-guide
DocPath: techniques/modifiers/overview
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Overview

Modifiers are non-destructive operations applied to meshes in a stack. They are essential
for creating high-quality geometry in VFX scenes — smooth glassware, beveled furniture,
repeating architectural elements, terrain, etc. Without modifiers, geometry looks flat and CG.

## Modifier Categories for VFX

### Generate Modifiers (Add Geometry)

| Modifier | Type String | Purpose |
|----------|-------------|---------|
| Array | `'ARRAY'` | Repeat geometry linearly/radially (fence posts, brick rows, circular arrangements) |
| Bevel | `'BEVEL'` | Add chamfered edges (critical for realistic hard surfaces) |
| Boolean | `'BOOLEAN'` | Cut/combine meshes (windows in walls, complex shapes) |
| Build | `'BUILD'` | Animate geometry appearing face-by-face |
| Mirror | `'MIRROR'` | Symmetric modeling (half object → full) |
| Screw | `'SCREW'` | Lathe/revolution (wine glass, vase, bottle from profile curve) |
| Solidify | `'SOLIDIFY'` | Add thickness to surfaces (glass panes, pottery walls, metal sheets) |
| Subdivision Surface | `'SUBSURF'` | Smooth mesh with Catmull-Clark or Simple subdivision |
| Wireframe | `'WIREFRAME'` | Convert edges to geometry (cages, lattice structures) |

### Deform Modifiers (Reshape Geometry)

| Modifier | Type String | Purpose |
|----------|-------------|---------|
| Curve | `'CURVE'` | Bend mesh along a curve path (objects following paths) |
| Displace | `'DISPLACE'` | Push vertices by texture (terrain, surface detail) |
| Lattice | `'LATTICE'` | Broad deformation via lattice cage |
| Shrinkwrap | `'SHRINKWRAP'` | Project/conform mesh to another surface |
| SimpleDeform | `'SIMPLE_DEFORM'` | Twist, Bend, Taper, Stretch operations |

## Adding Modifiers via Python

```python
# Add modifier to object
mod = obj.modifiers.new(name="MySubsurf", type='SUBSURF')
mod.levels = 2          # Viewport subdivision
mod.render_levels = 3   # Render subdivision

# Apply modifier (destructive — converts to mesh)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier="MySubsurf")
# REQUIRES: object must be active, OBJECT mode
```

## Modifier Stack Order (CRITICAL)

Order matters — modifiers execute top-to-bottom. Wrong order = wrong result.

| Stack Order (Top→Bottom) | Use Case |
|--------------------------|----------|
| Mirror → Array → Bevel → Subdivision Surface | Standard hard surface |
| Screw → Solidify → Subdivision Surface | Lathe objects (wine glass, vase) |
| Subdivision Surface → Displace | Detail displacement on smooth mesh |
| Array → Curve | Objects repeated along a path |
| Mirror → Solidify → Subdivision Surface | Symmetric thin shells |

## When to Apply vs Keep Live

- **Apply** when: mesh needs physics (cloth, rigid body), exporting, or Boolean operations
- **Keep live** when: still iterating, parametric control needed, render-time evaluation OK
- **ALWAYS** `bpy.ops.object.transform_apply(scale=True)` before adding modifiers — non-unit scale causes wrong results

## Decision Guide: Which Modifiers for Which Objects

| Object | Modifiers | Why |
|--------|-----------|-----|
| Wine glass | Screw + Solidify + Subdivision Surface | Profile → revolution → wall thickness → smooth |
| Table | Bevel + Subdivision Surface | Sharp-to-rounded edges, realistic look |
| Wall with window | Boolean (DIFFERENCE) + Bevel | Cut opening, soften edges |
| Fence | Array (linear) + Bevel | Repeating posts with realistic edges |
| Vase | Screw + Subdivision Surface | Profile → revolution → smooth |
| Lamp shade | Screw + Solidify | Profile → revolution → thickness |
| Terrain | Subdivision Surface + Displace | Smooth base → texture-driven height |
| Column | Array + SimpleDeform + Bevel | Repeat fluting → twist → edge detail |
| Chain | Array + Curve | Link → repeat along path |
| Bowl | Screw + Solidify + Subdivision Surface | Same as wine glass, different profile |

NOTE: `obj.modifiers.new(name, type)` — type must be the exact string from the table above. Common mistake: using `'SUBDIVISION_SURFACE'` instead of `'SUBSURF'`.
""",

    "technique_smooth_objects.md": """# Creating Smooth/Curved Objects with Modifiers (Wine Glass, Vases, Bottles)
DocType: technique-guide
DocPath: techniques/modifiers/smooth_objects
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Screw Modifier for Lathe-Turned Objects

The Screw modifier rotates a 2D profile around an axis to create a 3D solid of revolution.
This is the CORRECT way to make wine glasses, vases, bottles, bowls, goblets, and lamp shades.

Do NOT build these objects from primitives — Screw modifier produces perfect circular
cross-sections with controllable smoothness.

### Key Properties

```python
mod = obj.modifiers.new("Screw", type='SCREW')
mod.axis = 'Z'            # Revolution axis (usually Z for upright objects)
mod.steps = 64             # Viewport segments around revolution (higher = smoother)
mod.render_steps = 64      # Render segments (match or exceed viewport steps)
mod.angle = 6.283185       # Full revolution (2*pi radians = 360 degrees)
mod.screw_offset = 0.0     # Offset per revolution (0 = closed, >0 = spiral)
mod.use_merge_vertices = True   # Merge vertices at seam
mod.merge_threshold = 0.001
```

## Complete Wine Glass with Screw + Solidify + Subdivision

```python
import bpy
import bmesh
from mathutils import Vector
import math

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ====== 1. CREATE PROFILE MESH WITH BMESH ======
mesh = bpy.data.meshes.new("WineGlassProfile")
obj = bpy.data.objects.new("WineGlass", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

bm = bmesh.new()

# Profile vertices (X = radius from center, Z = height)
# Build from bottom to top: base → stem → bowl
profile_verts = [
    (0.035, 0, 0.0),      # Base center-ish
    (0.04,  0, 0.0),      # Base outer edge
    (0.04,  0, 0.005),    # Base thickness
    (0.006, 0, 0.015),    # Base-to-stem taper
    (0.005, 0, 0.06),     # Stem bottom
    (0.005, 0, 0.10),     # Stem top
    (0.008, 0, 0.11),     # Bowl junction
    (0.035, 0, 0.14),     # Bowl widest point
    (0.038, 0, 0.17),     # Bowl upper curve
    (0.032, 0, 0.19),     # Rim taper
    (0.030, 0, 0.20),     # Rim top
]

verts = [bm.verts.new(v) for v in profile_verts]
bm.verts.ensure_lookup_table()

# Connect vertices into edge chain (profile curve)
for i in range(len(verts) - 1):
    bm.edges.new((verts[i], verts[i + 1]))

bm.to_mesh(mesh)
bm.free()

# ====== 2. SCREW MODIFIER — Profile → 3D Revolution ======
screw = obj.modifiers.new("Screw", type='SCREW')
screw.axis = 'Z'
screw.steps = 64
screw.render_steps = 64
screw.use_merge_vertices = True
screw.merge_threshold = 0.001

# ====== 3. SOLIDIFY — Add Glass Wall Thickness ======
solid = obj.modifiers.new("Solidify", type='SOLIDIFY')
solid.thickness = 0.002         # 2mm glass wall thickness
solid.offset = -1.0             # Grow inward
solid.use_even_thickness = True  # Uniform thickness on curves

# ====== 4. SUBDIVISION SURFACE — Final Smoothing ======
subsurf = obj.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 2              # Viewport
subsurf.render_levels = 3       # Render (higher = smoother)
subsurf.subdivision_type = 'CATMULL_CLARK'

# ====== 5. SMOOTH SHADING ======
bpy.ops.object.shade_smooth()
```

## Smooth Vase with Subdivision Surface

For objects where Screw isn't ideal (irregular shapes), use a base mesh + Subdivision Surface:

```python
# Create base vase shape from circle
bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.15, depth=0.4, location=(0, 0, 0.2))
vase = bpy.context.active_object
vase.name = "Vase"

# Enter edit mode to shape the profile
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(vase.data)

# Select and scale edge loops to create vase profile
# Top ring — narrow opening
for v in bm.verts:
    if abs(v.co.z - 0.4) < 0.01:
        v.co.x *= 0.6
        v.co.y *= 0.6

# Bottom ring — flat base
for v in bm.verts:
    if abs(v.co.z) < 0.01:
        v.co.x *= 0.9
        v.co.y *= 0.9

bmesh.update_edit_mesh(vase.data)
bpy.ops.object.mode_set(mode='OBJECT')

# Add Subdivision Surface for smoothing
subsurf = vase.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 3
subsurf.subdivision_type = 'CATMULL_CLARK'

bpy.ops.object.shade_smooth()
```

## Crease Edges for Sharp Features on Subdivided Meshes

Subdivision Surface rounds ALL edges by default. To keep some edges sharp (like a rim):

```python
# In edit mode, set crease on specific edges
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

crease_layer = bm.edges.layers.crease.verify()
for edge in bm.edges:
    # Crease top rim edges to keep them sharp
    if all(v.co.z > 0.19 for v in edge.verts):
        edge[crease_layer] = 1.0  # 1.0 = fully sharp, 0.0 = fully smooth

bmesh.update_edit_mesh(obj.data)
bpy.ops.object.mode_set(mode='OBJECT')
```

## Common Pitfalls

NOTE: Apply Screw BEFORE Solidify if you need to apply them. Applying Solidify first on the profile gives wrong result (thickens the 2D profile, not the 3D surface).
NOTE: `bpy.ops.object.transform_apply(scale=True)` MUST be called before Screw if object has non-uniform scale.
NOTE: Screw `steps=64` is good for close-up hero objects. For background objects, `steps=32` saves geometry.
NOTE: Subdivision `levels` (viewport) vs `render_levels` (render) — use lower viewport for performance.
NOTE: `subdivision_type='CATMULL_CLARK'` for smooth organic shapes, `'SIMPLE'` for uniform subdivision without smoothing.

TECHNIQUE: For wine glasses and similar tableware, the Screw modifier profile approach gives the BEST results. Define the silhouette as a 2D edge chain, then revolve it. This guarantees perfect circular cross-sections that no amount of manual modeling can match.
""",

    "technique_hard_surface.md": """# Hard Surface Modeling with Modifiers (Tables, Walls, Frames)
DocType: technique-guide
DocPath: techniques/modifiers/hard_surface
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Boolean Modifier — Cutting and Combining Meshes

Boolean operations (DIFFERENCE, UNION, INTERSECT) create complex shapes from simple ones.
Essential for cutting windows in walls, creating door openings, combining shapes.

### Key Properties

```python
mod = obj.modifiers.new("Boolean", type='BOOLEAN')
mod.operation = 'DIFFERENCE'    # Cut cutter from object
# Also: 'UNION' (combine), 'INTERSECT' (keep overlap only)
mod.object = cutter_object      # The object to cut with
mod.solver = 'EXACT'            # 'EXACT' = clean topology, 'FAST' = quicker but messy
```

### Window Cut in Wall

```python
import bpy

# Create wall
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1.5))
wall = bpy.context.active_object
wall.name = "Wall"
wall.scale = (3.0, 0.15, 1.5)
bpy.ops.object.transform_apply(scale=True)

# Create cutter for window opening
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.5, 0, 1.5))
cutter = bpy.context.active_object
cutter.name = "WindowCutter"
cutter.scale = (0.6, 0.3, 0.8)  # Window dimensions
bpy.ops.object.transform_apply(scale=True)

# Apply Boolean DIFFERENCE
bpy.context.view_layer.objects.active = wall
bool_mod = wall.modifiers.new("WindowCut", type='BOOLEAN')
bool_mod.operation = 'DIFFERENCE'
bool_mod.object = cutter
bool_mod.solver = 'EXACT'
bpy.ops.object.modifier_apply(modifier="WindowCut")

# Hide cutter (or delete it)
cutter.hide_set(True)
cutter.hide_render = True

# Bevel the cut edges for realism
bevel = wall.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.01
bevel.segments = 2
bevel.limit_method = 'ANGLE'
bevel.angle_limit = 0.785  # ~45 degrees — only bevel sharp edges
```

## Bevel Modifier — Realistic Edges

Objects without beveled edges look CG. Real-world objects always have slightly rounded edges.
Bevel is the #1 modifier for realism on hard surface objects.

### Key Properties

```python
mod = obj.modifiers.new("Bevel", type='BEVEL')
mod.width = 0.005              # Bevel width in Blender units (meters)
mod.segments = 3               # Subdivision segments (more = smoother curve)
mod.limit_method = 'ANGLE'     # Only bevel edges sharper than angle_limit
mod.angle_limit = 1.0472       # 60 degrees in radians
# Also: 'WEIGHT' (per-edge control), 'NONE' (all edges), 'VGROUP' (vertex group)
mod.profile = 0.5              # 0.5 = circular profile (default), <0.5 = concave, >0.5 = convex
mod.affect = 'EDGES'           # 'EDGES' or 'VERTICES'
```

## Solidify Modifier — Thin Shells

Creates thickness from a surface. Essential for glass panes, pottery, metal sheets, lampshades.

```python
mod = obj.modifiers.new("Solidify", type='SOLIDIFY')
mod.thickness = 0.003          # Wall thickness (3mm for glass)
mod.offset = -1.0              # -1 = grow inward, 0 = both sides, 1 = grow outward
mod.use_even_thickness = True  # Uniform thickness on curved surfaces
mod.use_rim = True             # Close edges (default True)
mod.use_rim_only = False       # Only create rim geometry
```

## Mirror Modifier — Symmetric Objects

Model half the object, Mirror creates the other half. Saves work and guarantees symmetry.

```python
mod = obj.modifiers.new("Mirror", type='MIRROR')
mod.use_axis[0] = True         # Mirror on X axis
mod.use_axis[1] = False        # Mirror on Y axis
mod.use_axis[2] = False        # Mirror on Z axis
mod.use_bisect_axis[0] = True  # Clip vertices at mirror plane
mod.merge_threshold = 0.001    # Merge vertices within this distance at seam
mod.use_clip = True            # Prevent vertices from crossing mirror plane
```

## Complete Table with Beveled Edges

```python
import bpy

# Table top
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.75))
top = bpy.context.active_object
top.name = "TableTop"
top.scale = (0.8, 0.5, 0.03)
bpy.ops.object.transform_apply(scale=True)

# Bevel edges for realism
bevel = top.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.005
bevel.segments = 3
bevel.limit_method = 'ANGLE'
bevel.angle_limit = 1.0472

# Subdivision for final smoothing
subsurf = top.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 1
subsurf.render_levels = 2

bpy.ops.object.shade_smooth()

# Table legs (4x)
leg_positions = [(0.65, 0.35, 0.375), (-0.65, 0.35, 0.375),
                 (0.65, -0.35, 0.375), (-0.65, -0.35, 0.375)]

for i, pos in enumerate(leg_positions):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    leg = bpy.context.active_object
    leg.name = f"TableLeg_{i}"
    leg.scale = (0.03, 0.03, 0.375)
    bpy.ops.object.transform_apply(scale=True)

    bevel = leg.modifiers.new("Bevel", type='BEVEL')
    bevel.width = 0.003
    bevel.segments = 2
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = 1.0472

    bpy.ops.object.shade_smooth()
```

## Common Pitfalls

NOTE: Boolean `solver='EXACT'` is slower but produces clean topology suitable for further operations. Use `'FAST'` only for final geometry with no further modifiers.
NOTE: Bevel `limit_method='ANGLE'` is almost always what you want — it only bevels edges sharper than the threshold, leaving flat surfaces untouched.
NOTE: `bpy.ops.object.transform_apply(scale=True)` MUST be called before Boolean, Bevel, and Solidify. Non-unit scale causes wrong widths/thicknesses.
NOTE: After Boolean DIFFERENCE, the cutter object still exists. Hide it (`hide_render=True`) or delete it.

TECHNIQUE: Every hard surface object in a VFX scene should have a Bevel modifier. Even `width=0.002` with `segments=2` makes a dramatic realism difference because it catches specular highlights along edges — something perfectly sharp CG edges cannot do.
""",

    "technique_repeating_patterns.md": """# Arrays, Instances, and Repeating Geometry
DocType: technique-guide
DocPath: techniques/modifiers/repeating_patterns
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Array Modifier — Repeating Elements

The Array modifier duplicates geometry in a pattern. Essential for fences, brick rows,
stairs, circular arrangements, chain links, and any repeating VFX element.

### Key Properties

```python
mod = obj.modifiers.new("Array", type='ARRAY')
mod.count = 10                           # Number of copies
mod.fit_type = 'FIXED_COUNT'             # 'FIXED_COUNT', 'FIT_LENGTH', 'FIT_CURVE'
# For FIT_LENGTH:
# mod.fit_length = 5.0                   # Total length in meters

# Relative offset (fraction of object bounding box)
mod.use_relative_offset = True
mod.relative_offset_displace = (1.0, 0.0, 0.0)  # X=1.0 means each copy starts where previous ends

# Constant offset (absolute distance in meters)
mod.use_constant_offset = False
# mod.constant_offset_displace = (0.0, 0.0, 0.1)  # Add fixed gap

# Object offset (use empty for rotation/scaling per copy)
mod.use_object_offset = False
# mod.offset_object = empty_object       # Each copy transformed by this empty
```

### Linear Array (Fence Posts)

```python
import bpy

# Create single fence post
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
post = bpy.context.active_object
post.name = "FencePost"
post.scale = (0.05, 0.05, 0.5)
bpy.ops.object.transform_apply(scale=True)

# Bevel for realism
bevel = post.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.003
bevel.segments = 2

# Array to repeat posts
array = post.modifiers.new("Array", type='ARRAY')
array.count = 20
array.use_relative_offset = True
array.relative_offset_displace = (3.0, 0.0, 0.0)  # 3x post width spacing
```

### Circular Array (Radial Symmetry)

A circular array uses an Empty as an offset object. Each copy is rotated by `360/count` degrees.

```python
import bpy
import math

# Create the element to repeat
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.5, 0, 0.1))
element = bpy.context.active_object
element.name = "CandleBase"
element.scale = (0.03, 0.03, 0.1)
bpy.ops.object.transform_apply(scale=True)

# Create rotation empty at the center of the circle
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rot_empty = bpy.context.active_object
rot_empty.name = "ArrayRotator"

num_copies = 12
rot_empty.rotation_euler.z = math.radians(360.0 / num_copies)  # 30 degrees

# Add Array modifier with object offset
array = element.modifiers.new("CircularArray", type='ARRAY')
array.count = num_copies
array.use_relative_offset = False    # Disable relative offset!
array.use_object_offset = True       # Use empty for transformation
array.offset_object = rot_empty
```

## Curve Modifier — Objects Along a Path

Deforms a mesh to follow a curve. Combined with Array, creates objects repeated along a path
(railings on curved stairs, cables following a route, ivy on a wall).

```python
# Create a bezier curve path
bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
path = bpy.context.active_object
path.name = "RopePath"

# Create rope segment (short cylinder)
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.01, depth=0.1, location=(0, 0, 0))
segment = bpy.context.active_object
segment.name = "RopeSegment"
bpy.ops.object.transform_apply(scale=True)

# Array to fill the curve length
array = segment.modifiers.new("Array", type='ARRAY')
array.fit_type = 'FIT_CURVE'
array.curve = path
array.use_relative_offset = True
array.relative_offset_displace = (0.0, 0.0, 1.0)  # Stack along Z (curve deform axis)

# Curve modifier to bend along path
curve_mod = segment.modifiers.new("Curve", type='CURVE')
curve_mod.object = path
curve_mod.deform_axis = 'POS_Z'  # Must match array stack direction
```

## Complete Stone Wall (Array + Randomized Displacement)

```python
import bpy
import random
from mathutils import Vector

# Create single stone block
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
stone = bpy.context.active_object
stone.name = "StoneBlock"
stone.scale = (0.3, 0.15, 0.1)
bpy.ops.object.transform_apply(scale=True)

# Subdivision for displacement detail
subsurf = stone.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 3

# Displacement texture for stone surface irregularity
tex = bpy.data.textures.new("StoneTex", type='CLOUDS')
tex.noise_scale = 0.3

displace = stone.modifiers.new("Displace", type='DISPLACE')
displace.texture = tex
displace.strength = 0.01  # Subtle surface variation

# Bevel for slightly rounded edges
bevel = stone.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.005
bevel.segments = 2

# Array for row of stones
array = stone.modifiers.new("StoneRow", type='ARRAY')
array.count = 10
array.use_relative_offset = True
array.relative_offset_displace = (1.05, 0.0, 0.0)  # Small gap between stones

bpy.ops.object.shade_smooth()
```

## Common Pitfalls

NOTE: For circular array, `use_relative_offset = False` is required — otherwise both relative and object offsets compound and the result is wrong.
NOTE: Curve modifier `deform_axis` must match the direction the array stacks copies. If Array uses `relative_offset_displace = (1, 0, 0)`, Curve axis should be `'POS_X'`.
NOTE: `fit_type='FIT_CURVE'` requires `array.curve` to be set. It calculates how many copies fit along the curve length.
NOTE: Array creates GEOMETRY copies, not instances. For 1000+ copies, use geometry nodes collection instances instead.

TECHNIQUE: For circular arrangements (candles around a centerpiece, chairs around a table, columns around a rotunda), the Array + Empty rotation pattern is the standard approach. Set Empty rotation to `360/N` degrees on the appropriate axis.
""",

    "technique_deformation.md": """# Deform Modifiers for Dynamic Shapes (Terrain, Twisted Columns)
DocType: technique-guide
DocPath: techniques/modifiers/deformation
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Displace Modifier — Texture-Driven Surface Deformation

The Displace modifier pushes vertices along their normals based on a texture.
Primary use: terrain from flat plane, surface weathering, organic irregularity.

### Key Properties

```python
mod = obj.modifiers.new("Displace", type='DISPLACE')
mod.texture = texture_object       # bpy.data.textures reference
mod.strength = 1.0                 # Displacement amount (meters)
mod.mid_level = 0.5                # Texture value that produces zero displacement
mod.direction = 'NORMAL'           # 'NORMAL', 'X', 'Y', 'Z', 'RGB_TO_XYZ'
mod.texture_coords = 'LOCAL'       # 'LOCAL', 'GLOBAL', 'OBJECT', 'UV'
```

### Complete Terrain from Subdivided Plane

```python
import bpy

# Create flat base plane
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
terrain = bpy.context.active_object
terrain.name = "Terrain"

# Subdivide for displacement detail
subsurf = terrain.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 6               # High subdivision for detailed terrain
subsurf.render_levels = 6
subsurf.subdivision_type = 'SIMPLE'  # SIMPLE = uniform grid (no smoothing)

# Create displacement texture — clouds for natural-looking terrain
tex = bpy.data.textures.new("TerrainHeight", type='CLOUDS')
tex.noise_scale = 2.0            # Scale of terrain features
tex.noise_depth = 6              # Detail levels (higher = more detail)
tex.noise_basis = 'BLENDER_ORIGINAL'

# Apply displacement
displace = terrain.modifiers.new("TerrainDisplace", type='DISPLACE')
displace.texture = tex
displace.strength = 3.0          # Peak height in meters
displace.mid_level = 0.0         # Zero displacement at texture value 0
displace.direction = 'NORMAL'
displace.texture_coords = 'LOCAL'

# Optional: second displacement layer for fine detail
tex2 = bpy.data.textures.new("TerrainDetail", type='MUSGRAVE')
tex2.noise_scale = 0.5
tex2.noise_intensity = 1.0

displace2 = terrain.modifiers.new("DetailDisplace", type='DISPLACE')
displace2.texture = tex2
displace2.strength = 0.3         # Subtle surface roughness
displace2.mid_level = 0.5

bpy.ops.object.shade_smooth()
```

## SimpleDeform Modifier — Twist, Bend, Taper, Stretch

Applies simple geometric deformations to meshes. Each mode has a single parameter (`factor`).

### Key Properties

```python
mod = obj.modifiers.new("Deform", type='SIMPLE_DEFORM')
mod.deform_method = 'TWIST'     # 'TWIST', 'BEND', 'TAPER', 'STRETCH'
mod.deform_axis = 'Z'           # Axis of deformation
mod.angle = 1.5708              # Deformation amount (radians for TWIST/BEND)
mod.factor = 0.5                # Deformation amount (for TAPER/STRETCH)
mod.lock_x = False              # Lock axes from deformation
mod.lock_y = False
mod.limits = (0.0, 1.0)         # Limit deformation to portion of mesh (0-1 range)
```

### Twisted Column

```python
import bpy

# Create tall cylinder
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.15, depth=3.0, location=(0, 0, 1.5))
column = bpy.context.active_object
column.name = "TwistedColumn"

# Add edge loops for twist resolution
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
# Subdivide along length for smooth twist (need ~20 cuts)
for _ in range(4):
    bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.object.mode_set(mode='OBJECT')

# Twist deformation
twist = column.modifiers.new("Twist", type='SIMPLE_DEFORM')
twist.deform_method = 'TWIST'
twist.deform_axis = 'Z'
twist.angle = 1.5708   # 90 degrees = quarter twist over full height

# Bevel edges
bevel = column.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.005
bevel.segments = 2
bevel.limit_method = 'ANGLE'
bevel.angle_limit = 0.7854  # ~45 degrees

# Subdivision for smooth surface
subsurf = column.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 3

bpy.ops.object.shade_smooth()
```

## Bend Deformation

Bends a flat object into a curve. Useful for creating curved walls, bent metal sheets, arched bridges.

```python
# Bend a flat plane into an arch
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
sheet = bpy.context.active_object
sheet.name = "CurvedWall"
sheet.scale = (1, 0.05, 2)
bpy.ops.object.transform_apply(scale=True)

# Subdivide for smooth bend
subsurf = sheet.modifiers.new("Subdiv", type='SUBSURF')
subsurf.levels = 4
subsurf.subdivision_type = 'SIMPLE'

# Bend 90 degrees
bend = sheet.modifiers.new("Bend", type='SIMPLE_DEFORM')
bend.deform_method = 'BEND'
bend.deform_axis = 'Z'
bend.angle = 1.5708  # 90 degrees
```

## Lattice Modifier — Broad Deformations

A lattice is an invisible cage that deforms everything inside it. Useful for bending/stretching
multiple objects at once or applying broad shape changes.

```python
# Create lattice
bpy.ops.object.add(type='LATTICE', location=(0, 0, 1))
lattice = bpy.context.active_object
lattice.name = "DeformLattice"
lattice.scale = (2, 2, 2)
lattice.data.points_u = 4  # Resolution of lattice cage
lattice.data.points_v = 4
lattice.data.points_w = 4

# Assign lattice modifier to target object
mod = target_obj.modifiers.new("Lattice", type='LATTICE')
mod.object = lattice

# Deform by moving lattice points (in edit mode)
# This is typically done interactively or via scripted point manipulation
```

## Shrinkwrap Modifier — Conform to Surface

Projects a mesh onto another surface. Useful for labels on bottles, decals on walls,
cloth draped on furniture.

```python
mod = obj.modifiers.new("Shrinkwrap", type='SHRINKWRAP')
mod.target = surface_object          # Object to project onto
mod.wrap_method = 'NEAREST_SURFACEPOINT'  # 'NEAREST_SURFACEPOINT', 'PROJECT', 'NEAREST_VERTEX', 'TARGET_PROJECT'
mod.offset = 0.001                   # Small offset to prevent z-fighting
```

## Common Pitfalls

NOTE: SimpleDeform TWIST and BEND need sufficient geometry (edge loops) along the deform axis. A cylinder with 1 segment along its height will not twist smoothly — add subdivisions first.
NOTE: Displace `subdivision_type='SIMPLE'` for terrain — `'CATMULL_CLARK'` would smooth the plane before displacement, losing resolution at edges.
NOTE: For terrain, `mid_level=0.0` makes all displacement go upward. `mid_level=0.5` gives valleys and peaks (texture values below 0.5 go down, above go up).
NOTE: Lattice resolution (`points_u/v/w`) determines deformation smoothness. Too low = blocky deformation. 4-6 is typical.

TECHNIQUE: For terrain, stack TWO Displace modifiers: one with large `noise_scale` and high `strength` for major features (hills, valleys), and one with small `noise_scale` and low `strength` for surface detail (rocks, bumps). This gives realistic multi-scale terrain.
""",

    "technique_modifier_stacking.md": """# Modifier Stack Best Practices and Order Rules
DocType: technique-guide
DocPath: techniques/modifiers/stacking
DocVersion: 5.0.1
PhysicsDomain: modeling
---

## Modifier Stack Order Rules

Modifiers execute top-to-bottom. Wrong order produces wrong results or crashes.
Follow these proven stack orders:

### Standard Hard Surface Object
```
1. Mirror        — model half, mirror to full
2. Array         — repeat if needed
3. Bevel         — round edges for realism
4. Subdivision   — final smoothing
```

### Lathe Object (Wine Glass, Vase, Bowl)
```
1. Screw         — profile → 3D revolution
2. Solidify      — add wall thickness
3. Subdivision   — final smoothing
```

### Detail Displacement (Terrain, Weathered Surfaces)
```
1. Subdivision (SIMPLE)  — add resolution for displacement
2. Displace              — push vertices by texture
3. Displace (optional)   — second layer for fine detail
```

### Objects Along Path (Chain, Railing, Cable)
```
1. Array (FIT_CURVE)  — repeat element along curve length
2. Curve              — deform to follow path
```

### Symmetric Thin Shell (Pottery, Metal Sheet)
```
1. Mirror     — model half
2. Solidify   — add thickness
3. Subdivision — smooth
```

### Deformed Repeating Element (Twisted Column Arcade)
```
1. Array           — repeat column
2. SimpleDeform    — twist/bend
3. Bevel           — edge detail
4. Subdivision     — smooth
```

## When to Apply Modifiers

### APPLY when:
- Object needs physics simulation (cloth, rigid body) — physics engines need real mesh data
- Exporting to external formats (FBX, OBJ, glTF)
- Performing Boolean operations on the result
- Other modifiers need the resolved geometry

### KEEP LIVE when:
- Still iterating on design (change parameters without re-modeling)
- Render-time evaluation is acceptable (Subdivision, Displace)
- Object is background/non-interactive

### Apply Order
Apply in stack order (top-to-bottom). Applying out of order changes the result.

```python
# Apply modifiers in correct order
obj = bpy.context.active_object
bpy.context.view_layer.objects.active = obj

# Apply each modifier from top of stack
for mod_name in ["Mirror", "Solidify", "Subdivision"]:
    if mod_name in obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=mod_name)
```

## Performance: Viewport vs Render Levels

Subdivision Surface has separate viewport and render levels. Use lower viewport
for interactive performance, higher render for final quality.

```python
subsurf = obj.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 1         # Viewport: fast, coarse
subsurf.render_levels = 3  # Render: smooth, detailed
# Level 1 = 4x faces, Level 2 = 16x, Level 3 = 64x, Level 4 = 256x
```

| Level | Face Multiplier | Use Case |
|-------|-----------------|----------|
| 1 | 4x | Background objects, viewport preview |
| 2 | 16x | Mid-ground objects, general use |
| 3 | 64x | Hero objects, close-up |
| 4 | 256x | Extreme close-up only (HEAVY) |

## Complete Modifier Stack: Detailed Column

```python
import bpy
import math

# Create base column shape
bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.2, depth=0.5, location=(0, 0, 0.25))
column_seg = bpy.context.active_object
column_seg.name = "ColumnSegment"
bpy.ops.object.transform_apply(scale=True)

# Add edge loops for deformation resolution
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=8)
bpy.ops.object.mode_set(mode='OBJECT')

# 1. ARRAY — stack segments vertically
array = column_seg.modifiers.new("Stack", type='ARRAY')
array.count = 6
array.use_relative_offset = True
array.relative_offset_displace = (0.0, 0.0, 1.0)

# 2. SIMPLE DEFORM — gentle twist
twist = column_seg.modifiers.new("Twist", type='SIMPLE_DEFORM')
twist.deform_method = 'TWIST'
twist.deform_axis = 'Z'
twist.angle = math.radians(45)  # Gentle 45-degree twist over full height

# 3. BEVEL — edge detail
bevel = column_seg.modifiers.new("EdgeDetail", type='BEVEL')
bevel.width = 0.005
bevel.segments = 2
bevel.limit_method = 'ANGLE'
bevel.angle_limit = math.radians(50)

# 4. SUBDIVISION — final smoothing
subsurf = column_seg.modifiers.new("Smooth", type='SUBSURF')
subsurf.levels = 1
subsurf.render_levels = 2

bpy.ops.object.shade_smooth()
```

## Applying Modifiers for Physics

Cloth, rigid body, and soft body simulations need real mesh geometry.
Apply modifiers before adding physics:

```python
# Apply all modifiers before physics
bpy.context.view_layer.objects.active = obj
for mod in list(obj.modifiers):
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError as e:
        print(f"Cannot apply {mod.name}: {e}")

# NOW add physics
bpy.ops.rigidbody.object_add(type='ACTIVE')
```

## Common Mistakes

NOTE: Applying modifiers in wrong order changes the result. Mirror THEN Subdivision is different from Subdivision THEN Mirror.
NOTE: `bpy.ops.object.transform_apply(scale=True)` BEFORE adding modifiers. Non-unit scale makes Bevel widths wrong, Solidify thickness wrong, Array offsets wrong.
NOTE: `bpy.ops.object.modifier_apply(modifier=name)` requires the object to be active (`view_layer.objects.active = obj`) and in OBJECT mode. Fails silently or throws RuntimeError otherwise.
NOTE: Subdivision level 4+ on high-poly meshes can cause memory issues. Use level 3 max for hero objects, level 2 for everything else.
NOTE: Boolean operations on meshes with modifiers may produce unexpected results. Apply existing modifiers first, then Boolean.

TECHNIQUE: When building complex objects, add modifiers incrementally and check the result after each one. The viewport shows the modifier stack result in real-time — verify each modifier is producing the expected effect before adding the next one.
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "wine_glass_screw_solidify_subdivision",
        "issue": "wine glass or vase geometry looks faceted/crude instead of smooth and realistic",
        "code_snippet": """# Wine glass using Screw + Solidify + Subdivision Surface
import bpy, bmesh

mesh = bpy.data.meshes.new("WineGlassProfile")
obj = bpy.data.objects.new("WineGlass", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

bm = bmesh.new()
# Profile: (radius, 0, height) — base to rim
profile = [
    (0.035, 0, 0.0), (0.04, 0, 0.0), (0.04, 0, 0.005),
    (0.006, 0, 0.015), (0.005, 0, 0.06), (0.005, 0, 0.10),
    (0.008, 0, 0.11), (0.035, 0, 0.14), (0.038, 0, 0.17),
    (0.032, 0, 0.19), (0.030, 0, 0.20),
]
verts = [bm.verts.new(v) for v in profile]
bm.verts.ensure_lookup_table()
for i in range(len(verts) - 1):
    bm.edges.new((verts[i], verts[i + 1]))
bm.to_mesh(mesh)
bm.free()

# Screw — revolve profile around Z axis
screw = obj.modifiers.new("Screw", type='SCREW')
screw.axis = 'Z'
screw.steps = 64
screw.render_steps = 64
screw.use_merge_vertices = True
screw.merge_threshold = 0.001

# Solidify — glass wall thickness
solid = obj.modifiers.new("Solidify", type='SOLIDIFY')
solid.thickness = 0.002
solid.offset = -1.0
solid.use_even_thickness = True

# Subdivision — final smoothing
subsurf = obj.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 3
subsurf.subdivision_type = 'CATMULL_CLARK'

bpy.ops.object.shade_smooth()""",
        "effect_type": "modeling",
        "improvement": 50.0,
        "experiment_id": "seed_modifiers_v1",
        "context_before": "Wine glass built from cylinders/cones looks faceted, crude, and obviously CG",
        "context_after": "Screw modifier revolves a 2D profile into perfect circular cross-sections; Solidify adds uniform glass wall thickness; Subdivision smooths everything",
    },
    {
        "name": "hard_surface_bevel_subdivision",
        "issue": "hard surface objects (tables, shelves, frames) look flat and CG — missing edge highlights",
        "code_snippet": """# Hard surface with Bevel + Subdivision for realistic edges
import bpy

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.75))
obj = bpy.context.active_object
obj.scale = (0.8, 0.5, 0.03)
bpy.ops.object.transform_apply(scale=True)  # MUST apply scale before modifiers

# Bevel — round edges to catch specular highlights
bevel = obj.modifiers.new("Bevel", type='BEVEL')
bevel.width = 0.005            # 5mm bevel
bevel.segments = 3             # Smooth curve
bevel.limit_method = 'ANGLE'   # Only bevel sharp edges
bevel.angle_limit = 1.0472     # 60 degrees

# Subdivision — final smoothing
subsurf = obj.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 1
subsurf.render_levels = 2

bpy.ops.object.shade_smooth()""",
        "effect_type": "modeling",
        "improvement": 35.0,
        "experiment_id": "seed_modifiers_v1",
        "context_before": "Cube-based objects have perfectly sharp edges that never exist in reality — no specular highlights along edges",
        "context_after": "Bevel modifier with angle limit rounds only sharp edges; even 5mm bevel dramatically improves realism by catching light at edges",
    },
    {
        "name": "circular_array_empty_offset",
        "issue": "circular arrangement of objects (candles, columns, chairs) positioned manually with trigonometry errors",
        "code_snippet": """# Circular array using Empty rotation offset
import bpy, math

# Create element at radius distance from center
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.5, 0, 0.1))
element = bpy.context.active_object
element.scale = (0.03, 0.03, 0.1)
bpy.ops.object.transform_apply(scale=True)

# Create rotation empty at center
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rot_empty = bpy.context.active_object
rot_empty.name = "ArrayRotator"

num_copies = 12
rot_empty.rotation_euler.z = math.radians(360.0 / num_copies)

# Array with object offset (NOT relative offset)
bpy.context.view_layer.objects.active = element
array = element.modifiers.new("CircularArray", type='ARRAY')
array.count = num_copies
array.use_relative_offset = False   # MUST disable relative offset
array.use_object_offset = True
array.offset_object = rot_empty""",
        "effect_type": "modeling",
        "improvement": 30.0,
        "experiment_id": "seed_modifiers_v1",
        "context_before": "Manually positioning objects in a circle using sin/cos leads to off-by-one errors and non-uniform spacing",
        "context_after": "Array modifier with Empty rotation offset guarantees perfect circular spacing; changing count auto-adjusts angle",
    },
    {
        "name": "terrain_displace_subdivision",
        "issue": "terrain is flat plane or uses manually placed vertices — no natural variation",
        "code_snippet": """# Terrain from subdivided plane + cloud texture displacement
import bpy

bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
terrain = bpy.context.active_object
terrain.name = "Terrain"

# High subdivision for displacement detail (SIMPLE = no smoothing)
subsurf = terrain.modifiers.new("Subdivision", type='SUBSURF')
subsurf.levels = 6
subsurf.render_levels = 6
subsurf.subdivision_type = 'SIMPLE'

# Large-scale terrain features
tex_large = bpy.data.textures.new("TerrainLarge", type='CLOUDS')
tex_large.noise_scale = 2.0
tex_large.noise_depth = 6

disp_large = terrain.modifiers.new("LargeFeatures", type='DISPLACE')
disp_large.texture = tex_large
disp_large.strength = 3.0
disp_large.mid_level = 0.0

# Fine surface detail
tex_fine = bpy.data.textures.new("TerrainFine", type='MUSGRAVE')
tex_fine.noise_scale = 0.5

disp_fine = terrain.modifiers.new("FineDetail", type='DISPLACE')
disp_fine.texture = tex_fine
disp_fine.strength = 0.3
disp_fine.mid_level = 0.5

bpy.ops.object.shade_smooth()""",
        "effect_type": "modeling",
        "improvement": 40.0,
        "experiment_id": "seed_modifiers_v1",
        "context_before": "Flat plane with no variation — terrain looks artificial and boring",
        "context_after": "Two-layer displacement (large clouds + fine musgrave) on subdivided plane creates realistic multi-scale terrain with hills, valleys, and surface roughness",
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
    print("\nTest query: 'wine glass screw modifier solidify subdivision'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="wine glass screw modifier solidify subdivision surface",
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
    """Seed code pattern memory with modifier techniques."""
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
            # Mark as applicable to all modeling-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["modeling", "scene_setup", "geometry", "hard_surface"]
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
        description="Seed modifier technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED MODIFIER TECHNIQUES")
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
