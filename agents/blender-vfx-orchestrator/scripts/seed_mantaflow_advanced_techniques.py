#!/usr/bin/env python3
"""
Seed vector store + code pattern memory with advanced Mantaflow technique knowledge.

Creates LLM-optimized markdown documents describing advanced Mantaflow features
in Blender 5.0 (liquid simulation, multi-flow effects, dissolve, export),
then uploads them to the rewritten manual vector store.
Also seeds the code pattern memory with proven Mantaflow snippets.

Basic fire/smoke setup is already well-covered in the vector store. This fills
the gaps on advanced Mantaflow features: liquid sims, multi-flow domains,
dissolve/dissipation effects, and OpenVDB export for the DXR renderer.

Usage:
    # Upload technique docs to vector store + seed code patterns
    python scripts/seed_mantaflow_advanced_techniques.py

    # Dry run — show what would be uploaded
    python scripts/seed_mantaflow_advanced_techniques.py --dry-run

    # Only seed code patterns (no vector store upload)
    python scripts/seed_mantaflow_advanced_techniques.py --patterns-only

    # Only upload technique docs (no code patterns)
    python scripts/seed_mantaflow_advanced_techniques.py --docs-only
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
    "mantaflow_advanced_overview.md": """# Advanced Mantaflow Techniques in Blender 5.0
DocType: technique-guide
DocPath: techniques/mantaflow/advanced_overview
DocVersion: 5.0.1
PhysicsDomain: mantaflow
---

## Beyond Basic Fire and Smoke

Basic single-emitter fire/smoke is well documented. This guide covers advanced
Mantaflow features that produce diverse, high-quality VFX.

## Technique Comparison Table

| Technique | Domain Type | Complexity | Best For |
|-----------|-------------|------------|----------|
| Multi-Flow Domain | GAS | MEDIUM | Campfire (separate flame/smoke), colored smoke |
| Liquid Simulation | LIQUID | MEDIUM | Water pour, splash, puddle, waterfall |
| Liquid + Obstacles | LIQUID | HIGH | Waterfall over rocks, water around objects |
| Colored Smoke | GAS | MEDIUM | Colored plumes, multi-source smoke art |
| Adaptive Domain | GAS/LIQUID | LOW | Performance optimization for sparse effects |
| Noise Enhancement | GAS | LOW | Adding fine detail to coarse simulations |
| Dissolve Effects | GAS | LOW | Ethereal fog, fading smoke trails |

## Decision Guide

- **Water / liquid effects** → `domain_type='LIQUID'`, FLIP solver, `use_mesh=True` for surface
- **Multiple fire/smoke sources** → Single GAS domain, multiple flow objects, each with own color/temp
- **Smoke that fades** → GAS domain, `use_dissolve_smoke=True`, tune `dissolve_speed`
- **Performance on large scenes** → Enable `use_adaptive_domain=True` to shrink domain to active area
- **More detail without higher resolution** → `use_noise=True` with `noise_scale=2` on GAS domains
- **Liquid spray/foam/bubbles** → LIQUID domain with `use_spray_particles=True`, `use_foam_particles=True`

## API Gotchas (Blender 5.0)

| CORRECT | WRONG (will error) |
|---------|--------------------|
| `resolution_max` | `resolution_divisions` |
| `use_adaptive_timesteps` | `use_adaptive_time_steps` |
| `use_dissolve_smoke` | `use_dissolve` |
| `timesteps_max` | `timesteps_per_frame`, `timesteps_maximum` |
| `cache_type = 'ALL'` | `cache_type = 'REPLAY'` (incomplete data) |
| `cache_particle_format = 'UNI'` | `cache_particle_format = 'OPENVDB'` (GAS only) |
| `cache_mesh_format = 'BOBJECT'` | `cache_mesh_format = 'UNI'` (GAS only) |
| `bpy.ops.object.effector_add()` | `bpy.ops.object.forcefield_add()` (removed) |
| `openvdb_data_depth = '32'` | `openvdb_data_depth = 'NONE'` (runtime error) |

## Common Domain Setup Pattern

```python
import bpy

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 1.0))
domain_obj = bpy.context.active_object
domain_obj.name = "FluidDomain"
mod = domain_obj.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'          # or 'LIQUID'
ds.resolution_max = 64          # 32=fast/low, 64=balanced, 128=high/slow
ds.use_adaptive_domain = True   # Shrink to active region
ds.cache_type = 'ALL'           # REQUIRED for full bake (not 'REPLAY')

# GAS-specific cache format
ds.cache_particle_format = 'UNI'       # NOT 'OPENVDB' for GAS
ds.cache_mesh_format = 'BOBJECT'       # NOT 'UNI' for GAS
```

TECHNIQUE: Always set `cache_type='ALL'` before baking. `REPLAY` mode does not produce
complete bake data and will cause missing frames in renders.
""",

    "technique_liquid_simulation.md": """# Liquid Simulation (Water/Fluid Effects)
DocType: technique-guide
DocPath: techniques/mantaflow/liquid_simulation
DocVersion: 5.0.1
PhysicsDomain: mantaflow
---

## Liquid Simulation Setup

Mantaflow LIQUID domain simulates incompressible fluid (water, oil, lava) using
FLIP or APIC particle-based methods. Produces mesh surface + optional spray/foam/bubbles.

## Domain Setup for Liquid

```python
import bpy

# Domain — must be large enough to contain all liquid
bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0, 0, 2.0))
domain = bpy.context.active_object
domain.name = "LiquidDomain"
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = 64              # 64 minimum for decent liquid detail
ds.use_plane_init = True            # CRITICAL: required or bakes are empty
ds.use_adaptive_timesteps = True    # Stability for fast-moving fluid
ds.timesteps_max = 4                # Max substeps per frame
ds.cache_type = 'ALL'               # Full bake required

# Mesh surface generation
ds.use_mesh = True                  # Generate surface mesh from particles
ds.mesh_scale = 1                   # Mesh resolution relative to domain
ds.mesh_concave_upper = 3.5         # Concavity threshold (higher = smoother)
ds.mesh_concave_lower = 0.5

# Spray / foam / bubble particles (optional)
ds.use_spray_particles = True
ds.use_foam_particles = True
ds.use_bubble_particles = True
```

## Inflow (Liquid Source)

```python
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 3.0))
inflow = bpy.context.active_object
inflow.name = "WaterInflow"
mod = inflow.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'LIQUID'
fs.flow_behavior = 'INFLOW'        # Continuous source
fs.use_initial_velocity = True
fs.velocity_normal = -2.0          # Downward flow speed
```

## FLIP vs APIC Solver

| Property | FLIP | APIC |
|----------|------|------|
| Speed | Faster | Slower (~30% more) |
| Splashiness | More splashy, energetic | Smoother, calmer |
| Volume loss | More volume loss at low res | Better volume preservation |
| Best for | Waterfalls, impacts, action | Pools, calm pours, viscous |

```python
ds.simulation_method = 'FLIP'      # or 'APIC'
```

## Complete: Water Pouring Into Glass

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# --- Glass container (passive obstacle) ---
bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=0.8, location=(0, 0, 0.4))
glass = bpy.context.active_object
glass.name = "Glass"
# Hollow out: solidify modifier to make it a shell
solid = glass.modifiers.new("Solidify", type='SOLIDIFY')
solid.thickness = 0.02
bpy.ops.object.modifier_apply(modifier="Solidify")

mod = glass.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'EFFECTOR'
es = mod.effector_settings
es.effector_type = 'COLLISION'
es.surface_distance = 0.005

# --- Domain (encompasses glass + stream path) ---
bpy.ops.mesh.primitive_cube_add(size=3.0, location=(0, 0, 1.5))
domain = bpy.context.active_object
domain.name = "LiquidDomain"
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = 96
ds.use_plane_init = True            # CRITICAL for liquid domains
ds.use_mesh = True
ds.use_adaptive_timesteps = True
ds.timesteps_max = 4
ds.cache_type = 'ALL'
ds.simulation_method = 'FLIP'

# --- Water inflow (small stream above glass) ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(0, 0, 2.5))
inflow = bpy.context.active_object
inflow.name = "WaterStream"
mod = inflow.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'LIQUID'
fs.flow_behavior = 'INFLOW'
fs.use_initial_velocity = True
fs.velocity_normal = -3.0          # Downward pour

# --- Water material ---
mat = bpy.data.materials.new("M_Water")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.8, 0.9, 1.0, 1)
bsdf.inputs["Roughness"].default_value = 0.02
bsdf.inputs["Transmission Weight"].default_value = 0.95
bsdf.inputs["IOR"].default_value = 1.333          # Water IOR
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
domain.data.materials.append(mat)

# --- Bake ---
bpy.context.view_layer.objects.active = domain
bpy.ops.fluid.bake_all()
```

## Resolution vs Quality

| resolution_max | Detail Level | Sim Time (120 frames) | Use Case |
|----------------|-------------|----------------------|----------|
| 32 | Low — blocky liquid | ~30 seconds | Previewing |
| 64 | Medium — decent surface | ~2 minutes | Background liquid |
| 96 | Good — visible detail | ~8 minutes | Hero shot |
| 128 | High — fine splashes | ~20 minutes | Close-up hero |

CRITICAL: `use_plane_init = True` is REQUIRED for liquid domains. Without it, the initial
fluid plane is not created and bakes produce zero liquid — the domain will be completely empty.

NOTE: Liquid domains must have `cache_type='ALL'` and be baked with `bpy.ops.fluid.bake_all()`.
The domain object must be active and in OBJECT mode when calling bake.
""",

    "technique_multi_flow_effects.md": """# Multi-Flow Effects (Multiple Sources in One Domain)
DocType: technique-guide
DocPath: techniques/mantaflow/multi_flow_effects
DocVersion: 5.0.1
PhysicsDomain: mantaflow
---

## Multiple Flow Sources

A single Mantaflow GAS domain can contain multiple flow objects, each emitting different
smoke/fire with different properties. This enables campfires (separate flame/smoke),
colored smoke effects, and complex multi-source pyro.

## Key Concept: Flow Objects Are Independent

Each flow object inside a domain has its own:
- `flow_type` (SMOKE, FIRE, BOTH)
- `flow_behavior` (INFLOW, OUTFLOW, GEOMETRY)
- Temperature and density
- Color (for colored smoke)
- Velocity

The domain simulates all sources together in one unified fluid grid.

## Colored Smoke Setup

```python
import bpy

# --- Domain ---
bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0, 0, 2.0))
domain = bpy.context.active_object
domain.name = "SmokeDomain"
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'
ds.resolution_max = 64
ds.use_color_ramp = True              # Enable colored smoke visualization
ds.cache_type = 'ALL'
ds.cache_particle_format = 'UNI'     # UNI for GAS domains
ds.cache_mesh_format = 'BOBJECT'     # BOBJECT for GAS domains

# --- Red smoke source ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(-1.0, 0, 0.2))
red_flow = bpy.context.active_object
red_flow.name = "RedSmoke"
mod = red_flow.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'SMOKE'
fs.flow_behavior = 'INFLOW'
fs.smoke_color = (1.0, 0.1, 0.05)    # Red
fs.density = 1.0
fs.temperature = 0.5

# --- Blue smoke source ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(1.0, 0, 0.2))
blue_flow = bpy.context.active_object
blue_flow.name = "BlueSmoke"
mod = blue_flow.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'SMOKE'
fs.flow_behavior = 'INFLOW'
fs.smoke_color = (0.05, 0.2, 1.0)    # Blue
fs.density = 1.0
fs.temperature = 0.3
```

## Effector Objects (Obstacles and Guides)

```python
# Obstacle — fluid flows around it
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 1.5))
obstacle = bpy.context.active_object
obstacle.name = "Obstacle"
mod = obstacle.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'EFFECTOR'
es = mod.effector_settings
es.effector_type = 'COLLISION'        # Solid obstacle
es.surface_distance = 0.01           # Collision offset

# Guide — directs fluid flow along a path
bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=2.0, location=(0, 0, 1.0))
guide = bpy.context.active_object
guide.name = "FlowGuide"
mod = guide.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'EFFECTOR'
es = mod.effector_settings
es.effector_type = 'GUIDE'
es.guide_mode = 'OVERRIDE'           # OVERRIDE, AVERAGED, MIN, MAX
es.velocity_factor = 2.0             # Guide strength
```

## Force Fields Affecting Fluid

Force fields (wind, turbulence, vortex) placed in the scene affect Mantaflow simulation.

```python
# Wind force field — pushes flames sideways
bpy.ops.object.effector_add(type='WIND', location=(2, 0, 1))
wind = bpy.context.active_object
wind.name = "WindForce"
wind.field.strength = 3.0
wind.field.noise = 0.5                # Turbulent wind
wind.rotation_euler = (0, 1.57, 0)    # Point along -X axis

# Turbulence — adds chaotic swirling
bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 1.5))
turb = bpy.context.active_object
turb.name = "Turbulence"
turb.field.strength = 2.0
turb.field.size = 1.0                 # Scale of turbulence pattern
turb.field.noise = 1.0
```

NOTE: Use `bpy.ops.object.effector_add()` NOT `bpy.ops.object.forcefield_add()` (removed in Blender 5.0).

## Complete: Campfire With Separate Flame, Smoke, Embers

```python
import bpy, random

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 100

# --- Domain ---
bpy.ops.mesh.primitive_cube_add(size=3.0, location=(0, 0, 1.5))
domain = bpy.context.active_object
domain.name = "CampfireDomain"
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'
ds.resolution_max = 80
ds.use_noise = True                   # Wavelet noise for detail
ds.noise_scale = 2
ds.vorticity = 0.3                    # Swirl in flames
ds.use_dissolve_smoke = True
ds.dissolve_speed = 40                # Smoke fades over 40 frames
ds.alpha = 1.0                        # Buoyancy (higher = more rise)
ds.beta = 0.5                         # Heat response
ds.cache_type = 'ALL'
ds.cache_particle_format = 'UNI'
ds.cache_mesh_format = 'BOBJECT'

# --- Flame source (bottom, high temp) ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(0, 0, 0.3))
flame = bpy.context.active_object
flame.name = "FlameSource"
mod = flame.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'BOTH'                 # Fire + smoke
fs.flow_behavior = 'INFLOW'
fs.fuel_amount = 1.5                  # High fuel for bright flames
fs.temperature = 2.0                  # High temperature
fs.density = 0.8

# --- Smoke plume (above flames, cooler) ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, 0.8))
smoke = bpy.context.active_object
smoke.name = "SmokePlume"
mod = smoke.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'SMOKE'
fs.flow_behavior = 'INFLOW'
fs.density = 0.5
fs.temperature = 0.3
fs.smoke_color = (0.15, 0.12, 0.1)   # Dark grey-brown smoke

# --- Embers (particle system on separate object, not fluid) ---
bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, 0, 0.5))
ember_emitter = bpy.context.active_object
ember_emitter.name = "EmberEmitter"
ember_emitter.hide_render = True
ps_mod = ember_emitter.modifiers.new("Embers", type='PARTICLE_SYSTEM')
ps = ps_mod.particle_system.settings
ps.count = 60
ps.frame_start = 1
ps.frame_end = 80
ps.lifetime = 30
ps.lifetime_random = 0.5
ps.physics_type = 'NEWTON'
ps.mass = 0.001
ps.normal_factor = 3.0
ps.factor_random = 2.0
ps.damping = 0.05
ps.particle_size = 0.005

# --- Light wind ---
bpy.ops.object.effector_add(type='WIND', location=(2, 0, 1))
wind = bpy.context.active_object
wind.field.strength = 1.5
wind.field.noise = 0.3
wind.rotation_euler = (0, 1.57, 0)

# --- Bake ---
bpy.context.view_layer.objects.active = domain
bpy.ops.fluid.bake_all()
```

TECHNIQUE: Separate flame and smoke flows give better artistic control. Flame source
emits BOTH with high fuel/temp; smoke source emits SMOKE only with lower density.
Wind force field adds natural flickering direction.
""",

    "technique_mantaflow_dissolve_effects.md": """# Dissolve, Dissipation, and Temporal Effects
DocType: technique-guide
DocPath: techniques/mantaflow/dissolve_effects
DocVersion: 5.0.1
PhysicsDomain: mantaflow
---

## Smoke Dissolve (Fadeout)

`use_dissolve_smoke` makes smoke density decrease over time, causing it to fade.
Essential for effects where smoke should not accumulate indefinitely.

```python
ds = domain.modifiers["Fluid"].domain_settings
ds.use_dissolve_smoke = True
ds.dissolve_speed = 25               # Frames for smoke to fully dissolve
ds.use_dissolve_smoke_log = False    # Linear dissolve (True = logarithmic/slow start)
```

## Dissolve Speed Guide

| dissolve_speed | Effect | Use Case |
|----------------|--------|----------|
| 5-10 | Very fast fade | Breath in cold air, quick puffs |
| 15-30 | Medium fade | Campfire smoke, chimney plume |
| 40-60 | Slow fade | Fog, atmospheric haze |
| 80-120 | Very slow | Persistent cloud, heavy smoke |
| 0 (disabled) | No dissolve | Accumulating smoke, filling a room |

## Buoyancy and Heat Settings

These control how smoke/fire rises and responds to temperature differences.

```python
ds.alpha = 1.0                        # Buoyancy density (higher = more rise)
ds.beta = 0.5                         # Buoyancy heat (temp difference drives rise)
ds.vorticity = 0.2                    # Swirl/curl in the flow (0=laminar, 1=turbulent)
```

- **`alpha`**: Density-based buoyancy. Higher values make denser smoke rise faster.
  Range: 0.0 (no buoyancy) to 5.0+ (extreme rise). Default ~1.0.
- **`beta`**: Heat-based buoyancy. Hotter regions rise relative to cooler regions.
  Range: 0.0 to 5.0+. Default ~0.5. Fire typically needs beta >= 0.5.
- **`vorticity`**: Curling motion in the fluid. Higher = more turbulent, swirling detail.
  Range: 0.0 (smooth) to 1.0 (very turbulent). Default ~0.1.

## Adaptive Timesteps for Fast-Moving Fluid

When fluid moves fast (explosions, high-velocity jets), fixed timesteps can cause
instability or missed collisions. Adaptive timesteps subdivide frames as needed.

```python
ds.use_adaptive_timesteps = True      # NOT use_adaptive_time_steps
ds.timesteps_max = 4                  # Max substeps per frame (higher = more stable)
ds.timesteps_min = 1                  # Min substeps per frame
ds.cfl_condition = 4.0                # CFL number (lower = more conservative)
```

NOTE: The attribute is `use_adaptive_timesteps` (no underscore between "time" and "steps").
`use_adaptive_time_steps` will cause AttributeError.

## Noise / Wavelet Enhancement

Adds high-frequency detail to a coarse simulation without increasing domain resolution.

```python
ds.use_noise = True                   # Enable noise enhancement
ds.noise_scale = 2                    # Noise upscale factor (2 = 2x detail)
ds.noise_strength = 1.0              # How much noise to add (0=none, 1=full)
```

## Complete: Ethereal Fog That Dissipates

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 150

# --- Low flat domain for ground fog ---
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.5))
domain = bpy.context.active_object
domain.name = "FogDomain"
domain.scale = (4.0, 4.0, 1.5)
bpy.ops.object.transform_apply(scale=True)

mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'
ds.resolution_max = 64
ds.use_dissolve_smoke = True
ds.dissolve_speed = 60               # Slow dissolve for lingering fog
ds.use_dissolve_smoke_log = True     # Logarithmic = slow start, fast end
ds.alpha = 0.1                       # Low buoyancy — fog stays low
ds.beta = 0.0                        # No heat response
ds.vorticity = 0.05                  # Very smooth, laminar flow
ds.cache_type = 'ALL'
ds.cache_particle_format = 'UNI'
ds.cache_mesh_format = 'BOBJECT'

# --- Fog emitter (flat plane on ground) ---
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 0.05))
fog_flow = bpy.context.active_object
fog_flow.name = "FogEmitter"
mod = fog_flow.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'SMOKE'
fs.flow_behavior = 'INFLOW'
fs.density = 0.4                     # Thin fog
fs.temperature = 0.0                 # No heat — stays on ground
fs.smoke_color = (0.8, 0.82, 0.85)  # Light grey-blue

# --- Gentle wind to move fog ---
bpy.ops.object.effector_add(type='WIND', location=(3, 0, 0.5))
wind = bpy.context.active_object
wind.field.strength = 0.8
wind.field.noise = 0.2
wind.rotation_euler = (0, 1.57, 0)

# --- Bake ---
bpy.context.view_layer.objects.active = domain
bpy.ops.fluid.bake_all()
```

## Complete: Fast Explosion With Dissolving Smoke Trail

```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 80

# --- Domain ---
bpy.ops.mesh.primitive_cube_add(size=6.0, location=(0, 0, 3.0))
domain = bpy.context.active_object
domain.name = "ExplosionDomain"
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'
ds.resolution_max = 96
ds.use_adaptive_timesteps = True
ds.timesteps_max = 6                  # High for fast-moving explosion
ds.use_dissolve_smoke = True
ds.dissolve_speed = 30               # Smoke fades after initial blast
ds.alpha = 2.0                       # Strong buoyancy — mushroom cloud rise
ds.beta = 1.5                        # Strong heat response
ds.vorticity = 0.6                   # Turbulent swirl
ds.use_noise = True
ds.noise_scale = 2
ds.burning_rate = 1.2                # Fast fuel consumption
ds.cache_type = 'ALL'
ds.cache_particle_format = 'UNI'
ds.cache_mesh_format = 'BOBJECT'

# --- Explosion source (burst) ---
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(0, 0, 0.5))
blast = bpy.context.active_object
blast.name = "ExplosionSource"
mod = blast.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'FLOW'
fs = mod.flow_settings
fs.flow_type = 'BOTH'
fs.flow_behavior = 'INFLOW'
fs.fuel_amount = 3.0                  # Very high fuel for bright fireball
fs.temperature = 5.0                  # Extreme heat
fs.density = 2.0                      # Dense initial blast

# Keyframe: emit only frames 1-5 (short burst)
fs.keyframe_insert(data_path="density", frame=1)
fs.density = 0.0
fs.keyframe_insert(data_path="density", frame=6)

# --- Bake ---
bpy.context.view_layer.objects.active = domain
bpy.ops.fluid.bake_all()
```

TECHNIQUE: For explosions, use short-burst inflow (keyframe density to 0 after a few frames)
combined with high alpha/beta for mushroom cloud rise. `dissolve_speed` of 25-40 gives the
smoke trail that fades after the fireball, matching real explosion behavior.

NOTE: `burning_rate` controls fuel consumption speed. NOT `reaction_speed` (wrong attribute name).
""",

    "technique_mantaflow_export.md": """# Mantaflow Export and Volume Rendering
DocType: technique-guide
DocPath: techniques/mantaflow/export_rendering
DocVersion: 5.0.1
PhysicsDomain: mantaflow
---

## OpenVDB Export Settings

OpenVDB is the standard format for volumetric data exchange. Mantaflow bakes can be
exported as OpenVDB for use in external renderers (including NanoVDB for DXR).

```python
ds = domain.modifiers["Fluid"].domain_settings

# Cache format for OpenVDB export
ds.cache_data_format = 'OPENVDB'     # Store sim data as OpenVDB
ds.openvdb_data_depth = '32'         # '32' or '16' — NOT 'NONE' (runtime error)
ds.openvdb_cache_compress_type = 'ZIP'  # ZIP or NONE — BLOSC removed in 5.0

# For GAS domains specifically:
ds.cache_particle_format = 'UNI'     # NOT 'OPENVDB' for GAS
ds.cache_mesh_format = 'BOBJECT'     # NOT 'UNI' for GAS
```

## OpenVDB Data Depth

| Value | Precision | File Size | Use Case |
|-------|-----------|-----------|----------|
| `'32'` | Full float | Large | Hero effects, close-up fire |
| `'16'` | Half float | ~50% smaller | Background effects, distant smoke |

CRITICAL: `openvdb_data_depth='NONE'` will raise a runtime error. Always use '32' or '16'.
CRITICAL: BLOSC compression was removed in Blender 5.0. Use 'ZIP' or 'NONE' only.

## Cache Location

```python
import os

# Set explicit cache directory (reliable in headless mode)
cache_dir = os.environ.get('BLENDER_CACHE_DIR', '/tmp/blender_cache')
ds.cache_directory = cache_dir

# After bake, VDB files will be at:
# {cache_dir}/data_{frame:04d}.vdb
```

## Volume Shader Setup (Principled Volume)

For rendering Mantaflow gas simulations in Blender, assign a volume shader to the domain.

```python
import bpy

mat = bpy.data.materials.new("M_FireSmoke")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")

# Principled Volume — all-in-one for fire/smoke
vol = nt.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Density"].default_value = 5.0           # Smoke opacity
vol.inputs["Anisotropy"].default_value = 0.3         # Forward scattering
vol.inputs["Absorption Color"].default_value = (0.4, 0.3, 0.25, 1)  # Warm smoke
vol.inputs["Blackbody Intensity"].default_value = 1.0 # Fire glow from temperature
vol.inputs["Temperature"].default_value = 1500.0      # Base temperature for color

nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])

# Assign to domain
domain.data.materials.append(mat)
```

## Temperature-to-Color Mapping for Fire

Fire color depends on temperature. Use the `Blackbody Intensity` input on Principled Volume
to automatically map simulation temperature to physically correct fire colors.

| Temperature (K) | Color |
|------------------|-------|
| 800-1000 | Deep red / dark orange |
| 1200-1500 | Orange / yellow-orange |
| 1500-2000 | Bright yellow |
| 2000-3000 | Yellow-white |
| 3000+ | Blue-white |

```python
# For custom color mapping, use Attribute node + ColorRamp
attr = nt.nodes.new("ShaderNodeAttribute")
attr.attribute_name = "temperature"        # Mantaflow temperature field

ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0, 0, 0, 1)          # No fire = black
el1 = ramp.color_ramp.elements.new(0.3)
el1.color = (0.8, 0.15, 0.0, 1)                            # Low temp = red-orange
el2 = ramp.color_ramp.elements.new(0.6)
el2.color = (1.0, 0.7, 0.1, 1)                             # Mid temp = orange-yellow
ramp.color_ramp.elements[1].position = 1.0
ramp.color_ramp.elements[1].color = (1.0, 1.0, 0.8, 1)    # High temp = bright yellow

nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], vol.inputs["Emission Color"])
```

## Complete: Fire Material With Color Ramp

```python
import bpy

mat = bpy.data.materials.new("M_FireVolume")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out = nt.nodes.new("ShaderNodeOutputMaterial")
vol = nt.nodes.new("ShaderNodeVolumePrincipled")

# Density from sim attribute
density_attr = nt.nodes.new("ShaderNodeAttribute")
density_attr.attribute_name = "density"
nt.links.new(density_attr.outputs["Fac"], vol.inputs["Density"])

# Temperature-driven emission color
temp_attr = nt.nodes.new("ShaderNodeAttribute")
temp_attr.attribute_name = "temperature"

ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
el_mid = ramp.color_ramp.elements.new(0.25)
el_mid.color = (0.9, 0.15, 0.0, 1)         # Deep orange
el_hot = ramp.color_ramp.elements.new(0.5)
el_hot.color = (1.0, 0.6, 0.05, 1)         # Bright orange-yellow
ramp.color_ramp.elements[1].position = 1.0
ramp.color_ramp.elements[1].color = (1.0, 1.0, 0.7, 1)  # Near-white

nt.links.new(temp_attr.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], vol.inputs["Emission Color"])

# Blackbody for physically-based fire glow
vol.inputs["Blackbody Intensity"].default_value = 2.0
vol.inputs["Temperature"].default_value = 1200.0

# Smoke settings
vol.inputs["Absorption Color"].default_value = (0.35, 0.25, 0.2, 1)
vol.inputs["Anisotropy"].default_value = 0.3

nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
domain.data.materials.append(mat)
```

## NanoVDB Conversion for DXR Renderer

OpenVDB files from Mantaflow bakes can be converted to NanoVDB for real-time DXR rendering.

```
# Command-line conversion (outside Blender)
# nanovdb_convert takes .vdb and outputs .nvdb
nanovdb_convert input.vdb output.nvdb --grid density --grid temperature

# PlasmaDXR config reference
# In configs/user/default.json, point to .nvdb file:
# "volume_path": "assets/volumes/fire_001.nvdb"
```

The DXR renderer uses NanoVDB grids for ray-marched volumetric rendering with:
- Density grid for smoke/absorption
- Temperature grid for fire emission color mapping
- Flame grid for fuel visualization

TECHNIQUE: Export with `openvdb_data_depth='16'` for DXR — half-float precision is sufficient
for real-time rendering and halves file size. Use '32' only when banding artifacts appear.

NOTE: Ensure `cache_data_format='OPENVDB'` is set BEFORE baking. Changing format after bake
requires re-baking the entire simulation.
""",
}

# ============================================================
# CODE PATTERNS — For code_pattern_memory seeding
# ============================================================

CODE_PATTERNS = [
    {
        "name": "campfire_multi_flow_with_dissolve",
        "issue": "campfire smoke accumulates and fills the entire domain instead of dissipating naturally",
        "code_snippet": """# Campfire domain with dissolve and separate flame/smoke flows
bpy.ops.mesh.primitive_cube_add(size=3.0, location=(0, 0, 1.5))
domain = bpy.context.active_object
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'GAS'
ds.resolution_max = 80
ds.use_dissolve_smoke = True          # NOT use_dissolve
ds.dissolve_speed = 40                # Frames to fully dissolve
ds.alpha = 1.0                        # Density buoyancy
ds.beta = 0.5                         # Heat buoyancy
ds.vorticity = 0.3                    # Swirl
ds.cache_type = 'ALL'
ds.cache_particle_format = 'UNI'     # NOT OPENVDB for GAS
ds.cache_mesh_format = 'BOBJECT'     # NOT UNI for GAS

# Flame source — high fuel/temp
flame_flow = flame_obj.modifiers["Fluid"].flow_settings
flame_flow.flow_type = 'BOTH'
flame_flow.fuel_amount = 1.5
flame_flow.temperature = 2.0
flame_flow.density = 0.8

# Smoke source — lower density, dark color
smoke_flow = smoke_obj.modifiers["Fluid"].flow_settings
smoke_flow.flow_type = 'SMOKE'
smoke_flow.density = 0.5
smoke_flow.temperature = 0.3
smoke_flow.smoke_color = (0.15, 0.12, 0.1)""",
        "effect_type": "fire",
        "improvement": 35.0,
        "experiment_id": "seed_mantaflow_adv_v1",
        "context_before": "Single flow source produces uniform fire+smoke; smoke fills domain and obscures the scene",
        "context_after": "Separate flame/smoke flows with use_dissolve_smoke=True and dissolve_speed=40 gives natural campfire behavior",
    },
    {
        "name": "water_pour_liquid_domain",
        "issue": "liquid simulation bakes produce empty domain with zero liquid visible",
        "code_snippet": """# Liquid domain with REQUIRED use_plane_init and mesh surface
mod = domain.modifiers.new("Fluid", type='FLUID')
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = 96
ds.use_plane_init = True              # CRITICAL — without this, bakes are empty
ds.use_mesh = True                    # Generate mesh surface from particles
ds.mesh_scale = 1
ds.use_adaptive_timesteps = True      # NOT use_adaptive_time_steps
ds.timesteps_max = 4                  # NOT timesteps_per_frame
ds.cache_type = 'ALL'
ds.simulation_method = 'FLIP'

# Inflow (water stream)
inflow_mod = inflow.modifiers.new("Fluid", type='FLUID')
inflow_mod.fluid_type = 'FLOW'
fs = inflow_mod.flow_settings
fs.flow_type = 'LIQUID'
fs.flow_behavior = 'INFLOW'
fs.use_initial_velocity = True
fs.velocity_normal = -3.0            # Downward pour""",
        "effect_type": "water",
        "improvement": 50.0,
        "experiment_id": "seed_mantaflow_adv_v1",
        "context_before": "Liquid domain bake completes but renders show empty domain — no water visible",
        "context_after": "use_plane_init=True is REQUIRED for liquid domains; use_mesh=True generates renderable surface mesh",
    },
    {
        "name": "colored_smoke_multi_source",
        "issue": "colored smoke from multiple emitters all renders as grey/white instead of distinct colors",
        "code_snippet": """# Colored smoke requires use_color_ramp on domain + smoke_color on flows
ds = domain.modifiers["Fluid"].domain_settings
ds.domain_type = 'GAS'
ds.use_color_ramp = True              # REQUIRED for colored smoke display

# Red source
red_fs = red_obj.modifiers["Fluid"].flow_settings
red_fs.flow_type = 'SMOKE'
red_fs.flow_behavior = 'INFLOW'
red_fs.smoke_color = (1.0, 0.1, 0.05)
red_fs.density = 1.0

# Blue source
blue_fs = blue_obj.modifiers["Fluid"].flow_settings
blue_fs.flow_type = 'SMOKE'
blue_fs.flow_behavior = 'INFLOW'
blue_fs.smoke_color = (0.05, 0.2, 1.0)
blue_fs.density = 1.0""",
        "effect_type": "smoke",
        "improvement": 30.0,
        "experiment_id": "seed_mantaflow_adv_v1",
        "context_before": "Multiple colored smoke emitters all produce identical grey smoke — colors not visible",
        "context_after": "Enable use_color_ramp on domain settings and set smoke_color on each flow object for distinct colored plumes",
    },
    {
        "name": "openvdb_export_gas_domain",
        "issue": "OpenVDB export fails with runtime error or produces empty/corrupt VDB files",
        "code_snippet": """# Correct OpenVDB export settings for GAS domain in Blender 5.0
ds = domain.modifiers["Fluid"].domain_settings
ds.domain_type = 'GAS'

# Data format
ds.cache_data_format = 'OPENVDB'
ds.openvdb_data_depth = '32'          # or '16' — NEVER 'NONE' (runtime error)
ds.openvdb_cache_compress_type = 'ZIP'  # ZIP or NONE — BLOSC removed in 5.0

# GAS-specific particle/mesh formats (different from LIQUID)
ds.cache_particle_format = 'UNI'     # OPENVDB not valid for GAS particles
ds.cache_mesh_format = 'BOBJECT'     # UNI not valid for GAS mesh

ds.cache_type = 'ALL'                # Required for full bake

# Set cache dir before baking
import os
ds.cache_directory = os.environ.get('BLENDER_CACHE_DIR', '/tmp/blender_cache')

# Bake (domain must be active object in OBJECT mode)
bpy.context.view_layer.objects.active = domain
bpy.ops.fluid.bake_all()""",
        "effect_type": "fire",
        "improvement": 40.0,
        "experiment_id": "seed_mantaflow_adv_v1",
        "context_before": "VDB export crashes or produces empty files due to wrong format settings",
        "context_after": "openvdb_data_depth must be '32' or '16' (not 'NONE'), compression must be 'ZIP' or 'NONE' (not 'BLOSC'), GAS uses UNI/BOBJECT formats",
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
    print("\nTest query: 'mantaflow liquid simulation water pour mesh surface'")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query="mantaflow liquid simulation water pour mesh surface",
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
    """Seed code pattern memory with advanced Mantaflow techniques."""
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
            # Mark as applicable to all mantaflow-related effect types
            p = memory.patterns[pid]
            p.effect_types = ["fire", "smoke", "explosion", "water", "fog", "nebula"]
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
        description="Seed advanced Mantaflow technique knowledge into vector store + code pattern memory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--patterns-only", action="store_true", help="Only seed code patterns")
    parser.add_argument("--docs-only", action="store_true", help="Only upload technique docs")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED ADVANCED MANTAFLOW TECHNIQUES")
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
