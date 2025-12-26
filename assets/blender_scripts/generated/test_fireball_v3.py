#!/usr/bin/env python3
"""
GPT-5.2 — Test Fireball v3 (Improved) - Blender 5.0+

Iteration 2: More dramatic explosion with:
- Higher turbulence/vorticity
- Better fire-to-smoke ratio
- Volumetric material included
- Proper camera framing
"""

import bpy
import sys
from pathlib import Path
from math import radians

# =============================================================================
# Configuration
# =============================================================================

class Config:
    RESOLUTION = 96  # Increased from 64
    FRAME_START = 1
    FRAME_END = 50   # More frames for better motion
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/test_fireball_v3"
    BAKE = True
    DOMAIN_SCALE = 5.0  # Larger domain

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--resolution" and i + 1 < len(argv):
            Config.RESOLUTION = int(argv[i + 1])
            i += 2
        elif arg == "--frame_end" and i + 1 < len(argv):
            Config.FRAME_END = int(argv[i + 1])
            i += 2
        elif arg == "--bake" and i + 1 < len(argv):
            Config.BAKE = argv[i + 1].lower() in ("1", "true", "yes")
            i += 2
        else:
            i += 1

parse_args()

# =============================================================================
# Scene Setup
# =============================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_scene():
    scene = bpy.context.scene
    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # Better render settings
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 128

# =============================================================================
# Domain with Volumetric Material
# =============================================================================

def create_domain():
    # Create domain cube
    bpy.ops.mesh.primitive_cube_add(size=Config.DOMAIN_SCALE, location=(0, 0, Config.DOMAIN_SCALE/2))
    domain = bpy.context.active_object
    domain.name = "FluidDomain"

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'GAS'

    # Resolution
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = True

    # Cache settings
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'

    # IMPROVED: More dramatic fire behavior
    settings.burning_rate = 0.9          # Faster burning
    settings.flame_smoke = 0.8           # More smoke from flames
    settings.flame_vorticity = 0.8       # More turbulent swirls
    settings.flame_max_temp = 4.0        # Hotter flames

    # Buoyancy for rising effect
    settings.alpha = 1.0                 # Density affects rising
    settings.beta = 2.0                  # Temperature affects rising more

    # Add volumetric material
    add_volume_material(domain)

    return domain

def add_volume_material(domain):
    """Add Principled Volume shader for fire/smoke visualization."""
    mat = bpy.data.materials.new(name="FireSmokeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Principled Volume with fire settings
    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.inputs['Color'].default_value = (1.0, 0.25, 0.02, 1.0)  # Deep orange
    volume.inputs['Density'].default_value = 3.0
    volume.inputs['Anisotropy'].default_value = 0.3  # Forward scattering
    volume.inputs['Blackbody Intensity'].default_value = 3.0  # Bright fire glow
    volume.inputs['Temperature'].default_value = 2200.0  # Hot!

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    domain.data.materials.append(mat)
    print("[script] Volumetric material applied")

# =============================================================================
# Emitter
# =============================================================================

def create_emitter():
    # Icosphere for more interesting emission shape
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.4, location=(0, 0, 0.5))
    emitter = bpy.context.active_object
    emitter.name = "FlowEmitter"
    emitter.hide_render = True  # Don't render the emitter mesh

    bpy.ops.object.modifier_add(type='FLUID')
    emitter.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = emitter.modifiers["Fluid"].flow_settings
    flow.flow_type = 'BOTH'  # Fire AND smoke
    flow.flow_behavior = 'INFLOW'
    flow.fuel_amount = 1.5   # More fuel
    flow.temperature = 3.0   # Hotter initial temp

    # Animate: burst then stop (explosion-like)
    flow.keyframe_insert(data_path="fuel_amount", frame=1)
    flow.fuel_amount = 0.0
    flow.keyframe_insert(data_path="fuel_amount", frame=10)

    return emitter

# =============================================================================
# Camera & Lighting
# =============================================================================

def setup_camera():
    bpy.ops.object.camera_add(location=(6, -6, 4))
    cam = bpy.context.active_object
    cam.name = "Camera"

    # Point at explosion center
    bpy.ops.object.constraint_add(type='TRACK_TO')
    cam.constraints["Track To"].target = bpy.data.objects.get("FluidDomain")
    cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints["Track To"].up_axis = 'UP_Y'

    bpy.context.scene.camera = cam

def setup_lighting():
    # Key light (sun)
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-4, -4, 3))
    fill = bpy.context.active_object
    fill.data.energy = 100.0
    fill.data.size = 3.0

# =============================================================================
# Baking
# =============================================================================

def bake_simulation(domain):
    print(f"[script] Baking: frames {Config.FRAME_START}-{Config.FRAME_END}")
    print(f"[script] Resolution: {Config.RESOLUTION}")
    print(f"[script] Output: {Config.OUTPUT_DIR}")

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)
    bpy.ops.fluid.bake_all()

    print("[script] Bake complete!")

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Test Fireball v3 - Improved Explosion")
    print("=" * 60)

    clear_scene()
    setup_scene()

    domain = create_domain()
    emitter = create_emitter()
    setup_camera()
    setup_lighting()

    if Config.BAKE:
        bake_simulation(domain)

    # Save blend file
    blend_path = Path(Config.OUTPUT_DIR) / "test_fireball_v3.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {blend_path}")

if __name__ == "__main__":
    main()
