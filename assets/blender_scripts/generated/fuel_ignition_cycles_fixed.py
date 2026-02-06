#!/usr/bin/env python3
"""
Fire test with Cycles - CRASH-SAFE volume material (no MixRGB nodes).
"""

import bpy
import os

ASSET_NAME = "fuel_ignition_cycles_fixed"
OUTPUT_DIR = f"/tmp/{ASSET_NAME}"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
RENDER_DIR = os.path.join(OUTPUT_DIR, "renders")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Setup scene
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 24
scene.render.fps = 24

# Use Cycles with reduced samples
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.05

# Try GPU
try:
    scene.cycles.device = 'GPU'
except:
    pass

scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.image_settings.file_format = 'PNG'

# Color management
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Medium Contrast'

# Create emitter plane (fire source)
bpy.ops.mesh.primitive_plane_add(size=0.3, location=(0, 0, 0.01))
emitter = bpy.context.active_object
emitter.name = "FireEmitter"

# Use quick_smoke to create domain and flow automatically
bpy.ops.object.quick_smoke(style='BOTH', show_flows=True)

# Find the domain that was created
domain_obj = None
for obj in scene.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'domain_settings') and mod.domain_settings is not None:
            domain_obj = obj
            dset = mod.domain_settings
            break
    if domain_obj:
        break

if domain_obj:
    print(f"Domain found: {domain_obj.name}")

    # Configure domain
    dset.resolution_max = 64
    dset.use_noise = True
    dset.noise_scale = 2
    dset.noise_strength = 0.5
    dset.vorticity = 0.7
    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 8
    dset.cache_directory = CACHE_DIR
    dset.cache_type = 'ALL'

    # Create CRASH-SAFE volume material
    # KEY: No MixRGB nodes - direct connections only
    mat = bpy.data.materials.new("FireVolume_Safe")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)

    pvol = nt.nodes.new('ShaderNodeVolumePrincipled')
    pvol.location = (100, 0)

    # Attribute nodes for sim data
    attr_dens = nt.nodes.new('ShaderNodeAttribute')
    attr_dens.location = (-300, 100)
    attr_dens.attribute_name = "density"

    attr_flame = nt.nodes.new('ShaderNodeAttribute')
    attr_flame.location = (-300, -100)
    attr_flame.attribute_name = "flame"

    attr_temp = nt.nodes.new('ShaderNodeAttribute')
    attr_temp.location = (-300, -300)
    attr_temp.attribute_name = "temperature"

    # Density multiplier
    dens_mul = nt.nodes.new('ShaderNodeMath')
    dens_mul.location = (-100, 100)
    dens_mul.operation = 'MULTIPLY'
    dens_mul.inputs[1].default_value = 5.0

    # Emission strength from flame
    flame_mul = nt.nodes.new('ShaderNodeMath')
    flame_mul.location = (-100, -100)
    flame_mul.operation = 'MULTIPLY'
    flame_mul.inputs[1].default_value = 8.0

    # Temperature to Kelvin mapping
    temp_map = nt.nodes.new('ShaderNodeMapRange')
    temp_map.location = (-100, -300)
    temp_map.inputs['From Min'].default_value = 0.0
    temp_map.inputs['From Max'].default_value = 2.0
    temp_map.inputs['To Min'].default_value = 1000.0
    temp_map.inputs['To Max'].default_value = 2000.0

    # Blackbody for emission color (NO MixRGB!)
    bb = nt.nodes.new('ShaderNodeBlackbody')
    bb.location = (100, -200)

    # Wire up - SIMPLE, NO MIXING
    nt.links.new(attr_dens.outputs['Fac'], dens_mul.inputs[0])
    nt.links.new(dens_mul.outputs['Value'], pvol.inputs['Density'])

    nt.links.new(attr_flame.outputs['Fac'], flame_mul.inputs[0])
    nt.links.new(flame_mul.outputs['Value'], pvol.inputs['Emission Strength'])

    nt.links.new(attr_temp.outputs['Fac'], temp_map.inputs['Value'])
    nt.links.new(temp_map.outputs['Result'], bb.inputs['Temperature'])

    # Direct blackbody to emission (NO MIXING!)
    nt.links.new(bb.outputs['Color'], pvol.inputs['Emission Color'])

    # Set other volume properties
    pvol.inputs['Anisotropy'].default_value = 0.1
    pvol.inputs['Blackbody Intensity'].default_value = 1.0

    nt.links.new(pvol.outputs['Volume'], out.inputs['Volume'])

    # Apply material to domain
    if domain_obj.data.materials:
        domain_obj.data.materials[0] = mat
    else:
        domain_obj.data.materials.append(mat)

    print("Crash-safe volume material applied")

# Add a simple ground plane
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground_mat = bpy.data.materials.new("GroundMat")
ground_mat.use_nodes = True
ground_mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.2, 0.2, 0.2, 1)
ground.data.materials.append(ground_mat)

# World background
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg:
    bg.inputs['Color'].default_value = (0.05, 0.05, 0.07, 1)
    bg.inputs['Strength'].default_value = 0.5

# Add camera
bpy.ops.object.camera_add(location=(1.0, -1.0, 0.8))
cam = bpy.context.active_object
cam.name = "Camera"
cam.rotation_euler = (1.1, 0, 0.8)
scene.camera = cam

# Add light
bpy.ops.object.light_add(type='AREA', location=(0.5, -0.5, 1.5))
light = bpy.context.active_object
light.data.energy = 100
light.data.size = 1.0

# Bake simulation
if domain_obj:
    bpy.context.view_layer.objects.active = domain_obj
    domain_obj.select_set(True)

    try:
        bpy.ops.fluid.bake_all()
        print("Bake completed!")
    except Exception as e:
        print(f"Bake failed: {e}")

# Render frame 12
scene.frame_set(12)
scene.render.filepath = os.path.join(RENDER_DIR, f"{ASSET_NAME}_f0012.png")
print("Starting Cycles render...")
bpy.ops.render.render(write_still=True)
print(f"Rendered: {scene.render.filepath}")

# Save blend
blend_path = os.path.join(OUTPUT_DIR, f"{ASSET_NAME}.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Saved: {blend_path}")

print("=== CYCLES FIRE TEST COMPLETE ===")
