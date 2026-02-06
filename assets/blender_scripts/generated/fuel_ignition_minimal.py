#!/usr/bin/env python3
"""
Minimal fire test - uses quick_smoke with the simplest possible volume material.
"""

import bpy
import os

ASSET_NAME = "fuel_ignition_minimal"
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

# Use EEVEE to avoid Cycles crash (for now)
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.image_settings.file_format = 'PNG'

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

    # Create a SIMPLE volume material - no complex nodes
    mat = bpy.data.materials.new("FireVolume")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pvol = nt.nodes.new('ShaderNodeVolumePrincipled')

    # Use built-in attributes
    pvol.inputs['Density'].default_value = 1.0
    pvol.inputs['Emission Strength'].default_value = 5.0
    pvol.inputs['Emission Color'].default_value = (1.0, 0.4, 0.1, 1.0)  # Orange
    pvol.inputs['Blackbody Intensity'].default_value = 1.0
    pvol.inputs['Temperature'].default_value = 1500.0

    nt.links.new(pvol.outputs['Volume'], out.inputs['Volume'])

    # Apply material to domain
    if domain_obj.data.materials:
        domain_obj.data.materials[0] = mat
    else:
        domain_obj.data.materials.append(mat)

# Add a simple ground plane
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"

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
bpy.ops.render.render(write_still=True)
print(f"Rendered: {scene.render.filepath}")

# Save blend
blend_path = os.path.join(OUTPUT_DIR, f"{ASSET_NAME}.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Saved: {blend_path}")

print("=== MINIMAL FIRE TEST COMPLETE ===")
