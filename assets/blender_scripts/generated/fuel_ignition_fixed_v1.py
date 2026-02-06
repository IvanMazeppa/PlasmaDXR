#!/usr/bin/env python3
"""
fuel_ignition_fixed_v1

Fixed version of fuel_ignition_20260203_232244_fire_mantaflow_v1.py
- Removed MixRGB nodes from volume material (causes Cycles crash)
- Reduced frames for quick testing (1-24 instead of 1-96)
- Single render frame for validation
"""

import bpy
import os
import math
from mathutils import Vector, Euler

# ==============================================================
# Fire VFX: Ethanol pool ignition (Mantaflow) - FIXED VERSION
# ==============================================================

ASSET_NAME = "fuel_ignition_fixed_v1"
OUTPUT_DIR = os.environ.get("BLENDER_OUTPUT_DIR", f"/tmp/{ASSET_NAME}")
CACHE_DIR = os.path.join(OUTPUT_DIR, "mantaflow_cache")
RENDER_DIR = os.path.join(OUTPUT_DIR, "renders")

# Reduced for testing
FRAME_START = 1
FRAME_END = 24  # Reduced from 96
FPS = 24

# Physical dimensions (meters)
TRAY_X = 0.30
TRAY_Y = 0.20
TRAY_DEPTH = 0.025
POOL_DEPTH = 0.01

# Tighter domain to reduce memory
DOMAIN_X = 0.45
DOMAIN_Y = 0.35
DOMAIN_Z = 0.40

# Lower resolution for testing
RESOLUTION_MAX = 64  # Reduced from 128

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    try:
        for _ in range(3):
            bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def set_world(scene):
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    n_out = nodes.new("ShaderNodeOutputWorld")
    n_bg = nodes.new("ShaderNodeBackground")
    n_bg.inputs[0].default_value = (0.06, 0.07, 0.085, 1.0)
    n_bg.inputs[1].default_value = 0.6
    links.new(n_bg.outputs[0], n_out.inputs[0])


def setup_scene_and_render():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CACHE_DIR)
    ensure_dir(RENDER_DIR)

    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128  # Reduced for testing
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02

    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences
        cycles_addon = prefs.addons.get('cycles')
        if cycles_addon is not None:
            cprefs = cycles_addon.preferences
            for t in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    cprefs.compute_device_type = t
                    break
                except Exception:
                    continue
    except Exception:
        pass

    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = -0.6

    scene.render.resolution_x = 1280  # Reduced for testing
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False

    set_world(scene)
    return scene


def look_at(obj, target: Vector):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def create_material_concrete(name="MAT_Concrete"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.22, 0.24, 0.26, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.92
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def create_material_steel_tray(name="MAT_SteelTray"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.35, 0.36, 0.37, 1.0)
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.52
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def create_material_ethanol(name="MAT_Ethanol"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Blender 5.0: Transmission Weight
    try:
        bsdf.inputs['Transmission Weight'].default_value = 1.0
    except KeyError:
        pass

    bsdf.inputs['Roughness'].default_value = 0.02
    bsdf.inputs['IOR'].default_value = 1.36
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def create_material_fire_volume(name="MAT_FireVolume"):
    """CRASH-SAFE volume material - NO MixRGB nodes.

    Uses only Principled Volume with Blackbody for emission color.
    Avoids the Cycles bug in ShaderGraph::optimize_volume_output.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)

    pv = nodes.new('ShaderNodeVolumePrincipled')
    pv.location = (100, 0)

    # Attributes from the sim cache
    attr_density = nodes.new('ShaderNodeAttribute')
    attr_density.location = (-400, 100)
    attr_density.attribute_name = "density"

    attr_temp = nodes.new('ShaderNodeAttribute')
    attr_temp.location = (-400, -50)
    attr_temp.attribute_name = "temperature"

    attr_flame = nodes.new('ShaderNodeAttribute')
    attr_flame.location = (-400, -200)
    attr_flame.attribute_name = "flame"

    # Temperature to Kelvin mapping
    temp_map = nodes.new('ShaderNodeMapRange')
    temp_map.location = (-150, -50)
    temp_map.inputs['From Min'].default_value = 0.0
    temp_map.inputs['From Max'].default_value = 2.0
    temp_map.inputs['To Min'].default_value = 1000.0
    temp_map.inputs['To Max'].default_value = 2200.0
    temp_map.clamp = True

    # Blackbody for emission color (NO MixRGB!)
    bb = nodes.new('ShaderNodeBlackbody')
    bb.location = (-150, -200)

    # Density scaling
    dens_mul = nodes.new('ShaderNodeMath')
    dens_mul.location = (-150, 100)
    dens_mul.operation = 'MULTIPLY'
    dens_mul.inputs[1].default_value = 6.0

    # Flame to emission strength
    flame_mul = nodes.new('ShaderNodeMath')
    flame_mul.location = (-150, -350)
    flame_mul.operation = 'MULTIPLY'
    flame_mul.inputs[1].default_value = 8.0

    # Wire up
    links.new(attr_density.outputs['Fac'], dens_mul.inputs[0])
    links.new(dens_mul.outputs['Value'], pv.inputs['Density'])

    links.new(attr_temp.outputs['Fac'], temp_map.inputs['Value'])
    links.new(temp_map.outputs['Result'], bb.inputs['Temperature'])

    # Direct blackbody to emission color (no MixRGB!)
    links.new(bb.outputs['Color'], pv.inputs['Emission Color'])

    links.new(attr_flame.outputs['Fac'], flame_mul.inputs[0])
    links.new(flame_mul.outputs['Value'], pv.inputs['Emission Strength'])

    # Blackbody intensity for fire glow
    pv.inputs['Blackbody Intensity'].default_value = 1.0
    pv.inputs['Anisotropy'].default_value = 0.1

    links.new(pv.outputs['Volume'], out.inputs['Volume'])

    return mat


def build_environment(mats):
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "Floor_Concrete"
    floor.data.materials.append(mats['concrete'])

    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, -1.6, 1.0), rotation=(math.radians(90), 0, 0))
    back = bpy.context.active_object
    back.name = "Backdrop"
    back.data.materials.append(mats['concrete'])

    return floor


def build_tray_and_pool(mats):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, TRAY_DEPTH * 0.5))
    tray = bpy.context.active_object
    tray.name = "DripTray"
    tray.scale = (TRAY_X * 0.5, TRAY_Y * 0.5, TRAY_DEPTH * 0.5)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    solid = tray.modifiers.new(name="Solidify", type='SOLIDIFY')
    solid.thickness = 0.002
    solid.offset = 1.0

    bev = tray.modifiers.new(name="Bevel", type='BEVEL')
    bev.width = 0.002
    bev.segments = 3

    tray.data.materials.append(mats['steel'])

    pool_z = 0.003
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, pool_z))
    pool = bpy.context.active_object
    pool.name = "Ethanol_Pool"
    pool.scale = (TRAY_X * 0.5 * 0.92, TRAY_Y * 0.5 * 0.92, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pool.data.materials.append(mats['ethanol'])

    return tray, pool


def setup_lighting(tray_obj):
    bpy.ops.object.light_add(type='AREA', location=(0.55, 0.35, 0.75))
    key = bpy.context.active_object
    key.name = "Key_Area"
    key.data.energy = 180
    key.data.size = 0.6
    key.data.color = (0.85, 0.90, 1.0)

    bpy.ops.object.light_add(type='POINT', location=(-0.55, -0.35, 0.35))
    fill = bpy.context.active_object
    fill.name = "Fill_Point"
    fill.data.energy = 40
    fill.data.color = (1.0, 0.85, 0.70)

    return [key, fill]


def setup_camera(target_obj):
    bpy.ops.object.camera_add(location=(0.55, -0.55, 0.55))
    cam = bpy.context.active_object
    cam.name = "Camera_Main"
    cam.data.lens = 50
    look_at(cam, target_obj.location + Vector((0.0, 0.0, 0.03)))
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = target_obj
    cam.data.dof.aperture_fstop = 2.8
    bpy.context.scene.camera = cam
    return cam


def create_domain(mats, location=(0.0, 0.0, 0.20)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    domain_obj = bpy.context.active_object
    domain_obj.name = "Mantaflow_Domain"
    domain_obj.scale = (DOMAIN_X * 0.5, DOMAIN_Y * 0.5, DOMAIN_Z * 0.5)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    domain_obj.data.materials.append(mats['fire_volume'])

    mod = domain_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'DOMAIN'
    dset = mod.domain_settings

    dset.domain_type = 'GAS'
    dset.resolution_max = int(RESOLUTION_MAX)

    dset.use_noise = True
    dset.noise_strength = 0.5
    dset.noise_scale = 2  # Reduced from 3

    dset.vorticity = 0.75

    dset.use_adaptive_timesteps = True
    dset.timesteps_max = 8

    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 10

    dset.cache_directory = CACHE_DIR
    dset.cache_type = 'ALL'
    dset.openvdb_cache_compress_type = 'ZIP'

    domain_obj.display_type = 'WIRE'

    return domain_obj


def add_flow_object(obj, flow_type='BOTH', temperature=1.0, density=0.5, vel_normal=0.1):
    mod = obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'FLOW'
    fset = mod.flow_settings

    fset.flow_type = flow_type
    fset.flow_behavior = 'INFLOW'
    fset.use_initial_velocity = True
    fset.velocity_normal = float(vel_normal)
    fset.temperature = float(temperature)
    fset.density = float(density)

    return mod


def build_emitters(pool_obj):
    emitters = {}

    # Single emitter for quick test - expanding plane
    plane_z = pool_obj.location.z + 0.004
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, plane_z))
    emitter = bpy.context.active_object
    emitter.name = "Emitter_Fire"
    emitter.scale = (TRAY_X * 0.5 * 0.92, TRAY_Y * 0.5 * 0.92, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    em_mod = add_flow_object(emitter, flow_type='BOTH', temperature=1.2, density=0.3, vel_normal=0.15)

    # Simple animation: start thin, expand
    emitter.scale = (0.1, 1.0, 1.0)
    emitter.keyframe_insert(data_path="scale", frame=1)
    emitter.scale = (1.0, 1.0, 1.0)
    emitter.keyframe_insert(data_path="scale", frame=12)

    emitters['main'] = emitter

    return emitters


def bake_simulation(domain_obj):
    bpy.context.view_layer.objects.active = domain_obj
    domain_obj.select_set(True)

    try:
        if bpy.ops.fluid.free_all.poll():
            bpy.ops.fluid.free_all()
    except Exception:
        pass

    try:
        if bpy.ops.fluid.bake_all.poll():
            bpy.ops.fluid.bake_all()
            print("Mantaflow bake completed.")
    except Exception as e:
        print(f"Bake failed: {e}")


def render_frames(scene, frames):
    for fr in frames:
        scene.frame_set(fr)
        scene.render.filepath = os.path.join(RENDER_DIR, f"{ASSET_NAME}_f{fr:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered frame {fr}")


def save_blend():
    blend_path = os.path.join(OUTPUT_DIR, f"{ASSET_NAME}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved .blend to: {blend_path}")


def main():
    clear_scene()
    scene = setup_scene_and_render()

    mats = {
        'concrete': create_material_concrete(),
        'steel': create_material_steel_tray(),
        'ethanol': create_material_ethanol(),
        'fire_volume': create_material_fire_volume(),
    }

    build_environment(mats)
    tray, pool = build_tray_and_pool(mats)

    setup_lighting(tray)
    setup_camera(pool)

    domain = create_domain(mats, location=(0.0, 0.0, DOMAIN_Z * 0.5 * 0.9))
    build_emitters(pool)

    # Bake simulation
    bake_simulation(domain)

    # Single test render at frame 12 (fire should be spreading)
    render_frames(scene, frames=[12])

    save_blend()

    print("\n=== FUEL IGNITION TEST COMPLETE ===")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Render: {RENDER_DIR}")


if __name__ == "__main__":
    main()
