#!/usr/bin/env python3
"""
Water Pouring Into Bowl - Blender 5.0+

Continuous water stream pouring into a bowl, showing splash and ripple physics.
Features collision with bowl geometry and realistic water accumulation.

Usage:
    blender --background --python water_pouring_bowl.py -- [options]
"""

import bpy
import sys
import math
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

class Config:
    RESOLUTION = 96  # Increased from 64 for better collision detection
    FRAME_START = 1
    FRAME_END = 100  # Extended for more water and settling time
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/water_pouring_bowl_v3"
    BAKE = True
    RENDER = True
    RENDER_FRAMES = "mid"

    # Render settings
    RENDER_RESOLUTION_X = 512
    RENDER_RESOLUTION_Y = 512
    RENDER_SAMPLES = 128

    # Domain size
    DOMAIN_SIZE = 5.0

    # Bowl settings
    BOWL_RADIUS = 1.2
    BOWL_DEPTH = 0.8
    BOWL_THICKNESS = 0.12  # Increased from 0.08 for better collision

    # Pour stream settings - increased radius for more water
    STREAM_RADIUS = 0.12  # Increased from 0.08
    STREAM_HEIGHT = 2.0  # Height above bowl
    STREAM_OFFSET_X = 0.3  # Offset from center for angle


def parse_args():
    """Parse command line arguments."""
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
        elif arg == "--frame_start" and i + 1 < len(argv):
            Config.FRAME_START = int(argv[i + 1])
            i += 2
        elif arg == "--frame_end" and i + 1 < len(argv):
            Config.FRAME_END = int(argv[i + 1])
            i += 2
        elif arg == "--output_dir" and i + 1 < len(argv):
            Config.OUTPUT_DIR = argv[i + 1]
            i += 2
        elif arg == "--bake" and i + 1 < len(argv):
            Config.BAKE = argv[i + 1].lower() in ("1", "true", "yes")
            i += 2
        elif arg == "--render" and i + 1 < len(argv):
            Config.RENDER = argv[i + 1].lower() in ("1", "true", "yes")
            i += 2
        elif arg == "--render_frames" and i + 1 < len(argv):
            Config.RENDER_FRAMES = argv[i + 1]
            i += 2
        else:
            i += 1

parse_args()


# =============================================================================
# Scene Setup
# =============================================================================

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def setup_scene():
    """Configure scene settings."""
    scene = bpy.context.scene
    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True


# =============================================================================
# Domain Creation
# =============================================================================

def create_domain():
    """Create fluid domain for liquid simulation."""
    bpy.ops.mesh.primitive_cube_add(
        size=Config.DOMAIN_SIZE,
        location=(0, 0, Config.DOMAIN_SIZE / 2 - 0.5)
    )
    domain = bpy.context.active_object
    domain.name = "LiquidDomain"

    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'LIQUID'

    # Resolution
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = False

    # FLIP ratio for splashes
    settings.flip_ratio = 0.85

    # More timesteps for accurate stream behavior
    settings.timesteps_max = 6
    settings.timesteps_min = 2
    settings.cfl_condition = 3.0

    # Cache settings
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'

    # Mesh generation
    settings.use_mesh = True
    settings.mesh_scale = 1

    # Secondary particles for splash spray
    settings.use_spray_particles = True
    settings.use_foam_particles = True
    settings.use_bubble_particles = True

    # Particle settings
    settings.sndparticle_potential_max_wavecrest = 4.0
    settings.sndparticle_potential_min_wavecrest = 0.3
    settings.sndparticle_potential_max_energy = 4.0
    settings.sndparticle_potential_min_energy = 0.1
    settings.sndparticle_sampling_wavecrest = 40
    settings.sndparticle_sampling_trappedair = 40
    settings.sndparticle_life_max = 60.0
    settings.sndparticle_life_min = 15.0

    # Standard gravity
    settings.gravity = (0, 0, -9.81)

    # Domain visible as wireframe in viewport
    domain.display_type = 'WIRE'

    return domain


# =============================================================================
# Bowl Creation (Effector)
# =============================================================================

def create_bowl():
    """Create a bowl as fluid effector/obstacle."""
    # Create bowl using a boolean difference (outer - inner sphere)
    # Or use a simple cylinder with open top

    # Method: Create torus-like bowl shape
    # Outer shell
    bpy.ops.mesh.primitive_cylinder_add(
        radius=Config.BOWL_RADIUS,
        depth=Config.BOWL_DEPTH,
        location=(0, 0, Config.BOWL_DEPTH / 2)
    )
    bowl_outer = bpy.context.active_object
    bowl_outer.name = "BowlOuter"

    # Inner cavity (to subtract)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=Config.BOWL_RADIUS - Config.BOWL_THICKNESS,
        depth=Config.BOWL_DEPTH - Config.BOWL_THICKNESS,
        location=(0, 0, Config.BOWL_DEPTH / 2 + Config.BOWL_THICKNESS / 2)
    )
    bowl_inner = bpy.context.active_object
    bowl_inner.name = "BowlInner"

    # Boolean difference to create hollow bowl
    bpy.context.view_layer.objects.active = bowl_outer
    bool_mod = bowl_outer.modifiers.new(name="BooleanCut", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = bowl_inner

    # Apply the boolean
    bpy.ops.object.modifier_apply(modifier="BooleanCut")

    # Delete the inner object
    bpy.data.objects.remove(bowl_inner, do_unlink=True)

    bowl = bowl_outer
    bowl.name = "Bowl"

    # Make it a fluid effector (collision object)
    bpy.ops.object.modifier_add(type='FLUID')
    bowl.modifiers["Fluid"].fluid_type = 'EFFECTOR'

    effector = bowl.modifiers["Fluid"].effector_settings
    effector.effector_type = 'COLLISION'
    effector.use_effector = True

    # CRITICAL: Prevent fluid leaking through thin walls
    effector.surface_distance = 0.15  # Extra collision padding around mesh (prevents leaking)
    effector.subframes = 5  # More substeps for collision detection

    print("[script] Bowl effector: surface_distance=0.15, subframes=5 (leak prevention)")

    # Add ceramic/porcelain material
    add_bowl_material(bowl)

    return bowl


def add_bowl_material(bowl):
    """Add glass material to bowl for transparent look."""
    mat = bpy.data.materials.new(name="GlassBowlMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes and use Glass BSDF
    nodes.clear()

    # Glass BSDF for transparent bowl
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 0)
    glass.inputs['Color'].default_value = (0.95, 0.98, 1.0, 1.0)  # Very slight blue tint
    glass.inputs['Roughness'].default_value = 0.02  # Nearly smooth glass
    glass.inputs['IOR'].default_value = 1.52  # Glass IOR

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    bowl.data.materials.append(mat)
    print("[script] Bowl material: Glass BSDF (IOR 1.52, transparent)")


# =============================================================================
# Water Stream (Inflow)
# =============================================================================

def create_pour_stream():
    """Create continuous water inflow stream."""
    # Small cylinder for stream source
    bpy.ops.mesh.primitive_cylinder_add(
        radius=Config.STREAM_RADIUS,
        depth=0.15,
        location=(Config.STREAM_OFFSET_X, 0, Config.STREAM_HEIGHT)
    )
    stream = bpy.context.active_object
    stream.name = "PourStream"

    # Slight rotation for angled pour
    stream.rotation_euler = (math.radians(10), 0, 0)

    # Make it a fluid inflow
    bpy.ops.object.modifier_add(type='FLUID')
    stream.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = stream.modifiers["Fluid"].flow_settings
    flow.flow_type = 'LIQUID'
    flow.flow_behavior = 'INFLOW'  # Continuous flow

    # Flow rate and velocity - increased for more water
    flow.use_initial_velocity = True
    flow.velocity_normal = 3.0  # Increased downward velocity
    flow.velocity_coord = (0, 0, -4.0)  # Stronger downward push

    # Hide stream geometry
    stream.hide_render = True

    return stream


def animate_pour_stream(stream):
    """Animate the pour to start and stop."""
    flow = stream.modifiers["Fluid"].flow_settings

    scene = bpy.context.scene

    # Start pouring at frame 1
    scene.frame_set(1)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow')

    # Continue pouring until frame 75 (extended for more water)
    scene.frame_set(75)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow')

    # Stop pouring at frame 76
    scene.frame_set(76)
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow')

    # Keep stopped for settling
    scene.frame_set(Config.FRAME_END)
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow')

    scene.frame_set(1)
    print("[script] Pour animation: frames 1-75 pouring, 76+ settling (more water)")


# =============================================================================
# Water Material
# =============================================================================

def add_water_material(domain):
    """Add realistic water material."""
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Glass BSDF for water
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 0)
    glass.inputs['Color'].default_value = (0.85, 0.92, 1.0, 1.0)  # Slight blue
    glass.inputs['Roughness'].default_value = 0.0
    glass.inputs['IOR'].default_value = 1.33

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    domain.data.materials.append(mat)
    print("[script] Water material applied (Glass BSDF, IOR 1.33)")


# =============================================================================
# Camera and Lighting
# =============================================================================

def setup_camera_and_lighting():
    """Setup camera and lighting for bowl pour shot."""
    # Camera - 3/4 view to see bowl and stream
    bpy.ops.object.camera_add(
        location=(3.0, -3.0, 2.5)
    )
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    cam.data.lens = 50
    bpy.context.scene.camera = cam

    # Key light (main)
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 4))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 400
    key.data.size = 2.5
    key.rotation_euler = (math.radians(50), 0, math.radians(20))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -1, 2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 150
    fill.data.size = 3

    # Rim light for water highlights
    bpy.ops.object.light_add(type='SPOT', location=(0, 3, 2))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 600
    rim.data.spot_size = math.radians(50)
    rim.rotation_euler = (math.radians(110), 0, math.radians(180))

    # World background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.15, 0.18, 0.22, 1.0)  # Neutral gray-blue
    bg.inputs['Strength'].default_value = 0.8

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (200, 0)
    world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

    # Add a floor/surface
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Surface"

    floor_mat = bpy.data.materials.new(name="SurfaceMaterial")
    floor_mat.use_nodes = True
    floor_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    if floor_bsdf:
        floor_bsdf.inputs['Base Color'].default_value = (0.25, 0.25, 0.28, 1.0)
        floor_bsdf.inputs['Roughness'].default_value = 0.5
    floor.data.materials.append(floor_mat)


# =============================================================================
# Baking & Rendering
# =============================================================================

def bake_simulation(domain):
    """Bake the liquid simulation."""
    print(f"[script] Baking liquid: frames {Config.FRAME_START}-{Config.FRAME_END}")
    print(f"[script] Resolution: {Config.RESOLUTION}")
    print(f"[script] Output: {Config.OUTPUT_DIR}")

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()
    print("[script] Bake complete!")


def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True

    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    print(f"[script] Render: {Config.RENDER_RESOLUTION_X}x{Config.RENDER_RESOLUTION_Y}, {Config.RENDER_SAMPLES} samples")


def get_frames_to_render():
    """Determine which frames to render."""
    if Config.RENDER_FRAMES == "mid":
        mid = (Config.FRAME_START + Config.FRAME_END) // 2
        return [mid]
    elif Config.RENDER_FRAMES == "all":
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 6)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    elif Config.RENDER_FRAMES == "pour":
        # Key moments: start, mid-pour, splash, settling
        return [10, 25, 40, 55, 70]
    else:
        try:
            return [int(f.strip()) for f in Config.RENDER_FRAMES.split(",")]
        except ValueError:
            return [(Config.FRAME_START + Config.FRAME_END) // 2]


def render_previews():
    """Render preview images."""
    setup_render_settings()

    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = get_frames_to_render()
    print(f"[script] Rendering {len(frames)} frame(s): {frames}")

    for frame in frames:
        frame = max(Config.FRAME_START, min(frame, Config.FRAME_END))
        scene.frame_set(frame)

        render_path = output_dir / f"render_{frame:04d}.png"
        scene.render.filepath = str(render_path)

        print(f"[script] Rendering frame {frame}...")
        bpy.ops.render.render(write_still=True)
        print(f"[script] Saved: {render_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Water Pouring Into Bowl")
    print("=" * 60)

    clear_scene()
    setup_scene()

    # Create simulation elements
    domain = create_domain()
    bowl = create_bowl()
    stream = create_pour_stream()
    animate_pour_stream(stream)

    # Materials
    add_water_material(domain)

    # Camera and lighting
    setup_camera_and_lighting()

    if Config.BAKE:
        bake_simulation(domain)
    else:
        print("[script] Skipping bake (--bake 0)")

    if Config.RENDER:
        render_previews()
    else:
        print("[script] Skipping render (--render 0)")

    # Save blend file
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_path = output_dir / "water_pouring_bowl.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {blend_path}")


if __name__ == "__main__":
    main()
