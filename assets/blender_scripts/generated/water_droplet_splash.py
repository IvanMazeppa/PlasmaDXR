#!/usr/bin/env python3
"""
Water Droplet Splash Simulation - Blender 5.0+

A slow-motion water droplet hitting a horizontal plane, showing splash physics.
Optimized for detailed splash with spray particles.

Usage:
    blender --background --python water_droplet_splash.py -- [options]

Options:
    --resolution INT    Simulation resolution (default: 64)
    --frame_start INT   Start frame (default: 1)
    --frame_end INT     End frame (default: 60)
    --output_dir PATH   Output directory for cache files
    --bake BOOL         Run bake (1) or skip (0)
    --render BOOL       Render preview images (1) or skip (0)
"""

import bpy
import sys
import math
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

class Config:
    RESOLUTION = 64  # Higher = more detail
    FRAME_START = 1
    FRAME_END = 60
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/water_droplet_splash"
    BAKE = True
    RENDER = True
    RENDER_FRAMES = "mid"

    # Render settings
    RENDER_RESOLUTION_X = 512
    RENDER_RESOLUTION_Y = 512
    RENDER_SAMPLES = 128  # Higher for glass/water caustics

    # Domain size (tall to capture splash height)
    DOMAIN_WIDTH = 4.0
    DOMAIN_HEIGHT = 6.0

    # Water droplet settings
    DROPLET_RADIUS = 0.15
    DROPLET_HEIGHT = 2.5  # Starting height above water surface
    DROPLET_VELOCITY = -8.0  # Downward velocity


def parse_args():
    """Parse command line arguments after '--'."""
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
    """Configure scene settings for slow-motion water simulation."""
    scene = bpy.context.scene
    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END

    # Cycles for realistic water rendering
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # Enable caustics for water (important for realism)
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True


# =============================================================================
# Domain Creation
# =============================================================================

def create_domain():
    """Create fluid domain for liquid simulation."""
    # Tall domain to capture splash height
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, Config.DOMAIN_HEIGHT / 2 - 1)
    )
    domain = bpy.context.active_object
    domain.name = "LiquidDomain"

    # Scale to proper dimensions
    domain.scale = (Config.DOMAIN_WIDTH, Config.DOMAIN_WIDTH, Config.DOMAIN_HEIGHT)
    bpy.ops.object.transform_apply(scale=True)

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'LIQUID'

    # Resolution
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = False  # Keep full domain for splash

    # FLIP ratio - lower for detailed splashes (0.7-0.9)
    settings.flip_ratio = 0.80

    # Timesteps - more steps for accurate splash physics
    settings.timesteps_max = 8
    settings.timesteps_min = 2

    # Cache settings
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'

    # Enable mesh generation for liquid surface
    settings.use_mesh = True
    settings.mesh_scale = 1  # Int type required
    settings.mesh_particle_radius = 1.5  # Slightly larger for smoother mesh

    # Enable secondary particles for spray!
    settings.use_spray_particles = True
    settings.use_foam_particles = True
    settings.use_bubble_particles = True

    # Particle settings for nice splash spray
    settings.sndparticle_potential_max_wavecrest = 5.0
    settings.sndparticle_potential_min_wavecrest = 0.5
    settings.sndparticle_potential_max_energy = 5.0
    settings.sndparticle_potential_min_energy = 0.2
    settings.sndparticle_sampling_wavecrest = 50
    settings.sndparticle_sampling_trappedair = 50
    settings.sndparticle_life_max = 80.0
    settings.sndparticle_life_min = 20.0

    # Gravity (standard Earth gravity)
    settings.gravity = (0, 0, -9.81)

    # Domain displays as wire in viewport but renders as liquid mesh
    domain.display_type = 'WIRE'
    # domain.hide_render = False  # Domain IS the liquid mesh - must be visible!

    return domain


def create_water_surface():
    """Create initial water body (shallow pool)."""
    # Flat plane of water at the bottom
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, -0.25)
    )
    water = bpy.context.active_object
    water.name = "WaterSurface"

    # Scale to pool shape (wide, shallow)
    water.scale = (Config.DOMAIN_WIDTH * 0.9, Config.DOMAIN_WIDTH * 0.9, 0.5)
    bpy.ops.object.transform_apply(scale=True)

    # Make it a fluid source
    bpy.ops.object.modifier_add(type='FLUID')
    water.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = water.modifiers["Fluid"].flow_settings
    flow.flow_type = 'LIQUID'
    flow.flow_behavior = 'GEOMETRY'  # Initial geometry fills with liquid
    flow.use_initial_velocity = False

    # Hide the geometry mesh
    water.hide_render = True

    return water


def create_water_droplet():
    """Create falling water droplet."""
    # Sphere for the droplet
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=Config.DROPLET_RADIUS,
        location=(0, 0, Config.DROPLET_HEIGHT),
        segments=16,
        ring_count=8
    )
    droplet = bpy.context.active_object
    droplet.name = "WaterDroplet"

    # Make it a fluid inflow
    bpy.ops.object.modifier_add(type='FLUID')
    droplet.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = droplet.modifiers["Fluid"].flow_settings
    flow.flow_type = 'LIQUID'
    flow.flow_behavior = 'GEOMETRY'  # Single geometry emission
    flow.use_initial_velocity = True
    flow.velocity_normal = 0
    flow.velocity_coord = (0, 0, Config.DROPLET_VELOCITY)  # Downward

    # Hide geometry
    droplet.hide_render = True

    return droplet


def create_collision_plane():
    """Create invisible collision plane at bottom (optional, domain floor works)."""
    # The domain floor acts as collision by default
    # But we can add a visible plane for rendering
    bpy.ops.mesh.primitive_plane_add(
        size=Config.DOMAIN_WIDTH * 2,
        location=(0, 0, -0.5)
    )
    floor = bpy.context.active_object
    floor.name = "Floor"

    # Add floor material (gray matte)
    mat = bpy.data.materials.new(name="FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.3, 0.3, 0.35, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
        bsdf.inputs['Metallic'].default_value = 0.0

    floor.data.materials.append(mat)

    return floor


def add_water_material(domain):
    """Add realistic water material to the domain's liquid mesh."""
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Glass BSDF for water (IOR = 1.33)
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 0)
    glass.inputs['Color'].default_value = (0.8, 0.9, 1.0, 1.0)  # Slight blue tint
    glass.inputs['Roughness'].default_value = 0.0  # Perfectly smooth water
    glass.inputs['IOR'].default_value = 1.33  # Water IOR

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    domain.data.materials.append(mat)
    print("[script] Water material applied (Glass BSDF, IOR 1.33)")


# =============================================================================
# Camera and Lighting
# =============================================================================

def setup_camera_and_lighting():
    """Setup camera and lighting for dramatic slow-motion splash shot."""
    # Camera - close-up angle to see splash detail
    bpy.ops.object.camera_add(
        location=(3.5, -3.5, 1.5)
    )
    cam = bpy.context.active_object
    cam.name = "Camera"

    # Point at splash area
    cam.rotation_euler = (math.radians(75), 0, math.radians(45))
    bpy.context.scene.camera = cam

    # Adjust focal length for close-up
    cam.data.lens = 50

    # Key light (main light from above-right)
    bpy.ops.object.light_add(type='AREA', location=(3, -2, 4))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 500
    key.data.size = 2
    key.rotation_euler = (math.radians(45), 0, math.radians(30))

    # Fill light (softer, from left)
    bpy.ops.object.light_add(type='AREA', location=(-3, -1, 2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 200
    fill.data.size = 3

    # Back light for rim/splash highlights
    bpy.ops.object.light_add(type='SPOT', location=(0, 3, 3))
    back = bpy.context.active_object
    back.name = "BackLight"
    back.data.energy = 800
    back.data.spot_size = math.radians(60)
    back.rotation_euler = (math.radians(120), 0, math.radians(180))

    # World background (gradient for depth)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.05, 0.08, 0.15, 1.0)  # Dark blue
    bg.inputs['Strength'].default_value = 0.5

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (200, 0)

    world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])


# =============================================================================
# Baking
# =============================================================================

def bake_simulation(domain):
    """Bake the liquid simulation."""
    print(f"[script] Baking liquid simulation: frames {Config.FRAME_START}-{Config.FRAME_END}")
    print(f"[script] Resolution: {Config.RESOLUTION}")
    print(f"[script] Output: {Config.OUTPUT_DIR}")
    print(f"[script] Features: Mesh + Spray + Foam + Bubbles")

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()

    print("[script] Bake complete!")


# =============================================================================
# Rendering
# =============================================================================

def setup_render_settings():
    """Configure render settings for water simulation."""
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES

    # Important for water: enable caustics
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True

    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Transparent background for compositing
    scene.render.film_transparent = False  # Keep background visible

    print(f"[script] Render: {Config.RENDER_RESOLUTION_X}x{Config.RENDER_RESOLUTION_Y}, {Config.RENDER_SAMPLES} samples")


def get_frames_to_render():
    """Determine which frames to render."""
    if Config.RENDER_FRAMES == "mid":
        mid = (Config.FRAME_START + Config.FRAME_END) // 2
        return [mid]
    elif Config.RENDER_FRAMES == "all":
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 5)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    elif Config.RENDER_FRAMES == "splash":
        # Key splash moments
        return [15, 20, 25, 30, 40]
    else:
        try:
            return [int(f.strip()) for f in Config.RENDER_FRAMES.split(",")]
        except ValueError:
            mid = (Config.FRAME_START + Config.FRAME_END) // 2
            return [mid]


def render_previews():
    """Render preview images."""
    setup_render_settings()

    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = get_frames_to_render()
    rendered_files = []

    print(f"[script] Rendering {len(frames)} frame(s): {frames}")

    for frame in frames:
        frame = max(Config.FRAME_START, min(frame, Config.FRAME_END))
        scene.frame_set(frame)

        render_path = output_dir / f"render_{frame:04d}.png"
        scene.render.filepath = str(render_path)

        print(f"[script] Rendering frame {frame}...")
        bpy.ops.render.render(write_still=True)

        rendered_files.append(str(render_path))
        print(f"[script] Saved: {render_path}")

    return rendered_files


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Water Droplet Splash Simulation")
    print("=" * 60)

    clear_scene()
    setup_scene()

    # Create simulation elements
    domain = create_domain()
    water = create_water_surface()
    droplet = create_water_droplet()
    floor = create_collision_plane()

    # Add materials
    add_water_material(domain)

    # Setup camera and lights
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
    blend_path = output_dir / "water_droplet_splash.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {blend_path}")


if __name__ == "__main__":
    main()
