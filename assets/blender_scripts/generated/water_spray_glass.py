"""
Water Spray on Glass - Splashing Simulation

Creates a water spray hitting a glass pane with dramatic splashing effects.
Optimized for 100 frames to reduce bake time.

Features:
- High-velocity spray inflow for dramatic splashing
- Glass pane as collision surface
- Splash-optimized simulation settings
- 100 frame bake (not 250) for faster iteration
"""

import bpy
import math
import os
import shutil
from pathlib import Path


class Config:
    """Simulation configuration."""
    RESOLUTION = 96
    FRAME_START = 1
    FRAME_END = 100
    CACHE_FRAME_START = 1
    CACHE_FRAME_END = 100  # Explicitly set cache to 100 frames
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/water_spray_glass_v1"

    # Domain size (spray needs vertical space for splashing)
    DOMAIN_SIZE = (3.0, 3.0, 3.0)

    # Glass pane dimensions
    GLASS_WIDTH = 1.5
    GLASS_HEIGHT = 2.0
    GLASS_THICKNESS = 0.05

    # Spray settings
    SPRAY_DISTANCE = 1.0  # Distance from glass
    SPRAY_RADIUS = 0.15   # Nozzle radius
    SPRAY_VELOCITY = 8.0  # High velocity for splashing
    SPRAY_ANGLE = 15.0    # Cone spread angle in degrees

    # Simulation quality
    TIMESTEPS_MIN = 2
    TIMESTEPS_MAX = 6
    CFL_CONDITION = 3.0


def clean_scene():
    """Remove all objects and start fresh."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_output_directory():
    """Ensure output directory exists and is clean."""
    output_path = Path(Config.OUTPUT_DIR)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    return str(output_path)


def create_domain():
    """Create fluid domain with splash-optimized settings."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "FluidDomain"
    domain.scale = (
        Config.DOMAIN_SIZE[0],
        Config.DOMAIN_SIZE[1],
        Config.DOMAIN_SIZE[2]
    )
    bpy.ops.object.transform_apply(scale=True)

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'LIQUID'
    settings.resolution_max = Config.RESOLUTION

    # Cache settings - explicitly 100 frames
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_type = 'ALL'
    settings.cache_data_format = 'OPENVDB'
    settings.cache_frame_start = Config.CACHE_FRAME_START
    settings.cache_frame_end = Config.CACHE_FRAME_END

    # Splash-optimized timing
    settings.timesteps_min = Config.TIMESTEPS_MIN
    settings.timesteps_max = Config.TIMESTEPS_MAX
    settings.cfl_condition = Config.CFL_CONDITION

    # Enable mesh for rendering
    settings.use_mesh = True
    settings.mesh_concave_upper = 3.5
    settings.mesh_concave_lower = 1.5
    settings.mesh_smoothen_pos = 2
    settings.mesh_smoothen_neg = 2

    # Splash particles (droplets)
    settings.use_spray_particles = True
    settings.use_foam_particles = True
    settings.use_bubble_particles = False

    print(f"Domain created: {Config.RESOLUTION} resolution, {Config.CACHE_FRAME_END} frames")
    return domain


def create_glass_pane():
    """Create a vertical glass pane as collision target."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    glass = bpy.context.active_object
    glass.name = "GlassPane"

    # Scale to pane dimensions
    glass.scale = (
        Config.GLASS_THICKNESS,
        Config.GLASS_WIDTH,
        Config.GLASS_HEIGHT
    )
    # Position slightly off-center, vertical
    glass.location = (0, 0, 0)

    bpy.ops.object.transform_apply(scale=True)

    # Add fluid modifier as effector
    bpy.ops.object.modifier_add(type='FLUID')
    glass.modifiers["Fluid"].fluid_type = 'EFFECTOR'

    effector = glass.modifiers["Fluid"].effector_settings
    effector.effector_type = 'COLLISION'
    effector.surface_distance = 0.1
    effector.subframes = 5

    # Add glass material
    add_glass_material(glass)

    print("Glass pane created as collision effector")
    return glass


def add_glass_material(obj):
    """Add Glass BSDF material to object."""
    mat = bpy.data.materials.new(name="GlassMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Glass BSDF
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 0)
    glass.inputs['Color'].default_value = (0.95, 0.98, 1.0, 1.0)
    glass.inputs['Roughness'].default_value = 0.0
    glass.inputs['IOR'].default_value = 1.52

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)


def create_spray_nozzle():
    """Create spray inflow with high velocity and cone spread."""
    # Create a cone-shaped nozzle for spray effect
    bpy.ops.mesh.primitive_cone_add(
        radius1=Config.SPRAY_RADIUS,
        radius2=Config.SPRAY_RADIUS * 0.3,
        depth=0.3,
        location=(
            -Config.SPRAY_DISTANCE,  # In front of glass
            0,
            0.3  # Slightly above center
        )
    )
    nozzle = bpy.context.active_object
    nozzle.name = "SprayNozzle"

    # Rotate to point at glass (90 degrees around Y)
    nozzle.rotation_euler = (0, math.radians(90), 0)
    bpy.ops.object.transform_apply(rotation=True)

    # Add fluid modifier as inflow
    bpy.ops.object.modifier_add(type='FLUID')
    nozzle.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = nozzle.modifiers["Fluid"].flow_settings
    flow.flow_type = 'LIQUID'
    flow.flow_behavior = 'INFLOW'
    flow.use_inflow = True

    # High velocity for splashing
    flow.velocity_factor = 1.0
    flow.velocity_normal = Config.SPRAY_VELOCITY

    # Add some turbulence for spray spread
    flow.use_initial_velocity = True

    # Animate spray (continuous for most of simulation)
    animate_spray(nozzle)

    # Add water material for particles
    add_water_material(nozzle)

    print(f"Spray nozzle created: velocity {Config.SPRAY_VELOCITY}")
    return nozzle


def animate_spray(nozzle):
    """Animate spray on/off for interesting dynamics."""
    scene = bpy.context.scene
    flow = nozzle.modifiers["Fluid"].flow_settings

    # Start spray at frame 1
    scene.frame_set(1)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow')

    # Continuous spray until frame 70
    scene.frame_set(70)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow')

    # Stop spray at frame 71 (let splashing settle)
    scene.frame_set(71)
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow')

    print("Spray animated: frames 1-70 active, 71-100 settling")


def add_water_material(obj):
    """Add water material."""
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Glass BSDF for water
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 0)
    glass.inputs['Color'].default_value = (0.8, 0.9, 1.0, 1.0)
    glass.inputs['Roughness'].default_value = 0.0
    glass.inputs['IOR'].default_value = 1.33  # Water IOR

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)


def create_collection_basin():
    """Create a basin at bottom to catch splashed water."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -1.3))
    basin = bpy.context.active_object
    basin.name = "CollectionBasin"

    basin.scale = (2.0, 2.0, 0.1)
    bpy.ops.object.transform_apply(scale=True)

    # Add fluid modifier as effector
    bpy.ops.object.modifier_add(type='FLUID')
    basin.modifiers["Fluid"].fluid_type = 'EFFECTOR'

    effector = basin.modifiers["Fluid"].effector_settings
    effector.effector_type = 'COLLISION'
    effector.surface_distance = 0.1
    effector.subframes = 3

    print("Collection basin created")
    return basin


def setup_lighting():
    """Create lighting for the scene."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 500
    key.data.size = 2
    key.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, 2, 2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 200
    fill.data.size = 3

    # Backlight for rim effect on water
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 1))
    back = bpy.context.active_object
    back.name = "BackLight"
    back.data.energy = 300
    back.data.size = 2


def setup_camera():
    """Position camera for spray visualization."""
    bpy.ops.object.camera_add(
        location=(3, -3, 1.5),
        rotation=(math.radians(70), 0, math.radians(45))
    )
    camera = bpy.context.active_object
    camera.name = "MainCamera"
    bpy.context.scene.camera = camera

    camera.data.lens = 35


def setup_render_settings():
    """Configure render settings for quality output."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 50

    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END


def bake_simulation(domain):
    """Bake the fluid simulation."""
    print("\n" + "=" * 60)
    print("BAKING FLUID SIMULATION")
    print(f"Frames: {Config.CACHE_FRAME_START} to {Config.CACHE_FRAME_END} (100 frames)")
    print(f"Resolution: {Config.RESOLUTION}")
    print("=" * 60 + "\n")

    # Ensure domain is selected
    bpy.ops.object.select_all(action='DESELECT')
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain

    # Bake all
    bpy.ops.fluid.bake_all()

    print("\nBake complete!")


def render_test_frames(output_dir):
    """Render key frames to verify simulation."""
    test_frames = [20, 40, 60, 80, 95]
    scene = bpy.context.scene

    print("\nRendering test frames...")

    for frame in test_frames:
        scene.frame_set(frame)
        output_path = os.path.join(output_dir, f"render_{frame:04d}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"  Rendered frame {frame}")

    print("Test renders complete!")


def save_blend_file(output_dir):
    """Save the blend file for manual inspection."""
    blend_path = os.path.join(output_dir, "water_spray_glass.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved: {blend_path}")


def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("WATER SPRAY ON GLASS SIMULATION")
    print("100 frames, high-velocity splashing")
    print("=" * 60 + "\n")

    # Setup
    clean_scene()
    output_dir = setup_output_directory()

    # Create scene elements
    domain = create_domain()
    glass = create_glass_pane()
    nozzle = create_spray_nozzle()
    basin = create_collection_basin()

    # Lighting and camera
    setup_lighting()
    setup_camera()
    setup_render_settings()

    # Bake simulation
    bake_simulation(domain)

    # Render test frames
    render_test_frames(output_dir)

    # Save blend file
    save_blend_file(output_dir)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"Output: {output_dir}")
    print(f"VDB files: fluid_data_*.vdb (100 frames)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
