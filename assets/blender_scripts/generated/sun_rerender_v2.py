"""
Sun Post-Bake Rerender Script v2
Based on user's working viewport settings extracted from user_modified.blend

Key insight: Use Principled Volume's BUILT-IN Blackbody system with Temperature Attribute.
No complex node trees needed!
"""
import bpy
import sys


class Config:
    # === Principled Volume Settings (from user's working config) ===
    # Scattering color - BRIGHT ORANGE is key for warm_ratio!
    COLOR_R = 1.0
    COLOR_G = 0.11
    COLOR_B = 0.0

    # Density
    DENSITY = 17.3

    # Anisotropy: -1 = full backscatter (keeps light inside volume)
    ANISOTROPY = -1.0

    # Blackbody (built-in temperature-based emission)
    BLACKBODY_INTENSITY = 0.77  # Enables temperature-based fire color
    TEMPERATURE = 717.5         # Base temperature in Kelvin
    TEMP_ATTRIBUTE = "temperature"  # Read from simulation

    # Manual emission (disabled - let blackbody handle it)
    EMISSION_STRENGTH = 0.0

    # Absorption color
    ABSORPTION_R = 0.34
    ABSORPTION_G = 0.34
    ABSORPTION_B = 0.34

    # === Color Management (critical for not blowing out!) ===
    VIEW_TRANSFORM = "Standard"
    EXPOSURE = 1.0
    GAMMA = 0.593  # Lower gamma darkens the image

    # === Render Settings ===
    FRAME = 15
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_ground_truth_v1"
    OUTPUT_NAME = "rerender_v8_user_settings.png"
    SAMPLES = 64
    RESOLUTION = 512


def parse_args():
    """Parse command line arguments after '--'."""
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = []

    arg_map = {
        "--output": ("OUTPUT_NAME", str),
        "--frame": ("FRAME", int),
        "--samples": ("SAMPLES", int),
        "--density": ("DENSITY", float),
        "--anisotropy": ("ANISOTROPY", float),
        "--blackbody": ("BLACKBODY_INTENSITY", float),
        "--temperature": ("TEMPERATURE", float),
        "--gamma": ("GAMMA", float),
        "--exposure": ("EXPOSURE", float),
        "--color_r": ("COLOR_R", float),
        "--color_g": ("COLOR_G", float),
        "--color_b": ("COLOR_B", float),
    }

    i = 0
    while i < len(args):
        if args[i] in arg_map and i + 1 < len(args):
            attr, type_fn = arg_map[args[i]]
            setattr(Config, attr, type_fn(args[i + 1]))
            i += 2
        else:
            i += 1


def update_material():
    """Update volumetric material using user's working settings.

    Key: Simple Principled Volume with built-in Blackbody system.
    No extra nodes - just configure the shader directly!
    """
    # Find domain object
    domain = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                    domain = obj
                    break
        if domain:
            break

    if not domain:
        print("[ERROR] No fluid domain found!")
        return False

    # Get or create material
    if not domain.data.materials:
        mat = bpy.data.materials.new(name="SunMaterial_v2")
        domain.data.materials.append(mat)
    else:
        mat = domain.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear and rebuild simple node tree
    nodes.clear()

    # === Simple setup: just Principled Volume → Output ===
    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.name = "PrincipledVolume"

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Connect
    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # === Configure Principled Volume with user's working settings ===

    # Scattering color - ORANGE for warm sun
    volume.inputs['Color'].default_value = (
        Config.COLOR_R, Config.COLOR_G, Config.COLOR_B, 1.0
    )

    # Density
    volume.inputs['Density'].default_value = Config.DENSITY
    volume.inputs['Density Attribute'].default_value = "density"

    # Anisotropy (-1 = backscatter)
    volume.inputs['Anisotropy'].default_value = Config.ANISOTROPY

    # Absorption color
    volume.inputs['Absorption Color'].default_value = (
        Config.ABSORPTION_R, Config.ABSORPTION_G, Config.ABSORPTION_B, 1.0
    )

    # Manual emission disabled (blackbody handles it)
    volume.inputs['Emission Strength'].default_value = Config.EMISSION_STRENGTH

    # BLACKBODY - the key to temperature-based fire colors!
    volume.inputs['Blackbody Intensity'].default_value = Config.BLACKBODY_INTENSITY
    volume.inputs['Temperature'].default_value = Config.TEMPERATURE
    volume.inputs['Temperature Attribute'].default_value = Config.TEMP_ATTRIBUTE

    print(f"[rerender] === Principled Volume Settings (User Config) ===")
    print(f"  Color (scattering): ({Config.COLOR_R:.2f}, {Config.COLOR_G:.2f}, {Config.COLOR_B:.2f})")
    print(f"  Density: {Config.DENSITY}")
    print(f"  Anisotropy: {Config.ANISOTROPY}")
    print(f"  Blackbody Intensity: {Config.BLACKBODY_INTENSITY}")
    print(f"  Temperature: {Config.TEMPERATURE}K")
    print(f"  Temperature Attribute: {Config.TEMP_ATTRIBUTE}")

    return True


def setup_color_management():
    """Configure color management to prevent blown-out renders."""
    scene = bpy.context.scene
    view = scene.view_settings

    view.view_transform = Config.VIEW_TRANSFORM
    view.look = 'None'
    view.exposure = Config.EXPOSURE
    view.gamma = Config.GAMMA

    print(f"[rerender] === Color Management ===")
    print(f"  View Transform: {Config.VIEW_TRANSFORM}")
    print(f"  Exposure: {Config.EXPOSURE}")
    print(f"  Gamma: {Config.GAMMA} (lower = darker)")


def setup_render():
    """Configure render settings."""
    scene = bpy.context.scene
    render = scene.render
    cycles = scene.cycles

    render.engine = 'CYCLES'
    render.resolution_x = Config.RESOLUTION
    render.resolution_y = Config.RESOLUTION
    render.resolution_percentage = 100
    render.film_transparent = True

    cycles.samples = Config.SAMPLES
    cycles.use_denoising = False

    # Volume settings
    cycles.volume_step_rate = 1.0
    cycles.volume_max_steps = 1024


def render_frame():
    """Render the specified frame."""
    import os

    scene = bpy.context.scene
    scene.frame_set(Config.FRAME)

    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_NAME)
    scene.render.filepath = output_path

    print(f"[rerender] Rendering frame {Config.FRAME} @ {Config.RESOLUTION}x{Config.RESOLUTION}, {Config.SAMPLES} samples...")

    bpy.ops.render.render(write_still=True)

    print(f"[rerender] Saved: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Sun Post-Bake Rerender v2 (User's Working Settings)")
    print("=" * 60)

    parse_args()
    update_material()
    setup_color_management()
    setup_render()
    render_frame()

    print("[rerender] Done!")


if __name__ == "__main__":
    main()
