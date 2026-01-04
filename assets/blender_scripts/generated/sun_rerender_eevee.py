"""
Sun Post-Bake Rerender Script - EEVEE Version
Uses EEVEE engine to match viewport rendered preview.

Key discovery: Viewport "Rendered" mode uses EEVEE, not Cycles!
That's why F12 (Cycles) looks completely different.
"""
import bpy
import sys


class Config:
    # === Principled Volume Settings (from user's working viewport) ===
    COLOR_R = 1.0
    COLOR_G = 0.11
    COLOR_B = 0.0

    DENSITY = 17.3
    ANISOTROPY = -1.0

    BLACKBODY_INTENSITY = 0.77
    TEMPERATURE = 717.5
    TEMP_ATTRIBUTE = "temperature"

    EMISSION_STRENGTH = 0.0

    ABSORPTION_R = 0.34
    ABSORPTION_G = 0.34
    ABSORPTION_B = 0.34

    # === Color Management ===
    VIEW_TRANSFORM = "Standard"
    EXPOSURE = 1.0
    GAMMA = 0.593

    # === Render Settings ===
    FRAME = 15
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_ground_truth_v1"
    OUTPUT_NAME = "rerender_eevee.png"
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
        "--blackbody": ("BLACKBODY_INTENSITY", float),
        "--temperature": ("TEMPERATURE", float),
        "--gamma": ("GAMMA", float),
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
    """Update volumetric material."""
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

    if not domain.data.materials:
        mat = bpy.data.materials.new(name="SunMaterial_EEVEE")
        domain.data.materials.append(mat)
    else:
        mat = domain.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # Configure
    volume.inputs['Color'].default_value = (Config.COLOR_R, Config.COLOR_G, Config.COLOR_B, 1.0)
    volume.inputs['Density'].default_value = Config.DENSITY
    volume.inputs['Density Attribute'].default_value = "density"
    volume.inputs['Anisotropy'].default_value = Config.ANISOTROPY
    volume.inputs['Absorption Color'].default_value = (Config.ABSORPTION_R, Config.ABSORPTION_G, Config.ABSORPTION_B, 1.0)
    volume.inputs['Emission Strength'].default_value = Config.EMISSION_STRENGTH
    volume.inputs['Blackbody Intensity'].default_value = Config.BLACKBODY_INTENSITY
    volume.inputs['Temperature'].default_value = Config.TEMPERATURE
    volume.inputs['Temperature Attribute'].default_value = Config.TEMP_ATTRIBUTE

    print(f"[rerender] Material configured for EEVEE")
    return True


def setup_eevee():
    """Configure EEVEE render settings for volumes."""
    scene = bpy.context.scene
    render = scene.render

    # Switch to EEVEE!
    render.engine = 'BLENDER_EEVEE'

    render.resolution_x = Config.RESOLUTION
    render.resolution_y = Config.RESOLUTION
    render.resolution_percentage = 100
    render.film_transparent = True

    # EEVEE settings - use try/except for API compatibility
    eevee = scene.eevee

    # Try to set available settings (API varies between Blender versions)
    try:
        eevee.taa_render_samples = Config.SAMPLES
    except AttributeError:
        pass

    # Volume settings for EEVEE (try common attributes)
    volumetric_attrs = {
        'volumetric_start': 0.1,
        'volumetric_end': 100.0,
        'volumetric_tile_size': '4',
        'volumetric_samples': 128,
        'use_volumetric_lights': True,
        'use_volumetric_shadows': True,
    }
    for attr, val in volumetric_attrs.items():
        try:
            setattr(eevee, attr, val)
        except (AttributeError, TypeError):
            print(f"[warning] Could not set eevee.{attr}")

    print(f"[rerender] === EEVEE Settings ===")
    print(f"  Engine: BLENDER_EEVEE")


def setup_color_management():
    """Configure color management."""
    scene = bpy.context.scene
    view = scene.view_settings

    view.view_transform = Config.VIEW_TRANSFORM
    view.look = 'None'
    view.exposure = Config.EXPOSURE
    view.gamma = Config.GAMMA

    print(f"[rerender] === Color Management ===")
    print(f"  View Transform: {Config.VIEW_TRANSFORM}")
    print(f"  Gamma: {Config.GAMMA}")


def render_frame():
    """Render the frame."""
    import os

    scene = bpy.context.scene
    scene.frame_set(Config.FRAME)

    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_NAME)
    scene.render.filepath = output_path

    print(f"[rerender] Rendering frame {Config.FRAME} with EEVEE...")

    bpy.ops.render.render(write_still=True)

    print(f"[rerender] Saved: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Sun Post-Bake Rerender - EEVEE (Viewport Match)")
    print("=" * 60)

    parse_args()
    update_material()
    setup_eevee()
    setup_color_management()
    render_frame()

    print("[rerender] Done!")


if __name__ == "__main__":
    main()
