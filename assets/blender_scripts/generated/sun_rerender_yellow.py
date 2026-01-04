"""
Sun Post-Bake Rerender - Yellow Solar Color
Shift scattering from orange toward yellow to match real sun.
Real sun color temperature is ~5778K (more white/yellow than orange).
"""
import bpy
import sys


class Config:
    # === Yellow-shifted scattering color (like real sun) ===
    # More yellow, less orange for realistic solar appearance
    COLOR_R = 1.0
    COLOR_G = 0.7    # Increased from 0.11 to 0.7 for yellow
    COLOR_B = 0.4    # Add some blue for white-hot center

    DENSITY = 10.0
    ANISOTROPY = -0.5

    BLACKBODY_INTENSITY = 0.01
    TEMPERATURE = 717.5

    EMISSION_STRENGTH = 0.0

    # === Color Management ===
    VIEW_TRANSFORM = "Standard"
    LOOK = "None"
    EXPOSURE = 1.0
    GAMMA = 0.593

    # === Render ===
    FRAME = 15
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_ground_truth_v1"
    OUTPUT_NAME = "test_yellow_sun.png"
    SAMPLES = 64
    RESOLUTION = 512


def parse_args():
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = []

    arg_map = {
        "--output": ("OUTPUT_NAME", str),
        "--frame": ("FRAME", int),
        "--density": ("DENSITY", float),
        "--blackbody": ("BLACKBODY_INTENSITY", float),
        "--exposure": ("EXPOSURE", float),
        "--gamma": ("GAMMA", float),
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
        mat = bpy.data.materials.new(name="SunMaterial_Yellow")
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

    # Configure with yellow-shifted color
    volume.inputs['Color'].default_value = (Config.COLOR_R, Config.COLOR_G, Config.COLOR_B, 1.0)
    volume.inputs['Density'].default_value = Config.DENSITY
    volume.inputs['Density Attribute'].default_value = "density"
    volume.inputs['Anisotropy'].default_value = Config.ANISOTROPY

    volume.inputs['Emission Strength'].default_value = Config.EMISSION_STRENGTH
    volume.inputs['Blackbody Intensity'].default_value = Config.BLACKBODY_INTENSITY
    volume.inputs['Temperature'].default_value = Config.TEMPERATURE
    volume.inputs['Temperature Attribute'].default_value = "temperature"

    print(f"[rerender] === Yellow Solar Color ===")
    print(f"  Scattering Color: ({Config.COLOR_R:.2f}, {Config.COLOR_G:.2f}, {Config.COLOR_B:.2f})")
    print(f"  Blackbody Intensity: {Config.BLACKBODY_INTENSITY}")
    print(f"  Density: {Config.DENSITY}")

    return True


def setup_render():
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

    # Color management
    view = scene.view_settings
    view.view_transform = Config.VIEW_TRANSFORM
    view.look = Config.LOOK
    view.exposure = Config.EXPOSURE
    view.gamma = Config.GAMMA

    print(f"[rerender] View: {Config.VIEW_TRANSFORM}, Gamma: {Config.GAMMA}")


def render_frame():
    import os
    scene = bpy.context.scene
    scene.frame_set(Config.FRAME)

    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_NAME)
    scene.render.filepath = output_path

    print(f"[rerender] Rendering frame {Config.FRAME}...")
    bpy.ops.render.render(write_still=True)
    print(f"[rerender] Saved: {output_path}")


def main():
    print("=" * 60)
    print("Sun Rerender - Yellow Solar Color")
    print("=" * 60)

    parse_args()
    update_material()
    setup_render()
    render_frame()
    print("[rerender] Done!")


if __name__ == "__main__":
    main()
