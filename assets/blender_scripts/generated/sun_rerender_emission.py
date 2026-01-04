"""
Sun Post-Bake Rerender - Pure Emission Color (No Temperature/Blackbody)
Use direct emission color instead of temperature-driven blackbody.
"""
import bpy
import sys


class Config:
    # === ORANGE emission color directly ===
    EMISSION_COLOR_R = 1.0
    EMISSION_COLOR_G = 0.4
    EMISSION_COLOR_B = 0.1

    # Emission strength - START LOW
    EMISSION_STRENGTH = 0.5

    # Scattering color
    COLOR_R = 1.0
    COLOR_G = 0.3
    COLOR_B = 0.05

    DENSITY = 5.0
    ANISOTROPY = 0.0

    # DISABLE blackbody - it's the problem!
    BLACKBODY_INTENSITY = 0.0

    # === Color Management ===
    VIEW_TRANSFORM = "Filmic"
    LOOK = "High Contrast"
    EXPOSURE = -2.0
    GAMMA = 1.0

    # === Render ===
    FRAME = 15
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_ground_truth_v1"
    OUTPUT_NAME = "test_emission_only.png"
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
        "--emission": ("EMISSION_STRENGTH", float),
        "--exposure": ("EXPOSURE", float),
        "--emit_r": ("EMISSION_COLOR_R", float),
        "--emit_g": ("EMISSION_COLOR_G", float),
        "--emit_b": ("EMISSION_COLOR_B", float),
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
        mat = bpy.data.materials.new(name="SunMaterial_Emission")
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

    # Configure - use EMISSION COLOR, not temperature
    volume.inputs['Color'].default_value = (Config.COLOR_R, Config.COLOR_G, Config.COLOR_B, 1.0)
    volume.inputs['Density'].default_value = Config.DENSITY
    volume.inputs['Density Attribute'].default_value = "density"
    volume.inputs['Anisotropy'].default_value = Config.ANISOTROPY

    # DIRECT EMISSION - orange color!
    volume.inputs['Emission Strength'].default_value = Config.EMISSION_STRENGTH
    volume.inputs['Emission Color'].default_value = (
        Config.EMISSION_COLOR_R, Config.EMISSION_COLOR_G, Config.EMISSION_COLOR_B, 1.0
    )

    # DISABLE blackbody (was causing blowout)
    volume.inputs['Blackbody Intensity'].default_value = 0.0

    print(f"[rerender] === Emission-Only Material ===")
    print(f"  Emission Color: ({Config.EMISSION_COLOR_R:.2f}, {Config.EMISSION_COLOR_G:.2f}, {Config.EMISSION_COLOR_B:.2f})")
    print(f"  Emission Strength: {Config.EMISSION_STRENGTH}")
    print(f"  Blackbody: DISABLED")
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

    print(f"[rerender] === Color Management ===")
    print(f"  View: {Config.VIEW_TRANSFORM}, Look: {Config.LOOK}")
    print(f"  Exposure: {Config.EXPOSURE}")


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
    print("Sun Rerender - Pure Emission Color (No Blackbody)")
    print("=" * 60)

    parse_args()
    update_material()
    setup_render()
    render_frame()
    print("[rerender] Done!")


if __name__ == "__main__":
    main()
