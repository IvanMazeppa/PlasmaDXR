#!/usr/bin/env python3
"""
Post-bake material iteration script - FAST (~20-30 sec per render)

This script loads an existing baked simulation and re-renders with
different material settings WITHOUT re-baking.

Usage:
    blender --background sun_ground_truth_v1.blend --python sun_rerender_iteration.py -- [options]

Options:
    --blackbody FLOAT     Blackbody intensity (default: 0.3)
    --temperature FLOAT   Temperature in Kelvin (default: 1500)
    --density FLOAT       Volume density (default: 20.0)
    --frame INT           Frame to render (default: 15)
    --output PATH         Output filename (default: rerender_v{iteration}.png)
"""

import bpy
import sys
from pathlib import Path

class Config:
    # === Temperature Remapping (Blender 5.0 Volume Info → Blackbody) ===
    # Volume Info outputs 0-1 (maps to 0-1000K internally)
    # We remap to: output = (input * TEMP_MULTIPLY) + TEMP_OFFSET
    TEMP_MULTIPLY = 3200.0     # Scale factor (3200 gives good sun range)
    TEMP_OFFSET = 800.0        # Base temperature (800K = deep red)
    # Result: 0→800K (deep red), 1→4000K (yellow-white)

    # === Material Properties ===
    BLACKBODY_INTENSITY = 0.0  # Set to 0 - we use Emission Color from Blackbody node instead
    DENSITY = 5.0              # Base density (multiplied by flame field)
    DENSITY_MULT = 15.0        # Flame density multiplier
    ANISOTROPY = 0.3           # Light scattering direction (-1 to 1)
    EMISSION_STRENGTH = 3.0    # Emission multiplier (higher for bright sun)

    # === Scattering Color (dark = less scatter, shows emission better) ===
    COLOR_R = 0.1              # Red channel
    COLOR_G = 0.05             # Green channel
    COLOR_B = 0.02             # Blue channel (very dark to show emission)

    # === Color Ramp for Fire Colors (Flame intensity → Color) ===
    # Position 0.0 = deep red (low flame intensity)
    RAMP_COLOR_0_R = 0.8
    RAMP_COLOR_0_G = 0.1
    RAMP_COLOR_0_B = 0.02
    # Position 0.5 = bright orange (mid flame intensity)
    RAMP_COLOR_MID_R = 1.0
    RAMP_COLOR_MID_G = 0.4
    RAMP_COLOR_MID_B = 0.05
    # Position 1.0 = yellow-white (high flame intensity)
    RAMP_COLOR_1_R = 1.0
    RAMP_COLOR_1_G = 0.8
    RAMP_COLOR_1_B = 0.3

    # === Post-Processing / Color Correction ===
    EXPOSURE = 0.0             # Exposure adjustment (EV)
    GAMMA = 1.0                # Gamma correction

    # === Render Settings ===
    FRAME = 15
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_ground_truth_v1"
    OUTPUT_NAME = "rerender_v6.png"
    SAMPLES = 64
    RESOLUTION = 512

def parse_args():
    """Parse command line arguments for material and post-processing experimentation."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    arg_map = {
        # Temperature remapping (Volume Info → Blackbody)
        "--temp_mult": ("TEMP_MULTIPLY", float),
        "--temp_offset": ("TEMP_OFFSET", float),
        # Material properties
        "--density": ("DENSITY", float),
        "--density_mult": ("DENSITY_MULT", float),
        "--anisotropy": ("ANISOTROPY", float),
        "--emission": ("EMISSION_STRENGTH", float),
        "--blackbody": ("BLACKBODY_INTENSITY", float),
        # Scattering color
        "--color_r": ("COLOR_R", float),
        "--color_g": ("COLOR_G", float),
        "--color_b": ("COLOR_B", float),
        # Color ramp (fire colors)
        "--ramp0_r": ("RAMP_COLOR_0_R", float),
        "--ramp0_g": ("RAMP_COLOR_0_G", float),
        "--ramp0_b": ("RAMP_COLOR_0_B", float),
        "--ramp_mid_r": ("RAMP_COLOR_MID_R", float),
        "--ramp_mid_g": ("RAMP_COLOR_MID_G", float),
        "--ramp_mid_b": ("RAMP_COLOR_MID_B", float),
        "--ramp1_r": ("RAMP_COLOR_1_R", float),
        "--ramp1_g": ("RAMP_COLOR_1_G", float),
        "--ramp1_b": ("RAMP_COLOR_1_B", float),
        # Post-processing
        "--exposure": ("EXPOSURE", float),
        "--gamma": ("GAMMA", float),
        # Render settings
        "--frame": ("FRAME", int),
        "--output": ("OUTPUT_NAME", str),
        "--samples": ("SAMPLES", int),
        "--resolution": ("RESOLUTION", int),
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in arg_map and i + 1 < len(argv):
            attr, converter = arg_map[arg]
            setattr(Config, attr, converter(argv[i + 1]))
            i += 2
        else:
            i += 1

parse_args()


def update_material():
    """Update volumetric material using PROPER Blender 5.0 fire shading.

    Key insight from Blender 5.0 manual:
    - Volume Info → Flame output gives fire intensity (0-1)
    - Color Ramp maps flame intensity to warm orange-red colors
    - This is MORE RELIABLE than Temperature which may not vary much

    Node setup:
    Volume Info (Flame) → Color Ramp (fire colors) → Principled Volume (Emission Color)
    Volume Info (Flame) → Math (Multiply) → Principled Volume (Density)
    """
    # Find the domain object
    domain = bpy.data.objects.get("FluidDomain")
    if not domain:
        print("[rerender] ERROR: FluidDomain not found!")
        return False

    # Get or create material
    if not domain.data.materials:
        mat = bpy.data.materials.new(name="SunFireMaterial")
        domain.data.materials.append(mat)
    else:
        mat = domain.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes for clean rebuild
    nodes.clear()

    # === Create node tree for proper fire/sun shading (Blender 5.0) ===

    # 1. Volume Info node - gets Flame intensity from simulation
    vol_info = nodes.new('ShaderNodeVolumeInfo')
    vol_info.location = (-600, 200)
    vol_info.name = "VolumeInfo"

    # 2. Color Ramp - map flame intensity (0-1) to fire colors
    # 0 = dark red, 0.5 = bright orange, 1 = yellow-white
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.location = (-300, 200)
    color_ramp.name = "FireColorRamp"

    # Configure color ramp for sun/fire colors (orange-red dominated)
    # Access color stops and set them
    ramp = color_ramp.color_ramp
    ramp.interpolation = 'LINEAR'

    # Clear existing stops and create new ones
    # Position 0.0 = deep red (low flame)
    ramp.elements[0].position = 0.0
    ramp.elements[0].color = (Config.RAMP_COLOR_0_R, Config.RAMP_COLOR_0_G, Config.RAMP_COLOR_0_B, 1.0)

    # Position 1.0 = bright orange-yellow (high flame)
    ramp.elements[1].position = 1.0
    ramp.elements[1].color = (Config.RAMP_COLOR_1_R, Config.RAMP_COLOR_1_G, Config.RAMP_COLOR_1_B, 1.0)

    # Add middle stop for orange (optional, for more control)
    mid_stop = ramp.elements.new(0.5)
    mid_stop.color = (Config.RAMP_COLOR_MID_R, Config.RAMP_COLOR_MID_G, Config.RAMP_COLOR_MID_B, 1.0)

    # 3. Principled Volume shader
    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (200, 0)
    volume.name = "PrincipledVolume"

    # Configure Principled Volume
    volume.inputs['Density'].default_value = Config.DENSITY
    volume.inputs['Anisotropy'].default_value = Config.ANISOTROPY
    volume.inputs['Emission Strength'].default_value = Config.EMISSION_STRENGTH
    volume.inputs['Blackbody Intensity'].default_value = 0.0  # Disabled - using Color Ramp instead

    # Scattering color (dark = less scattering, shows emission better)
    volume.inputs['Color'].default_value = (Config.COLOR_R, Config.COLOR_G, Config.COLOR_B, 1.0)

    # 4. Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (500, 0)

    # === Connect nodes ===

    # Volume Info Flame → Color Ramp
    links.new(vol_info.outputs['Flame'], color_ramp.inputs['Fac'])

    # Color Ramp → Principled Volume Emission Color
    if 'Emission Color' in volume.inputs:
        links.new(color_ramp.outputs['Color'], volume.inputs['Emission Color'])
        print("[rerender] Connected Color Ramp → Emission Color")

    # Volume → Output
    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # Also connect Volume Info Flame to density (fire regions are more visible)
    density_mult = nodes.new('ShaderNodeMath')
    density_mult.operation = 'MULTIPLY'
    density_mult.location = (-200, -100)
    density_mult.name = "DensityMult"
    density_mult.inputs[1].default_value = Config.DENSITY_MULT

    links.new(vol_info.outputs['Flame'], density_mult.inputs[0])
    links.new(density_mult.outputs[0], volume.inputs['Density'])

    print(f"[rerender] === Blender 5.0 Fire Shading (Color Ramp) ===")
    print(f"  Volume Info Flame → Color Ramp → Emission Color")
    print(f"  Color Ramp: Red({Config.RAMP_COLOR_0_R:.2f},{Config.RAMP_COLOR_0_G:.2f},{Config.RAMP_COLOR_0_B:.2f})")
    print(f"            → Orange({Config.RAMP_COLOR_MID_R:.2f},{Config.RAMP_COLOR_MID_G:.2f},{Config.RAMP_COLOR_MID_B:.2f})")
    print(f"            → Yellow({Config.RAMP_COLOR_1_R:.2f},{Config.RAMP_COLOR_1_G:.2f},{Config.RAMP_COLOR_1_B:.2f})")
    print(f"  Emission Strength: {Config.EMISSION_STRENGTH}")
    print(f"  Density Multiplier: {Config.DENSITY_MULT}")

    return True


def setup_color_management():
    """Configure Blender's color management for post-processing.

    Uses Blender's built-in color management which is more robust
    than compositor nodes in background mode.
    """
    scene = bpy.context.scene

    # Use Filmic for better HDR handling
    scene.view_settings.view_transform = 'AgX'  # or 'Filmic' for Blender < 4.0

    # Apply exposure
    if Config.EXPOSURE != 0.0:
        scene.view_settings.exposure = Config.EXPOSURE
        print(f"[rerender] Exposure: {Config.EXPOSURE:+.2f} EV")

    # Apply gamma
    if Config.GAMMA != 1.0:
        scene.view_settings.gamma = Config.GAMMA
        print(f"[rerender] Gamma: {Config.GAMMA:.2f}")

    # Note: Additional color grading would require compositor nodes
    # For now, temperature is controlled via the Blackbody node setup


def render_frame():
    """Render a single frame with current settings."""
    scene = bpy.context.scene

    # Set frame
    scene.frame_set(Config.FRAME)

    # Render settings
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.SAMPLES
    scene.render.resolution_x = Config.RESOLUTION
    scene.render.resolution_y = Config.RESOLUTION
    scene.render.film_transparent = True

    # Output path
    output_path = Path(Config.OUTPUT_DIR) / Config.OUTPUT_NAME
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'PNG'

    # Render
    print(f"[rerender] Rendering frame {Config.FRAME} @ {Config.RESOLUTION}x{Config.RESOLUTION}, {Config.SAMPLES} samples...")
    bpy.ops.render.render(write_still=True)
    print(f"[rerender] Saved: {output_path}")

    return str(output_path)


def main():
    print("=" * 60)
    print("Post-Bake Material Iteration + Post-Processing")
    print("=" * 60)

    if not update_material():
        print("[rerender] Failed to update material!")
        return

    # Setup color management for post-processing
    setup_color_management()

    # Render
    output_path = render_frame()
    print("[rerender] Done!")

    return output_path


if __name__ == "__main__":
    main()
