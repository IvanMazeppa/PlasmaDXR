#!/usr/bin/env python3
"""
Script Generator MCP Server

Generates and modifies Blender Python scripts for NanoVDB asset creation.
Uses templates from working scripts and incorporates evaluation feedback.

Part of the Self-Improving NanoVDB Asset Generation Pipeline.

Tools:
    - generate_script: Create new Blender script from description
    - modify_script: Improve script based on evaluation feedback
    - analyze_script: Understand what a script does
    - list_templates: Show available template scripts
    - get_template: Get content of a template script

Usage:
    python server.py  # Run as MCP server (stdio transport)
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import technique catalog for variety in generation
from technique_catalog import (
    PYRO_TECHNIQUES,
    ADVANCED_TECHNIQUES,
    get_technique_by_keywords,
    get_technique_with_randomized_params,
    get_random_technique,
    list_all_techniques,
)

# =============================================================================
# Blender API Parameter Ranges (from Blender 5.0 Python API)
# Source: bpy.types.FluidDomainSettings, bpy.types.FluidFlowSettings
# These ranges are used for validation to prevent invalid parameter errors
# =============================================================================

BLENDER_PARAM_RANGES = {
    # FluidDomainSettings - GAS domain
    "burning_rate": {"min": 0.01, "max": 4.0, "default": 0.75, "type": "float"},
    "flame_smoke": {"min": 0.0, "max": 8.0, "default": 1.0, "type": "float"},
    "flame_vorticity": {"min": 0.0, "max": 2.0, "default": 0.5, "type": "float"},
    "flame_max_temp": {"min": 1.0, "max": 10.0, "default": 3.0, "type": "float"},
    "flame_ignition": {"min": 0.5, "max": 5.0, "default": 1.5, "type": "float"},
    "alpha": {"min": -5.0, "max": 5.0, "default": 1.0, "type": "float"},  # density buoyancy
    "beta": {"min": -5.0, "max": 5.0, "default": 1.0, "type": "float"},   # heat buoyancy
    "dissolve_speed": {"min": 1, "max": 10000, "default": 5, "type": "int"},
    "vorticity": {"min": 0.0, "max": 4.0, "default": 0.0, "type": "float"},

    # Noise upres parameters
    "noise_scale": {"min": 1, "max": 10, "default": 2, "type": "int"},
    "noise_strength": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
    "noise_pos_scale": {"min": 0.0001, "max": 10.0, "default": 2.0, "type": "float"},

    # FluidFlowSettings
    "fuel_amount": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
    "temperature": {"min": -10.0, "max": 10.0, "default": 1.0, "type": "float"},
    "velocity_normal": {"min": -100.0, "max": 100.0, "default": 0.0, "type": "float"},
    "velocity_random": {"min": 0.0, "max": 10.0, "default": 0.0, "type": "float"},

    # Domain settings
    "resolution_max": {"min": 8, "max": 4096, "default": 64, "type": "int"},
}


def validate_parameter(param_name: str, value: float) -> Dict[str, Any]:
    """
    Validate a single parameter against Blender API ranges.

    Returns dict with:
        valid: bool - True if in range
        clamped_value: The value clamped to valid range (if out of range)
        warning: Optional warning message
    """
    if param_name not in BLENDER_PARAM_RANGES:
        return {"valid": True, "clamped_value": value, "warning": None}

    range_info = BLENDER_PARAM_RANGES[param_name]
    min_val = range_info["min"]
    max_val = range_info["max"]

    if value < min_val:
        return {
            "valid": False,
            "clamped_value": min_val,
            "warning": f"{param_name}={value} below minimum {min_val}, clamped"
        }
    elif value > max_val:
        return {
            "valid": False,
            "clamped_value": max_val,
            "warning": f"{param_name}={value} above maximum {max_val}, clamped"
        }

    return {"valid": True, "clamped_value": value, "warning": None}


def validate_and_clamp_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate all parameters in a dict and clamp to valid ranges.

    Returns:
        - Validated params dict with clamped values
        - List of warning messages for out-of-range params
    """
    validated = {}
    warnings = []

    for param_name, value in params.items():
        if isinstance(value, (int, float)):
            result = validate_parameter(param_name, value)
            validated[param_name] = result["clamped_value"]
            if result["warning"]:
                warnings.append(result["warning"])
        else:
            validated[param_name] = value

    return validated, warnings


def generate_effector_code(effector_config: Dict[str, Any]) -> str:
    """
    Generate Blender Python code to create fluid effectors from config.

    Args:
        effector_config: Dict of effector name -> effector properties
                        {"ground_plane": {"type": "COLLISION", "shape": "plane", ...}}

    Returns:
        Python code string for the create_effectors() function body
    """
    if not effector_config:
        return "    # No effectors for this technique\n    pass"

    lines = []
    lines.append("    effectors = []")
    lines.append("")

    for effector_name, props in effector_config.items():
        effector_type = props.get("type", "COLLISION")
        shape = props.get("shape", "plane")

        # Position
        pos_x = props.get("position_x", 0.0)
        pos_y = props.get("position_y", 0.0)
        pos_z = props.get("position_z", 0.0)

        lines.append(f"    # {effector_name}: {props.get('description', effector_type)}")

        if shape == "plane":
            scale = props.get("scale", 10.0)
            lines.append(f"    bpy.ops.mesh.primitive_plane_add(size={scale}, location=({pos_x}, {pos_y}, {pos_z}))")
            lines.append(f"    effector = bpy.context.active_object")
            lines.append(f"    effector.name = '{effector_name}'")

            # Handle rotation for planes (default is XY plane, rotation_x=90 makes XZ plane, etc.)
            rot_x = props.get("rotation_x", 0)
            rot_y = props.get("rotation_y", 0)
            rot_z = props.get("rotation_z", 0)
            if rot_x != 0 or rot_y != 0 or rot_z != 0:
                import math
                rot_x_rad = rot_x * 3.14159 / 180
                rot_y_rad = rot_y * 3.14159 / 180
                rot_z_rad = rot_z * 3.14159 / 180
                lines.append(f"    effector.rotation_euler = ({rot_x_rad:.4f}, {rot_y_rad:.4f}, {rot_z_rad:.4f})")

        elif shape == "cylinder":
            radius = props.get("radius", 1.0)
            height = props.get("height", 2.0)
            lines.append(f"    bpy.ops.mesh.primitive_cylinder_add(radius={radius}, depth={height}, location=({pos_x}, {pos_y}, {pos_z}))")
            lines.append(f"    effector = bpy.context.active_object")
            lines.append(f"    effector.name = '{effector_name}'")

            # Rotation for cylinders
            rot_x = props.get("rotation_x", 0)
            rot_y = props.get("rotation_y", 0)
            rot_z = props.get("rotation_z", 0)
            if rot_x != 0 or rot_y != 0 or rot_z != 0:
                rot_x_rad = rot_x * 3.14159 / 180
                rot_y_rad = rot_y * 3.14159 / 180
                rot_z_rad = rot_z * 3.14159 / 180
                lines.append(f"    effector.rotation_euler = ({rot_x_rad:.4f}, {rot_y_rad:.4f}, {rot_z_rad:.4f})")

        elif shape == "cube":
            scale = props.get("scale", 2.0)
            lines.append(f"    bpy.ops.mesh.primitive_cube_add(size={scale}, location=({pos_x}, {pos_y}, {pos_z}))")
            lines.append(f"    effector = bpy.context.active_object")
            lines.append(f"    effector.name = '{effector_name}'")

        else:
            # Default to sphere
            radius = props.get("radius", 1.0)
            lines.append(f"    bpy.ops.mesh.primitive_uv_sphere_add(radius={radius}, location=({pos_x}, {pos_y}, {pos_z}))")
            lines.append(f"    effector = bpy.context.active_object")
            lines.append(f"    effector.name = '{effector_name}'")

        # Add Fluid modifier with Effector type
        lines.append(f"    bpy.ops.object.modifier_add(type='FLUID')")
        lines.append(f"    effector.modifiers['Fluid'].fluid_type = 'EFFECTOR'")
        lines.append(f"    eff_settings = effector.modifiers['Fluid'].effector_settings")
        lines.append(f"    eff_settings.effector_type = '{effector_type}'")

        # GUIDE-specific settings
        if effector_type == "GUIDE":
            guide_mode = props.get("guide_mode", "MAXIMUM")
            velocity_factor = props.get("velocity_factor", 1.0)
            lines.append(f"    eff_settings.guide_mode = '{guide_mode}'")
            lines.append(f"    eff_settings.velocity_factor = {velocity_factor}")

        # Hide effector in render (invisible collision surface)
        lines.append(f"    effector.hide_render = True")
        lines.append(f"    effectors.append(effector)")
        lines.append("")

    lines.append(f"    print(f'[script] Created {{len(effectors)}} effector(s)')")
    lines.append(f"    return effectors")

    return "\n".join(lines)


# Load environment
load_dotenv()

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    SCRIPT_DIR.parent.parent
))

# Script directories
TEMPLATE_DIR = PROJECT_ROOT / "assets/blender_scripts/GPT-5.2"
OUTPUT_DIR = PROJECT_ROOT / "assets/blender_scripts/generated"

# Blender path for script headers
BLENDER_PATH = "/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender"

# Create FastMCP server
mcp = FastMCP("script-generator")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScriptAnalysis:
    """Analysis of a Blender script."""
    script_path: str
    script_type: str  # "pyro", "liquid", "mesh", "shader", "utility"
    domain_type: Optional[str]  # "GAS", "LIQUID", None
    features: List[str]  # ["mantaflow", "openvdb_export", "animation", etc.]
    parameters: Dict[str, Any]  # Extracted configurable parameters
    frame_range: Optional[tuple]
    resolution: Optional[int]
    output_format: str  # "openvdb", "blend", "render"
    summary: str


@dataclass
class GeneratedScript:
    """Result of script generation."""
    success: bool
    script_path: str
    script_content: str
    template_used: Optional[str]
    parameters: Dict[str, Any]
    notes: List[str]


@dataclass
class ScriptModification:
    """Result of script modification."""
    success: bool
    original_path: str
    modified_path: str
    changes_made: List[str]
    parameters_changed: Dict[str, Any]


# =============================================================================
# Script Templates
# =============================================================================

# Common script header template
SCRIPT_HEADER = '''#!/usr/bin/env python3
"""
{title}

Generated by PlasmaDXR Script Generator
Date: {date}
Template: {template}

Description:
    {description}

Usage:
    blender --background --python {filename} -- [options]

Options:
    --resolution INT    Simulation resolution (default: {resolution})
    --frame_start INT   Start frame (default: {frame_start})
    --frame_end INT     End frame (default: {frame_end})
    --output_dir PATH   Output directory for VDB files
    --bake BOOL         Run bake (1) or skip (0)
    --render BOOL       Render preview images (1) or skip (0)
    --render_frames STR Frames to render: "mid", "all", or "1,25,50" (default: mid)
"""

import bpy
import sys
import os
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Script configuration - modify these values or pass via command line."""

    RESOLUTION = {resolution}
    FRAME_START = {frame_start}
    FRAME_END = {frame_end}
    OUTPUT_DIR = "{output_dir}"
    BAKE = True
    RENDER = True  # Enable preview rendering for evaluation
    RENDER_FRAMES = "mid"  # "mid", "all", or comma-separated frame numbers

    # Render settings for evaluation
    RENDER_RESOLUTION_X = 512
    RENDER_RESOLUTION_Y = 512
    RENDER_SAMPLES = 64  # Cycles samples (lower for speed)

    # Domain settings
    DOMAIN_SCALE = {domain_scale}

    # Simulation settings
{simulation_settings}


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
'''

# Pyro (smoke/fire) domain template
PYRO_DOMAIN_TEMPLATE = '''
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


# =============================================================================
# Domain Creation
# =============================================================================

def create_domain():
    """Create and configure fluid domain for {effect_type}."""
    # Create domain cube
    bpy.ops.mesh.primitive_cube_add(size=Config.DOMAIN_SCALE, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "FluidDomain"

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'GAS'

    # Resolution
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = True

    # Cache settings
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'

    # Gas behavior
{gas_settings}

    # Noise upres for fine detail (2-4x visual improvement)
{noise_settings}

    # Add volumetric material for rendering
    add_volume_material(domain)

    return domain


def add_volume_material(domain):
    """Add Principled Volume shader for fire/smoke visualization."""
    mat = bpy.data.materials.new(name="FireSmokeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    # Aggressive settings for visible, dramatic fire/smoke
    volume.inputs['Color'].default_value = (1.0, 0.3, 0.05, 1.0)  # Rich orange
    volume.inputs['Density'].default_value = 10.0    # High density for thick smoke
    volume.inputs['Anisotropy'].default_value = 0.5  # Forward scattering
    volume.inputs['Blackbody Intensity'].default_value = 8.0  # Intense flame glow
    volume.inputs['Temperature'].default_value = 2500.0  # Hot flames

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    domain.data.materials.append(mat)
    print("[script] Volumetric material applied (high density + emission)")


def setup_camera_and_lighting():
    """Add camera and lights for rendering."""
    bpy.ops.object.camera_add(location=(6, -6, 4))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0


def create_emitter():
    """Create flow emitter for {effect_type}."""
    # Create emitter geometry
{emitter_geometry}

    emitter = bpy.context.active_object
    emitter.name = "FlowEmitter"

    # Add fluid modifier as flow
    bpy.ops.object.modifier_add(type='FLUID')
    emitter.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = emitter.modifiers["Fluid"].flow_settings
    flow.flow_type = 'BOTH'  # Fire AND smoke
    flow.flow_behavior = 'INFLOW'
{flow_settings}

    # Hide emitter mesh in render
    emitter.hide_render = True

    return emitter


def setup_emission_dynamics(emitter):
    """Setup animated emission dynamics with keyframes."""
{emission_keyframes}


def create_effectors(domain):
    """Create fluid effectors (collision/guide objects) for shaping flow."""
{effector_code}


def setup_camera_and_lighting():
    """Add camera and lights for rendering."""
    # Camera
    bpy.ops.object.camera_add(location=(6, -6, 4))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = cam

    # Sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-4, -4, 3))
    fill = bpy.context.active_object
    fill.data.energy = 100.0


# =============================================================================
# Baking & Export
# =============================================================================

def bake_simulation(domain):
    """Bake the fluid simulation."""
    print(f"[script] Baking simulation: frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")
    print(f"[script] Resolution: {{Config.RESOLUTION}}")
    print(f"[script] Output: {{Config.OUTPUT_DIR}}")

    # Ensure output directory exists
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Select domain and bake
    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()

    print("[script] Bake complete!")


# =============================================================================
# Rendering for Evaluation
# =============================================================================

def setup_render_settings():
    """Configure render settings for evaluation previews."""
    scene = bpy.context.scene

    # Use Cycles for volumetrics
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES

    # Resolution
    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100

    # Output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 15

    # Transparent background for compositing flexibility
    scene.render.film_transparent = True

    print(f"[script] Render settings: {{Config.RENDER_RESOLUTION_X}}x{{Config.RENDER_RESOLUTION_Y}}, {{Config.RENDER_SAMPLES}} samples")


def get_frames_to_render():
    """Determine which frames to render based on Config.RENDER_FRAMES."""
    if Config.RENDER_FRAMES == "mid":
        mid = (Config.FRAME_START + Config.FRAME_END) // 2
        return [mid]
    elif Config.RENDER_FRAMES == "all":
        # Render every 10th frame for "all"
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 5)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    else:
        # Parse comma-separated frame numbers
        try:
            return [int(f.strip()) for f in Config.RENDER_FRAMES.split(",")]
        except ValueError:
            mid = (Config.FRAME_START + Config.FRAME_END) // 2
            return [mid]


def render_previews():
    """Render preview images for quality evaluation."""
    setup_render_settings()

    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = get_frames_to_render()
    rendered_files = []

    print(f"[script] Rendering {{len(frames)}} preview frame(s): {{frames}}")

    for frame in frames:
        # Clamp frame to valid range
        frame = max(Config.FRAME_START, min(frame, Config.FRAME_END))

        scene.frame_set(frame)

        # Output path
        render_path = output_dir / f"render_{{frame:04d}}.png"
        scene.render.filepath = str(render_path)

        # Render
        print(f"[script] Rendering frame {{frame}}...")
        bpy.ops.render.render(write_still=True)

        rendered_files.append(str(render_path))
        print(f"[script] Saved: {{render_path}}")

    print(f"[script] Rendered {{len(rendered_files)}} preview(s)")
    return rendered_files


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("{title}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    domain = create_domain()
    emitter = create_emitter()
    setup_emission_dynamics(emitter)  # Animated emission profiles
    create_effectors(domain)  # Create collision/guide effectors
    setup_camera_and_lighting()

    if Config.BAKE:
        bake_simulation(domain)
    else:
        print("[script] Skipping bake (--bake 0)")

    # Render preview images for evaluation
    if Config.RENDER:
        render_previews()
    else:
        print("[script] Skipping render (--render 0)")

    # Save blend file
    blend_path = Path(Config.OUTPUT_DIR) / "{filename_stem}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {{blend_path}}")


if __name__ == "__main__":
    main()
'''

# Liquid domain template
LIQUID_DOMAIN_TEMPLATE = '''
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


# =============================================================================
# Domain Creation
# =============================================================================

def create_domain():
    """Create and configure fluid domain for liquid simulation."""
    bpy.ops.mesh.primitive_cube_add(size=Config.DOMAIN_SCALE, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "LiquidDomain"

    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'LIQUID'

    settings.resolution_max = Config.RESOLUTION
    settings.use_mesh = True
    settings.mesh_scale = 1.0

    # Cache
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'

{liquid_settings}

    return domain


def create_inflow():
    """Create liquid inflow source."""
{inflow_geometry}

    inflow = bpy.context.active_object
    inflow.name = "LiquidInflow"

    bpy.ops.object.modifier_add(type='FLUID')
    inflow.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = inflow.modifiers["Fluid"].flow_settings
    flow.flow_type = 'LIQUID'
    flow.flow_behavior = 'INFLOW'
{inflow_settings}

    return inflow


# =============================================================================
# Baking
# =============================================================================

def bake_simulation(domain):
    """Bake liquid simulation."""
    print(f"[script] Baking liquid: frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()
    print("[script] Bake complete!")


# =============================================================================
# Rendering for Evaluation
# =============================================================================

def setup_render_settings():
    """Configure render settings for evaluation previews."""
    scene = bpy.context.scene

    # Use Cycles for volumetrics
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES

    # Resolution
    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100

    # Output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 15

    print(f"[script] Render settings: {{Config.RENDER_RESOLUTION_X}}x{{Config.RENDER_RESOLUTION_Y}}, {{Config.RENDER_SAMPLES}} samples")


def get_frames_to_render():
    """Determine which frames to render based on Config.RENDER_FRAMES."""
    if Config.RENDER_FRAMES == "mid":
        mid = (Config.FRAME_START + Config.FRAME_END) // 2
        return [mid]
    elif Config.RENDER_FRAMES == "all":
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 5)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    else:
        try:
            return [int(f.strip()) for f in Config.RENDER_FRAMES.split(",")]
        except ValueError:
            mid = (Config.FRAME_START + Config.FRAME_END) // 2
            return [mid]


def render_previews():
    """Render preview images for quality evaluation."""
    setup_render_settings()

    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = get_frames_to_render()
    rendered_files = []

    print(f"[script] Rendering {{len(frames)}} preview frame(s): {{frames}}")

    for frame in frames:
        frame = max(Config.FRAME_START, min(frame, Config.FRAME_END))
        scene.frame_set(frame)

        render_path = output_dir / f"render_{{frame:04d}}.png"
        scene.render.filepath = str(render_path)

        print(f"[script] Rendering frame {{frame}}...")
        bpy.ops.render.render(write_still=True)

        rendered_files.append(str(render_path))
        print(f"[script] Saved: {{render_path}}")

    print(f"[script] Rendered {{len(rendered_files)}} preview(s)")
    return rendered_files


def main():
    print("=" * 60)
    print("{title}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    domain = create_domain()
    inflow = create_inflow()

    if Config.BAKE:
        bake_simulation(domain)
    else:
        print("[script] Skipping bake (--bake 0)")

    # Render preview images for evaluation
    if Config.RENDER:
        render_previews()
    else:
        print("[script] Skipping render (--render 0)")

    blend_path = Path(Config.OUTPUT_DIR) / "{filename_stem}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {{blend_path}}")


if __name__ == "__main__":
    main()
'''


# =============================================================================
# Helper Functions
# =============================================================================

def extract_script_info(content: str) -> Dict[str, Any]:
    """Extract information from a script's content."""
    info = {
        "domain_type": None,
        "resolution": None,
        "frame_range": None,
        "features": [],
        "parameters": {}
    }

    # Detect domain type
    if "domain_type = 'GAS'" in content or 'domain_type = "GAS"' in content:
        info["domain_type"] = "GAS"
        info["features"].append("pyro")
    elif "domain_type = 'LIQUID'" in content or 'domain_type = "LIQUID"' in content:
        info["domain_type"] = "LIQUID"
        info["features"].append("liquid")

    # Detect features
    if "OPENVDB" in content or "openvdb" in content.lower():
        info["features"].append("openvdb_export")
    if "animation" in content.lower() or "keyframe" in content.lower():
        info["features"].append("animation")
    if "noise" in content.lower():
        info["features"].append("procedural_noise")
    if "turbulence" in content.lower():
        info["features"].append("turbulence")

    # Extract resolution
    res_match = re.search(r'resolution_max\s*=\s*(\d+)', content)
    if res_match:
        info["resolution"] = int(res_match.group(1))

    # Extract frame range
    start_match = re.search(r'frame_start\s*=\s*(\d+)', content, re.IGNORECASE)
    end_match = re.search(r'frame_end\s*=\s*(\d+)', content, re.IGNORECASE)
    if start_match and end_match:
        info["frame_range"] = (int(start_match.group(1)), int(end_match.group(1)))

    return info


def determine_script_type(content: str, filename: str) -> str:
    """Determine the type of script from content and filename."""
    content_lower = content.lower()
    filename_lower = filename.lower()

    if "domain_type = 'gas'" in content_lower or "smoke" in filename_lower or "fire" in filename_lower or "explosion" in filename_lower:
        return "pyro"
    elif "domain_type = 'liquid'" in content_lower or "liquid" in filename_lower or "water" in filename_lower:
        return "liquid"
    elif "bpy.types.shader" in content_lower or "node_tree" in content_lower:
        return "shader"
    elif "bpy.ops.mesh" in content_lower and "modifier" not in content_lower:
        return "mesh"
    else:
        return "utility"


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def list_templates(
    script_type: Optional[str] = None
) -> str:
    """
    List available template scripts.

    Args:
        script_type: Optional filter by type ("pyro", "liquid", "mesh", etc.)

    Returns:
        JSON array of template scripts with metadata

    Example:
        list_templates("pyro")  # List only pyro/explosion templates
    """
    templates = []

    if not TEMPLATE_DIR.exists():
        return json.dumps({"error": f"Template directory not found: {TEMPLATE_DIR}"})

    for script_path in TEMPLATE_DIR.rglob("*.py"):
        if "__pycache__" in str(script_path):
            continue

        try:
            content = script_path.read_text()
            detected_type = determine_script_type(content, script_path.name)

            # Apply filter
            if script_type and detected_type != script_type:
                continue

            # Extract docstring
            description = ""
            if content.startswith('"""') or content.startswith("'''"):
                quote = '"""' if content.startswith('"""') else "'''"
                end = content.find(quote, 3)
                if end > 0:
                    description = content[3:end].strip().split("\n")[0]

            info = extract_script_info(content)

            templates.append({
                "path": str(script_path.relative_to(PROJECT_ROOT)),
                "name": script_path.stem,
                "type": detected_type,
                "description": description[:100] if description else "",
                "domain_type": info["domain_type"],
                "resolution": info["resolution"],
                "features": info["features"]
            })
        except Exception as e:
            continue

    # Sort by type then name
    templates.sort(key=lambda x: (x["type"], x["name"]))

    return json.dumps({
        "count": len(templates),
        "filter": script_type,
        "templates": templates
    }, indent=2)


@mcp.tool()
async def get_template(
    template_name: str
) -> str:
    """
    Get the full content of a template script.

    Args:
        template_name: Name or path of template (e.g., "blender_hydrogen_cloud" or full path)

    Returns:
        JSON with script content and metadata

    Example:
        get_template("blender_explosion_grenade_bake")
    """
    # Find the template
    template_path = None

    # Check if it's a full path
    if "/" in template_name:
        template_path = PROJECT_ROOT / template_name
    else:
        # Search for it
        for script_path in TEMPLATE_DIR.rglob("*.py"):
            if script_path.stem == template_name or template_name in script_path.stem:
                template_path = script_path
                break

    if not template_path or not template_path.exists():
        return json.dumps({"error": f"Template not found: {template_name}"})

    content = template_path.read_text()
    info = extract_script_info(content)

    return json.dumps({
        "path": str(template_path.relative_to(PROJECT_ROOT)),
        "name": template_path.stem,
        "type": determine_script_type(content, template_path.name),
        "content": content,
        "info": info
    }, indent=2)


@mcp.tool()
async def analyze_script(
    script_path: str
) -> str:
    """
    Analyze a Blender script to understand its structure and parameters.

    Args:
        script_path: Path to script to analyze

    Returns:
        JSON with ScriptAnalysis containing type, features, parameters

    Example:
        analyze_script("assets/blender_scripts/GPT-5.2/blender_hydrogen_cloud.py")
    """
    path = Path(script_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / script_path

    if not path.exists():
        return json.dumps({"error": f"Script not found: {path}"})

    content = path.read_text()
    info = extract_script_info(content)
    script_type = determine_script_type(content, path.name)

    # Extract docstring for summary
    summary = ""
    if content.startswith('"""') or content.startswith("'''"):
        quote = '"""' if content.startswith('"""') else "'''"
        end = content.find(quote, 3)
        if end > 0:
            docstring = content[3:end].strip()
            # Get first paragraph
            paras = docstring.split("\n\n")
            summary = paras[0].replace("\n", " ")[:200]

    # Determine output format
    output_format = "blend"
    if "OPENVDB" in content:
        output_format = "openvdb"
    elif "render" in content.lower():
        output_format = "render"

    result = ScriptAnalysis(
        script_path=str(path),
        script_type=script_type,
        domain_type=info["domain_type"],
        features=info["features"],
        parameters=info["parameters"],
        frame_range=info["frame_range"],
        resolution=info["resolution"],
        output_format=output_format,
        summary=summary
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def generate_script(
    effect_type: str,
    description: str,
    output_name: str,
    resolution: int = 96,
    frame_start: int = 1,
    frame_end: int = 50,
    template_name: Optional[str] = None,
    technique_name: Optional[str] = None,
    force_random_technique: bool = False
) -> str:
    """
    Generate a new Blender script from a description.

    Uses the Technique Catalog to select categorically different approaches,
    ensuring variety across generated scripts.

    Args:
        effect_type: Type of effect ("pyro", "liquid", "explosion", "nebula", etc.)
        description: Description of what to create
        output_name: Name for the output script (without .py)
        resolution: Simulation resolution (default 96)
        frame_start: Start frame (default 1)
        frame_end: End frame (default 50)
        template_name: Optional template to base on
        technique_name: Optional specific technique from catalog (e.g., "rising_mushroom")
        force_random_technique: If True, ignore keywords and pick randomly for maximum variety

    Returns:
        JSON with GeneratedScript containing path and content

    Example:
        generate_script(
            effect_type="pyro",
            description="A rising mushroom cloud explosion with bright orange fire",
            output_name="mushroom_cloud",
            resolution=128,
            frame_end=100
        )
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{output_name}.py"
    notes = []

    # Determine base template
    if template_name:
        template_result = await get_template(template_name)
        template_data = json.loads(template_result)
        if "error" in template_data:
            notes.append(f"Template not found, using built-in: {template_name}")
            template_name = None

    # Select appropriate template based on effect type
    effect_lower = effect_type.lower()

    if effect_lower in ["pyro", "smoke", "fire", "explosion", "nebula", "gas"]:
        domain_template = PYRO_DOMAIN_TEMPLATE
        domain_type = "GAS"

        # =====================================================================
        # TECHNIQUE SELECTION - Core variety mechanism
        # =====================================================================
        selected_technique = None

        if technique_name and technique_name in PYRO_TECHNIQUES:
            # Explicit technique requested
            selected_technique = get_technique_with_randomized_params(technique_name, "pyro")
            notes.append(f"Using requested technique: {technique_name}")
        elif force_random_technique:
            # Force random for maximum variety
            selected_technique = get_random_technique("pyro")
            selected_technique = get_technique_with_randomized_params(
                selected_technique["name"], "pyro"
            )
            notes.append(f"Random technique selected: {selected_technique['name']}")
        else:
            # Keyword matching with fallback to random
            selected_technique = get_technique_with_randomized_params(description, "pyro")
            notes.append(f"Technique: {selected_technique['name']} (keyword match)")

        # Extract parameters from technique
        domain_params = selected_technique.get("domain_params", {})
        flow_params = selected_technique.get("flow_params", {})
        emitter_config = selected_technique.get("emitter", {})
        noise_params = selected_technique.get("noise_params", {})
        emission_dynamics = selected_technique.get("emission_dynamics", {})
        effector_config = selected_technique.get("effectors", {})

        # =====================================================================
        # PARAMETER VALIDATION - Clamp to Blender API ranges
        # =====================================================================
        domain_params, domain_warnings = validate_and_clamp_params(domain_params)
        flow_params, flow_warnings = validate_and_clamp_params(flow_params)
        noise_params, noise_warnings = validate_and_clamp_params(noise_params)

        all_warnings = domain_warnings + flow_warnings + noise_warnings
        if all_warnings:
            notes.append(f"Validation: {len(all_warnings)} param(s) clamped to valid ranges")
            for w in all_warnings[:3]:  # Show first 3 warnings
                notes.append(f"  - {w}")

        # Build gas settings from technique parameters
        gas_lines = []
        if "burning_rate" in domain_params:
            gas_lines.append(f"    settings.burning_rate = {domain_params['burning_rate']:.2f}")
        if "flame_smoke" in domain_params:
            gas_lines.append(f"    settings.flame_smoke = {domain_params['flame_smoke']:.2f}")
        if "flame_vorticity" in domain_params:
            gas_lines.append(f"    settings.flame_vorticity = {domain_params['flame_vorticity']:.2f}")
        if "flame_max_temp" in domain_params:
            gas_lines.append(f"    settings.flame_max_temp = {domain_params['flame_max_temp']:.2f}")
        if "flame_ignition" in domain_params:
            gas_lines.append(f"    settings.flame_ignition = {domain_params['flame_ignition']:.2f}")
        if "alpha" in domain_params:
            gas_lines.append(f"    settings.alpha = {domain_params['alpha']:.2f}  # Density buoyancy")
        if "beta" in domain_params:
            gas_lines.append(f"    settings.beta = {domain_params['beta']:.2f}  # Heat buoyancy")
        if "dissolve_speed" in domain_params:
            gas_lines.append(f"    settings.dissolve_speed = {int(domain_params['dissolve_speed'])}")
            gas_lines.append("    settings.use_dissolve_smoke = True")

        # Add technique signature comment
        gas_lines.insert(0, f"    # Technique: {selected_technique['name']}")
        gas_lines.insert(1, f"    # {selected_technique.get('visual_signature', 'Custom effect')}")

        gas_settings = "\n".join(gas_lines)

        # =====================================================================
        # NOISE UPRES SETTINGS - 2-4x visual detail improvement
        # =====================================================================
        noise_lines = []
        if noise_params.get("use_noise", False):
            noise_lines.append("    settings.use_noise = True")
            if "noise_scale" in noise_params:
                noise_lines.append(f"    settings.noise_scale = {int(noise_params['noise_scale'])}")
            if "noise_strength" in noise_params:
                noise_lines.append(f"    settings.noise_strength = {noise_params['noise_strength']:.2f}")
            if "noise_pos_scale" in noise_params:
                noise_lines.append(f"    settings.noise_pos_scale = {noise_params['noise_pos_scale']:.2f}")
            notes.append(f"Noise upres: scale={noise_params.get('noise_scale', 2)}, strength={noise_params.get('noise_strength', 1.0):.1f}")
        else:
            noise_lines.append("    # Noise upres disabled for this technique")
        noise_settings = "\n".join(noise_lines)

        # =====================================================================
        # EMISSION DYNAMICS KEYFRAMES - Realistic combustion animation
        # =====================================================================
        emission_lines = []
        if emission_dynamics:
            profile = emission_dynamics.get("profile", "sustained")
            peak_frame = int(emission_dynamics.get("peak_frame", 5))
            decay_start = int(emission_dynamics.get("decay_start", 25))
            decay_end = int(emission_dynamics.get("decay_end", 50))
            peak_fuel = emission_dynamics.get("peak_fuel_multiplier", 1.5)
            decay_fuel = emission_dynamics.get("decay_fuel_multiplier", 0.2)

            emission_lines.append(f"    # Emission profile: {profile}")
            emission_lines.append("    flow = emitter.modifiers['Fluid'].flow_settings")
            emission_lines.append(f"    base_fuel = flow.fuel_amount")
            emission_lines.append("")
            emission_lines.append("    # Keyframe emission dynamics")

            if profile == "burst":
                # Explosive start with rapid decay
                emission_lines.append(f"    # Frame 1: Initial state (low)")
                emission_lines.append(f"    bpy.context.scene.frame_set(1)")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * 0.3")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {peak_frame}: Peak explosion")
                emission_lines.append(f"    bpy.context.scene.frame_set({peak_frame})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {peak_fuel:.1f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {decay_start}: Decay begins")
                emission_lines.append(f"    bpy.context.scene.frame_set({decay_start})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {(peak_fuel + decay_fuel) / 2:.1f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {decay_end}: Nearly burned out")
                emission_lines.append(f"    bpy.context.scene.frame_set({decay_end})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {decay_fuel:.2f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")

            elif profile == "sustained":
                # Steady burn with gradual ramp up and down
                emission_lines.append(f"    # Frame 1: Ramp up start")
                emission_lines.append(f"    bpy.context.scene.frame_set(1)")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * 0.5")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {peak_frame}: Full intensity")
                emission_lines.append(f"    bpy.context.scene.frame_set({peak_frame})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {peak_fuel:.1f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {decay_start}: Still burning strong")
                emission_lines.append(f"    bpy.context.scene.frame_set({decay_start})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {peak_fuel:.1f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {decay_end}: Fade out")
                emission_lines.append(f"    bpy.context.scene.frame_set({decay_end})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {decay_fuel:.2f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")

            elif profile == "pulsing":
                # Rhythmic emission waves
                pulse_period = int(emission_dynamics.get("pulse_period", 10))
                emission_lines.append(f"    # Pulsing emission with {pulse_period}-frame period")
                emission_lines.append(f"    for frame in range(1, Config.FRAME_END, {pulse_period}):")
                emission_lines.append(f"        bpy.context.scene.frame_set(frame)")
                emission_lines.append(f"        flow.fuel_amount = base_fuel * {peak_fuel:.1f}")
                emission_lines.append(f"        flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append(f"        bpy.context.scene.frame_set(frame + {pulse_period // 2})")
                emission_lines.append(f"        flow.fuel_amount = base_fuel * {decay_fuel:.1f}")
                emission_lines.append(f"        flow.keyframe_insert(data_path='fuel_amount')")

            elif profile == "decay_only":
                # Already burning, just fading
                emission_lines.append(f"    # Frame 1: Already smoldering")
                emission_lines.append(f"    bpy.context.scene.frame_set(1)")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {peak_fuel:.1f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")
                emission_lines.append("")
                emission_lines.append(f"    # Frame {decay_end}: Embers dying")
                emission_lines.append(f"    bpy.context.scene.frame_set({decay_end})")
                emission_lines.append(f"    flow.fuel_amount = base_fuel * {decay_fuel:.2f}")
                emission_lines.append(f"    flow.keyframe_insert(data_path='fuel_amount')")

            emission_lines.append("")
            emission_lines.append("    # Reset to frame 1")
            emission_lines.append("    bpy.context.scene.frame_set(1)")
            emission_lines.append(f"    print(f'[script] Emission dynamics: {profile} profile applied')")

            notes.append(f"Emission: {profile} (peak@{peak_frame}, decay@{decay_start}-{decay_end})")
        else:
            emission_lines.append("    # No emission dynamics for this technique")
            emission_lines.append("    pass")

        emission_keyframes = "\n".join(emission_lines)

        # Build emitter geometry based on technique
        emitter_shape = emitter_config.get("shape", "sphere")
        emitter_z = emitter_config.get("position_z", "ground")

        # Map position to Z coordinate
        z_positions = {
            "ground": -2.0,
            "bottom": -3.0,
            "mid": 0.0,
            "elevated": 2.0,
            "ceiling": 4.0,
            "surface": 0.0,
        }
        z_coord = z_positions.get(emitter_z, 0.0)

        if emitter_shape == "sphere":
            radius = emitter_config.get("radius", (0.8, 1.2))
            if isinstance(radius, tuple):
                radius = (radius[0] + radius[1]) / 2
            emitter_geometry = f"    bpy.ops.mesh.primitive_uv_sphere_add(radius={radius:.2f}, location=(0, 0, {z_coord:.1f}))"
        elif emitter_shape == "plane":
            scale = emitter_config.get("scale_xy", (2.0, 2.0))
            if isinstance(scale, tuple):
                scale_val = (scale[0] + scale[1]) / 2
            else:
                scale_val = scale
            emitter_geometry = f"    bpy.ops.mesh.primitive_plane_add(size={scale_val:.1f}, location=(0, 0, {z_coord:.1f}))"
        elif emitter_shape == "cone":
            emitter_geometry = f"    bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0, location=(0, 0, {z_coord:.1f}))"
        else:
            emitter_geometry = f"    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, {z_coord:.1f}))"

        # Build flow settings from technique
        flow_lines = []
        flow_type = flow_params.get("flow_type", "BOTH")
        flow_lines.append(f"    flow.flow_type = '{flow_type}'")

        if flow_params.get("flow_behavior"):
            flow_lines.append(f"    flow.flow_behavior = '{flow_params['flow_behavior']}'")

        if "fuel_amount" in flow_params:
            flow_lines.append(f"    flow.fuel_amount = {flow_params['fuel_amount']:.2f}")
        if "temperature" in flow_params:
            flow_lines.append(f"    flow.temperature = {flow_params['temperature']:.2f}")
        if "velocity_normal" in flow_params:
            flow_lines.append(f"    flow.velocity_normal = {flow_params['velocity_normal']:.2f}")
        if "velocity_random" in flow_params:
            flow_lines.append(f"    flow.velocity_random = {flow_params['velocity_random']:.2f}")

        flow_settings = "\n".join(flow_lines)

        # Domain scale based on technique
        domain_scale = 8.0 if selected_technique["name"] in ["volcanic_plume", "stellar_flare"] else 6.0

        # Add technique info to notes
        notes.append(f"Visual: {selected_technique.get('visual_signature', 'N/A')}")

        # Generate effector code from technique config
        effector_code = generate_effector_code(effector_config)
        if effector_config:
            notes.append(f"Effectors: {len(effector_config)} ({', '.join(effector_config.keys())})")

    elif effect_lower in ["liquid", "water", "fluid"]:
        domain_template = LIQUID_DOMAIN_TEMPLATE
        domain_type = "LIQUID"
        gas_settings = ""
        noise_settings = "    # Noise not applicable for liquid simulations"
        emission_keyframes = "    # No emission dynamics for liquid\n    pass"
        emitter_geometry = """    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 1))"""
        flow_settings = """    flow.use_initial_velocity = True
    flow.velocity_factor = 1.0"""
        domain_scale = 4.0
        effector_code = "    # No effectors for liquid simulation\n    pass"
    else:
        # Default to pyro with random technique for variety
        domain_template = PYRO_DOMAIN_TEMPLATE
        domain_type = "GAS"
        selected_technique = get_random_technique("pyro")
        notes.append(f"Unknown effect '{effect_type}', using random pyro technique: {selected_technique['name']}")
        gas_settings = """    settings.vorticity = 0.3
    settings.flame_vorticity = 0.5"""
        noise_settings = "    settings.use_noise = True\n    settings.noise_scale = 2\n    settings.noise_strength = 1.0"
        emission_keyframes = "    # Default emission (no keyframes)\n    pass"
        emitter_geometry = """    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0))"""
        flow_settings = """    flow.flow_type = 'BOTH'"""
        domain_scale = 6.0
        # Use effector config from selected random technique if available
        effector_config = selected_technique.get("effectors", {})
        effector_code = generate_effector_code(effector_config)

    # Build simulation settings for Config class
    sim_settings = f"""    # Effect: {effect_type}
    # {description[:50]}..."""

    # Format the header
    header = SCRIPT_HEADER.format(
        title=f"GPT-5.2 — {output_name.replace('_', ' ').title()} (OpenVDB) - Blender 5.0+",
        date=datetime.now().strftime("%Y-%m-%d"),
        template=template_name or f"built-in {effect_type}",
        description=description,
        filename=f"{output_name}.py",
        resolution=resolution,
        frame_start=frame_start,
        frame_end=frame_end,
        output_dir=str(PROJECT_ROOT / "build/vdb_output" / output_name),
        domain_scale=domain_scale,
        simulation_settings=sim_settings
    )

    # Format the body
    body = domain_template.format(
        effect_type=effect_type,
        title=f"{output_name.replace('_', ' ').title()}",
        filename_stem=output_name,
        gas_settings=gas_settings,
        noise_settings=noise_settings,
        emission_keyframes=emission_keyframes,
        emitter_geometry=emitter_geometry,
        flow_settings=flow_settings,
        effector_code=effector_code,
        liquid_settings=gas_settings,
        inflow_geometry=emitter_geometry,
        inflow_settings=flow_settings
    )

    # Combine
    script_content = header + body

    # Write script
    output_path.write_text(script_content)
    notes.append(f"Generated {len(script_content)} bytes")

    result = GeneratedScript(
        success=True,
        script_path=str(output_path.relative_to(PROJECT_ROOT)),
        script_content=script_content,
        template_used=template_name,
        parameters={
            "effect_type": effect_type,
            "resolution": resolution,
            "frame_range": [frame_start, frame_end],
            "domain_type": domain_type
        },
        notes=notes
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def modify_script(
    script_path: str,
    modifications: Dict[str, Any],
    output_name: Optional[str] = None
) -> str:
    """
    Modify an existing script based on feedback or parameter changes.

    Args:
        script_path: Path to script to modify
        modifications: Dict of changes to make, can include:
            - resolution: New resolution value
            - frame_end: New end frame
            - turbulence: Turbulence/vorticity value (0-1)
            - density: Density multiplier
            - temperature: Temperature value
            - custom_code: Dict of {search_pattern: replacement}
        output_name: Optional new filename (default: adds "_modified" suffix)

    Returns:
        JSON with ScriptModification result

    Example:
        modify_script(
            "assets/blender_scripts/generated/mushroom_cloud.py",
            {
                "resolution": 128,
                "turbulence": 0.8,
                "frame_end": 150
            }
        )
    """
    path = Path(script_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / script_path

    if not path.exists():
        return json.dumps({"error": f"Script not found: {path}"})

    content = path.read_text()
    changes_made = []
    params_changed = {}

    # Apply modifications
    if "resolution" in modifications:
        new_res = modifications["resolution"]
        content = re.sub(
            r'(resolution_max\s*=\s*)\d+',
            f'\\g<1>{new_res}',
            content
        )
        content = re.sub(
            r'(RESOLUTION\s*=\s*)\d+',
            f'\\g<1>{new_res}',
            content
        )
        changes_made.append(f"Resolution: {new_res}")
        params_changed["resolution"] = new_res

    if "frame_end" in modifications:
        new_end = modifications["frame_end"]
        content = re.sub(
            r'(frame_end\s*=\s*)\d+',
            f'\\g<1>{new_end}',
            content,
            flags=re.IGNORECASE
        )
        content = re.sub(
            r'(FRAME_END\s*=\s*)\d+',
            f'\\g<1>{new_end}',
            content
        )
        changes_made.append(f"Frame end: {new_end}")
        params_changed["frame_end"] = new_end

    if "frame_start" in modifications:
        new_start = modifications["frame_start"]
        content = re.sub(
            r'(frame_start\s*=\s*)\d+',
            f'\\g<1>{new_start}',
            content,
            flags=re.IGNORECASE
        )
        changes_made.append(f"Frame start: {new_start}")
        params_changed["frame_start"] = new_start

    if "turbulence" in modifications or "vorticity" in modifications:
        new_turb = modifications.get("turbulence", modifications.get("vorticity"))
        content = re.sub(
            r'(vorticity\s*=\s*)[\d.]+',
            f'\\g<1>{new_turb}',
            content
        )
        changes_made.append(f"Vorticity: {new_turb}")
        params_changed["vorticity"] = new_turb

    if "temperature" in modifications:
        new_temp = modifications["temperature"]
        content = re.sub(
            r'(temperature\s*=\s*)[\d.]+',
            f'\\g<1>{new_temp}',
            content
        )
        changes_made.append(f"Temperature: {new_temp}")
        params_changed["temperature"] = new_temp

    if "domain_scale" in modifications:
        new_scale = modifications["domain_scale"]
        content = re.sub(
            r'(DOMAIN_SCALE\s*=\s*)[\d.]+',
            f'\\g<1>{new_scale}',
            content
        )
        changes_made.append(f"Domain scale: {new_scale}")
        params_changed["domain_scale"] = new_scale

    # Custom code replacements
    if "custom_code" in modifications:
        for search, replace in modifications["custom_code"].items():
            if search in content:
                content = content.replace(search, replace)
                changes_made.append(f"Custom: {search[:30]}...")

    # Determine output path
    if output_name:
        output_path = OUTPUT_DIR / f"{output_name}.py"
    else:
        output_path = OUTPUT_DIR / f"{path.stem}_modified.py"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    result = ScriptModification(
        success=True,
        original_path=str(path.relative_to(PROJECT_ROOT)),
        modified_path=str(output_path.relative_to(PROJECT_ROOT)),
        changes_made=changes_made,
        parameters_changed=params_changed
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def list_techniques(
    effect_type: str = "pyro"
) -> str:
    """
    List available techniques from the catalog for variety in generation.

    Each technique produces categorically different visual results.
    Use technique_name parameter in generate_script to select specific techniques.

    Args:
        effect_type: Type of effect ("pyro" currently supported)

    Returns:
        JSON with available techniques and their descriptions

    Example:
        list_techniques("pyro")
    """
    if effect_type.lower() == "pyro":
        techniques = []
        for name, tech in PYRO_TECHNIQUES.items():
            domain_params = tech.get("domain_params", {})
            techniques.append({
                "name": name,
                "description": tech.get("description", ""),
                "visual_signature": tech.get("visual_signature", ""),
                "keywords": tech.get("keywords", []),
                "key_differences": {
                    "burning_rate": domain_params.get("burning_rate", "default"),
                    "flame_smoke": domain_params.get("flame_smoke", "default"),
                    "flame_vorticity": domain_params.get("flame_vorticity", "default"),
                    "beta_buoyancy": domain_params.get("beta", "default"),
                }
            })

        return json.dumps({
            "effect_type": effect_type,
            "count": len(techniques),
            "techniques": techniques,
            "usage": "Use technique_name='rising_mushroom' in generate_script()",
            "tip": "Use force_random_technique=True for maximum variety"
        }, indent=2)

    return json.dumps({
        "error": f"Unknown effect type: {effect_type}",
        "supported": ["pyro"]
    })


@mcp.tool()
async def validate_parameters(
    params: Dict[str, Any]
) -> str:
    """
    Validate Blender fluid simulation parameters against API ranges.

    Uses documented Blender 5.0 Python API ranges to validate parameters.
    Returns validation results with any necessary clamping.

    Args:
        params: Dictionary of parameter name -> value to validate

    Returns:
        JSON with validation results, including:
        - valid: True if all params in range
        - validated_params: Clamped values
        - warnings: List of any out-of-range issues

    Example:
        validate_parameters({
            "burning_rate": 5.0,  # Above max of 4.0
            "flame_smoke": 3.5,   # Valid
            "temperature": 15.0   # Above max of 10.0
        })
    """
    validated, warnings = validate_and_clamp_params(params)

    return json.dumps({
        "valid": len(warnings) == 0,
        "validated_params": validated,
        "warnings": warnings,
        "param_count": len(params),
        "clamped_count": len(warnings),
    }, indent=2)


@mcp.tool()
async def get_parameter_ranges() -> str:
    """
    Get all documented Blender 5.0 fluid simulation parameter ranges.

    Returns the valid ranges for all known parameters, sourced from
    bpy.types.FluidDomainSettings and bpy.types.FluidFlowSettings.

    Useful for understanding what values are valid before generation.

    Returns:
        JSON with all parameter ranges

    Example:
        get_parameter_ranges()
    """
    return json.dumps({
        "source": "Blender 5.0 Python API",
        "parameter_ranges": BLENDER_PARAM_RANGES,
        "categories": {
            "domain_gas": ["burning_rate", "flame_smoke", "flame_vorticity",
                          "flame_max_temp", "flame_ignition", "alpha", "beta",
                          "dissolve_speed", "vorticity"],
            "noise_upres": ["noise_scale", "noise_strength", "noise_pos_scale"],
            "flow": ["fuel_amount", "temperature", "velocity_normal", "velocity_random"],
            "general": ["resolution_max"],
        }
    }, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
