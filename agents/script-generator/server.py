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
from dataclasses import dataclass, asdict, field
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

# Import script validator for pre-execution validation
from validator import (
    BlenderScriptValidator,
    ValidationResult,
    ValidationSeverity,
    ValidationIssue,
    validate_script as validate_script_file,
    validate_script_content,
)

# Import effect registry for simulation type extensibility (Phase 2.5)
from effect_registry import (
    EFFECT_TYPES,
    get_effect_type,
    get_category,
    get_output_format,
    get_execution_pattern,
    is_volumetric,
    is_mesh_physics,
    requires_live_render,
    get_evaluation_metrics,
    get_recommended_settings,
    get_blender_physics_type,
    validate_effect_type,
    SimulationCategory,
    OutputFormat,
    ExecutionPattern,
)

# Import technique selector for UCB1-based intelligent selection (Phase 3)
from technique_selector import (
    TechniquePerformanceStore,
    TechniqueRecommendation,
    recommend_technique as ucb1_recommend_technique,
)

# Import knowledge client for mandatory warning checks (Phase 4)
from knowledge_client import (
    KnowledgeClient,
    WarningCheck,
    check_before_modify as kb_check_before_modify,
    preload_knowledge,
    get_knowledge_client,
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
    # Phase 4.1: Knowledge base integration
    warnings: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    has_critical_warnings: bool = False


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
# MESH PHYSICS TEMPLATES (Phase 2.5 - Simulation Type Extensibility)
# =============================================================================
# These templates use live_render execution pattern (frame-by-frame physics)
# instead of bake_export pattern used by volumetric simulations.
# =============================================================================

# Soft Body Template - Jelly, rubber, organic deformable objects
SOFT_BODY_TEMPLATE = '''
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
# Soft Body Object
# =============================================================================

def create_soft_body_object():
    """Create and configure soft body object."""
{object_geometry}
    obj = bpy.context.active_object
    obj.name = "SoftBodyObject"

    # Apply subdivision for smoother deformation
    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers["Subdivision"].levels = 2
    obj.modifiers["Subdivision"].render_levels = 2

    # Add soft body physics
    bpy.ops.object.modifier_add(type='SOFT_BODY')
    settings = obj.modifiers["Soft Body"].settings

    # CRITICAL: Stability settings (from Gemini 3 Pro discoveries)
    # Default values cause jitter explosions for joined primitives
    settings.step_min = {step_min}  # Default 5 - prevents jitter
    settings.step_max = {step_max}  # Default 10 - adaptive stepping

    # Damping prevents energy buildup
    settings.ball_damp = {damping}  # Default 0.5

    # Friction and goal settings
    settings.friction = {friction}
{soft_body_settings}

    return obj


def create_collision_ground():
    """Create ground plane for soft body collision."""
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -2))
    ground = bpy.context.active_object
    ground.name = "CollisionGround"

    # Add collision physics
    bpy.ops.object.modifier_add(type='COLLISION')
    ground.modifiers["Collision"].settings.thickness_outer = 0.1

    return ground


# =============================================================================
# Live Render Pattern (frame-by-frame physics evaluation)
# =============================================================================

def setup_render_settings():
    """Configure render settings for live physics capture."""
    scene = bpy.context.scene

    # Use Cycles for quality or Eevee for speed
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES

    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100


def render_with_physics():
    """
    Live render with per-frame physics update.

    CRITICAL: bpy.context.view_layer.update() forces physics solve each frame.
    Without this, physics bake is unreliable in headless mode.
    """
    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[script] Live rendering frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")

    for frame in range(Config.FRAME_START, Config.FRAME_END + 1):
        scene.frame_set(frame)

        # CRITICAL: Force physics solve for this frame
        bpy.context.view_layer.update()

        if frame in Config.RENDER_FRAMES or frame == Config.FRAME_END:
            render_path = output_dir / f"render_{{frame:04d}}.png"
            scene.render.filepath = str(render_path)

            print(f"[script] Rendering frame {{frame}} with physics...")
            bpy.ops.render.render(write_still=True)
            print(f"[script] Saved: {{render_path}}")

    print("[script] Live render complete!")


def main():
    print("=" * 60)
    print("{title}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    obj = create_soft_body_object()
    ground = create_collision_ground()

    # Setup camera and lighting
    bpy.ops.object.camera_add(location=(5, -5, 3))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    setup_render_settings()

    if Config.RENDER:
        render_with_physics()
    else:
        print("[script] Skipping render (--render 0)")

    blend_path = Path(Config.OUTPUT_DIR) / "{filename_stem}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {{blend_path}}")


if __name__ == "__main__":
    main()
'''


# Cloth Template - Fabric, flags, curtains
CLOTH_TEMPLATE = '''
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
# Cloth Object
# =============================================================================

def create_cloth_object():
    """Create and configure cloth object."""
{object_geometry}
    cloth = bpy.context.active_object
    cloth.name = "ClothObject"

    # Subdivide for better draping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add cloth physics
    bpy.ops.object.modifier_add(type='CLOTH')
    settings = cloth.modifiers["Cloth"].settings

    # Quality settings
    settings.quality = {quality}
    settings.vertex_group_mass = ""

{cloth_settings}

    # Collision settings
    collision = cloth.modifiers["Cloth"].collision_settings
    collision.collision_quality = {collision_quality}
{collision_settings}

    return cloth


def create_collision_object():
    """Create object for cloth to drape over/interact with."""
{collision_geometry}
    collider = bpy.context.active_object
    collider.name = "ClothCollider"

    bpy.ops.object.modifier_add(type='COLLISION')
    collider.modifiers["Collision"].settings.thickness_outer = 0.02

    return collider


# =============================================================================
# Live Render Pattern
# =============================================================================

def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES
    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y


def render_with_physics():
    """Live render with per-frame physics."""
    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[script] Live rendering frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")

    for frame in range(Config.FRAME_START, Config.FRAME_END + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()  # Force physics solve

        if frame in Config.RENDER_FRAMES or frame == Config.FRAME_END:
            render_path = output_dir / f"render_{{frame:04d}}.png"
            scene.render.filepath = str(render_path)

            print(f"[script] Rendering frame {{frame}}...")
            bpy.ops.render.render(write_still=True)

    print("[script] Complete!")


def main():
    print("=" * 60)
    print("{title}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    cloth = create_cloth_object()
    collider = create_collision_object()

    # Camera and lighting
    bpy.ops.object.camera_add(location=(5, -5, 3))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))

    setup_render_settings()

    if Config.RENDER:
        render_with_physics()

    blend_path = Path(Config.OUTPUT_DIR) / "{filename_stem}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {{blend_path}}")


if __name__ == "__main__":
    main()
'''


# Rigid Body Template - Destruction, dominos, shatter effects
RIGID_BODY_TEMPLATE = '''
# =============================================================================
# Scene Setup
# =============================================================================

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_scene():
    """Configure scene settings and rigid body world."""
    scene = bpy.context.scene
    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END

    # Create rigid body world if not exists
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()

    scene.rigidbody_world.time_scale = 1.0
    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 10


# =============================================================================
# Rigid Body Objects
# =============================================================================

def create_active_rigid_body():
    """Create active rigid body object(s)."""
{object_geometry}
    obj = bpy.context.active_object
    obj.name = "RigidBodyActive"

    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.collision_shape = '{collision_shape}'

{rigid_body_settings}

    return obj


def create_passive_rigid_body():
    """Create ground/static rigid body."""
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -2))
    ground = bpy.context.active_object
    ground.name = "RigidBodyGround"

    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.collision_shape = 'MESH'

    return ground


{additional_objects}


# =============================================================================
# Live Render Pattern
# =============================================================================

def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = Config.RENDER_SAMPLES
    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y


def render_with_physics():
    """Live render with per-frame physics."""
    scene = bpy.context.scene
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[script] Live rendering frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")

    for frame in range(Config.FRAME_START, Config.FRAME_END + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        if frame in Config.RENDER_FRAMES or frame == Config.FRAME_END:
            render_path = output_dir / f"render_{{frame:04d}}.png"
            scene.render.filepath = str(render_path)

            print(f"[script] Rendering frame {{frame}}...")
            bpy.ops.render.render(write_still=True)

    print("[script] Complete!")


def main():
    print("=" * 60)
    print("{title}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    active = create_active_rigid_body()
    ground = create_passive_rigid_body()

    # Camera and lighting
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))

    setup_render_settings()

    if Config.RENDER:
        render_with_physics()

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

    # =========================================================================
    # MESH PHYSICS (Phase 2.5 - Simulation Type Extensibility)
    # =========================================================================
    elif is_mesh_physics(effect_lower):
        # Mesh-based physics (soft body, cloth, rigid body)
        # Uses live_render execution pattern instead of bake_export
        domain_type = "MESH"
        notes.append(f"Mesh physics: {effect_lower} (live_render pattern)")

        # Get recommended settings from effect registry
        recommended = get_recommended_settings(effect_lower)

        if effect_lower == "soft_body":
            domain_template = SOFT_BODY_TEMPLATE
            # Apply stability settings from registry
            step_min = recommended.get("step_min", 20)
            step_max = recommended.get("step_max", 100)
            damping = recommended.get("damping", 2.0)
            friction = recommended.get("friction", 0.5)
            goal_spring = recommended.get("goal_spring", 0.5)
            goal_friction = recommended.get("goal_friction", 0.5)

            notes.append(f"Soft body: step_min={step_min}, damping={damping}")

            # Placeholders for template formatting
            gas_settings = ""
            noise_settings = ""
            emission_keyframes = ""
            emitter_geometry = ""
            flow_settings = ""
            effector_code = ""
            domain_scale = 4.0

        elif effect_lower == "cloth":
            domain_template = CLOTH_TEMPLATE
            quality = recommended.get("quality", 10)
            collision_quality = recommended.get("collision_quality", 5)
            self_collision = recommended.get("self_collision", True)

            notes.append(f"Cloth: quality={quality}, self_collision={self_collision}")

            gas_settings = ""
            noise_settings = ""
            emission_keyframes = ""
            emitter_geometry = ""
            flow_settings = ""
            effector_code = ""
            domain_scale = 4.0

        elif effect_lower == "rigid_body":
            domain_template = RIGID_BODY_TEMPLATE
            collision_shape = recommended.get("collision_shape", "CONVEX_HULL")
            friction = recommended.get("friction", 0.5)
            bounciness = recommended.get("bounciness", 0.5)

            notes.append(f"Rigid body: collision_shape={collision_shape}")

            gas_settings = ""
            noise_settings = ""
            emission_keyframes = ""
            emitter_geometry = ""
            flow_settings = ""
            effector_code = ""
            domain_scale = 6.0

        else:
            # Fallback for unknown mesh physics type
            domain_template = SOFT_BODY_TEMPLATE
            notes.append(f"Unknown mesh physics '{effect_lower}', using soft_body")
            gas_settings = ""
            noise_settings = ""
            emission_keyframes = ""
            emitter_geometry = ""
            flow_settings = ""
            effector_code = ""
            domain_scale = 4.0

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

    # Determine output directory based on simulation type
    if domain_type == "MESH":
        output_dir_path = str(PROJECT_ROOT / "build/mesh_output" / output_name)
        title_suffix = "(Mesh Physics)"
    else:
        output_dir_path = str(PROJECT_ROOT / "build/vdb_output" / output_name)
        title_suffix = "(OpenVDB)"

    # Format the header
    header = SCRIPT_HEADER.format(
        title=f"GPT-5.2 — {output_name.replace('_', ' ').title()} {title_suffix} - Blender 5.0+",
        date=datetime.now().strftime("%Y-%m-%d"),
        template=template_name or f"built-in {effect_type}",
        description=description,
        filename=f"{output_name}.py",
        resolution=resolution,
        frame_start=frame_start,
        frame_end=frame_end,
        output_dir=output_dir_path,
        domain_scale=domain_scale,
        simulation_settings=sim_settings
    )

    # Format the body based on domain type
    if domain_type == "MESH":
        # =====================================================================
        # MESH PHYSICS TEMPLATES - Different placeholder format
        # =====================================================================
        recommended = get_recommended_settings(effect_lower)

        if effect_lower == "soft_body":
            # Default object geometry - can be customized based on description
            if "cube" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 2))"
            elif "sphere" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 2))"
            elif "torus" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.3, location=(0, 0, 2))"
            else:
                object_geometry = "    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(0, 0, 2))"

            soft_body_settings = """    # Goal settings (shape retention)
    settings.use_goal = True
    settings.goal_spring = {goal_spring}
    settings.goal_friction = {goal_friction}""".format(
                goal_spring=recommended.get("goal_spring", 0.5),
                goal_friction=recommended.get("goal_friction", 0.5)
            )

            body = domain_template.format(
                object_geometry=object_geometry,
                step_min=recommended.get("step_min", 20),
                step_max=recommended.get("step_max", 100),
                damping=recommended.get("damping", 2.0),
                friction=recommended.get("friction", 0.5),
                soft_body_settings=soft_body_settings
            )

        elif effect_lower == "cloth":
            # Default cloth geometry
            if "flag" in description.lower() or "banner" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 2))\n    bpy.ops.transform.resize(value=(0.5, 2.0, 1.0))"
            elif "cape" in description.lower() or "cloak" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, 0, 2))"
            else:
                object_geometry = "    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 2))"

            # Collision object geometry
            collision_geometry = "    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0.5))"

            cloth_settings = """    # Cloth presets
    settings.vertex_group_bending = ""
    settings.bending_stiffness = 0.5"""

            collision_settings = """    collision.use_self_collision = {self_collision}
    collision.self_distance_min = 0.015""".format(
                self_collision="True" if recommended.get("self_collision", True) else "False"
            )

            body = domain_template.format(
                object_geometry=object_geometry,
                quality=recommended.get("quality", 10),
                collision_quality=recommended.get("collision_quality", 5),
                cloth_settings=cloth_settings,
                collision_settings=collision_settings,
                collision_geometry=collision_geometry
            )

        elif effect_lower == "rigid_body":
            # Default rigid body geometry based on description keywords
            if "shatter" in description.lower() or "break" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 2))"
                additional_objects = """    # Add ground plane for collision
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type='PASSIVE')"""
            elif "domino" in description.lower():
                object_geometry = """    # Create first domino
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0.5))
    bpy.ops.transform.resize(value=(0.1, 0.3, 0.5))"""
                additional_objects = """    # Create domino chain
    for i in range(1, 10):
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=(i * 0.7, 0, 0.5))
        bpy.ops.transform.resize(value=(0.1, 0.3, 0.5))
        obj = bpy.context.active_object
        obj.name = f"Domino_{i}"
        bpy.ops.rigidbody.object_add(type='ACTIVE')

    # Add ground plane
    bpy.ops.mesh.primitive_plane_add(size=15, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type='PASSIVE')"""
            elif "pile" in description.lower() or "stack" in description.lower():
                object_geometry = "    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 2))"
                additional_objects = """    # Create additional objects for pile
    import random
    for i in range(20):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(2.5, 5)
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(x, y, z))
        obj = bpy.context.active_object
        obj.name = f"Block_{i}"
        bpy.ops.rigidbody.object_add(type='ACTIVE')

    # Add ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type='PASSIVE')"""
            else:
                object_geometry = "    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 2))"
                additional_objects = """    # Add ground plane for collision
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type='PASSIVE')"""

            # Build rigid body settings string
            rigid_body_settings = """    # Rigid body physical properties
    rb = obj.rigid_body
    rb.friction = {friction}
    rb.restitution = {bounciness}  # Bounciness
    rb.linear_damping = 0.04
    rb.angular_damping = 0.1""".format(
                friction=recommended.get("friction", 0.5),
                bounciness=recommended.get("bounciness", 0.5)
            )

            body = domain_template.format(
                object_geometry=object_geometry,
                collision_shape=recommended.get("collision_shape", "CONVEX_HULL"),
                rigid_body_settings=rigid_body_settings,
                additional_objects=additional_objects
            )

        else:
            # Fallback - use soft body template with defaults
            body = domain_template.format(
                object_geometry="    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(0, 0, 2))",
                step_min=20,
                step_max=100,
                damping=2.0,
                friction=0.5,
                soft_body_settings="    # Default settings"
            )

    else:
        # =====================================================================
        # VOLUMETRIC TEMPLATES (VDB output)
        # =====================================================================
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
    warnings_collected = []
    mitigations_applied = []

    # Phase 4.1: Mandatory warning checks before any modification
    # Check each parameter against the knowledge base
    for param_name, new_value in modifications.items():
        if param_name == "custom_code":
            continue  # Skip custom code, check individual params

        # Determine change type based on parameter name patterns
        change_type = "modify"
        if isinstance(new_value, (int, float)):
            # Try to detect if this is an increase or decrease
            # by checking current value in script
            current_match = re.search(
                rf'{param_name}\s*=\s*([\d.]+)',
                content,
                re.IGNORECASE
            )
            if current_match:
                try:
                    current_val = float(current_match.group(1))
                    if new_value > current_val:
                        change_type = "increase"
                    elif new_value < current_val:
                        change_type = "decrease"
                except (ValueError, TypeError):
                    pass

        # Check knowledge base for warnings
        warning_check = kb_check_before_modify(
            parameter=param_name,
            change_type=change_type,
            new_value=new_value
        )

        # Collect warnings
        if warning_check.warnings:
            warnings_collected.extend([
                f"[{param_name}] {w}" for w in warning_check.warnings
            ])

        # Handle critical warnings
        if warning_check.has_critical_warnings:
            # Add mitigation to response but still proceed
            # (orchestrator can decide to abort based on warnings)
            if warning_check.mitigation:
                mitigations_applied.append(
                    f"[{param_name}] {warning_check.mitigation}"
                )
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

    # Determine if there are critical warnings
    has_critical = any("MUST" in w.upper() or "NEVER" in w.upper() for w in warnings_collected)

    result = ScriptModification(
        success=True,
        original_path=str(path.relative_to(PROJECT_ROOT)),
        modified_path=str(output_path.relative_to(PROJECT_ROOT)),
        changes_made=changes_made,
        parameters_changed=params_changed,
        warnings=warnings_collected,
        mitigations=mitigations_applied,
        has_critical_warnings=has_critical
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


# =============================================================================
# Intelligent Technique Selection Tool (Phase 3 - Task 3.3)
# =============================================================================

@mcp.tool()
async def recommend_technique(
    effect_type: str,
    description: str,
    keyword_weight: float = 0.3,
    prefer_untried: bool = True
) -> str:
    """
    Recommend a technique using UCB1 algorithm for exploration/exploitation balance.

    Uses keyword filtering combined with Upper Confidence Bound (UCB1) algorithm
    to select techniques. Untried techniques get exploration bonus to ensure
    variety. Performance data is persisted across sessions for learning.

    UCB1 Formula: avg_reward + sqrt(2 * ln(total_trials) / trials)

    Args:
        effect_type: Type of effect (pyro, explosion, fire, etc.)
        description: Description of desired effect (used for keyword matching)
        keyword_weight: How much to weight keyword matches (0-1, default 0.3)
        prefer_untried: If True, untried techniques get exploration bonus (default True)

    Returns:
        JSON with:
        - technique_name: Recommended technique
        - confidence: 0-1 confidence in this recommendation
        - selection_reason: Why this technique was selected
        - ucb_score: Raw UCB1 score
        - alternatives: Top 3 alternative techniques with stats
        - keyword_matches: Number of keyword matches for selected technique
        - exploration_mode: True if this is an untried technique

    Example:
        recommend_technique(
            effect_type="pyro",
            description="A rising mushroom cloud explosion with bright orange flames"
        )
    """
    try:
        # Get recommendation using UCB1 algorithm
        recommendation = ucb1_recommend_technique(
            effect_type=effect_type,
            description=description,
            keyword_weight=keyword_weight,
            prefer_untried=prefer_untried
        )

        # Convert dataclass to dict for JSON serialization
        return json.dumps({
            "technique_name": recommendation.technique_name,
            "confidence": recommendation.confidence,
            "selection_reason": recommendation.selection_reason,
            "ucb_score": recommendation.ucb_score,
            "alternatives": recommendation.alternatives,
            "keyword_matches": recommendation.keyword_matches,
            "exploration_mode": recommendation.exploration_mode,
            "effect_type": effect_type,
            "description_preview": description[:100] + "..." if len(description) > 100 else description,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Technique recommendation failed: {str(e)}",
            "technique_name": "",
            "confidence": 0.0,
            "selection_reason": f"Error: {str(e)}",
            "ucb_score": 0.0,
            "alternatives": [],
            "keyword_matches": 0,
            "exploration_mode": False,
        }, indent=2)


@mcp.tool()
async def record_technique_outcome(
    technique_name: str,
    effect_type: str,
    success: bool,
    final_score: float,
    iterations: int = 1
) -> str:
    """
    Record the outcome of using a technique for learning.

    Call this after an asset generation session completes to update
    the technique performance statistics. This enables UCB1 to learn
    which techniques work best for different effect types.

    Args:
        technique_name: Name of the technique that was used
        effect_type: Type of effect (pyro, explosion, etc.)
        success: True if the technique passed quality thresholds
        final_score: Final quality score achieved (0-100)
        iterations: Number of iterations to pass (only relevant if success=True)

    Returns:
        JSON confirmation with updated statistics

    Example:
        record_technique_outcome(
            technique_name="rising_mushroom",
            effect_type="pyro",
            success=True,
            final_score=85.0,
            iterations=3
        )
    """
    try:
        store = TechniquePerformanceStore()

        if success:
            store.record_success(technique_name, effect_type, final_score, iterations)
            action = "success"
        else:
            store.record_failure(technique_name, effect_type, final_score)
            action = "failure"

        # Get updated stats
        perf = store.get(technique_name, effect_type)

        return json.dumps({
            "recorded": True,
            "action": action,
            "technique_name": technique_name,
            "effect_type": effect_type,
            "updated_stats": {
                "trials": perf.trials,
                "success_count": perf.success_count,
                "failure_count": perf.failure_count,
                "success_rate": round(perf.success_rate, 2),
                "avg_score": round(perf.avg_score, 1),
                "avg_iterations": round(perf.avg_iterations, 1) if perf.success_count > 0 else None,
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "recorded": False,
            "error": f"Failed to record outcome: {str(e)}",
        }, indent=2)


@mcp.tool()
async def get_technique_stats(
    effect_type: str = "pyro"
) -> str:
    """
    Get performance statistics for all techniques of an effect type.

    Useful for understanding which techniques have been tried and
    their success rates. Helps in debugging UCB1 selection behavior.

    Args:
        effect_type: Type of effect (pyro, explosion, etc.)

    Returns:
        JSON with performance stats for all techniques

    Example:
        get_technique_stats("pyro")
    """
    try:
        store = TechniquePerformanceStore()
        all_perfs = store.get_all_for_effect(effect_type)

        techniques_stats = []
        for perf in all_perfs:
            techniques_stats.append({
                "technique_name": perf.technique_name,
                "trials": perf.trials,
                "success_count": perf.success_count,
                "failure_count": perf.failure_count,
                "success_rate": round(perf.success_rate, 2),
                "avg_score": round(perf.avg_score, 1),
                "avg_iterations": round(perf.avg_iterations, 1) if perf.success_count > 0 else None,
                "last_used": perf.last_used,
            })

        # Sort by trials (most used first)
        techniques_stats.sort(key=lambda x: x["trials"], reverse=True)

        return json.dumps({
            "effect_type": effect_type,
            "total_techniques_tried": len(techniques_stats),
            "total_trials": sum(t["trials"] for t in techniques_stats),
            "overall_success_rate": round(
                sum(t["success_count"] for t in techniques_stats) /
                max(sum(t["trials"] for t in techniques_stats), 1),
                2
            ),
            "techniques": techniques_stats,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to get stats: {str(e)}",
            "effect_type": effect_type,
            "techniques": [],
        }, indent=2)


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
# Pre-Execution Validation Tool (Phase 2 - Task 2.1)
# =============================================================================

@mcp.tool()
async def validate_script(
    script_path: str,
    strict: bool = False
) -> str:
    """
    Validate a Blender script before execution.

    Performs comprehensive pre-execution validation:
    - Python syntax validation using AST
    - Parameter extraction and range checking
    - Required pattern detection (domain, flow, physics)
    - Security/safety checks
    - Output path validation

    This tool should be called after generate_script() and before
    blender-executor to catch issues early.

    Args:
        script_path: Path to the Blender Python script to validate
        strict: If True, treat warnings as errors (default False)

    Returns:
        JSON with validation result:
        - valid: True if script passes validation
        - script_path: Path that was validated
        - issues: List of validation issues found
        - extracted_params: Parameters detected in script
        - detected_effect_type: volumetric, mesh, or unknown
        - detected_simulation_pattern: bake_export or live_render
        - error_count: Number of errors
        - warning_count: Number of warnings

    Example:
        validate_script("assets/blender_scripts/generated/explosion_v1.py")
    """
    try:
        result = validate_script_file(script_path)

        # In strict mode, warnings become errors
        if strict and result.to_dict()["warning_count"] > 0:
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="strict_mode",
                message=f"Strict mode: {result.to_dict()['warning_count']} warnings treated as errors",
            ))

        return json.dumps(result.to_dict(), indent=2)

    except Exception as e:
        return json.dumps({
            "valid": False,
            "script_path": script_path,
            "issues": [{
                "severity": "error",
                "category": "validation_error",
                "message": f"Validation failed: {str(e)}",
            }],
            "extracted_params": {},
            "detected_effect_type": None,
            "detected_simulation_pattern": None,
            "error_count": 1,
            "warning_count": 0,
        }, indent=2)


@mcp.tool()
async def validate_script_inline(
    script_content: str,
    script_name: str = "<inline>",
    strict: bool = False
) -> str:
    """
    Validate Blender script content directly without a file.

    Useful for validating scripts before writing them to disk,
    or for validating script fragments.

    Args:
        script_content: The Python script content to validate
        script_name: Name for error messages (default "<inline>")
        strict: If True, treat warnings as errors (default False)

    Returns:
        JSON with validation result (same format as validate_script)

    Example:
        validate_script_inline('''
            import bpy
            bpy.ops.mesh.primitive_cube_add()
        ''')
    """
    try:
        result = validate_script_content(script_content, script_name)

        # In strict mode, warnings become errors
        if strict and result.to_dict()["warning_count"] > 0:
            result.valid = False

        return json.dumps(result.to_dict(), indent=2)

    except Exception as e:
        return json.dumps({
            "valid": False,
            "script_path": script_name,
            "issues": [{
                "severity": "error",
                "category": "validation_error",
                "message": f"Validation failed: {str(e)}",
            }],
            "extracted_params": {},
            "detected_effect_type": None,
            "detected_simulation_pattern": None,
            "error_count": 1,
            "warning_count": 0,
        }, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
