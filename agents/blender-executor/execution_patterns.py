"""
Blender Execution Patterns for Multi-Agent VFX Pipeline

Phase 2.5.2: Execution patterns for different simulation types.

Two primary patterns:
1. BAKE_EXPORT: For volumetric simulations (Mantaflow -> OpenVDB)
   - bpy.ops.fluid.bake_all() then export VDB files
   - Used for: pyro, explosion, fire, smoke, nebula, sun, liquid

2. LIVE_RENDER: For mesh physics (soft body, cloth, rigid body)
   - Frame-by-frame render with physics update
   - CRITICAL: bpy.context.view_layer.update() forces physics solve
   - Used for: soft_body, cloth, rigid_body

Problem Solved:
- bpy.ops.ptcache.bake_all() is unreliable in headless mode
- Mesh physics requires explicit view_layer.update() per frame
- Different physics types have different bake vs render requirements

References:
- MULTI_AGENT_IMPROVEMENT_PLAN_V2.md: Phase 2.5.2 specification
- GEMINI_FEEDBACK_ANALYSIS_AND_PLAN_AMENDMENTS.md: Live render pattern discovery
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum


class ExecutionPattern(Enum):
    """Execution pattern for running simulations."""
    BAKE_EXPORT = "bake_export"
    LIVE_RENDER = "live_render"
    AUTO = "auto"


# =============================================================================
# BAKE AND EXPORT PATTERN (Volumetric)
# =============================================================================

BAKE_EXPORT_TEMPLATE = '''
def run_bake_export(domain, output_dir: str, frame_start: int, frame_end: int):
    """
    Bake volumetric simulation and export VDB files.

    Pattern for: pyro, explosion, fire, smoke, nebula, sun, liquid
    Output: OpenVDB files (fluid_data_XXXX.vdb)

    Args:
        domain: Blender fluid domain object
        output_dir: Directory for VDB output
        frame_start: Start frame
        frame_end: End frame
    """
    import bpy
    from pathlib import Path

    print(f"[bake_export] Starting bake: frames {frame_start}-{frame_end}")
    print(f"[bake_export] Output: {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Configure cache settings
    settings = domain.modifiers["Fluid"].domain_settings
    settings.cache_directory = output_dir
    settings.cache_type = 'ALL'
    settings.cache_data_format = 'OPENVDB'

    # CRITICAL: Set cache frame range to match scene
    # This fixes the 250-frame default bug
    settings.cache_frame_start = frame_start
    settings.cache_frame_end = frame_end
    settings.cache_frame_offset = 0

    # Select domain and bake
    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()

    # Verify VDB files were created
    vdb_files = list(Path(output_dir).glob("fluid_data_*.vdb"))
    print(f"[bake_export] Bake complete: {len(vdb_files)} VDB files created")

    return {
        "success": len(vdb_files) > 0,
        "vdb_count": len(vdb_files),
        "output_dir": output_dir,
    }
'''


# =============================================================================
# LIVE RENDER PATTERN (Mesh Physics)
# =============================================================================

LIVE_RENDER_TEMPLATE = '''
def run_live_render_loop(output_dir: str, frame_start: int, frame_end: int):
    """
    Frame-by-frame rendering with live physics calculation.

    Pattern for: soft_body, cloth, rigid_body
    Output: PNG image sequence

    CRITICAL: bpy.context.view_layer.update() forces physics solve
    Without this call, mesh physics deformations won't be calculated.

    Args:
        output_dir: Directory for render output
        frame_start: Start frame
        frame_end: End frame
    """
    import bpy
    from pathlib import Path

    print(f"[live_render] Starting render: frames {frame_start}-{frame_end}")
    print(f"[live_render] Output: {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    rendered_files = []

    # Configure render settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    for frame in range(frame_start, frame_end + 1):
        # Set current frame
        scene.frame_set(frame)

        # CRITICAL: Force physics calculation
        # This is essential for soft body, cloth, and rigid body
        # Without this, the mesh won't deform
        bpy.context.view_layer.update()

        # Set output path for this frame
        render_path = Path(output_dir) / f"render_{frame:04d}.png"
        scene.render.filepath = str(render_path)

        # Render frame
        print(f"[live_render] Rendering frame {frame}/{frame_end}...")
        bpy.ops.render.render(write_still=True)

        rendered_files.append(str(render_path))

    print(f"[live_render] Complete: {len(rendered_files)} frames rendered")

    return {
        "success": len(rendered_files) == (frame_end - frame_start + 1),
        "frame_count": len(rendered_files),
        "output_dir": output_dir,
        "rendered_files": rendered_files,
    }
'''


# =============================================================================
# SOFT BODY SETUP PATTERN
# =============================================================================

SOFT_BODY_SETUP_TEMPLATE = '''
def setup_soft_body(obj, settings: dict = None):
    """
    Configure soft body physics with stable settings.

    Gemini 3 Pro discovered that default soft body settings cause
    jitter and explosions. This function applies stable defaults.

    CRITICAL SETTINGS (from Gemini 3 Pro):
    - step_min: 20 (default 5) - prevents jitter explosions
    - step_max: 100 (default 10) - adaptive stepping
    - damping: 2.0 (default 0.5) - prevents energy buildup

    Args:
        obj: Blender object to add soft body to
        settings: Optional override settings dict
    """
    import bpy

    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add soft body modifier if not present
    if "Soft Body" not in [m.name for m in obj.modifiers]:
        bpy.ops.object.modifier_add(type='SOFT_BODY')

    sb = obj.modifiers["Soft Body"].settings

    # STABLE DEFAULTS (from Gemini 3 Pro discoveries)
    # These prevent the common "jelly explosion" bug
    sb.step_min = 20       # Default is 5 - TOO LOW
    sb.step_max = 100      # Default is 10 - TOO LOW
    sb.damping = 2.0       # Default is 0.5 - TOO LOW
    sb.friction = 0.5

    # Goal settings (attachment to original shape)
    sb.goal_spring = 0.5
    sb.goal_friction = 0.5

    # Edge settings
    sb.use_edges = True
    sb.use_edge_collision = True
    sb.use_face_collision = True

    # Apply custom overrides
    if settings:
        for key, value in settings.items():
            if hasattr(sb, key):
                setattr(sb, key, value)

    print(f"[soft_body] Configured {obj.name} with stable settings")
    print(f"[soft_body] step_min={sb.step_min}, step_max={sb.step_max}, damping={sb.damping}")

    return sb
'''


# =============================================================================
# CLOTH SETUP PATTERN
# =============================================================================

CLOTH_SETUP_TEMPLATE = '''
def setup_cloth(obj, preset: str = "cotton", settings: dict = None):
    """
    Configure cloth physics simulation.

    Args:
        obj: Blender object (should be a subdivided plane/mesh)
        preset: Cloth preset ("cotton", "silk", "leather", "rubber")
        settings: Optional override settings dict
    """
    import bpy

    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add cloth modifier if not present
    if "Cloth" not in [m.name for m in obj.modifiers]:
        bpy.ops.object.modifier_add(type='CLOTH')

    cloth = obj.modifiers["Cloth"]
    cs = cloth.settings
    cc = cloth.collision_settings

    # Preset configurations
    PRESETS = {
        "cotton": {"mass": 0.3, "stiffness": 15.0, "damping": 5.0},
        "silk": {"mass": 0.1, "stiffness": 5.0, "damping": 0.5},
        "leather": {"mass": 0.4, "stiffness": 80.0, "damping": 25.0},
        "rubber": {"mass": 0.3, "stiffness": 25.0, "damping": 25.0},
    }

    preset_config = PRESETS.get(preset, PRESETS["cotton"])

    # Apply preset
    cs.mass = preset_config["mass"]
    cs.tension_stiffness = preset_config["stiffness"]
    cs.compression_stiffness = preset_config["stiffness"]
    cs.tension_damping = preset_config["damping"]

    # Quality settings
    cs.quality = 10

    # Collision settings
    cc.use_collision = True
    cc.use_self_collision = True
    cc.collision_quality = 5

    # Apply custom overrides
    if settings:
        for key, value in settings.items():
            if hasattr(cs, key):
                setattr(cs, key, value)
            elif hasattr(cc, key):
                setattr(cc, key, value)

    print(f"[cloth] Configured {obj.name} with '{preset}' preset")

    return cloth
'''


# =============================================================================
# RIGID BODY SETUP PATTERN
# =============================================================================

RIGID_BODY_SETUP_TEMPLATE = '''
def setup_rigid_body(obj, body_type: str = "ACTIVE", settings: dict = None):
    """
    Configure rigid body physics.

    Args:
        obj: Blender object
        body_type: "ACTIVE" (moves) or "PASSIVE" (collider)
        settings: Optional override settings dict
    """
    import bpy

    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add rigid body
    bpy.ops.rigidbody.object_add()

    rb = obj.rigid_body
    rb.type = body_type

    # Default settings
    rb.collision_shape = 'CONVEX_HULL'  # Good balance of accuracy/speed
    rb.friction = 0.5
    rb.restitution = 0.5  # Bounciness
    rb.use_margin = True
    rb.collision_margin = 0.001

    # Mass from volume (auto-calculate)
    rb.mass = 1.0

    # Apply custom overrides
    if settings:
        for key, value in settings.items():
            if hasattr(rb, key):
                setattr(rb, key, value)

    print(f"[rigid_body] Configured {obj.name} as {body_type}")

    return rb


def setup_cell_fracture(obj, source_limit: int = 100, recursion: int = 0):
    """
    Use Cell Fracture addon to shatter an object.

    Requires: Cell Fracture addon enabled

    Args:
        obj: Object to fracture
        source_limit: Maximum number of fragments
        recursion: Recursion depth for sub-fracturing
    """
    import bpy

    # Ensure Cell Fracture addon is enabled
    try:
        bpy.ops.preferences.addon_enable(module="object_fracture_cell")
    except Exception as e:
        print(f"[cell_fracture] Warning: Could not enable addon: {e}")

    # Select object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Run cell fracture
    bpy.ops.object.add_fracture_cell_objects(
        source_limit=source_limit,
        recursion=recursion,
        use_remove_original=True,
    )

    # Get fractured pieces
    fracture_pieces = [o for o in bpy.context.selected_objects]

    print(f"[cell_fracture] Created {len(fracture_pieces)} fragments from {obj.name}")

    return fracture_pieces
'''


# =============================================================================
# PREPROCESSING PATTERNS
# =============================================================================

PREPROCESSING_TEMPLATE = '''
def apply_voxel_remesh(obj, voxel_size: float = 0.05):
    """
    Apply voxel remesh modifier - required for soft body on joined primitives.

    Gemini 3 Pro discovered that joined primitives (like a rabbit made
    of spheres and cylinders) need voxel remesh before soft body,
    otherwise the mesh explodes.

    Args:
        obj: Object to remesh
        voxel_size: Voxel size (smaller = more detail, slower)
    """
    import bpy

    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add voxel remesh modifier
    bpy.ops.object.modifier_add(type='REMESH')
    remesh = obj.modifiers["Remesh"]
    remesh.mode = 'VOXEL'
    remesh.voxel_size = voxel_size
    remesh.use_smooth_shade = True

    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier="Remesh")

    print(f"[voxel_remesh] Applied to {obj.name} with voxel_size={voxel_size}")

    return obj
'''


# =============================================================================
# PATTERN SELECTION
# =============================================================================

def get_execution_pattern(effect_type: str) -> str:
    """
    Get the appropriate execution pattern for an effect type.

    Args:
        effect_type: Effect type name (e.g., "pyro", "soft_body")

    Returns:
        Pattern name: "bake_export" or "live_render"
    """
    # Import effect registry to check effect type
    try:
        from effect_registry import get_execution_pattern as get_pattern
        pattern = get_pattern(effect_type)
        return pattern.value
    except ImportError:
        pass

    # Fallback: hardcoded mappings
    VOLUMETRIC_TYPES = ["pyro", "explosion", "fire", "smoke", "nebula", "sun", "liquid"]
    MESH_TYPES = ["soft_body", "cloth", "rigid_body"]

    if effect_type.lower() in VOLUMETRIC_TYPES:
        return "bake_export"
    elif effect_type.lower() in MESH_TYPES:
        return "live_render"
    else:
        return "auto"


def get_pattern_template(pattern: str) -> str:
    """
    Get the code template for an execution pattern.

    Args:
        pattern: "bake_export" or "live_render"

    Returns:
        Python code template string
    """
    if pattern == "bake_export":
        return BAKE_EXPORT_TEMPLATE
    elif pattern == "live_render":
        return LIVE_RENDER_TEMPLATE
    else:
        # Default to live render for safety (works for all types)
        return LIVE_RENDER_TEMPLATE


def get_physics_setup_template(physics_type: str) -> Optional[str]:
    """
    Get the physics setup template for a physics type.

    Args:
        physics_type: "SOFT_BODY", "CLOTH", or "RIGID_BODY"

    Returns:
        Python code template string or None
    """
    TEMPLATES = {
        "SOFT_BODY": SOFT_BODY_SETUP_TEMPLATE,
        "CLOTH": CLOTH_SETUP_TEMPLATE,
        "RIGID_BODY": RIGID_BODY_SETUP_TEMPLATE,
    }
    return TEMPLATES.get(physics_type.upper())


# =============================================================================
# COMBINED SCRIPT GENERATION
# =============================================================================

def generate_execution_code(
    effect_type: str,
    output_dir: str,
    frame_start: int,
    frame_end: int,
    physics_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate complete execution code for an effect type.

    Args:
        effect_type: Effect type name
        output_dir: Output directory for results
        frame_start: Start frame
        frame_end: End frame
        physics_settings: Optional physics configuration

    Returns:
        Python code string to embed in Blender script
    """
    pattern = get_execution_pattern(effect_type)

    code_parts = []

    # Add appropriate execution pattern
    code_parts.append(get_pattern_template(pattern))

    # Add physics setup if needed
    if effect_type.lower() == "soft_body":
        code_parts.append(SOFT_BODY_SETUP_TEMPLATE)
        code_parts.append(PREPROCESSING_TEMPLATE)  # Voxel remesh
    elif effect_type.lower() == "cloth":
        code_parts.append(CLOTH_SETUP_TEMPLATE)
    elif effect_type.lower() == "rigid_body":
        code_parts.append(RIGID_BODY_SETUP_TEMPLATE)

    # Generate main execution call
    if pattern == "bake_export":
        main_call = f'''
# Execute bake and export
result = run_bake_export(domain, "{output_dir}", {frame_start}, {frame_end})
print(f"[main] Bake result: {{result}}")
'''
    else:
        main_call = f'''
# Execute live render loop
result = run_live_render_loop("{output_dir}", {frame_start}, {frame_end})
print(f"[main] Render result: {{result}}")
'''

    code_parts.append(main_call)

    return "\n\n".join(code_parts)


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("=== Execution Patterns ===\n")

    test_types = ["pyro", "soft_body", "cloth", "rigid_body", "explosion", "custom"]

    for effect_type in test_types:
        pattern = get_execution_pattern(effect_type)
        print(f"{effect_type}: {pattern}")

    print("\n=== Generated Code Example ===")
    code = generate_execution_code(
        effect_type="soft_body",
        output_dir="/tmp/test_output",
        frame_start=1,
        frame_end=50,
    )
    print(code[:500] + "...")
