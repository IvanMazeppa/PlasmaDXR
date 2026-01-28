"""
Blender 5.0 API Fixer.

Pre-execution validation and auto-fix for known-bad API patterns.
This breaks the infinite loop caused by Coordinator diagnoses that
_modify_script_impl cannot handle.

The pattern:
1. Script fails with deprecated/removed API call
2. Coordinator correctly diagnoses the issue
3. _modify_script_impl can only change Config class params
4. Fix never applied → same error → infinite loop

This module fixes known-bad patterns BEFORE execution.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Known-bad patterns and their fixes for Blender 5.0
# Format: (pattern_regex, replacement, description)
BLENDER_50_FIXES: List[Tuple[str, str, str]] = [
    # ColorRamp.elements.clear() was removed in Blender 5.0
    (
        r"(\w+\.color_ramp\.elements)\.clear\(\)",
        r"""# Blender 5.0: elements.clear() removed - remove all but first element
while len(\1) > 1:
    \1.remove(\1[0])""",
        "ColorRamp.elements.clear() → while loop removal"
    ),

    # use_nodes deprecated, use tree instead
    (
        r"(\w+)\.use_nodes\s*=\s*True",
        r"\1.use_nodes = True  # Deprecated in 5.0, but still works",
        "use_nodes deprecation warning"
    ),

    # lamp → light (old API)
    (
        r"bpy\.types\.Lamp",
        r"bpy.types.Light",
        "bpy.types.Lamp → bpy.types.Light"
    ),
    (
        r"\.lamp\b",
        r".light",
        ".lamp → .light"
    ),

    # render.image_settings.color_mode changes
    (
        r"color_mode\s*=\s*['\"]BW['\"]",
        r"color_mode = 'RGBA'  # BW removed in 5.0, use RGBA with compositor",
        "color_mode BW → RGBA"
    ),

    # bpy.ops.import_scene.obj renamed
    (
        r"bpy\.ops\.import_scene\.obj\(",
        r"bpy.ops.wm.obj_import(",
        "import_scene.obj → wm.obj_import"
    ),

    # bpy.ops.export_scene.obj renamed
    (
        r"bpy\.ops\.export_scene\.obj\(",
        r"bpy.ops.wm.obj_export(",
        "export_scene.obj → wm.obj_export"
    ),

    # Principled BSDF input name changes in 4.0+
    (
        r"\.inputs\[['\"]Subsurface['\"]",
        r".inputs['Subsurface Weight'",
        "BSDF Subsurface → Subsurface Weight"
    ),
    (
        r"\.inputs\[['\"]Transmission['\"]",
        r".inputs['Transmission Weight'",
        "BSDF Transmission → Transmission Weight"
    ),

    # Cycles samples path changes
    (
        r"scene\.cycles\.progressive\s*=",
        r"# progressive removed in 4.0+\n# scene.cycles.progressive =",
        "cycles.progressive removed"
    ),

    # Object.select deprecated → Object.select_set()
    (
        r"(\w+)\.select\s*=\s*True",
        r"\1.select_set(True)",
        "obj.select = True → obj.select_set(True)"
    ),
    (
        r"(\w+)\.select\s*=\s*False",
        r"\1.select_set(False)",
        "obj.select = False → obj.select_set(False)"
    ),

    # scene.update() removed → depsgraph.update()
    (
        r"bpy\.context\.scene\.update\(\)",
        r"bpy.context.view_layer.depsgraph.update()",
        "scene.update() → depsgraph.update()"
    ),

    # FluidDomainSettings.use_dissolve does NOT EXIST - correct is use_dissolve_smoke
    (
        r"\.use_dissolve\s*=",
        r".use_dissolve_smoke =",
        "use_dissolve → use_dissolve_smoke (typo fix)"
    ),
    # Fluid domain adaptive flag renamed
    (
        r"\.adaptive_domain\s*=",
        r".use_adaptive_domain =",
        "adaptive_domain → use_adaptive_domain"
    ),
    # Fix accidental double-prefix from corrections
    (
        r"\.use_use_adaptive_domain\s*=",
        r".use_adaptive_domain =",
        "use_use_adaptive_domain → use_adaptive_domain"
    ),
    # Invalid flow behavior enum (GAS -> INFLOW)
    (
        r"flow_behavior\s*=\s*['\"]GAS['\"]",
        r"flow_behavior = 'INFLOW'",
        "flow_behavior GAS → INFLOW"
    ),
    # Invalid flow type enum (GAS -> FIRE)
    (
        r"flow_type\s*=\s*['\"]GAS['\"]",
        r"flow_type = 'FIRE'",
        "flow_type GAS → FIRE"
    ),

    # Fluid behavior constants changed
    (
        r"flow_behavior\s*=\s*['\"]INFLOW_OUTFLOW['\"]",
        r"flow_behavior = 'INFLOW'",
        "INFLOW_OUTFLOW → INFLOW"
    ),
    # FluidDomainSettings.use_caching removed in Blender 5.0
    (
        r"(?m)^.*\.use_caching\s*=\s*.*$",
        r"# Blender 5.0: use_caching removed (line deleted)",
        "use_caching removed"
    ),
    # FluidDomainSettings.noise_res_factor removed in Blender 5.0
    (
        r"(?m)^.*\.noise_res_factor\s*=\s*.*$",
        r"# Blender 5.0: noise_res_factor removed (line deleted)",
        "noise_res_factor removed"
    ),
    # bpy.ops.fluid.free_all() fails on fresh scenes with "grids still in use"
    # Remove this call entirely - just bake directly
    (
        r"(?m)^\s*bpy\.ops\.fluid\.free_all\(\)\s*#?.*$",
        r"# Blender 5.0: free_all() removed - causes 'grids still in use' on fresh scenes",
        "free_all() removed (grids in use error)"
    ),
    # Also catch the with context override version
    (
        r"(?m)^\s*with\s+bpy\.context\.temp_override.*:\s*\n\s*bpy\.ops\.fluid\.free_all\(\)",
        r"# Blender 5.0: free_all() removed - causes 'grids still in use' on fresh scenes",
        "free_all() with override removed"
    ),
    # FluidDomainSettings.resolution_divisions does NOT exist - use resolution_max
    (
        r"\.resolution_divisions\b",
        r".resolution_max",
        "resolution_divisions → resolution_max"
    ),
    (
        r"['\"]resolution_divisions['\"]",
        r"'resolution_max'",
        "resolution_divisions key → resolution_max"
    ),
    # LLM TYPO: use_adaptive_time_steps → use_adaptive_timesteps (no underscore between time/steps)
    (
        r"\.use_adaptive_time_steps\b",
        r".use_adaptive_timesteps",
        "use_adaptive_time_steps → use_adaptive_timesteps (typo fix)"
    ),
    (
        r"['\"]use_adaptive_time_steps['\"]",
        r"'use_adaptive_timesteps'",
        "use_adaptive_time_steps key → use_adaptive_timesteps (typo fix)"
    ),
    # LLM TRUNCATION: cache_format → cache_data_format (missing 'data' part)
    (
        r"\.cache_format\b",
        r".cache_data_format",
        "cache_format → cache_data_format (typo fix)"
    ),
    (
        r"['\"]cache_format['\"]",
        r"'cache_data_format'",
        "cache_format key → cache_data_format (typo fix)"
    ),
    # LLM HALLUCINATION: FluidFlowSettings.velocity does NOT exist
    # Correct attributes: velocity_factor, velocity_normal, velocity_random, use_initial_velocity
    # Note: Use negative lookbehind to avoid double-applying (velocity_factor → velocity_factor_factor)
    (
        r"(flow_settings|flow)\.velocity(?!_)\s*=",
        r"\1.velocity_factor =",
        "velocity → velocity_factor (FluidFlowSettings)"
    ),
    # LLM HALLUCINATION: velocity_multi does NOT exist
    (
        r"\.velocity_multi\s*=",
        r".velocity_factor =",
        "velocity_multi → velocity_factor (FluidFlowSettings)"
    ),
    # Type fix: noise_scale must be int, not float
    # Match patterns like: noise_scale = 1.0, noise_scale = 2.0, etc.
    (
        r"(\.noise_scale\s*=\s*)(\d+)\.0\b",
        r"\1\2  # Must be int, not float",
        "noise_scale float → int"
    ),
    # P0 FIX: Volume shader density too low - boost from 3.0 to 10.0
    # This fixes "grey sphere" renders where the volume is too transparent
    (
        r"(multiply\.inputs\[1\]\.default_value\s*=\s*)([0-5]\.0)",
        r"\g<1>10.0  # Boosted from \2 for visibility",
        "Volume density multiplier boosted (grey sphere fix)"
    ),
    # P0 FIX: Emission strength too low for fire visibility
    (
        r"(bb_emission\.inputs\['Strength'\]\.default_value\s*=\s*)([0-5]\.0)",
        r"\g<1>20.0  # Boosted from \2 for fire visibility",
        "Emission strength boosted (fire visibility fix)"
    ),
    # P0 FIX: Blackbody intensity too low
    (
        r"(inputs\['Blackbody Intensity'\]\.default_value\s*=\s*)([0-3]\.0)",
        r"\g<1>8.0  # Boosted from \2 for fire glow",
        "Blackbody intensity boosted (fire glow fix)"
    ),

    # Render only representative frames for quality evaluation, not full animation
    (
        r"(frames\s*=\s*)range\(\s*FRAME_START\s*,\s*FRAME_END\s*\+\s*1\s*\)",
        r"\1[min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames instead of full animation"
    ),

    # Mantaflow cache_type REPLAY doesn't produce full bake data - must be ALL
    (
        r"cache_type\s*=\s*['\"]REPLAY['\"]",
        r"cache_type = 'ALL'  # API Fixer: REPLAY doesn't produce full bake data",
        "cache_type REPLAY → ALL (full bake data)"
    ),
]


# =============================================================================
# VOLUME MATERIAL INJECTION (P0 FIX - Grey Sphere Problem)
# =============================================================================
# Without a volume shader on the domain, Cycles renders the mesh geometry
# instead of the smoke/fire volume, resulting in a grey sphere.

VOLUME_MATERIAL_SNIPPET = '''
# ==== API FIXER: Volume Material Setup (Mantaflow) ====
# Without this, renders show grey mesh instead of smoke/fire
def _api_fixer_setup_volume_material(domain_obj, effect_type="SMOKE"):
    """Ensure domain has a volume shader for rendering smoke/fire."""
    mat_name = f"{domain_obj.name}_VolumeMaterial"

    # Check if material already exists
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes based on effect type
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)

        if effect_type in ("FIRE", "BOTH"):
            # Fire/smoke: Principled Volume with blackbody emission
            volume = nodes.new('ShaderNodeVolumePrincipled')
            volume.location = (0, 0)
            volume.inputs['Density'].default_value = 5.0
            volume.inputs['Anisotropy'].default_value = 0.3
            volume.inputs['Blackbody Intensity'].default_value = 1.0
            volume.inputs['Blackbody Tint'].default_value = (1.0, 0.8, 0.5, 1.0)

            # Connect density and flame attributes
            attr_density = nodes.new('ShaderNodeAttribute')
            attr_density.location = (-400, 100)
            attr_density.attribute_name = 'density'
            attr_density.attribute_type = 'GEOMETRY'

            attr_flame = nodes.new('ShaderNodeAttribute')
            attr_flame.location = (-400, -100)
            attr_flame.attribute_name = 'flame'
            attr_flame.attribute_type = 'GEOMETRY'

            # Multiply density
            multiply = nodes.new('ShaderNodeMath')
            multiply.location = (-200, 100)
            multiply.operation = 'MULTIPLY'
            multiply.inputs[1].default_value = 5.0

            links.new(attr_density.outputs['Fac'], multiply.inputs[0])
            links.new(multiply.outputs['Value'], volume.inputs['Density'])
            links.new(attr_flame.outputs['Fac'], volume.inputs['Blackbody Intensity'])
            links.new(volume.outputs['Volume'], output.inputs['Volume'])

        else:
            # Smoke only: simpler volume scatter
            volume = nodes.new('ShaderNodeVolumePrincipled')
            volume.location = (0, 0)
            volume.inputs['Density'].default_value = 5.0
            volume.inputs['Anisotropy'].default_value = 0.3
            volume.inputs['Blackbody Intensity'].default_value = 0.0

            attr_density = nodes.new('ShaderNodeAttribute')
            attr_density.location = (-400, 100)
            attr_density.attribute_name = 'density'
            attr_density.attribute_type = 'GEOMETRY'

            multiply = nodes.new('ShaderNodeMath')
            multiply.location = (-200, 100)
            multiply.operation = 'MULTIPLY'
            multiply.inputs[1].default_value = 5.0

            links.new(attr_density.outputs['Fac'], multiply.inputs[0])
            links.new(multiply.outputs['Value'], volume.inputs['Density'])
            links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # Assign material to domain
    if domain_obj.data.materials:
        domain_obj.data.materials[0] = mat
    else:
        domain_obj.data.materials.append(mat)

    return mat

# Find domain object and apply volume material
_api_fixer_domain = None
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'fluid_type') and mod.fluid_type == 'DOMAIN':
            _api_fixer_domain = obj
            break
    if _api_fixer_domain:
        break

if _api_fixer_domain:
    # Determine effect type from flow settings
    _api_fixer_effect_type = "SMOKE"
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and hasattr(mod, 'fluid_type') and mod.fluid_type == 'FLOW':
                if hasattr(mod, 'flow_settings') and mod.flow_settings:
                    ft = getattr(mod.flow_settings, 'flow_type', 'SMOKE')
                    if ft in ('FIRE', 'BOTH'):
                        _api_fixer_effect_type = ft
                        break
    _api_fixer_setup_volume_material(_api_fixer_domain, _api_fixer_effect_type)
    print(f"[API Fixer] Applied volume material to {_api_fixer_domain.name} ({_api_fixer_effect_type})")
# ==== END API FIXER: Volume Material Setup ====
'''


ANIMATION_TO_STILLS_SNIPPET = '''
# ==== API FIXER: Animation → Representative Still Renders ====
# bpy.ops.render.render(animation=True) does NOT write individual frame files in
# headless mode. Replace with per-frame still renders of 3 representative frames.
_api_fixer_scene = bpy.context.scene
_api_fixer_fs = _api_fixer_scene.frame_start
_api_fixer_fe = _api_fixer_scene.frame_end
_api_fixer_base_path = _api_fixer_scene.render.filepath
_api_fixer_frames = [min(_api_fixer_fs + 5, _api_fixer_fe), (_api_fixer_fs + _api_fixer_fe) // 2, _api_fixer_fe]
for _api_fixer_f in _api_fixer_frames:
    _api_fixer_scene.frame_set(_api_fixer_f)
    _api_fixer_scene.render.filepath = f"{_api_fixer_base_path}{_api_fixer_f:04d}"
    bpy.ops.render.render(write_still=True)
    print(f"[API Fixer] Rendered frame {_api_fixer_f}")
_api_fixer_scene.render.filepath = _api_fixer_base_path
# ==== END API FIXER: Animation → Representative Still Renders ====
'''


BAKE_FRAME_ALIGNMENT_SNIPPET = '''
# ==== API FIXER: Bake Frame Range Alignment ====
# Mantaflow cache_frame_start/cache_frame_end default to 1-120 independent of scene.frame_end.
# Mismatched ranges waste bake time or cause missing frames.
_api_fixer_scene = bpy.context.scene
for _api_fixer_obj in bpy.data.objects:
    for _api_fixer_mod in _api_fixer_obj.modifiers:
        if _api_fixer_mod.type == 'FLUID' and hasattr(_api_fixer_mod, 'fluid_type') and _api_fixer_mod.fluid_type == 'DOMAIN':
            _ds = _api_fixer_mod.domain_settings
            _ds.cache_frame_start = _api_fixer_scene.frame_start
            _ds.cache_frame_end = _api_fixer_scene.frame_end
            print(f"[API Fixer] Aligned bake range: {_ds.cache_frame_start}-{_ds.cache_frame_end} (scene: {_api_fixer_scene.frame_start}-{_api_fixer_scene.frame_end})")
            break
# ==== END API FIXER: Bake Frame Range Alignment ====
'''


def _inject_animation_to_stills(content: str) -> tuple[str, bool]:
    """
    Replace bpy.ops.render.render(animation=True) with per-frame still renders.

    In headless Blender, animation=True renders frames internally but does NOT
    write individual image files that the pipeline can find for quality evaluation.
    This replaces the call with a loop that renders 3 representative frames using
    write_still=True.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Match render(animation=True) with optional extra kwargs
    animation_match = re.search(
        r"^(\s*)bpy\.ops\.render\.render\([^)]*animation\s*=\s*True[^)]*\).*$",
        content,
        re.MULTILINE
    )

    if animation_match and "API FIXER: Animation" not in content:
        indent = animation_match.group(1)
        indented_snippet = "\n".join(
            indent + line if line.strip() else line
            for line in ANIMATION_TO_STILLS_SNIPPET.split("\n")
        )
        content = (
            content[:animation_match.start()]
            + indented_snippet
            + content[animation_match.end():]
        )
        return content, True

    return content, False


def _inject_bake_frame_alignment(content: str) -> tuple[str, bool]:
    """
    Inject bake frame range alignment if script uses Mantaflow and baking.

    Ensures cache_frame_start/cache_frame_end match scene.frame_start/frame_end
    so the bake doesn't waste time on 120 default frames when scene is 25 frames.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script uses Mantaflow domain
    has_fluid_domain = re.search(
        r"fluid_type\s*=\s*['\"]DOMAIN['\"]|type\s*=\s*['\"]FLUID['\"]",
        content
    ) is not None

    # Check if script has a bake call
    has_bake = re.search(
        r"bpy\.ops\.fluid\.bake_data|bpy\.ops\.fluid\.bake_all",
        content
    ) is not None

    # Check if already aligned
    already_aligned = re.search(
        r"cache_frame_start|cache_frame_end|Bake Frame Range Alignment",
        content
    ) is not None

    if has_fluid_domain and has_bake and not already_aligned:
        # Inject before the first bake call
        bake_match = re.search(
            r"^(\s*)(bpy\.ops\.fluid\.bake_data|bpy\.ops\.fluid\.bake_all)",
            content,
            re.MULTILINE
        )
        if bake_match:
            insert_pos = bake_match.start()
            indent = bake_match.group(1)
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in BAKE_FRAME_ALIGNMENT_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True

    return content, False


def _inject_volume_material_setup(content: str) -> tuple[str, bool]:
    """
    Inject volume material setup if script uses Mantaflow but lacks volume shader.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script uses Mantaflow domain
    has_fluid_domain = re.search(
        r"fluid_type\s*=\s*['\"]DOMAIN['\"]|type\s*=\s*['\"]FLUID['\"]",
        content
    ) is not None

    # Check if domain is LIQUID — liquid domains render via mesh surface, not volume shader.
    # Volume material injection would overwrite the correct water/glass material.
    is_liquid_domain = re.search(
        r"domain_type\s*[=,]\s*['\"]LIQUID['\"]",
        content
    ) is not None

    # Check if script already has volume material setup
    has_volume_material = re.search(
        r"ShaderNodeVolumePrincipled|ShaderNodeVolumeScatter|ShaderNodeVolumeAbsorption|"
        r"Volume\s*Scatter|Volume\s*Absorption|VolumeMaterial|_setup_volume_material",
        content,
        re.IGNORECASE
    ) is not None

    # Check if there's a render call (otherwise no point adding material)
    has_render = re.search(r"bpy\.ops\.render\.render", content) is not None

    if has_fluid_domain and not is_liquid_domain and not has_volume_material and has_render:
        # Find injection point: after bpy.ops.fluid.bake or before bpy.ops.render.render
        # Prefer injecting before render call
        render_match = re.search(
            r"^(\s*)(scene\.render\.filepath\s*=|bpy\.ops\.render\.render)",
            content,
            re.MULTILINE
        )

        if render_match:
            insert_pos = render_match.start()
            indent = render_match.group(1)
            # Indent the snippet
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in VOLUME_MATERIAL_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True
        else:
            # Fallback: append at the end before any final print statements
            content = content.rstrip() + "\n\n" + VOLUME_MATERIAL_SNIPPET
            return content, True

    return content, False


def validate_and_fix_script(script_path: str) -> Dict:
    """
    Validate script for known-bad Blender 5.0 API patterns and auto-fix.

    Args:
        script_path: Path to the Blender Python script

    Returns:
        Dict with:
            - fixed: bool - whether any fixes were applied
            - fixes_applied: List[str] - descriptions of fixes
            - original_path: str - original script path
            - fixed_path: str - path to fixed script (same if no fixes)
    """
    path = Path(script_path)
    if not path.exists():
        return {
            "fixed": False,
            "fixes_applied": [],
            "original_path": str(path),
            "fixed_path": str(path),
            "error": f"Script not found: {path}"
        }

    content = path.read_text()
    original_content = content
    fixes_applied = []

    for pattern, replacement, description in BLENDER_50_FIXES:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            fixes_applied.append(description)

    # Camera safety fix: add camera if rendering but no camera setup exists
    has_render_call = re.search(r"bpy\.ops\.render\.render", content) is not None
    has_camera_setup = re.search(
        r"(scene\.camera|bpy\.context\.scene\.camera|camera_add)",
        content
    ) is not None

    if has_render_call and not has_camera_setup:
        camera_snippet = (
            "\n# API Fixer: Ensure camera exists for rendering\n"
            "scene = bpy.context.scene\n"
            "if scene.camera is None:\n"
            "    bpy.ops.object.camera_add(location=(6, -6, 4))\n"
            "    cam = bpy.context.active_object\n"
            "    cam.rotation_euler = (1.1, 0, 0.8)\n"
            "    scene.camera = cam\n"
        )

        insert_pos = None
        scene_match = re.search(r"^scene\s*=\s*bpy\.context\.scene.*$", content, re.MULTILINE)
        if scene_match:
            insert_pos = scene_match.end()
        else:
            import_match = re.search(r"^import bpy.*$", content, re.MULTILINE)
            if import_match:
                insert_pos = import_match.end()

        if insert_pos is not None:
            content = content[:insert_pos] + camera_snippet + content[insert_pos:]
        else:
            content = camera_snippet + "\n" + content

        fixes_applied.append("Added camera setup before render (missing scene.camera)")

    # P0 FIX: Inject volume material for Mantaflow domains (fixes grey sphere)
    content, volume_fixed = _inject_volume_material_setup(content)
    if volume_fixed:
        fixes_applied.append("Injected volume material for Mantaflow domain (fixes grey sphere)")

    # P1 FIX: Align bake frame range with scene frame range
    content, bake_fixed = _inject_bake_frame_alignment(content)
    if bake_fixed:
        fixes_applied.append("Aligned bake cache frame range with scene frame range")

    # P0 FIX: Replace animation=True with per-frame still renders (headless compat)
    content, stills_fixed = _inject_animation_to_stills(content)
    if stills_fixed:
        fixes_applied.append("Replaced animation=True with representative still renders (headless fix)")

    if fixes_applied:
        # Write fixed content back
        path.write_text(content)
        print(f"[API Fixer] Applied {len(fixes_applied)} fixes to {path.name}")
        for fix in fixes_applied:
            print(f"  - {fix}")

        return {
            "fixed": True,
            "fixes_applied": fixes_applied,
            "original_path": str(path),
            "fixed_path": str(path),
            "content_changed": True
        }
    else:
        return {
            "fixed": False,
            "fixes_applied": [],
            "original_path": str(path),
            "fixed_path": str(path),
            "content_changed": False
        }


def check_for_known_issues(script_path: str) -> List[str]:
    """
    Check script for known-bad patterns WITHOUT fixing.

    Args:
        script_path: Path to the Blender Python script

    Returns:
        List of issue descriptions found
    """
    path = Path(script_path)
    if not path.exists():
        return [f"Script not found: {path}"]

    content = path.read_text()
    issues = []

    for pattern, _, description in BLENDER_50_FIXES:
        if re.search(pattern, content):
            issues.append(description)

    return issues


def add_custom_fix(pattern: str, replacement: str, description: str) -> None:
    """
    Add a custom fix pattern at runtime.

    Useful when the Coordinator discovers a new issue that should be
    automatically fixed in future runs.

    Args:
        pattern: Regex pattern to match
        replacement: Replacement string (can use \\1, \\2 for groups)
        description: Human-readable description
    """
    BLENDER_50_FIXES.append((pattern, replacement, description))
    print(f"[API Fixer] Added custom fix: {description}")


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blender_api_fixer.py <script_path> [--check-only]")
        sys.exit(1)

    script_path = sys.argv[1]
    check_only = "--check-only" in sys.argv

    if check_only:
        issues = check_for_known_issues(script_path)
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No known issues found")
    else:
        result = validate_and_fix_script(script_path)
        if result["fixed"]:
            print(f"Fixed {len(result['fixes_applied'])} issues")
        else:
            print("No fixes needed")
