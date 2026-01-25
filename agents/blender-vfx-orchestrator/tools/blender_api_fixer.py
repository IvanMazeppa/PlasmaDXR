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
]


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
