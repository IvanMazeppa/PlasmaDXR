"""
Function tool wrappers for script generation.

IN-PROCESS IMPLEMENTATION - Does NOT spawn MCP subprocesses.

Exposes script-generator capabilities to OpenAI Agents SDK agents:
- generate_script: Create new Blender script
- modify_script: Improve script based on feedback
- validate_script: Pre-execution validation
- list_techniques: Show available techniques
- recommend_technique: UCB1-based selection
- record_technique_outcome: Learning feedback

This uses the two-layer pattern:
- _impl functions: Plain functions with actual logic (for internal use)
- @function_tool wrappers: Exposed to agents, call the _impl functions

IMPORTANT: This imports directly from the script-generator modules rather than
spawning MCP subprocesses, avoiding anyio TaskGroup conflicts.
"""

from __future__ import annotations

import json
import os
import sys
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from agents import (
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
    ToolGuardrailFunctionOutput,
)


# =============================================================================
# PATH SETUP - Add script-generator to Python path
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ORCHESTRATOR_ROOT.parent.parent  # agents -> PlasmaDXR

# Add script-generator to path for imports
SCRIPT_GENERATOR_DIR = PROJECT_ROOT / "agents/script-generator"
if str(SCRIPT_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_GENERATOR_DIR))

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "assets/blender_scripts/generated"

# Delete sentinel for structured modifications
DELETE_SENTINEL = "__DELETE__"

# Known hallucination patterns to block before execution
HALLUCINATION_PATTERNS = [
    (r"\bresolution_divisions\b", "resolution_divisions removed in Blender 5.0 (use resolution_max)"),
    (r"\buse_adaptive_time_steps\b", "use_adaptive_time_steps is invalid (use use_adaptive_timesteps)"),
    (r"\bvelocity_multi\b", "velocity_multi is invalid (use velocity_factor)"),
    (r"\bnoise_res_factor\b", "noise_res_factor removed in Blender 5.0"),
    (r"\btime_scale\b", "time_scale removed in Blender 5.0"),
    (r"\bdomain_resolution\b\s*=", "domain_resolution is read-only (use resolution_max)"),
    (r"\bflow\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
    (r"\bobject\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
    (r"\bobj\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
]


def _scan_for_hallucinated_api(script_text: str) -> List[str]:
    """Return list of hallucination issues found in script text."""
    issues: List[str] = []
    if not script_text:
        return issues

    in_triple = False
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if '"""' in stripped or "'''" in stripped:
            # Toggle triple-quote state when encountering docstrings
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_triple = not in_triple
            continue
        if in_triple or stripped.startswith("#"):
            continue

        for pattern, message in HALLUCINATION_PATTERNS:
            if re.search(pattern, line):
                issues.append(f"{message} | line: {stripped}")
    return issues


def _parse_tool_arguments(data: Any) -> Dict[str, Any]:
    """Parse tool arguments from guardrail context."""
    raw = getattr(getattr(data, "context", None), "tool_arguments", None)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


@tool_input_guardrail
def guard_write_script_input(data: Any) -> ToolGuardrailFunctionOutput:
    """Block write_script() if generated code includes known hallucinations."""
    args = _parse_tool_arguments(data)
    code = args.get("code", "")
    issues = _scan_for_hallucinated_api(code)
    if issues:
        return ToolGuardrailFunctionOutput.reject_content(
            "Blocked write_script due to hallucinated APIs: " + "; ".join(issues)
        )
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def guard_write_script_output(data: Any) -> ToolGuardrailFunctionOutput:
    """Validate written script file before allowing pipeline to proceed."""
    output = getattr(data, "output", None)
    script_path = None
    if isinstance(output, dict):
        script_path = output.get("script_path")
    elif isinstance(output, str):
        try:
            payload = json.loads(output)
            script_path = payload.get("script_path")
        except Exception:
            script_path = None

    if script_path:
        try:
            text = Path(script_path).read_text()
            issues = _scan_for_hallucinated_api(text)
            if issues:
                return ToolGuardrailFunctionOutput.reject_content(
                    "Script contains hallucinated APIs: " + "; ".join(issues)
                )
        except Exception:
            # If we can't read, allow but surface later via validate_script
            return ToolGuardrailFunctionOutput.allow()

    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def guard_modify_script_output(data: Any) -> ToolGuardrailFunctionOutput:
    """Validate modified script file before allowing pipeline to proceed."""
    output = getattr(data, "output", None)
    modified_path = None
    payload = None
    if isinstance(output, dict):
        payload = output
        modified_path = output.get("modified_path")
    elif isinstance(output, str):
        try:
            payload = json.loads(output)
            modified_path = payload.get("modified_path")
        except Exception:
            modified_path = None

    if payload and payload.get("success") and not payload.get("changes_made"):
        return ToolGuardrailFunctionOutput.reject_content(
            "modify_script produced no changes; modifications must be patchable."
        )

    if modified_path:
        try:
            text = Path(modified_path).read_text()
            issues = _scan_for_hallucinated_api(text)
            if issues:
                return ToolGuardrailFunctionOutput.reject_content(
                    "Modified script contains hallucinated APIs: " + "; ".join(issues)
                )
        except Exception:
            return ToolGuardrailFunctionOutput.allow()

    return ToolGuardrailFunctionOutput.allow()


# =============================================================================
# LAZY IMPORTS - Import from script-generator modules
# =============================================================================

# These are lazy-imported to avoid import errors if modules are not present
_technique_catalog = None
_technique_selector = None
_validator = None
_effect_registry = None


def _get_technique_catalog():
    """Lazy import technique_catalog module."""
    global _technique_catalog
    if _technique_catalog is None:
        try:
            import technique_catalog
            _technique_catalog = technique_catalog
        except ImportError as e:
            raise ImportError(f"Could not import technique_catalog: {e}")
    return _technique_catalog


def _get_technique_selector():
    """Lazy import technique_selector module."""
    global _technique_selector
    if _technique_selector is None:
        try:
            import technique_selector
            _technique_selector = technique_selector
        except ImportError as e:
            raise ImportError(f"Could not import technique_selector: {e}")
    return _technique_selector


def _get_validator():
    """Lazy import validator module."""
    global _validator
    if _validator is None:
        try:
            import validator
            _validator = validator
        except ImportError as e:
            raise ImportError(f"Could not import validator: {e}")
    return _validator


def _get_effect_registry():
    """Lazy import effect_registry module."""
    global _effect_registry
    if _effect_registry is None:
        try:
            import effect_registry
            _effect_registry = effect_registry
        except ImportError as e:
            raise ImportError(f"Could not import effect_registry: {e}")
    return _effect_registry


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeneratedScript:
    """Result of script generation."""
    success: bool
    script_path: str
    script_content: str
    template_used: Optional[str]
    parameters: Dict[str, Any]
    notes: List[str]
    technique_name: Optional[str] = None


@dataclass
class ScriptModification:
    """Result of script modification."""
    success: bool
    original_path: str
    modified_path: str
    changes_made: List[str]
    parameters_changed: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    has_critical_warnings: bool = False


# =============================================================================
# INTERNAL IMPLEMENTATION FUNCTIONS
# =============================================================================

def _list_techniques_impl(effect_type: str = "pyro") -> str:
    """
    List available techniques from the catalog.

    Args:
        effect_type: Type of effect ("pyro" currently supported)

    Returns:
        JSON with available techniques and their descriptions
    """
    try:
        catalog = _get_technique_catalog()

        if effect_type.lower() == "pyro":
            techniques = []
            for name, tech in catalog.PYRO_TECHNIQUES.items():
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

    except Exception as e:
        return json.dumps({
            "error": f"Failed to list techniques: {str(e)}"
        })


def _recommend_technique_impl(
    effect_type: str,
    description: str,
    keyword_weight: float = 0.3,
    prefer_untried: bool = True
) -> str:
    """
    Recommend a technique using UCB1 algorithm.

    Args:
        effect_type: Type of effect
        description: Description of desired effect
        keyword_weight: How much to weight keyword matches (0-1)
        prefer_untried: If True, untried techniques get exploration bonus

    Returns:
        JSON with recommendation details
    """
    try:
        selector = _get_technique_selector()

        recommendation = selector.recommend_technique(
            effect_type=effect_type,
            description=description,
            keyword_weight=keyword_weight,
            prefer_untried=prefer_untried
        )

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


def _record_technique_outcome_impl(
    technique_name: str,
    effect_type: str,
    success: bool,
    final_score: float,
    iterations: int = 1
) -> str:
    """
    Record the outcome of using a technique for learning.

    Args:
        technique_name: Name of the technique that was used
        effect_type: Type of effect
        success: True if the technique passed quality thresholds
        final_score: Final quality score achieved (0-100)
        iterations: Number of iterations needed to pass

    Returns:
        JSON confirmation with updated statistics
    """
    try:
        selector = _get_technique_selector()
        store = selector.TechniquePerformanceStore()

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


def _validate_script_impl(script_path: str, strict: bool = False) -> str:
    """
    Validate a Blender script before execution.

    Args:
        script_path: Path to the Blender Python script to validate
        strict: If True, treat warnings as errors

    Returns:
        JSON with validation results
    """
    try:
        validator = _get_validator()

        # Resolve path
        path = Path(script_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / script_path

        if not path.exists():
            return json.dumps({
                "valid": False,
                "error": f"Script not found: {path}",
                "issues": [],
                "extracted_params": {},
                "detected_effect_type": "unknown",
                "error_count": 1,
                "warning_count": 0
            })

        # Validate (validator.validate_script doesn't accept strict parameter)
        result = validator.validate_script(str(path))

        # Calculate counts from issues list (not direct attributes on ValidationResult)
        from validator import ValidationSeverity
        error_count = sum(1 for i in result.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in result.issues if i.severity == ValidationSeverity.WARNING)
        is_valid = result.valid

        if strict and warning_count > 0:
            # In strict mode, any warnings make the script invalid
            is_valid = False
            error_count += warning_count

        return json.dumps({
            "valid": is_valid,
            "issues": [
                {
                    "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                    "category": issue.category,  # Note: ValidationIssue uses 'category', not 'code'
                    "message": issue.message,
                    "line": issue.line,
                    "suggestion": issue.suggestion
                }
                for issue in result.issues
            ],
            "extracted_params": result.extracted_params,
            "detected_effect_type": result.detected_effect_type,
            "error_count": error_count,
            "warning_count": warning_count if not strict else 0
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "valid": False,
            "error": f"Validation failed: {str(e)}",
            "issues": [],
            "extracted_params": {},
            "detected_effect_type": "unknown",
            "error_count": 1,
            "warning_count": 0
        })


def _generate_script_impl(
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
    Generate a new Blender script.

    This is a simplified in-process implementation that generates
    basic Blender scripts without the full MCP server complexity.

    Args:
        effect_type: Type of effect
        description: Description of what to create
        output_name: Name for the output script
        resolution: Simulation resolution
        frame_start: Start frame
        frame_end: End frame
        template_name: Optional template to base on
        technique_name: Optional specific technique
        force_random_technique: If True, pick randomly

    Returns:
        JSON with generated script details
    """
    try:
        catalog = _get_technique_catalog()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{output_name}.py"
        notes = []

        # Select technique
        selected_technique = None
        effect_lower = effect_type.lower()

        if effect_lower in ["pyro", "smoke", "fire", "explosion", "nebula", "gas", "sun"]:
            if technique_name and technique_name in catalog.PYRO_TECHNIQUES:
                selected_technique = catalog.get_technique_with_randomized_params(technique_name, "pyro")
                notes.append(f"Using requested technique: {technique_name}")
            elif force_random_technique:
                selected_technique = catalog.get_random_technique("pyro")
                selected_technique = catalog.get_technique_with_randomized_params(
                    selected_technique["name"], "pyro"
                )
                notes.append(f"Random technique selected: {selected_technique['name']}")
            else:
                selected_technique = catalog.get_technique_with_randomized_params(description, "pyro")
                notes.append(f"Technique: {selected_technique['name']} (keyword match)")

        if not selected_technique:
            # Fallback to first technique
            if catalog.PYRO_TECHNIQUES:
                first_name = list(catalog.PYRO_TECHNIQUES.keys())[0]
                selected_technique = catalog.get_technique_with_randomized_params(first_name, "pyro")
                notes.append(f"Fallback technique: {first_name}")
            else:
                return json.dumps({
                    "success": False,
                    "error": "No techniques available",
                    "script_path": "",
                    "notes": notes
                })

        # Generate script content
        technique_desc = selected_technique.get("description", "VFX effect")
        domain_params = selected_technique.get("domain_params", {})
        flow_params = selected_technique.get("flow_params", {})

        # SELF-LEARNING: NO HARDCODED PHYSICS RULES
        # Previously had space physics override here (beta=0.0 for sun/nebula)
        # Now physics rules emerge from learning via dynamic_instructions.py
        # The system will observe outcomes and build validated rules

        script_content = f'''#!/usr/bin/env python3
"""
{output_name.replace("_", " ").title()}

Generated by PlasmaDXR Script Generator
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Technique: {selected_technique.get("name", "default")}

Description:
    {description}

Effect Type: {effect_type}
Visual Signature: {selected_technique.get("visual_signature", technique_desc)}
"""

import bpy
import sys
from pathlib import Path


class Config:
    """Script configuration."""
    RESOLUTION = {resolution}
    FRAME_START = {frame_start}
    FRAME_END = {frame_end}
    OUTPUT_DIR = "build/vdb_output/{output_name}"
    BAKE = True
    RENDER = True
    RENDER_FRAMES = "mid"
    RENDER_RESOLUTION_X = 512
    RENDER_RESOLUTION_Y = 512
    RENDER_SAMPLES = 64
    DOMAIN_SCALE = 4.0

    # Technique parameters
    BURNING_RATE = {domain_params.get("burning_rate", 0.75)}
    FLAME_SMOKE = {domain_params.get("flame_smoke", 1.0)}
    FLAME_VORTICITY = {domain_params.get("flame_vorticity", 0.5)}
    BETA_BUOYANCY = {domain_params.get("beta", 1.0)}
    FUEL_AMOUNT = {flow_params.get("fuel_amount", 1.0)}
    TEMPERATURE = {flow_params.get("temperature", 1.0)}


def parse_args():
    """Parse command line arguments."""
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
        else:
            i += 1

parse_args()


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


def create_domain():
    """Create and configure fluid domain."""
    bpy.ops.mesh.primitive_cube_add(size=Config.DOMAIN_SCALE, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "FluidDomain"

    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = 'GAS'
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = True

    # Cache settings
    settings.cache_type = 'ALL'
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = 'OPENVDB'
    settings.cache_frame_start = Config.FRAME_START
    settings.cache_frame_end = Config.FRAME_END

    # Gas behavior (from technique)
    settings.burning_rate = Config.BURNING_RATE
    settings.flame_smoke = Config.FLAME_SMOKE
    settings.flame_vorticity = Config.FLAME_VORTICITY
    settings.beta = Config.BETA_BUOYANCY

    # Add volume material
    add_volume_material(domain)

    return domain


def add_volume_material(domain):
    """Add volumetric material for rendering."""
    mat = bpy.data.materials.new(name="FireSmokeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.inputs['Color'].default_value = (1.0, 0.3, 0.05, 1.0)
    volume.inputs['Density'].default_value = 10.0
    volume.inputs['Anisotropy'].default_value = 0.5
    volume.inputs['Blackbody Intensity'].default_value = 8.0
    volume.inputs['Temperature'].default_value = 2500.0

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    domain.data.materials.append(mat)
    print("[script] Volumetric material applied")


def create_emitter():
    """Create flow emitter."""
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(0, 0, -1.5))
    emitter = bpy.context.active_object
    emitter.name = "FlowEmitter"

    bpy.ops.object.modifier_add(type='FLUID')
    emitter.modifiers["Fluid"].fluid_type = 'FLOW'

    flow = emitter.modifiers["Fluid"].flow_settings
    flow.flow_type = 'BOTH'
    flow.flow_behavior = 'INFLOW'
    flow.fuel_amount = Config.FUEL_AMOUNT
    flow.temperature = Config.TEMPERATURE

    emitter.hide_render = True

    return emitter


def setup_camera_and_lighting():
    """Add camera and lights."""
    bpy.ops.object.camera_add(location=(6, -6, 4))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0


def bake_simulation(domain):
    """Bake the fluid simulation."""
    print(f"[script] Baking: frames {{Config.FRAME_START}}-{{Config.FRAME_END}}")
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)

    bpy.ops.fluid.bake_all()
    print("[script] Bake complete!")


def render_previews():
    """Render preview images."""
    scene = bpy.context.scene
    scene.cycles.samples = Config.RENDER_SAMPLES
    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.image_settings.file_format = 'PNG'

    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    mid = (Config.FRAME_START + Config.FRAME_END) // 2
    scene.frame_set(mid)

    render_path = output_dir / f"render_{{mid:04d}}.png"
    scene.render.filepath = str(render_path)

    print(f"[script] Rendering frame {{mid}}...")
    bpy.ops.render.render(write_still=True)
    print(f"[script] Saved: {{render_path}}")


def main():
    print("=" * 60)
    print("{output_name.replace("_", " ").title()}")
    print("=" * 60)

    clear_scene()
    setup_scene()

    domain = create_domain()
    emitter = create_emitter()
    setup_camera_and_lighting()

    if Config.BAKE:
        bake_simulation(domain)

    if Config.RENDER:
        render_previews()

    # Save blend file
    blend_path = Path(Config.OUTPUT_DIR) / "{output_name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[script] Saved: {{blend_path}}")


if __name__ == "__main__":
    main()
'''

        # Write script
        output_path.write_text(script_content)
        notes.append(f"Script written to: {output_path}")

        result = GeneratedScript(
            success=True,
            script_path=str(output_path),
            script_content=script_content[:1000] + "...",  # Preview only
            template_used="built-in pyro",
            parameters={
                "resolution": resolution,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "technique": selected_technique.get("name", "default"),
            },
            notes=notes,
            technique_name=selected_technique.get("name")
        )

        return json.dumps(asdict(result), indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Script generation failed: {str(e)}",
            "script_path": "",
            "notes": [str(e)]
        })


def _modify_script_impl(
    script_path: str,
    modifications: Dict[str, Any],
    output_name: Optional[str] = None
) -> str:
    """
    Modify an existing Blender script.

    Handles multiple input formats from Coordinator/Learning Agent:
    1. Simple param names: 'noise_scale' -> Config.NOISE_SCALE or settings.noise_scale
    2. Blender API paths: 'FluidDomainSettings.noise_scale' -> settings.noise_scale
    3. Prefixed vars: 'dsettings.noise_scale' -> dsettings.noise_scale

    Args:
        script_path: Path to the script to modify
        modifications: Dict of changes to make
        output_name: Optional new filename

    Returns:
        JSON with modification results
    """
    # Map Coordinator Blender API class names to common script variable patterns
    # The Coordinator outputs 'FluidDomainSettings.param' but scripts use 'settings.param'
    BLENDER_CLASS_TO_VAR = {
        'FluidDomainSettings': ['settings', 'dsettings', 'domain_settings', 'dom', 'domain'],
        'FluidFlowSettings': ['flow', 'fsettings', 'flow_settings'],
        'FluidEffectorSettings': ['effector', 'effector_settings'],
        'Scene': ['scene', 'bpy.context.scene'],
        'Object': ['obj', 'domain', 'emitter'],
    }

    try:
        # Resolve path
        path = Path(script_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / script_path

        if not path.exists():
            return json.dumps({
                "success": False,
                "error": f"Script not found: {path}",
                "original_path": str(path),
                "modified_path": "",
                "changes_made": [],
                "parameters_changed": {}
            })

        # Read original content
        content = path.read_text()
        changes_made = []
        params_changed = {}

        def _format_value_for_python(val: Any, old_val_str: str = "") -> str:
            """Format a value for insertion into Python source code.

            String values that represent enum constants (e.g. 'REPLAY', 'LIQUID')
            must be quoted in the output, otherwise they become bare NameErrors.
            If the old value was already quoted, preserve quoting for the new value.
            """
            if isinstance(val, str):
                # Check if old value was a quoted string — preserve quoting
                old_stripped = old_val_str.strip()
                was_quoted = (
                    (old_stripped.startswith("'") and old_stripped.endswith("'"))
                    or (old_stripped.startswith('"') and old_stripped.endswith('"'))
                )
                if was_quoted:
                    return f"'{val}'"
                # If it looks like a Python identifier/enum (all uppercase, underscores),
                # it likely needs quotes to avoid NameError
                if val.replace('_', '').isalpha() and val == val.upper() and len(val) > 1:
                    return f"'{val}'"
                # Other string values that don't look like numbers or Python literals
                try:
                    float(val)
                except ValueError:
                    if val not in ('True', 'False', 'None'):
                        return f"'{val}'"
            return str(val)

        # Apply modifications via regex replacements
        # CRITICAL: Only modify parameters within the Config class section
        # This prevents accidental modification of parse_args() or other functions
        for param, value in modifications.items():
            param_upper = param.upper()

            # Parse Coordinator-style API path: 'FluidDomainSettings.noise_scale' -> ('FluidDomainSettings', 'noise_scale')
            blender_class = None
            attr_name = param
            if '.' in param:
                parts = param.split('.', 1)
                if parts[0] in BLENDER_CLASS_TO_VAR:
                    blender_class = parts[0]
                    attr_name = parts[1]

            # Handle explicit deletion directives or delete sentinel
            if value == DELETE_SENTINEL or (
                isinstance(value, str) and any(
                    marker in value.lower()
                    for marker in ("delete this line", "remove this line", "delete line", "remove line")
                )
            ):
                # Extract target token to remove
                if value == DELETE_SENTINEL:
                    target = attr_name or param
                else:
                    target = value.split("(")[0].strip()
                    target = target if target else param

                lines = content.split("\n")
                removed_any = False
                for i, line in enumerate(lines):
                    if target in line and not line.strip().startswith("#"):
                        lines[i] = f"# REMOVED: {line.strip()}"
                        removed_any = True

                if removed_any:
                    content = "\n".join(lines)
                    changes_made.append(f"REMOVED line containing '{target}'")
                    params_changed[param] = {"from": target, "to": "REMOVED"}
                    continue

            # First, find the Config class boundaries
            config_class_pattern = r"(class Config:.*?)((?=\ndef\s|\nclass\s|\Z))"
            config_match = re.search(config_class_pattern, content, re.DOTALL)

            if config_match:
                config_section = config_match.group(0)
                config_start = config_match.start()

                # Look for PARAM = value ONLY within Config class
                param_pattern = rf"(\s+{param_upper}\s*=\s*)([^\n]+)"
                param_match = re.search(param_pattern, config_section, re.IGNORECASE)

                if param_match:
                    old_val = param_match.group(2).strip()
                    formatted = _format_value_for_python(value, old_val)
                    # Replace only within Config section
                    new_config = re.sub(param_pattern, f"\\g<1>{formatted}", config_section, count=1, flags=re.IGNORECASE)
                    content = content[:config_start] + new_config + content[config_start + len(config_section):]
                    changes_made.append(f"{param_upper}: {old_val} -> {formatted}")
                    params_changed[param] = {"from": old_val, "to": formatted}
                    continue

            # Fallback: Try Config.PARAM references anywhere (for dynamically assigned values)
            pattern_config_ref = rf"(Config\.{param_upper}\s*=\s*)([^\n]+)"
            match = re.search(pattern_config_ref, content, re.IGNORECASE)
            if match and param not in params_changed:
                old_val = match.group(2).strip()
                formatted = _format_value_for_python(value, old_val)
                new_content = re.sub(pattern_config_ref, f"\\g<1>{formatted}", content, flags=re.IGNORECASE)
                if new_content != content:
                    content = new_content
                    changes_made.append(f"Config.{param_upper}: {old_val} -> {formatted}")
                    params_changed[param] = {"from": old_val, "to": formatted}
                    continue

            # DIRECT ATTRIBUTE PATTERN (Coordinator contract fix)
            # Handle scripts using direct assignment: settings.noise_scale = X
            # When Coordinator outputs 'FluidDomainSettings.noise_scale', we search for
            # settings.noise_scale, dsettings.noise_scale, domain_settings.noise_scale, etc.
            if param not in params_changed:
                direct_match_found = False

                # Build list of variable names to search for
                var_names_to_try = []
                if blender_class and blender_class in BLENDER_CLASS_TO_VAR:
                    var_names_to_try = BLENDER_CLASS_TO_VAR[blender_class]
                else:
                    # If no class prefix, try common variable names
                    var_names_to_try = ['settings', 'dsettings', 'flow', 'fsettings', 'domain_settings', 'flow_settings']

                for var_name in var_names_to_try:
                    # Pattern: var_name.attr_name = value (case-sensitive for attr_name)
                    # IMPORTANT: Use word boundary \b to prevent matching substrings
                    # (e.g., 'settings' should NOT match inside 'dsettings')
                    direct_pattern = rf"(\b{re.escape(var_name)}\.{re.escape(attr_name)}\s*=\s*)([^\n]+)"
                    direct_match = re.search(direct_pattern, content)

                    if direct_match:
                        old_val = direct_match.group(2).strip()
                        formatted = _format_value_for_python(value, old_val)
                        new_content = re.sub(direct_pattern, f"\\g<1>{formatted}", content, count=1)
                        if new_content != content:
                            content = new_content
                            changes_made.append(f"{var_name}.{attr_name}: {old_val} -> {formatted}")
                            params_changed[param] = {"from": old_val, "to": formatted, "pattern": "direct_attr"}
                            direct_match_found = True
                            break

                # Also try: attr on domain_settings property access (e.g., domain.modifiers["Fluid"].domain_settings.X)
                if not direct_match_found:
                    # More permissive pattern for chained property access
                    chained_pattern = rf"(\.{re.escape(attr_name)}\s*=\s*)([^\n]+)"
                    chained_matches = list(re.finditer(chained_pattern, content))
                    for chained_match in chained_matches:
                        # Verify this is not inside a string or comment
                        line_start = content.rfind('\n', 0, chained_match.start()) + 1
                        line = content[line_start:chained_match.end()]
                        if line.strip().startswith('#') or line.strip().startswith('"""'):
                            continue

                        old_val = chained_match.group(2).strip()
                        formatted = _format_value_for_python(value, old_val)
                        # Replace only this occurrence
                        content = content[:chained_match.start()] + f".{attr_name} = {formatted}" + content[chained_match.end():]
                        changes_made.append(f"*.{attr_name}: {old_val} -> {formatted}")
                        params_changed[param] = {"from": old_val, "to": formatted, "pattern": "chained_attr"}
                        break

            # SHADER NODE PATTERN (Learning Agent feedback loop fix)
            # Handle shader node input modifications: node.inputs['X'].default_value = Y
            # This enables the feedback loop to actually modify visual appearance
            # The Learning Agent analyzes the script and provides EXACT patterns from its analysis
            if param not in params_changed:
                # Check if param matches shader node pattern: var.inputs['Name'].default_value
                shader_pattern_match = re.match(
                    r"(\w+)\.inputs\[(['\"])(.+?)\2\]\.default_value",
                    param
                )
                if shader_pattern_match:
                    node_var = shader_pattern_match.group(1)
                    input_name = shader_pattern_match.group(3)

                    # Build pattern to find and replace
                    # Match: node_var.inputs['input_name'].default_value = old_value
                    search_pattern = rf"({re.escape(node_var)}\.inputs\[['\"]" + re.escape(input_name) + rf"['\"]\]\.default_value\s*=\s*)([^\n]+)"
                    shader_match = re.search(search_pattern, content)

                    if shader_match:
                        old_val = shader_match.group(2).strip()
                        formatted = _format_value_for_python(value, old_val)
                        new_content = re.sub(search_pattern, f"\\g<1>{formatted}", content, count=1)
                        if new_content != content:
                            content = new_content
                            changes_made.append(f"SHADER: {node_var}.inputs['{input_name}']: {old_val} -> {formatted}")
                            params_changed[param] = {"from": old_val, "to": formatted, "pattern": "shader_node"}

            # MATH NODE PATTERN (density multipliers, etc.)
            # Handle: multiply.inputs[N].default_value = Y
            if param not in params_changed:
                math_pattern_match = re.match(
                    r"(\w+)\.inputs\[(\d+)\]\.default_value",
                    param
                )
                if math_pattern_match:
                    node_var = math_pattern_match.group(1)
                    input_idx = math_pattern_match.group(2)

                    search_pattern = rf"({re.escape(node_var)}\.inputs\[{input_idx}\]\.default_value\s*=\s*)([^\n]+)"
                    math_match = re.search(search_pattern, content)

                    if math_match:
                        old_val = math_match.group(2).strip()
                        formatted = _format_value_for_python(value, old_val)
                        new_content = re.sub(search_pattern, f"\\g<1>{formatted}", content, count=1)
                        if new_content != content:
                            content = new_content
                            changes_made.append(f"MATH: {node_var}.inputs[{input_idx}]: {old_val} -> {formatted}")
                            params_changed[param] = {"from": old_val, "to": formatted, "pattern": "math_node"}

        # Determine output path
        if output_name:
            modified_path = OUTPUT_DIR / f"{output_name}.py"
            # Also update the script's internal OUTPUT_DIR to prevent render overwrites
            # This ensures each iteration writes to a unique directory
            old_output_dir_pattern = r'(OUTPUT_DIR\s*=\s*["\'])([^"\']+)(["\'])'
            output_dir_match = re.search(old_output_dir_pattern, content)
            if output_dir_match:
                old_output_path = output_dir_match.group(2)
                # Replace the base name with the new output_name (e.g., nasa_sun_test_v1 -> nasa_sun_test_v1_qfix3)
                new_output_path = re.sub(r'[^/]+$', output_name, old_output_path)
                content = re.sub(
                    old_output_dir_pattern,
                    f'\\g<1>{new_output_path}\\g<3>',
                    content
                )
                changes_made.append(f"OUTPUT_DIR: {old_output_path} -> {new_output_path}")
                params_changed["OUTPUT_DIR"] = {"from": old_output_path, "to": new_output_path}
        else:
            modified_path = path.with_stem(path.stem + "_modified")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        modified_path.write_text(content)

        result = ScriptModification(
            success=True,
            original_path=str(path),
            modified_path=str(modified_path),
            changes_made=changes_made,
            parameters_changed=params_changed,
            warnings=[],
            mitigations=[],
            has_critical_warnings=False
        )

        return json.dumps(asdict(result), indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Modification failed: {str(e)}",
            "original_path": script_path,
            "modified_path": "",
            "changes_made": [],
            "parameters_changed": {}
        })


# =============================================================================
# FUNCTION TOOL WRAPPERS (exposed to agents)
# =============================================================================

@function_tool
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
    Generate a new Blender Python script for VFX asset creation.

    Uses the Technique Catalog to select categorically different approaches,
    ensuring variety across generated scripts.

    Args:
        effect_type: Type of effect (pyro, explosion, fire, smoke, nebula, sun)
        description: Natural language description of what to create
        output_name: Name for the output script (without .py)
        resolution: Blender simulation resolution (default 96)
        frame_start: Animation start frame (default 1)
        frame_end: Animation end frame (default 50)
        template_name: Optional specific template to base on
        technique_name: Optional specific technique from catalog
        force_random_technique: If True, ignore keywords for maximum variety

    Returns:
        JSON with generated script path, content preview, and technique info
    """
    return _generate_script_impl(
        effect_type=effect_type,
        description=description,
        output_name=output_name,
        resolution=resolution,
        frame_start=frame_start,
        frame_end=frame_end,
        template_name=template_name,
        technique_name=technique_name,
        force_random_technique=force_random_technique
    )


@function_tool(
    tool_output_guardrails=[guard_modify_script_output],
)
async def modify_script(
    script_path: str,
    modifications_json: str,
    output_name: Optional[str] = None
) -> str:
    """
    Modify an existing Blender script based on evaluation feedback.

    Applies parameter changes and includes knowledge base warnings
    for potentially problematic modifications.

    Args:
        script_path: Path to the script to modify
        modifications_json: JSON string of changes to make:
            - resolution: New simulation resolution
            - frame_end: New end frame
            - turbulence: Turbulence/vorticity value (0-1)
            - density: Density multiplier
            - temperature: Temperature value
            - flame_smoke: Flame smoke ratio
            - domain_scale: Domain size multiplier
            - custom_code: Dict of {search_pattern: replacement}
            - "__DELETE__": Sentinel value to remove a line matching the key
            Example: '{"resolution": 128, "turbulence": 0.8}'
        output_name: Optional new filename (default: adds "_modified" suffix)

    Returns:
        JSON with modified script path, changes made, and warnings
    """
    modifications = json.loads(modifications_json) if modifications_json else {}
    return _modify_script_impl(
        script_path=script_path,
        modifications=modifications,
        output_name=output_name
    )


# REMOVED: apply_space_physics_fix tool
# This tool was removed as part of the self-learning architecture.
# Physics rules should emerge from experimentation, not be hardcoded.
# See docs/SELF_LEARNING_ARCHITECTURE.md for details.


@function_tool
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

    Args:
        script_path: Path to the Blender Python script to validate
        strict: If True, treat warnings as errors (default False)

    Returns:
        JSON with:
        - valid: True if script passes validation
        - issues: List of validation issues found
        - extracted_params: Parameters detected in script
        - detected_effect_type: volumetric, mesh, or unknown
        - error_count: Number of errors
        - warning_count: Number of warnings
    """
    return _validate_script_impl(script_path, strict)


# =============================================================================
# DIRECT CODE WRITING - LLM generates code, this saves it
# =============================================================================

def _write_script_impl(
    code: str,
    output_name: str,
    effect_type: str = "vfx",
    technique_name: str = "llm_generated",
    description: str = ""
) -> str:
    """
    Write LLM-generated Blender Python code directly to a file.

    This bypasses the technique catalog entirely - the LLM writes the code
    based on research findings, and this tool just saves it.

    Args:
        code: The complete Blender Python script code
        output_name: Name for the output file (without .py extension)
        effect_type: Type of effect being created
        technique_name: Name to identify this technique/approach
        description: Description of what the script does

    Returns:
        JSON with script path and validation info
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{output_name}.py"

        # Add header comment if not present
        if not code.strip().startswith('"""') and not code.strip().startswith('#'):
            header = f'''#!/usr/bin/env python3
"""
{output_name}

Generated by Script Writer Agent (LLM-generated, not from template)
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Technique: {technique_name}
Effect Type: {effect_type}

Description:
    {description}
"""

'''
            code = header + code

        # Write the file
        output_path.write_text(code)

        # Basic validation
        issues = []
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        # Check for required Blender imports
        if "import bpy" not in code:
            issues.append("Warning: Missing 'import bpy'")

        return json.dumps({
            "success": True,
            "script_path": str(output_path),
            "technique_name": technique_name,
            "effect_type": effect_type,
            "code_length": len(code),
            "issues": issues,
            "message": f"Script written to {output_path}"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "script_path": None
        }, indent=2)


@function_tool(
    tool_input_guardrails=[guard_write_script_input],
    tool_output_guardrails=[guard_write_script_output],
)
async def write_script(
    code: str,
    output_name: str,
    effect_type: str = "vfx",
    technique_name: str = "llm_generated",
    description: str = ""
) -> str:
    """
    Write YOUR generated Blender Python code to a file.

    USE THIS instead of generate_script! You are the code generator.
    Based on research findings, write complete Blender Python code and
    use this tool to save it.

    IMPORTANT: Generate code that matches the RESEARCH FINDINGS, not generic templates.
    For sun effects: Use shader-based approach (Emission materials, Volume Scatter, compositor glare)
    For explosions: Use appropriate simulation or shader approach from research

    Args:
        code: Your complete Blender Python script code (must include 'import bpy')
        output_name: Name for the output file (e.g., "sun_shader_v1")
        effect_type: Type of effect (sun, explosion, fire, nebula, etc.)
        technique_name: Descriptive name for your approach (e.g., "layered_shader_corona")
        description: Brief description of what your script does

    Returns:
        JSON with script path, technique info, and any validation issues

    Example:
        code = '''
import bpy
# Create sun photosphere with emission material
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
sun = bpy.context.active_object
...
'''
        write_script(code=code, output_name="sun_v1", effect_type="sun",
                     technique_name="photosphere_corona_shader")
    """
    return _write_script_impl(
        code=code,
        output_name=output_name,
        effect_type=effect_type,
        technique_name=technique_name,
        description=description
    )


# =============================================================================
# DEPRECATED: Template-based generation (kept for backwards compatibility)
# =============================================================================

@function_tool
async def list_techniques(
    effect_type: str = "pyro"
) -> str:
    """
    DEPRECATED: Use write_script() instead and generate code from research findings.

    List available techniques from the catalog for variety in generation.

    Each technique produces categorically different visual results.

    Args:
        effect_type: Type of effect (pyro, explosion, etc.)

    Returns:
        JSON with available techniques and their descriptions
    """
    return _list_techniques_impl(effect_type)


@function_tool
async def recommend_technique(
    effect_type: str,
    description: str,
    keyword_weight: float = 0.3,
    prefer_untried: bool = True
) -> str:
    """
    Recommend a technique using UCB1 algorithm for exploration/exploitation balance.

    Uses keyword filtering combined with Upper Confidence Bound (UCB1)
    to select techniques. Untried techniques get exploration bonus.
    Performance data persists across sessions for learning.

    Args:
        effect_type: Type of effect (pyro, explosion, fire, etc.)
        description: Description of desired effect (used for keyword matching)
        keyword_weight: How much to weight keyword matches (0-1, default 0.3)
        prefer_untried: If True, untried techniques get exploration bonus

    Returns:
        JSON with:
        - technique_name: Recommended technique
        - confidence: 0-1 confidence score
        - selection_reason: Why this technique was selected
        - ucb_score: Raw UCB1 score
        - alternatives: Top 3 alternative techniques
        - exploration_mode: True if this is an untried technique
    """
    return _recommend_technique_impl(
        effect_type=effect_type,
        description=description,
        keyword_weight=keyword_weight,
        prefer_untried=prefer_untried
    )


@function_tool
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
    technique performance statistics. Enables UCB1 to learn which
    techniques work best for different effect types.

    Args:
        technique_name: Name of the technique that was used
        effect_type: Type of effect (pyro, explosion, etc.)
        success: True if the technique passed quality thresholds
        final_score: Final quality score achieved (0-100)
        iterations: Number of iterations needed to pass

    Returns:
        JSON with updated technique statistics
    """
    return _record_technique_outcome_impl(
        technique_name=technique_name,
        effect_type=effect_type,
        success=success,
        final_score=final_score,
        iterations=iterations
    )
