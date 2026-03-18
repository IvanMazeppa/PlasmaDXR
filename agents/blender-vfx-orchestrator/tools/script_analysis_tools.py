"""
Script Analysis Tools for Self-Learning Feedback Loop.

Enables the Learning Agent to understand script structure BEFORE suggesting
modifications, ensuring suggestions match what the script can actually handle.

KEY PRINCIPLE: Not hardcoded fixes - the agent learns what works by analyzing
the script and tracking modification outcomes.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import function_tool


@dataclass
class ModifiablePattern:
    """A pattern in the script that can be modified."""
    pattern_type: str  # "config_class", "settings_attr", "shader_node", "direct_assignment"
    identifier: str    # The full path/name (e.g., "DENSITY", "settings.noise_scale")
    current_value: Any
    line_number: int
    line_text: str
    modification_format: str  # How to specify this in parameter_modifications


@dataclass
class ScriptAnalysis:
    """Complete analysis of a script's modifiable patterns."""
    script_path: str
    config_values: Dict[str, ModifiablePattern]
    settings_assignments: Dict[str, ModifiablePattern]
    shader_node_inputs: Dict[str, ModifiablePattern]
    other_assignments: Dict[str, ModifiablePattern]
    total_modifiable: int
    analysis_notes: List[str]


def _extract_config_class_values(content: str, lines: List[str]) -> Dict[str, ModifiablePattern]:
    """Extract values from Config class definition."""
    patterns = {}

    # Find Config class boundaries
    config_match = re.search(r"class Config:.*?(?=\ndef\s|\nclass\s|\Z)", content, re.DOTALL)
    if not config_match:
        return patterns

    config_section = config_match.group()
    config_start_line = content[:config_match.start()].count('\n') + 1

    # Find all UPPERCASE = value assignments in Config
    for match in re.finditer(r"^\s+([A-Z][A-Z_0-9]*)\s*=\s*(.+)$", config_section, re.MULTILINE):
        name = match.group(1)
        value_str = match.group(2).strip()

        # Calculate line number
        line_offset = config_section[:match.start()].count('\n')
        line_num = config_start_line + line_offset

        # Parse value
        try:
            value = ast.literal_eval(value_str.split('#')[0].strip())
        except Exception:
            value = value_str

        patterns[name] = ModifiablePattern(
            pattern_type="config_class",
            identifier=name,
            current_value=value,
            line_number=line_num,
            line_text=lines[line_num - 1].strip() if line_num <= len(lines) else "",
            modification_format=f'{{"Config.{name}": new_value}} or {{"{name}": new_value}}'
        )

    return patterns


def _extract_settings_assignments(content: str, lines: List[str]) -> Dict[str, ModifiablePattern]:
    """Extract settings.X = value patterns (FluidDomainSettings, FlowSettings)."""
    patterns = {}

    # Common variable names for settings objects
    var_patterns = [
        r"(settings|dsettings|fsettings|domain_settings|flow_settings|flow)\.([\w_]+)\s*=\s*(.+)",
    ]

    for var_pattern in var_patterns:
        for match in re.finditer(var_pattern, content, re.MULTILINE):
            var_name = match.group(1)
            attr_name = match.group(2)
            value_str = match.group(3).strip()

            # Calculate line number
            line_num = content[:match.start()].count('\n') + 1

            # Skip if in comment or docstring
            line = lines[line_num - 1] if line_num <= len(lines) else ""
            if line.strip().startswith('#') or '"""' in line:
                continue

            # Parse value
            try:
                value = ast.literal_eval(value_str.split('#')[0].strip())
            except Exception:
                value = value_str

            identifier = f"{var_name}.{attr_name}"
            patterns[identifier] = ModifiablePattern(
                pattern_type="settings_attr",
                identifier=identifier,
                current_value=value,
                line_number=line_num,
                line_text=line.strip(),
                modification_format=f'{{"{identifier}": new_value}} or {{"FluidDomainSettings.{attr_name}": new_value}}'
            )

    return patterns


def _extract_shader_node_inputs(content: str, lines: List[str]) -> Dict[str, ModifiablePattern]:
    """Extract shader node input assignments: node.inputs['X'].default_value = Y."""
    patterns = {}

    # Pattern for node.inputs['Name'].default_value = value
    node_input_pattern = r"(\w+)\.inputs\[(['\"])(.+?)\2\]\.default_value\s*=\s*(.+)"

    for match in re.finditer(node_input_pattern, content, re.MULTILINE):
        node_var = match.group(1)
        input_name = match.group(3)
        value_str = match.group(4).strip()

        # Calculate line number
        line_num = content[:match.start()].count('\n') + 1

        # Skip if in comment or docstring
        line = lines[line_num - 1] if line_num <= len(lines) else ""
        if line.strip().startswith('#') or '"""' in line:
            continue

        # Parse value
        try:
            value = ast.literal_eval(value_str.split('#')[0].strip())
        except Exception:
            value = value_str

        identifier = f"{node_var}.inputs['{input_name}'].default_value"
        patterns[identifier] = ModifiablePattern(
            pattern_type="shader_node",
            identifier=identifier,
            current_value=value,
            line_number=line_num,
            line_text=line.strip(),
            modification_format=f'{{"{identifier}": new_value}} - MUST use exact pattern'
        )

    return patterns


def _extract_other_assignments(content: str, lines: List[str]) -> Dict[str, ModifiablePattern]:
    """Extract other potentially modifiable assignments."""
    patterns = {}

    # Math node multiply.inputs[1].default_value (common for density multipliers)
    multiply_pattern = r"(\w+)\.inputs\[(\d+)\]\.default_value\s*=\s*(.+)"

    for match in re.finditer(multiply_pattern, content, re.MULTILINE):
        node_var = match.group(1)
        input_idx = match.group(2)
        value_str = match.group(3).strip()

        # Calculate line number
        line_num = content[:match.start()].count('\n') + 1

        # Skip if already captured by shader_node pattern
        line = lines[line_num - 1] if line_num <= len(lines) else ""
        if "inputs['" in line or line.strip().startswith('#'):
            continue

        try:
            value = ast.literal_eval(value_str.split('#')[0].strip())
        except Exception:
            value = value_str

        identifier = f"{node_var}.inputs[{input_idx}].default_value"
        patterns[identifier] = ModifiablePattern(
            pattern_type="math_node",
            identifier=identifier,
            current_value=value,
            line_number=line_num,
            line_text=line.strip(),
            modification_format=f'{{"{identifier}": new_value}} - for math/mix node inputs'
        )

    return patterns


def analyze_script_structure(script_path: str) -> ScriptAnalysis:
    """
    Analyze a Blender script to identify all modifiable value patterns.

    Args:
        script_path: Path to the script file

    Returns:
        ScriptAnalysis with all modifiable patterns categorized
    """
    path = Path(script_path)
    if not path.exists():
        return ScriptAnalysis(
            script_path=script_path,
            config_values={},
            settings_assignments={},
            shader_node_inputs={},
            other_assignments={},
            total_modifiable=0,
            analysis_notes=["ERROR: Script file not found"]
        )

    content = path.read_text()
    lines = content.splitlines()

    config_values = _extract_config_class_values(content, lines)
    settings_assignments = _extract_settings_assignments(content, lines)
    shader_node_inputs = _extract_shader_node_inputs(content, lines)
    other_assignments = _extract_other_assignments(content, lines)

    total = len(config_values) + len(settings_assignments) + len(shader_node_inputs) + len(other_assignments)

    notes = []
    if not config_values:
        notes.append("No Config class found - Config.X modifications won't work")
    if not shader_node_inputs:
        notes.append("No shader node inputs found - volume material may be missing")
    if shader_node_inputs:
        notes.append(f"Found {len(shader_node_inputs)} shader node inputs - these control visual appearance")

    return ScriptAnalysis(
        script_path=script_path,
        config_values=config_values,
        settings_assignments=settings_assignments,
        shader_node_inputs=shader_node_inputs,
        other_assignments=other_assignments,
        total_modifiable=total,
        analysis_notes=notes
    )


def _impl_analyze_script_modifiable_patterns(script_path: str) -> str:
    """Implementation for analyze_script_modifiable_patterns."""
    analysis = analyze_script_structure(script_path)

    # Format for LLM consumption
    result = {
        "script_path": analysis.script_path,
        "total_modifiable_patterns": analysis.total_modifiable,
        "analysis_notes": analysis.analysis_notes,
        "patterns_by_type": {
            "config_class": {
                "count": len(analysis.config_values),
                "description": "Modify via {'PARAM_NAME': value} - affects Config.X values",
                "patterns": {
                    k: {
                        "current_value": v.current_value,
                        "line": v.line_number,
                        "modification_hint": v.modification_format
                    }
                    for k, v in analysis.config_values.items()
                }
            },
            "settings_assignments": {
                "count": len(analysis.settings_assignments),
                "description": "Modify via {'settings.attr': value} - Blender simulation params",
                "patterns": {
                    k: {
                        "current_value": v.current_value,
                        "line": v.line_number,
                        "modification_hint": v.modification_format
                    }
                    for k, v in analysis.settings_assignments.items()
                }
            },
            "shader_node_inputs": {
                "count": len(analysis.shader_node_inputs),
                "description": "Modify via exact pattern - CRITICAL for visual appearance",
                "patterns": {
                    k: {
                        "current_value": v.current_value,
                        "line": v.line_number,
                        "modification_hint": v.modification_format
                    }
                    for k, v in analysis.shader_node_inputs.items()
                }
            },
            "math_nodes": {
                "count": len(analysis.other_assignments),
                "description": "Math/mix node inputs - often control density multipliers",
                "patterns": {
                    k: {
                        "current_value": v.current_value,
                        "line": v.line_number,
                        "modification_hint": v.modification_format
                    }
                    for k, v in analysis.other_assignments.items()
                }
            }
        },
        "recommendation": _generate_modification_recommendation(analysis)
    }

    return json.dumps(result, indent=2)


def _generate_modification_recommendation(analysis: ScriptAnalysis) -> str:
    """Generate actionable recommendation based on analysis."""
    if analysis.shader_node_inputs:
        density_inputs = [k for k in analysis.shader_node_inputs if 'Density' in k]
        intensity_inputs = [k for k in analysis.shader_node_inputs if 'Intensity' in k or 'Strength' in k]

        if density_inputs or intensity_inputs:
            return (
                f"To fix visual issues: Modify shader nodes directly. "
                f"Density controls: {density_inputs or 'none found'}. "
                f"Intensity controls: {intensity_inputs or 'none found'}. "
                f"Use EXACT identifier in parameter_modifications."
            )

    if analysis.config_values:
        return (
            "Script uses Config class. Modify via {'PARAM_NAME': value}. "
            "Check if Config values are actually used in shader setup."
        )

    return "Limited modification patterns found. Consider regenerating script."


@function_tool
async def analyze_script_modifiable_patterns(script_path: str) -> str:
    """
    Analyze a Blender script to identify all modifiable value patterns.

    Call this BEFORE suggesting parameter_modifications to ensure your
    suggestions match the script's actual structure.

    Returns categorized patterns:
    - config_class: Config.X values (modify via {"PARAM_NAME": value})
    - settings_assignments: settings.X values (modify via {"settings.attr": value})
    - shader_node_inputs: node.inputs['X'].default_value (CRITICAL for visuals)
    - math_nodes: multiply.inputs[N].default_value (density multipliers)

    Args:
        script_path: Absolute path to the Blender Python script

    Returns:
        JSON with all modifiable patterns and modification hints

    Example usage:
        1. Call analyze_script_modifiable_patterns(script_path)
        2. Review patterns_by_type.shader_node_inputs for visual controls
        3. Suggest parameter_modifications using EXACT identifiers from analysis
    """
    return _impl_analyze_script_modifiable_patterns(script_path)


@function_tool
async def get_effective_modification_for_issue(
    issue: str,
    script_path: str,
    effect_type: str = "pyro"
) -> str:
    """
    Get the most effective modification pattern for a visual issue.

    Analyzes the script and recommends which pattern to modify based on
    the issue description.

    Args:
        issue: The visual issue (e.g., "too dark", "not enough fire", "grey mesh")
        script_path: Path to the current script
        effect_type: Effect type for context

    Returns:
        JSON with recommended modification pattern and example
    """
    analysis = analyze_script_structure(script_path)

    # Map common issues to relevant patterns
    issue_lower = issue.lower()

    recommendations = []

    if any(term in issue_lower for term in ["dark", "not visible", "grey", "density"]):
        # Density/visibility issues - look for density controls
        for k, v in analysis.shader_node_inputs.items():
            if 'Density' in k:
                recommendations.append({
                    "issue_type": "visibility/density",
                    "pattern": v.identifier,
                    "current_value": v.current_value,
                    "suggested_modification": {v.identifier: v.current_value * 2},
                    "reasoning": "Increase density for visibility"
                })

        for k, v in analysis.other_assignments.items():
            if 'multiply' in k.lower():
                recommendations.append({
                    "issue_type": "visibility/density",
                    "pattern": v.identifier,
                    "current_value": v.current_value,
                    "suggested_modification": {v.identifier: v.current_value * 2},
                    "reasoning": "Density multiplier - common for smoke visibility"
                })

    if any(term in issue_lower for term in ["fire", "flame", "glow", "emission", "bright"]):
        # Fire/emission issues
        for k, v in analysis.shader_node_inputs.items():
            if any(term in k for term in ['Intensity', 'Strength', 'Emission', 'Blackbody']):
                recommendations.append({
                    "issue_type": "fire/emission",
                    "pattern": v.identifier,
                    "current_value": v.current_value,
                    "suggested_modification": {v.identifier: v.current_value * 2 if isinstance(v.current_value, (int, float)) else 10.0},
                    "reasoning": "Increase emission/intensity for fire visibility"
                })

    if not recommendations:
        recommendations.append({
            "issue_type": "unrecognized",
            "note": "No specific pattern found for this issue",
            "available_shader_nodes": list(analysis.shader_node_inputs.keys()),
            "available_config": list(analysis.config_values.keys()),
            "suggestion": "Try modifying shader node inputs or regenerate script"
        })

    return json.dumps({
        "issue": issue,
        "script_path": script_path,
        "recommendations": recommendations,
        "total_patterns_available": analysis.total_modifiable
    }, indent=2)


# =============================================================================
# LOOK-DEV COVERAGE METRICS (Task 2: deterministic script-level richness)
# =============================================================================


@dataclass
class LookDevCoverage:
    """Deterministic look-dev coverage metrics for a Blender script.

    Measures scene richness at the script level without LLM cost.
    """
    # Color management
    has_color_management: bool = False  # AgX, Filmic, or explicit view_transform
    color_management_type: str = ""

    # Depth of field
    has_dof: bool = False

    # World/background strategy
    has_world_setup: bool = False  # Any world node tree or background color
    world_strategy: str = ""  # "procedural_sky" | "env_texture" | "dark_world" | "gradient" | ""

    # Materials
    material_count: int = 0
    principled_bsdf_count: int = 0
    has_transmission: bool = False  # Glass/liquid/SSS
    has_roughness_variation: bool = False  # Multiple roughness values

    # Hero object refinement
    modifier_count: int = 0  # Subdivision, bevel, solidify, etc.
    has_subdivision: bool = False
    has_bevel: bool = False
    has_solidify: bool = False

    # Atmosphere / volume
    has_atmosphere: bool = False  # Volume scatter in world or standalone volume

    # Lighting
    light_count: int = 0
    light_types: List[str] = field(default_factory=list)  # POINT, SUN, AREA, SPOT

    # Summary score (0-100, rough proxy for script richness)
    coverage_score: float = 0.0


def compute_lookdev_coverage(script_path: str) -> LookDevCoverage:
    """Compute deterministic look-dev coverage metrics from a script file.

    No LLM cost — pure regex/string analysis of the Python source.
    """
    path = Path(script_path)
    if not path.exists():
        return LookDevCoverage()

    content = path.read_text()
    c = content.lower()
    cov = LookDevCoverage()

    # -- Color management --
    for cm_type in ("agx", "filmic", "khronos pbr neutral"):
        if cm_type in c:
            cov.has_color_management = True
            cov.color_management_type = cm_type
            break
    if "view_transform" in c:
        cov.has_color_management = True
        if not cov.color_management_type:
            cov.color_management_type = "custom"

    # -- DOF --
    cov.has_dof = "use_dof" in c and ("true" in c[c.index("use_dof"):c.index("use_dof") + 30] if "use_dof" in c else False)
    if not cov.has_dof:
        cov.has_dof = bool(re.search(r"\.use_dof\s*=\s*True", content))

    # -- World setup --
    if "world" in c and ("node_tree" in c or "background" in c or "use_nodes" in c):
        cov.has_world_setup = True
    if "sky_texture" in c or "ShaderNodeTexSky" in content:
        cov.world_strategy = "procedural_sky"
    elif "environment_texture" in c or "ShaderNodeTexEnvironment" in content:
        cov.world_strategy = "env_texture"
    elif re.search(r"world.*color.*=.*\(\s*0", content):
        cov.world_strategy = "dark_world"
    elif "colorramp" in c and "world" in c:
        cov.world_strategy = "gradient"
    if cov.world_strategy:
        cov.has_world_setup = True

    # -- Materials --
    cov.material_count = len(re.findall(r"bpy\.data\.materials\.new\(", content))
    cov.principled_bsdf_count = len(re.findall(r"ShaderNodeBsdfPrincipled|Principled BSDF", content))
    cov.has_transmission = bool(re.search(r"Transmission Weight|transmission|\.ior\s*=|IOR", content, re.IGNORECASE))
    roughness_values = re.findall(r"roughness.*?=\s*([0-9.]+)", c)
    if len(set(roughness_values)) > 1:
        cov.has_roughness_variation = True

    # -- Modifiers (hero object refinement) --
    modifier_types = re.findall(r"type\s*=\s*['\"](\w+)['\"]", content)
    refinement_mods = {"SUBSURF", "BEVEL", "SOLIDIFY", "SMOOTH", "EDGE_SPLIT", "WEIGHTED_NORMAL", "REMESH"}
    for mt in modifier_types:
        if mt in refinement_mods:
            cov.modifier_count += 1
    cov.has_subdivision = "SUBSURF" in content
    cov.has_bevel = "BEVEL" in content or "bevel" in c
    cov.has_solidify = "SOLIDIFY" in content

    # -- Atmosphere --
    cov.has_atmosphere = bool(
        re.search(r"ShaderNodeVolumeScatter|Volume Scatter|volume_scatter", content)
        or re.search(r"ShaderNodeVolumePrincipled|Principled Volume", content)
    )

    # -- Lighting --
    light_matches = re.findall(r"type\s*=\s*['\"](\w+)['\"]", content)
    for lt in ("POINT", "SUN", "AREA", "SPOT"):
        count = light_matches.count(lt)
        if count > 0:
            cov.light_count += count
            cov.light_types.append(lt)

    # -- Coverage score (weighted proxy) --
    score = 0.0
    if cov.has_color_management:
        score += 15
    if cov.has_dof:
        score += 10
    if cov.has_world_setup:
        score += 10
    if cov.world_strategy:
        score += 5
    if cov.material_count >= 2:
        score += 10
    elif cov.material_count >= 1:
        score += 5
    if cov.has_transmission:
        score += 5
    if cov.has_roughness_variation:
        score += 5
    if cov.modifier_count >= 2:
        score += 15
    elif cov.modifier_count >= 1:
        score += 10
    if cov.has_atmosphere:
        score += 10
    if cov.light_count >= 3:
        score += 15
    elif cov.light_count >= 2:
        score += 10
    elif cov.light_count >= 1:
        score += 5
    cov.coverage_score = min(100.0, score)

    return cov


@function_tool
async def compute_lookdev_coverage_tool(script_path: str) -> str:
    """Compute deterministic look-dev coverage metrics for a Blender script.

    Returns coverage metrics including: color management, DOF, world strategy,
    material count, hero object refinement (modifiers), atmosphere presence,
    and lighting. Also returns a coverage_score (0-100) as a richness proxy.

    Zero LLM cost — pure static analysis.

    Args:
        script_path: Absolute path to the Blender Python script

    Returns:
        JSON with look-dev coverage metrics
    """
    cov = compute_lookdev_coverage(script_path)
    return json.dumps(asdict(cov), indent=2)


# For direct import in orchestrator
analyze_script_structure_impl = analyze_script_structure
