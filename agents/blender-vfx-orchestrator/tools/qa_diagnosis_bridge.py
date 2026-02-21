"""
QA Diagnosis Bridge: Grounds QA visual feedback in actual script code.

Solves the broken feedback loop where QA sees the render but not the code,
leading to suggestions like "too dark" instead of actionable feedback like
"Line 245: light energy=10, truth pack range [0, 1000000], suggest 100-500".

Usage:
    grounded = create_code_grounded_feedback(quality_output, script_path, truth_pack)
    # Returns dense LLM-optimized text with line numbers and variable names
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.script_analysis_tools import (
    analyze_script_structure,
    ScriptAnalysis,
    ModifiablePattern,
)


# =============================================================================
# ISSUE → SCRIPT PARAMETER MAPPING
# =============================================================================

# Maps QA issue keywords to script analysis categories and patterns
ISSUE_TO_PARAMS: Dict[str, List[Dict[str, Any]]] = {
    "dark": [
        {"category": "config_class", "keywords": ["LIGHT", "ENERGY", "POWER", "BRIGHTNESS", "INTENSITY"]},
        {"category": "settings_attr", "keywords": ["energy", "power", "intensity", "strength"]},
        {"category": "shader_node", "keywords": ["Emission Strength", "Color", "Density", "Blackbody Intensity"]},
    ],
    "dim": [
        {"category": "config_class", "keywords": ["LIGHT", "ENERGY", "POWER", "EMISSION"]},
        {"category": "settings_attr", "keywords": ["energy", "power", "emission"]},
        {"category": "shader_node", "keywords": ["Emission Strength", "Blackbody Intensity"]},
    ],
    "bright": [
        {"category": "config_class", "keywords": ["LIGHT", "ENERGY", "POWER", "EXPOSURE"]},
        {"category": "settings_attr", "keywords": ["energy", "power", "film_exposure"]},
        {"category": "shader_node", "keywords": ["Emission Strength"]},
    ],
    "overexposed": [
        {"category": "config_class", "keywords": ["LIGHT", "ENERGY", "EXPOSURE"]},
        {"category": "settings_attr", "keywords": ["energy", "film_exposure"]},
    ],
    "density": [
        {"category": "config_class", "keywords": ["DENSITY", "FLAME_SMOKE"]},
        {"category": "settings_attr", "keywords": ["density", "flame_smoke"]},
        {"category": "shader_node", "keywords": ["Density"]},
    ],
    "smoke": [
        {"category": "config_class", "keywords": ["DENSITY", "SMOKE", "DISSOLVE", "VORTICITY"]},
        {"category": "settings_attr", "keywords": ["density", "use_dissolve_smoke", "dissolve_speed", "vorticity"]},
        {"category": "shader_node", "keywords": ["Density", "Anisotropy"]},
    ],
    "fire": [
        {"category": "config_class", "keywords": ["FLAME", "TEMPERATURE", "BURNING", "FIRE"]},
        {"category": "settings_attr", "keywords": ["flame_smoke", "temperature", "burning_rate"]},
        {"category": "shader_node", "keywords": ["Blackbody Intensity", "Temperature"]},
    ],
    "resolution": [
        {"category": "config_class", "keywords": ["RESOLUTION", "RES"]},
        {"category": "settings_attr", "keywords": ["resolution_max", "use_noise", "noise_scale"]},
    ],
    "clipping": [
        {"category": "config_class", "keywords": ["DOMAIN", "SIZE", "SCALE"]},
        {"category": "settings_attr", "keywords": ["domain_type", "use_adaptive_domain"]},
    ],
    "camera": [
        {"category": "config_class", "keywords": ["CAMERA", "CAM"]},
        {"category": "settings_attr", "keywords": ["location", "rotation", "lens"]},
    ],
    "liquid": [
        {"category": "config_class", "keywords": ["VISCOSITY", "SURFACE", "PARTICLE"]},
        {"category": "settings_attr", "keywords": ["viscosity_base", "use_flip_particles", "use_plane_init"]},
    ],
    "empty": [
        {"category": "config_class", "keywords": ["FLOW", "DENSITY", "VELOCITY"]},
        {"category": "settings_attr", "keywords": ["flow_behavior", "density", "velocity_normal", "use_plane_init"]},
    ],
}


def create_code_grounded_feedback(
    quality_output: Any,
    script_path: str,
    truth_pack: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create code-grounded feedback by pairing QA visual critique with script analysis.

    Transforms vague feedback like "too dark" into actionable feedback like:
    "Line 245: light energy=10, truth pack range [0, 1000000], suggest 100-500"

    Args:
        quality_output: QualityOutput from the QA phase (has issues, suggestions, primary_issue)
        script_path: Path to the current Blender script
        truth_pack: Optional truth pack for range information

    Returns:
        Dense, LLM-optimized feedback string with line numbers and variable names
    """
    if not script_path or not Path(script_path).exists():
        return "WARNING: No script available for code-grounded feedback."

    # Analyze the script
    analysis = analyze_script_structure(script_path)
    if analysis.total_modifiable == 0:
        return "WARNING: Script has no modifiable patterns detected."

    # Extract issues from quality output
    issues = _extract_issues(quality_output)
    if not issues:
        return "No issues identified — script appears to be working correctly."

    # Build grounded feedback
    sections = []
    sections.append("## CODE-GROUNDED FEEDBACK (from script analysis)")
    sections.append(f"Script: {script_path} ({analysis.total_modifiable} modifiable patterns)")
    sections.append("")

    matched_any = False

    for issue_text in issues:
        matches = _find_relevant_params(issue_text, analysis, truth_pack)
        if matches:
            matched_any = True
            sections.append(f"### Issue: {issue_text}")
            for match in matches:
                sections.append(match)
            sections.append("")

    # If no matches found, provide full script overview
    if not matched_any:
        sections.append("### No direct parameter matches for reported issues")
        sections.append("Available modifiable parameters:")
        sections.extend(_format_all_params(analysis, truth_pack))

    return "\n".join(sections)


def _extract_issues(quality_output: Any) -> List[str]:
    """Extract issue strings from QualityOutput."""
    issues = []
    if hasattr(quality_output, 'primary_issue') and quality_output.primary_issue:
        issues.append(quality_output.primary_issue)
    if hasattr(quality_output, 'issues'):
        for issue in quality_output.issues:
            if issue and issue not in issues:
                issues.append(issue)
    if hasattr(quality_output, 'suggestions'):
        for suggestion in quality_output.suggestions[:3]:  # Limit suggestions
            if suggestion and suggestion not in issues:
                issues.append(suggestion)
    return issues[:5]  # Cap at 5 issues


def _find_relevant_params(
    issue_text: str,
    analysis: ScriptAnalysis,
    truth_pack: Optional[Dict[str, Any]],
) -> List[str]:
    """Find script parameters relevant to an issue."""
    lines = []
    issue_lower = issue_text.lower()

    # Find which categories match this issue
    for keyword, param_specs in ISSUE_TO_PARAMS.items():
        if keyword not in issue_lower:
            continue

        for spec in param_specs:
            category = spec["category"]
            search_keywords = spec["keywords"]

            patterns = _get_patterns_by_category(analysis, category)

            for kw in search_keywords:
                for name, pattern in patterns.items():
                    if kw.lower() in name.lower():
                        line = _format_param_with_truth_pack(
                            name, pattern, truth_pack
                        )
                        if line not in lines:
                            lines.append(line)

    return lines


def _get_patterns_by_category(
    analysis: ScriptAnalysis,
    category: str,
) -> Dict[str, ModifiablePattern]:
    """Get patterns from analysis by category name."""
    if category == "config_class":
        return analysis.config_values
    elif category == "settings_attr":
        return analysis.settings_assignments
    elif category == "shader_node":
        return analysis.shader_node_inputs
    else:
        return {}


def _format_param_with_truth_pack(
    name: str,
    pattern: ModifiablePattern,
    truth_pack: Optional[Dict[str, Any]],
) -> str:
    """Format a parameter with truth pack range information."""
    line = f"- Line {pattern.line_number}: {name} = {pattern.current_value}"

    # Try to find truth pack range for this parameter
    if truth_pack and pattern.pattern_type == "settings_attr":
        # Extract the attribute name from patterns like "settings.resolution_max"
        attr_name = name.split(".")[-1] if "." in name else name

        for type_name, type_data in truth_pack.items():
            if not isinstance(type_data, dict) or "properties" not in type_data:
                continue
            props = type_data["properties"]
            if attr_name in props:
                prop_info = props[attr_name]
                if "range" in prop_info:
                    r = prop_info["range"]
                    line += f" (range: [{r[0]}, {r[1]}])"
                if "default" in prop_info:
                    line += f" (default: {prop_info['default']})"
                break

    return line


def _format_all_params(
    analysis: ScriptAnalysis,
    truth_pack: Optional[Dict[str, Any]],
) -> List[str]:
    """Format all modifiable parameters as a summary."""
    lines = []
    all_patterns = {}
    all_patterns.update(analysis.config_values)
    all_patterns.update(analysis.settings_assignments)
    all_patterns.update(analysis.shader_node_inputs)

    for name, pattern in sorted(all_patterns.items(), key=lambda x: x[1].line_number):
        lines.append(_format_param_with_truth_pack(name, pattern, truth_pack))

    return lines[:20]  # Cap at 20 parameters
