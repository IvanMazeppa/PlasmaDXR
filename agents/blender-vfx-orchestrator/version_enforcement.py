"""
VERSION ENFORCEMENT - STOPS AI TRAINING DATA CONTAMINATION

This module CRASHES LOUDLY if outdated patterns are detected.
It is imported at startup and validates all critical assumptions.

THE PROBLEM:
AI models revert to training data instead of using current documentation.
This causes hallucinated model names, API attributes, and SDK patterns.

THE SOLUTION:
Make wrong code fail immediately with clear error messages pointing
to the correct documentation.

CURRENT VERSIONS (2026-01):
- OpenAI Models: gpt-5.2, gpt-5-mini, o3, o4-mini (NOT gpt-4o, gpt-4-mini!)
- Blender: 5.0 (NOT 2.8x, 3.x, 4.x!)
- Agents SDK: v0.8.3 (NOT pre-v0.6!)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

# =============================================================================
# AUTHORITATIVE VERSION CONSTANTS
# =============================================================================

# OpenAI Models - CURRENT as of 2026-01
VALID_OPENAI_MODELS: Set[str] = {
    # GPT-5 family (current)
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5",
    # o-series (current)
    "o3",
    "o3-mini",
    "o4-mini",
    # Codex (current)
    "codex",
    "gpt-5.2-codex",
}

DEPRECATED_OPENAI_MODELS: Set[str] = {
    # GPT-4 family (DEPRECATED - DO NOT USE)
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    # GPT-3.5 (ANCIENT - DO NOT USE)
    "gpt-3.5-turbo",
    "gpt-3.5",
}

# Blender API - attributes that DON'T EXIST in 5.0
INVALID_BLENDER_ATTRIBUTES: Set[str] = {
    # FluidDomainSettings - hallucinated/old
    "resolution_divisions",      # → resolution_max
    "use_adaptive_time_steps",   # → use_adaptive_timesteps (no underscore)
    "use_dissolve",              # → use_dissolve_smoke
    "adaptive_domain",           # → use_adaptive_domain
    "noise_res_factor",          # REMOVED in 5.0
    "use_caching",               # REMOVED in 5.0
    "cache_format",              # → cache_data_format (missing 'data')
    # FluidFlowSettings - hallucinated/old
    "absolute_density",          # → use_absolute (flag) + density (value)
    # "velocity" as standalone doesn't exist - use velocity_factor, velocity_normal, velocity_random
    # Old node names
    "ShaderNodeSeparateRGB",     # → ShaderNodeSeparateColor
    "ShaderNodeCombineRGB",      # → ShaderNodeCombineColor
    # Old type names
    "bpy.types.Lamp",            # → bpy.types.Light
}

# Agents SDK - patterns from OLD versions
DEPRECATED_SDK_PATTERNS: List[Tuple[str, str]] = [
    # Old import patterns
    (r"from openai\.agents", "Use: from agents import ..."),
    (r"import openai\.agents", "Use: from agents import ..."),
    # Old class names
    (r"AgentRunner", "Use: Runner (not AgentRunner)"),
    (r"ToolDefinition", "Use: function_tool decorator"),
    # Old method names
    (r"\.run_sync\(", "Use: await Runner.run() with async"),
    (r"agent\.execute\(", "Use: await Runner.run(agent, prompt)"),
    # Old handoff patterns (deprecated in favor of agents-as-tools)
    (r"transfer_to_agent", "Use: agent.as_tool() pattern instead of handoffs"),
]

# =============================================================================
# ENFORCEMENT FUNCTIONS
# =============================================================================

class VersionViolationError(Exception):
    """Raised when outdated/invalid patterns are detected."""
    pass


def check_model_name(model: str, context: str = "") -> None:
    """
    Validate that a model name is current.

    Args:
        model: Model name to check
        context: Where this model name was found (for error message)

    Raises:
        VersionViolationError: If model is deprecated
    """
    model_lower = model.lower().strip()

    if model_lower in DEPRECATED_OPENAI_MODELS:
        raise VersionViolationError(
            f"\n{'='*70}\n"
            f"DEPRECATED MODEL DETECTED: '{model}'\n"
            f"Context: {context}\n"
            f"{'='*70}\n"
            f"\n"
            f"This model is from AI training data, NOT current reality.\n"
            f"\n"
            f"VALID MODELS (2026-01):\n"
            f"  - gpt-5.2      (high capability)\n"
            f"  - gpt-5-mini   (fast, cheap)\n"
            f"  - o3           (reasoning)\n"
            f"  - o4-mini      (fast reasoning)\n"
            f"\n"
            f"DO NOT USE: gpt-4o, gpt-4-mini, gpt-4, gpt-3.5-turbo\n"
            f"{'='*70}\n"
        )


def check_blender_attribute(attr: str, context: str = "") -> None:
    """
    Validate that a Blender attribute exists in 5.0.

    Args:
        attr: Attribute name to check
        context: Where this attribute was found

    Raises:
        VersionViolationError: If attribute doesn't exist in Blender 5.0
    """
    if attr in INVALID_BLENDER_ATTRIBUTES:
        raise VersionViolationError(
            f"\n{'='*70}\n"
            f"INVALID BLENDER 5.0 ATTRIBUTE: '{attr}'\n"
            f"Context: {context}\n"
            f"{'='*70}\n"
            f"\n"
            f"This attribute is from AI training data (Blender 2.8-4.x).\n"
            f"It does NOT exist in Blender 5.0.\n"
            f"\n"
            f"CONSULT DOCUMENTATION:\n"
            f"  https://docs.blender.org/api/current/\n"
            f"\n"
            f"Use semantic_search_blender_docs() to find correct attribute.\n"
            f"{'='*70}\n"
        )


def scan_file_for_violations(filepath: Path) -> List[str]:
    """
    Scan a Python file for version violations.

    Args:
        filepath: Path to Python file

    Returns:
        List of violation descriptions
    """
    violations = []

    try:
        content = filepath.read_text()
    except Exception:
        return violations

    # Check for deprecated models
    for model in DEPRECATED_OPENAI_MODELS:
        # Match model in strings or assignments
        patterns = [
            rf'["\']({re.escape(model)})["\']',
            rf'model\s*=\s*["\']({re.escape(model)})["\']',
        ]
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(
                    f"{filepath}: Deprecated model '{model}' - use gpt-5.2 or gpt-5-mini"
                )
                break

    # Check for invalid Blender attributes
    for attr in INVALID_BLENDER_ATTRIBUTES:
        if f".{attr}" in content or f"['{attr}']" in content or f'["{attr}"]' in content:
            violations.append(
                f"{filepath}: Invalid Blender 5.0 attribute '{attr}'"
            )

    # Check for deprecated SDK patterns
    for pattern, fix in DEPRECATED_SDK_PATTERNS:
        if re.search(pattern, content):
            violations.append(
                f"{filepath}: Deprecated SDK pattern - {fix}"
            )

    return violations


def scan_directory(directory: Path, exclude_dirs: Set[str] = None) -> List[str]:
    """
    Scan all Python files in directory for violations.

    Args:
        directory: Root directory to scan
        exclude_dirs: Directory names to skip

    Returns:
        List of all violations found
    """
    if exclude_dirs is None:
        exclude_dirs = {"venv", ".venv", "__pycache__", ".git", "node_modules"}

    all_violations = []

    for filepath in directory.rglob("*.py"):
        # Skip excluded directories
        if any(excl in filepath.parts for excl in exclude_dirs):
            continue

        violations = scan_file_for_violations(filepath)
        all_violations.extend(violations)

    return all_violations


def enforce_on_startup(strict: bool = False) -> None:
    """
    Run enforcement checks at module import time.

    Args:
        strict: If True, raise exception on violations. If False, print warnings.
    """
    # Get the orchestrator directory
    this_dir = Path(__file__).parent

    violations = scan_directory(this_dir)

    if violations:
        msg = (
            f"\n{'='*70}\n"
            f"VERSION ENFORCEMENT: {len(violations)} VIOLATIONS DETECTED\n"
            f"{'='*70}\n\n"
        )
        for v in violations[:10]:  # Limit output
            msg += f"  - {v}\n"
        if len(violations) > 10:
            msg += f"  ... and {len(violations) - 10} more\n"
        msg += (
            f"\n"
            f"These are likely from AI training data contamination.\n"
            f"AI models must use current documentation, not memory.\n"
            f"{'='*70}\n"
        )

        if strict:
            raise VersionViolationError(msg)
        else:
            print(msg, file=sys.stderr)


# =============================================================================
# RUNTIME VALIDATORS (call these before using values)
# =============================================================================

def validate_model_config(config: dict) -> None:
    """
    Validate all model names in a configuration dict.

    Args:
        config: Configuration dict that may contain model names

    Raises:
        VersionViolationError: If any deprecated models found
    """
    def _check_recursive(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if k in ("model", "model_name") and isinstance(v, str):
                    check_model_name(v, context=new_path)
                _check_recursive(v, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check_recursive(item, f"{path}[{i}]")

    _check_recursive(config)


def validate_script_content(script: str, script_name: str = "unknown") -> List[str]:
    """
    Validate Blender script content for invalid attributes.

    Args:
        script: Python script content
        script_name: Name for error messages

    Returns:
        List of invalid attributes found
    """
    invalid_found = []

    for attr in INVALID_BLENDER_ATTRIBUTES:
        if f".{attr}" in script or f"['{attr}']" in script or f'["{attr}"]' in script:
            invalid_found.append(attr)

    return invalid_found


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan for AI training data contamination (outdated patterns)"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory or file to scan"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if violations found"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if path.is_file():
        violations = scan_file_for_violations(path)
    else:
        violations = scan_directory(path)

    if violations:
        print(f"\n{'='*70}")
        print(f"VIOLATIONS FOUND: {len(violations)}")
        print(f"{'='*70}\n")
        for v in violations:
            print(f"  - {v}")
        print(f"\n{'='*70}")
        print("These patterns are from outdated AI training data.")
        print("Consult current documentation before making changes.")
        print(f"{'='*70}\n")

        if args.strict:
            sys.exit(1)
    else:
        print("No violations found.")
