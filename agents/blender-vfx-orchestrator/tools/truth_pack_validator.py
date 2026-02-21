"""
Truth Pack Validator: Validates and auto-fixes generated scripts.

Wraps truth_pack.py validation into a convenient pipeline step and
provides an @function_tool for agent use.

Usage:
    # Direct call from orchestrator
    fixed_path, fixes = validate_and_fix_script(script_path, truth_pack)

    # As agent tool
    result = await validate_script_truth_pack(script_path="/path/to/script.py")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents import function_tool

from tools.truth_pack import (
    ValidationError,
    validate_script_against_truth_pack,
    auto_fix_script,
    build_substitution_table,
)


def validate_and_fix_script(
    script_path: str,
    truth_pack: Dict[str, Any],
    max_fix_attempts: int = 2,
) -> Tuple[str, List[str]]:
    """
    Read a script, validate against truth pack, auto-fix, and write back.

    Args:
        script_path: Path to the Blender Python script
        truth_pack: Truth pack from build_truth_pack()
        max_fix_attempts: Maximum fix-then-revalidate cycles

    Returns:
        Tuple of (fixed_script_path, list_of_all_fixes_applied)

    Raises:
        FileNotFoundError: If script_path doesn't exist
        ValueError: If unfixable errors remain after max attempts
    """
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    script = path.read_text()
    all_fixes = []

    for attempt in range(max_fix_attempts):
        errors = validate_script_against_truth_pack(script, truth_pack)

        if not errors:
            if all_fixes:
                print(f"[TruthPackValidator] Script clean after {len(all_fixes)} fixes",
                      file=sys.stderr)
            else:
                print(f"[TruthPackValidator] Script clean — no fixes needed",
                      file=sys.stderr)
            break

        print(f"[TruthPackValidator] Attempt {attempt + 1}: "
              f"{len(errors)} validation errors found", file=sys.stderr)

        # Log errors
        for err in errors[:10]:  # Cap logging at 10
            print(f"  Line {err.line}: {err.message}"
                  f"{f' → suggest: {err.suggestion}' if err.suggestion else ''}",
                  file=sys.stderr)

        # Auto-fix
        script, fixes = auto_fix_script(script, errors, truth_pack)
        all_fixes.extend(fixes)

        if not fixes:
            # No fixes could be applied — remaining errors are unfixable
            unfixable = [
                e for e in errors
                if e.object_type != "KNOWN_HALLUCINATION" and e.suggestion is None
            ]
            if unfixable:
                msg = "; ".join(
                    f"Line {e.line}: {e.attribute} on {e.object_type}"
                    for e in unfixable[:5]
                )
                print(f"[TruthPackValidator] WARNING: {len(unfixable)} unfixable errors: {msg}",
                      file=sys.stderr)
            break

    # Write fixed script back
    if all_fixes:
        path.write_text(script)
        print(f"[TruthPackValidator] Wrote fixed script: {path}", file=sys.stderr)

    return str(path), all_fixes


def format_validation_report(
    errors: List[ValidationError],
    fixes: List[str],
) -> str:
    """
    Format validation results as dense LLM-readable text.

    Args:
        errors: Validation errors found
        fixes: Fixes that were applied

    Returns:
        Formatted report string
    """
    lines = []

    if not errors and not fixes:
        return "VALIDATION: PASS — No invalid attributes detected."

    if fixes:
        lines.append(f"VALIDATION: {len(fixes)} fixes applied:")
        for fix in fixes:
            lines.append(f"  - {fix}")

    # Check for remaining errors after fixes
    if errors:
        remaining = [e for e in errors if e.suggestion is None]
        if remaining:
            lines.append(f"WARNING: {len(remaining)} unfixable errors:")
            for err in remaining[:5]:
                lines.append(f"  Line {err.line}: {err.message}")

    return "\n".join(lines)


# =============================================================================
# AGENT TOOL WRAPPER
# =============================================================================

# Module-level truth pack reference (set by orchestrator before agent runs)
_current_truth_pack: Optional[Dict[str, Any]] = None


def set_truth_pack(truth_pack: Dict[str, Any]) -> None:
    """Set the current truth pack for agent tool use."""
    global _current_truth_pack
    _current_truth_pack = truth_pack


@function_tool
async def validate_script_truth_pack(
    script_path: str,
) -> str:
    """Validate a Blender script against the truth pack and auto-fix invalid attributes.

    Checks every attribute access (domain_settings.X, flow_settings.X, etc.)
    against the truth pack built from Blender introspection. Auto-fixes
    known hallucinations and close matches.

    Args:
        script_path: Absolute path to the Blender Python script to validate

    Returns:
        Validation report with fixes applied
    """
    if _current_truth_pack is None:
        return "ERROR: Truth pack not loaded. Build truth pack first."

    try:
        fixed_path, fixes = validate_and_fix_script(script_path, _current_truth_pack)

        # Re-validate to check remaining issues
        script = Path(fixed_path).read_text()
        remaining = validate_script_against_truth_pack(script, _current_truth_pack)

        return format_validation_report(remaining, fixes)

    except FileNotFoundError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: Validation failed: {e}"
