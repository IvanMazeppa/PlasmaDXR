"""
Phase 2A-3: Tool-level guardrails for truth pack enforcement.

These are TOOL guardrails (ToolInputGuardrail / ToolOutputGuardrail) — they wrap
individual tools, not agents. They run automatically before/after every invocation
of the guarded tool, making validation impossible to bypass.

Cost: $0 (pure Python, no LLM calls).

Three guardrails:
1. truth_pack_input_guardrail — validates scripts against truth pack BEFORE execution
2. critical_failure_output_guardrail — catches BLACK_SCREEN/ZERO_LIGHTS in evaluation output
3. script_length_output_guardrail — warns when generated scripts are too basic (<500 lines)

Usage:
    These guardrails are attached to tools in their respective modules
    via attach_tool_guardrails() called during orchestrator initialization.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agents import (
    tool_input_guardrail,
    tool_output_guardrail,
    ToolGuardrailFunctionOutput,
)

# Path setup for truth pack imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from tools.truth_pack import (
    validate_script_against_truth_pack,
    auto_fix_script,
)
from tools.truth_pack_validator import set_truth_pack as _get_truth_pack_ref

# Module-level truth pack reference (set by orchestrator before runs)
_truth_pack: Optional[Dict[str, Any]] = None

# Kill switch
ENABLE_TOOL_GUARDRAILS = os.getenv("ENABLE_TOOL_GUARDRAILS", "true").lower() != "false"

# Script length thresholds
MIN_SCRIPT_LINES_WARNING = 500
MIN_SCRIPT_LINES_BASIC = 300

# Critical failure keywords in quality evaluation output
CRITICAL_FAILURE_KEYWORDS = [
    "ZERO_LIGHTS",
    "BLACK_SCREEN",
    "WHITE_SCREEN",
    "CLIPPING_ARTIFACTS",
]


def set_truth_pack(truth_pack: Dict[str, Any]) -> None:
    """Set the truth pack for guardrail validation. Called by orchestrator."""
    global _truth_pack
    _truth_pack = truth_pack


def _parse_tool_arguments(data: Any) -> Dict[str, Any]:
    """Extract tool arguments from guardrail data context."""
    ctx = getattr(data, "context", None)
    raw = getattr(ctx, "tool_arguments", None) if ctx else None
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# =============================================================================
# 1. TRUTH PACK INPUT GUARDRAIL (on execute_blender_script)
# =============================================================================

@tool_input_guardrail
def truth_pack_input_guardrail(data: Any) -> ToolGuardrailFunctionOutput:
    """
    Validate script against truth pack BEFORE Blender execution.

    Reads the script_path from tool arguments, validates all attribute
    accesses against the truth pack, and auto-fixes hallucinations.
    If unfixable errors remain, the tool call is rejected.
    """
    if not ENABLE_TOOL_GUARDRAILS or _truth_pack is None:
        return ToolGuardrailFunctionOutput.allow()

    args = _parse_tool_arguments(data)
    script_path = args.get("script_path", "")
    if not script_path:
        return ToolGuardrailFunctionOutput.allow()

    path = Path(script_path)
    if not path.exists():
        # Let the tool itself handle FileNotFoundError
        return ToolGuardrailFunctionOutput.allow()

    try:
        script = path.read_text()
    except Exception:
        return ToolGuardrailFunctionOutput.allow()

    # Validate
    errors = validate_script_against_truth_pack(script, _truth_pack)
    if not errors:
        return ToolGuardrailFunctionOutput.allow()

    # Auto-fix
    fixed_script, fixes = auto_fix_script(script, errors, _truth_pack)

    if fixes:
        # Write fixed script back
        path.write_text(fixed_script)
        print(
            f"[ToolGuardrail] Truth pack auto-fixed {len(fixes)} issues in {path.name}",
            file=sys.stderr,
        )

    # Re-validate after fixes
    remaining = validate_script_against_truth_pack(fixed_script, _truth_pack)
    unfixable = [e for e in remaining if e.suggestion is None]

    if unfixable:
        msg_parts = [f"Line {e.line}: {e.attribute} on {e.object_type}" for e in unfixable[:5]]
        return ToolGuardrailFunctionOutput.reject_content(
            f"Script has {len(unfixable)} unfixable truth pack errors after auto-fix. "
            f"Errors: {'; '.join(msg_parts)}. "
            f"Regenerate the script with correct Blender 5.0 API."
        )

    # All errors were auto-fixed — allow execution with the fixed script
    return ToolGuardrailFunctionOutput.allow()


# =============================================================================
# 2. CRITICAL FAILURE OUTPUT GUARDRAIL (on evaluate_render / analyze_with_vision)
# =============================================================================

@tool_output_guardrail
def critical_failure_output_guardrail(data: Any) -> ToolGuardrailFunctionOutput:
    """
    Catch critical rendering failures in quality evaluation output.

    When BLACK_SCREEN, ZERO_LIGHTS, WHITE_SCREEN, or CLIPPING_ARTIFACTS
    are detected, reject with a targeted message so the agent doesn't
    waste iterations on minor tweaks.
    """
    if not ENABLE_TOOL_GUARDRAILS:
        return ToolGuardrailFunctionOutput.allow()

    output = str(getattr(data, "output", "") or "")

    found = [kw for kw in CRITICAL_FAILURE_KEYWORDS if kw in output]
    if found:
        return ToolGuardrailFunctionOutput.reject_content(
            f"CRITICAL rendering failure: {', '.join(found)}. "
            f"Do NOT iterate on minor parameters — this requires a fundamental fix "
            f"(lighting setup, camera position, or technique switch)."
        )

    return ToolGuardrailFunctionOutput.allow()


# =============================================================================
# 3. SCRIPT LENGTH OUTPUT GUARDRAIL (on generate_script)
# =============================================================================

@tool_output_guardrail
def script_length_output_guardrail(data: Any) -> ToolGuardrailFunctionOutput:
    """
    Check generated script length to catch overly basic scripts.

    Scripts under 500 lines typically lack proper scene setup (lighting,
    camera, materials). This guardrail injects a warning into the output
    but does not reject — the script may still be valid for simple effects.
    """
    if not ENABLE_TOOL_GUARDRAILS:
        return ToolGuardrailFunctionOutput.allow()

    output = getattr(data, "output", None)
    if not output:
        return ToolGuardrailFunctionOutput.allow()

    # Parse the generate_script output to find the script path
    script_path = None
    try:
        if isinstance(output, str):
            payload = json.loads(output)
        elif isinstance(output, dict):
            payload = output
        else:
            return ToolGuardrailFunctionOutput.allow()
        script_path = payload.get("script_path")
    except (json.JSONDecodeError, TypeError):
        return ToolGuardrailFunctionOutput.allow()

    if not script_path:
        return ToolGuardrailFunctionOutput.allow()

    try:
        text = Path(script_path).read_text()
        line_count = len(text.splitlines())
    except Exception:
        return ToolGuardrailFunctionOutput.allow()

    if line_count < MIN_SCRIPT_LINES_BASIC:
        return ToolGuardrailFunctionOutput.reject_content(
            f"Generated script is only {line_count} lines — too basic for a complete scene. "
            f"Minimum {MIN_SCRIPT_LINES_BASIC} lines expected. Regenerate with full scene setup "
            f"(domain, emitter, materials, lighting, camera, bake, render)."
        )

    if line_count < MIN_SCRIPT_LINES_WARNING:
        # Allow but the warning is surfaced in output_info
        return ToolGuardrailFunctionOutput.allow(
            output_info={
                "warning": f"Script is {line_count} lines — may lack full scene setup. "
                f"Scripts 500+ lines typically produce better results."
            }
        )

    return ToolGuardrailFunctionOutput.allow()


# =============================================================================
# ATTACHMENT FUNCTION
# =============================================================================

def attach_tool_guardrails() -> dict:
    """
    Attach tool guardrails to their target tools.

    Called during orchestrator initialization, after truth pack is loaded.
    Mutates FunctionTool instances directly (they're mutable dataclasses).

    Returns:
        Dict with attachment status for logging.
    """
    from tools.blender_executor_tools import execute_blender_script
    from tools.asset_evaluator_tools import evaluate_render, analyze_with_vision
    from tools.script_generator_tools import generate_script

    attached = {}

    # 1. Truth pack input guardrail on execute_blender_script
    if execute_blender_script.tool_input_guardrails is None:
        execute_blender_script.tool_input_guardrails = []
    execute_blender_script.tool_input_guardrails.append(truth_pack_input_guardrail)
    attached["execute_blender_script"] = "truth_pack_input_guardrail"

    # 2. Critical failure output guardrail on evaluate_render and analyze_with_vision
    for tool in (evaluate_render, analyze_with_vision):
        if tool.tool_output_guardrails is None:
            tool.tool_output_guardrails = []
        tool.tool_output_guardrails.append(critical_failure_output_guardrail)
        attached[tool.name] = "critical_failure_output_guardrail"

    # 3. Script length output guardrail on generate_script
    if generate_script.tool_output_guardrails is None:
        generate_script.tool_output_guardrails = []
    generate_script.tool_output_guardrails.append(script_length_output_guardrail)
    attached["generate_script"] = "script_length_output_guardrail"

    return attached
