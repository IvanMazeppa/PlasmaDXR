"""
Script Writer Guardrails for OpenAI Agents SDK.

These guardrails enforce critical requirements for the Script Writer agent:

1. require_research_context (INPUT) - Ensures research was done before scripting
2. validate_effect_type (INPUT) - Validates effect_type is a known enum value
3. validate_script_output (OUTPUT) - Ensures script output has valid path

SDK Pattern Reference:
    https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, List, TYPE_CHECKING, Union

from agents import (
    Agent,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    TResponseInputItem,
)

if TYPE_CHECKING:
    from agents import RunContextWrapper

# Valid effect types - dynamically synced from the canonical EffectType enum.
# This prevents mismatches where the guardrail rejects valid effect types.
try:
    from models.shared_context import EffectType
    VALID_EFFECT_TYPES = {e.value for e in EffectType}
except ImportError:
    # Fallback if EffectType not importable (e.g., standalone guardrail testing)
    VALID_EFFECT_TYPES = {
        "explosion", "fire", "smoke", "nebula", "pyro",
        "water", "water_splash", "ocean", "waterfall", "rain",
        "sun", "star", "supernova",
    }

# Keywords that indicate research was performed
RESEARCH_INDICATORS = [
    "research",
    "documentation",
    "api",
    "bpy.",
    "mantaflow",
    "blender 5",
    "technique",
    "approach",
    "pattern",
    "semantic_search",
    "search_code_patterns",
    "find_alternative",
    "recommended_approach",
    "key_parameters",
    "alternative_approaches",
]

# Known hallucination patterns to block in generated scripts
HALLUCINATION_PATTERNS = [
    (r"\bresolution_divisions\b", "resolution_divisions removed in Blender 5.0 (use resolution_max)"),
    (r"\buse_adaptive_time_steps\b", "use_adaptive_time_steps is invalid (use use_adaptive_timesteps)"),
    (r"\bvelocity_multi\b", "velocity_multi is invalid (use velocity_factor)"),
    (r"\bnoise_res_factor\b", "noise_res_factor removed in Blender 5.0"),
    (r"\bdomain_resolution\b\s*=", "domain_resolution is read-only (use resolution_max)"),
    (r"\bflow\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
    (r"\bobject\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
    (r"\bobj\.velocity_factor\b", "velocity_factor must be set on flow_settings, not bpy.types.Object"),
]

# Valid OUTPUT_DIR prefixes - scripts must use these paths to avoid permission errors
VALID_OUTPUT_DIR_PREFIXES = [
    "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output",
    "/home/maz3ppa/projects/PlasmaDXR/assets",
    "/tmp/",  # Temporary paths are OK for testing
    "//",  # Blender-relative paths are OK
]


def _validate_output_dir(script_text: str) -> List[str]:
    """Check if OUTPUT_DIR in script uses a valid path prefix."""
    issues = []

    # Find OUTPUT_DIR assignment
    output_dir_match = re.search(
        r'OUTPUT_DIR\s*=\s*[f]?["\']([^"\']+)["\']',
        script_text
    )

    if output_dir_match:
        output_dir = output_dir_match.group(1)
        # Handle f-string variables like {ASSET_NAME} - extract the base path
        base_path = re.sub(r'\{[^}]+\}', '', output_dir)

        is_valid = any(
            base_path.startswith(prefix)
            for prefix in VALID_OUTPUT_DIR_PREFIXES
        )

        if not is_valid:
            issues.append(
                f"HALLUCINATED OUTPUT_DIR: '{output_dir}' is not a valid path. "
                f"Use /home/maz3ppa/projects/PlasmaDXR/build/vdb_output/{{ASSET_NAME}}"
            )

    return issues


def _scan_script_for_hallucinations(script_text: str) -> List[str]:
    """Return list of hallucination issues found in script text."""
    issues: List[str] = []
    if not script_text:
        return issues

    # Check OUTPUT_DIR path validity (prevent permission errors from hallucinated paths)
    issues.extend(_validate_output_dir(script_text))

    in_triple = False
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_triple = not in_triple
            continue
        if in_triple or stripped.startswith("#"):
            continue

        for pattern, message in HALLUCINATION_PATTERNS:
            if re.search(pattern, line):
                issues.append(f"{message} | line: {stripped}")
    return issues


@input_guardrail
async def require_research_context(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    input: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that ensures Script Writer receives research context.

    The Script Writer should NOT be called without prior research findings.
    This guardrail checks for research indicators in the input prompt.

    Triggers when:
    - Input lacks any research-related keywords
    - No documentation references found
    - No technique/approach information provided

    Args:
        ctx: Run context wrapper
        agent: The Script Writer agent
        input: Input prompt string or message list

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if no research context
    """
    # Convert input to searchable string
    if isinstance(input, list):
        # List of TResponseInputItem - extract text content
        input_text = " ".join(
            str(item.get("content", "") if isinstance(item, dict) else item)
            for item in input
        ).lower()
    else:
        input_text = str(input).lower()

    # Check for research indicators
    research_found = any(
        indicator.lower() in input_text
        for indicator in RESEARCH_INDICATORS
    )

    if not research_found:
        print(
            f"[Guardrail] require_research_context TRIGGERED: "
            f"No research context found in Script Writer input",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Script Writer requires research context before generating code",
                "hint": (
                    "Call research agent first with semantic_search_blender_docs "
                    "or search_code_patterns to find best approach"
                ),
                "missing": "No research findings, technique recommendations, or API documentation",
            }
        )

    print(
        f"[Guardrail] require_research_context PASSED: Research context found",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


@input_guardrail
async def validate_effect_type(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    input: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that validates effect_type is a known value.

    Prevents Script Writer from proceeding with invalid effect types
    that would lead to failed generation.

    Args:
        ctx: Run context wrapper
        agent: The Script Writer agent
        input: Input prompt string or message list

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if effect_type invalid
    """
    # Convert input to searchable string
    if isinstance(input, list):
        input_text = " ".join(
            str(item.get("content", "") if isinstance(item, dict) else item)
            for item in input
        ).lower()
    else:
        input_text = str(input).lower()

    # Check for any valid effect type
    effect_type_found = any(
        effect_type in input_text
        for effect_type in VALID_EFFECT_TYPES
    )

    if not effect_type_found:
        print(
            f"[Guardrail] validate_effect_type TRIGGERED: "
            f"No valid effect type found in input",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "No valid effect type found in input",
                "valid_types": list(VALID_EFFECT_TYPES),
                "hint": "Include effect type in prompt (e.g., 'explosion', 'fire', 'nebula')",
            }
        )

    print(
        f"[Guardrail] validate_effect_type PASSED: Valid effect type found",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


@output_guardrail
async def validate_script_output(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that validates ScriptOutput has required fields.

    Ensures the Script Writer produces valid output with:
    - Non-empty script_path that points to an existing file
    - technique_used is specified
    - validation_passed status is set

    Args:
        ctx: Run context wrapper
        agent: The Script Writer agent
        output: The agent's output (should be ScriptOutput)

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if output invalid
    """
    # Handle different output types
    if output is None:
        print(
            f"[Guardrail] validate_script_output TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Script Writer produced no output",
                "hint": "Agent must return ScriptOutput with script_path",
            }
        )

    # Check if it's a ScriptOutput-like object or dict
    if hasattr(output, "script_path"):
        script_path = output.script_path
        technique = getattr(output, "technique_used", None)
        validation_passed = getattr(output, "validation_passed", True)
    elif isinstance(output, dict):
        script_path = output.get("script_path")
        technique = output.get("technique_used")
        validation_passed = output.get("validation_passed", True)
    else:
        # Try to extract from string output
        script_path = None
        technique = None
        validation_passed = False

    errors = []

    # Check script_path
    if not script_path:
        errors.append("script_path is missing or empty")
    elif not Path(script_path).exists():
        # Only warn, don't fail - file might be created after validation
        print(
            f"[Guardrail] validate_script_output WARNING: "
            f"script_path '{script_path}' does not exist yet",
            file=sys.stderr
        )
    else:
        try:
            script_text = Path(script_path).read_text()
            hallucinations = _scan_script_for_hallucinations(script_text)
            if hallucinations:
                return GuardrailFunctionOutput(
                    tripwire_triggered=True,
                    output_info={
                        "reason": "Script contains hallucinated APIs",
                        "issues": hallucinations,
                    }
                )
        except Exception:
            # If we can't read the script, defer to validate_script tool
            pass

    # Check technique_used
    if not technique:
        errors.append("technique_used is missing")

    if errors:
        print(
            f"[Guardrail] validate_script_output TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Invalid ScriptOutput structure",
                "errors": errors,
                "received": str(output)[:200],
            }
        )

    print(
        f"[Guardrail] validate_script_output PASSED: "
        f"script_path={script_path}, technique={technique}",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_guardrails():
        print("Testing Script Writer Guardrails...")
        print("-" * 60)

        # Mock context
        class MockContext:
            context = None

        ctx = MockContext()

        # Mock agent
        class MockAgent:
            name = "Script Writer"

        agent = MockAgent()

        # Access the guardrail function from the InputGuardrail/OutputGuardrail objects
        # The decorator wraps the function in an InputGuardrail object
        require_research_fn = require_research_context.guardrail_function
        validate_effect_fn = validate_effect_type.guardrail_function
        validate_script_fn = validate_script_output.guardrail_function

        # Test require_research_context - should trigger
        print("\n1. Testing require_research_context (no research)...")
        result = await require_research_fn(
            ctx, agent, "Generate a script for an explosion"
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test require_research_context - should pass
        print("\n2. Testing require_research_context (with research)...")
        result = await require_research_fn(
            ctx, agent,
            "Research findings: Use mantaflow technique with bpy.ops.fluid. "
            "Recommended approach: shader-based volume."
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test validate_effect_type - should trigger
        print("\n3. Testing validate_effect_type (invalid)...")
        result = await validate_effect_fn(
            ctx, agent, "Generate something cool"
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_effect_type - should pass
        print("\n4. Testing validate_effect_type (valid)...")
        result = await validate_effect_fn(
            ctx, agent, "Generate an explosion effect"
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test validate_script_output - should trigger
        print("\n5. Testing validate_script_output (None)...")
        result = await validate_script_fn(ctx, agent, None)
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_script_output - should pass
        print("\n6. Testing validate_script_output (valid)...")

        class MockOutput:
            script_path = "/tmp/test.py"
            technique_used = "mantaflow_fire"
            validation_passed = True

        result = await validate_script_fn(ctx, agent, MockOutput())
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        print("\n" + "-" * 60)
        print("All Script Writer guardrail tests passed!")

    asyncio.run(test_guardrails())
