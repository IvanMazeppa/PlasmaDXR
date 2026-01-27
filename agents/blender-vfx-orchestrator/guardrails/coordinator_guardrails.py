"""
Coordinator Agent Guardrails for OpenAI Agents SDK.

These guardrails validate the structured outputs from Coordinator agents:

1. validate_technique_decision (OUTPUT) - TechniqueDecision has selected_technique
2. validate_modification_decision (OUTPUT) - ModificationDecision has action
3. validate_quality_decision (OUTPUT) - QualityDecision has passed and next_action

SDK Pattern Reference:
    https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md
"""

from __future__ import annotations

import re
import sys
from typing import Any, TYPE_CHECKING

from agents import (
    Agent,
    GuardrailFunctionOutput,
    output_guardrail,
)

if TYPE_CHECKING:
    from agents import RunContextWrapper

# Valid actions for modification decisions
VALID_MODIFICATION_ACTIONS = {
    "modify_params",
    "switch_technique",
    "continue",
}

DELETE_SENTINEL = "__DELETE__"

# Valid next actions for quality gate decisions
VALID_QUALITY_NEXT_ACTIONS = {
    "complete",
    "iterate",
    "switch_technique",
    "request_guidance",
}


@output_guardrail
async def validate_technique_decision(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail for TechniqueDecision structured output.

    Ensures the Technique Selector Coordinator produces valid output with:
    - selected_technique is non-empty
    - reasoning is provided
    - key_parameters is a dict (can be empty)
    - alternative_techniques is a list (can be empty)

    Args:
        ctx: Run context wrapper
        agent: The Technique Selector agent
        output: The agent's output (should be TechniqueDecision)

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if output invalid
    """
    if output is None:
        print(
            f"[Guardrail] validate_technique_decision TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Technique Selector produced no output",
                "hint": "Agent must return TechniqueDecision with selected_technique",
            }
        )

    # Extract fields
    if hasattr(output, "selected_technique"):
        selected_technique = output.selected_technique
        reasoning = getattr(output, "reasoning", "")
        key_parameters = getattr(output, "key_parameters", {})
        alternative_techniques = getattr(output, "alternative_techniques", [])
    elif isinstance(output, dict):
        selected_technique = output.get("selected_technique")
        reasoning = output.get("reasoning", "")
        key_parameters = output.get("key_parameters", {})
        alternative_techniques = output.get("alternative_techniques", [])
    else:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": f"Unrecognized output type: {type(output)}",
            }
        )

    errors = []

    # selected_technique is required and must be non-empty
    if not selected_technique:
        errors.append("selected_technique is missing or empty")
    elif not isinstance(selected_technique, str):
        errors.append(f"selected_technique must be string, got {type(selected_technique)}")

    # reasoning should be non-empty
    if not reasoning:
        errors.append("reasoning is missing - explain why this technique was selected")

    # key_parameters should be dict
    if not isinstance(key_parameters, dict):
        errors.append(f"key_parameters must be dict, got {type(key_parameters)}")

    # alternative_techniques should be list
    if not isinstance(alternative_techniques, list):
        errors.append(f"alternative_techniques must be list, got {type(alternative_techniques)}")

    if errors:
        print(
            f"[Guardrail] validate_technique_decision TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Invalid TechniqueDecision structure",
                "errors": errors,
            }
        )

    print(
        f"[Guardrail] validate_technique_decision PASSED: "
        f"technique={selected_technique}, "
        f"params_count={len(key_parameters)}, "
        f"alternatives={len(alternative_techniques)}",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


@output_guardrail
async def validate_modification_decision(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail for ModificationDecision structured output.

    Ensures the Modification Strategist Coordinator produces valid output with:
    - action is one of: modify_params, switch_technique, continue
    - parameter_changes is a dict when action is modify_params
    - new_technique is set when action is switch_technique
    - confidence is between 0.0 and 1.0

    Args:
        ctx: Run context wrapper
        agent: The Modification Strategist agent
        output: The agent's output (should be ModificationDecision)

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if output invalid
    """
    if output is None:
        print(
            f"[Guardrail] validate_modification_decision TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Modification Strategist produced no output",
                "hint": "Agent must return ModificationDecision with action",
            }
        )

    # Extract fields
    if hasattr(output, "action"):
        action = output.action
        parameter_changes = getattr(output, "parameter_changes", {})
        new_technique = getattr(output, "new_technique", None)
        reasoning = getattr(output, "reasoning", "")
        confidence = getattr(output, "confidence", 0.5)
    elif isinstance(output, dict):
        action = output.get("action")
        parameter_changes = output.get("parameter_changes", {})
        new_technique = output.get("new_technique")
        reasoning = output.get("reasoning", "")
        confidence = output.get("confidence", 0.5)
    else:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": f"Unrecognized output type: {type(output)}",
            }
        )

    errors = []

    # action is required and must be valid
    if not action:
        errors.append("action is missing")
    elif action not in VALID_MODIFICATION_ACTIONS:
        errors.append(
            f"action must be one of {VALID_MODIFICATION_ACTIONS}, got '{action}'"
        )

    # Conditional validation based on action
    if action == "modify_params":
        if not parameter_changes:
            errors.append(
                "action='modify_params' but parameter_changes is empty. "
                "Provide concrete parameter changes like {\"temperature\": 3.0}"
            )
        elif not isinstance(parameter_changes, dict):
            errors.append(f"parameter_changes must be dict, got {type(parameter_changes)}")
        else:
            nested_keys = [
                key for key, value in parameter_changes.items()
                if isinstance(value, (dict, list))
            ]
            if nested_keys:
                errors.append(
                    "parameter_changes must be flat (no nested dict/list). "
                    f"Nested keys: {nested_keys}"
                )
            else:
                # Enforce structured values: numbers/bools/short enums/DELETE sentinel
                invalid_values = []
                for key, value in parameter_changes.items():
                    if isinstance(value, (int, float, bool)) or value is None:
                        continue
                    if isinstance(value, str):
                        if value == DELETE_SENTINEL:
                            continue
                        if re.search(r"\s", value) or len(value) > 32:
                            invalid_values.append(key)
                            continue
                        # Short enum tokens like GAS, LIQUID, REPLAY, MODULAR, GEOMETRY
                        continue
                    invalid_values.append(key)

                if invalid_values:
                    errors.append(
                        "parameter_changes values must be numeric/bool, short enum tokens, "
                        f"or '{DELETE_SENTINEL}' for removals. "
                        f"Invalid keys: {invalid_values}"
                    )

    if action == "switch_technique":
        if not new_technique:
            errors.append(
                "action='switch_technique' but new_technique is not specified"
            )

    # confidence validation
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            errors.append(f"confidence must be number, got {type(confidence)}")
        elif not 0.0 <= confidence <= 1.0:
            errors.append(f"confidence must be 0.0-1.0, got {confidence}")

    # reasoning should be present
    if not reasoning:
        print(
            f"[Guardrail] validate_modification_decision WARNING: "
            f"reasoning is empty",
            file=sys.stderr
        )

    if errors:
        print(
            f"[Guardrail] validate_modification_decision TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Invalid ModificationDecision structure",
                "errors": errors,
            }
        )

    print(
        f"[Guardrail] validate_modification_decision PASSED: "
        f"action={action}, confidence={confidence}",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


@output_guardrail
async def validate_quality_decision(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail for QualityDecision structured output.

    Ensures the Quality Gate Judge Coordinator produces valid output with:
    - passed is a boolean
    - should_continue is a boolean
    - next_action is one of: complete, iterate, switch_technique, request_guidance
    - escape_level is an integer 0-4

    Args:
        ctx: Run context wrapper
        agent: The Quality Gate Judge agent
        output: The agent's output (should be QualityDecision)

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if output invalid
    """
    if output is None:
        print(
            f"[Guardrail] validate_quality_decision TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Quality Gate Judge produced no output",
                "hint": "Agent must return QualityDecision with passed and next_action",
            }
        )

    # Extract fields
    if hasattr(output, "passed"):
        passed = output.passed
        should_continue = getattr(output, "should_continue", None)
        next_action = getattr(output, "next_action", None)
        escape_level = getattr(output, "escape_level", 0)
        reasoning = getattr(output, "reasoning", "")
    elif isinstance(output, dict):
        passed = output.get("passed")
        should_continue = output.get("should_continue")
        next_action = output.get("next_action")
        escape_level = output.get("escape_level", 0)
        reasoning = output.get("reasoning", "")
    else:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": f"Unrecognized output type: {type(output)}",
            }
        )

    errors = []

    # passed is required boolean
    if passed is None:
        errors.append("passed is missing")
    elif not isinstance(passed, bool):
        errors.append(f"passed must be boolean, got {type(passed)}")

    # should_continue is required boolean
    if should_continue is None:
        errors.append("should_continue is missing")
    elif not isinstance(should_continue, bool):
        errors.append(f"should_continue must be boolean, got {type(should_continue)}")

    # next_action is required and must be valid
    if not next_action:
        errors.append("next_action is missing")
    elif next_action not in VALID_QUALITY_NEXT_ACTIONS:
        errors.append(
            f"next_action must be one of {VALID_QUALITY_NEXT_ACTIONS}, got '{next_action}'"
        )

    # escape_level validation
    if not isinstance(escape_level, int):
        errors.append(f"escape_level must be int, got {type(escape_level)}")
    elif not 0 <= escape_level <= 4:
        errors.append(f"escape_level must be 0-4, got {escape_level}")

    # Consistency checks
    if passed is True and next_action != "complete":
        print(
            f"[Guardrail] validate_quality_decision WARNING: "
            f"passed=True but next_action='{next_action}' (expected 'complete')",
            file=sys.stderr
        )

    if passed is False and next_action == "complete":
        errors.append(
            "Inconsistent: passed=False but next_action='complete'. "
            "Cannot complete if quality gate not passed."
        )

    if errors:
        print(
            f"[Guardrail] validate_quality_decision TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Invalid QualityDecision structure",
                "errors": errors,
            }
        )

    print(
        f"[Guardrail] validate_quality_decision PASSED: "
        f"passed={passed}, next_action={next_action}, escape_level={escape_level}",
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
        print("Testing Coordinator Guardrails...")
        print("-" * 60)

        # Mock context
        class MockContext:
            context = None

        ctx = MockContext()

        # Mock agent
        class MockAgent:
            name = "Coordinator"

        agent = MockAgent()

        # Access the guardrail function from the OutputGuardrail objects
        validate_technique_fn = validate_technique_decision.guardrail_function
        validate_modification_fn = validate_modification_decision.guardrail_function
        validate_quality_fn = validate_quality_decision.guardrail_function

        # Test validate_technique_decision - should trigger (None)
        print("\n1. Testing validate_technique_decision (None)...")
        result = await validate_technique_fn(ctx, agent, None)
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_technique_decision - should trigger (missing technique)
        print("\n2. Testing validate_technique_decision (missing technique)...")
        result = await validate_technique_fn(
            ctx, agent, {"reasoning": "test"}
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_technique_decision - should pass
        print("\n3. Testing validate_technique_decision (valid)...")
        result = await validate_technique_fn(
            ctx, agent,
            {
                "selected_technique": "mantaflow_fire",
                "reasoning": "Best for fire effects",
                "key_parameters": {"temperature": 3.0},
                "alternative_techniques": ["shader_volume"],
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test validate_modification_decision - should trigger (invalid action)
        print("\n4. Testing validate_modification_decision (invalid action)...")
        result = await validate_modification_fn(
            ctx, agent, {"action": "invalid_action"}
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_modification_decision - should trigger (modify without params)
        print("\n5. Testing validate_modification_decision (modify without params)...")
        result = await validate_modification_fn(
            ctx, agent,
            {"action": "modify_params", "parameter_changes": {}, "reasoning": "test"}
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_modification_decision - should pass
        print("\n6. Testing validate_modification_decision (valid modify)...")
        result = await validate_modification_fn(
            ctx, agent,
            {
                "action": "modify_params",
                "parameter_changes": {"temperature": 3.5, "density": 5.0},
                "reasoning": "Increase heat for brighter flames",
                "confidence": 0.8
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test validate_quality_decision - should trigger (passed but not complete)
        print("\n7. Testing validate_quality_decision (passed=False, complete)...")
        result = await validate_quality_fn(
            ctx, agent,
            {
                "passed": False,
                "should_continue": False,
                "next_action": "complete",  # Inconsistent!
                "escape_level": 0,
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_quality_decision - should pass
        print("\n8. Testing validate_quality_decision (valid)...")
        result = await validate_quality_fn(
            ctx, agent,
            {
                "passed": True,
                "should_continue": False,
                "next_action": "complete",
                "escape_level": 0,
                "reasoning": "Quality gate passed with score 75"
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        print("\n" + "-" * 60)
        print("All Coordinator guardrail tests passed!")

    asyncio.run(test_guardrails())
