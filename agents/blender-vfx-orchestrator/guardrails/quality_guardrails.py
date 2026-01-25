"""
Quality Analyst Guardrails for OpenAI Agents SDK.

These guardrails enforce critical requirements for the Quality Analyst agent:

1. check_budget_before_quality (INPUT) - Blocks evaluation if budget exhausted
2. validate_quality_output (OUTPUT) - Ensures quality output has required fields

SDK Pattern Reference:
    https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md
"""

from __future__ import annotations

import sys
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
    from models.shared_context import SharedContext

# Critical issues that should auto-fail quality gate
CRITICAL_ISSUES = {
    "ZERO_LIGHTS_ACTIVE",
    "BLACK_SCREEN",
    "WHITE_SCREEN",
    "CLIPPING_ARTIFACTS",
    "NO_RENDER_OUTPUT",
}


@input_guardrail
async def check_budget_before_quality(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    input: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that checks budget before quality evaluation.

    Quality evaluation uses vision APIs which are expensive. This guardrail
    blocks evaluation if the budget is exhausted.

    Note: This requires access to the shared context's budget_tracker.
    If context is not available, the guardrail passes (fails open).

    Args:
        ctx: Run context wrapper (should contain SharedContext)
        agent: The Quality Analyst agent
        input: Input prompt string or message list

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if budget exhausted
    """
    # Try to get budget tracker from context
    try:
        # ctx.context should be SharedContext
        shared_context = ctx.context
        if shared_context is None:
            print(
                f"[Guardrail] check_budget_before_quality: No context, passing",
                file=sys.stderr
            )
            return GuardrailFunctionOutput(
                tripwire_triggered=False,
                output_info={"status": "passed"}
            )

        # Get budget tracker from context or utils
        if hasattr(shared_context, "budget_tracker"):
            budget_tracker = shared_context.budget_tracker
        else:
            # Fall back to global budget tracker
            from utils import get_budget_tracker
            budget_tracker = get_budget_tracker()

        if hasattr(budget_tracker, "get_spent"):
            spent = budget_tracker.get_spent()
        elif hasattr(budget_tracker, "state") and hasattr(budget_tracker.state, "total_spent"):
            spent = budget_tracker.state.total_spent()
        else:
            spent = 0.0
        remaining = max(0.0, budget_tracker.monthly_limit - spent)

        if not budget_tracker.can_afford_evaluation():
            print(
                f"[Guardrail] check_budget_before_quality TRIGGERED: "
                f"Budget exhausted (spent: ${spent:.2f})",
                file=sys.stderr
            )
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info={
                    "reason": "Budget exhausted - cannot perform quality evaluation",
                    "total_spent": spent,
                    "monthly_limit": budget_tracker.monthly_limit,
                    "remaining": remaining,
                    "hint": "Return best result so far without further evaluation",
                }
            )

        # Check if at 80% budget - warn but don't block
        remaining_pct = remaining / budget_tracker.monthly_limit if budget_tracker.monthly_limit else 0.0
        if remaining_pct < 0.2:
            print(
                f"[Guardrail] check_budget_before_quality WARNING: "
                f"Budget at {(1-remaining_pct)*100:.0f}% "
                f"(remaining: ${remaining:.2f})",
                file=sys.stderr
            )

        print(
            f"[Guardrail] check_budget_before_quality PASSED: "
            f"Budget OK (remaining: ${remaining:.2f})",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=False,
            output_info={"status": "passed"}
        )

    except Exception as e:
        # Fail open on errors - let the agent run
        print(
            f"[Guardrail] check_budget_before_quality: Error checking budget: {e}, passing",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )


@output_guardrail
async def validate_quality_output(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that validates QualityOutput has required fields.

    Ensures the Quality Analyst produces valid output with:
    - overall_score in range [0, 100]
    - passed boolean is set
    - primary_issue is specified when not passed
    - Critical issues are properly flagged

    Args:
        ctx: Run context wrapper
        agent: The Quality Analyst agent
        output: The agent's output (should be QualityOutput)

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True if output invalid
    """
    # Handle None output
    if output is None:
        print(
            f"[Guardrail] validate_quality_output TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Quality Analyst produced no output",
                "hint": "Agent must return QualityOutput with overall_score and passed",
            }
        )

    # Extract fields from output
    if hasattr(output, "overall_score"):
        overall_score = output.overall_score
        passed = getattr(output, "passed", None)
        primary_issue = getattr(output, "primary_issue", None)
        issues = getattr(output, "issues", [])
        vision_assessment = getattr(output, "vision_assessment", "")
    elif isinstance(output, dict):
        overall_score = output.get("overall_score")
        passed = output.get("passed")
        primary_issue = output.get("primary_issue")
        issues = output.get("issues", [])
        vision_assessment = output.get("vision_assessment", "")
    else:
        print(
            f"[Guardrail] validate_quality_output TRIGGERED: "
            f"Unrecognized output type: {type(output)}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Output is not a valid QualityOutput structure",
                "received_type": str(type(output)),
            }
        )

    errors = []

    # Check overall_score
    if overall_score is None:
        errors.append("overall_score is missing")
    elif not isinstance(overall_score, (int, float)):
        errors.append(f"overall_score must be a number, got {type(overall_score)}")
    elif not 0 <= overall_score <= 100:
        errors.append(f"overall_score must be 0-100, got {overall_score}")

    # Check passed boolean
    if passed is None:
        errors.append("passed is missing")
    elif not isinstance(passed, bool):
        errors.append(f"passed must be boolean, got {type(passed)}")

    # Check consistency: if passed=True but critical issues exist, that's wrong
    if passed is True and issues:
        critical_found = [
            issue for issue in issues
            if any(critical in str(issue).upper() for critical in CRITICAL_ISSUES)
        ]
        if critical_found:
            errors.append(
                f"passed=True but critical issues found: {critical_found}. "
                f"Critical issues should auto-fail."
            )

    # Check consistency: if passed=False, should have primary_issue
    if passed is False and not primary_issue:
        print(
            f"[Guardrail] validate_quality_output WARNING: "
            f"passed=False but no primary_issue specified",
            file=sys.stderr
        )
        # Don't fail, just warn

    # Check vision_assessment is non-empty for thorough evaluation
    if not vision_assessment:
        print(
            f"[Guardrail] validate_quality_output WARNING: "
            f"vision_assessment is empty - evaluation may be incomplete",
            file=sys.stderr
        )

    if errors:
        print(
            f"[Guardrail] validate_quality_output TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={
                "reason": "Invalid QualityOutput structure",
                "errors": errors,
                "received": str(output)[:300],
            }
        )

    print(
        f"[Guardrail] validate_quality_output PASSED: "
        f"score={overall_score}, passed={passed}, "
        f"issues_count={len(issues) if issues else 0}",
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
        print("Testing Quality Analyst Guardrails...")
        print("-" * 60)

        # Mock context
        class MockContext:
            context = None

        ctx = MockContext()

        # Mock agent
        class MockAgent:
            name = "Quality Analyst"

        agent = MockAgent()

        # Access the guardrail function from the InputGuardrail/OutputGuardrail objects
        check_budget_fn = check_budget_before_quality.guardrail_function
        validate_quality_fn = validate_quality_output.guardrail_function

        # Test check_budget_before_quality - should pass (no context)
        print("\n1. Testing check_budget_before_quality (no context)...")
        result = await check_budget_fn(
            ctx, agent, "Evaluate this render"
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test validate_quality_output - should trigger (None)
        print("\n2. Testing validate_quality_output (None)...")
        result = await validate_quality_fn(ctx, agent, None)
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_quality_output - should trigger (missing fields)
        print("\n3. Testing validate_quality_output (missing fields)...")
        result = await validate_quality_fn(
            ctx, agent, {"overall_score": 75}  # missing 'passed'
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_quality_output - should trigger (score out of range)
        print("\n4. Testing validate_quality_output (score out of range)...")
        result = await validate_quality_fn(
            ctx, agent, {"overall_score": 150, "passed": True}
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_quality_output - should trigger (critical issue but passed)
        print("\n5. Testing validate_quality_output (critical issue but passed=True)...")
        result = await validate_quality_fn(
            ctx, agent,
            {
                "overall_score": 70,
                "passed": True,
                "issues": ["ZERO_LIGHTS_ACTIVE - no lighting"]
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        # Test validate_quality_output - should pass
        print("\n6. Testing validate_quality_output (valid)...")

        class MockOutput:
            overall_score = 75.5
            passed = True
            primary_issue = None
            issues = []
            vision_assessment = "Good explosion with visible flames"
            suggestions = []
            reference_similarity = None

        result = await validate_quality_fn(ctx, agent, MockOutput())
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        # Test valid failed output
        print("\n7. Testing validate_quality_output (valid fail)...")
        result = await validate_quality_fn(
            ctx, agent,
            {
                "overall_score": 45,
                "passed": False,
                "primary_issue": "smoke too thin",
                "issues": ["smoke density low", "lacks brightness"],
                "vision_assessment": "Smoke is barely visible"
            }
        )
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        assert result.tripwire_triggered is False

        print("\n" + "-" * 60)
        print("All Quality Analyst guardrail tests passed!")

    asyncio.run(test_guardrails())
