"""
Guardrails module for OpenAI Agents SDK.

Phase 3 of Architecture Optimization Plan - Input/Output validation guardrails.

Guardrails are decorators that validate agent inputs and outputs BEFORE/AFTER
the agent runs. Unlike RunHooks (which operate at tool-call level), guardrails
operate at the agent level.

SDK Pattern:
    from agents import Agent, input_guardrail, GuardrailFunctionOutput

    @input_guardrail
    async def validate_input(ctx, agent, input):
        if not valid:
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info={"reason": "why invalid"}
            )
        return GuardrailFunctionOutput(tripwire_triggered=False)

    agent = Agent(input_guardrails=[validate_input])

When tripwire is triggered:
- Input guardrails raise InputGuardrailTripwireTriggered
- Output guardrails raise OutputGuardrailTripwireTriggered

These exceptions should be caught and handled in the pipeline.

Usage:
    from guardrails import (
        require_research_context,
        validate_effect_type,
        check_budget_before_quality,
        validate_script_output,
        validate_quality_output,
    )

    script_writer = Agent(
        input_guardrails=[require_research_context, validate_effect_type],
        output_guardrails=[validate_script_output],
    )
"""

from guardrails.script_guardrails import (
    require_research_context,
    validate_effect_type,
    validate_script_output,
)

from guardrails.research_guardrails import (
    validate_research_output,
)

from guardrails.quality_guardrails import (
    check_budget_before_quality,
    validate_quality_output,
)

from guardrails.coordinator_guardrails import (
    validate_technique_decision,
    validate_modification_decision,
    validate_quality_decision,
)

# Phase 12: Spec-First Pipeline guardrails (API Hallucination Prevention)
from guardrails.api_spec_guardrails import (
    validate_api_spec,
    validate_code_against_spec,
)

__all__ = [
    # Research Agent guardrails
    "validate_research_output",
    # Script Writer guardrails
    "require_research_context",
    "validate_effect_type",
    "validate_script_output",
    # Quality Analyst guardrails
    "check_budget_before_quality",
    "validate_quality_output",
    # Coordinator guardrails
    "validate_technique_decision",
    "validate_modification_decision",
    "validate_quality_decision",
    # Spec-First Pipeline guardrails (Phase 12)
    "validate_api_spec",
    "validate_code_against_spec",
]
