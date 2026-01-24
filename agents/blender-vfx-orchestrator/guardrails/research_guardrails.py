"""
Research Agent Guardrails for OpenAI Agents SDK.

Phase 3 enforcement: ensure structured ResearchOutput is complete and doc-grounded.
"""

from __future__ import annotations

import sys
from typing import Any, TYPE_CHECKING

from agents import Agent, GuardrailFunctionOutput, output_guardrail

if TYPE_CHECKING:
    from agents import RunContextWrapper


@output_guardrail
async def validate_research_output(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that validates ResearchOutput has required fields.

    Requirements:
    - recommended_approach must be non-empty
    - doc_refs must be a non-empty list
    """
    if output is None:
        print(
            "[Guardrail] validate_research_output TRIGGERED: Output is None",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "Research Agent produced no output"}
        )

    if hasattr(output, "recommended_approach"):
        recommended_approach = output.recommended_approach
        doc_refs = getattr(output, "doc_refs", [])
    elif isinstance(output, dict):
        recommended_approach = output.get("recommended_approach")
        doc_refs = output.get("doc_refs", [])
    else:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": f"Invalid ResearchOutput type: {type(output)}"}
        )

    errors = []
    if not isinstance(recommended_approach, str) or not recommended_approach.strip():
        errors.append("recommended_approach must be a non-empty string")
    if not isinstance(doc_refs, list) or len(doc_refs) == 0:
        errors.append("doc_refs must be a non-empty list")

    if errors:
        print(
            f"[Guardrail] validate_research_output TRIGGERED: {errors}",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"errors": errors}
        )

    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )
