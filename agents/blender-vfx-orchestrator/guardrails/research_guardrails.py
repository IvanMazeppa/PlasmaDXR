"""
Research Agent Guardrails for OpenAI Agents SDK.

Phase 3 enforcement: ensure structured ResearchOutput is complete and doc-grounded.
"""

from __future__ import annotations

import re
import sys
from typing import Any, TYPE_CHECKING

from agents import Agent, GuardrailFunctionOutput, output_guardrail

if TYPE_CHECKING:
    from agents import RunContextWrapper


# Strict doc grounding patterns for Blender 5 manual/API references.
# Rejects sentinels, filenames, and null-ish placeholders.
_DOC_REF_PATTERNS = (
    re.compile(r"^blender_manual_html/[^\s#]+#[^\s]+$"),
    re.compile(r"^blender_python_reference_5_0/[^\s#]+#[^\s]+$"),
    re.compile(r"^bpy\.(?:types|ops)\.[^\s#]+\.html(?:#[^\s]+)?$"),
)
_API_DOC_REF_PATTERNS = (
    re.compile(r"^blender_python_reference_5_0/bpy\.(?:types|ops)\.[^\s#]+#[^\s]+$"),
    re.compile(r"^bpy\.(?:types|ops)\.[^\s#]+\.html(?:#[^\s]+)?$"),
)


def _is_valid_doc_ref(ref: Any) -> bool:
    if not isinstance(ref, str):
        return False
    token = ref.strip()
    if not token:
        return False
    if token.lower().startswith(("doc_search_", "none", "null", "unknown")):
        return False
    if token.endswith(".md"):
        return False
    return any(pattern.match(token) for pattern in _DOC_REF_PATTERNS)


def _is_api_doc_ref(ref: str) -> bool:
    token = ref.strip()
    return any(pattern.match(token) for pattern in _API_DOC_REF_PATTERNS)


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
    - doc_refs must be a non-empty list of strict Blender 5 doc paths
    - must include at least one API doc reference (`bpy.types.*` or `bpy.ops.*`)
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
    else:
        normalized_refs = [ref.strip() for ref in doc_refs if isinstance(ref, str)]
        invalid_refs = [ref for ref in normalized_refs if not _is_valid_doc_ref(ref)]
        if invalid_refs:
            errors.append(
                "doc_refs contain invalid/non-grounded entries: "
                + ", ".join(invalid_refs[:5])
            )
        api_refs = [ref for ref in normalized_refs if _is_api_doc_ref(ref)]
        if not api_refs:
            errors.append("doc_refs must include at least one API reference (bpy.types.* or bpy.ops.*)")

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
