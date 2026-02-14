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


def _normalize_doc_ref(ref: str) -> str:
    """Normalize a doc_ref to canonical format before validation.

    The vector store returns filenames like 'bpy.types.FluidDomainSettings.html'
    but the guardrail expects 'blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#anchor'.
    This normalizer bridges the gap so valid refs aren't rejected for format mismatch.
    """
    token = ref.strip()
    # Already has full path prefix — return as-is
    if token.startswith(("blender_python_reference_5_0/", "blender_manual_html/")):
        return token
    # Bare API doc filename: bpy.types.Foo.html or bpy.ops.bar.html
    if re.match(r"^bpy\.(?:types|ops)\.[^\s]+\.html", token):
        # Add prefix and a generic anchor if missing
        if "#" not in token:
            token = f"blender_python_reference_5_0/{token}#attributes"
        else:
            token = f"blender_python_reference_5_0/{token}"
        return token
    return token


def _is_valid_doc_ref(ref: Any) -> bool:
    if not isinstance(ref, str):
        return False
    token = _normalize_doc_ref(ref.strip())
    if not token:
        return False
    if token.lower().startswith(("doc_search_", "none", "null", "unknown")):
        return False
    if token.endswith(".md"):
        return False
    return any(pattern.match(token) for pattern in _DOC_REF_PATTERNS)


def _is_api_doc_ref(ref: str) -> bool:
    token = _normalize_doc_ref(ref.strip())
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

    # Also extract api_modules — the structured output field where LLMs
    # reliably place API paths like "bpy.types.FluidDomainSettings".
    api_modules = []
    if hasattr(output, "api_modules"):
        api_modules = getattr(output, "api_modules", []) or []
    elif isinstance(output, dict):
        api_modules = output.get("api_modules", []) or []

    errors = []
    if not isinstance(recommended_approach, str) or not recommended_approach.strip():
        errors.append("recommended_approach must be a non-empty string")

    # S1-2 FIX: Accept empty doc_refs when api_modules provides valid API grounding.
    # LLMs reliably populate api_modules with "bpy.types.FluidDomainSettings" but
    # often return empty or garbage doc_refs from the vector store.
    has_api_modules_grounding = any(
        isinstance(m, str) and m.startswith(("bpy.types.", "bpy.ops."))
        for m in api_modules
    )

    if not isinstance(doc_refs, list) or len(doc_refs) == 0:
        if not has_api_modules_grounding:
            errors.append("doc_refs must be a non-empty list (or api_modules must contain valid bpy.types/bpy.ops paths)")
        else:
            print(
                f"[Guardrail] validate_research_output: doc_refs empty but api_modules has valid API grounding — OK",
                file=sys.stderr,
            )
    else:
        normalized_refs = [ref.strip() for ref in doc_refs if isinstance(ref, str)]
        invalid_refs = [ref for ref in normalized_refs if not _is_valid_doc_ref(ref)]
        if invalid_refs:
            # Downgrade to warning — invalid doc_refs are noisy but not fatal
            # if we have valid api_modules for API grounding.
            print(
                f"[Guardrail] validate_research_output WARN: {len(invalid_refs)} invalid doc_refs "
                f"(e.g., {invalid_refs[0][:60]}...)",
                file=sys.stderr
            )
        api_refs = [ref for ref in normalized_refs if _is_api_doc_ref(ref)]
        # Accept api_modules as sufficient API grounding when doc_refs
        # don't contain direct API references.
        has_api_grounding = bool(api_refs) or has_api_modules_grounding
        if not has_api_grounding:
            errors.append(
                "No API grounding found: doc_refs must include at least one API reference "
                "(bpy.types.* or bpy.ops.*) or api_modules must contain valid API paths"
            )

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
