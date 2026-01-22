"""
API Validator Agent using OpenAI Agents SDK.

Validates Blender 5.0 API calls in Python scripts against documentation.
This agent is called BEFORE code is written/executed to catch API errors
at the source rather than discovering them through runtime failures.

Key capabilities:
- Extract bpy.types.*, bpy.ops.*, bpy.context.* calls from code
- Validate each call against Blender 5.0 documentation via vector store
- Return structured validation results with corrections
- Callable via as_tool() for integration with Script Writer

SDK Pattern: This agent uses structured output (Pydantic) and is designed
to be called as a tool from the orchestrator via agent.as_tool().

PROBLEM ADDRESSED: Problem 1 from Architecture Optimization Plan
- LLM generates code from training data (Blender 2.8x-4.x) instead of 5.0
- This agent REQUIRES documentation lookup before approving any API usage
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from agents import Agent, ModelSettings, function_tool

if TYPE_CHECKING:
    from agents import RunContextWrapper

# Add parent directory to path for imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import semantic docs tools for API validation
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    search_blender_api_by_intent,
)


# =============================================================================
# STRUCTURED OUTPUT MODELS (SDK Pattern: Pydantic for type safety)
# =============================================================================

class APICallValidation(BaseModel):
    """Validation result for a single API call."""
    api_call: str = Field(description="The original API call (e.g., 'bpy.types.FluidDomainSettings.flame_smoke')")
    is_valid: bool = Field(description="Whether the API call exists in Blender 5.0")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in validation (0.0-1.0)")
    correction: Optional[str] = Field(default=None, description="Corrected API call if invalid")
    documentation_source: Optional[str] = Field(default=None, description="Source of validation info")
    notes: Optional[str] = Field(default=None, description="Additional validation notes")


class CodeValidationResult(BaseModel):
    """Complete validation result for a code snippet."""
    is_valid: bool = Field(description="True if ALL API calls are valid")
    total_calls_checked: int = Field(description="Number of API calls validated")
    valid_calls: int = Field(description="Number of valid API calls")
    invalid_calls: int = Field(description="Number of invalid API calls")
    validations: List[APICallValidation] = Field(description="Individual validation results")
    corrections_needed: List[str] = Field(description="List of corrections to apply")
    summary: str = Field(description="Human-readable summary of validation")


# =============================================================================
# KNOWN BLENDER 5.0 API CHANGES (High-confidence corrections)
# =============================================================================

# These are KNOWN breaking changes from Blender 4.x to 5.0
# Used as high-confidence corrections when documentation search is ambiguous
KNOWN_API_CHANGES: Dict[str, Dict[str, str]] = {
    # Principled Volume shader socket renames
    "inputs[\"Smoke\"]": {
        "correction": "inputs[\"Grid\"]",
        "reason": "Renamed in Blender 5.0 Principled Volume shader"
    },
    "inputs[\"Smoke Color\"]": {
        "correction": "inputs[\"Grid Color\"]",
        "reason": "Renamed in Blender 5.0 Principled Volume shader"
    },
    # FluidDomainSettings changes
    "modifier.effector_weights": {
        "correction": "effector_weights",
        "reason": "effector_weights moved from modifier to domain settings in 5.0"
    },
    # FluidFlowSettings changes
    "flow_type": {
        "correction": "flow_behavior",
        "reason": "Renamed in Blender 5.0 FluidFlowSettings"
    },
}

# Known valid Blender 5.0 API patterns (don't warn about these)
KNOWN_VALID_5_0_PATTERNS = [
    r"bpy\.types\.FluidDomainSettings\.\w+",
    r"bpy\.types\.FluidFlowSettings\.\w+",
    r"bpy\.types\.FluidEffectorSettings\.\w+",
    r"bpy\.ops\.fluid\.\w+",
    r"bpy\.ops\.object\.\w+",
    r"bpy\.context\.\w+",
    r"bpy\.data\.\w+",
]


# =============================================================================
# API EXTRACTION TOOLS
# =============================================================================

def extract_api_calls_from_code(code: str) -> List[str]:
    """
    Extract Blender API calls from Python code.

    Looks for patterns:
    - bpy.types.* (type references)
    - bpy.ops.* (operators)
    - bpy.context.* (context access)
    - bpy.data.* (data access)
    - .inputs["*"] (node socket access)
    - domain.* settings (FluidDomainSettings)

    Args:
        code: Python source code

    Returns:
        List of unique API calls found
    """
    api_calls = set()

    # Pattern 1: bpy.types.ClassName
    for match in re.finditer(r'bpy\.types\.(\w+)', code):
        api_calls.add(f"bpy.types.{match.group(1)}")

    # Pattern 2: bpy.types.ClassName.property
    for match in re.finditer(r'bpy\.types\.(\w+)\.(\w+)', code):
        api_calls.add(f"bpy.types.{match.group(1)}.{match.group(2)}")

    # Pattern 3: bpy.ops.category.operator
    for match in re.finditer(r'bpy\.ops\.(\w+)\.(\w+)', code):
        api_calls.add(f"bpy.ops.{match.group(1)}.{match.group(2)}")

    # Pattern 4: Node socket access .inputs["Name"]
    for match in re.finditer(r'\.inputs\["([^"]+)"\]', code):
        api_calls.add(f'inputs["{match.group(1)}"]')

    # Pattern 5: domain.property (FluidDomainSettings)
    for match in re.finditer(r'domain\.(\w+)', code):
        prop = match.group(1)
        # Skip common non-API attributes
        if prop not in ['name', 'data', 'location', 'scale', 'rotation']:
            api_calls.add(f"domain.{prop}")

    # Pattern 6: flow_settings.property (FluidFlowSettings)
    for match in re.finditer(r'flow_settings\.(\w+)', code):
        api_calls.add(f"flow_settings.{match.group(1)}")

    # Pattern 7: modifier.effector_weights (deprecated pattern)
    if 'modifier.effector_weights' in code:
        api_calls.add('modifier.effector_weights')

    return sorted(api_calls)


@function_tool
def extract_blender_api_calls(code: str) -> str:
    """
    Extract all Blender API calls from Python code for validation.

    Args:
        code: Python source code to analyze

    Returns:
        JSON with extracted API calls and their locations
    """
    api_calls = extract_api_calls_from_code(code)

    # Check for known problematic patterns
    known_issues = []
    for call in api_calls:
        for pattern, fix in KNOWN_API_CHANGES.items():
            if pattern in call:
                known_issues.append({
                    "api_call": call,
                    "known_issue": pattern,
                    "correction": fix["correction"],
                    "reason": fix["reason"]
                })

    return json.dumps({
        "total_api_calls": len(api_calls),
        "api_calls": api_calls,
        "known_issues_found": len(known_issues),
        "known_issues": known_issues
    }, indent=2)


@function_tool
def check_known_api_changes(api_call: str) -> str:
    """
    Check if an API call matches known Blender 5.0 breaking changes.

    Args:
        api_call: The API call to check (e.g., 'inputs["Smoke"]')

    Returns:
        JSON with known correction if exists, otherwise "no_known_issue"
    """
    for pattern, fix in KNOWN_API_CHANGES.items():
        if pattern in api_call:
            return json.dumps({
                "has_known_issue": True,
                "original": api_call,
                "pattern_matched": pattern,
                "correction": fix["correction"],
                "reason": fix["reason"],
                "confidence": 1.0  # Known issues are high confidence
            })

    return json.dumps({
        "has_known_issue": False,
        "api_call": api_call,
        "note": "No known breaking changes for this API call"
    })


@function_tool
def validate_api_call_against_docs(api_call: str) -> str:
    """
    Validate a single API call against Blender 5.0 documentation.

    This tool searches the vector store for documentation about the API call
    and determines if it exists in Blender 5.0.

    Args:
        api_call: The API call to validate (e.g., 'bpy.types.FluidDomainSettings.flame_smoke')

    Returns:
        JSON with validation result, confidence, and any corrections
    """
    # First check known issues (high confidence)
    for pattern, fix in KNOWN_API_CHANGES.items():
        if pattern in api_call:
            return json.dumps({
                "api_call": api_call,
                "is_valid": False,
                "confidence": 1.0,
                "correction": api_call.replace(pattern, fix["correction"]),
                "source": "known_breaking_changes",
                "reason": fix["reason"]
            })

    # For other calls, we need to rely on the agent to use semantic search
    # This tool just marks the call as "needs_verification"
    return json.dumps({
        "api_call": api_call,
        "status": "needs_verification",
        "instruction": "Use semantic_search_blender_docs to verify this API exists in Blender 5.0",
        "search_query": f"Blender 5.0 API {api_call}"
    })


@function_tool
def format_validation_report(
    code: str,
    validations: str  # JSON string of validation results
) -> str:
    """
    Format a complete validation report for code review.

    Args:
        code: The original code being validated
        validations: JSON string containing validation results

    Returns:
        Formatted report with summary and corrections
    """
    try:
        results = json.loads(validations)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid validations JSON"})

    # Count results
    total = len(results) if isinstance(results, list) else 0
    valid = sum(1 for r in results if r.get("is_valid", False)) if isinstance(results, list) else 0
    invalid = total - valid

    corrections = []
    if isinstance(results, list):
        for r in results:
            if not r.get("is_valid", True) and r.get("correction"):
                corrections.append({
                    "original": r.get("api_call"),
                    "corrected": r.get("correction"),
                    "reason": r.get("reason", "See Blender 5.0 docs")
                })

    report = {
        "summary": {
            "total_api_calls": total,
            "valid": valid,
            "invalid": invalid,
            "pass": invalid == 0
        },
        "corrections_needed": corrections,
        "recommendation": (
            "Code is valid for Blender 5.0" if invalid == 0
            else f"Apply {len(corrections)} corrections before execution"
        )
    }

    return json.dumps(report, indent=2)


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================

API_VALIDATOR_INSTRUCTIONS = """## ROLE
You are the Blender 5.0 API Validator. Your job is to validate Python code
against Blender 5.0 documentation BEFORE it is executed.

## CRITICAL: Blender 5.0 API Changes
Blender 5.0 has BREAKING CHANGES from 4.x. Common issues:
- inputs["Smoke"] → inputs["Grid"] (Principled Volume shader)
- modifier.effector_weights → effector_weights (direct on domain)
- flow_type → flow_behavior (FluidFlowSettings)

## TURN BUDGET: MAX 4 TURNS
T1: extract_blender_api_calls(code) - get all API calls
T2: check_known_api_changes for each suspicious call
T3: semantic_search_blender_docs for unknown APIs (ONE search max)
T4: Return CodeValidationResult

## WORKFLOW
1. Extract API calls from code
2. Check each against known breaking changes (FAST, no LLM)
3. For unknown APIs, search docs ONCE (avoid research loops)
4. Return structured validation result

## OUTPUT: CodeValidationResult
- is_valid: bool (True only if ALL calls valid)
- total_calls_checked: int
- valid_calls: int
- invalid_calls: int
- validations: List[APICallValidation]
- corrections_needed: List[str] (specific corrections to apply)
- summary: str

## RULES
- NEVER approve code without checking known_api_changes
- ONE semantic search max (avoid loops)
- If uncertain about an API, mark it as needs_manual_review
- Confidence > 0.8 for known issues, < 0.5 for uncertain
- Always provide specific corrections, not vague suggestions
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_api_validator(custom_instructions: str = "") -> Agent:
    """
    Create an API Validator agent for Blender 5.0 code validation.

    This agent is designed to be called via as_tool() from the orchestrator
    or Script Writer to validate code BEFORE execution.

    Args:
        custom_instructions: Additional context (e.g., current effect type)

    Returns:
        Agent instance ready for use
    """
    instructions = API_VALIDATOR_INSTRUCTIONS
    if custom_instructions:
        instructions = instructions + "\n\n" + custom_instructions

    return Agent(
        name="API Validator",
        instructions=instructions,
        model=os.getenv("API_VALIDATOR_MODEL", "gpt-4.1"),  # Fast model for validation
        model_settings=ModelSettings(
            temperature=0.0,  # Deterministic for validation
        ),
        tools=[
            # API extraction and validation
            extract_blender_api_calls,
            check_known_api_changes,
            validate_api_call_against_docs,
            format_validation_report,
            # Documentation search (use sparingly - one search max)
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ],
    )


# =============================================================================
# AS_TOOL WRAPPER FOR ORCHESTRATOR INTEGRATION
# =============================================================================

def get_api_validator_as_tool() -> Any:
    """
    Get the API Validator agent wrapped as a tool for orchestrator use.

    SDK Pattern: agent.as_tool() allows calling this agent as a function
    tool from another agent, with control returning to the caller.

    Returns:
        Tool instance that can be added to another agent's tools list

    Usage:
        orchestrator = Agent(
            tools=[
                get_api_validator_as_tool(),
                # ... other tools
            ]
        )
    """
    validator = create_api_validator()
    return validator.as_tool(
        tool_name="validate_blender_api",
        tool_description=(
            "Validate Blender Python code against Blender 5.0 API documentation. "
            "Call BEFORE writing or executing scripts to catch API errors. "
            "Returns validation result with specific corrections for any invalid APIs."
        ),
    )


# =============================================================================
# STANDALONE VALIDATION FUNCTION (for direct Python calls)
# =============================================================================

async def validate_code_api(code: str) -> CodeValidationResult:
    """
    Validate code API calls without running the full agent.

    This is a lightweight validation that only checks known issues.
    For full validation with documentation search, use the agent.

    Args:
        code: Python source code to validate

    Returns:
        CodeValidationResult with validation details
    """
    api_calls = extract_api_calls_from_code(code)
    validations = []
    corrections_needed = []

    for call in api_calls:
        validation = APICallValidation(
            api_call=call,
            is_valid=True,
            confidence=0.5,  # Default confidence for unknown
        )

        # Check known issues
        for pattern, fix in KNOWN_API_CHANGES.items():
            if pattern in call:
                corrected = call.replace(pattern, fix["correction"])
                validation = APICallValidation(
                    api_call=call,
                    is_valid=False,
                    confidence=1.0,
                    correction=corrected,
                    documentation_source="known_breaking_changes",
                    notes=fix["reason"]
                )
                corrections_needed.append(f"{call} → {corrected}")
                break

        validations.append(validation)

    valid_count = sum(1 for v in validations if v.is_valid)
    invalid_count = len(validations) - valid_count

    return CodeValidationResult(
        is_valid=(invalid_count == 0),
        total_calls_checked=len(validations),
        valid_calls=valid_count,
        invalid_calls=invalid_count,
        validations=validations,
        corrections_needed=corrections_needed,
        summary=(
            f"Validated {len(validations)} API calls: {valid_count} valid, {invalid_count} invalid"
            if validations else "No API calls found to validate"
        )
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing API Validator Agent...")
        print("-" * 60)

        # Test code with known issues
        test_code = '''
import bpy

# Create domain
bpy.ops.mesh.primitive_cube_add()
domain = bpy.context.active_object
bpy.ops.object.modifier_add(type='FLUID')
domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

# BAD: Old API pattern (should be caught)
domain.modifier.effector_weights.wind = 1.0

# Set up shader
mat = bpy.data.materials.new("VolumeShader")
mat.use_nodes = True
principled_volume = mat.node_tree.nodes.new('ShaderNodeVolumePrincipled')

# BAD: Old socket name (should be caught)
principled_volume.inputs["Smoke"].default_value = 0.5
principled_volume.inputs["Smoke Color"].default_value = (1, 0.5, 0.2, 1)

# GOOD: New socket name
principled_volume.inputs["Grid"].default_value = 0.8
'''

        print("Test code:")
        print(test_code[:200] + "...")
        print()

        # Test 1: Lightweight validation (no agent)
        print("Test 1: Lightweight validation (known issues only)")
        result = await validate_code_api(test_code)
        print(f"  Valid: {result.is_valid}")
        print(f"  Checked: {result.total_calls_checked} calls")
        print(f"  Invalid: {result.invalid_calls}")
        print(f"  Corrections: {result.corrections_needed}")
        print()

        # Test 2: Extract API calls
        print("Test 2: API call extraction")
        api_calls = extract_api_calls_from_code(test_code)
        print(f"  Found {len(api_calls)} API calls:")
        for call in api_calls[:10]:
            print(f"    - {call}")
        print()

        # Test 3: Full agent test (requires API key)
        if os.getenv("OPENAI_API_KEY"):
            print("Test 3: Full agent validation")
            agent = create_api_validator()
            result = await Runner.run(
                agent,
                f"Validate this Blender Python code:\n\n```python\n{test_code}\n```",
                max_turns=5
            )
            print(f"  Response: {str(result.final_output)[:500]}...")
        else:
            print("Test 3: Skipped (OPENAI_API_KEY not set)")

        print("-" * 60)
        print("Tests complete!")

    asyncio.run(test())
