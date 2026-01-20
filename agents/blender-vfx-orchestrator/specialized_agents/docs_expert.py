"""
Documentation Expert Agent using OpenAI Agents SDK.

UPDATED: Now uses vector store-based semantic search tools instead of
deprecated local file-based tools. This provides much better search
results through AI embeddings.

Key changes:
- Replaced 12 local blender_docs_tools with 3 vector store tools
- Uses semantic_search_blender_docs for natural language queries
- Uses search_blender_api_by_intent for API discovery
- Uses find_alternative_approaches when stuck
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agents import Agent, ModelSettings, function_tool
from openai.types.shared import Reasoning

# Add parent directory to path for imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# NEW: Import vector store-based semantic search tools (Strategy 1)
# These replace the deprecated local blender_docs_tools
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    find_alternative_approaches,
    search_blender_api_by_intent,
)


# =============================================================================
# PARAMETER VALIDATION TOOLS (kept - still useful)
# =============================================================================

# Blender 5.0 API parameter ranges (from script-generator)
BLENDER_PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "turbulence": {"min": 0.0, "max": 1.0, "default": 0.3},
    "vorticity": {"min": 0.0, "max": 1.0, "default": 0.3},
    "temperature": {"min": 0.0, "max": 100000.0, "default": 5778.0},
    "flame_max_temp": {"min": 0.0, "max": 100000.0, "default": 3000.0},
    "flame_smoke": {"min": 0.0, "max": 8.0, "default": 1.0},
    "burning_rate": {"min": 0.01, "max": 4.0, "default": 0.75},
    "smoke_density": {"min": 0.0, "max": 1.0, "default": 0.5},
    "emission_intensity": {"min": 0.0, "max": 100.0, "default": 1.0},
    "domain_resolution": {"min": 32, "max": 512, "default": 96},
    "timesteps_max": {"min": 1, "max": 45, "default": 4},
    "cfl_condition": {"min": 0.0, "max": 10.0, "default": 4.0},
}


@function_tool
def validate_parameter_range(parameter: str, value: float) -> str:
    """
    Validate a parameter value against Blender 5.0 API ranges.

    Args:
        parameter: Parameter name (e.g., "turbulence", "flame_smoke")
        value: Proposed value

    Returns:
        JSON with valid (bool), clamped_value, and warning if out of range
    """
    if parameter not in BLENDER_PARAMETER_RANGES:
        return json.dumps({
            "valid": True,
            "value": value,
            "warning": f"Unknown parameter '{parameter}', cannot validate range"
        })

    range_info = BLENDER_PARAMETER_RANGES[parameter]
    min_val, max_val = range_info["min"], range_info["max"]

    if min_val <= value <= max_val:
        return json.dumps({
            "valid": True,
            "value": value,
            "range": [min_val, max_val]
        })

    clamped = max(min_val, min(max_val, value))
    return json.dumps({
        "valid": False,
        "original_value": value,
        "clamped_value": clamped,
        "range": [min_val, max_val],
        "warning": f"Value {value} out of range [{min_val}, {max_val}], clamped to {clamped}"
    })


@function_tool
def get_parameter_defaults(effect_type: str) -> str:
    """
    Get recommended default parameters for an effect type.

    Args:
        effect_type: Type of effect (sun, explosion, fire, smoke, nebula)

    Returns:
        JSON with recommended default parameter values
    """
    defaults = {
        "sun": {
            "flame_max_temp": 5778.0,
            "emission_intensity": 10.0,
            "turbulence": 0.1,
            "smoke_density": 0.2,
        },
        "explosion": {
            "flame_max_temp": 3000.0,
            "emission_intensity": 5.0,
            "turbulence": 0.8,
            "vorticity": 0.6,
            "burning_rate": 1.5,
        },
        "fire": {
            "flame_max_temp": 2500.0,
            "emission_intensity": 3.0,
            "turbulence": 0.5,
            "burning_rate": 0.75,
        },
        "smoke": {
            "smoke_density": 0.8,
            "turbulence": 0.3,
            "emission_intensity": 0.1,
        },
        "nebula": {
            "smoke_density": 0.3,
            "emission_intensity": 0.5,
            "turbulence": 0.2,
        },
        "pyro": {
            "flame_max_temp": 3000.0,
            "emission_intensity": 4.0,
            "turbulence": 0.6,
            "vorticity": 0.5,
            "burning_rate": 1.0,
        },
    }

    if effect_type not in defaults:
        return json.dumps({
            "warning": f"Unknown effect type '{effect_type}'",
            "available_types": list(defaults.keys())
        })

    return json.dumps({
        "effect_type": effect_type,
        "recommended_defaults": defaults[effect_type]
    })


# =============================================================================
# AI-optimized Docs Expert instructions - compact, search-focused
# =============================================================================

DOC_EXPERT_INSTRUCTIONS = """## ROLE
Search Blender 5.0 docs, find APIs, validate parameters.

## TURN BUDGET: MAX 3 TURNS
T1: semantic_search_blender_docs OR search_blender_api_by_intent
T2: validate_parameter_range (if needed)
T3: Return DocsOutput

## TOOLS
- semantic_search_blender_docs(query) → conceptual search, finds related terms
- search_blender_api_by_intent(intent, domain) → API discovery ("what to do" → "which API")
- find_alternative_approaches(current, issue) → when stuck (2+ iter same issue)
- validate_parameter_range(param, value) → check Blender 5.0 limits
- get_parameter_defaults(effect_type) → sun/explosion/fire/smoke/nebula defaults

## STRATEGY
General question → semantic_search_blender_docs
API unknown → search_blender_api_by_intent
Stuck → find_alternative_approaches
Always validate params before recommending

## OUTPUT (DocsOutput)
- answer: 1-2 sentence summary
- modifications: {param: value}
- rationale: why changes help (cite docs)
- confidence: 0.0-1.0 (docs found → high, no docs → ≤0.3)
- citations: [sources]
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_docs_expert(custom_instructions: str = "") -> Agent:
    """
    Create a documentation expert agent with vector store search tools.

    This function is SYNCHRONOUS (no await needed) because all tools
    are in-process function wrappers.

    Args:
        custom_instructions: Additional instructions to append (e.g., session context)

    Returns:
        Agent instance ready for use (no initialization needed)
    """
    instructions = DOC_EXPERT_INSTRUCTIONS
    if custom_instructions:
        instructions = instructions + "\n\n" + custom_instructions

    return Agent(
        name="Documentation Expert",
        instructions=instructions,
        model=os.getenv("DOC_EXPERT_MODEL", "gpt-5.2"),  # Full reasoning capabilities
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"),
        ),
        tools=[
            # Vector store semantic search tools (NEW - replaces deprecated local tools)
            semantic_search_blender_docs,
            search_blender_api_by_intent,
            find_alternative_approaches,
            # Parameter validation (kept - still useful)
            validate_parameter_range,
            get_parameter_defaults,
        ],
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from agents import Runner
    import asyncio

    async def test():
        print("Testing DocsExpert Agent (vector store mode)...")
        print("-" * 60)

        # Note: create_docs_expert() is synchronous!
        agent = create_docs_expert()
        print(f"Created agent: {agent.name}")
        print(f"Tools: {len(agent.tools)} total")
        tool_names = [t.name for t in agent.tools if hasattr(t, 'name')]
        print(f"Tool names: {tool_names}")

        # Note: FunctionTool objects are not directly callable
        # They must be used through the agent
        print("\nVector store tools ready for agent use")

        # Full agent test (requires OPENAI_API_KEY)
        if os.getenv("OPENAI_API_KEY"):
            print("\nTesting agent with LLM (semantic search)...")
            result = await Runner.run(
                agent,
                "How do I increase smoke density in a Mantaflow simulation?",
                max_turns=15
            )
            print(f"Response: {str(result.final_output)[:500]}...")
        else:
            print("\nSkipping LLM test (OPENAI_API_KEY not set)")

        print("-" * 60)
        print("Test complete!")

    asyncio.run(test())
