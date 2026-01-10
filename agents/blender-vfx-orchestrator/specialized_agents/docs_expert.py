"""
Documentation Expert Agent using OpenAI Agents SDK.

Uses direct function_tool imports instead of MCP transport
to avoid task group conflicts (MCP nesting architectural issue).

Key changes from previous MCP-based implementation:
- Removed DocsExpertAgent class (used MCPServerStdio)
- Removed DocsExpertConnectionPool class
- Uses function_tool imports from shared.blender_docs_tools
- create_docs_expert() is now synchronous (no await needed!)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from agents import Agent, ModelSettings, function_tool
from openai.types.shared import Reasoning

# Import the 12 function_tools from shared module (in-process, no MCP)
from ..shared import (
    search_manual,
    search_tutorials,
    browse_hierarchy,
    search_vdb_workflow,
    search_python_api,
    search_nodes,
    search_modifiers,
    read_page,
    list_api_modules,
    search_bpy_operators,
    search_bpy_types,
    search_semantic,
)


# =============================================================================
# PARAMETER VALIDATION TOOLS
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
# AGENT INSTRUCTIONS
# =============================================================================

DOC_EXPERT_INSTRUCTIONS = """You are a Blender 5.0 documentation expert.

Your role:
1. Search the official Blender documentation for relevant information
2. Read specific pages when more detail is needed
3. Provide accurate, cited answers with documentation paths
4. Validate parameter values against API ranges before recommending

SEARCH STRATEGY (use these tools in order of preference):
1. search_semantic - For natural language questions
2. search_vdb_workflow - For VDB/volume/caching topics
3. search_python_api - For bpy.ops, bpy.types questions
4. search_bpy_types - For specific type properties (FluidDomainSettings, etc.)
5. search_nodes - For shader/material/geometry node questions
6. read_page - To get full content of promising search results

AVAILABLE SEARCH TOOLS (12 total):
- search_manual: General keyword search across all Blender docs
- search_tutorials: Tutorial and learning resources
- browse_hierarchy: Directory tree navigation
- search_vdb_workflow: VDB/NanoVDB specialized search
- search_python_api: bpy.ops/types documentation
- search_nodes: Shader/compositor/geometry nodes
- search_modifiers: Modifier documentation
- read_page: Full page content retrieval
- list_api_modules: API module listing
- search_bpy_operators: bpy.ops.* search
- search_bpy_types: bpy.types.* search
- search_semantic: AI embedding-based semantic search

WHEN RECOMMENDING PARAMETERS:
- Always use validate_parameter_range() to check values
- Use get_parameter_defaults() for baseline values
- Prefer conservative changes (small increments)
- Maximum 2-3 parameter changes per recommendation

OUTPUT FORMAT:
Return JSON with:
- answer: 1-2 sentence summary of what was found
- modifications: dict of parameter_name -> value
- rationale: Why these changes should help based on documentation
- confidence: 0.0-1.0 (set to 0.3 or lower if docs not found)
- citations: list of documentation paths used

If you cannot find relevant documentation, set confidence to 0.3 or lower
and note in the rationale that the recommendation is based on general
knowledge rather than official docs.
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_docs_expert(custom_instructions: str = "") -> Agent:
    """
    Create a documentation expert agent with direct function tools.

    This is the main factory function. Unlike the previous MCP-based
    implementation, this function is SYNCHRONOUS (no await needed)
    because all tools are in-process function_tool wrappers.

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
        model=os.getenv("DOC_EXPERT_MODEL", "gpt-4o"),  # Cost-effective default
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"),
        ),
        tools=[
            # Blender documentation search (12 tools from shared module)
            search_manual,
            search_tutorials,
            browse_hierarchy,
            search_vdb_workflow,
            search_python_api,
            search_nodes,
            search_modifiers,
            read_page,
            list_api_modules,
            search_bpy_operators,
            search_bpy_types,
            search_semantic,
            # Parameter validation (2 local tools)
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
        print("Testing DocsExpert Agent (function_tool mode)...")
        print("-" * 60)

        # Note: create_docs_expert() is now synchronous!
        agent = create_docs_expert()
        print(f"Created agent: {agent.name}")
        print(f"Tools: {len(agent.tools)} total")

        # Quick test without LLM call
        print("\nTesting direct tool calls:")
        result = validate_parameter_range("turbulence", 0.5)
        print(f"  validate_parameter_range('turbulence', 0.5): {result}")

        result = get_parameter_defaults("explosion")
        print(f"  get_parameter_defaults('explosion'): {result[:100]}...")

        # Full agent test (requires OPENAI_API_KEY)
        if os.getenv("OPENAI_API_KEY"):
            print("\nTesting agent with LLM...")
            result = await Runner.run(
                agent,
                "How do I increase smoke density in a Mantaflow simulation?"
            )
            print(f"Response: {result.final_output}")
        else:
            print("\nSkipping LLM test (OPENAI_API_KEY not set)")

        print("-" * 60)
        print("Test complete!")

    asyncio.run(test())
