"""
API Spec Agent for Verified Blender API Specifications.

This agent creates verified API specifications by querying Blender 5.0
documentation for every attribute and operation. The output is an APISpec
that the Code Writer can trust.

Key insight: The agent MUST call documentation tools for every attribute.
This is enforced by:
1. Output guardrail rejects specs with missing/invalid doc_refs
2. RunHooks can track doc queries and reject if insufficient

SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agents import Agent, ModelSettings
from agents.agent_output import AgentOutputSchema

if TYPE_CHECKING:
    from agents import RunContextWrapper

# Import SharedContext for Agent generic typing
from models.shared_context import SharedContext

# Import the APISpec model
from models.api_spec import APISpec

# Import the output guardrail
from guardrails.api_spec_guardrails import validate_api_spec

# Import documentation tools
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    search_blender_api_by_intent,
    blender_doc_search_bundle,
)

# Import parameter validation tool if available
try:
    from tools.script_generator_tools import validate_parameter_range
except ImportError:
    validate_parameter_range = None


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================

API_SPEC_AGENT_INSTRUCTIONS = """## ROLE
You are the API Specification Agent. Create a VERIFIED APISpec for Blender 5.0.

## DOC_REF FORMAT - CRITICAL (Read First!)

The doc_ref field MUST include the ATTRIBUTE NAME in the anchor. The validator REJECTS refs without it.

CORRECT (attribute name in anchor):
- "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max"
- "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html#temperature"
- "blender_python_reference_5_0/bpy.types.Scene.html#frame_start"

WRONG (generic class ref - REJECTED BY VALIDATOR):
- "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#bpy.types.FluidDomainSettings.bl_rna_get_subclass_py"
- "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html" (no anchor)

Pattern: blender_python_reference_5_0/bpy.types.{CLASS}.html#{ATTRIBUTE_NAME}

## ENUM VALUES (CRITICAL)

If value_type is **enum**, you MUST include `enum_values`:
- Extract exact enum values from docs (case-sensitive)
- Example: `flow_behavior` → `['INFLOW', 'OUTFLOW', 'GEOMETRY']`
- If enum values cannot be verified, **omit the attribute**

## EXECUTION PLAN

### TURN 1: Bundle-first doc search (MANDATORY)
Call `blender_doc_search_bundle` ONCE with:
- effect_type, description, intent, domain

Use bundle results to populate as many attributes/ops as possible.

### TURN 2: Targeted attribute searches (MAX 6 TOTAL)
Only if an attribute is still missing a valid doc_ref:
- Call `semantic_search_blender_docs` for that specific attribute
- **Hard limit:** 6 calls total

### TURN 3: Ops search (MAX 1)
Call `search_blender_api_by_intent` only if an op doc_ref is missing.

### TURN 4: OUTPUT - NO MORE SEARCHING
Output the APISpec. Do NOT search again after Turn 3.

**TOTAL DOC SEARCHES:**
- 1 bundle + <=6 semantic_search + <=1 intent = <=8 total

## EXTRACTING DOC_REF FROM RESULTS

When you search "FluidDomainSettings resolution_max", construct the doc_ref as:
  "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max"

The anchor (#resolution_max) MUST be the exact attribute_name you're documenting.
Do NOT use class methods like bl_rna_get_subclass_py as anchors.

## COMMON ATTRIBUTES

Domain: resolution_max, domain_type, use_adaptive_timesteps, time_scale, vorticity
Flow: flow_type, flow_behavior, temperature, density, fuel_amount, velocity_normal
Scene: frame_start, frame_end

## HALLUCINATED - DO NOT USE
- resolution_divisions (use resolution_max)
- timesteps_per_frame (use timesteps_maximum)
- bake_frame_start (use scene.frame_start)

## OUTPUT RULES
1. Every attribute's doc_ref MUST contain that attribute's name
2. Incomplete spec is OK - do NOT exceed turn budget
3. After Turn 3, OUTPUT immediately
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

class APISpecAgent:
    """
    API Specification Agent for creating verified Blender API specs.

    This agent queries documentation for every attribute and operation,
    ensuring the Code Writer has a verified spec to work from.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        use_high_reasoning: bool = True,
    ):
        """
        Initialize the API Spec Agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for best reasoning)
            use_high_reasoning: If True, use high reasoning effort
        """
        self.model = os.getenv("API_SPEC_MODEL", model)
        self.use_high_reasoning = use_high_reasoning
        self._agent: Agent | None = None

    def initialize(self) -> None:
        """Initialize the agent with tools and guardrails."""
        # Build tool list
        tools = [
            blender_doc_search_bundle,
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ]
        if validate_parameter_range:
            tools.append(validate_parameter_range)

        # Build model settings
        model_settings_kwargs = {}
        if self.use_high_reasoning:
            model_settings_kwargs["reasoning"] = {"effort": "high"}
        # Avoid large parallel batches that trip loop detection
        model_settings_kwargs["parallel_tool_calls"] = False

        self._agent = Agent[SharedContext](
            name="API Spec Agent",
            instructions=API_SPEC_AGENT_INSTRUCTIONS,
            model=self.model,
            model_settings=ModelSettings(**model_settings_kwargs),
            tools=tools,
            output_type=AgentOutputSchema(APISpec, strict_json_schema=False),
            output_guardrails=[validate_api_spec],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance."""
        if self._agent is None:
            raise RuntimeError("APISpecAgent not initialized. Call initialize() first.")
        return self._agent


def create_api_spec_agent(
    model: str = "gpt-5.2",
    use_high_reasoning: bool = True,
) -> Agent:
    """
    Factory function to create and initialize an API Spec Agent.

    Args:
        model: OpenAI model to use
        use_high_reasoning: If True, use high reasoning effort

    Returns:
        Initialized Agent instance
    """
    agent_wrapper = APISpecAgent(
        model=model,
        use_high_reasoning=use_high_reasoning,
    )
    agent_wrapper.initialize()
    return agent_wrapper.agent


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing APISpecAgent...")
        print("-" * 60)

        agent = create_api_spec_agent()
        print(f"Agent created: {agent.name}")
        print(f"  Model: {agent.model}")
        print(f"  Tools: {[t.name for t in agent.tools]}")
        print(f"  Output type: {agent.output_type}")
        print(f"  Guardrails: {[g.name for g in agent.output_guardrails]}")

        # Quick test prompt
        prompt = """Create API specification for a simple smoke effect.

Effect type: pyro
Technique: mantaflow_smoke

Key parameters needed:
- Domain resolution
- Flow type and behavior
- Temperature and density

Search the documentation for each attribute and include the doc_ref."""

        print(f"\nTest prompt:\n{prompt[:200]}...")
        print("\nNOTE: Full test requires OpenAI API key. Skipping Runner.run().")

    asyncio.run(test())
