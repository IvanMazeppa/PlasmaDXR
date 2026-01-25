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
You are the API Specification Agent. Your job is to create VERIFIED API
specifications for Blender 5.0 script generation.

## CRITICAL RULES
1. You MUST call semantic_search_blender_docs for EVERY attribute you plan to use.
2. You MUST include the doc_ref from the search result in your output.
3. Doc refs MUST be API docs (blender_python_reference_5_0/...), not manual pages.
4. You MUST include bpy.ops.* calls in the spec with their doc refs.
5. If you CANNOT find documentation for an attribute, DO NOT include it.
6. NEVER guess or invent attribute names - only use EXACT names from documentation.

## TURN BUDGET (4-6 turns maximum)
T1: semantic_search_blender_docs for FluidDomainSettings attributes
    - Query: "FluidDomainSettings resolution_max" or similar
    - Extract: object_type, attribute_name, value_type, doc_ref

T2: semantic_search_blender_docs for FluidFlowSettings attributes
    - Query: "FluidFlowSettings flow_type" or similar
    - Extract: object_type, attribute_name, value_type, doc_ref

T3: search_blender_api_by_intent for any uncertain APIs or operators
    - Query: "bpy.ops.fluid.bake_data" or similar
    - Extract: op_path, doc_ref

T4: Return complete APISpec with ALL verified attributes

## OUTPUT CONTRACT
Your output MUST be an APISpec with:

domain_attributes: List of FluidDomainSettings attributes
  - Each attribute MUST have:
    - object_type: "FluidDomainSettings"
    - attribute_name: Exact name from docs (CASE SENSITIVE)
    - value_type: From docs (int, float, bool, enum)
    - doc_ref: API doc path (REQUIRED - guardrail will reject without this)

flow_attributes: List of FluidFlowSettings attributes
  - Same requirements as domain_attributes

ops: List of bpy.ops operations
  - Each operation MUST have:
    - op_path: Exact path (e.g., "bpy.ops.fluid.bake_data")
    - doc_ref: API doc path (REQUIRED)

## WHAT TO SEARCH FOR (by effect type)

### Pyro/Fire/Smoke effects:
Domain attributes to verify:
- resolution_max (NOT resolution_divisions!)
- domain_type
- use_adaptive_timesteps (NOT use_adaptive_time_steps!)
- timesteps_maximum (NOT timesteps_per_frame!)
- time_scale
- cache_type
- cache_directory
- cache_data_format
- use_noise
- noise_strength

Flow attributes to verify:
- flow_type
- flow_behavior
- flow_source
- temperature
- density
- velocity_normal
- velocity_random
- use_initial_velocity

Operations to verify:
- bpy.ops.fluid.bake_data
- bpy.ops.fluid.bake_noise (if using noise)

### IMPORTANT: Common Hallucinated Attributes (DO NOT USE)
These attributes do NOT exist in Blender 5.0:
- bake_frame_start (use scene.frame_start instead)
- bake_frame_end (use scene.frame_end instead)
- resolution_divisions (use resolution_max)
- timesteps_per_frame (use timesteps_maximum)
- use_adaptive_time_steps (use use_adaptive_timesteps)

## EXAMPLE SEARCH QUERIES
For FluidDomainSettings.resolution_max:
  semantic_search_blender_docs(
    query="FluidDomainSettings resolution_max",
    max_results=3
  )

For bpy.ops.fluid.bake_data:
  search_blender_api_by_intent(
    intent="bake fluid simulation data",
    domain="fluid"
  )

## STOP CONDITIONS
- After 6 turns, return whatever spec you have
- Do NOT invent attributes you couldn't verify
- Empty spec is better than hallucinated spec
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
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ]
        if validate_parameter_range:
            tools.append(validate_parameter_range)

        # Build model settings
        model_settings_kwargs = {}
        if self.use_high_reasoning:
            model_settings_kwargs["reasoning"] = {"effort": "high"}

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
