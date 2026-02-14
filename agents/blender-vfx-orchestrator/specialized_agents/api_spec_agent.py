"""
API Spec Agent for Verified Blender API Specifications.

This agent creates verified API specifications by querying Blender 5.0
documentation for every attribute and operation. The output is an APISpec
that the Code Writer can trust.

Key insight: The agent MUST call documentation tools for every attribute.
This is enforced by:
1. Output guardrail rejects specs with missing/invalid doc_refs
2. RunHooks can track doc queries and reject if insufficient

SDK Reference: https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md
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

## IMPORTANT: BUNDLE RESULTS ARE PRE-LOADED
The orchestrator has ALREADY called `blender_doc_search_bundle` and injected the results
into your prompt. You do NOT need to call the bundle tool - just parse the provided results.

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

## EXECUTION PLAN - FOLLOW EXACTLY

### TURN 1: Parse pre-loaded bundle results
The prompt contains JSON from `blender_doc_search_bundle`. Extract:
- Attribute names from `related_apis` and `results`
- Doc refs from `doc_refs` field
- Code patterns from `code_snippets`

Construct doc_refs as: blender_python_reference_5_0/bpy.types.{CLASS}.html#{ATTRIBUTE}

### TURN 2: Targeted searches for GAPS ONLY (MAX 4 TOTAL)
Only if a CRITICAL attribute is missing from the bundle:
- Call `semantic_search_blender_docs` for that specific attribute
- **Hard limit:** 4 calls total
- Skip non-critical attributes rather than exceeding limit

### TURN 3: OUTPUT - NO MORE SEARCHING
Output the APISpec. Do NOT search again after Turn 2.

**TOTAL DOC SEARCHES:** <=4 semantic_search (bundle already done)

## EXTRACTING DOC_REF FROM BUNDLE RESULTS

The bundle results contain `doc_refs` and `related_apis`. Map them:
- related_apis: ["bpy.types.FluidDomainSettings.resolution_max"]
  → doc_ref: "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max"

The anchor (#resolution_max) MUST be the exact attribute_name you're documenting.
Do NOT use class methods like bl_rna_get_subclass_py as anchors.

## KNOWN-GOOD ATTRIBUTES (Always Allowed)
The following attributes are WHITELISTED and do not need doc_ref anchor verification.
You SHOULD include them when relevant to the effect. Use the class-level doc_ref (no anchor needed):
  doc_ref: "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html"

### FluidDomainSettings (Domain Setup - CRITICAL)
- domain_type: enum [GAS, LIQUID] — DETERMINES EFFECT TYPE
- resolution_max: int [32-512] — simulation quality
- use_noise: bool — turbulence
- noise_strength: float — turbulence intensity
- noise_scale: int [1-10] — noise detail
- vorticity: float [0-1] — swirl
- use_adaptive_timesteps: bool
- timesteps_max: int [1-45]
- use_dissolve_smoke: bool
- dissolve_speed: int
- use_flip_particles: bool — enable FLIP (liquid)
- flip_ratio: float [0-1] — FLIP particle ratio (liquid)
- particle_radius: float — particle size
- use_mesh: bool — liquid mesh generation
- mesh_concave_upper: float
- mesh_smoothen_pos: int
- mesh_smoothen_neg: int
- cache_directory: str
- cache_type: enum [REPLAY, MODULAR, ALL]
- openvdb_cache_compress_type: enum [ZIP, NONE]
- use_spray_particles: bool (liquid)
- use_foam_particles: bool (liquid)
- use_bubble_particles: bool (liquid)

### FluidFlowSettings (Inflow/Outflow)
- flow_type: enum [SMOKE, FIRE, BOTH, LIQUID]
- flow_behavior: enum [INFLOW, OUTFLOW, GEOMETRY]
- use_initial_velocity: bool
- velocity_normal: float — initial velocity
- temperature: float — heat
- density: float — smoke density

### FluidModifier
- fluid_type: enum [NONE, DOMAIN, FLOW, EFFECTOR]

## COMMON ATTRIBUTES

Domain: resolution_max, domain_type, use_adaptive_timesteps, use_noise, noise_strength, vorticity
Flow: flow_type, flow_behavior, temperature, density, fuel_amount, velocity_normal
Scene: frame_start, frame_end

## HALLUCINATED - DO NOT USE
- resolution_divisions (use resolution_max)
- timesteps_per_frame (use timesteps_max)
- timesteps_maximum (use timesteps_max)
- bake_frame_start (use scene.frame_start)

## OUTPUT RULES
1. Every attribute's doc_ref MUST contain that attribute's name OR be on the known-good whitelist
2. Include ALL whitelisted attributes relevant to the effect type
3. For LIQUID effects: domain_type=LIQUID, use_flip_particles, use_mesh are MANDATORY
4. After Turn 2, OUTPUT immediately - no more searching
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
