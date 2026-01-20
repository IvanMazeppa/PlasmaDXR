"""
Script Writer Agent using OpenAI Agents SDK.

Specialized agent for generating and modifying Blender 5.0 Mantaflow
Python scripts for VFX effects.

Key capabilities:
- Script generation with UCB1 technique selection
- Script modification based on quality feedback
- Parameter validation against Blender API ranges
- Integration with script-generator MCP server
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from tools.script_generator_tools import (
        generate_script,
        modify_script,
        validate_script,
        list_techniques,
        recommend_technique,
        record_technique_outcome,
    )

# Import tools at runtime to avoid circular imports
from tools.script_generator_tools import (
    generate_script,
    modify_script,
    validate_script,
    list_techniques,
    recommend_technique,
    record_technique_outcome,
    apply_space_physics_fix,  # Critical for sun/star/nebula effects
)

# Import vector store documentation tools for API verification
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    search_blender_api_by_intent,
)

# Compact parameter reference (Blender 5.0 FluidDomainSettings/FluidFlowSettings)
PARAM_RANGES = {
    "vorticity": (0.0, 4.0), "flame_vorticity": (0.0, 2.0), "flame_smoke": (0.0, 8.0),
    "burning_rate": (0.01, 4.0), "flame_max_temp": (1.0, 10.0), "flame_ignition": (0.5, 5.0),
    "resolution_max": (32, 512), "alpha": (-5.0, 5.0), "beta": (-5.0, 5.0),
    "dissolve_speed": (1, 10000), "noise_strength": (0.0, 10.0), "temperature": (-10.0, 10.0),
    "fuel_amount": (0.0, 10.0), "velocity_normal": (-100.0, 100.0), "velocity_random": (0.0, 10.0),
}

# AI-optimized Script Writer instructions - compact, research-driven
SCRIPT_WRITER_INSTRUCTIONS = """## ROLE
Generate/modify Blender 5.0 Mantaflow scripts. Research-first approach for unknown issues.

## TURN BUDGET: MAX 5 TURNS
T1: recommend_technique OR analyze feedback
T2: generate_script OR modify_script (SINGLE CALL with ALL changes batched)
T3: apply_space_physics_fix (if sun/star/nebula)
T4: validate_script
T5: Return ScriptOutput

CRITICAL: Batch ALL parameter changes into ONE modify_script call. Never call modify_script multiple times.

## NEW SCRIPT WORKFLOW
1. recommend_technique(effect_type, description)
2. generate_script(effect_type, description, output_name, technique_name=recommended)
3. IF sun/star/nebula: apply_space_physics_fix(script_path)
4. validate_script(script_path)
5. Return {script_path, technique_used, parameters, validation}

## MODIFICATION WORKFLOW
1. Parse quality feedback → identify ALL visual issues
2. Research unknown issues: search_blender_api_by_intent(issue, "fluid")
3. Map issues → parameters (batch ALL into single dict)
4. modify_script(script_path, modifications_json, output_name) — ONCE
5. validate_script → Return

## RESEARCH-FIRST APPROACH
Unknown issue? DON'T guess. Search first:
- search_blender_api_by_intent("what causes upward motion", "fluid") → discovers scene.gravity, alpha, beta
- semantic_search_blender_docs("emitter position fluid simulation")

## PARAMETER MAPPING (use as hints, verify via search for unknowns)
VISUAL→PARAM:
- thin/transparent → density↑, flame_smoke↑
- dark/dim → emission_intensity↑, blackbody_intensity↑
- rising/drifting → alpha=0, beta=0, scene.gravity=(0,0,0)
- no turbulence → vorticity↑, flame_vorticity↑
- clipped edges → domain_scale↑, emitter position
- burns too fast → burning_rate↓

## SPACE EFFECTS (sun/star/nebula)
ALWAYS call apply_space_physics_fix() — handles: scene.gravity=0, alpha=0, beta=0, centered emitter.
Mantaflow may not be ideal for static stellar objects. Consider shader-based approaches if sim produces no motion.

## PATH HANDLING
Use EXACT paths from tool responses. Never modify/prefix paths.

## OUTPUT
Return ScriptOutput with: script_path, technique_used, parameters, validation, warnings
"""


class ScriptWriterAgent:
    """
    Script Writer Agent for Blender 5.0 VFX script generation.

    Uses gpt-5.2 with high reasoning effort for intelligent script generation
    with UCB1-based technique selection for exploration/exploitation balance.
    """

    def __init__(self, model: str = "gpt-5.2"):
        """
        Initialize the script writer agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for intelligent code generation)
        """
        self.model = os.getenv("SCRIPT_WRITER_MODEL", model)
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., current session state)
        """
        instructions = SCRIPT_WRITER_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Script Writer",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning={
                    "effort": "high"  # High reasoning for complex code generation
                },
                # Note: temperature not supported with gpt-5.2 reasoning models
            ),
            tools=[
                # Script generation tools
                generate_script,
                modify_script,
                validate_script,
                list_techniques,
                recommend_technique,
                record_technique_outcome,
                # Space physics correction (critical for sun/star/nebula)
                apply_space_physics_fix,
                # Documentation search tools (verify API usage, find alternatives)
                semantic_search_blender_docs,
                search_blender_api_by_intent,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("ScriptWriterAgent not initialized. Call initialize() first.")
        return self._agent


def create_script_writer(custom_instructions: str = "") -> Agent:
    """
    Factory function to create and initialize a script writer agent.

    Args:
        custom_instructions: Additional context to append

    Returns:
        Initialized Agent instance ready for use
    """
    writer = ScriptWriterAgent()
    writer.initialize(custom_instructions=custom_instructions)
    return writer.agent


# Convenience alias matching the agents/__init__.py export
ScriptWriterAgent = ScriptWriterAgent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing ScriptWriterAgent...")
        print("-" * 60)

        agent = create_script_writer()

        # Test script generation
        result = await Runner.run(
            agent,
            "Generate a Blender script for a rising mushroom cloud explosion with bright orange flames"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
