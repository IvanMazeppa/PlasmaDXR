"""
Script Writer Agent using OpenAI Agents SDK.

Specialized agent for generating and modifying Blender 5.0 Mantaflow
Python scripts for VFX effects.

Key capabilities:
- Script generation with UCB1 technique selection
- Script modification based on quality feedback
- Parameter validation against Blender API ranges
- Integration with script-generator MCP server

SELF-LEARNING: This agent uses DYNAMIC INSTRUCTIONS that inject validated
rules from the knowledge base. NO HARDCODED PHYSICS RULES - rules emerge
from experimentation and are only applied when they have a success rate > 70%.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from agents import RunContextWrapper
    from models.shared_context import SharedContext
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
    # NOTE: apply_space_physics_fix REMOVED - we don't hardcode physics anymore
)

# Import vector store documentation tools for API verification
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    search_blender_api_by_intent,
)

# Import dynamic instructions for self-learning
from tools.dynamic_instructions import (
    dynamic_script_writer_instructions,
    get_script_writer_instructions_static,
    SCRIPT_WRITER_BASE_INSTRUCTIONS,
)

# Compact parameter reference (Blender 5.0 FluidDomainSettings/FluidFlowSettings)
PARAM_RANGES = {
    "vorticity": (0.0, 4.0), "flame_vorticity": (0.0, 2.0), "flame_smoke": (0.0, 8.0),
    "burning_rate": (0.01, 4.0), "flame_max_temp": (1.0, 10.0), "flame_ignition": (0.5, 5.0),
    "resolution_max": (32, 512), "alpha": (-5.0, 5.0), "beta": (-5.0, 5.0),
    "dissolve_speed": (1, 10000), "noise_strength": (0.0, 10.0), "temperature": (-10.0, 10.0),
    "fuel_amount": (0.0, 10.0), "velocity_normal": (-100.0, 100.0), "velocity_random": (0.0, 10.0),
}

# DEPRECATED: Hardcoded instructions replaced by dynamic_instructions.py
# Keeping for reference only - DO NOT USE
SCRIPT_WRITER_INSTRUCTIONS_DEPRECATED = """
[DEPRECATED - See tools/dynamic_instructions.py for current instructions]
"""


class ScriptWriterAgent:
    """
    Script Writer Agent for Blender 5.0 VFX script generation.

    Uses gpt-5.2 with high reasoning effort for intelligent script generation
    with UCB1-based technique selection for exploration/exploitation balance.

    SELF-LEARNING: Uses dynamic_instructions that inject validated rules from
    the knowledge base at runtime. No hardcoded physics rules.
    """

    def __init__(self, model: str = "gpt-5.2", use_dynamic_instructions: bool = True):
        """
        Initialize the script writer agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for intelligent code generation)
            use_dynamic_instructions: If True, use dynamic instructions that query KB.
                                      If False, use static base instructions.
        """
        self.model = os.getenv("SCRIPT_WRITER_MODEL", model)
        self.use_dynamic_instructions = use_dynamic_instructions
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., current session state)
        """
        # Choose instruction source
        if self.use_dynamic_instructions:
            # Dynamic instructions: function that generates instructions at runtime
            # based on accumulated knowledge from the knowledge base
            instructions = dynamic_script_writer_instructions
        else:
            # Static fallback: base instructions without KB integration
            instructions = get_script_writer_instructions_static()
            if custom_instructions:
                instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Script Writer",
            instructions=instructions,  # Can be function OR string
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
                # NOTE: apply_space_physics_fix REMOVED
                # We don't hardcode physics anymore - rules emerge from learning
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


def create_script_writer(
    custom_instructions: str = "",
    use_dynamic_instructions: bool = True
) -> Agent:
    """
    Factory function to create and initialize a script writer agent.

    Args:
        custom_instructions: Additional context to append
        use_dynamic_instructions: If True (default), use dynamic instructions that
                                  inject validated learnings from the knowledge base.
                                  If False, use static base instructions.

    Returns:
        Initialized Agent instance ready for use

    Note:
        When use_dynamic_instructions=True, the instructions are a FUNCTION that
        gets called at the start of each agent run. This allows us to inject
        learnings that have been validated through experimentation.
    """
    writer = ScriptWriterAgent(use_dynamic_instructions=use_dynamic_instructions)
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
