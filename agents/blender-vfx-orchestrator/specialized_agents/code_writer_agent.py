"""
Code Writer Agent for Verified Blender Script Generation.

This agent receives a VERIFIED APISpec from the API Spec Agent and
writes Blender Python code using ONLY the verified attributes.

Key insight: The agent cannot hallucinate attributes because:
1. It only receives verified attributes in the APISpec
2. The output guardrail validates code against the spec
3. Any unverified attribute usage is rejected

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

# Import the output model
from models.api_spec import VerifiedScriptOutput

# Import the output guardrail
from guardrails.api_spec_guardrails import validate_code_against_spec

# Import script writing tools
from tools.script_generator_tools import (
    write_script,
    validate_script,
)


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================

CODE_WRITER_AGENT_INSTRUCTIONS = """## ROLE
You are the Code Writer Agent. Your job is to write Blender Python scripts
using ONLY the verified attributes provided in the APISpec.

## CRITICAL RULES
1. You MUST ONLY use attributes listed in the APISpec.
2. Do NOT invent or guess ANY attribute names.
3. Copy attribute names EXACTLY from the spec (CASE SENSITIVE).
4. If you need an attribute not in the spec, STOP and report it.
5. Use variable names `dset` for domain_settings and `fset` for flow_settings.

## VARIABLE NAMING CONVENTION (REQUIRED)
The guardrail validates your code by looking for these patterns:
- dset.attribute_name = value  (for FluidDomainSettings)
- fset.attribute_name = value  (for FluidFlowSettings)

You MUST use these exact variable names:
```python
# After creating the fluid modifier:
mod = domain_obj.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'DOMAIN'
dset = mod.domain_settings  # MUST be named 'dset'

# For flow objects:
mod = flow_obj.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'FLOW'
fset = mod.flow_settings  # MUST be named 'fset'
```

## TURN BUDGET (4-6 turns maximum)
T1: Review the APISpec and plan the script structure
T2: Write the complete Blender Python script
T3: Call write_script(code=..., output_name=..., technique_name=...)
T4: Call validate_script(script_path)
T5: Return VerifiedScriptOutput

## SCRIPT STRUCTURE TEMPLATE
```python
import bpy
from pathlib import Path

# ASSET/OUTPUT SETTINGS (from request)
ASSET_NAME = "{asset_name}"
OUTPUT_DIR = f"/path/to/output/{ASSET_NAME}"
RENDER_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.png"

# EFFECT PARAMETERS (from APISpec)
DOMAIN_RESOLUTION = {resolution}  # From spec
FRAME_START = 1
FRAME_END = {frames}

# Setup directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Clear scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Create domain
def create_domain():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 1))
    domain = bpy.context.active_object
    domain.name = 'FluidDomain'

    mod = domain.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'DOMAIN'
    dset = mod.domain_settings  # REQUIRED NAME

    # Set ONLY attributes from APISpec
    dset.domain_type = 'GAS'  # If in spec
    dset.resolution_max = DOMAIN_RESOLUTION  # If in spec
    # ... other verified attributes ...

    return domain

# Create flow
def create_flow():
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.5, location=(0, 0, 0.5))
    flow = bpy.context.active_object
    flow.name = 'FluidFlow'

    mod = flow.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'FLOW'
    fset = mod.flow_settings  # REQUIRED NAME

    # Set ONLY attributes from APISpec
    fset.flow_type = 'SMOKE'  # If in spec
    fset.flow_behavior = 'FLOW'  # If in spec
    # ... other verified attributes ...

    return flow

# Main
clear_scene()
scene = bpy.context.scene
scene.frame_start = FRAME_START
scene.frame_end = FRAME_END

domain = create_domain()
flow = create_flow()

# ... rest of script ...
```

## WHAT NOT TO DO
DO NOT use any attributes not in the APISpec:
- ❌ dset.bake_frame_start (doesn't exist)
- ❌ dset.resolution_divisions (wrong name)
- ❌ dset.timesteps_per_frame (wrong name)
- ❌ Any attribute you "remember" but isn't in spec

## OUTPUT CONTRACT
Your output MUST be a VerifiedScriptOutput with:
- script_path: Path to the generated script
- technique_used: Name of the technique used
- parameters_set: Key parameters configured
- apis_used: List of API attributes used (for verification)
- ops_used: List of bpy.ops calls used

## IF MISSING ATTRIBUTES
If the APISpec is missing attributes you need:
1. Do NOT invent them
2. Return a script that works with available attributes
3. Note the missing attributes in your response

Better to generate a simpler working script than a broken one.
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

class CodeWriterAgent:
    """
    Code Writer Agent for generating verified Blender Python scripts.

    This agent receives an APISpec with verified attributes and writes
    code that only uses those attributes. The output guardrail validates
    that no unverified attributes are used.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        use_high_reasoning: bool = True,
    ):
        """
        Initialize the Code Writer Agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for best code generation)
            use_high_reasoning: If True, use high reasoning effort
        """
        self.model = os.getenv("CODE_WRITER_MODEL", model)
        self.use_high_reasoning = use_high_reasoning
        self._agent: Agent | None = None

    def initialize(self) -> None:
        """Initialize the agent with tools and guardrails."""
        # Build model settings
        model_settings_kwargs = {}
        if self.use_high_reasoning:
            model_settings_kwargs["reasoning"] = {"effort": "high"}

        self._agent = Agent[SharedContext](
            name="Code Writer",
            instructions=CODE_WRITER_AGENT_INSTRUCTIONS,
            model=self.model,
            model_settings=ModelSettings(**model_settings_kwargs),
            tools=[
                write_script,
                validate_script,
            ],
            output_type=AgentOutputSchema(VerifiedScriptOutput, strict_json_schema=False),
            output_guardrails=[validate_code_against_spec],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance."""
        if self._agent is None:
            raise RuntimeError("CodeWriterAgent not initialized. Call initialize() first.")
        return self._agent


def create_code_writer_agent(
    model: str = "gpt-5.2",
    use_high_reasoning: bool = True,
) -> Agent:
    """
    Factory function to create and initialize a Code Writer Agent.

    Args:
        model: OpenAI model to use
        use_high_reasoning: If True, use high reasoning effort

    Returns:
        Initialized Agent instance
    """
    agent_wrapper = CodeWriterAgent(
        model=model,
        use_high_reasoning=use_high_reasoning,
    )
    agent_wrapper.initialize()
    return agent_wrapper.agent


def format_api_spec_for_prompt(api_spec) -> str:
    """
    Format an APISpec as a string for inclusion in the Code Writer prompt.

    Args:
        api_spec: APISpec instance

    Returns:
        Formatted string describing the verified attributes
    """
    lines = []
    lines.append("## VERIFIED API SPECIFICATION")
    lines.append(f"Effect type: {api_spec.effect_type}")
    lines.append(f"Technique: {api_spec.technique}")
    lines.append("")

    # Domain attributes
    lines.append("### FluidDomainSettings (use with dset.attribute_name)")
    if api_spec.domain_attributes:
        for attr in api_spec.domain_attributes:
            line = f"- {attr.attribute_name}: {attr.value_type}"
            if attr.example_value is not None:
                line += f" (example: {attr.example_value})"
            if attr.value_range:
                line += f" range: {attr.value_range}"
            lines.append(line)
    else:
        lines.append("- (none verified)")
    lines.append("")

    # Flow attributes
    lines.append("### FluidFlowSettings (use with fset.attribute_name)")
    if api_spec.flow_attributes:
        for attr in api_spec.flow_attributes:
            line = f"- {attr.attribute_name}: {attr.value_type}"
            if attr.example_value is not None:
                line += f" (example: {attr.example_value})"
            if attr.enum_values:
                line += f" values: {attr.enum_values}"
            lines.append(line)
    else:
        lines.append("- (none verified)")
    lines.append("")

    # Operations
    lines.append("### Verified Operations (bpy.ops)")
    if api_spec.ops:
        for op in api_spec.ops:
            lines.append(f"- {op.op_path}")
    else:
        lines.append("- (none verified)")
    lines.append("")

    # Deprecation warnings
    if api_spec.deprecation_warnings:
        lines.append("### Deprecation Warnings")
        for warning in api_spec.deprecation_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## IMPORTANT: ONLY use the attributes listed above!")
    return "\n".join(lines)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from models.api_spec import APISpec, APIAttribute, APIOperation

    async def test():
        print("Testing CodeWriterAgent...")
        print("-" * 60)

        agent = create_code_writer_agent()
        print(f"Agent created: {agent.name}")
        print(f"  Model: {agent.model}")
        print(f"  Tools: {[t.name for t in agent.tools]}")
        print(f"  Output type: {agent.output_type}")
        print(f"  Guardrails: {[g.name for g in agent.output_guardrails]}")

        # Create a sample APISpec
        sample_spec = APISpec(
            effect_type="pyro",
            technique="mantaflow_smoke",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="resolution_max",
                    value_type="int",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max",
                    example_value=128,
                    value_range="[32, 512]",
                ),
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="domain_type",
                    value_type="enum",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#domain_type",
                    example_value="GAS",
                ),
            ],
            flow_attributes=[
                APIAttribute(
                    object_type="FluidFlowSettings",
                    attribute_name="flow_type",
                    value_type="enum",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidFlowSettings.html#flow_type",
                    enum_values=["SMOKE", "FIRE", "BOTH"],
                ),
            ],
            ops=[
                APIOperation(
                    op_path="bpy.ops.fluid.bake_data",
                    doc_ref="blender_python_reference_5_0/bpy.ops.fluid.html#bake_data",
                ),
            ],
        )

        # Format for prompt
        formatted = format_api_spec_for_prompt(sample_spec)
        print(f"\nFormatted API Spec:\n{formatted}")

        print("\nNOTE: Full test requires OpenAI API key. Skipping Runner.run().")

    asyncio.run(test())
