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
You are the Code Writer Agent. You write COMPLETE, self-contained Blender Python
scripts that build an entire scene from scratch and produce rendered output.

## TWO CATEGORIES OF API USAGE

### Category 1: Fluid attributes (SPEC-LOCKED)
`dset.*` and `fset.*` assignments MUST use ONLY attribute names from the APISpec.
Do NOT invent or guess fluid attribute names. Copy them EXACTLY (case-sensitive).
- ✅ dset.resolution_max = 128  (if in spec)
- ❌ dset.resolution_divisions = 128  (hallucinated name)
- ❌ dset.bake_frame_start = 1  (doesn't exist)

### Category 2: Standard Blender scene construction (FREE TO USE)
You MUST build a complete scene. The following are NOT constrained by the APISpec:
- Object creation: bpy.ops.mesh.primitive_cube_add, cylinder_add, plane_add, etc.
- Modifiers: bpy.ops.object.modifier_add, obj.modifiers.new()
- Scene setup: bpy.context.scene.frame_start/end, render settings, output paths
- Camera and lighting: bpy.ops.object.camera_add, bpy.ops.object.light_add
- Materials and shaders: bpy.data.materials, node trees, Principled BSDF, Volume shaders
- Rendering: bpy.ops.render.render, scene.render.filepath, image_settings
- Fluid baking: bpy.ops.fluid.bake_all, bpy.ops.fluid.bake_data
- Context: bpy.context, bpy.data, depsgraph, view_layer

Use standard Blender Python freely for everything EXCEPT dset.*/fset.* attribute names.

## VARIABLE NAMING CONVENTION (REQUIRED)
The guardrail validates fluid attributes by looking for these patterns:
- dset.attribute_name = value  (for FluidDomainSettings)
- fset.attribute_name = value  (for FluidFlowSettings)

You MUST use these exact variable names:
```python
mod = domain_obj.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'DOMAIN'
dset = mod.domain_settings  # MUST be named 'dset'

mod = flow_obj.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'FLOW'
fset = mod.flow_settings  # MUST be named 'fset'
```

## WHAT A COMPLETE SCRIPT LOOKS LIKE
Your script MUST include ALL of these sections:
1. Scene clearing and setup (frame range, render engine, samples, output path)
2. Geometry creation (domain cube, emitter objects, environment objects)
3. Fluid modifier setup (domain + flow with dset/fset from APISpec)
4. Materials and shaders (volume shaders for gas, water materials for liquid)
5. Lighting (area lights, environment lighting — use creative judgment)
6. Camera setup (position, focal length, orientation)
7. Baking (bpy.ops.fluid.bake_all or bake_data)
8. Rendering (per-frame still renders with write_still=True)

A complete script is typically 200-400 lines. If your script is under 100 lines,
you are almost certainly missing required sections.

## TURN BUDGET (4-6 turns maximum)
T1: Review the APISpec and plan the script structure
T2: Write the complete Blender Python script
T3: Call write_script(code=..., output_name=..., technique_name=...)
T4: Call validate_script(script_path)
T5: Return VerifiedScriptOutput

## WHAT NOT TO DO
DO NOT hallucinate fluid attribute names:
- ❌ dset.bake_frame_start (doesn't exist)
- ❌ dset.resolution_divisions (wrong name — correct: resolution_max)
- ❌ dset.timesteps_per_frame (wrong name — correct: timesteps_max)
- ❌ Any dset.*/fset.* attribute you "remember" but isn't in the spec

DO NOT generate incomplete scripts:
- ❌ Settings-only scripts that expect pre-existing scene objects
- ❌ Scripts that skip geometry creation, lighting, or materials
- ❌ Scripts that reference undefined variables (dset/fset without creating modifiers)

## OUTPUT CONTRACT
Your output MUST be a VerifiedScriptOutput with:
- script_path: Path to the generated script
- technique_used: Name of the technique used
- parameters_set: Key parameters configured
- apis_used: List of dset.*/fset.* attributes used (for verification)
- ops_used: List of bpy.ops calls used

## IF MISSING FLUID ATTRIBUTES
If the APISpec is missing a fluid attribute you need:
1. Do NOT invent it — skip that specific setting
2. Use sensible Blender defaults for that property
3. Note the missing attribute in validation_errors
The rest of the script (scene, geometry, materials, lighting, rendering) is unaffected.
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
