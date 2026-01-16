"""
Executor Agent using OpenAI Agents SDK.

Specialized agent for executing Blender scripts and handling errors.
Wraps the blender-executor MCP server tools.

Key capabilities:
- Execute Blender Python scripts via CLI
- Parse and categorize Blender/Python errors
- Suggest fixes for common errors
- List and retrieve run outputs
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from tools.blender_executor_tools import (
        execute_blender_script,
        parse_blender_errors,
        list_run_outputs,
        get_latest_run,
    )

# Import tools at runtime
from tools.blender_executor_tools import (
    execute_blender_script,
    parse_blender_errors,
    list_run_outputs,
    get_latest_run,
)


# Agent instructions for Blender execution
EXECUTOR_INSTRUCTIONS = """You execute Blender scripts and handle errors.

Your role:
1. Execute Blender Python scripts with appropriate arguments
2. Parse and categorize errors when execution fails
3. Suggest fixes for common errors
4. Report execution results with output paths

WORKFLOW:
1. Call execute_blender_script() with the script path
2. If execution succeeds:
   - Call list_run_outputs() to get output files (VDB, renders)
   - Return success with output paths
3. If execution fails:
   - Call parse_blender_errors() on stderr/stdout
   - Return structured error with suggested_fix

COMMON ERRORS AND FIXES:

1. AttributeError: 'FluidDomainSettings' has no attribute 'openvdb_cache_compress_type'
   → Blender 5.0 API change. Remove or update the attribute.

2. RuntimeError: Operator bpy.ops.fluid.bake_all poll failed, context is incorrect
   → Set active object: bpy.context.view_layer.objects.active = domain_obj
   → Set mode: bpy.ops.object.mode_set(mode='OBJECT')

3. KeyError: 'bpy_struct[key]: key "xxx" not found'
   → Node/socket name mismatch. Check exact names with node.inputs.keys()

4. MemoryError or GPU out of memory
   → Reduce domain_resolution or frame_end
   → Close other GPU applications

5. FileNotFoundError for output path
   → Ensure output directory exists (os.makedirs with exist_ok=True)

6. Timeout (script takes too long)
   → Reduce resolution, frame_end, or timesteps_max
   → Consider quick preview render first

OUTPUT FORMAT:
Return JSON with:
- success: bool
- script_path: Path that was executed
- run_dir: Directory containing outputs (if success)
- vdb_files: List of VDB files generated (if success)
- render_files: List of render images (if success)
- error: Error message (if failed)
- error_type: PYTHON, BLENDER, CONTEXT, API, etc. (if failed)
- suggested_fix: Recommended fix (if failed)
- execution_time_seconds: How long execution took
"""


class ExecutorAgent:
    """
    Executor Agent for Blender script execution.

    Uses gpt-5.2 with medium reasoning for intelligent error parsing and fix suggestions.
    """

    def __init__(self, model: str = "gpt-5.2"):
        """
        Initialize the executor agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for intelligent error handling)
        """
        self.model = os.getenv("EXECUTOR_MODEL", model)
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., timeout settings)
        """
        instructions = EXECUTOR_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Blender Executor",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning={
                    "effort": "medium"  # Medium reasoning for error analysis
                },
            ),
            tools=[
                execute_blender_script,
                parse_blender_errors,
                list_run_outputs,
                get_latest_run,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("ExecutorAgent not initialized. Call initialize() first.")
        return self._agent


def create_executor(custom_instructions: str = "") -> Agent:
    """
    Factory function to create and initialize an executor agent.

    Args:
        custom_instructions: Additional context to append

    Returns:
        Initialized Agent instance ready for use
    """
    executor = ExecutorAgent()
    executor.initialize(custom_instructions=custom_instructions)
    return executor.agent


# Convenience alias
ExecutorAgent = ExecutorAgent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing ExecutorAgent...")
        print("-" * 60)

        agent = create_executor()

        # Test execution query
        result = await Runner.run(
            agent,
            "Execute the script at assets/blender_scripts/generated/test_explosion.py"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
