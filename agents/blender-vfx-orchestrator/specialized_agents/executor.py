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


# AI-optimized Executor instructions - compact, error-aware
EXECUTOR_INSTRUCTIONS = """## ROLE
Execute Blender scripts, parse errors, suggest fixes.

## TURN BUDGET: MAX 3 TURNS
T1: execute_blender_script(script_path)
T2: list_run_outputs() (success) OR parse_blender_errors(stderr) (fail)
T3: Return ExecutionOutput

## WORKFLOW
SUCCESS: execute → list_run_outputs → return {success, render_path, vdb_files, execution_time}
FAIL: execute → parse_blender_errors → return {success=false, error, error_type, suggested_fix}

## COMMON ERRORS → FIXES
- "FluidDomainSettings has no attribute" → Blender 5.0 API change, remove/update attr
- "bake_all poll failed, context incorrect" → set view_layer.objects.active, mode='OBJECT'
- "KeyError: key not found" → node/socket name mismatch, check node.inputs.keys()
- MemoryError/GPU OOM → reduce resolution or frame_end
- FileNotFoundError → ensure output dir exists (makedirs exist_ok=True)
- Timeout → reduce resolution, frame_end, timesteps_max

## OUTPUT (ExecutionOutput)
- success: bool
- render_path: path to rendered output (if success)
- vdb_files: [VDB paths] (if success)
- error_message: error text (if fail)
- error_type: PYTHON|BLENDER|CONTEXT|API|MEMORY|TIMEOUT
- suggested_fix: fix recommendation (if fail)
- execution_time_seconds: float
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
