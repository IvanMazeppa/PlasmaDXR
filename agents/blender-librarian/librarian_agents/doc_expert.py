"""
Documentation Expert Agent using OpenAI Agents SDK.

Specialized agent for searching and synthesizing Blender 5.0 documentation
using native MCP support to connect to the blender-manual server.

Key capabilities:
- Native MCP integration with blender-manual (12 tools)
- Parameter validation against Blender API ranges
- Structured output for script-generator compatibility
- Automatic tool caching for reduced latency
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from pathlib import Path
from agents import Agent, ModelSettings, function_tool
from agents.mcp import MCPServerStdio
from openai.types.shared import Reasoning


# Blender 5.0 API parameter ranges (from script-generator)
BLENDER_PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "turbulence": {"min": 0.0, "max": 1.0, "default": 0.3},
    "vorticity": {"min": 0.0, "max": 1.0, "default": 0.3},
    "temperature": {"min": 0.0, "max": 100000.0, "default": 5778.0},  # Kelvin
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


# Agent instructions for documentation search and synthesis
DOC_EXPERT_INSTRUCTIONS = """You are a Blender 5.0 documentation expert.

Your role:
1. Search the official Blender documentation for relevant information
2. Read specific pages when more detail is needed
3. Provide accurate, cited answers with documentation paths
4. Validate parameter values against API ranges before recommending

SEARCH STRATEGY (use these MCP tools in order of preference):
1. search_semantic - For natural language questions
2. search_vdb_workflow - For VDB/volume/caching topics
3. search_python_api - For bpy.ops, bpy.types questions
4. search_bpy_types - For specific type properties (FluidDomainSettings, etc.)
5. search_nodes - For shader/material/geometry node questions
6. read_page - To get full content of promising search results

WHEN RECOMMENDING PARAMETERS:
- Always use validate_parameter_range() to check values
- Use get_parameter_defaults() for baseline values
- Prefer conservative changes (small increments)
- Maximum 2-3 parameter changes per recommendation

OUTPUT FORMAT (always respond with valid JSON):
{
    "answer": "1-2 sentence summary of what was found",
    "modifications": {
        "parameter_name": value,
        ...
    },
    "rationale": "Why these changes should help based on documentation",
    "confidence": 0.0-1.0,
    "citations": ["path/to/doc1.html", "path/to/doc2.html"]
}

If you cannot find relevant documentation, set confidence to 0.3 or lower and note that the recommendation is based on general knowledge rather than official docs."""


class DocExpertAgent:
    """
    Documentation Expert using OpenAI Agents SDK with native MCP support.

    Spawns the blender-manual MCP server via stdio and uses GPT-5.2
    to intelligently search and synthesize documentation.
    """

    # Default path to blender-manual server
    DEFAULT_BLENDER_MANUAL_PATH = Path(__file__).parent.parent.parent / "blender-manual"

    def __init__(
        self,
        blender_manual_path: Optional[str] = None,
        model: str = "gpt-5.2",
        cache_tools: bool = True
    ):
        """
        Initialize the documentation expert.

        Args:
            blender_manual_path: Path to blender-manual server directory
            model: OpenAI model to use (default: gpt-5.2)
            cache_tools: Whether to cache MCP tool definitions
        """
        self.blender_manual_path = Path(blender_manual_path) if blender_manual_path else self.DEFAULT_BLENDER_MANUAL_PATH
        self.model = os.getenv("OPENAI_MODEL", model)
        self.cache_tools = cache_tools
        self._agent: Optional[Agent] = None
        self._mcp_server: Optional[MCPServerStdio] = None

    async def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with MCP server connection.

        Args:
            custom_instructions: Additional instructions to append (e.g., session context)
        """
        # Create MCP server connection via stdio (spawns server subprocess)
        venv_python = self.blender_manual_path / "venv" / "bin" / "python"
        server_script = self.blender_manual_path / "blender_server.py"

        # Prepare environment with pre-warm flag for embeddings
        # This eliminates the 10-30s cold start on first semantic search query
        env = os.environ.copy()
        env["BLENDER_MCP_PREWARM_EMBEDDINGS"] = "1"

        self._mcp_server = MCPServerStdio(
            name="blender-manual",
            params={
                "command": str(venv_python),
                "args": [str(server_script)],
                "cwd": str(self.blender_manual_path),
                "env": env  # Pass environment with pre-warm flag
            },
            cache_tools_list=self.cache_tools,
            client_session_timeout_seconds=120.0  # Increased to allow for embedding pre-warm (was 60s)
        )

        # IMPORTANT: Explicitly connect to the MCP server before using it
        await self._mcp_server.connect()

        # Combine base instructions with any custom context
        instructions = DOC_EXPERT_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        # Create the agent with MCP tools and local function tools
        # Use medium reasoning effort for agentic documentation search tasks
        self._agent = Agent(
            name="Documentation Expert",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
                verbosity="low"
            ),
            mcp_servers=[self._mcp_server],
            tools=[validate_parameter_range, get_parameter_defaults]
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("DocExpertAgent not initialized. Call initialize() first.")
        return self._agent

    async def close(self) -> None:
        """Clean up MCP server connection."""
        if self._mcp_server:
            try:
                await self._mcp_server.cleanup()
            except Exception:
                pass  # Ignore cleanup errors
            self._mcp_server = None
        self._agent = None


# =============================================================================
# CONNECTION POOLING
# =============================================================================

class MCPConnectionPool:
    """
    Singleton connection pool for blender-manual MCP server.

    Instead of spawning a new MCP server subprocess for each query,
    this pool maintains a persistent connection that can be reused.

    Benefits:
    - Eliminates 15-30s startup time for each query
    - Keeps embeddings warm in the server process
    - Reduces resource usage from repeated subprocess spawning
    """

    _instance: Optional['MCPConnectionPool'] = None
    _lock: Optional['asyncio.Lock'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        import asyncio
        self._lock = asyncio.Lock()
        self._mcp_server: Optional[MCPServerStdio] = None
        self._connected = False
        self._blender_manual_path = DocExpertAgent.DEFAULT_BLENDER_MANUAL_PATH
        self._initialized = True

    async def get_server(self) -> MCPServerStdio:
        """
        Get the pooled MCP server connection, creating it if necessary.

        Returns:
            Connected MCPServerStdio instance

        Thread-safe via asyncio.Lock.
        """
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._mcp_server is not None and self._connected:
                return self._mcp_server

            # Create new connection
            venv_python = self._blender_manual_path / "venv" / "bin" / "python"
            server_script = self._blender_manual_path / "blender_server.py"

            # Prepare environment with pre-warm flag
            env = os.environ.copy()
            env["BLENDER_MCP_PREWARM_EMBEDDINGS"] = "1"

            self._mcp_server = MCPServerStdio(
                name="blender-manual-pooled",
                params={
                    "command": str(venv_python),
                    "args": [str(server_script)],
                    "cwd": str(self._blender_manual_path),
                    "env": env
                },
                cache_tools_list=True,
                client_session_timeout_seconds=120.0
            )

            await self._mcp_server.connect()
            self._connected = True
            return self._mcp_server

    async def close(self) -> None:
        """Close the pooled connection."""
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._mcp_server:
                try:
                    await self._mcp_server.cleanup()
                except Exception:
                    pass
                self._mcp_server = None
                self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the pool has an active connection."""
        return self._connected and self._mcp_server is not None


# Module-level singleton for connection pooling
_connection_pool: Optional[MCPConnectionPool] = None


def get_connection_pool() -> MCPConnectionPool:
    """Get the global MCP connection pool singleton."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = MCPConnectionPool()
    return _connection_pool


async def create_doc_expert_pooled(
    custom_instructions: str = ""
) -> Agent:
    """
    Factory function to create a doc expert using the pooled MCP connection.

    This version reuses the persistent MCP server connection instead of
    spawning a new subprocess for each agent instance.

    Args:
        custom_instructions: Additional instructions to append

    Returns:
        Initialized Agent instance ready for use
    """
    pool = get_connection_pool()
    mcp_server = await pool.get_server()

    instructions = DOC_EXPERT_INSTRUCTIONS
    if custom_instructions:
        instructions = instructions + "\n\n" + custom_instructions

    agent = Agent(
        name="Documentation Expert (Pooled)",
        instructions=instructions,
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"),
            verbosity="low"
        ),
        mcp_servers=[mcp_server],
        tools=[validate_parameter_range, get_parameter_defaults]
    )

    return agent


async def create_doc_expert(
    blender_manual_path: Optional[str] = None,
    custom_instructions: str = ""
) -> Agent:
    """
    Factory function to create and initialize a documentation expert agent.

    Args:
        blender_manual_path: Path to blender-manual server directory (auto-detected if None)
        custom_instructions: Additional instructions to append

    Returns:
        Initialized Agent instance ready for use
    """
    expert = DocExpertAgent(blender_manual_path=blender_manual_path)
    await expert.initialize(custom_instructions=custom_instructions)
    return expert.agent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing DocExpertAgent...")
        print("-" * 60)

        agent = await create_doc_expert()

        # Test a simple query
        result = await Runner.run(
            agent,
            "How do I increase smoke density in a Mantaflow simulation?"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
