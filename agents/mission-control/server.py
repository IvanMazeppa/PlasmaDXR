#!/usr/bin/env python3
"""
Mission-Control Agent - Strategic Orchestrator

Multi-agent RAG orchestrator for PlasmaDX-Clean DirectX 12 rendering project.
Coordinates 4 specialist councils (rendering, materials, physics, diagnostics)
in a hierarchical workflow based on NVIDIA's multi-agent RAG architecture.

Usage:
    python server.py                    # Interactive mode
    python server.py "dispatch task"    # Single query mode

Environment:
    ANTHROPIC_API_KEY - Optional (Max subscribers don't need this)
    PROJECT_ROOT - Project root directory (default: auto-detect)
    LOG_LEVEL - Logging level (default: INFO)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, query
from dotenv import load_dotenv

# Import custom tools
from tools.dispatch import dispatch_plan
from tools.handoff import handoff_to_agent
from tools.record import record_decision
from tools.status import publish_status

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mission-control")


def get_project_root() -> Path:
    """
    Auto-detect project root directory.

    Returns:
        Path to project root (PlasmaDX-Clean directory)
    """
    # Try environment variable first
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # Auto-detect from current file location
    # agents/mission-control/server.py -> ../../
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


def create_mission_control_server() -> Any:
    """
    Create the mission-control MCP server with all custom tools.

    Returns:
        MCP server instance with registered tools

    Note:
        Uses in-process mode (not stdio) for low-latency orchestration.
        All tools are registered and available for agent invocation.
    """
    logger.info("Creating mission-control MCP server...")

    # Create server with all custom tools
    server = create_sdk_mcp_server(
        name="mission-control",
        version="0.1.0",
        tools=[
            dispatch_plan,
            record_decision,
            publish_status,
            handoff_to_agent,
        ],
    )

    logger.info("MCP server created with 4 custom tools")
    return server


def create_agent_options() -> ClaudeAgentOptions:
    """
    Create ClaudeAgentOptions with mission-control configuration.

    Returns:
        Configured options for the agent

    Configuration:
        - Working directory: Project root
        - MCP server: In-process mission-control server
        - Allowed tools: All mission-control tools
        - System prompt: Strategic orchestrator persona

    TODO: Add integration with existing external MCP agents
          (log-analysis-rag, dxr-image-quality-analyst, etc.)
    """
    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")

    # Create the MCP server
    mcp_server = create_mission_control_server()

    # Define system prompt for strategic orchestrator
    system_prompt = """You are Mission-Control, a strategic orchestrator agent for the PlasmaDX-Clean
DirectX 12 volumetric rendering project.

Your responsibilities:
1. Coordinate 4 specialist councils: rendering, materials, physics, diagnostics
2. Dispatch tasks based on priority and dependencies
3. Record all strategic decisions with rationale and artifact links
4. Enforce quality gates (LPIPS visual similarity, FPS performance)
5. Maintain context persistence across sessions

Available tools:
- dispatch_plan: Send work to specialist councils
- record_decision: Log decisions to docs/sessions/SESSION_<date>.md
- publish_status: Query councils and generate status reports
- handoff_to_agent: Delegate to specialist agents (PIX debugging, log analysis, etc.)

Project context available in:
- CLAUDE.md: Project overview and architecture
- docs/MASTER_ROADMAP_V2.md: Development roadmap
- docs/CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md: Multi-agent RAG architecture

Be strategic, data-driven, and thorough in decision-making.
"""

    # Configure options
    options = ClaudeAgentOptions(
        cwd=str(project_root),
        mcp_servers={"mission-control": mcp_server},
        allowed_tools=[
            "mcp__mission-control__dispatch_plan",
            "mcp__mission-control__record_decision",
            "mcp__mission-control__publish_status",
            "mcp__mission-control__handoff_to_agent",
        ],
        system_prompt=system_prompt,
    )

    return options


async def interactive_mode() -> None:
    """
    Run mission-control in interactive mode with continuous conversation.

    Maintains context across multiple queries using ClaudeSDKClient.
    User can ask follow-up questions and the agent remembers context.
    """
    logger.info("Starting interactive mode...")
    print("\n" + "=" * 70)
    print("Mission-Control Agent - Interactive Mode")
    print("=" * 70)
    print("\nType your requests below. Type 'exit' or 'quit' to end session.\n")

    options = create_agent_options()

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("\nEnding session. Goodbye!")
                    break

                if not user_input:
                    continue

                # Send query to agent
                await client.query(user_input)

                # Stream response
                print("\nMission-Control: ", end="", flush=True)
                async for message in client.receive_response():
                    print(message, end="", flush=True)
                print("\n")

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError: {e}\n")


async def single_query_mode(prompt: str) -> None:
    """
    Run a single query and exit.

    Args:
        prompt: User's query/task

    Note:
        Uses query() function which creates a new session for each call.
        For multi-turn conversations, use interactive_mode() instead.
    """
    logger.info(f"Single query mode: {prompt[:50]}...")
    options = create_agent_options()

    print("\n" + "=" * 70)
    print("Mission-Control Agent - Single Query")
    print("=" * 70)
    print(f"\nQuery: {prompt}\n")
    print("Response:\n")

    # Stream response
    async for message in query(prompt=prompt, options=options):
        print(message, end="", flush=True)
    print("\n")


async def main() -> None:
    """
    Main entry point for mission-control agent.

    Modes:
        - No arguments: Interactive mode (continuous conversation)
        - With arguments: Single query mode (one-off task)
    """
    if len(sys.argv) > 1:
        # Single query mode
        prompt = " ".join(sys.argv[1:])
        await single_query_mode(prompt)
    else:
        # Interactive mode
        await interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
