#!/usr/bin/env python3
"""
Mission-Control Agent - Strategic Orchestrator with Supervised Autonomy

Tier 1 Strategic Agent (AGENT_HIERARCHY_AND_ROLES.md)
Provides supervised autonomy: AI-powered decision-making with human oversight.

Architecture:
- Claude Agent SDK for AI reasoning and conversation
- MCP servers for tool integration (own tools + external agents)
- Session persistence via record_decision
- Quality gates and multi-agent coordination

Usage:
    python server.py                    # Interactive mode (supervised autonomy)
    python server.py "dispatch task"    # Single query mode

Environment:
    ANTHROPIC_API_KEY - Optional for Max subscribers
    PROJECT_ROOT - Project root directory (auto-detected)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

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


def create_mission_control_server():
    """
    Create Mission-Control's MCP server with all custom tools.

    Returns:
        MCP server instance with 4 strategic tools registered

    Tools:
        - record_decision: Log decisions to session files
        - dispatch_plan: Route tasks to specialist councils
        - publish_status: Aggregate status from all councils
        - handoff_to_agent: Delegate to specialist agents
    """
    logger.info("Creating mission-control MCP server...")

    server = create_sdk_mcp_server(
        name="mission-control",
        version="0.1.0",
        tools=[
            record_decision,
            dispatch_plan,
            publish_status,
            handoff_to_agent,
        ],
    )

    logger.info("MCP server created with 4 custom tools")
    return server


def create_agent_options() -> ClaudeAgentOptions:
    """
    Create ClaudeAgentOptions for Mission-Control.

    Configuration:
        - Working directory: Project root
        - MCP servers: Mission-Control's own tools
        - System prompt: Strategic orchestrator persona
        - Model: Claude Sonnet 4.5 (latest)

    TODO: Add external MCP server connections:
          - pix-debug (PIX capture analysis)
          - log-analysis-rag (Log/buffer RAG search)
          - dxr-image-quality-analyst (LPIPS ML comparison)
          - path-and-probe (Probe grid specialist)
    """
    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")

    # Create Mission-Control's own MCP server
    mcp_server = create_mission_control_server()

    # Define system prompt for strategic orchestrator
    system_prompt = """You are Mission-Control, a strategic orchestrator agent for the PlasmaDX-Clean
DirectX 12 volumetric rendering project.

**Supervised Autonomy Mode**: You work autonomously but seek approval for major decisions.

**Your Responsibilities:**
1. **Strategic Coordination**: Coordinate 4 specialist councils (rendering, materials, physics, diagnostics)
2. **Decision Recording**: Log all strategic decisions with rationale and evidence to session files
3. **Quality Gates**: Enforce LPIPS visual similarity and FPS performance thresholds
4. **Context Persistence**: Maintain session context across conversations
5. **Human Oversight**: Ask for approval on major architecture or performance decisions

**Available Tools:**
- `record_decision`: Log decisions to docs/sessions/SESSION_<date>.md
- `dispatch_plan`: Route tasks to specialist councils (TODO: implementation pending)
- `publish_status`: Query councils and generate status reports (TODO: implementation pending)
- `handoff_to_agent`: Delegate to specialist agents (TODO: implementation pending)

**Project Context:**
- CLAUDE.md: Project overview, architecture, build system
- docs/MASTER_ROADMAP_V2.md: Development roadmap and current status
- docs/CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md: Multi-agent RAG architecture

**Decision-Making Guidelines:**
1. **Analyze First**: Gather data before recommending changes
2. **Seek Approval**: For major decisions (architecture, performance, quality trade-offs)
3. **Record Everything**: Use `record_decision` for all strategic choices
4. **Be Transparent**: Explain rationale with evidence (PIX captures, FPS metrics, LPIPS scores)
5. **Data-Driven**: Base recommendations on measurements, not assumptions

**Communication Style:**
- Direct and technical
- Evidence-based recommendations
- Clear approval requests (e.g., "Recommend X because Y. Approve?")
- No sugar-coating (brutal honesty preferred per CLAUDE.md)

Remember: You are an autonomous strategic advisor, but the user (Ben) has final approval on major decisions.
"""

    # Configure agent options
    options = ClaudeAgentOptions(
        cwd=str(project_root),
        mcp_servers={"mission-control": mcp_server},
        allowed_tools=[
            "mcp__mission-control__record_decision",
            "mcp__mission-control__dispatch_plan",
            "mcp__mission-control__publish_status",
            "mcp__mission-control__handoff_to_agent",
        ],
        system_prompt=system_prompt,
    )

    return options


async def interactive_mode():
    """
    Run Mission-Control in supervised autonomy mode.

    Provides:
        - Continuous conversation with context retention
        - Autonomous analysis and recommendations
        - Human oversight for major decisions
        - Session persistence via record_decision

    The agent will:
        - Analyze problems autonomously
        - Coordinate specialist agents
        - Make recommendations with evidence
        - Seek approval for major changes
        - Record decisions to session logs
    """
    logger.info("Starting supervised autonomy mode...")
    print("\n" + "=" * 80)
    print("Mission-Control Agent - Supervised Autonomy Mode")
    print("=" * 80)
    print("\nI'm your strategic orchestrator with AI-powered analysis and decision-making.")
    print("I'll work autonomously but seek your approval for major decisions.")
    print("\nType your requests below. Type 'exit' or 'quit' to end session.\n")

    options = create_agent_options()

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("\nEnding supervised autonomy session. Goodbye!")
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


async def single_query_mode(prompt: str):
    """
    Run a single query and exit.

    Args:
        prompt: User's query/task

    Use case:
        Quick one-off tasks without maintaining session context.
        For multi-turn conversations, use interactive_mode() instead.
    """
    logger.info(f"Single query mode: {prompt[:50]}...")
    options = create_agent_options()

    print("\n" + "=" * 80)
    print("Mission-Control Agent - Single Query")
    print("=" * 80)
    print(f"\nQuery: {prompt}\n")
    print("Response:\n")

    # Stream response
    async for message in query(prompt=prompt, options=options):
        print(message, end="", flush=True)
    print("\n")


async def main():
    """
    Main entry point for Mission-Control agent.

    Modes:
        - No arguments: Interactive supervised autonomy mode
        - With arguments: Single query mode
    """
    if len(sys.argv) > 1:
        # Single query mode
        prompt = " ".join(sys.argv[1:])
        await single_query_mode(prompt)
    else:
        # Interactive supervised autonomy mode
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
