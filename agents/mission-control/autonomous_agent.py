#!/usr/bin/env python3
"""
Mission-Control Autonomous Agent

CORRECT IMPLEMENTATION: Agent SDK with autonomous reasoning
- Uses ClaudeSDKClient for independent AI decision-making
- Calls specialist MCP tools (dxr-image-quality-analyst, log-analysis-rag, etc.)
- Strategic orchestration with full autonomy

Architecture:
    Mission-Control (THIS - Autonomous Agent)
        ├─ Uses → dxr-image-quality-analyst (MCP tool server)
        ├─ Uses → log-analysis-rag (MCP tool server)
        ├─ Uses → path-and-probe (MCP tool server)
        └─ Uses → Other specialist MCP servers

Usage:
    python autonomous_agent.py                    # Interactive mode
    python autonomous_agent.py "dispatch task"    # Single query mode

    Or via HTTP (for Claude Code integration):
    uvicorn autonomous_agent:app --port 8001
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mission-control")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class MissionControlAgent:
    """
    Autonomous strategic orchestrator with AI reasoning.

    Key Capabilities:
    - Autonomous decision-making
    - Strategic task coordination
    - Multi-agent workflow orchestration
    - Quality gate enforcement
    - Session persistence
    """

    def __init__(self):
        self.client: Optional[ClaudeSDKClient] = None
        self.options: Optional[ClaudeAgentOptions] = None

    def create_options(self) -> ClaudeAgentOptions:
        """
        Create ClaudeAgentOptions for autonomous agent.

        Configuration:
        - Access to all specialist MCP tool servers
        - Strategic orchestrator system prompt
        - Supervised autonomy mode
        """
        logger.info("Creating autonomous agent configuration...")

        # System prompt for strategic orchestrator
        system_prompt = """You are Mission-Control, an autonomous strategic orchestrator for PlasmaDX-Clean.

**Your Role:**
You have INDEPENDENT AI REASONING for strategic decision-making and multi-agent coordination.

**Core Responsibilities:**
1. **Strategic Analysis** - Analyze problems holistically across rendering, materials, physics, diagnostics
2. **Task Coordination** - Dispatch work to specialist agents via their MCP tools
3. **Quality Gates** - Enforce LPIPS visual similarity (≥0.85) and FPS performance thresholds
4. **Decision Recording** - Log all strategic decisions with rationale and evidence
5. **Supervised Autonomy** - Work autonomously but seek approval for major architecture changes

**Available Specialist MCP Tools:**

**Rendering Specialists:**
- path-and-probe: Probe grid lighting (6 tools: analyze_probe_grid, validate_coverage, diagnose_interpolation, etc.)
- dxr-shadow-engineer: Shadow techniques research, performance analysis, shader generation

**Materials Specialists:**
- gaussian-analyzer: 3D Gaussian structure analysis, performance estimation, material property simulation
- material-system-engineer: Codebase file ops, shader generation, particle struct generation

**Diagnostics Specialists:**
- dxr-image-quality-analyst: LPIPS ML comparison, visual quality assessment, performance comparison
- log-analysis-rag: RAG-based log/buffer search, autonomous diagnosis, specialist routing
- pix-debug: Buffer validation, GPU hang diagnosis, shader execution validation

**How to Work:**
1. **Analyze First** - Use diagnostic tools to gather evidence (logs, PIX, screenshots, buffer dumps)
2. **Coordinate Specialists** - Route tasks to appropriate specialist tools
3. **Make Decisions** - Autonomously recommend solutions with data-driven rationale
4. **Seek Approval** - For major changes (architecture, performance trade-offs), ask user explicitly
5. **Record Everything** - Document decisions, rationale, and evidence

**Communication Style (Per CLAUDE.md):**
- **Brutal Honesty**: "ZERO LIGHTS ACTIVE - catastrophic" NOT "lighting could use refinement"
- **Specific & Quantified**: "LPIPS 0.34 (target: ≥0.85), 66% visual degradation" NOT "some differences"
- **Evidence-Based**: Reference PIX captures, FPS metrics, buffer dumps, screenshots
- **Actionable**: Every problem gets concrete next steps

**Decision Framework:**
1. Gather evidence (diagnostic tools)
2. Analyze holistically (consider rendering + materials + physics + diagnostics)
3. Evaluate trade-offs (performance vs quality vs complexity)
4. Recommend with confidence level (high/medium/low)
5. Seek approval for major decisions
6. Record decision to session log

**Example Autonomous Workflow:**
User: "Probe grid lighting is dim"

You autonomously:
1. Call log-analysis-rag to search recent logs for probe grid issues
2. Call path-and-probe to analyze probe grid configuration
3. Call dxr-image-quality-analyst to assess current visual quality
4. Synthesize findings: "Dispatch mismatch - only 32³ probes updated, not 48³"
5. Recommend fix with evidence
6. After approval, coordinate implementation

Remember: You are an AUTONOMOUS AI AGENT with independent reasoning, not just a tool router.
Work confidently, but seek approval for major architectural changes.
"""

        # Configure agent options
        options = ClaudeAgentOptions(
            cwd=str(PROJECT_ROOT),

            # External MCP specialist servers (stdio transport)
            mcp_servers={
                "dxr-image-quality-analyst": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/dxr-image-quality-analyst && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
                "log-analysis-rag": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/log-analysis-rag && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
                "path-and-probe": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/path-and-probe && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
                "pix-debug": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/pix-debug && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
                "gaussian-analyzer": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/gaussian-analyzer && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
                "material-system-engineer": {
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "cd agents/material-system-engineer && ./run_server.sh"],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
                },
            },

            # Allow access to all specialist MCP tools
            allowed_tools=[
                # DXR Image Quality Analyst
                "mcp__dxr-image-quality-analyst__list_recent_screenshots",
                "mcp__dxr-image-quality-analyst__compare_screenshots_ml",
                "mcp__dxr-image-quality-analyst__assess_visual_quality",
                "mcp__dxr-image-quality-analyst__compare_performance",
                "mcp__dxr-image-quality-analyst__analyze_pix_capture",

                # Log Analysis RAG
                "mcp__log-analysis-rag__ingest_logs",
                "mcp__log-analysis-rag__query_logs",
                "mcp__log-analysis-rag__diagnose_issue",
                "mcp__log-analysis-rag__analyze_pix_capture",
                "mcp__log-analysis-rag__read_buffer_dump",
                "mcp__log-analysis-rag__route_to_specialist",

                # Path & Probe
                "mcp__path-and-probe__analyze_probe_grid",
                "mcp__path-and-probe__validate_probe_coverage",
                "mcp__path-and-probe__diagnose_interpolation",
                "mcp__path-and-probe__optimize_update_pattern",
                "mcp__path-and-probe__validate_sh_coefficients",
                "mcp__path-and-probe__compare_vs_restir",

                # PIX Debug
                "mcp__pix-debug__capture_buffers",
                "mcp__pix-debug__analyze_restir_reservoirs",
                "mcp__pix-debug__analyze_particle_buffers",
                "mcp__pix-debug__diagnose_visual_artifact",
                "mcp__pix-debug__diagnose_gpu_hang",
                "mcp__pix-debug__analyze_dxil_root_signature",
                "mcp__pix-debug__validate_shader_execution",

                # Gaussian Analyzer
                "mcp__gaussian-analyzer__analyze_gaussian_parameters",
                "mcp__gaussian-analyzer__simulate_material_properties",
                "mcp__gaussian-analyzer__estimate_performance_impact",
                "mcp__gaussian-analyzer__compare_rendering_techniques",
                "mcp__gaussian-analyzer__validate_particle_struct",

                # Material System Engineer
                "mcp__material-system-engineer__read_codebase_file",
                "mcp__material-system-engineer__write_codebase_file",
                "mcp__material-system-engineer__search_codebase",
                "mcp__material-system-engineer__generate_material_shader",
                "mcp__material-system-engineer__generate_particle_struct",
                "mcp__material-system-engineer__generate_material_config",
            ],

            system_prompt=system_prompt,
        )

        logger.info("Autonomous agent configuration created")
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"MCP servers configured: {len(options.mcp_servers)}")

        return options

    async def start(self):
        """Initialize autonomous agent"""
        logger.info("Starting Mission-Control autonomous agent...")

        self.options = self.create_options()
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.__aenter__()

        logger.info("✅ Mission-Control autonomous agent ready")

    async def stop(self):
        """Shutdown autonomous agent"""
        if self.client:
            await self.client.__aexit__(None, None, None)
            logger.info("Mission-Control autonomous agent stopped")

    async def query(self, prompt: str) -> str:
        """
        Send query to autonomous agent and get response.

        The agent will autonomously:
        1. Analyze the query
        2. Call appropriate specialist MCP tools
        3. Synthesize findings
        4. Make recommendations
        5. Seek approval for major decisions

        Args:
            prompt: User query or task description

        Returns:
            Full autonomous response (reasoning + tool calls + recommendations)
        """
        if not self.client:
            raise RuntimeError("Agent not started. Call start() first.")

        logger.info(f"Processing query: {prompt[:100]}...")

        # Send query to autonomous agent
        await self.client.query(prompt)

        # Stream autonomous response
        full_response = ""
        async for message in self.client.receive_response():
            # Convert message to string if needed
            message_text = str(message) if not isinstance(message, str) else message
            full_response += message_text
            print(message_text, end="", flush=True)

        print()  # Newline after streaming

        return full_response


# ============================================================================
# Interactive Mode (for testing autonomous agent)
# ============================================================================

async def interactive_mode():
    """
    Interactive autonomous agent mode.

    Run standalone for testing autonomous reasoning.
    Agent will coordinate specialists autonomously.
    """
    print("\n" + "=" * 80)
    print("Mission-Control Autonomous Agent - Interactive Mode")
    print("=" * 80)
    print("\nAutonomous strategic orchestrator with AI reasoning.")
    print("I can coordinate specialist agents, analyze problems, and make decisions.")
    print("\nType your requests below. Type 'exit' or 'quit' to end.\n")

    agent = MissionControlAgent()

    try:
        await agent.start()

        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("\nShutting down Mission-Control. Goodbye!")
                    break

                if not user_input:
                    continue

                # Process with autonomous reasoning
                print("\n[Mission-Control]: ", end="", flush=True)
                await agent.query(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError: {e}\n")

    finally:
        await agent.stop()


async def single_query_mode(prompt: str):
    """
    Single query mode for one-off autonomous tasks.

    Args:
        prompt: Task description
    """
    agent = MissionControlAgent()

    try:
        await agent.start()

        print("\n" + "=" * 80)
        print("Mission-Control Autonomous Agent - Single Query")
        print("=" * 80)
        print(f"\nQuery: {prompt}\n")
        print("Response:\n")

        await agent.query(prompt)

    finally:
        await agent.stop()


async def main():
    """
    Main entry point.

    Modes:
    - No args: Interactive mode
    - With args: Single query mode
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
