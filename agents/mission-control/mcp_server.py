#!/usr/bin/env python3
"""
Mission-Control MCP Server - FastMCP implementation for Claude Code

This file exposes Mission-Control's strategic tools as an MCP server that
Claude Code can connect to via stdio transport.

Architecture:
- This MCP server provides 4 strategic tools to Claude Code
- The tools coordinate with the autonomous agent (autonomous_agent.py)
- Sessions are logged to docs/sessions/SESSION_<date>.md

Usage:
    As MCP server (Claude Code connects via stdio):
        python mcp_server.py

    For development testing:
        python mcp_server.py --test
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging - use stderr to avoid polluting stdio MCP channel
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("mission-control-mcp")

# Initialize FastMCP server
mcp = FastMCP("mission-control")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ===========================================================================
# MCP TOOLS - Strategic Orchestration
# ===========================================================================

@mcp.tool()
def record_decision(
    decision: str,
    rationale: str,
    artifacts: Optional[list[str]] = None
) -> str:
    """
    Record a strategic decision with rationale and supporting artifacts.

    Decisions are logged to docs/sessions/SESSION_<date>.md with:
    - Decision description
    - Rationale explaining why this choice was made
    - Links to supporting artifacts (PIX captures, screenshots, buffer dumps)
    - Timestamp and agent context

    Args:
        decision: What was decided
        rationale: Why it was decided (evidence-based reasoning)
        artifacts: Optional list of file paths to supporting evidence

    Returns:
        Confirmation of successful recording with file path and timestamp

    Example:
        record_decision(
            decision="Switch from PCSS to raytraced shadows",
            rationale="PIX capture shows PCSS blocker search is bottleneck at 4.2ms. RT shadows projected at 2.1ms.",
            artifacts=["PIX/Captures/shadow_analysis.wpix", "screenshots/before_pcss.png"]
        )
    """
    if artifacts is None:
        artifacts = []

    # Generate session file path
    today = datetime.now().strftime("%Y-%m-%d")
    sessions_dir = PROJECT_ROOT / "docs" / "sessions"
    session_file = sessions_dir / f"SESSION_{today}.md"

    # Create sessions directory if needed
    sessions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format artifact links
    artifact_links = "\n".join(f"  - `{art}`" for art in artifacts) if artifacts else "  - None"

    decision_entry = f"""
## Decision: {decision}

**Timestamp**: {timestamp}
**Agent**: mission-control

**Rationale**:
{rationale}

**Supporting Artifacts**:
{artifact_links}

---
"""

    try:
        is_new_file = not session_file.exists()

        if is_new_file:
            header = f"""# Session Summary - {today}

**Project**: PlasmaDX-Clean Multi-Agent RAG System
**Date**: {today}
**Orchestrator**: mission-control

---

"""
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(header)

        with open(session_file, "a", encoding="utf-8") as f:
            f.write(decision_entry)

        relative_path = session_file.relative_to(PROJECT_ROOT)
        return f"""Decision recorded successfully.

**File**: `{relative_path}`
**Timestamp**: {timestamp}
**Status**: {"New session file created" if is_new_file else "Appended to existing session"}

**Decision**: {decision}
**Rationale**: {rationale[:200]}{'...' if len(rationale) > 200 else ''}
**Artifacts**: {len(artifacts)} file(s) linked"""

    except Exception as e:
        return f"Failed to record decision: {e}"


@mcp.tool()
def dispatch_plan(plan: str, priority: str, council: str) -> str:
    """
    Dispatch a task plan to one of the 4 specialist councils.

    Councils:
    - rendering: DXR raytracing, RTXDI, volumetric rendering, shadows, Gaussian splatting
    - materials: Particle properties, material systems, celestial body types
    - physics: PINN ML, GPU physics, accretion disk dynamics, black hole simulation
    - diagnostics: PIX debugging, buffer analysis, performance profiling, visual quality

    Args:
        plan: Description of the task to execute
        priority: Priority level ("high", "medium", "low")
        council: Target council ("rendering", "materials", "physics", "diagnostics")

    Returns:
        Dispatch confirmation with task ID and status

    Example:
        dispatch_plan(
            plan="Optimize RTXDI M5 temporal accumulation to reduce patchwork artifacts",
            priority="high",
            council="rendering"
        )
    """
    valid_councils = ["rendering", "materials", "physics", "diagnostics"]
    valid_priorities = ["high", "medium", "low"]

    if council not in valid_councils:
        return f"Error: Invalid council '{council}'. Valid councils: {', '.join(valid_councils)}"

    if priority not in valid_priorities:
        return f"Error: Invalid priority '{priority}'. Valid priorities: {', '.join(valid_priorities)}"

    # Generate task ID
    task_id = f"{council}_{priority}_{hash(plan) % 10000:04d}"

    return f"""Task dispatched successfully.

**Council**: {council}
**Priority**: {priority}
**Task ID**: {task_id}
**Status**: queued

**Plan**: {plan[:200]}{'...' if len(plan) > 200 else ''}

**Next Steps**:
1. Council will analyze task requirements
2. Specialist agents will be invoked as needed
3. Results reported via publish_status

Note: Council dispatch currently returns placeholder. Full implementation pending."""


@mcp.tool()
def publish_status() -> str:
    """
    Query all specialist councils and publish a consolidated status report.

    Reports include:
    - Active tasks per council
    - Completion status
    - Blocked tasks and dependencies
    - Quality gate results (LPIPS, FPS, visual quality)
    - Recent decisions and outcomes

    Returns:
        Structured status summary with quality gate metrics

    Example:
        status = publish_status()
        # Returns: "Councils Status: Rendering: healthy (2 active)..."
    """
    # TODO: Query actual council status from session logs
    # TODO: Check quality gates against real metrics

    mock_status = {
        "councils": {
            "rendering": {"status": "healthy", "active": 2, "work": "RTXDI M5 temporal accumulation"},
            "materials": {"status": "idle", "active": 0, "work": None},
            "physics": {"status": "healthy", "active": 1, "work": "PINN C++ integration"},
            "diagnostics": {"status": "idle", "active": 0, "work": None},
        },
        "quality_gates": {
            "lpips": 0.03,
            "fps": 142,
            "visual_quality": 8.5,
        },
    }

    return f"""Mission-Control Status Report

**Councils**:
- Rendering: {mock_status['councils']['rendering']['status']} ({mock_status['councils']['rendering']['active']} active) - {mock_status['councils']['rendering']['work'] or 'idle'}
- Materials: {mock_status['councils']['materials']['status']} ({mock_status['councils']['materials']['active']} active)
- Physics: {mock_status['councils']['physics']['status']} ({mock_status['councils']['physics']['active']} active) - {mock_status['councils']['physics']['work'] or 'idle'}
- Diagnostics: {mock_status['councils']['diagnostics']['status']} ({mock_status['councils']['diagnostics']['active']} active)

**Quality Gates**:
- LPIPS Similarity: {mock_status['quality_gates']['lpips']:.3f} (target: < 0.05)
- FPS Performance: {mock_status['quality_gates']['fps']} (target: > 90)
- Visual Quality: {mock_status['quality_gates']['visual_quality']}/10 (target: > 7)

**Status**: All quality gates PASSING

Note: This is placeholder data. Real implementation will query session logs and agent status."""


@mcp.tool()
def handoff_to_agent(agent: str, task: str, context: Optional[dict[str, Any]] = None) -> str:
    """
    Hand off a task to a specific specialist agent.

    Available agents:
    - dxr-image-quality-analyst: Visual quality analysis, LPIPS ML comparison, screenshot assessment
    - pix-debug: PIX capture analysis, buffer validation, GPU hang diagnosis
    - log-analysis-rag: Log ingestion, diagnostic queries, specialist routing
    - material-system-engineer: Particle materials, shader generation, struct validation
    - path-and-probe: Probe grid optimization, SH coefficient validation, lighting coverage
    - gaussian-analyzer: 3D Gaussian parameters, material property simulation
    - dxr-shadow-engineer: Shadow technique research, shader generation

    Args:
        agent: Target agent name
        task: Task description
        context: Optional dictionary of context data (file paths, parameters, etc.)

    Returns:
        Handoff confirmation with agent acknowledgment

    Example:
        handoff_to_agent(
            agent="dxr-image-quality-analyst",
            task="Compare screenshots before and after RTXDI change",
            context={"before": "screenshots/before.png", "after": "screenshots/after.png"}
        )
    """
    if context is None:
        context = {}

    valid_agents = [
        "dxr-image-quality-analyst",
        "pix-debug",
        "log-analysis-rag",
        "material-system-engineer",
        "path-and-probe",
        "gaussian-analyzer",
        "dxr-shadow-engineer",
        "dxr-volumetric-pyro-specialist",
    ]

    if agent not in valid_agents:
        return f"Error: Unknown agent '{agent}'. Valid agents:\n" + "\n".join(f"  - {a}" for a in valid_agents)

    context_str = "\n".join(f"  - {k}: {v}" for k, v in context.items()) if context else "  - None"

    return f"""Task handed off successfully.

**Agent**: {agent}
**Task**: {task[:200]}{'...' if len(task) > 200 else ''}

**Context Provided**:
{context_str}

**Status**: Agent acknowledged task
**Estimated Completion**: 2-5 minutes (varies by complexity)

The agent will execute the task and report results.

Note: This is a placeholder. To actually invoke the agent, use Claude Code's Task tool
with the appropriate subagent_type, or invoke the agent's MCP tools directly."""


# ===========================================================================
# SERVER STARTUP
# ===========================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        # Test mode - run some sample tool calls
        print("Testing mission-control MCP server tools...\n", file=sys.stderr)

        print("1. Testing record_decision:", file=sys.stderr)
        result = record_decision(
            decision="Test decision",
            rationale="Testing the MCP server",
            artifacts=["test.txt"]
        )
        print(result, file=sys.stderr)

        print("\n2. Testing dispatch_plan:", file=sys.stderr)
        result = dispatch_plan(
            plan="Test plan",
            priority="high",
            council="rendering"
        )
        print(result, file=sys.stderr)

        print("\n3. Testing publish_status:", file=sys.stderr)
        result = publish_status()
        print(result, file=sys.stderr)

        print("\n4. Testing handoff_to_agent:", file=sys.stderr)
        result = handoff_to_agent(
            agent="dxr-image-quality-analyst",
            task="Test task",
            context={"test_key": "test_value"}
        )
        print(result, file=sys.stderr)

        print("\n Tests complete!", file=sys.stderr)
    else:
        # Normal MCP server mode
        logger.info("Starting Mission-Control MCP Server...")
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info("Tools: record_decision, dispatch_plan, publish_status, handoff_to_agent")
        mcp.run()
