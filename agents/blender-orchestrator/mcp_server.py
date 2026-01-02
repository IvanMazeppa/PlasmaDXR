#!/usr/bin/env python3
"""
Blender VFX Orchestrator - MCP Server (FastMCP implementation)

This file exposes the orchestrator's tools as an MCP server that
Claude Code can connect to via stdio transport.

Architecture:
- This MCP server provides 4 tools to Claude Code
- The tools wrap the BlenderOrchestratorAgent (Claude Agent SDK client)
- Sessions are persisted to build/orchestrator_state/

Usage:
    As MCP server (Claude Code connects via stdio):
        python mcp_server.py

    For testing:
        python mcp_server.py --test
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Configure logging - use stderr to avoid polluting stdio MCP channel
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("blender-orchestrator-mcp")

# Initialize FastMCP server
mcp = FastMCP("blender-orchestrator")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ===========================================================================
# MCP TOOLS - VFX Asset Orchestration
# ===========================================================================

@mcp.tool()
def get_status() -> str:
    """
    Get the current status of the Blender VFX orchestrator.

    Returns:
    - Trust score (0.0 - 1.0)
    - Current autonomy level (supervised, guided, autonomous, trusted)
    - Token usage limits and current usage
    - Active session info if any

    Returns:
        JSON string with orchestrator status
    """
    try:
        from orchestrator import BlenderOrchestratorAgent

        agent = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

        trust_score = agent.autonomy.get_trust_score()
        autonomy_level = agent.autonomy.get_level()

        status = {
            "trust_score": trust_score,
            "autonomy_level": autonomy_level.value,
            "token_limits": {
                "per_session": agent.guardrail_config.token_limits.per_session,
                "per_iteration": agent.guardrail_config.token_limits.per_iteration,
                "absolute_max": agent.guardrail_config.token_limits.absolute_max,
            },
            "cost_limits": {
                "per_session_usd": agent.guardrail_config.cost_limits.per_session,
                "per_day_usd": agent.guardrail_config.cost_limits.per_day,
            },
            "workflow_config": {
                "max_iterations": agent.workflow_config.max_iterations,
                "max_retries": agent.workflow_config.max_retries,
            },
            "quality_thresholds": {
                "vfx_pass": agent.quality_config.vfx_pass_threshold,
                "temporal": agent.quality_config.temporal_threshold,
            },
        }

        # Check for active sessions
        sessions = agent.list_sessions(status_filter="in_progress")
        if sessions:
            status["active_sessions"] = [
                {"session_id": s["session_id"], "asset_name": s.get("asset_name", "unknown")}
                for s in sessions[:5]
            ]

        return json.dumps(status, indent=2)

    except Exception as e:
        logger.error(f"get_status error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_sessions(
    status_filter: Optional[str] = None,
    limit: int = 20
) -> str:
    """
    List all VFX asset generation sessions.

    Returns sessions with their:
    - Session ID
    - Asset name
    - Status (in_progress, completed, failed, paused)
    - Best score achieved
    - Iteration count
    - Creation timestamp

    Args:
        status_filter: Optional filter by status (in_progress, completed, failed)
        limit: Maximum number of sessions to return (default 20)

    Returns:
        JSON string with list of session summaries
    """
    try:
        from orchestrator import BlenderOrchestratorAgent

        agent = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

        sessions = agent.list_sessions(
            status_filter=status_filter,
            limit=limit
        )

        return json.dumps({
            "count": len(sessions),
            "sessions": sessions,
        }, indent=2)

    except Exception as e:
        logger.error(f"list_sessions error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def create_asset(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: Optional[str] = None,
    semantic_query: Optional[str] = None,
    resolution: int = 96,
    frame_end: int = 50,
    technique_name: Optional[str] = None
) -> str:
    """
    Start a new VFX asset generation session with the Blender orchestrator.

    The orchestrator will:
    1. Generate a Blender Python script using script-generator
    2. Execute the simulation using blender-executor
    3. Evaluate quality using asset-evaluator (VFX metrics)
    4. Iterate until quality threshold met or max iterations reached

    Args:
        asset_name: Name for the asset (e.g., "explosion_v1")
        effect_type: Type of effect (pyro, explosion, fire, smoke, nebula, sun)
        description: Description of what to create
        reference_path: Optional path to reference image for evaluation
        semantic_query: Optional text description for CLIP evaluation
        resolution: Blender simulation resolution (default 96)
        frame_end: Animation end frame (default 50)
        technique_name: Optional specific technique to use

    Returns:
        JSON string with session status and results
    """
    try:
        from orchestrator import BlenderOrchestratorAgent

        agent = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

        logger.info(f"Starting asset generation: {asset_name}")
        logger.info(f"  Effect: {effect_type}")
        logger.info(f"  Description: {description[:50]}...")

        await agent.start()

        result = await agent.create_asset(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            semantic_query=semantic_query,
            resolution=resolution,
            frame_end=frame_end,
            technique_name=technique_name,
        )

        await agent.stop()

        return json.dumps({
            "status": "completed",
            "session_id": result.get("session_id", "unknown"),
            "best_score": result.get("best_score", 0),
            "total_iterations": result.get("total_iterations", 0),
            "quality_passed": result.get("quality_passed", False),
            "trust_score": result.get("trust_score", 0),
            "autonomy_level": result.get("autonomy_level", "unknown"),
            "best_render_path": result.get("best_render_path", ""),
            "best_vdb_path": result.get("best_vdb_path", ""),
        }, indent=2)

    except Exception as e:
        logger.error(f"create_asset error: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e),
            "asset_name": asset_name,
        })


@mcp.tool()
async def resume_session(session_id: str) -> str:
    """
    Resume a paused or incomplete VFX asset generation session.

    Continues from the last checkpoint:
    - Loads session state (parameters, scores, iteration count)
    - Continues iteration loop from where it stopped
    - Uses same quality thresholds and configuration

    Args:
        session_id: ID of the session to resume

    Returns:
        JSON string with resumed session results
    """
    if not session_id:
        return json.dumps({
            "status": "error",
            "error": "session_id is required",
        })

    try:
        from orchestrator import BlenderOrchestratorAgent

        agent = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

        logger.info(f"Resuming session: {session_id}")

        await agent.start()
        result = await agent.resume_session(session_id)
        await agent.stop()

        return json.dumps({
            "status": "resumed",
            "session_id": session_id,
            "best_score": result.get("best_score", 0),
            "total_iterations": result.get("total_iterations", 0),
            "quality_passed": result.get("quality_passed", False),
            "trust_score": result.get("trust_score", 0),
            "best_render_path": result.get("best_render_path", ""),
        }, indent=2)

    except Exception as e:
        logger.error(f"resume_session error: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e),
            "session_id": session_id,
        })


# ===========================================================================
# MAIN - Server Entry Point
# ===========================================================================

def main():
    """Run the MCP server."""
    if "--test" in sys.argv:
        # Test mode - run some quick checks
        print("Testing blender-orchestrator MCP server...")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Tools registered: get_status, list_sessions, create_asset, resume_session")

        # Test get_status
        result = get_status()
        print(f"\nget_status result:\n{result[:200]}...")
        print("\nTests passed!")
        return

    # Normal MCP server mode
    logger.info("Starting Blender VFX Orchestrator MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
