#!/usr/bin/env python3
"""
Blender Orchestrator MCP Tools

This module provides MCP tool wrappers for the blender-orchestrator server.
Tools are implemented as async functions decorated with @mcp.tool() from FastMCP.

These tools are the external API - they can be called via MCP by Claude Code.
The internal orchestration logic lives in orchestrator.py and uses direct
Python calls to the underlying MCP servers (script-generator, blender-executor,
asset-evaluator).

Tools:
    - create_asset: Start a new VFX asset generation session
    - resume_session: Resume an interrupted session
    - list_sessions: List available sessions with status
    - get_status: Get current orchestrator status (trust, autonomy, etc.)
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from mcp.server.fastmcp import FastMCP

# =============================================================================
# Configuration
# =============================================================================

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", SCRIPT_DIR.parent.parent))

# State persistence directory
STATE_DIR = PROJECT_ROOT / "build" / "orchestrator_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("blender-orchestrator.tools")

# Create FastMCP server instance for tool registration
mcp = FastMCP("blender-orchestrator")


# =============================================================================
# Orchestrator Instance (Lazy-loaded)
# =============================================================================

_orchestrator_instance = None


def get_orchestrator():
    """
    Lazy-load orchestrator instance.

    This avoids import cycle and ensures orchestrator is only created
    when tools are actually called (not at import time).
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        from orchestrator import BlenderOrchestratorAgent
        _orchestrator_instance = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)
    return _orchestrator_instance


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def create_asset(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: Optional[str] = None,
    semantic_query: Optional[str] = None,
    resolution: int = 96,
    frame_end: int = 50,
    technique_name: Optional[str] = None,
) -> str:
    """
    Start a new VFX asset generation session.

    Creates a Blender simulation, evaluates quality, and iterates until
    quality thresholds are met or max iterations reached.

    Args:
        asset_name: Name for the asset (e.g., "explosion_v1")
        effect_type: Type of effect - one of: pyro, explosion, fire, smoke, nebula, sun
        description: Description of what to create
        reference_path: Optional path to reference image for evaluation
        semantic_query: Optional text description for CLIP evaluation
        resolution: Blender simulation resolution (default 96)
        frame_end: Animation end frame (default 50)
        technique_name: Optional specific technique from catalog

    Returns:
        JSON string with session result including:
        - session_id: Unique session identifier
        - status: completed, failed, or paused
        - best_score: Highest VFX quality score achieved
        - iterations: Number of iterations performed
        - output_dir: Path to VDB/render output
        - passed: Whether quality thresholds were met

    Example:
        create_asset(
            asset_name="mushroom_explosion_v1",
            effect_type="pyro",
            description="Rising mushroom cloud with orange/red fire",
            resolution=96,
            frame_end=50
        )
    """
    orchestrator = get_orchestrator()

    try:
        result = await orchestrator.create_asset(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            semantic_query=semantic_query,
            resolution=resolution,
            frame_end=frame_end,
            technique_name=technique_name,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"create_asset failed: {e}")
        return json.dumps({
            "error": str(e),
            "status": "failed",
            "asset_name": asset_name,
        })


@mcp.tool()
async def resume_session(session_id: str) -> str:
    """
    Resume an interrupted or paused session.

    Loads session state from disk and continues from the last checkpoint.
    Useful for continuing after crashes, manual pauses, or hitting iteration limits.

    Args:
        session_id: Session ID to resume (from list_sessions output)

    Returns:
        JSON string with resumed session result including:
        - session_id: The session ID
        - status: Current status after resumption
        - best_score: Best score (may improve during resumed run)
        - iterations: Total iterations including resumed run

    Example:
        resume_session("mushroom_explosion_v1_20250101_120000")
    """
    orchestrator = get_orchestrator()

    try:
        result = await orchestrator.resume_session(session_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"resume_session failed: {e}")
        return json.dumps({
            "error": str(e),
            "session_id": session_id,
            "status": "resume_failed",
        })


@mcp.tool()
def list_sessions(
    status_filter: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List available sessions with their status and scores.

    Args:
        status_filter: Optional filter - one of: in_progress, completed, failed, paused
        limit: Maximum number of sessions to return (default 20)

    Returns:
        JSON string with list of sessions, each containing:
        - session_id: Unique identifier
        - asset_name: Name of the asset being created
        - status: Current status
        - best_score: Best VFX quality score achieved
        - iterations: Number of iterations performed
        - updated_at: Last update timestamp

    Example:
        list_sessions(status_filter="in_progress")
    """
    orchestrator = get_orchestrator()

    try:
        sessions = orchestrator.list_sessions(
            status_filter=status_filter,
            limit=limit,
        )
        return json.dumps(sessions, indent=2)
    except Exception as e:
        logger.error(f"list_sessions failed: {e}")
        return json.dumps({
            "error": str(e),
            "sessions": [],
        })


@mcp.tool()
def get_status() -> str:
    """
    Get current orchestrator status including trust score and autonomy level.

    Returns:
        JSON string with:
        - autonomy:
            - trust_score: 0.0-1.0 trust level
            - autonomy_level: supervised, guided, autonomous, or trusted
            - override_active: Whether level is manually overridden
            - recent_events: List of trust-affecting events
        - guardrails:
            - session: Token/cost limits for current session
            - daily: Daily cost limits
        - workflow:
            - stage: Current workflow stage
            - iteration: Current iteration number
            - best_score: Best score so far
        - session: Current active session info (if any)

    Example:
        get_status()
    """
    orchestrator = get_orchestrator()

    try:
        status = orchestrator.get_status()
        return json.dumps(status, indent=2)
    except Exception as e:
        logger.error(f"get_status failed: {e}")
        return json.dumps({
            "error": str(e),
            "autonomy": {"trust_score": 0.0, "autonomy_level": "unknown"},
        })


# =============================================================================
# Additional Utility Functions (Not MCP Tools, used internally)
# =============================================================================

def get_resumable_sessions() -> List[Dict[str, Any]]:
    """Get list of sessions that can be resumed."""
    orchestrator = get_orchestrator()
    return orchestrator.get_resumable_sessions()


def set_trust_score(score: float) -> None:
    """Set the trust score manually."""
    orchestrator = get_orchestrator()
    orchestrator.set_trust_score(score)


def set_autonomy_override(level: Optional[str]) -> None:
    """Override the autonomy level (or None to use trust score)."""
    orchestrator = get_orchestrator()
    orchestrator.set_autonomy_override(level)


# =============================================================================
# FastMCP Server Access
# =============================================================================

def get_mcp_server() -> FastMCP:
    """Get the FastMCP server instance for this module."""
    return mcp


if __name__ == "__main__":
    # For testing: list registered tools
    print("Blender Orchestrator Tools")
    print("=" * 40)
    print("\nRegistered MCP tools:")
    for tool_name in ["create_asset", "resume_session", "list_sessions", "get_status"]:
        func = globals().get(tool_name)
        if func:
            doc = func.__doc__ or "No documentation"
            first_line = doc.strip().split('\n')[0]
            print(f"  - {tool_name}: {first_line}")
