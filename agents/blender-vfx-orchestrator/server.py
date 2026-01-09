"""
Blender VFX Orchestrator MCP Server.

FastMCP entry point that exposes the BlenderVFXOrchestrator as an MCP server.
Provides tools for autonomous VFX asset generation using the OpenAI Agents SDK.

Key features:
- create_asset: Start autonomous VFX generation
- resume_session: Continue paused sessions
- list_sessions: View session history
- get_status: Budget and orchestrator status
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    BlenderVFXOrchestrator,
    get_orchestrator,
    is_orchestrator_initialized,
    create_vfx_asset,
    resume_vfx_session,
)
from models.shared_context import (
    AssetRequest,
    EffectType,
    SessionState,
    SessionStatus,
)
from utils import (
    BudgetTracker,
    get_budget_tracker,
    SessionPersistence,
    get_persistence,
)
# DocsExpert is optional (disabled by default for MCP compatibility)
from orchestrator import ENABLE_DOCS_EXPERT


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_VERSION = "1.0.0"


# =============================================================================
# LIFESPAN HANDLER
# =============================================================================

@asynccontextmanager
async def _lifespan(server: FastMCP):
    """
    Lifespan handler for MCP server startup/shutdown.

    NOTE: DocsExpert MCP connection is lazily initialized when first used
    (during create_asset). Pre-warming during lifespan causes TaskGroup
    conflicts because spawning child MCP processes during parent MCP startup
    is not supported.
    """
    startup_time = datetime.now()
    print(f"[blender-vfx-orchestrator] Server started at {startup_time.isoformat()}", file=sys.stderr)

    yield

    # Shutdown: close MCP connections if they were initialized
    if ENABLE_DOCS_EXPERT:
        try:
            from specialized_agents.docs_expert import get_docs_connection_pool
            pool = get_docs_connection_pool()
            if pool.is_connected:
                await pool.close()
                print("[blender-vfx-orchestrator] DocsExpert MCP connection closed", file=sys.stderr)
        except Exception:
            pass

    # Close orchestrator if it was initialized
    # Note: get_orchestrator() would create one if not exists, so we just try/except
    try:
        from orchestrator import _orchestrator
        if _orchestrator is not None:
            await _orchestrator.close()
    except Exception:
        pass

    print("[blender-vfx-orchestrator] Server shutdown complete", file=sys.stderr)


# =============================================================================
# MCP SERVER INSTANCE
# =============================================================================

mcp = FastMCP(
    "blender-vfx-orchestrator",
    lifespan=_lifespan,
)


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
async def create_asset(
    asset_name: str,
    description: str,
    effect_type: str = "pyro",
    reference_path: str = "",
    semantic_query: str = "",
    resolution: int = 96,
    frame_end: int = 50,
    quality_threshold: float = 60.0,
    max_iterations: int = 5,
    ctx: Context = None,
) -> str:
    """
    Start autonomous VFX asset generation.

    The orchestrator will coordinate 5 specialized agents to generate
    a high-quality Blender VFX asset through iterative improvement:

    1. Script Writer - Generates Blender Python scripts
    2. Executor - Runs scripts and handles errors
    3. Quality Analyst - Evaluates render quality with ML metrics
    4. Learning Agent - Tracks experiments and suggests fixes
    5. Docs Expert - Searches Blender documentation when stuck

    Args:
        asset_name: Name for the asset (used for output files)
        description: Description of what to create (e.g., "bright orange mushroom cloud")
        effect_type: Type of effect - pyro, explosion, fire, smoke, nebula, sun
        reference_path: Optional path to reference image for LPIPS comparison
        semantic_query: Optional semantic description for CLIP evaluation
        resolution: Blender simulation resolution (32-512, default 96)
        frame_end: Animation end frame (default 50)
        quality_threshold: Minimum quality score to pass (0-100, default 60)
        max_iterations: Maximum iteration attempts (default 5)

    Returns:
        JSON with session results including:
        - session_id: Unique session identifier
        - status: passed, failed, max_iterations, in_progress
        - best_score: Best quality score achieved
        - best_iteration: Which iteration had best score
        - final_render_path: Path to best render
        - vdb_files: List of generated VDB files
        - total_cost: Session cost in USD
    """
    try:
        # Validate effect type
        try:
            effect = EffectType(effect_type.lower())
        except ValueError:
            valid_types = [e.value for e in EffectType]
            return json.dumps({
                "error": f"Invalid effect_type '{effect_type}'",
                "valid_types": valid_types
            })

        # Check budget before starting
        budget = get_budget_tracker()
        if not budget.can_afford_evaluation():
            status = budget.get_status()
            return json.dumps({
                "error": "Budget exhausted",
                "budget_status": status
            })

        # Create request
        request = AssetRequest(
            asset_name=asset_name,
            description=description,
            effect_type=effect,
            reference_path=reference_path if reference_path else None,
            semantic_query=semantic_query if semantic_query else None,
            resolution=resolution,
            frame_start=1,
            frame_end=frame_end,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )

        # Get orchestrator and create asset
        orchestrator = await get_orchestrator()
        session = await orchestrator.create_asset(request)

        # Format result
        result = {
            "session_id": session.session_id,
            "asset_name": session.request.asset_name,
            "status": session.status.value,
            "current_iteration": session.current_iteration,
            "best_score": session.best_score,
            "best_iteration": session.best_iteration,
            "final_render_path": session.final_render_path,
            "vdb_dir": session.final_vdb_dir,
            "current_issues": session.current_issues,
            "total_cost": budget.get_spent(),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def resume_session(
    session_id: str,
    ctx: Context = None,
) -> str:
    """
    Resume a paused or incomplete VFX generation session.

    Sessions are automatically saved after each iteration. If a session
    was interrupted (e.g., context limit, error), it can be resumed
    from the last checkpoint.

    Args:
        session_id: ID of the session to resume (from create_asset result)

    Returns:
        JSON with updated session state (same format as create_asset)
    """
    try:
        # Get orchestrator and resume
        orchestrator = await get_orchestrator()
        session = await orchestrator.resume_session(session_id)

        budget = get_budget_tracker()

        result = {
            "session_id": session.session_id,
            "asset_name": session.request.asset_name,
            "status": session.status.value,
            "current_iteration": session.current_iteration,
            "best_score": session.best_score,
            "best_iteration": session.best_iteration,
            "final_render_path": session.final_render_path,
            "vdb_dir": session.final_vdb_dir,
            "current_issues": session.current_issues,
            "total_cost": budget.get_spent(),
        }

        return json.dumps(result, indent=2)

    except ValueError as e:
        return json.dumps({
            "error": str(e),
            "hint": "Use list_sessions() to find valid session IDs"
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def list_sessions(
    status_filter: str = "",
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """
    List saved VFX generation sessions.

    Args:
        status_filter: Optional filter by status (in_progress, passed, failed, max_iterations)
        limit: Maximum number of sessions to return (default 20)

    Returns:
        JSON array of session summaries with:
        - session_id, asset_name, status, best_score, iteration_count, created_at
    """
    try:
        persistence = get_persistence()
        all_sessions = persistence.list_sessions()
        total_available = len(all_sessions)

        # Apply filter
        if status_filter:
            try:
                status = SessionStatus(status_filter.lower())
                filtered_sessions = [s for s in all_sessions if s.get("status") == status.value]
            except ValueError:
                valid_statuses = [s.value for s in SessionStatus]
                return json.dumps({
                    "error": f"Invalid status_filter '{status_filter}'",
                    "valid_statuses": valid_statuses
                })
        else:
            filtered_sessions = all_sessions

        # Limit results
        sessions = filtered_sessions[:limit]

        return json.dumps({
            "sessions": sessions,
            "count": len(sessions),
            "total": total_available,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def get_session_details(
    session_id: str,
    ctx: Context = None,
) -> str:
    """
    Get detailed information about a specific session.

    Args:
        session_id: ID of the session to retrieve

    Returns:
        JSON with full session state including iteration history
    """
    try:
        persistence = get_persistence()
        session = persistence.load_session(session_id)

        if not session:
            return json.dumps({
                "error": f"Session not found: {session_id}",
                "hint": "Use list_sessions() to find valid session IDs"
            })

        # Format iteration history
        iterations = []
        for it in session.iterations:
            iterations.append({
                "iteration": it.iteration,
                "score": it.score,
                "passed": it.passed,
                "primary_issue": it.primary_issue,
                "render_path": it.render_path,
            })

        result = {
            "session_id": session.session_id,
            "request": {
                "asset_name": session.request.asset_name,
                "description": session.request.description,
                "effect_type": session.request.effect_type.value,
                "resolution": session.request.resolution,
                "frame_end": session.request.frame_end,
                "quality_threshold": session.request.quality_threshold,
                "max_iterations": session.request.max_iterations,
            },
            "status": session.status.value,
            "current_iteration": session.current_iteration,
            "best_score": session.best_score,
            "best_iteration": session.best_iteration,
            "final_render_path": session.final_render_path,
            "current_script_path": session.current_script_path,
            "vdb_files": session.vdb_files,
            "current_issues": session.current_issues,
            "iterations": iterations,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def get_status(ctx: Context = None) -> str:
    """
    Get orchestrator and budget status.

    Returns:
        JSON with:
        - orchestrator_version: Server version
        - initialized: Whether orchestrator is ready
        - budget: Monthly budget status (spent, remaining, limits)
        - docs_expert_enabled: Whether DocsExpert is enabled
        - docs_expert_connected: Whether DocsExpert MCP is connected (if enabled)
    """
    try:
        budget = get_budget_tracker()
        budget_status = budget.get_status()

        # Check DocsExpert connection (only if enabled)
        docs_connected = False
        if ENABLE_DOCS_EXPERT:
            try:
                from specialized_agents.docs_expert import get_docs_connection_pool
                pool = get_docs_connection_pool()
                docs_connected = pool.is_connected
            except Exception:
                pass

        # Check orchestrator initialization WITHOUT triggering it
        # Calling get_orchestrator() would create MCP client connections,
        # which causes asyncio TaskGroup conflicts inside MCP tool handlers
        initialized = is_orchestrator_initialized()

        # Extract category spending from nested structure
        categories = budget_status.get("categories", {})
        vision_spent = categories.get("vision", {}).get("spent", 0)
        docs_spent = categories.get("docs", {}).get("spent", 0)
        monthly_limit = budget_status["total_spent"] + budget_status["total_remaining"]

        result = {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "initialized": initialized,
            "budget": {
                "monthly_limit_usd": monthly_limit,
                "total_spent_usd": budget_status["total_spent"],
                "remaining_usd": budget_status["total_remaining"],
                "vision_spent_usd": vision_spent,
                "doc_spent_usd": docs_spent,
                "at_80_percent": budget_status["total_spent"] >= monthly_limit * 0.8,
                "exhausted": budget_status["total_spent"] >= monthly_limit,
            },
            "docs_expert_enabled": ENABLE_DOCS_EXPERT,
            "docs_expert_connected": docs_connected,
            "project_root": str(PROJECT_ROOT),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def cancel_session(
    session_id: str,
    ctx: Context = None,
) -> str:
    """
    Cancel an in-progress session.

    Marks the session as cancelled and saves the current state.
    The session cannot be resumed after cancellation.

    Args:
        session_id: ID of the session to cancel

    Returns:
        JSON with cancellation confirmation
    """
    try:
        persistence = get_persistence()
        session = persistence.load_session(session_id)

        if not session:
            return json.dumps({
                "error": f"Session not found: {session_id}"
            })

        if session.status in [SessionStatus.PASSED, SessionStatus.CANCELLED]:
            return json.dumps({
                "error": f"Session already completed with status: {session.status.value}",
                "session_id": session_id
            })

        # Update status and save
        session.status = SessionStatus.CANCELLED
        session.update_timestamp()
        persistence.save_session(session)

        return json.dumps({
            "cancelled": True,
            "session_id": session_id,
            "final_score": session.best_score,
            "iterations_completed": session.current_iteration,
        })

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })


@mcp.tool()
async def get_effect_types(ctx: Context = None) -> str:
    """
    Get available VFX effect types with descriptions.

    Returns:
        JSON with effect types and their recommended use cases
    """
    effect_info = {
        "pyro": {
            "description": "General pyrotechnic effects (fire, smoke, explosions)",
            "use_cases": ["Explosions", "Fire effects", "Smoke plumes"],
            "recommended_resolution": 96,
        },
        "explosion": {
            "description": "Fast, violent detonation effects",
            "use_cases": ["Grenades", "Bombs", "Shockwaves"],
            "recommended_resolution": 128,
        },
        "fire": {
            "description": "Sustained combustion effects",
            "use_cases": ["Campfires", "Torches", "Building fires"],
            "recommended_resolution": 96,
        },
        "smoke": {
            "description": "Non-combustion smoke and vapor",
            "use_cases": ["Fog", "Steam", "Dust clouds"],
            "recommended_resolution": 64,
        },
        "nebula": {
            "description": "Cosmic gas clouds and stellar phenomena",
            "use_cases": ["Space scenes", "Stellar nurseries", "Cosmic backgrounds"],
            "recommended_resolution": 128,
        },
        "sun": {
            "description": "Solar surface and prominence effects",
            "use_cases": ["Star surfaces", "Solar flares", "Prominences"],
            "recommended_resolution": 96,
        },
    }

    return json.dumps({
        "effect_types": effect_info,
        "default": "pyro"
    }, indent=2)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mcp.run()
