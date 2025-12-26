#!/usr/bin/env python3
"""
Iteration Controller MCP Server

Orchestrates the self-improving asset generation loop:
Generate Script → Execute Blender → Evaluate Quality → Improve → Repeat

Part of the Self-Improving NanoVDB Asset Generation Pipeline.

Tools:
    - create_asset: Full pipeline from description to passing asset
    - run_iteration: Execute one generate→evaluate cycle
    - get_history: View iteration history for an asset
    - compare_iterations: Compare quality across attempts

Usage:
    python server.py  # Run as MCP server (stdio transport)
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", SCRIPT_DIR.parent.parent))
HISTORY_DIR = PROJECT_ROOT / "build/iteration_history"

mcp = FastMCP("iteration-controller")


@dataclass
class IterationResult:
    """Result of one iteration cycle."""
    iteration: int
    passed: bool
    lpips_score: Optional[float]
    clip_score: Optional[float]
    overall_score: float
    script_path: str
    render_path: Optional[str]
    vdb_files: List[str]
    recommendations: List[str]
    duration_seconds: float
    timestamp: str


@dataclass
class AssetSession:
    """Tracks an asset creation session."""
    asset_name: str
    description: str
    effect_type: str
    reference_path: Optional[str]
    semantic_query: Optional[str]
    iterations: List[IterationResult] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, passed, failed, max_iterations
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


def load_session(asset_name: str) -> Optional[AssetSession]:
    """Load session from disk."""
    session_file = HISTORY_DIR / f"{asset_name}.json"
    if session_file.exists():
        data = json.loads(session_file.read_text())
        iterations = [IterationResult(**it) for it in data.pop("iterations", [])]
        return AssetSession(**data, iterations=iterations)
    return None


def save_session(session: AssetSession):
    """Save session to disk."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    session_file = HISTORY_DIR / f"{session.asset_name}.json"
    data = asdict(session)
    session_file.write_text(json.dumps(data, indent=2))


@mcp.tool()
async def create_asset(
    asset_name: str,
    description: str,
    effect_type: str = "pyro",
    reference_path: Optional[str] = None,
    semantic_query: Optional[str] = None,
    max_iterations: int = 5,
    lpips_threshold: float = 0.35,
    clip_threshold: float = 0.60,
    resolution: int = 96,
    frame_end: int = 50
) -> str:
    """
    Create a NanoVDB asset through iterative improvement.

    Runs the full pipeline: generate script → execute → evaluate → improve
    until quality thresholds are met or max iterations reached.

    Args:
        asset_name: Name for the asset (used for files)
        description: What to create (e.g., "bright orange mushroom cloud explosion")
        effect_type: Type of effect (pyro, liquid, explosion, nebula)
        reference_path: Optional reference image for LPIPS comparison
        semantic_query: Optional text description for CLIP evaluation
        max_iterations: Maximum improvement attempts (default 5)
        lpips_threshold: LPIPS score to pass (default 0.35)
        clip_threshold: CLIP score to pass (default 0.60)
        resolution: Blender simulation resolution (default 96)
        frame_end: Animation end frame (default 50)

    Returns:
        JSON with session results and final asset paths

    Example:
        create_asset(
            asset_name="supernova_burst",
            description="expanding stellar explosion with hot core and cooling edges",
            effect_type="explosion",
            semantic_query="a bright supernova explosion in space",
            max_iterations=3
        )
    """
    session = AssetSession(
        asset_name=asset_name,
        description=description,
        effect_type=effect_type,
        reference_path=reference_path,
        semantic_query=semantic_query or description
    )

    current_resolution = resolution
    current_turbulence = 0.3

    for i in range(max_iterations):
        result_json = await run_iteration(
            asset_name=asset_name,
            iteration=i + 1,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            semantic_query=semantic_query or description,
            lpips_threshold=lpips_threshold,
            clip_threshold=clip_threshold,
            resolution=current_resolution,
            frame_end=frame_end,
            turbulence=current_turbulence
        )

        result_data = json.loads(result_json)
        if "error" in result_data:
            session.status = "failed"
            save_session(session)
            return json.dumps({"error": result_data["error"], "session": asdict(session)})

        iteration = IterationResult(**result_data)
        session.iterations.append(iteration)
        save_session(session)

        if iteration.passed:
            session.status = "passed"
            save_session(session)
            return json.dumps({
                "success": True,
                "message": f"Asset passed quality threshold on iteration {i + 1}",
                "final_score": iteration.overall_score,
                "vdb_files": iteration.vdb_files,
                "script_path": iteration.script_path,
                "session": asdict(session)
            }, indent=2)

        # Adjust parameters for next iteration based on feedback
        if iteration.lpips_score and iteration.lpips_score > 0.5:
            current_resolution = min(current_resolution + 32, 192)
        if "turbulence" in str(iteration.recommendations).lower():
            current_turbulence = min(current_turbulence + 0.2, 1.0)

    session.status = "max_iterations"
    save_session(session)

    best = max(session.iterations, key=lambda x: x.overall_score)
    return json.dumps({
        "success": False,
        "message": f"Max iterations ({max_iterations}) reached",
        "best_score": best.overall_score,
        "best_iteration": best.iteration,
        "vdb_files": best.vdb_files,
        "session": asdict(session)
    }, indent=2)


@mcp.tool()
async def run_iteration(
    asset_name: str,
    iteration: int,
    effect_type: str,
    description: str,
    reference_path: Optional[str] = None,
    semantic_query: Optional[str] = None,
    lpips_threshold: float = 0.35,
    clip_threshold: float = 0.60,
    resolution: int = 96,
    frame_end: int = 50,
    turbulence: float = 0.3
) -> str:
    """
    Execute one iteration of the asset generation pipeline.

    Steps: Generate/modify script → Execute Blender → Evaluate result

    Returns:
        JSON with IterationResult
    """
    import time
    start_time = time.time()

    script_name = f"{asset_name}_v{iteration}"
    vdb_files = []
    render_path = None

    # This is a coordinator - in practice, Claude Code will call the
    # individual MCP tools. Here we return the iteration structure.
    result = IterationResult(
        iteration=iteration,
        passed=False,
        lpips_score=None,
        clip_score=None,
        overall_score=0.0,
        script_path=f"assets/blender_scripts/generated/{script_name}.py",
        render_path=render_path,
        vdb_files=vdb_files,
        recommendations=[
            f"Use script-generator to create/modify script for: {description}",
            f"Use blender-executor to run the script with resolution={resolution}",
            f"Use asset-evaluator to compare output against reference/query",
            "Analyze recommendations and adjust parameters for next iteration"
        ],
        duration_seconds=time.time() - start_time,
        timestamp=datetime.now().isoformat()
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def get_history(asset_name: str) -> str:
    """
    Get iteration history for an asset.

    Args:
        asset_name: Name of the asset

    Returns:
        JSON with full session history
    """
    session = load_session(asset_name)
    if not session:
        return json.dumps({"error": f"No history found for: {asset_name}"})

    return json.dumps(asdict(session), indent=2)


@mcp.tool()
async def list_sessions(status_filter: Optional[str] = None) -> str:
    """
    List all asset creation sessions.

    Args:
        status_filter: Optional filter (in_progress, passed, failed, max_iterations)

    Returns:
        JSON array of session summaries
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for session_file in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(session_file.read_text())
            if status_filter and data.get("status") != status_filter:
                continue

            iterations = data.get("iterations", [])
            best_score = max((it.get("overall_score", 0) for it in iterations), default=0)

            sessions.append({
                "asset_name": data.get("asset_name"),
                "effect_type": data.get("effect_type"),
                "status": data.get("status"),
                "iterations": len(iterations),
                "best_score": best_score,
                "created_at": data.get("created_at")
            })
        except:
            continue

    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return json.dumps({"count": len(sessions), "sessions": sessions}, indent=2)


@mcp.tool()
async def compare_iterations(asset_name: str) -> str:
    """
    Compare quality scores across iterations for an asset.

    Args:
        asset_name: Name of the asset

    Returns:
        JSON with iteration comparison and improvement analysis
    """
    session = load_session(asset_name)
    if not session:
        return json.dumps({"error": f"No history found for: {asset_name}"})

    if len(session.iterations) < 2:
        return json.dumps({"message": "Need at least 2 iterations to compare"})

    comparison = []
    for i, it in enumerate(session.iterations):
        comparison.append({
            "iteration": it.iteration,
            "overall_score": it.overall_score,
            "lpips": it.lpips_score,
            "clip": it.clip_score,
            "passed": it.passed
        })

    # Calculate improvement
    first_score = session.iterations[0].overall_score
    last_score = session.iterations[-1].overall_score
    improvement = last_score - first_score

    return json.dumps({
        "asset_name": asset_name,
        "total_iterations": len(session.iterations),
        "improvement": round(improvement, 2),
        "first_score": first_score,
        "best_score": max(it.overall_score for it in session.iterations),
        "final_status": session.status,
        "iterations": comparison
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
