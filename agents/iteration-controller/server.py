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
    temporal_score: Optional[float]  # Temporal consistency (0-1, higher = better)
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


# =============================================================================
# Iteration Decision Engine
# =============================================================================

# Maps issues to parameter fixes with expected impact
ISSUE_TO_FIXES: Dict[str, List[Dict[str, Any]]] = {
    "TOO DARK": [
        {"param": "flame_max_temp", "action": "increase", "delta": 500, "blender_param": "flame_max_temp"},
        {"param": "emission_multiplier", "action": "increase", "delta": 0.5, "blender_param": "emission_strength"},
        {"param": "burning_rate", "action": "increase", "delta": 0.3, "blender_param": "burning_rate"},
    ],
    "TOO SMALL": [
        {"param": "domain_scale", "action": "increase", "delta": 2.0, "blender_param": "domain_scale"},
        {"param": "flow_radius", "action": "increase", "delta": 0.5, "blender_param": "flow_radius"},
        {"param": "flow_velocity", "action": "increase", "delta": 2.0, "blender_param": "initial_velocity"},
    ],
    "WRONG COLOR": [
        {"param": "burning_rate", "action": "increase", "delta": 0.5, "blender_param": "burning_rate"},
        {"param": "flame_smoke", "action": "increase", "delta": 1.0, "blender_param": "flame_smoke"},
        {"param": "flame_max_temp", "action": "increase", "delta": 300, "blender_param": "flame_max_temp"},
    ],
    "NO STRUCTURE": [
        {"param": "turbulence", "action": "increase", "delta": 0.2, "blender_param": "turbulence_strength"},
        {"param": "vorticity", "action": "increase", "delta": 0.3, "blender_param": "flame_vorticity"},
        {"param": "noise_scale", "action": "increase", "delta": 1.5, "blender_param": "noise_strength"},
    ],
    "LOW CONTRAST": [
        {"param": "density_multiplier", "action": "increase", "delta": 0.3, "blender_param": "density"},
        {"param": "flame_smoke", "action": "increase", "delta": 0.5, "blender_param": "flame_smoke"},
    ],
    "Overexposed": [
        {"param": "flame_max_temp", "action": "decrease", "delta": 300, "blender_param": "flame_max_temp"},
        {"param": "emission_multiplier", "action": "decrease", "delta": 0.3, "blender_param": "emission_strength"},
    ],
    "clipped": [
        {"param": "domain_height", "action": "increase", "delta": 2.0, "blender_param": "domain_size_z"},
        {"param": "flow_position_z", "action": "decrease", "delta": 0.5, "blender_param": "flow_position"},
    ],
}


def diagnose_issues_from_quality(quality_result: Dict) -> List[str]:
    """Extract issue keywords from VFX quality result."""
    issues = quality_result.get("issues", [])
    critical = quality_result.get("critical_issues", [])
    return critical + [i for i in issues if i not in critical]


def suggest_fixes_for_issues(issues: List[str], max_fixes: int = 3) -> List[Dict]:
    """
    Suggest parameter fixes based on identified issues.

    Returns up to max_fixes suggestions, prioritizing critical issues.
    """
    fixes = []
    seen_params = set()

    for issue in issues:
        # Find matching fix category
        for keyword, fix_list in ISSUE_TO_FIXES.items():
            if keyword.upper() in issue.upper():
                for fix in fix_list:
                    if fix["param"] not in seen_params:
                        fixes.append({
                            "issue": issue,
                            "parameter": fix["param"],
                            "blender_parameter": fix["blender_param"],
                            "action": fix["action"],
                            "suggested_delta": fix["delta"],
                            "rationale": f"Addresses: {keyword}"
                        })
                        seen_params.add(fix["param"])
                        if len(fixes) >= max_fixes:
                            return fixes
                break  # Move to next issue after finding match

    return fixes


# =============================================================================
# Session State Persistence (for Claude Code orchestration)
# =============================================================================

ORCHESTRATION_STATE_DIR = PROJECT_ROOT / "build/orchestration_state"


@dataclass
class OrchestrationState:
    """State that persists across Claude Code context resets."""
    session_id: str
    asset_name: str
    effect_type: str
    current_iteration: int
    best_score: float
    best_iteration: int
    parameters_current: Dict[str, Any]
    parameters_tried: List[Dict[str, Any]]
    issues_history: List[List[str]]
    fixes_applied: List[Dict[str, Any]]
    next_action: str
    status: str  # in_progress, paused, completed, failed
    saved_at: str = ""

    def __post_init__(self):
        if not self.saved_at:
            self.saved_at = datetime.now().isoformat()


def save_orchestration_state(state: OrchestrationState):
    """Save orchestration state to disk."""
    ORCHESTRATION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = ORCHESTRATION_STATE_DIR / f"{state.session_id}_state.json"
    state.saved_at = datetime.now().isoformat()
    state_file.write_text(json.dumps(asdict(state), indent=2))


def load_orchestration_state(session_id: str) -> Optional[OrchestrationState]:
    """Load orchestration state from disk."""
    state_file = ORCHESTRATION_STATE_DIR / f"{session_id}_state.json"
    if state_file.exists():
        data = json.loads(state_file.read_text())
        return OrchestrationState(**data)
    return None


# =============================================================================
# MCP Tools - Decision Engine
# =============================================================================

@mcp.tool()
async def diagnose_vfx_issues(
    quality_json: str
) -> str:
    """
    Diagnose issues from VFX quality evaluation and suggest fixes.

    Takes the output from asset-evaluator's evaluate_vfx_quality tool
    and returns specific parameter changes to try.

    Args:
        quality_json: JSON string from evaluate_vfx_quality result

    Returns:
        JSON with:
        - diagnosed_issues: List of issues found
        - suggested_fixes: Parameter changes to try
        - priority_order: Which fix to try first
        - rationale: Why these fixes should help

    Example:
        diagnose_vfx_issues('{"quality": {"issues": ["TOO DARK: ..."], ...}}')
    """
    try:
        data = json.loads(quality_json)
        quality = data.get("quality", data)  # Handle nested or flat structure

        issues = diagnose_issues_from_quality(quality)
        fixes = suggest_fixes_for_issues(issues)

        if not issues:
            return json.dumps({
                "status": "no_issues",
                "message": "No issues found - quality is acceptable",
                "score": quality.get("composite_score", 0)
            }, indent=2)

        return json.dumps({
            "status": "issues_found",
            "diagnosed_issues": issues,
            "issue_count": len(issues),
            "suggested_fixes": fixes,
            "priority_order": [f["parameter"] for f in fixes],
            "recommendation": f"Apply fix for '{fixes[0]['parameter']}' first" if fixes else "Review issues manually"
        }, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_next_iteration_params(
    current_params: str,
    quality_json: str,
    iteration: int = 1
) -> str:
    """
    Get parameters for the next iteration based on current state and issues.

    Intelligently adjusts parameters to address identified quality issues.

    Args:
        current_params: JSON string of current Blender parameters
        quality_json: JSON string from evaluate_vfx_quality result
        iteration: Current iteration number

    Returns:
        JSON with:
        - next_params: Updated parameters for next iteration
        - changes: What was changed and why
        - expected_improvement: What should get better

    Example:
        get_next_iteration_params(
            '{"flame_max_temp": 2000, "turbulence": 0.3}',
            '{"quality": {"issues": ["TOO DARK: ..."]}}',
            iteration=2
        )
    """
    try:
        params = json.loads(current_params)
        quality_data = json.loads(quality_json)
        quality = quality_data.get("quality", quality_data)

        issues = diagnose_issues_from_quality(quality)
        fixes = suggest_fixes_for_issues(issues, max_fixes=2)  # Apply max 2 fixes per iteration

        next_params = params.copy()
        changes = []

        for fix in fixes:
            param = fix["parameter"]
            action = fix["action"]
            delta = fix["suggested_delta"]

            current_value = params.get(param, 0)

            if action == "increase":
                new_value = current_value + delta
            elif action == "decrease":
                new_value = max(0, current_value - delta)
            else:
                new_value = current_value

            next_params[param] = new_value
            changes.append({
                "parameter": param,
                "old_value": current_value,
                "new_value": new_value,
                "reason": fix["rationale"]
            })

        return json.dumps({
            "iteration": iteration + 1,
            "next_params": next_params,
            "changes": changes,
            "issues_addressed": [f["issue"] for f in fixes],
            "expected_improvement": [f["rationale"] for f in fixes] if fixes else ["No changes suggested"]
        }, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# MCP Tools - Orchestration State
# =============================================================================

@mcp.tool()
async def save_iteration_state(
    session_id: str,
    asset_name: str,
    effect_type: str,
    current_iteration: int,
    best_score: float,
    best_iteration: int,
    parameters_current: str,
    issues_current: str,
    next_action: str,
    status: str = "in_progress"
) -> str:
    """
    Save orchestration state so Claude Code can resume after context limit.

    Call this after each iteration to preserve progress.

    Args:
        session_id: Unique session identifier
        asset_name: Name of asset being created
        effect_type: Effect type (explosion, fire, etc.)
        current_iteration: Current iteration number
        best_score: Best VFX quality score so far
        best_iteration: Which iteration had best score
        parameters_current: JSON string of current parameters
        issues_current: JSON array of current issues
        next_action: What to do next (e.g., "run_blender", "evaluate", "adjust_params")
        status: Session status

    Returns:
        JSON confirmation with state file path

    Example:
        save_iteration_state(
            "sun_surface_2024",
            "sun_surface",
            "sun",
            5,
            72.5,
            4,
            '{"flame_max_temp": 3000}',
            '["Needs more structure"]',
            "adjust_params"
        )
    """
    try:
        # Load existing state or create new
        existing = load_orchestration_state(session_id)

        params = json.loads(parameters_current)
        issues = json.loads(issues_current)

        if existing:
            # Update existing state
            existing.current_iteration = current_iteration
            if best_score > existing.best_score:
                existing.best_score = best_score
                existing.best_iteration = best_iteration
            existing.parameters_tried.append(params)
            existing.issues_history.append(issues)
            existing.parameters_current = params
            existing.next_action = next_action
            existing.status = status
            state = existing
        else:
            # Create new state
            state = OrchestrationState(
                session_id=session_id,
                asset_name=asset_name,
                effect_type=effect_type,
                current_iteration=current_iteration,
                best_score=best_score,
                best_iteration=best_iteration,
                parameters_current=params,
                parameters_tried=[params],
                issues_history=[issues],
                fixes_applied=[],
                next_action=next_action,
                status=status
            )

        save_orchestration_state(state)

        return json.dumps({
            "success": True,
            "session_id": session_id,
            "saved_at": state.saved_at,
            "state_file": str(ORCHESTRATION_STATE_DIR / f"{session_id}_state.json"),
            "message": f"State saved at iteration {current_iteration}"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def load_iteration_state(session_id: str) -> str:
    """
    Load saved orchestration state to resume after context limit.

    Use this at the start of a new Claude Code session to continue
    where you left off.

    Args:
        session_id: Session ID to load

    Returns:
        JSON with full orchestration state, or error if not found

    Example:
        load_iteration_state("sun_surface_2024")
    """
    state = load_orchestration_state(session_id)

    if not state:
        # Try to find similar sessions
        ORCHESTRATION_STATE_DIR.mkdir(parents=True, exist_ok=True)
        available = [f.stem.replace("_state", "") for f in ORCHESTRATION_STATE_DIR.glob("*_state.json")]

        return json.dumps({
            "error": f"No state found for session: {session_id}",
            "available_sessions": available[:10],
            "hint": "Use list_orchestration_sessions to see all sessions"
        }, indent=2)

    return json.dumps({
        "success": True,
        "state": asdict(state),
        "resume_instructions": [
            f"Current iteration: {state.current_iteration}",
            f"Best score so far: {state.best_score} (iteration {state.best_iteration})",
            f"Next action: {state.next_action}",
            f"Current parameters: {json.dumps(state.parameters_current)}",
            f"Recent issues: {state.issues_history[-1] if state.issues_history else []}"
        ]
    }, indent=2)


@mcp.tool()
async def list_orchestration_sessions() -> str:
    """
    List all saved orchestration sessions.

    Returns:
        JSON with list of sessions and their status
    """
    ORCHESTRATION_STATE_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for state_file in ORCHESTRATION_STATE_DIR.glob("*_state.json"):
        try:
            data = json.loads(state_file.read_text())
            sessions.append({
                "session_id": data.get("session_id"),
                "asset_name": data.get("asset_name"),
                "effect_type": data.get("effect_type"),
                "current_iteration": data.get("current_iteration"),
                "best_score": data.get("best_score"),
                "status": data.get("status"),
                "saved_at": data.get("saved_at"),
                "next_action": data.get("next_action")
            })
        except:
            continue

    sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

    return json.dumps({
        "count": len(sessions),
        "sessions": sessions
    }, indent=2)


# =============================================================================
# MCP Tools - Asset Creation
# =============================================================================

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
    temporal_threshold: float = 0.7,
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
        temporal_threshold: Temporal consistency to pass (default 0.7, higher = smoother)
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
            temporal_threshold=temporal_threshold,
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
    temporal_threshold: float = 0.7,
    resolution: int = 96,
    frame_end: int = 50,
    turbulence: float = 0.3
) -> str:
    """
    Execute one iteration of the asset generation pipeline.

    Steps: Generate/modify script → Execute Blender → Evaluate spatial → Evaluate temporal

    Returns:
        JSON with IterationResult
    """
    import time
    start_time = time.time()

    script_name = f"{asset_name}_v{iteration}"
    vdb_files = []
    render_path = None
    render_dir = f"build/vdb_output/{asset_name}"

    # This is a coordinator - in practice, Claude Code will call the
    # individual MCP tools. Here we return the iteration structure.
    result = IterationResult(
        iteration=iteration,
        passed=False,
        lpips_score=None,
        clip_score=None,
        temporal_score=None,
        overall_score=0.0,
        script_path=f"assets/blender_scripts/generated/{script_name}.py",
        render_path=render_path,
        vdb_files=vdb_files,
        recommendations=[
            f"1. Use script-generator to create/modify script for: {description}",
            f"2. Use blender-executor to run the script with resolution={resolution}",
            f"3. Use asset-evaluator evaluate_render for spatial quality (LPIPS/CLIP)",
            f"4. If spatial quality passes (LPIPS<{lpips_threshold}, CLIP>{clip_threshold}):",
            f"   Run asset-evaluator analyze_temporal_quality on {render_dir}",
            f"   Temporal threshold: {temporal_threshold} (higher = smoother)",
            "5. If temporal has flickering: increase noise_scale or reduce turbulence",
            "6. Adjust parameters based on recommendations for next iteration"
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
