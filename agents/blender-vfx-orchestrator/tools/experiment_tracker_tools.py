"""
Function tool wrappers for experiment-tracker MCP server.

Exposes experiment-tracker capabilities to OpenAI Agents SDK agents:
- Session management (start, end, report)
- Experiment recording (baseline, result)
- Knowledge base queries and learning
- Fix suggestions based on accumulated knowledge

These wrappers translate MCP tool calls into @function_tool decorated
functions that agents can call directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents import function_tool

from utils.mcp_connection_pool import get_mcp_server


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@function_tool
async def start_experiment_session(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: str = "",
    semantic_query: str = ""
) -> str:
    """
    Start a new experiment session for tracking iterations on an asset.

    Creates a session that persists across iterations, accumulating:
    - Parameter change history
    - Quality score progression
    - Learnings and warnings
    - Causal relationships between changes and outcomes

    Args:
        asset_name: Name of the asset being created (e.g., "explosion_v1")
        effect_type: Type of effect (explosion, fire, nebula, sun, etc.)
        description: Description of what we're trying to create
        reference_path: Optional path to reference image
        semantic_query: Optional semantic description for evaluation

    Returns:
        JSON with:
        - success: True if session started
        - session_id: Unique identifier for this session
        - message: Confirmation message

    Example:
        start_experiment_session(
            asset_name="mushroom_cloud_v1",
            effect_type="pyro",
            description="A rising mushroom cloud explosion with bright orange fire",
            reference_path="assets/reference_images/explosion/mushroom_ref.jpg"
        )
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "start_experiment_session",
        {
            "asset_name": asset_name,
            "effect_type": effect_type,
            "description": description,
            "reference_path": reference_path,
            "semantic_query": semantic_query,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def end_experiment_session(
    final_status: str,
    best_score: float
) -> str:
    """
    End the current experiment session and generate final report.

    Should be called when:
    - Quality threshold is met (final_status="completed")
    - Max iterations reached (final_status="max_iterations")
    - User cancels (final_status="abandoned")

    Args:
        final_status: Final status ("completed", "abandoned", "max_iterations")
        best_score: Best quality score achieved (0-100)

    Returns:
        JSON with:
        - success: True if session ended
        - final_status: The status provided
        - best_score: Score recorded
        - session_report: Full session summary

    Example:
        end_experiment_session("completed", 78.5)
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "end_experiment_session",
        {
            "final_status": final_status,
            "best_score": best_score,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def get_session_report(
    session_id: str = ""
) -> str:
    """
    Generate a report for an experiment session.

    Includes:
    - Session metadata (asset name, effect type, duration)
    - All iterations with scores
    - Parameter changes and their effects
    - Key learnings discovered
    - Final recommendations

    Args:
        session_id: Session ID (leave empty for current session)

    Returns:
        JSON with session summary and experiment history

    Example:
        get_session_report()  # Current session
        get_session_report("session_20250107_143022")  # Specific session
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "get_session_report",
        {
            "session_id": session_id,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# EXPERIMENT RECORDING
# =============================================================================

@function_tool
async def record_baseline(
    params: str,
    scores: str,
    render_path: str,
    script_path: str
) -> str:
    """
    Record baseline state before running an experiment.

    Call this BEFORE making parameter changes to establish a baseline
    for comparison. The tracker uses this to compute score deltas
    and identify effective parameter changes.

    Args:
        params: JSON string of current parameters
        scores: JSON string of current evaluation scores
        render_path: Path to current render image
        script_path: Path to current Blender script

    Returns:
        JSON confirmation with:
        - success: True if baseline recorded
        - message: Confirmation message
        - render_path: Path recorded

    Example:
        record_baseline(
            params='{"resolution": 96, "turbulence": 0.3}',
            scores='{"overall_score": 55, "passed": false}',
            render_path="build/vdb_output/explosion_v1/render_0030.png",
            script_path="assets/blender_scripts/generated/explosion_v1.py"
        )
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "record_baseline",
        {
            "params": params,
            "scores": scores,
            "render_path": render_path,
            "script_path": script_path,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def record_experiment_result(
    hypothesis: str,
    issue_addressed: str,
    result_params: str,
    result_scores: str,
    result_render: str,
    result_script: str,
    success: bool,
    observed_effects: str,
    learnings: str,
    warnings: str,
    human_notes: str = ""
) -> str:
    """
    Record the result of an experiment after making changes.

    Call this AFTER making parameter changes and re-evaluating to record:
    - What was tested and why
    - Whether it worked
    - What we learned (added to knowledge base)

    Args:
        hypothesis: What we were testing (e.g., "Increasing domain height will fix clipping")
        issue_addressed: The problem we tried to fix (e.g., "clipping at top edge")
        result_params: JSON string of parameters after change
        result_scores: JSON string of evaluation scores after change
        result_render: Path to result render image
        result_script: Path to result Blender script
        success: Did the experiment achieve its goal?
        observed_effects: JSON array of observed effects (e.g., '["smoke less dense", "fire brighter"]')
        learnings: JSON array of strings - what we learned
        warnings: JSON array of strings - gotchas discovered
        human_notes: Optional human observations

    Returns:
        JSON with:
        - experiment_id: Unique ID for this experiment
        - success: Whether it succeeded
        - partial_success: True if some improvement
        - score_changes: Score deltas from baseline
        - learnings: Learnings recorded
        - warnings: Warnings recorded
        - recommendations: What to try next

    Example:
        record_experiment_result(
            hypothesis="Increasing resolution from 96 to 128 will add detail",
            issue_addressed="lack of fine detail",
            result_params='{"resolution": 128, "turbulence": 0.3}',
            result_scores='{"overall_score": 62, "passed": true}',
            result_render="build/vdb_output/explosion_v2/render_0030.png",
            result_script="assets/blender_scripts/generated/explosion_v2.py",
            success=True,
            observed_effects='["more detail visible", "smoke has finer wisps"]',
            learnings='["resolution 128 provides good detail without excessive render time"]',
            warnings='["render time increased by 40%"]'
        )
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "record_experiment_result",
        {
            "hypothesis": hypothesis,
            "issue_addressed": issue_addressed,
            "result_params": result_params,
            "result_scores": result_scores,
            "result_render": result_render,
            "result_script": result_script,
            "success": success,
            "observed_effects": observed_effects,
            "learnings": learnings,
            "warnings": warnings,
            "human_notes": human_notes,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

@function_tool
async def get_warnings_before_change(
    parameter: str,
    change_type: str
) -> str:
    """
    Get warnings from knowledge base before making a parameter change.

    ALWAYS call this BEFORE modifying parameters to check for:
    - Known failure modes
    - Side effects
    - Required companion changes

    Args:
        parameter: Parameter you're planning to change (e.g., "domain_scale")
        change_type: Type of change ("increase", "decrease", "modify")

    Returns:
        JSON with:
        - parameter: The parameter checked
        - change_type: The change type
        - warnings: List of relevant warnings
        - parameter_info: Accumulated knowledge about this parameter
        - recommendation: "Proceed with caution" or "No known issues"

    Example:
        get_warnings_before_change("domain_scale", "increase")
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "get_warnings_before_change",
        {
            "parameter": parameter,
            "change_type": change_type,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def suggest_experiments(
    issue: str,
    current_params: str = "{}",
    current_scores: str = "{}"
) -> str:
    """
    Get experiment suggestions for addressing an issue.

    Uses accumulated knowledge to suggest informed experiments,
    ranked by likelihood of success based on past outcomes.

    Args:
        issue: Description of the issue (e.g., "clipping at top edge", "lacks smoke")
        current_params: JSON string of current parameters
        current_scores: JSON string of current scores

    Returns:
        JSON with:
        - issue: The issue being addressed
        - suggestions_count: Number of suggestions
        - suggestions: Ranked list of experiments, each with:
            - hypothesis: What to test
            - parameter_changes: Parameters to modify
            - expected_effects: What should happen
            - risks: Potential problems
            - confidence: Likelihood of success (0-100%)

    Example:
        suggest_experiments(
            issue="smoke is too thin",
            current_params='{"density": 1.0, "turbulence": 0.3}',
            current_scores='{"overall_score": 48}'
        )
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "suggest_experiments",
        {
            "issue": issue,
            "current_params": current_params,
            "current_scores": current_scores,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def query_knowledge_base(
    query: str
) -> str:
    """
    Search the knowledge base for relevant information.

    Searches across:
    - Parameter rules and guidelines
    - Past experiment outcomes
    - Causal relationships
    - Warnings and gotchas

    Args:
        query: Search query (e.g., "domain_scale", "clipping", "smoke")

    Returns:
        JSON with matching knowledge entries

    Example:
        query_knowledge_base("turbulence")
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "query_knowledge_base",
        {
            "query": query,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def get_parameter_knowledge(
    parameter: str
) -> str:
    """
    Get all accumulated knowledge about a specific parameter.

    Returns comprehensive information:
    - Effective value ranges
    - Success/failure statistics
    - Related parameters
    - Known side effects

    Args:
        parameter: Parameter name (e.g., "domain_scale", "flame_smoke", "resolution")

    Returns:
        JSON with rules, warnings, and statistics for this parameter

    Example:
        get_parameter_knowledge("turbulence")
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "get_parameter_knowledge",
        {
            "parameter": parameter,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def add_manual_learning(
    parameter: str,
    rule: str,
    warning: str = "",
    context: str = ""
) -> str:
    """
    Manually add a learning to the knowledge base.

    Use this when you observe something that should be remembered
    but wasn't automatically captured by experiment recording.

    Args:
        parameter: Parameter this learning applies to
        rule: The rule or guideline (e.g., "Always adjust position when scaling domain")
        warning: Optional warning message
        context: Optional context for when this applies

    Returns:
        JSON confirmation with:
        - success: True if learning added
        - parameter: The parameter
        - rule_added: The rule recorded
        - warning_added: The warning recorded (if any)

    Example:
        add_manual_learning(
            parameter="domain_scale",
            rule="When increasing domain_scale, also increase domain_height by same ratio",
            warning="Failure to adjust height causes clipping at top",
            context="pyro effects with rising smoke"
        )
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "add_manual_learning",
        {
            "parameter": parameter,
            "rule": rule,
            "warning": warning,
            "context": context,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# STATISTICS
# =============================================================================

@function_tool
async def get_experiment_statistics() -> str:
    """
    Get overall experiment tracking statistics.

    Provides aggregate metrics across all sessions:
    - Total experiments and sessions
    - Success rates by effect type
    - Most effective parameter changes
    - Common failure modes

    Returns:
        JSON with:
        - statistics: Raw statistics object
        - summary: Human-readable summary string

    Example:
        get_experiment_statistics()
    """
    server = await get_mcp_server("experiment-tracker")

    result = await server.call_tool(
        "get_experiment_statistics",
        {}
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)
