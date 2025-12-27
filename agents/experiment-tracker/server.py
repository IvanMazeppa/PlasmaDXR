#!/usr/bin/env python3
"""
Experiment Tracker MCP Server

Provides tools for tracking experiments, learning from results,
and querying accumulated knowledge.
"""

import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from database import ExperimentDatabase, get_db
from tracker import ExperimentTracker, get_tracker

# Load environment
load_dotenv()

# Initialize MCP server using FastMCP (like blender-executor)
mcp = FastMCP("experiment-tracker")

# Global tracker instance
_tracker: Optional[ExperimentTracker] = None


def get_tracker_instance() -> ExperimentTracker:
    """Get or create tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def start_experiment_session(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: str = "",
    semantic_query: str = ""
) -> str:
    """
    Start a new experiment session for tracking iterations on an asset.

    Args:
        asset_name: Name of the asset being created
        effect_type: Type of effect (explosion, fire, nebula, etc.)
        description: Description of what we're trying to create
        reference_path: Optional path to reference image
        semantic_query: Optional semantic description for evaluation

    Returns:
        JSON with session ID and status
    """
    tracker = get_tracker_instance()

    session_id = tracker.start_session(
        asset_name=asset_name,
        effect_type=effect_type,
        description=description,
        reference_path=reference_path or None,
        semantic_query=semantic_query or None
    )

    return json.dumps({
        'success': True,
        'session_id': session_id,
        'message': f'Started experiment session: {session_id}'
    }, indent=2)


@mcp.tool()
def record_baseline(
    params: str,
    scores: str,
    render_path: str,
    script_path: str
) -> str:
    """
    Record baseline state before running an experiment.

    Args:
        params: JSON string of current parameters
        scores: JSON string of current evaluation scores
        render_path: Path to current render image
        script_path: Path to current Blender script

    Returns:
        JSON confirmation
    """
    tracker = get_tracker_instance()

    tracker.record_baseline(
        params=json.loads(params),
        scores=json.loads(scores),
        render_path=render_path,
        script_path=script_path
    )

    return json.dumps({
        'success': True,
        'message': 'Baseline recorded',
        'render_path': render_path
    }, indent=2)


@mcp.tool()
def record_experiment_result(
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

    Args:
        hypothesis: What we were testing (e.g., "Increasing domain height will fix clipping")
        issue_addressed: The problem we tried to fix
        result_params: JSON string of parameters after change
        result_scores: JSON string of evaluation scores after change
        result_render: Path to result render image
        result_script: Path to result Blender script
        success: Did the experiment achieve its goal?
        observed_effects: JSON array of observed effects
        learnings: JSON array of strings - what we learned
        warnings: JSON array of strings - gotchas discovered
        human_notes: Optional human observations

    Returns:
        JSON with experiment result analysis
    """
    tracker = get_tracker_instance()

    result = tracker.record_experiment(
        hypothesis=hypothesis,
        issue_addressed=issue_addressed,
        result_params=json.loads(result_params),
        result_scores=json.loads(result_scores),
        result_render=result_render,
        result_script=result_script,
        success=success,
        observed_effects=json.loads(observed_effects),
        learnings=json.loads(learnings),
        warnings=json.loads(warnings),
        human_notes=human_notes
    )

    return json.dumps({
        'experiment_id': result.experiment_id,
        'success': result.success,
        'partial_success': result.partial_success,
        'score_changes': result.score_changes,
        'learnings': result.learnings,
        'warnings': result.warnings,
        'recommendations': result.recommendations
    }, indent=2)


@mcp.tool()
def get_warnings_before_change(
    parameter: str,
    change_type: str
) -> str:
    """
    Get warnings from knowledge base before making a parameter change.

    Call this BEFORE making changes to see what might go wrong.

    Args:
        parameter: Parameter you're planning to change (e.g., "domain_scale")
        change_type: Type of change (e.g., "increase", "decrease", "modify")

    Returns:
        JSON with warnings and relevant knowledge
    """
    tracker = get_tracker_instance()

    warnings = tracker.get_warnings_for_change(parameter, change_type)
    param_info = tracker.get_parameter_info(parameter)

    return json.dumps({
        'parameter': parameter,
        'change_type': change_type,
        'warnings': warnings,
        'parameter_info': param_info,
        'recommendation': (
            "Proceed with caution" if warnings
            else "No known issues with this change"
        )
    }, indent=2)


@mcp.tool()
def suggest_experiments(
    issue: str,
    current_params: str = "{}",
    current_scores: str = "{}"
) -> str:
    """
    Get experiment suggestions for addressing an issue.

    Uses knowledge base to suggest informed experiments.

    Args:
        issue: Description of the issue (e.g., "clipping at top edge", "lacks smoke")
        current_params: JSON string of current parameters
        current_scores: JSON string of current scores

    Returns:
        JSON with ranked experiment suggestions
    """
    tracker = get_tracker_instance()

    suggestions = tracker.suggest_experiments(
        issue=issue,
        current_params=json.loads(current_params),
        current_scores=json.loads(current_scores)
    )

    return json.dumps({
        'issue': issue,
        'suggestions_count': len(suggestions),
        'suggestions': [
            {
                'hypothesis': s.hypothesis,
                'parameter_changes': s.parameter_changes,
                'expected_effects': s.expected_effects,
                'risks': s.risks,
                'confidence': f"{s.confidence:.0%}"
            }
            for s in suggestions
        ]
    }, indent=2)


@mcp.tool()
def query_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        query: Search query (e.g., "domain_scale", "clipping", "smoke")

    Returns:
        JSON with matching knowledge entries
    """
    tracker = get_tracker_instance()

    result = tracker.query_knowledge(query)

    return json.dumps(result, indent=2)


@mcp.tool()
def get_parameter_knowledge(parameter: str) -> str:
    """
    Get all accumulated knowledge about a specific parameter.

    Args:
        parameter: Parameter name (e.g., "domain_scale", "flame_smoke")

    Returns:
        JSON with rules, warnings, statistics for this parameter
    """
    tracker = get_tracker_instance()

    info = tracker.get_parameter_info(parameter)

    return json.dumps(info, indent=2)


@mcp.tool()
def add_manual_learning(
    parameter: str,
    rule: str,
    warning: str = "",
    context: str = ""
) -> str:
    """
    Manually add a learning to the knowledge base.

    Use this when you observe something that should be remembered.

    Args:
        parameter: Parameter this learning applies to
        rule: The rule or guideline (e.g., "Always adjust position when scaling domain")
        warning: Optional warning message
        context: Optional context for when this applies

    Returns:
        JSON confirmation
    """
    tracker = get_tracker_instance()

    tracker.add_manual_learning(
        parameter=parameter,
        rule=rule,
        warning=warning or None,
        context=context
    )

    return json.dumps({
        'success': True,
        'parameter': parameter,
        'rule_added': rule,
        'warning_added': warning or None,
        'message': f'Learning added for {parameter}'
    }, indent=2)


@mcp.tool()
def add_human_feedback(
    experiment_id: str,
    rating: int,
    notes: str
) -> str:
    """
    Add human feedback to a recorded experiment.

    Args:
        experiment_id: ID of the experiment
        rating: 1-5 rating (1=failed, 5=excellent)
        notes: Human observations and feedback

    Returns:
        JSON confirmation
    """
    tracker = get_tracker_instance()

    tracker.add_human_feedback(experiment_id, rating, notes)

    return json.dumps({
        'success': True,
        'experiment_id': experiment_id,
        'rating': rating,
        'message': 'Feedback recorded'
    }, indent=2)


@mcp.tool()
def get_experiment_statistics() -> str:
    """
    Get overall experiment tracking statistics.

    Returns:
        JSON with counts, success rates, and summary
    """
    tracker = get_tracker_instance()

    stats = tracker.get_statistics()

    return json.dumps({
        'statistics': stats,
        'summary': (
            f"Tracked {stats['total_experiments']} experiments across "
            f"{stats['total_sessions']} sessions. "
            f"Success rate: {stats['overall_success_rate']:.0%}. "
            f"Knowledge base: {stats['parameters_tracked']} parameters, "
            f"{stats['causal_relationships']} causal relationships."
        )
    }, indent=2)


@mcp.tool()
def get_session_report(session_id: str = "") -> str:
    """
    Generate a report for an experiment session.

    Args:
        session_id: Session ID (leave empty for current session)

    Returns:
        JSON with session summary and experiment history
    """
    tracker = get_tracker_instance()

    report = tracker.get_session_report(session_id or None)

    return json.dumps(report, indent=2)


@mcp.tool()
def end_experiment_session(
    final_status: str,
    best_score: float
) -> str:
    """
    End the current experiment session.

    Args:
        final_status: Final status (e.g., "completed", "abandoned", "max_iterations")
        best_score: Best score achieved

    Returns:
        JSON with session summary
    """
    tracker = get_tracker_instance()

    # Get report before ending
    report = tracker.get_session_report()

    tracker.end_session(final_status, best_score)

    return json.dumps({
        'success': True,
        'final_status': final_status,
        'best_score': best_score,
        'session_report': report
    }, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
