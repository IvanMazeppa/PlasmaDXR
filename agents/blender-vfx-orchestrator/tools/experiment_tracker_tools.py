"""
Function tool wrappers for experiment tracking.

IN-PROCESS IMPLEMENTATION - Does NOT spawn MCP subprocesses.

Exposes experiment-tracker capabilities to OpenAI Agents SDK agents:
- Session management (start, end, report)
- Experiment recording (baseline, result)
- Knowledge base queries and learning
- Fix suggestions based on accumulated knowledge

This uses the two-layer pattern:
- _impl functions: Plain functions with actual logic (for internal use)
- @function_tool wrappers: Exposed to agents, call the _impl functions

IMPORTANT: This imports directly from the experiment-tracker modules rather than
spawning MCP subprocesses, avoiding anyio TaskGroup conflicts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from agents import function_tool


# =============================================================================
# PATH SETUP - Add experiment-tracker to Python path
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ORCHESTRATOR_ROOT.parent.parent  # agents -> PlasmaDXR

# Add experiment-tracker to path for imports
EXPERIMENT_TRACKER_DIR = PROJECT_ROOT / "agents/experiment-tracker"
if str(EXPERIMENT_TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_TRACKER_DIR))


# =============================================================================
# LAZY IMPORTS - Import from experiment-tracker modules
# =============================================================================

_tracker = None
_tracker_instance = None


def _get_tracker_module():
    """Lazy import tracker module."""
    global _tracker
    if _tracker is None:
        try:
            import tracker
            _tracker = tracker
        except ImportError as e:
            raise ImportError(f"Could not import tracker: {e}")
    return _tracker


def _get_tracker_instance():
    """Get or create a shared ExperimentTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        tracker_mod = _get_tracker_module()
        _tracker_instance = tracker_mod.ExperimentTracker()
    return _tracker_instance


# =============================================================================
# INTERNAL IMPLEMENTATION FUNCTIONS
# =============================================================================

def _start_experiment_session_impl(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: str = "",
    semantic_query: str = ""
) -> str:
    """
    Start a new experiment session.

    Args:
        asset_name: Name of the asset being created
        effect_type: Type of effect (explosion, fire, nebula, sun, etc.)
        description: Description of what we're trying to create
        reference_path: Optional path to reference image
        semantic_query: Optional semantic description for evaluation

    Returns:
        JSON with session_id and confirmation
    """
    try:
        tracker = _get_tracker_instance()

        session_id = tracker.start_session(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            semantic_query=semantic_query
        )

        return json.dumps({
            "success": True,
            "session_id": session_id,
            "message": f"Session started for {asset_name}",
            "asset_name": asset_name,
            "effect_type": effect_type
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "session_id": ""
        })


def _end_experiment_session_impl(
    final_status: str,
    best_score: float
) -> str:
    """
    End the current experiment session.

    Args:
        final_status: Final status ("completed", "abandoned", "max_iterations")
        best_score: Best quality score achieved (0-100)

    Returns:
        JSON with session summary
    """
    try:
        tracker = _get_tracker_instance()

        report = tracker.end_session(
            final_status=final_status,
            best_score=best_score
        )

        return json.dumps({
            "success": True,
            "final_status": final_status,
            "best_score": best_score,
            "session_report": report if isinstance(report, dict) else {"summary": str(report)}
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "final_status": final_status,
            "best_score": best_score
        })


def _get_session_report_impl(session_id: str = "") -> str:
    """
    Get a report for an experiment session.

    Args:
        session_id: Session ID (leave empty for current session)

    Returns:
        JSON with session summary
    """
    try:
        tracker = _get_tracker_instance()

        report = tracker.get_session_report(session_id=session_id if session_id else None)

        return json.dumps(
            report if isinstance(report, dict) else {"report": str(report)},
            indent=2,
            default=str
        )

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "session_id": session_id
        })


def _record_baseline_impl(
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
    try:
        tracker = _get_tracker_instance()

        params_dict = json.loads(params) if isinstance(params, str) else params
        scores_dict = json.loads(scores) if isinstance(scores, str) else scores

        tracker.record_baseline(
            params=params_dict,
            scores=scores_dict,
            render_path=render_path,
            script_path=script_path
        )

        return json.dumps({
            "success": True,
            "message": "Baseline recorded",
            "render_path": render_path,
            "params_count": len(params_dict),
            "scores_count": len(scores_dict)
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def _record_experiment_result_impl(
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
    Record the result of an experiment.

    Args:
        hypothesis: What we were testing
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
        JSON with experiment result
    """
    try:
        tracker = _get_tracker_instance()

        # Parse JSON strings
        params_dict = json.loads(result_params) if isinstance(result_params, str) else result_params
        scores_dict = json.loads(result_scores) if isinstance(result_scores, str) else result_scores
        effects_list = json.loads(observed_effects) if isinstance(observed_effects, str) else observed_effects
        learnings_list = json.loads(learnings) if isinstance(learnings, str) else learnings
        warnings_list = json.loads(warnings) if isinstance(warnings, str) else warnings

        result = tracker.record_experiment(
            hypothesis=hypothesis,
            issue_addressed=issue_addressed,
            result_params=params_dict,
            result_scores=scores_dict,
            result_render=result_render,
            result_script=result_script,
            success=success,
            observed_effects=effects_list,
            learnings=learnings_list,
            warnings=warnings_list,
            human_notes=human_notes
        )

        # Handle various return types
        if hasattr(result, '__dict__'):
            result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"result": str(result)}

        return json.dumps({
            "experiment_id": result_dict.get("experiment_id", ""),
            "success": result_dict.get("success", success),
            "partial_success": result_dict.get("partial_success", False),
            "score_changes": result_dict.get("score_changes", {}),
            "learnings": learnings_list,
            "warnings": warnings_list,
            "recommendations": result_dict.get("recommendations", [])
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "experiment_id": "",
            "success": False,
            "error": str(e),
            "learnings": [],
            "warnings": [],
            "recommendations": []
        })


def _get_warnings_before_change_impl(
    parameter: str,
    change_type: str
) -> str:
    """
    Get warnings from knowledge base before making a parameter change.

    Args:
        parameter: Parameter you're planning to change
        change_type: Type of change ("increase", "decrease", "modify")

    Returns:
        JSON with warnings and recommendations
    """
    try:
        tracker = _get_tracker_instance()

        warnings = tracker.get_warnings_for_change(
            parameter=parameter,
            change_type=change_type
        )

        # Handle various return types
        if isinstance(warnings, list):
            return json.dumps({
                "parameter": parameter,
                "change_type": change_type,
                "warnings": warnings,
                "parameter_info": {},
                "recommendation": "Proceed with caution" if warnings else "No known issues"
            }, indent=2)
        elif isinstance(warnings, dict):
            return json.dumps(warnings, indent=2, default=str)
        else:
            return json.dumps({
                "parameter": parameter,
                "change_type": change_type,
                "warnings": [str(warnings)] if warnings else [],
                "recommendation": "Proceed with caution" if warnings else "No known issues"
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "parameter": parameter,
            "change_type": change_type,
            "warnings": [],
            "error": str(e),
            "recommendation": f"Error getting warnings: {e}"
        })


def _suggest_experiments_impl(
    issue: str,
    current_params: str = "{}",
    current_scores: str = "{}"
) -> str:
    """
    Get experiment suggestions for addressing an issue.

    Args:
        issue: Description of the issue
        current_params: JSON string of current parameters
        current_scores: JSON string of current scores

    Returns:
        JSON with ranked experiment suggestions
    """
    try:
        tracker = _get_tracker_instance()

        params_dict = json.loads(current_params) if isinstance(current_params, str) else current_params
        scores_dict = json.loads(current_scores) if isinstance(current_scores, str) else current_scores

        suggestions = tracker.suggest_experiments(
            issue=issue,
            current_params=params_dict,
            current_scores=scores_dict
        )

        # Handle various return types
        if isinstance(suggestions, list):
            return json.dumps({
                "issue": issue,
                "suggestions_count": len(suggestions),
                "suggestions": suggestions
            }, indent=2, default=str)
        elif isinstance(suggestions, dict):
            return json.dumps(suggestions, indent=2, default=str)
        else:
            return json.dumps({
                "issue": issue,
                "suggestions_count": 0,
                "suggestions": [str(suggestions)] if suggestions else [],
                "message": "No specific suggestions available"
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "issue": issue,
            "suggestions_count": 0,
            "suggestions": [],
            "error": str(e)
        })


def _query_knowledge_base_impl(query: str) -> str:
    """
    Search the knowledge base.

    Args:
        query: Search query

    Returns:
        JSON with matching knowledge entries
    """
    try:
        tracker = _get_tracker_instance()

        results = tracker.query_knowledge(query=query)

        if isinstance(results, list):
            return json.dumps({
                "query": query,
                "results_count": len(results),
                "results": results
            }, indent=2, default=str)
        elif isinstance(results, dict):
            return json.dumps(results, indent=2, default=str)
        else:
            return json.dumps({
                "query": query,
                "results_count": 1 if results else 0,
                "results": [str(results)] if results else []
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "query": query,
            "results_count": 0,
            "results": [],
            "error": str(e)
        })


def _get_parameter_knowledge_impl(parameter: str) -> str:
    """
    Get all accumulated knowledge about a specific parameter.

    Args:
        parameter: Parameter name

    Returns:
        JSON with rules, warnings, and statistics
    """
    try:
        tracker = _get_tracker_instance()

        knowledge = tracker.get_parameter_knowledge(parameter=parameter)

        if isinstance(knowledge, dict):
            return json.dumps(knowledge, indent=2, default=str)
        else:
            return json.dumps({
                "parameter": parameter,
                "knowledge": str(knowledge) if knowledge else "No knowledge available",
                "rules": [],
                "warnings": []
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "parameter": parameter,
            "error": str(e),
            "rules": [],
            "warnings": []
        })


def _add_manual_learning_impl(
    parameter: str,
    rule: str,
    warning: str = "",
    context: str = ""
) -> str:
    """
    Manually add a learning to the knowledge base.

    Args:
        parameter: Parameter this learning applies to
        rule: The rule or guideline
        warning: Optional warning message
        context: Optional context for when this applies

    Returns:
        JSON confirmation
    """
    try:
        tracker = _get_tracker_instance()

        tracker.add_manual_learning(
            parameter=parameter,
            rule=rule,
            warning=warning if warning else None,
            context=context if context else None
        )

        return json.dumps({
            "success": True,
            "parameter": parameter,
            "rule_added": rule,
            "warning_added": warning if warning else None,
            "context": context if context else None
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "parameter": parameter
        })


def _get_experiment_statistics_impl() -> str:
    """
    Get overall experiment tracking statistics.

    Returns:
        JSON with aggregate metrics
    """
    try:
        tracker = _get_tracker_instance()

        stats = tracker.get_statistics()

        if isinstance(stats, dict):
            return json.dumps({
                "statistics": stats,
                "summary": stats.get("summary", "Statistics retrieved successfully")
            }, indent=2, default=str)
        else:
            return json.dumps({
                "statistics": str(stats) if stats else {},
                "summary": "Statistics retrieved"
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "statistics": {},
            "error": str(e),
            "summary": f"Error getting statistics: {e}"
        })


# In-memory modification strategy tracking (persists within session)
# Tracks which modification patterns actually improve scores
_modification_strategy_stats: Dict[str, Dict[str, Any]] = {}


def _report_modification_outcome_impl(
    issue: str,
    modification_pattern: str,
    pattern_type: str,
    score_before: float,
    score_after: float,
    visual_change_observed: bool,
    effect_type: str = ""
) -> str:
    """
    Report whether a modification attempt actually worked.

    Builds knowledge about:
    - Which modification patterns (config, settings, shader_node) work for which issues
    - Which parameters have real visual impact
    - Which suggestions are "no-ops" that don't reach the render

    Args:
        issue: The issue being addressed (e.g., "too dark", "no fire visible")
        modification_pattern: The pattern used (e.g., "volume.inputs['Density'].default_value")
        pattern_type: Category (config_class, settings_attr, shader_node, math_node)
        score_before: Quality score before modification
        score_after: Quality score after modification
        visual_change_observed: Did the render look different?
        effect_type: Effect type for context

    Returns:
        JSON with updated strategy effectiveness statistics
    """
    global _modification_strategy_stats

    try:
        score_delta = score_after - score_before
        success = score_delta > 0 and visual_change_observed

        # Create strategy key
        strategy_key = f"{issue}|{pattern_type}"
        if strategy_key not in _modification_strategy_stats:
            _modification_strategy_stats[strategy_key] = {
                "issue": issue,
                "pattern_type": pattern_type,
                "total_attempts": 0,
                "successful": 0,
                "total_score_delta": 0.0,
                "patterns_tried": [],
                "effect_types": set()
            }

        stats = _modification_strategy_stats[strategy_key]
        stats["total_attempts"] += 1
        if success:
            stats["successful"] += 1
        stats["total_score_delta"] += score_delta
        if modification_pattern not in stats["patterns_tried"]:
            stats["patterns_tried"].append(modification_pattern)
        if effect_type:
            stats["effect_types"].add(effect_type)

        # Calculate success rate
        success_rate = stats["successful"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0.0
        avg_delta = stats["total_score_delta"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0.0

        # Also add to knowledge base if pattern is effective
        if success_rate >= 0.7 and stats["total_attempts"] >= 2:
            try:
                tracker = _get_tracker_instance()
                tracker.add_manual_learning(
                    parameter=f"modification_strategy:{issue}",
                    rule=f"For '{issue}' issues, use {pattern_type} modifications (success rate: {success_rate:.0%})",
                    context=f"Effect types: {', '.join(stats['effect_types'])}"
                )
            except Exception:
                pass  # Non-critical

        return json.dumps({
            "success": True,
            "this_attempt": {
                "success": success,
                "score_delta": score_delta,
                "visual_change": visual_change_observed
            },
            "strategy_stats": {
                "issue": issue,
                "pattern_type": pattern_type,
                "success_rate": success_rate,
                "average_score_delta": avg_delta,
                "total_attempts": stats["total_attempts"],
                "recommendation": "USE THIS PATTERN" if success_rate >= 0.5 else "TRY DIFFERENT PATTERN"
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def _get_effective_strategy_impl(issue: str) -> str:
    """
    Get the most effective modification strategy for an issue.

    Based on accumulated outcome tracking data.

    Args:
        issue: The issue to address

    Returns:
        JSON with best strategy and confidence
    """
    global _modification_strategy_stats

    # Find strategies for this issue
    matching = []
    for key, stats in _modification_strategy_stats.items():
        if issue.lower() in stats["issue"].lower():
            success_rate = stats["successful"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0.0
            matching.append({
                "pattern_type": stats["pattern_type"],
                "success_rate": success_rate,
                "attempts": stats["total_attempts"],
                "avg_score_delta": stats["total_score_delta"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0.0,
                "patterns_tried": stats["patterns_tried"]
            })

    # Sort by success rate
    matching.sort(key=lambda x: x["success_rate"], reverse=True)

    if matching:
        best = matching[0]
        return json.dumps({
            "issue": issue,
            "best_strategy": best["pattern_type"],
            "confidence": best["success_rate"],
            "recommendation": f"Use {best['pattern_type']} modifications. Patterns that worked: {best['patterns_tried'][:3]}",
            "all_strategies": matching
        }, indent=2)
    else:
        return json.dumps({
            "issue": issue,
            "best_strategy": "unknown",
            "confidence": 0.0,
            "recommendation": "No data yet. Try shader_node modifications for visual issues.",
            "all_strategies": []
        }, indent=2)


# =============================================================================
# FUNCTION TOOL WRAPPERS (exposed to agents)
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
        JSON with session_id and confirmation
    """
    return _start_experiment_session_impl(
        asset_name=asset_name,
        effect_type=effect_type,
        description=description,
        reference_path=reference_path,
        semantic_query=semantic_query
    )


@function_tool
async def end_experiment_session(
    final_status: str,
    best_score: float
) -> str:
    """
    End the current experiment session and generate final report.

    Args:
        final_status: Final status ("completed", "abandoned", "max_iterations")
        best_score: Best quality score achieved (0-100)

    Returns:
        JSON with session summary
    """
    return _end_experiment_session_impl(
        final_status=final_status,
        best_score=best_score
    )


@function_tool
async def get_session_report(
    session_id: str = ""
) -> str:
    """
    Generate a report for an experiment session.

    Args:
        session_id: Session ID (leave empty for current session)

    Returns:
        JSON with session summary and experiment history
    """
    return _get_session_report_impl(session_id)


@function_tool
async def record_baseline(
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
    return _record_baseline_impl(
        params=params,
        scores=scores,
        render_path=render_path,
        script_path=script_path
    )


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

    Args:
        hypothesis: What we were testing
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
        JSON with experiment result
    """
    return _record_experiment_result_impl(
        hypothesis=hypothesis,
        issue_addressed=issue_addressed,
        result_params=result_params,
        result_scores=result_scores,
        result_render=result_render,
        result_script=result_script,
        success=success,
        observed_effects=observed_effects,
        learnings=learnings,
        warnings=warnings,
        human_notes=human_notes
    )


@function_tool
async def get_warnings_before_change(
    parameter: str,
    change_type: str
) -> str:
    """
    Get warnings from knowledge base before making a parameter change.

    Args:
        parameter: Parameter you're planning to change
        change_type: Type of change ("increase", "decrease", "modify")

    Returns:
        JSON with warnings and recommendations
    """
    return _get_warnings_before_change_impl(parameter, change_type)


@function_tool
async def suggest_experiments(
    issue: str,
    current_params: str = "{}",
    current_scores: str = "{}"
) -> str:
    """
    Get experiment suggestions for addressing an issue.

    Args:
        issue: Description of the issue
        current_params: JSON string of current parameters
        current_scores: JSON string of current scores

    Returns:
        JSON with ranked experiment suggestions
    """
    return _suggest_experiments_impl(issue, current_params, current_scores)


@function_tool
async def query_knowledge_base(
    query: str
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        query: Search query

    Returns:
        JSON with matching knowledge entries
    """
    return _query_knowledge_base_impl(query)


@function_tool
async def get_parameter_knowledge(
    parameter: str
) -> str:
    """
    Get all accumulated knowledge about a specific parameter.

    Args:
        parameter: Parameter name

    Returns:
        JSON with rules, warnings, and statistics
    """
    return _get_parameter_knowledge_impl(parameter)


@function_tool
async def add_manual_learning(
    parameter: str,
    rule: str,
    warning: str = "",
    context: str = ""
) -> str:
    """
    Manually add a learning to the knowledge base.

    Args:
        parameter: Parameter this learning applies to
        rule: The rule or guideline
        warning: Optional warning message
        context: Optional context for when this applies

    Returns:
        JSON confirmation
    """
    return _add_manual_learning_impl(parameter, rule, warning, context)


@function_tool
async def get_experiment_statistics() -> str:
    """
    Get overall experiment tracking statistics.

    Returns:
        JSON with aggregate metrics
    """
    return _get_experiment_statistics_impl()


@function_tool
async def report_modification_outcome(
    issue: str,
    modification_pattern: str,
    pattern_type: str,
    score_before: float,
    score_after: float,
    visual_change_observed: bool,
    effect_type: str = ""
) -> str:
    """
    Report whether a modification attempt actually worked.

    Call this AFTER each iteration to track which modification strategies
    are effective. This builds knowledge about:
    - Which patterns (shader_node, config_class, etc.) work for which issues
    - Which parameters have real visual impact
    - Which suggestions are "no-ops"

    Args:
        issue: The issue being addressed (e.g., "too dark", "no fire visible")
        modification_pattern: The exact pattern used (e.g., "volume.inputs['Density'].default_value")
        pattern_type: Category: config_class, settings_attr, shader_node, math_node
        score_before: Quality score before modification
        score_after: Quality score after modification
        visual_change_observed: Did the render look different?
        effect_type: Effect type for context

    Returns:
        JSON with strategy effectiveness statistics
    """
    return _report_modification_outcome_impl(
        issue, modification_pattern, pattern_type,
        score_before, score_after, visual_change_observed, effect_type
    )


def _seed_technique_entry_impl(
    effect_type: str,
    technique_name: str,
    description: str,
    source: str = "manual_seed",
    parameters: str = "{}"
) -> str:
    """
    Seed a technique entry into the knowledge base.

    Creates a KB entry with emerging trust level for bootstrap seeding.

    Args:
        effect_type: Effect type this technique applies to (fire, smoke, etc.)
        technique_name: Short name of the technique
        description: What this technique does and when to use it
        source: Source identifier (default "manual_seed")
        parameters: Optional JSON string of relevant parameters

    Returns:
        JSON confirmation with entry details
    """
    try:
        tracker = _get_tracker_instance()

        dated_source = f"{source}_{datetime.now().strftime('%Y-%m-%d')}"

        tracker.add_manual_learning(
            parameter=f"technique:{effect_type}:{technique_name}",
            rule=description,
            warning="",
            context=json.dumps({
                "effect_type": effect_type,
                "technique_name": technique_name,
                "trust_level": "emerging",
                "source": dated_source,
                "retention": 1.0,
                "parameters": json.loads(parameters) if isinstance(parameters, str) else parameters,
            })
        )

        return json.dumps({
            "success": True,
            "effect_type": effect_type,
            "technique_name": technique_name,
            "source": dated_source,
            "trust_level": "emerging",
            "message": f"Seeded technique '{technique_name}' for {effect_type}"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "effect_type": effect_type,
            "technique_name": technique_name
        })


@function_tool
async def seed_technique_entry(
    effect_type: str,
    technique_name: str,
    description: str,
    source: str = "manual_seed",
    parameters: str = "{}"
) -> str:
    """
    Seed a technique entry into the knowledge base for bootstrap discovery.

    Use this to populate the KB with known techniques so the system can
    discover and try them during production runs. Entries start with
    trust_level="emerging" and must earn trust through successful use.

    Args:
        effect_type: Effect type (fire, smoke, liquid, rigid_body, particles, cloth)
        technique_name: Short technique name (e.g., "mantaflow_gas_burning")
        description: What the technique does and when to use it
        source: Source identifier (default "manual_seed")
        parameters: Optional JSON string of relevant Blender parameters

    Returns:
        JSON confirmation with seeded entry details
    """
    return _seed_technique_entry_impl(
        effect_type=effect_type,
        technique_name=technique_name,
        description=description,
        source=source,
        parameters=parameters
    )


# =============================================================================
# Phase 2B-6: Ebbinghaus Memory Decay — Reinforcement
# =============================================================================

def _reinforce_entry_impl(entry_id: str, quality_score: float = 60.0) -> str:
    """
    Reinforce a KB entry — resets decay clock and increments reinforcement count.

    Called when a pattern/technique from the KB is used successfully.

    Args:
        entry_id: The 'parameter' key in parameter_knowledge
                  (e.g., 'technique:fire:mantaflow_gas')
        quality_score: Quality score achieved (0-100), updates confidence via EMA

    Returns:
        JSON confirmation with updated reinforcement state
    """
    try:
        tracker = _get_tracker_instance()
        db = tracker.db

        now_iso = datetime.now().isoformat()

        with db._connection() as conn:
            row = conn.execute(
                "SELECT confidence, reinforcement_count FROM parameter_knowledge WHERE parameter = ?",
                (entry_id,)
            ).fetchone()

            if not row:
                return json.dumps({
                    "success": False,
                    "error": f"Entry '{entry_id}' not found in knowledge base"
                })

            old_confidence = row['confidence'] if row['confidence'] else 50.0
            old_count = row['reinforcement_count'] if row['reinforcement_count'] else 0

            # EMA update for confidence
            new_confidence = old_confidence + 0.1 * (quality_score - old_confidence)
            new_count = old_count + 1

            conn.execute(
                """UPDATE parameter_knowledge
                   SET last_reinforced = ?, reinforcement_count = ?,
                       confidence = ?, last_updated = ?
                   WHERE parameter = ?""",
                (now_iso, new_count, new_confidence, now_iso, entry_id)
            )

        return json.dumps({
            "success": True,
            "entry_id": entry_id,
            "reinforcement_count": new_count,
            "confidence": round(new_confidence, 2),
            "message": f"Reinforced '{entry_id}': count={new_count}, confidence={new_confidence:.1f}"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "entry_id": entry_id
        })


@function_tool
async def reinforce_entry(
    entry_id: str,
    quality_score: float = 60.0,
) -> str:
    """
    Reinforce a knowledge base entry after successful use.

    Resets the Ebbinghaus decay clock and increments the reinforcement count.
    Call this when a pattern or technique from the KB is used successfully
    in a production run.

    Args:
        entry_id: The parameter key in knowledge base
                  (e.g., 'technique:fire:mantaflow_gas')
        quality_score: Quality score achieved (0-100)

    Returns:
        JSON with updated reinforcement state
    """
    return _reinforce_entry_impl(entry_id, quality_score)


@function_tool
async def get_effective_strategy(issue: str) -> str:
    """
    Get the most effective modification strategy for an issue.

    Returns the pattern type (shader_node, config_class, etc.) that has
    historically worked best for this type of issue.

    Args:
        issue: The issue to address (e.g., "too dark", "grey sphere")

    Returns:
        JSON with best strategy and confidence level
    """
    return _get_effective_strategy_impl(issue)
