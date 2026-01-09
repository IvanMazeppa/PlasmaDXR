"""
Function tools for the Blender VFX Orchestrator.

Exposes MCP server capabilities as @function_tool decorated functions
that can be called by OpenAI Agents SDK agents.

Tool Categories:
- script_generator: Script generation and technique selection
- blender_executor: Blender script execution
- asset_evaluator: Quality evaluation (v2 consolidated API)
- experiment_tracker: Experiment tracking and learning
"""

from .script_generator_tools import (
    generate_script,
    modify_script,
    validate_script,
    list_techniques,
    recommend_technique,
    record_technique_outcome,
)

from .blender_executor_tools import (
    execute_blender_script,
    parse_blender_errors,
    list_run_outputs,
    get_latest_run,
)

from .asset_evaluator_tools import (
    evaluate_render,
    compare_renders,
    diagnose_issues,
    get_reference_stats,
    list_renders,
    analyze_temporal_quality,
)

from .experiment_tracker_tools import (
    start_experiment_session,
    end_experiment_session,
    get_session_report,
    record_baseline,
    record_experiment_result,
    get_warnings_before_change,
    suggest_experiments,
    query_knowledge_base,
    get_parameter_knowledge,
    add_manual_learning,
    get_experiment_statistics,
)

__all__ = [
    # Script Generator
    "generate_script",
    "modify_script",
    "validate_script",
    "list_techniques",
    "recommend_technique",
    "record_technique_outcome",
    # Blender Executor
    "execute_blender_script",
    "parse_blender_errors",
    "list_run_outputs",
    "get_latest_run",
    # Asset Evaluator
    "evaluate_render",
    "compare_renders",
    "diagnose_issues",
    "get_reference_stats",
    "list_renders",
    "analyze_temporal_quality",
    # Experiment Tracker
    "start_experiment_session",
    "end_experiment_session",
    "get_session_report",
    "record_baseline",
    "record_experiment_result",
    "get_warnings_before_change",
    "suggest_experiments",
    "query_knowledge_base",
    "get_parameter_knowledge",
    "add_manual_learning",
    "get_experiment_statistics",
]
