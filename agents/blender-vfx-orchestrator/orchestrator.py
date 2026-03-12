"""
Blender VFX Orchestrator using OpenAI Agents SDK.

Main orchestrator that coordinates specialized agents to autonomously
generate and iterate on VFX assets until quality thresholds are met.

Key capabilities:
- Multi-agent coordination via code-based pipeline
- Quality gate enforcement
- Stuck detection and escape strategies
- Budget enforcement ($20/month limit)
- Session persistence across context limits
"""

from __future__ import annotations

import asyncio
import inspect
import time
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dotenv import load_dotenv
load_dotenv()

# pydantic models moved to models/pipeline_models.py

from agents import Agent, ModelSettings, Runner, trace, RunContextWrapper, ItemHelpers, SQLiteSession, function_tool
try:
    from agents import RunConfig
except Exception:
    RunConfig = None
from agents.exceptions import OutputGuardrailTripwireTriggered
from agents.memory import OpenAIResponsesCompactionSession
from agents.memory.session import SessionABC

# Enable verbose logging for debugging agent interactions
# Set AGENTS_DEBUG=1 to enable, or call enable_verbose_stdout_logging() directly
import logging
if os.environ.get("AGENTS_DEBUG", "").lower() in ("1", "true", "yes"):
    from agents import enable_verbose_stdout_logging
    enable_verbose_stdout_logging()
    print("[Orchestrator] Verbose SDK logging ENABLED", file=sys.stderr)

# Configure agents logger for custom filtering
_agents_logger = logging.getLogger("openai.agents")

# Session Manager for deterministic experiment state tracking
from session_manager import SessionManager

# Config system for preset-based agent settings
from config import AgentConfigManager, get_config

# ExperimentTracker baseline recording for Learning Agent compatibility
from tools.experiment_tracker_tools import _record_baseline_impl as record_experiment_baseline
# Proactive research for pre-iteration checks (direct callable, no tool wrapper)
from tools.proactive_research_tools import pre_iteration_research_direct as pre_iteration_research
# Pattern outcome reporting and pattern search (direct callable for Phase 2 self-learning reuse)
from tools.code_pattern_tools import _report_pattern_outcome_impl as report_pattern_outcome
from tools.code_pattern_tools import search_patterns_impl as search_patterns_direct
# Knowledge base query for doc-grounded learnings (direct callable for Phase 2)
from tools.experiment_tracker_tools import _query_knowledge_base_impl as query_knowledge_direct
from tools.experiment_tracker_tools import _suggest_experiments_impl as suggest_experiments_direct
# Phase 2B-3: Deterministic render quality checks (Tier 1, $0)
from tools.deterministic_quality_checks import run_deterministic_checks
# Phase 2B-5: HITL framework (pipeline-level checkpoints)
from utils.hitl_handler import HITLHandler, CheckpointDecision
# Phase 2B-8: Artifact read tools for agents
from tools.artifact_tools import read_artifact, list_session_artifacts
# Phase 2B-2: Per-agent context trimming ($0, deterministic)
from utils.context_filter import vfx_context_filter
from agents.agent_output import AgentOutputSchema

# Enforcement Hooks for loop detection and doc query requirements
from hooks import (
    EnforcementHooks,
    EnforcementConfig,
    LoopDetectedError,
    DocQueryRequiredError,
    TurnBudgetExceededError,
    CommunicationFlowTracker,
    DiagnosticHooks,
)
from hooks.enforcement_hooks import (
    create_research_hooks,
    create_script_writer_hooks,
    create_fallback_script_writer_hooks,
    create_error_recovery_hooks,
    create_quality_analyst_hooks,
    create_learning_agent_hooks,
)

# Phase 3: Input/Output Guardrails for agent validation
from guardrails import (
    # Research Agent guardrails
    validate_research_output,
    # Script Writer guardrails
    require_research_context,
    validate_effect_type,
    validate_script_output,
    # Quality Analyst guardrails
    check_budget_before_quality,
    validate_quality_output,
    # Coordinator guardrails
    validate_technique_decision,
    validate_modification_decision,
    validate_quality_decision,
)
# Phase 7 (DEPRECATED 2A-6): Spec-First guardrails removed — truth pack handles validation at $0
# Artifact Gates: Deterministic validation of execution outputs (bake/render gates)
from guardrails.artifact_gates import (
    validate_execution_artifacts,
    format_gate_failure_for_diagnosis,
)
from openai.types.shared import Reasoning


# =============================================================================
# PIPELINE OUTPUT MODELS (extracted to models/pipeline_models.py)
# =============================================================================
# Re-exported here for backward compatibility with existing imports.
from models.pipeline_models import (
    ResearchOutput,
    ScriptOutput,
    ExecutionOutput,
    QualityOutput,
    LearningOutput,
    TechniqueContract,
    TechniqueDecision,
    ModificationDecision,
    QualityDecision,
    compute_sica_utility,
)

from models.shared_context import (
    AssetRequest,
    SessionState,
    SessionStatus,
    SharedContext,
    IterationResult,
    EscapeLevel,
    ScriptModification,
    BlenderExecution,
    QualityMetrics,
)
# API Spec models for Spec-First Pipeline
from models.api_spec import APISpec, VerifiedScriptOutput

# Phase 1 Reliability: Truth Pack (deterministic API validation)
from tools.truth_pack import (
    build_truth_pack,
    validate_script_against_truth_pack,
    auto_fix_script,
    format_truth_pack_for_prompt,
    truth_pack_to_api_spec,
    TECHNIQUE_ALIASES,
)
from tools.truth_pack_validator import (
    validate_and_fix_script,
    set_truth_pack as set_global_truth_pack,
)
from tools.qa_diagnosis_bridge import create_code_grounded_feedback

# Phase 2A-3: Tool-level guardrails for truth pack enforcement
from guardrails.tool_guardrails import (
    attach_tool_guardrails,
    set_truth_pack as set_guardrail_truth_pack,
)

# Phase 2A-4: PipelineMonitor (deterministic monitoring layer)
from tools.pipeline_monitor import PipelineMonitor, AlertLevel, AlertType

# Proactive research tools for Strategy 3: Early warning detection
# NOTE: pre_iteration_research (direct callable) is imported at line 48 for pipeline use
# Import the @function_tool version for agent tool lists
from tools.proactive_research_tools import (
    pre_iteration_research as pre_iteration_research_tool,  # @function_tool version for agents
    evaluate_escape_velocity,
    search_alternative_approaches,
)

# Semantic docs tools for Strategy 1: Vector Store for Blender Documentation
from tools.semantic_docs_tools import (
    blender_doc_search_bundle,
    bundle_search_impl,  # Direct callable (not FunctionTool) for programmatic use
    find_alternative_approaches,
)

# Code pattern tools for Strategy 4: Code Pattern Memory
from tools.code_pattern_tools import (
    record_code_pattern,
    search_code_patterns,
    get_pattern_code,
    report_pattern_outcome,
    get_pattern_library_stats,
    list_patterns_by_effect,
)

# Knowledge distillation tools for Strategy 2: Pattern Extraction
from tools.knowledge_distillation_tools import (
    extract_successful_pattern,
    apply_pattern_to_script,
    analyze_script_for_patterns,
    compare_scripts,
)

# Script modification for direct parameter changes
from tools.script_generator_tools import _modify_script_impl

# Phase 2A-5: Parameter bounds and damped convergence
from tools.parameter_bounds import apply_bounds as apply_parameter_bounds

# Deterministic quality-to-parameter mapping (zero LLM fallback)
from utils.quality_parameter_map import (
    map_quality_issues_to_params,
    merge_suggestions_to_modifications,
    format_suggestions_for_prompt,
)

# Dynamic instruction wrappers for standalone agents (SDK dynamic instructions pattern)
from tools.dynamic_instructions import (
    dynamic_script_writer_standalone_instructions,
    dynamic_quality_analyst_standalone_instructions,
    dynamic_learning_agent_standalone_instructions,
)

from specialized_agents import (
    create_script_writer,
    create_quality_analyst,
    create_learning_agent,
)
# DocsExpert now uses in-process function_tools instead of MCP client connections.
# This fixes the anyio TaskGroup conflict that previously blocked MCP tool handlers.
from specialized_agents.docs_expert import create_docs_expert
# Phase 7 (DEPRECATED 2A-6): Spec-First Pipeline agents removed.
# Truth pack + tool guardrails handle API validation at $0.
# See specialized_agents/api_spec_agent.py for historical reference.
# API Validator for Blender 5.0 API validation (Phase 6 of Architecture Optimization)
# Validates API calls in generated scripts BEFORE execution to catch errors at source
from specialized_agents.api_validator import (
    validate_code_api,
    CodeValidationResult,
    KNOWN_API_CHANGES,
)
from utils import (
    BudgetTracker,
    get_budget_tracker,
    SessionPersistence,
    get_persistence,
    generate_session_id,
    create_session_from_request,
    resume_or_create_session,
    # Artifact-First Handoffs
    ArtifactManager,
    get_artifact_manager,
)

# =============================================================================
# SDK SESSION MANAGEMENT (Phase 4: Conversation Persistence)
# =============================================================================

# Directory for SDK session databases
SDK_SESSIONS_DIR = Path(__file__).parent / "sessions" / "sdk"


def get_or_create_sdk_session(session_id: str) -> OpenAIResponsesCompactionSession:
    """
    Get or create an SDK session with automatic history compaction.

    Uses OpenAIResponsesCompactionSession to wrap a SQLiteSession. This provides
    conversation persistence while preventing unbounded token growth — the SDK
    automatically summarizes old history via the Responses API `responses.compact`
    method.

    Auto-compaction triggers when conversation history exceeds ~25K tokens
    (~100K chars). Manual compaction also runs at end of each iteration as a
    safety net. Truth pack lives in SharedContext (Python-managed) and current
    script path / quality issues live in SessionState, so both survive compaction.

    Args:
        session_id: Unique identifier for the VFX session

    Returns:
        OpenAIResponsesCompactionSession for use with Runner.run()
    """
    SDK_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = SDK_SESSIONS_DIR / "vfx_conversations.db"

    underlying = SQLiteSession(session_id, str(db_path))
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=underlying,
        # Auto-compact when conversation history exceeds ~25K tokens (~100K chars)
        # Truth pack is in SharedContext (Python-managed), not conversation history,
        # so it survives compaction. Current script path and quality issues are in
        # SessionState, which also survives compaction.
        should_trigger_compaction=lambda history: sum(
            len(str(item)) for item in (history or [])
        ) > 100_000,
    )


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def visualize_agents(agent, filename: str = "agent_graph") -> Optional[str]:
    """
    Generate a graphical visualization of the agent architecture.

    Requires: pip install "openai-agents[viz]"

    Args:
        agent: The root agent to visualize
        filename: Output filename (without extension)

    Returns:
        Path to generated PNG file, or None if visualization unavailable
    """
    try:
        from agents.extensions.visualization import draw_graph
        graph = draw_graph(agent, filename=filename)
        print(f"[Viz] Agent graph saved to {filename}.png", file=sys.stderr)
        return f"{filename}.png"
    except ImportError:
        print("[Viz] Visualization not available. Run: pip install 'openai-agents[viz]'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Viz] Error generating graph: {e}", file=sys.stderr)
        return None


# =============================================================================
# COORDINATOR AGENTS (extracted to coordinator_agents.py)
# =============================================================================
from coordinator_agents import (
    COORDINATOR_INSTRUCTIONS,
    create_research_tool_wrapper,
    create_technique_selection_coordinator,
    create_modification_coordinator,
    create_quality_gate_coordinator,
)


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class BlenderVFXOrchestrator:
    """
    Main orchestrator for autonomous VFX asset generation.

    Coordinates 5 specialized agents through DETERMINISTIC CODE-BASED
    orchestration. Python controls the pipeline sequence explicitly,
    ensuring reliable execution without relying on LLM instruction-following.

    Pipeline: Research → Script → Execute → Evaluate → Learn → (loop or complete)
    """

    def __init__(self):
        """Initialize the orchestrator (call initialize() before use)."""
        # Standalone agents for code-based orchestration (no handoffs)
        self._research_agent: Optional[Agent] = None
        self._script_agent_standalone: Optional[Agent] = None
        # Phase 2A-7: Executor agent removed — execution is deterministic via
        # _execute_blender_script_impl() (see PHASE 2: EXECUTION in pipeline).
        self._quality_agent_standalone: Optional[Agent] = None
        self._learning_agent_standalone: Optional[Agent] = None
        self._docs_expert_standalone: Optional[Agent] = None  # For parallel preflight

        # Phase 2: Coordinator agents for decision points (agents-as-tools pattern)
        # These agents are called at specific decision points, NOT for every step
        self._technique_coordinator: Optional[Agent] = None
        self._modification_coordinator: Optional[Agent] = None
        self._quality_gate_coordinator: Optional[Agent] = None

        # Phase 7 (DEPRECATED 2A-6): Spec-First agents removed. Truth pack handles validation at $0.

        self._budget_tracker: BudgetTracker = get_budget_tracker()
        self._persistence: SessionPersistence = get_persistence()
        self._pipeline_monitor: PipelineMonitor = PipelineMonitor()  # Phase 2A-4
        self._initialized = False

        # Config system for preset-based settings
        self._config: AgentConfigManager = get_config()
        print(f"[Orchestrator] Using preset: {self._config.preset.name} ({self._config.preset.description})", file=sys.stderr)

        # Phase 2B-5: HITL checkpoint handler (pipeline-level, not SDK-level)
        self._hitl = HITLHandler(
            autonomy_level=self._config.get_hitl_autonomy_level(),
            interactive=self._config.is_hitl_interactive(),
            enabled=self._config.use_hitl(),
        )

        # Local tracing for AI-parseable analysis
        # Tracks Coordinator → modify_script communication flow
        self._flow_tracker = CommunicationFlowTracker(
            log_file=str(Path(__file__).parent / "traces" / "communication_flow.jsonl")
        )
        self._diagnostic_hooks: Optional[DiagnosticHooks] = None
        self._enable_parallel_preflight = os.getenv("VFX_PARALLEL_PREFLIGHT", "1").lower() in ("1", "true", "yes")

    def _get_model_settings(self, agent_name: str) -> tuple[str, ModelSettings]:
        """
        Get model and ModelSettings for an agent from config.

        Args:
            agent_name: Name of agent (e.g., "script_writer", "research_agent")

        Returns:
            Tuple of (model_name, ModelSettings)
        """
        settings = self._config.get_agent_settings(agent_name)
        model_settings_kwargs = settings.to_model_settings()

        # Always set verbosity based on config
        if settings.verbose:
            model_settings_kwargs["verbosity"] = "high"
        else:
            model_settings_kwargs["verbosity"] = "medium"

        return settings.model, ModelSettings(**model_settings_kwargs)

    def _supports_run_config(self) -> bool:
        """Return True if Runner.run accepts run_config in this SDK version."""
        if RunConfig is None:
            return False
        try:
            return "run_config" in inspect.signature(Runner.run).parameters
        except Exception:
            return False

    def _build_run_config(
        self,
        session: "SessionState",
        request: "AssetRequest",
        iteration: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Optional["RunConfig"]:
        """Best-effort RunConfig builder. Falls back safely if unsupported."""
        if RunConfig is None:
            return None
        trace_meta = {
            "asset_name": request.asset_name,
            "effect_type": request.effect_type.value,
            "session_id": session.session_id,
            "generation_mode": "script_writer",  # Phase 2A-6: spec-first deprecated
        }
        if iteration is not None:
            trace_meta["iteration"] = iteration
        if phase:
            trace_meta["phase"] = phase
        try:
            return RunConfig(
                workflow_name="blender-vfx-orchestrator",
                group_id=session.session_id,
                trace_metadata=trace_meta,
                call_model_input_filter=vfx_context_filter,
            )
        except TypeError:
            # Older/newer SDK signature; return default config if possible.
            try:
                return RunConfig()
            except Exception:
                return None

    async def _apply_script_modifications(
        self,
        *,
        script_path: str,
        modifications: Dict[str, Any],
        output_name: Optional[str] = None,
        effect_type: Optional[str] = None,
        previous_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply script changes through guarded modify_script tool path only.

        If effect_type is provided, parameter bounds and damped convergence
        are applied before passing modifications to the script modifier (Phase 2A-5).
        """
        if not modifications:
            return {
                "success": False,
                "error": "No modifications provided",
                "original_path": script_path,
                "modified_path": "",
                "changes_made": [],
                "parameters_changed": {},
            }

        # Phase 2A-5: Apply parameter bounds before modification
        if effect_type and modifications:
            oscillating: set = set()
            if hasattr(self, '_pipeline_monitor'):
                status = self._pipeline_monitor.get_status_report()
                oscillating = set(status.get("oscillating_params", []))
            modifications = apply_parameter_bounds(
                effect_type=effect_type,
                modifications=modifications,
                previous_params=previous_params,
                oscillating_params=oscillating if oscillating else None,
            )

        try:
            # Call _modify_script_impl directly (sync function), NOT the
            # @function_tool-wrapped modify_script (which is a FunctionTool
            # object and not directly awaitable).
            result_raw = _modify_script_impl(
                script_path=script_path,
                modifications=modifications,
                output_name=output_name,
            )
            if isinstance(result_raw, str):
                parsed = json.loads(result_raw)
            elif isinstance(result_raw, dict):
                parsed = result_raw
            else:
                parsed = {
                    "success": False,
                    "error": f"Unexpected _modify_script_impl response type: {type(result_raw).__name__}",
                }
        except Exception as e:
            parsed = {
                "success": False,
                "error": f"modify_script failed: {e}",
            }

        if "original_path" not in parsed:
            parsed["original_path"] = script_path
        if "modified_path" not in parsed:
            parsed["modified_path"] = ""
        if "changes_made" not in parsed:
            parsed["changes_made"] = []
        if "parameters_changed" not in parsed:
            parsed["parameters_changed"] = {}
        return parsed

    def _require_artifact_file(self, artifact_path: str, label: str) -> None:
        """Fail fast when required artifact is missing/empty."""
        if not artifact_path:
            raise RuntimeError(f"{label} artifact path is empty")
        p = Path(artifact_path)
        if not p.exists():
            raise RuntimeError(f"{label} artifact missing: {artifact_path}")
        if p.stat().st_size <= 0:
            raise RuntimeError(f"{label} artifact is empty: {artifact_path}")

    async def _run_agent(
        self,
        agent: Agent,
        prompt: str,
        *,
        context: SharedContext,
        session: Optional[SessionABC] = None,
        hooks: Optional[EnforcementHooks] = None,
        max_turns: Optional[int] = None,
        run_config: Optional["RunConfig"] = None,
    ):
        """Wrapper for Runner.run with optional RunConfig support."""
        kwargs: Dict[str, Any] = {
            "context": context,
            "session": session,
            "hooks": hooks,
            "max_turns": max_turns,
        }
        if run_config is not None and self._supports_run_config():
            kwargs["run_config"] = run_config
        # Remove None entries to preserve older SDK compat
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return await Runner.run(agent, prompt, **kwargs)

    async def _run_parallel_preflight(
        self,
        request: "AssetRequest",
        context: SharedContext,
        sdk_session: Optional[SessionABC],
        research_hooks: EnforcementHooks,
        run_config: Optional["RunConfig"],
    ) -> tuple[Optional[Any], Optional[str]]:
        """Run research + docs expert in parallel (fan-out/fan-in).

        Delegates to phases.research.run_parallel_preflight.
        """
        from phases.research import run_parallel_preflight
        return await run_parallel_preflight(
            orch=self,
            request=request,
            context=context,
            sdk_session=sdk_session,
            research_hooks=research_hooks,
            run_config=run_config,
        )

    async def _run_parallel_learning_and_gate(
        self,
        iteration: int,
        request: "AssetRequest",
        context: SharedContext,
        session: "SessionState",
        sdk_session: Optional[SessionABC],
        script: "ScriptOutput",
        execution: "ExecutionOutput",
        quality: "QualityOutput",
        quality_artifact_path: str,
        scorecard_artifact_path: str,
        baseline_score_snapshot: float,
        baseline_params_snapshot: Dict,
        baseline_render_snapshot: Optional[str],
        learning_hooks: "EnforcementHooks",
        session_mgr: "SessionManager",
        artifact_mgr: "ArtifactManager",
        previous_score: float,
    ) -> tuple[Optional["LearningOutput"], Optional["QualityDecision"]]:
        """
        Run Learning Agent and Quality Gate Coordinator in parallel.

        Since Quality Gate can make decisions based on quality metrics alone
        (without needing Learning Agent's recommendation), we can run them
        concurrently and merge results.

        Returns:
            Tuple of (LearningOutput, QualityDecision) - either may be None on failure
        """
        import time as _time
        start = _time.perf_counter()

        parallel_enabled = os.getenv("VFX_PARALLEL_LEARNING_GATE", "1").lower() in ("1", "true", "yes")
        if not parallel_enabled:
            print(f"[Parallel Learn+Gate] DISABLED, running sequential", file=sys.stderr)
            return None, None  # Signal to caller to use sequential path

        print(f"[Parallel Learn+Gate] PARALLEL mode for iteration {iteration}", file=sys.stderr)

        # Sync baseline before running
        try:
            baseline_params_json = json.dumps(baseline_params_snapshot)
            baseline_scores_json = json.dumps({"overall": baseline_score_snapshot})
            record_experiment_baseline(
                params=baseline_params_json,
                scores=baseline_scores_json,
                render_path=baseline_render_snapshot or execution.render_path or "",
                script_path=script.script_path or ""
            )
        except Exception as e:
            print(f"[Parallel Learn+Gate] WARN: Baseline sync failed: {e}", file=sys.stderr)

        # Build prompts
        learn_ctx = session_mgr.get_context_for_agents()
        artifact_paths_section = artifact_mgr.get_artifact_paths_summary()
        iteration_summary = artifact_mgr.get_iteration_summary(max_iterations=3)

        # Generate deterministic parameter suggestions for Learning Agent
        _learn_det_suggestions = map_quality_issues_to_params(
            issues=quality.issues,
            primary_issue=quality.primary_issue,
            effect_type=request.effect_type.value if hasattr(request.effect_type, 'value') else str(request.effect_type),
        )
        _learn_det_section = format_suggestions_for_prompt(_learn_det_suggestions)

        learn_prompt = f"""Record the experiment results and suggest fixes.

## Current Iteration
Iteration: {iteration}
Script: {script.script_path}
Technique: {script.technique_used}
Render: {execution.render_path}
Score: {quality.overall_score:.1f}
Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}
Quality Artifact: {quality_artifact_path}
Scorecard Artifact: {scorecard_artifact_path}

{iteration_summary}

## Analysis Needed
1. Did the score improve? Delta: {learn_ctx.get('best_score', 0) - previous_score:+.1f}
2. Same issue {learn_ctx.get('consecutive_same_issue', 0)} times in a row
3. Techniques tried: {', '.join(learn_ctx.get('techniques_tried', []))}

{_learn_det_section}

## CRITICAL: Before Suggesting Modifications
FIRST call analyze_script_modifiable_patterns("{script.script_path}") to understand:
- What shader_node_inputs exist (these control visual appearance!)
- What Config class values exist (and if they're used)
- What settings assignments exist
Then provide parameter_modifications using EXACT identifiers from the analysis.
Refine the deterministic suggestions above using the ACTUAL identifiers you find in the script.

## Your Output
- If score improved significantly (delta >= 5), extract the pattern
- If same issue 3+ times, recommend 'switch_technique'
- If issues are parameter-related, provide parameter_modifications using EXACT patterns from script analysis
- Recommend: 'iterate' | 'switch_technique' | 'complete'"""

        ctx = session_mgr.get_context_for_agents()
        # Quality Gate prompt WITHOUT Learning Agent recommendation (independent decision)
        gate_prompt = f"""Interpret quality evaluation results for iteration {iteration}.

## Key Metrics
Score: {quality.overall_score:.1f} / Threshold: {request.quality_threshold} / Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}

## Artifacts (read if needed)
Scorecard: {scorecard_artifact_path}
Quality: {quality_artifact_path}

## Context
Iteration: {iteration}/{request.max_iterations} | Same Issue: {ctx.get('consecutive_same_issue', 0)}x | Best: {session.best_score:.1f} | Escape: {ctx.get('escape_level', 0)}

Decide: Is quality gate PASSED? What is the next action?
Note: Learning Agent is also analyzing in parallel - make your decision based on metrics."""

        # Create parallel tasks
        learn_task = self._run_agent(
            self._learning_agent_standalone,
            learn_prompt,
            context=context,
            session=sdk_session,
            hooks=learning_hooks,
            max_turns=8,
            run_config=self._build_run_config(
                session=session,
                request=request,
                iteration=iteration,
                phase="learning_parallel",
            ),
        )

        gate_task = self._run_agent(
            self._quality_gate_coordinator,
            gate_prompt,
            context=context,
            session=sdk_session,
            max_turns=3,
            run_config=self._build_run_config(
                session=session,
                request=request,
                iteration=iteration,
                phase="quality_gate_parallel",
            ),
        )

        # Fan-out/fan-in
        print(f"[Parallel Learn+Gate] Launching Learning + Quality Gate in parallel...", file=sys.stderr)
        results = await asyncio.gather(learn_task, gate_task, return_exceptions=True)

        elapsed = _time.perf_counter() - start

        # Extract results
        learning_result = results[0] if not isinstance(results[0], Exception) else None
        gate_result = results[1] if not isinstance(results[1], Exception) else None

        learning = None
        if learning_result and hasattr(learning_result, 'final_output'):
            learning = learning_result.final_output

        gate_decision = None
        if gate_result and hasattr(gate_result, 'final_output'):
            gate_decision = gate_result.final_output

        # Log results
        learn_ok = learning is not None
        gate_ok = gate_decision is not None

        if isinstance(results[0], Exception):
            print(f"[Parallel Learn+Gate] Learning FAILED: {results[0]}", file=sys.stderr)
        if isinstance(results[1], Exception):
            print(f"[Parallel Learn+Gate] Quality Gate FAILED: {results[1]}", file=sys.stderr)

        print(f"[Parallel Learn+Gate] Completed in {elapsed:.1f}s", file=sys.stderr)
        print(f"[Parallel Learn+Gate]   Learning: {'OK - ' + learning.next_action if learn_ok else 'FAILED'}", file=sys.stderr)
        print(f"[Parallel Learn+Gate]   Gate: {'OK - ' + gate_decision.next_action if gate_ok else 'FAILED'}", file=sys.stderr)

        return learning, gate_decision

    # Phase 2A-6: Removed 4 spec-first methods (_run_api_spec_only,
    # _run_parallel_technique_and_spec, _run_spec_first_pipeline,
    # _run_spec_first_modification). Truth pack + tool guardrails
    # handle API validation at $0. See git history for reference.

    async def _run_original_script_writer(
        self,
        effect_type: str,
        technique: str,
        request: "AssetRequest",
        context: SharedContext,
        sdk_session: Optional[SessionABC] = None,
    ) -> ScriptOutput:
        """
        Run the original Script Writer as fallback.

        This is used when the Spec-First pipeline fails.
        Uses fallback hooks that don't require doc queries (research already done).
        """
        script_hooks = create_fallback_script_writer_hooks()

        script_prompt = f"""Generate a Blender Python script for {effect_type} VFX.

## Technique: {technique}

## Parameters
- Asset Name: {request.asset_name}
- Effect Type: {effect_type}
- Description: {request.description}
- Resolution: {request.resolution}
- Frames: {request.frame_start}-{request.frame_end}

        Generate a complete, validated script. Return the script_path in your output."""

        try:
            script_result = await self._run_agent(
                self._script_agent_standalone,
                script_prompt,
                context=context,
                session=sdk_session,
                hooks=script_hooks,
                max_turns=15,
                run_config=self._build_run_config(
                    session=context.session,
                    request=request,
                    iteration=context.session.current_iteration or None,
                    phase="script_fallback",
                ),
            )
            return script_result.final_output
        except Exception as e:
            return ScriptOutput(
                script_path="",
                technique_used=technique,
                parameters_set={},
                validation_passed=False,
                validation_errors=[str(e)],
            )

    async def initialize(self) -> None:
        """
        Initialize all specialized agents for code-based pipeline orchestration.

        Must be called before create_asset_pipeline().

        All standalone agents are initialized synchronously. DocsExpert uses
        in-process function_tools (extracted from blender-manual MCP server),
        which avoids the anyio TaskGroup conflicts that blocked MCP tool handlers.
        """
        if self._initialized:
            return

        print("[Orchestrator] Initializing standalone agents...", file=sys.stderr)

        # Create STANDALONE agents for code-based orchestration
        # These have NO handoffs - Python controls the pipeline sequence directly
        # Using structured outputs (output_type) for type-safe data passing between agents
        research_model, research_settings = self._get_model_settings("research_agent")
        self._research_agent = Agent[SharedContext](
            name="Research Agent",
            instructions="""ROLE: Blender documentation research specialist.
INPUTS: effect_type, description.
TOOLS: blender_doc_search_bundle, search_code_patterns, list_patterns_by_effect.

CRITICAL: You have exactly 3 turns. Do NOT exceed them.

TURN 1 — RETRIEVE ALL AT ONCE:
Call ALL of these tools in parallel (one function call each):
  1. blender_doc_search_bundle(effect_type=<type>, description=<desc>, domain=<physics_system>)
     Set domain to the specific physics system (e.g. "rigid_body", "mantaflow", "particles").
  2. blender_doc_search_bundle(effect_type=<type>, description=<desc>, domain=<physics_system>)
     Use a DIFFERENT query focus (alternative techniques, API modules, known pitfalls).
  3. list_patterns_by_effect(effect_type=<type>)
  4. search_code_patterns(issue="", effect_type=<type>)

TURN 2 — ONE OPTIONAL FOLLOW-UP:
If Turn 1 results were empty or irrelevant for a specific sub-question, make ONE targeted follow-up call.
If Turn 1 results were sufficient, skip directly to Turn 3.

TURN 3 — SYNTHESIZE:
Combine ALL findings into ResearchOutput. Use whatever you have — do NOT search again.
If some fields are sparse, fill them with best-effort data from available results.

<output_contract>
ResearchOutput fields — ALL REQUIRED unless marked optional:
- recommended_approach: str — specific technique name, NOT generic (e.g. "Cell Fracture + Rigid Body" not "physics simulation")
- key_parameters: dict — actual Blender parameter names and suggested values
- api_modules: list[str] — bpy.types/bpy.ops references from doc search
- code_patterns: list[{pattern_id, issue, code_snippet}] — from pattern search
- warnings: list[str] — doc search warnings, known gotchas
- alternative_approaches: list[str] — at least 1 alternative technique (REQUIRED for technique diversity)
- doc_refs: list[str] — Blender 5.0 doc references (REQUIRED)
</output_contract>

STOP after synthesizing. Return structured output only.""",
            model=research_model,
            model_settings=research_settings,
            output_type=AgentOutputSchema(ResearchOutput, strict_json_schema=False),
            tools=[
                blender_doc_search_bundle,
                search_code_patterns,
                list_patterns_by_effect,
            ],
            # Phase 3: Output guardrail to enforce doc_refs
            output_guardrails=[validate_research_output],
        )

        # Script Writer with structured output
        # NOTE: Using static instructions for standalone agents (can't append to functions)
        # KB integration happens through tools (query_knowledge_base, get_physics_patterns)
        # No hardcoded physics rules - they emerge from experimentation
        # Phase 3: Input/output guardrails for validation
        # DYNAMIC INSTRUCTIONS: Uses wrapper that calls dynamic_script_writer_instructions + extras
        base_script_writer_standalone = create_script_writer(use_dynamic_instructions=False)
        script_model, script_settings = self._get_model_settings("script_writer")
        self._script_agent_standalone = Agent[SharedContext](
            name="Script Writer",
            instructions=dynamic_script_writer_standalone_instructions,  # Dynamic!
            model=script_model,
            model_settings=script_settings,
            output_type=AgentOutputSchema(ScriptOutput, strict_json_schema=False),
            tools=base_script_writer_standalone.tools,
            # Phase 3: Guardrails for Script Writer
            # - Input: Ensure research context provided, valid effect type
            # - Output: Ensure ScriptOutput has required fields
            input_guardrails=[require_research_context, validate_effect_type],
            output_guardrails=[validate_script_output],
        )

        # Phase 2A-7: Executor agent REMOVED — execution is deterministic.
        # Pipeline calls _execute_blender_script_impl() directly (see PHASE 2: EXECUTION).
        # Historical reference: specialized_agents/executor.py

        # Quality Analyst with structured output (LLM-as-judge pattern)
        # Physics observation happens through tools (observe_physics_anomaly, get_physics_patterns)
        # Phase 3: Input/output guardrails for validation
        # Phase 2A-1: Vision tools have is_enabled=budget_allows_vision (set in create_quality_analyst)
        # DYNAMIC INSTRUCTIONS: Uses wrapper that calls dynamic_quality_analyst_instructions + extras
        base_quality_standalone = create_quality_analyst(use_dynamic_instructions=False)
        quality_model, quality_settings = self._get_model_settings("quality_analyst")
        self._quality_agent_standalone = Agent[SharedContext](
            name="Quality Analyst",
            instructions=dynamic_quality_analyst_standalone_instructions,  # Dynamic!
            model=quality_model,
            model_settings=quality_settings,
            output_type=AgentOutputSchema(QualityOutput, strict_json_schema=False),
            tools=base_quality_standalone.tools,
            # Phase 3: Guardrails for Quality Analyst
            # - Input: Check budget before expensive vision evaluation
            # - Output: Validate QualityOutput structure and critical issue consistency
            input_guardrails=[check_budget_before_quality],
            output_guardrails=[validate_quality_output],
        )

        # Learning Agent with structured output
        # Core of the self-learning system - processes physics observations via tools
        # DYNAMIC INSTRUCTIONS: Uses wrapper that calls dynamic_learning_agent_instructions + extras
        base_learning_standalone = create_learning_agent(use_dynamic_instructions=False)
        learning_model, learning_settings = self._get_model_settings("learning_agent")
        self._learning_agent_standalone = Agent[SharedContext](
            name="Learning Agent",
            instructions=dynamic_learning_agent_standalone_instructions,  # Dynamic!
            model=learning_model,
            model_settings=learning_settings,
            output_type=AgentOutputSchema(LearningOutput, strict_json_schema=False),
            tools=base_learning_standalone.tools + [read_artifact, list_session_artifacts],  # Phase 2B-8
        )

        # Docs Expert Standalone - for parallel preflight (no handoffs)
        # This is used by _run_parallel_preflight to search docs in parallel with research
        # Phase 2A-1: Doc search tools have is_enabled=budget_allows_docs (set in create_docs_expert)
        docs_model, docs_settings = self._get_model_settings("docs_expert")
        base_docs_standalone = create_docs_expert()
        self._docs_expert_standalone = Agent[SharedContext](
            name="Documentation Expert (Standalone)",
            instructions=base_docs_standalone.instructions,  # No handoff instructions
            model=docs_model,
            model_settings=docs_settings,
            tools=base_docs_standalone.tools,
            # No handoffs - runs standalone via Runner.run()
        )

        # =============================================================
        # Phase 2: Create Coordinator Agents (agents-as-tools pattern)
        # =============================================================
        # These lightweight coordinators are called at DECISION POINTS only.
        # Python controls the pipeline; coordinators make intelligent choices.

        print("[Orchestrator] Creating Phase 2 Coordinator agents...", file=sys.stderr)

        # Technique Selection Coordinator (iteration 1)
        self._technique_coordinator = create_technique_selection_coordinator(
            research_agent=self._research_agent,
        )

        # Modification Strategy Coordinator (iteration 2+)
        self._modification_coordinator = create_modification_coordinator()

        # Quality Gate Coordinator (after each iteration)
        self._quality_gate_coordinator = create_quality_gate_coordinator()

        # Phase 7 (DEPRECATED 2A-6): Spec-First agents removed.
        # Truth pack + tool guardrails (2A-3) handle API validation at $0.

        # Phase 2A-3: Attach tool-level guardrails to FunctionTool instances
        try:
            attached = attach_tool_guardrails()
            print(f"[Orchestrator] Tool guardrails attached: {attached}", file=sys.stderr)
        except Exception as e:
            print(f"[Orchestrator] WARNING: Tool guardrail attachment failed: {e}", file=sys.stderr)

        self._initialized = True
        print(f"[Orchestrator] Initialization complete (5 standalone + 3 coordinators ready)", file=sys.stderr)

    async def create_asset_pipeline(
        self,
        request: AssetRequest,
        resume_session_id: Optional[str] = None,
        max_iterations_override: Optional[int] = None,
    ) -> SessionState:
        """
        CODE-BASED ORCHESTRATION: Python controls the pipeline sequence.

        This method implements deterministic workflow control instead of
        relying on LLM instruction-following. Each agent runs independently
        and Python manages the state transitions.

        Pipeline: Research → Script → Execute → Evaluate → Learn → (loop)

        When resume_session_id is provided, loads the existing session and
        resumes from the last completed iteration, skipping Phase 0/0.5.

        Args:
            request: Asset generation request parameters
            resume_session_id: If provided, resume this session instead of creating new
            max_iterations_override: Override max_iterations (useful for extending MAX_ITERATIONS sessions)

        Returns:
            SessionState with final results
        """
        if not self._initialized:
            await self.initialize()

        # Check budget before starting
        if not self._budget_tracker.can_afford_evaluation():
            raise RuntimeError(
                f"Budget exhausted. Monthly limit: ${self._budget_tracker.monthly_limit}"
            )

        # ====== SESSION SETUP: Resume or Create ======
        is_resuming = False

        if resume_session_id:
            # Resume path: load existing session
            session = self._persistence.load_session(resume_session_id)
            if not session:
                raise ValueError(f"Session not found: {resume_session_id}")

            # Terminal status check — already complete
            if session.status in [SessionStatus.PASSED, SessionStatus.CANCELLED]:
                print(f"[Pipeline] Session {resume_session_id} already {session.status.value}, returning", file=sys.stderr)
                return session

            # Apply max_iterations override (useful for MAX_ITERATIONS sessions)
            if max_iterations_override is not None:
                session.request.max_iterations = max_iterations_override

            # Set status to IN_PROGRESS for resumption
            session.status = SessionStatus.IN_PROGRESS

            # Create SharedContext from loaded session
            context = SharedContext(session=session)
            # Sync stuck_state from session to context
            context.stuck_state = session.stuck_state

            session_id = session.session_id
            is_resuming = True

            print(f"[Pipeline] RESUMING session: {session_id}", file=sys.stderr)
        else:
            # Fresh path: create new session
            session_id = generate_session_id(request.asset_name)
            context = create_session_from_request(request, session_id)
            session = context.session

        # Phase 4: SDK session with compaction — preserves conversation context
        # while preventing unbounded token growth. Compaction is triggered manually
        # at the end of each iteration (not auto) to preserve full intra-iteration context.
        sdk_session = get_or_create_sdk_session(session_id)
        print(f"[Pipeline] SDK Session (compacted): {session_id}", file=sys.stderr)

        # Phase 2B-1: Stateless iterations — each iteration gets fresh context
        use_stateless = self._config.use_stateless_iterations()
        iter_session = None if use_stateless else sdk_session
        print(f"[Pipeline] Stateless iterations: {'ON' if use_stateless else 'OFF'}", file=sys.stderr)

        # Artifact-First Handoffs: Create manager for file-based agent communication
        # Instead of passing full data inline in prompts, agents write to disk and pass file refs
        artifact_mgr = get_artifact_manager(session_id)
        print(f"[Pipeline] Artifact Manager: {artifact_mgr.artifact_dir}", file=sys.stderr)

        # Use overridden max_iterations if provided (applies to both paths)
        effective_max_iterations = max_iterations_override if max_iterations_override is not None else request.max_iterations

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"PIPELINE ORCHESTRATION: {request.asset_name}{' (RESUMED)' if is_resuming else ''}", file=sys.stderr)
        print(f"Effect: {request.effect_type.value} | Max Iterations: {effective_max_iterations}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)

        try:
            with trace(f"VFX Pipeline: {request.asset_name}", group_id=session.session_id):
                # Initialize variables set by both fresh and resume paths
                selected_technique: Optional[TechniqueDecision] = None
                research_text = ""

                # ====== PHASES 0 + 0.5 + 0.6: Research, Technique, Truth Pack ======
                from phases.research import run_research_phase
                research_text, selected_technique = await run_research_phase(
                    orch=self,
                    request=request,
                    context=context,
                    session=session,
                    sdk_session=sdk_session,
                    artifact_mgr=artifact_mgr,
                    is_resuming=is_resuming,
                )

                # ====== ITERATION LOOP SETUP ======
                if is_resuming and session.iterations:
                    # Resume path: reconstruct loop variables from last IterationResult
                    last_iter = session.iterations[-1]
                    iteration = last_iter.iteration  # Loop body increments at top
                    previous_script = ScriptOutput(
                        script_path=last_iter.script.script_path,
                        technique_used=last_iter.script.technique_name or "",
                        parameters_set=last_iter.script.modifications or {},
                        validation_passed=last_iter.script.validation_passed,
                        validation_errors=last_iter.script.validation_issues or [],
                    )
                    previous_score = last_iter.score
                    previous_params = last_iter.script.modifications or {}
                    quality = QualityOutput(
                        overall_score=last_iter.quality.overall_score,
                        passed=last_iter.quality.passed,
                        primary_issue=last_iter.quality.primary_issue,
                        issues=last_iter.quality.issues,
                        suggestions=last_iter.quality.suggestions,
                    )
                    learning: Optional[LearningOutput] = None  # Populated after first resumed iteration
                    session_mgr = SessionManager.from_session_state(session)
                    print(f"[Pipeline] RESUME: Restored from iteration {last_iter.iteration} "
                          f"(score={last_iter.score:.1f}, issues={session_mgr.issue_tracker.consecutive_same_issue}x same)",
                          file=sys.stderr)
                else:
                    # Fresh path (or resume with 0 completed iterations)
                    iteration = 0
                    previous_script: Optional[ScriptOutput] = None
                    previous_score = 0.0
                    previous_params: Dict[str, Any] = {}
                    quality: Optional[QualityOutput] = None
                    learning: Optional[LearningOutput] = None
                    code_grounded_feedback = ""  # Phase 3.5: init before loop
                    session_mgr = SessionManager(session_id=session.session_id)

                # Create enforcement hooks for each agent type
                # These prevent loops and enforce documentation-first patterns
                research_hooks = create_research_hooks()
                script_hooks = create_script_writer_hooks()
                quality_hooks = create_quality_analyst_hooks()
                learning_hooks = create_learning_agent_hooks()

                if not is_resuming:
                    # Record initial baseline for iteration 1
                    # This fixes the "No baseline recorded" bug for single-iteration runs
                    # The baseline represents the "before any experiments" state
                    session_mgr.record_baseline(
                        params={},  # No params before first iteration
                        score=0.0,  # Score starts at 0
                        script_path="",  # No script yet
                        render_path=None
                    )
                    print(f"[Pipeline] Initial baseline recorded (score=0.0)", file=sys.stderr)

                # Phase 2B-5: Check for pending HITL checkpoint on resume
                if session.pending_checkpoint and self._hitl:
                    print(f"[Pipeline] HITL: Pending checkpoint found on resume", file=sys.stderr)
                    pending_cp = self._hitl.check_pending(session)
                    if pending_cp and pending_cp.decision == CheckpointDecision.ABORT:
                        print(f"[Pipeline] HITL: Human chose ABORT on pending checkpoint", file=sys.stderr)
                        session.status = SessionStatus.PAUSED
                    elif pending_cp and pending_cp.decision == CheckpointDecision.SWITCH_TECHNIQUE:
                        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
                    session.pending_checkpoint = None  # Consumed

                while iteration < effective_max_iterations:
                    if session.status == SessionStatus.PAUSED:
                        break  # HITL abort from pending checkpoint
                    iteration += 1
                    session.current_iteration = iteration
                    print(f"\n[Pipeline] ====== ITERATION {iteration}/{effective_max_iterations} ======", file=sys.stderr)

                    # Track per-iteration cost and time for utility scoring
                    iteration_start_time = time.monotonic()
                    budget_spent_start = self._budget_tracker.get_spent()

                    # Phase 2: Initialize self-learning variables for this iteration
                    # These will be populated in Phase 0.95 for iteration 2+
                    pattern_to_apply = None
                    kb_suggestions = []

                    # Record baseline BEFORE making changes (iteration 2+)
                    # This fixes the "record_baseline never called" bug
                    if iteration > 1 and previous_script and previous_script.script_path:
                        session_mgr.record_baseline(
                            params=previous_params,
                            score=previous_score,
                            script_path=previous_script.script_path,
                            render_path=session.final_render_path
                        )
                        print(f"[Pipeline] Baseline recorded: score={previous_score:.1f}", file=sys.stderr)

                        # ====== PHASE 0.9: PRE-ITERATION RESEARCH (iter>1 only) ======
                        # Phase 1.1: MANDATORY pre_iteration_research() call
                        # This checks for early warning signals BEFORE modifications
                        print(f"[Pipeline] PHASE 0.9: Pre-Iteration Research", file=sys.stderr)
                        try:
                            pre_research = pre_iteration_research(
                                current_issue=quality.primary_issue if quality else "unknown",
                                current_approach=previous_script.technique_used if previous_script else "unknown",
                                iteration_history=session_mgr.get_iteration_history_json(),  # Use JSON for proper parsing
                                effect_type=request.effect_type.value
                            )
                            import json as _json
                            pre_research_data = _json.loads(pre_research)

                            warning_level = pre_research_data.get("warning_level", "none")
                            escape_action = pre_research_data.get("escape_action", "continue")

                            print(f"[Pipeline] Pre-research: warning={warning_level}, action={escape_action}", file=sys.stderr)

                            # If pre-research recommends technique switch, override learning agent
                            if escape_action == "switch_technique_or_mine_docs":
                                print(f"[Pipeline] Pre-research recommends TECHNIQUE SWITCH", file=sys.stderr)
                                if learning:
                                    learning.next_action = "switch_technique"
                            elif escape_action == "check_knowledge_then_modify":
                                print(f"[Pipeline] Pre-research recommends KNOWLEDGE CHECK first", file=sys.stderr)
                                # Let Modification Coordinator handle this
                        except Exception as e:
                            print(f"[Pipeline] WARN: Pre-iteration research failed: {e}", file=sys.stderr)
                            # Continue without pre-research

                        # ====== PHASE 0.95: SELF-LEARNING REUSE (Phase 2 Implementation) ======
                        # Search patterns and knowledge base BEFORE modifications
                        # This makes extracted patterns influence future iterations
                        # (pattern_to_apply and kb_suggestions initialized at iteration start)

                        if quality and quality.primary_issue:
                            print(f"[Pipeline] PHASE 0.95: Self-Learning Reuse", file=sys.stderr)

                            # Step 1: Search code patterns for matching fixes
                            try:
                                matching_patterns = search_patterns_direct(
                                    issue=quality.primary_issue,
                                    effect_type=request.effect_type.value,
                                    min_confidence=50.0  # Only consider patterns with reasonable confidence
                                )
                                if matching_patterns:
                                    # Find highest confidence pattern
                                    best_pattern = max(matching_patterns, key=lambda p: p.confidence)
                                    print(f"[Pipeline] Pattern found: {best_pattern.name} (confidence={best_pattern.confidence:.0f}%)", file=sys.stderr)

                                    # Apply if confidence >= 70 (high confidence threshold)
                                    if best_pattern.confidence >= 70:
                                        pattern_to_apply = best_pattern
                                        print(f"[Pipeline] HIGH CONFIDENCE pattern - will apply: {best_pattern.pattern_id}", file=sys.stderr)
                                        # Store pattern ID for outcome tracking
                                        context.pending_pattern_id = best_pattern.pattern_id
                                        context.pending_pattern_name = best_pattern.name
                                    else:
                                        print(f"[Pipeline] Pattern confidence too low ({best_pattern.confidence:.0f}%), skipping auto-apply", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Pattern search failed: {e}", file=sys.stderr)

                            # Step 2: Query knowledge base for doc-grounded learnings
                            try:
                                kb_result = query_knowledge_direct(quality.primary_issue)
                                import json as _json
                                kb_data = _json.loads(kb_result)
                                if kb_data.get("results"):
                                    kb_suggestions = kb_data["results"][:3]  # Top 3 learnings
                                    print(f"[Pipeline] Knowledge base: {len(kb_suggestions)} relevant learnings found", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Knowledge base query failed: {e}", file=sys.stderr)

                            # Step 3: Get experiment suggestions (combines patterns + KB)
                            try:
                                suggestions_result = suggest_experiments_direct(
                                    issue=quality.primary_issue,
                                    current_params=json.dumps(previous_params) if previous_params else "{}",
                                    current_scores=json.dumps({"overall": previous_score})
                                )
                                import json as _json
                                suggestions_data = _json.loads(suggestions_result)
                                if suggestions_data.get("suggestions"):
                                    print(f"[Pipeline] Experiment suggestions: {len(suggestions_data['suggestions'])} available", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Experiment suggestions failed: {e}", file=sys.stderr)

                    # ====== PHASE 1: SCRIPT GENERATION ======
                    print(f"[Pipeline] PHASE 1: Script Generation", file=sys.stderr)
                    if iteration == 1:
                        # Determine technique from Coordinator selection
                        technique_name = "unknown"
                        if selected_technique:
                            technique_name = selected_technique.selected_technique

                        # Phase 2A-6: Spec-First pipeline removed. Truth pack + tool guardrails
                        # handle API validation at $0 (see parameter_bounds, truth_pack).
                        print(f"[Pipeline] Script Writer with truth pack validation", file=sys.stderr)

                        # Build script prompt — use TechniqueContract if available (binding),
                        # fall back to advisory prose from TechniqueDecision if not.
                        technique_guidance = ""
                        starting_params = {}
                        active_contract: Optional[TechniqueContract] = None

                        if session.technique_contract:
                            # BINDING CONTRACT from capability pack registry
                            try:
                                active_contract = TechniqueContract(**session.technique_contract)
                                technique_guidance = active_contract.to_script_constraints()
                                starting_params = active_contract.key_parameters
                                print(f"[Pipeline] Using TechniqueContract (BINDING): {active_contract.technique_name}", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Failed to load TechniqueContract: {e}", file=sys.stderr)
                                active_contract = None

                        if not active_contract and selected_technique:
                            # ADVISORY prose (legacy path — no capability pack matched)
                            technique_guidance = f"""
## COORDINATOR SELECTED TECHNIQUE
Technique: {selected_technique.selected_technique}
Reasoning: {selected_technique.reasoning}
Starting Parameters: {json.dumps(selected_technique.key_parameters, indent=2) if selected_technique.key_parameters else 'None specified'}

YOU MUST USE THIS TECHNIQUE. The Coordinator has analyzed the research and selected this as optimal."""
                            starting_params = selected_technique.key_parameters or {}
                            print(f"[Pipeline] Using advisory technique guidance (no contract): {selected_technique.selected_technique}", file=sys.stderr)

                        script_prompt = f"""Generate a Blender Python script for {request.effect_type.value} VFX.
{technique_guidance}

## Research Findings
{research_text[:1500]}

## Parameters
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}
- Description: {request.description}
- Resolution: {request.resolution}
- Frames: {request.frame_start}-{request.frame_end}
{f"- Starting Parameters: {json.dumps(starting_params, indent=2)}" if starting_params else ""}

Generate a complete, validated script using the selected technique. Return the script_path in your output."""

                        # Run Script Writer for iteration 1 with enforcement hooks
                        try:
                            script_result = await self._run_agent(
                                self._script_agent_standalone,
                                script_prompt,
                                context=context,
                                session=iter_session,  # Phase 2B-1: Stateless iterations
                                hooks=script_hooks,
                                max_turns=15,
                                run_config=self._build_run_config(
                                    session=session,
                                    request=request,
                                    iteration=iteration,
                                    phase="script_writer",
                                ),
                            )
                            script: ScriptOutput = script_result.final_output
                        except LoopDetectedError as e:
                            print(f"[Pipeline] WARN: Script Writer loop: {e}", file=sys.stderr)
                            # Create a minimal script output to continue
                            script = ScriptOutput(
                                script_path="",
                                technique_used="loop_detected",
                                parameters_set={},
                                validation_passed=False,
                                validation_errors=[str(e)]
                            )
                        except DocQueryRequiredError as e:
                            print(f"[Pipeline] ERROR: Script Writer missing doc query: {e}", file=sys.stderr)
                            # Force research before continuing
                            script = ScriptOutput(
                                script_path="",
                                technique_used="doc_query_missing",
                                parameters_set={},
                                validation_passed=False,
                                validation_errors=[str(e)]
                            )
                        except Exception as e:
                            # SDK wraps LoopDetectedError in UserError - check for it
                            if "Loop detected" in str(e) or "LoopDetectedError" in str(e):
                                print(f"[Pipeline] WARN: Script Writer loop (wrapped): {e}", file=sys.stderr)
                                script = ScriptOutput(
                                    script_path="",
                                    technique_used="loop_detected",
                                    parameters_set={},
                                    validation_passed=False,
                                    validation_errors=["Loop detected - agent made too many consecutive doc queries"]
                                )
                            else:
                                raise  # Re-raise if it's a different error
                        print(f"[Pipeline] Script hooks stats: {script_hooks.get_stats()}", file=sys.stderr)
                    else:
                        # Track if we successfully modified the script directly
                        direct_modification_success = False
                        script: Optional[ScriptOutput] = None

                        # ====== EXECUTION FAILURE ESCALATION ======
                        # After 2+ consecutive identical execution failures, param tweaks
                        # can't help — force a full spec-first regeneration with a new technique.
                        exec_failure_prefix = "Execution failed:"
                        is_exec_failure = (
                            quality is not None
                            and quality.primary_issue is not None
                            and quality.primary_issue.startswith(exec_failure_prefix)
                        )
                        if (
                            is_exec_failure
                            and session_mgr.issue_tracker.consecutive_same_issue >= 2
                        ):
                            consec = session_mgr.issue_tracker.consecutive_same_issue
                            print(
                                f"[Pipeline] EXECUTION FAILURE x{consec} -- forcing full regeneration",
                                file=sys.stderr,
                            )
                            # Pick an untried technique
                            untried = [
                                a for a in session.alternative_approaches
                                if a not in session_mgr.techniques_tried
                            ]
                            new_technique = untried[0] if untried else f"alternative_{len(session_mgr.techniques_tried) + 1}"
                            print(f"[Pipeline] Switching to technique: {new_technique}", file=sys.stderr)
                            try:
                                script = await self._run_original_script_writer(
                                    effect_type=request.effect_type.value,
                                    technique=new_technique,
                                    request=request,
                                    context=context,
                                    sdk_session=iter_session,  # Phase 2B-1
                                )
                                direct_modification_success = True
                                session_mgr.reset_for_technique_switch(new_technique)
                                self._pipeline_monitor.reset()  # Phase 2A-4
                                print(f"[Pipeline] Regeneration succeeded: {script.script_path}", file=sys.stderr)
                            except Exception as e:
                                print(
                                    f"[Pipeline] Regeneration failed: {e}, falling back to Coordinator",
                                    file=sys.stderr,
                                )
                                # Fall through to existing Coordinator path below

                        # ====== PHASE 1.0.1: PATTERN APPLICATION (Phase 2 Self-Learning) ======
                        # Apply high-confidence pattern if found in Phase 0.95
                        if pattern_to_apply and previous_script and previous_script.script_path:
                            print(f"[Pipeline] PHASE 1.0.1: Applying pattern '{pattern_to_apply.name}'", file=sys.stderr)
                            print(f"[Pipeline] Pattern code:\n{pattern_to_apply.code_snippet[:200]}...", file=sys.stderr)

                            # Parse the pattern's code_snippet to extract parameters
                            # Pattern code_snippet format: "domain.dissolve_speed = 5\ndomain.flame_smoke = 3.0"
                            pattern_params = {}
                            try:
                                for line in pattern_to_apply.code_snippet.strip().split('\n'):
                                    line = line.strip()
                                    if '=' in line and not line.startswith('#'):
                                        # Extract parameter name and value
                                        # Handle patterns like "domain.param = value" or "obj.param = value"
                                        match = re.match(r'(?:\w+\.)?(\w+)\s*=\s*(.+)', line)
                                        if match:
                                            param_name = match.group(1)
                                            value_str = match.group(2).strip()
                                            # Try to parse the value
                                            try:
                                                if value_str.lower() == 'true':
                                                    pattern_params[param_name] = True
                                                elif value_str.lower() == 'false':
                                                    pattern_params[param_name] = False
                                                elif '.' in value_str:
                                                    pattern_params[param_name] = float(value_str)
                                                else:
                                                    pattern_params[param_name] = int(value_str)
                                            except ValueError:
                                                pattern_params[param_name] = value_str

                                if pattern_params:
                                    print(f"[Pipeline] Pattern parameters extracted: {pattern_params}", file=sys.stderr)
                                    output_name = f"{request.asset_name}_iter{iteration}_pattern"

                                    modify_result = await self._apply_script_modifications(
                                        script_path=previous_script.script_path,
                                        modifications=pattern_params,
                                        output_name=output_name,
                                        effect_type=request.effect_type.value,  # Phase 2A-5
                                        previous_params=previous_script.parameters_set,  # Phase 2A-5
                                    )

                                    if modify_result.get("success") and modify_result.get("modified_path"):
                                        print(f"[Pipeline] Pattern application SUCCESS: {modify_result['modified_path']}", file=sys.stderr)
                                        script = ScriptOutput(
                                            script_path=modify_result["modified_path"],
                                            technique_used=previous_script.technique_used + f" (pattern:{pattern_to_apply.pattern_id})",
                                            parameters_set=pattern_params,
                                            validation_passed=True,
                                            validation_errors=[],
                                        )
                                        direct_modification_success = True
                                        # Mark that we applied this pattern (for outcome tracking)
                                        context.last_applied_pattern_id = pattern_to_apply.pattern_id
                                    else:
                                        print(f"[Pipeline] Pattern application FAILED: {modify_result.get('error', 'Unknown')}", file=sys.stderr)
                                else:
                                    print(f"[Pipeline] WARN: Could not extract parameters from pattern", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Pattern parsing failed: {e}", file=sys.stderr)

                        # Check for concrete parameter modifications from Learning Agent
                        # If provided, apply directly without going through Script Writer interpretation
                        if not direct_modification_success and learning and learning.parameter_modifications and previous_script and previous_script.script_path:
                            print(f"[Pipeline] Applying {len(learning.parameter_modifications)} parameter modifications directly", file=sys.stderr)
                            print(f"[Pipeline] Modifications: {learning.parameter_modifications}", file=sys.stderr)

                            # Generate output name for modified script
                            output_name = f"{request.asset_name}_iter{iteration}_paramfix"

                            # Call modify_script directly with concrete parameters
                            modify_result = await self._apply_script_modifications(
                                script_path=previous_script.script_path,
                                modifications=learning.parameter_modifications,
                                output_name=output_name,
                                effect_type=request.effect_type.value,  # Phase 2A-5
                                previous_params=previous_script.parameters_set,  # Phase 2A-5
                            )

                            if modify_result.get("success") and modify_result.get("modified_path"):
                                print(f"[Pipeline] Direct modification SUCCESS: {modify_result['modified_path']}", file=sys.stderr)
                                print(f"[Pipeline] Changes: {modify_result.get('changes_made', [])}", file=sys.stderr)

                                # Create ScriptOutput for the modified script
                                script = ScriptOutput(
                                    script_path=modify_result["modified_path"],
                                    technique_used=previous_script.technique_used + " (param-modified)",
                                    parameters_set=learning.parameter_modifications,
                                    validation_passed=True,
                                    validation_errors=[],
                                )
                                direct_modification_success = True
                            else:
                                print(f"[Pipeline] Direct modification FAILED: {modify_result.get('error', 'Unknown')}", file=sys.stderr)

                        # ====== DETERMINISTIC FALLBACK (zero LLM cost) ======
                        # If neither pattern application nor Learning Agent provided params,
                        # use keyword matching on quality issues to suggest concrete changes.
                        if not direct_modification_success and quality and previous_script and previous_script.script_path:
                            det_suggestions = map_quality_issues_to_params(
                                issues=quality.issues,
                                primary_issue=quality.primary_issue,
                                effect_type=request.effect_type.value,
                            )
                            if det_suggestions:
                                det_params = merge_suggestions_to_modifications(det_suggestions)
                                if det_params:
                                    print(f"[Pipeline] DETERMINISTIC FALLBACK: {len(det_params)} params from keyword matching", file=sys.stderr)
                                    print(f"[Pipeline] Deterministic params: {det_params}", file=sys.stderr)

                                    output_name = f"{request.asset_name}_iter{iteration}_detfix"
                                    modify_result = await self._apply_script_modifications(
                                        script_path=previous_script.script_path,
                                        modifications=det_params,
                                        output_name=output_name,
                                        effect_type=request.effect_type.value,  # Phase 2A-5
                                        previous_params=previous_script.parameters_set,  # Phase 2A-5
                                    )

                                    if modify_result.get("success") and modify_result.get("modified_path"):
                                        matched_count = len(modify_result.get("changes_made", []))
                                        print(f"[Pipeline] Deterministic fallback SUCCESS: {modify_result['modified_path']} ({matched_count} changes)", file=sys.stderr)
                                        script = ScriptOutput(
                                            script_path=modify_result["modified_path"],
                                            technique_used=previous_script.technique_used + " (det-fallback)",
                                            parameters_set=det_params,
                                            validation_passed=True,
                                            validation_errors=[],
                                        )
                                        direct_modification_success = True
                                    else:
                                        print(f"[Pipeline] Deterministic fallback: no params matched script targets", file=sys.stderr)

                        # If no direct modifications (or they failed), consult Modification Coordinator
                        if not direct_modification_success:
                            # ====== PHASE 1.1: MODIFICATION STRATEGY (Coordinator Decision) ======
                            # Phase 2A-4: Record QA suggestion for cascade detection
                            if quality and quality.primary_issue:
                                self._pipeline_monitor.record_qa_suggestion(
                                    quality.primary_issue, quality.overall_score
                                )
                            # Phase 2 Enhancement: Use Modification Coordinator to decide strategy
                            print(f"[Pipeline] PHASE 1.1: Modification Strategy (Coordinator)", file=sys.stderr)

                            mod_decision: Optional[ModificationDecision] = None
                            ctx = session_mgr.get_context_for_agents()
                            iteration_summary = session_mgr.get_iteration_summary()

                            # Build deterministic suggestions section for coordinator prompt
                            _coord_det_suggestions = map_quality_issues_to_params(
                                issues=quality.issues if quality else [],
                                primary_issue=quality.primary_issue if quality else None,
                                effect_type=request.effect_type.value,
                            )
                            _coord_det_section = format_suggestions_for_prompt(_coord_det_suggestions)

                            mod_prompt = f"""Decide modification strategy for iteration {iteration}.

## Current State
- Current Score: {previous_score:.1f}
- Target Score: {request.quality_threshold}
- Primary Issue: {quality.primary_issue if quality else 'Unknown'}
- Consecutive Same Issue: {ctx.get('consecutive_same_issue', 0)}

## Quality Feedback
{quality.vision_assessment if quality else 'No assessment'}
Issues: {', '.join(quality.issues[:3]) if quality and quality.issues else 'None'}

{_coord_det_section}

## Iteration History
{iteration_summary}

## Techniques Tried
{', '.join(session_mgr.techniques_tried) if session_mgr.techniques_tried else 'None'}

## Available Alternatives
{', '.join(session.alternative_approaches[:3]) if session.alternative_approaches else 'None'}

{code_grounded_feedback if code_grounded_feedback else ''}

Decide: modify_params, modify_code, OR switch_technique.
- modify_params: Provide CONCRETE parameter values for tuning issues. Use the deterministic suggestions above as starting points.
- modify_code: Describe structural changes needed (missing objects, wrong setup, broken logic).
- switch_technique: When the current approach is fundamentally broken after 3+ attempts."""

                            try:
                                mod_result = await self._run_agent(
                                    self._modification_coordinator,
                                    mod_prompt,
                                    context=context,
                                    session=iter_session,  # Phase 2B-1: Stateless iterations
                                    max_turns=4,  # Coordinators should be fast
                                    run_config=self._build_run_config(
                                        session=session,
                                        request=request,
                                        iteration=iteration,
                                        phase="modify_strategy",
                                    ),
                                )
                                mod_decision = mod_result.final_output
                                print(f"[Pipeline] Coordinator decision: {mod_decision.action}", file=sys.stderr)
                                print(f"[Pipeline] Reasoning: {mod_decision.reasoning[:60]}...", file=sys.stderr)

                                # TECHNIQUE SWITCH: Generate a fresh script with a new technique
                                if mod_decision.action == 'switch_technique':
                                    print(f"[Pipeline] TECHNIQUE SWITCH → Generating fresh script", file=sys.stderr)

                                    # Pick new technique from alternatives
                                    new_technique = mod_decision.reasoning[:80] if mod_decision.reasoning else "alternative"
                                    untried = [a for a in session.alternative_approaches if a not in session_mgr.techniques_tried]
                                    if untried:
                                        new_technique = untried[0]

                                    # Verify new technique != previous technique
                                    if new_technique == session.current_technique:
                                        stuck_untried = session.stuck_state.get_untried_techniques(
                                            list(TECHNIQUE_ALIASES.keys()) + list(session.alternative_approaches)
                                        )
                                        if stuck_untried:
                                            new_technique = stuck_untried[0]
                                            print(f"[Pipeline] Forced different technique: {new_technique}", file=sys.stderr)

                                    # Rebuild truth pack for new technique
                                    try:
                                        truth_pack = await build_truth_pack(new_technique)
                                        context.truth_pack = truth_pack
                                        set_global_truth_pack(truth_pack)
                                        set_guardrail_truth_pack(truth_pack)  # Phase 2A-3
                                        context.api_spec = truth_pack_to_api_spec(
                                            truth_pack, request.effect_type.value, new_technique
                                        )
                                        print(f"[Pipeline] Truth pack rebuilt for '{new_technique}'", file=sys.stderr)
                                    except Exception as e:
                                        print(f"[Pipeline] WARNING: Truth pack rebuild failed: {e}", file=sys.stderr)

                                    try:
                                        script = await self._run_original_script_writer(
                                            effect_type=request.effect_type.value,
                                            technique=new_technique,
                                            request=request,
                                            context=context,
                                            sdk_session=iter_session,  # Phase 2B-1
                                        )
                                        direct_modification_success = True
                                        session_mgr.reset_for_technique_switch(new_technique)
                                        self._pipeline_monitor.reset()  # Phase 2A-4
                                    except Exception as e:
                                        print(f"[Pipeline] Technique switch failed: {e}, falling back to Script Writer", file=sys.stderr)
                                        # Fall through to existing Script Writer path

                                # MODIFY CODE: Script Writer rewrites with truth pack validation
                                if mod_decision.action == 'modify_code' and mod_decision.code_change_description and previous_script and previous_script.script_path:
                                    print(f"[Pipeline] MODIFY CODE → Script Writer rewrite", file=sys.stderr)
                                    print(f"[Pipeline] Changes requested: {mod_decision.code_change_description[:120]}", file=sys.stderr)

                                    code_fix_prompt = f"""REWRITE this Blender script to fix STRUCTURAL issues.

## Current Script
Path: {previous_script.script_path}
Technique: {previous_script.technique_used}

## What Must Change (from Coordinator)
{mod_decision.code_change_description}

## Quality Feedback
Score: {previous_score:.1f}
Primary Issue: {quality.primary_issue if quality else 'Unknown'}
Issues: {', '.join(quality.issues[:5]) if quality and quality.issues else 'None'}

## Effect Type: {request.effect_type.value}
## Description: {request.description}

## CRITICAL: Doc Query Required
You MUST call blender_doc_search_bundle("{request.effect_type.value}") FIRST to verify APIs before writing code.

## Instructions
1. FIRST: Call blender_doc_search_bundle("{request.effect_type.value}") to get verified Blender APIs
2. Read the current script to understand the existing setup
3. Make the structural changes described above using ONLY verified APIs from the bundle
4. Keep everything that's working — only fix what's broken
5. Use write_script to output the complete fixed script
6. Name the output: {request.asset_name}_script_iter{iteration}_codefix"""

                                    try:
                                        code_fix_hooks = create_error_recovery_hooks()
                                        code_fix_result = await self._run_agent(
                                            self._script_agent_standalone,
                                            code_fix_prompt,
                                            context=context,
                                            session=iter_session,  # Phase 2B-1
                                            hooks=code_fix_hooks,
                                            max_turns=8,
                                            run_config=self._build_run_config(
                                                session=session,
                                                request=request,
                                                iteration=iteration,
                                                phase="code_fix",
                                            ),
                                        )
                                        code_fix_output = code_fix_result.final_output
                                        if code_fix_output and code_fix_output.script_path:
                                            print(f"[Pipeline] Code fix SUCCESS: {code_fix_output.script_path}", file=sys.stderr)
                                            script = code_fix_output
                                            direct_modification_success = True
                                        else:
                                            print(f"[Pipeline] Code fix produced no script_path, falling back", file=sys.stderr)
                                    except Exception as e:
                                        print(f"[Pipeline] Code fix failed: {e}, falling back", file=sys.stderr)

                                # If Coordinator provides parameter changes, try direct modification
                                if mod_decision.action == 'modify_params' and mod_decision.parameter_changes and previous_script and previous_script.script_path:
                                    print(f"[Pipeline] Coordinator provided params: {mod_decision.parameter_changes}", file=sys.stderr)
                                    output_name = f"{request.asset_name}_iter{iteration}_coordfix"

                                    # Record Coordinator output for flow tracking
                                    self._flow_tracker.record_coordinator_output(
                                        coordinator="ModificationCoordinator",
                                        decision=mod_decision.action,
                                        parameters=mod_decision.parameter_changes,
                                        reasoning=mod_decision.reasoning[:200] if mod_decision.reasoning else ""
                                    )

                                    modify_result = await self._apply_script_modifications(
                                        script_path=previous_script.script_path,
                                        modifications=mod_decision.parameter_changes,
                                        output_name=output_name,
                                        effect_type=request.effect_type.value,  # Phase 2A-5
                                        previous_params=previous_script.parameters_set,  # Phase 2A-5
                                    )

                                    # Track changes made for communication breakdown detection
                                    changes_made = modify_result.get("changes_made", [])
                                    params_changed = modify_result.get("parameters_changed", {})

                                    # Record modify result for flow tracking
                                    self._flow_tracker.record_modify_result(
                                        changes_made=changes_made,
                                        success=modify_result.get("success", False) and len(changes_made) > 0,
                                        error=modify_result.get("error", "")
                                    )

                                    if modify_result.get("success") and modify_result.get("modified_path"):
                                        print(f"[Pipeline] Coordinator modification SUCCESS: {modify_result['modified_path']}", file=sys.stderr)
                                        print(f"[Pipeline] Changes applied: {changes_made}", file=sys.stderr)
                                        script = ScriptOutput(
                                            script_path=modify_result["modified_path"],
                                            technique_used=previous_script.technique_used + " (coord-modified)",
                                            parameters_set=mod_decision.parameter_changes,
                                            validation_passed=True,
                                            validation_errors=[],
                                        )
                                        direct_modification_success = True
                                    else:
                                        # COMMUNICATION BREAKDOWN DETECTION
                                        # Coordinator provided params but modify_script couldn't apply them
                                        print(f"[Pipeline] ⚠️ COORDINATOR→MODIFY BREAKDOWN DETECTED:", file=sys.stderr)
                                        print(f"  Coordinator params: {mod_decision.parameter_changes}", file=sys.stderr)
                                        print(f"  Changes made: {changes_made if changes_made else 'NONE'}", file=sys.stderr)
                                        print(f"  Error: {modify_result.get('error', 'Unknown')}", file=sys.stderr)
                                        print(f"  → Coordinator likely provided API fixes that require code changes, not Config params", file=sys.stderr)

                                        # Check if this looks like an API fix vs parameter change
                                        for param_key in mod_decision.parameter_changes.keys():
                                            if any(kw in param_key.lower() for kw in ['clear', 'remove', 'replace', 'fix', 'update']):
                                                print(f"  → HINT: '{param_key}' looks like an API fix, not a numeric parameter", file=sys.stderr)

                            except Exception as e:
                                print(f"[Pipeline] WARN: Modification Coordinator failed: {e}", file=sys.stderr)
                                # Fall through to Script Writer
                                mod_decision = None

                        # If still no success, use Spec-First modification (prevents hallucination)
                        if not direct_modification_success:
                            print(f"[Pipeline] Fallback modification via Spec-First (iter {iteration})", file=sys.stderr)

                            # Build modification instructions from all available feedback
                            mod_instructions_parts = []
                            ctx = session_mgr.get_context_for_agents()

                            if quality and quality.primary_issue:
                                mod_instructions_parts.append(f"Fix primary issue: {quality.primary_issue}")
                            if quality and quality.suggestions:
                                mod_instructions_parts.append(f"Suggestions: {'; '.join(quality.suggestions[:3])}")
                            if learning and learning.suggested_modifications:
                                mod_instructions_parts.append(f"Learning recommendations: {'; '.join(learning.suggested_modifications[:3])}")
                            if ctx.get("stuck_issues"):
                                mod_instructions_parts.append(f"Persistent issues (try different approach): {', '.join(ctx['stuck_issues'])}")

                            mod_instructions = "\n".join(mod_instructions_parts) or "Improve overall quality"

                            # Phase 2A-6: Script Writer with truth pack validation
                            print(f"[Pipeline] Fallback modification via Script Writer (iter {iteration})", file=sys.stderr)

                            # Build compact alternatives/techniques lists
                            untried = [a for a in session.alternative_approaches if a not in session.techniques_tried][:5]
                            untried_str = ", ".join(untried) if untried else "None"
                            tried_str = ", ".join(session.techniques_tried) if session.techniques_tried else "None"

                            script_prompt = f"""Modify the existing script to fix quality issues.

## Script
Path: {previous_script.script_path if previous_script else 'Unknown'}
Technique: {previous_script.technique_used if previous_script else 'Unknown'}

## Issue to Fix
{mod_instructions}

## Context
Untried approaches: {untried_str}
Already tried: {tried_str}

## Target
Previous: {previous_score:.1f} | Target: {request.quality_threshold}

## CRITICAL: Doc Query Required
You MUST call blender_doc_search_bundle("{request.effect_type.value}") FIRST to verify APIs before modifying code.

## Instructions
1. FIRST: Call blender_doc_search_bundle("{request.effect_type.value}") to get verified Blender APIs
2. Then fix the primary issue using ONLY verified APIs from the bundle"""

                            iter_script_hooks = create_fallback_script_writer_hooks()
                            try:
                                script_result = await self._run_agent(
                                    self._script_agent_standalone,
                                    script_prompt,
                                    context=context,
                                    session=iter_session,  # Phase 2B-1
                                    hooks=iter_script_hooks,
                                    max_turns=15,
                                    run_config=self._build_run_config(
                                        session=session,
                                        request=request,
                                        iteration=iteration,
                                        phase="script_modify",
                                    ),
                                )
                                script = script_result.final_output
                            except LoopDetectedError as e:
                                print(f"[Pipeline] WARN: Script Writer loop (iter {iteration}): {e}", file=sys.stderr)
                                script = ScriptOutput(
                                    script_path="",
                                    technique_used="loop_detected",
                                    parameters_set={},
                                    validation_passed=False,
                                    validation_errors=[str(e)]
                                )
                            except Exception as e:
                                if "Loop detected" in str(e) or "LoopDetectedError" in str(e):
                                    print(f"[Pipeline] WARN: Script Writer loop (iter {iteration}, wrapped): {e}", file=sys.stderr)
                                    script = ScriptOutput(
                                        script_path="",
                                        technique_used="loop_detected",
                                        parameters_set={},
                                        validation_passed=False,
                                        validation_errors=["Loop detected"]
                                    )
                                else:
                                    raise
                            print(f"[Pipeline] Script hooks stats (iter {iteration}): {iter_script_hooks.get_stats()}", file=sys.stderr)

                    # Always capture script_path if available (for subsequent iterations)
                    if script.script_path:
                        session.current_script_path = script.script_path
                        previous_script = script

                        # Track techniques tried (for technique switching logic)
                        if script.technique_used and script.technique_used not in session.techniques_tried:
                            session.techniques_tried.append(script.technique_used)
                            print(f"[Pipeline] New technique: {script.technique_used} (total tried: {len(session.techniques_tried)})", file=sys.stderr)
                        session.current_technique = script.technique_used

                        # ====== CONTRACT ADHERENCE CHECK ======
                        # If a TechniqueContract is active, verify the script actually
                        # follows it. This catches the core F1/F2 failure: script writer
                        # ignoring technique selection and defaulting to training priors.
                        if session.technique_contract and script.script_path:
                            try:
                                contract = TechniqueContract(**session.technique_contract)
                                script_content = Path(script.script_path).read_text()
                                adhered, violations = contract.check_adherence(script_content)
                                if adhered:
                                    print(f"[Pipeline] CONTRACT ADHERENCE: PASSED", file=sys.stderr)
                                else:
                                    print(f"[Pipeline] CONTRACT ADHERENCE: FAILED ({len(violations)} violations)", file=sys.stderr)
                                    for v in violations:
                                        print(f"[Pipeline]   VIOLATION: {v}", file=sys.stderr)
                                    # Add violations to script validation errors so downstream
                                    # phases are aware, but don't block execution — the script
                                    # may still produce a usable render.
                                    if not script.validation_errors:
                                        script.validation_errors = []
                                    script.validation_errors.extend(
                                        [f"CONTRACT: {v}" for v in violations]
                                    )
                            except Exception as e:
                                print(f"[Pipeline] WARN: Contract adherence check failed: {e}", file=sys.stderr)

                        print(f"[Pipeline] Script: {script.script_path} ({script.technique_used})", file=sys.stderr)
                    else:
                        print(f"[Pipeline] WARNING: No script_path returned", file=sys.stderr)

                    if not script.validation_passed:
                        print(f"[Pipeline] WARNING: Script validation failed: {script.validation_errors}", file=sys.stderr)
                        # If we have a script path, try executing anyway (some validation errors are warnings)
                        if not script.script_path:
                            # No script to execute - continue to next iteration
                            quality = QualityOutput(
                                overall_score=0,
                                passed=False,
                                primary_issue="Script generation failed - no path",
                                issues=script.validation_errors,
                                suggestions=["Fix script generation errors"]
                            )
                            # Record failure so stuck detection sees this iteration
                            current_params = script.parameters_set if hasattr(script, 'parameters_set') and script.parameters_set else {}
                            previous_params = current_params
                            previous_score = 0
                            session_mgr.record_result(
                                iteration=iteration,
                                params=current_params,
                                score=0,
                                issues=quality.issues,
                                primary_issue=quality.primary_issue,
                                technique=script.technique_used if script else None
                            )
                            print(f"[Pipeline] Script failure recorded: consecutive_same_issue={session_mgr.issue_tracker.consecutive_same_issue}", file=sys.stderr)
                            continue
                        # Else proceed with execution to see actual results

                    # ====== PHASE 1.5: API VALIDATION (Blender 5.0) ======
                    # Validate API calls BEFORE execution to catch Blender 5.0 breaking changes
                    # This addresses Problem 1 from Architecture Optimization Plan
                    if script.script_path:
                        print(f"[Pipeline] PHASE 1.5: API Validation", file=sys.stderr)
                        try:
                            # Read the script content
                            script_content = Path(script.script_path).read_text()

                            # Lightweight validation (no agent call - fast)
                            api_validation: CodeValidationResult = await validate_code_api(script_content)

                            print(f"[Pipeline] API Validation: {api_validation.valid_calls}/{api_validation.total_calls_checked} valid", file=sys.stderr)

                            if not api_validation.is_valid and api_validation.corrections_needed:
                                print(f"[Pipeline] API CORRECTIONS NEEDED:", file=sys.stderr)
                                for correction in api_validation.corrections_needed:
                                    print(f"[Pipeline]   - {correction}", file=sys.stderr)

                                # Apply corrections to script content
                                corrected_content = script_content
                                for validation in api_validation.validations:
                                    if not validation.is_valid and validation.correction:
                                        correction = validation.correction
                                        api_call = validation.api_call
                                        # If correction is a comment, comment out lines containing the API call
                                        if correction.startswith("#"):
                                            lines = corrected_content.split('\n')
                                            for i, line in enumerate(lines):
                                                if api_call in line and not line.strip().startswith('#'):
                                                    lines[i] = f"# REMOVED (Blender 5.0): {line.strip()}  {correction}"
                                            corrected_content = '\n'.join(lines)
                                        else:
                                            # Direct token replacement
                                            # BUGFIX 2026-01-27: Assignment patterns (containing '=') must use
                                            # full string replacement. Token splitting on '.' corrupts decimal
                                            # literals (e.g., 6.0 becomes 6.<correction> when replacing "0")
                                            if '=' in api_call:
                                                # Assignment pattern (noise_scale = 1.0) - replace full match
                                                corrected_content = corrected_content.replace(api_call, correction)
                                            else:
                                                # Property access (.use_caching) - use token replacement
                                                old_token = api_call.split('.')[-1] if '.' in api_call else api_call
                                                new_token = correction.split('.')[-1] if '.' in correction else correction
                                                corrected_content = corrected_content.replace(old_token, new_token)

                                # Also apply known patterns directly
                                for pattern, fix in KNOWN_API_CHANGES.items():
                                    if pattern in corrected_content:
                                        correction = fix["correction"]
                                        # If correction is a comment (line removal), comment out the entire line
                                        if correction.startswith("#"):
                                            # Find and comment out lines containing this pattern
                                            lines = corrected_content.split('\n')
                                            for i, line in enumerate(lines):
                                                if pattern in line and not line.strip().startswith('#'):
                                                    # Comment out the line with explanation
                                                    lines[i] = f"# REMOVED (Blender 5.0): {line.strip()}  {correction}"
                                            corrected_content = '\n'.join(lines)
                                        else:
                                            # Direct replacement for non-comment corrections
                                            corrected_content = corrected_content.replace(pattern, correction)
                                        print(f"[Pipeline]   Applied known fix: {pattern} → {correction[:50]}...", file=sys.stderr)

                                # Write corrected script
                                corrected_path = script.script_path.replace('.py', '_api_fixed.py')
                                Path(corrected_path).write_text(corrected_content)
                                print(f"[Pipeline] Corrected script saved: {corrected_path}", file=sys.stderr)

                                # Update script path to use corrected version
                                script = ScriptOutput(
                                    script_path=corrected_path,
                                    technique_used=script.technique_used + " (api-corrected)",
                                    parameters_set=script.parameters_set if hasattr(script, 'parameters_set') else {},
                                    validation_passed=True,  # Now validated
                                    validation_errors=[]
                                )
                                session.current_script_path = corrected_path
                                previous_script = script

                        except Exception as e:
                            print(f"[Pipeline] API Validation error (non-fatal): {e}", file=sys.stderr)
                            # Continue with original script if validation fails

                    # ====== PHASE 1.9: PIPELINE MONITOR (After Generation) ======
                    if script and hasattr(script, 'script_path') and script.script_path:
                        gen_alerts = self._pipeline_monitor.check_after_generation(
                            script.script_path, iteration
                        )
                        for alert in gen_alerts:
                            print(f"[Monitor] {alert.level.value}: {alert.message}", file=sys.stderr)

                    # ====== PHASES 2 + 2.5-2.7 + Artifact Gates ======
                    from phases.execution import run_execution_phase
                    exec_result = await run_execution_phase(
                        orch=self,
                        request=request,
                        context=context,
                        session=session,
                        sdk_session=sdk_session,
                        iter_session=iter_session,
                        artifact_mgr=artifact_mgr,
                        session_mgr=session_mgr,
                        script=script,
                        iteration=iteration,
                    )
                    execution = exec_result.execution
                    script = exec_result.script  # May be updated by recovery
                    if exec_result.should_continue:
                        quality = exec_result.quality
                        previous_params = exec_result.previous_params
                        previous_score = exec_result.previous_score
                        continue

                    # Capture baseline snapshot BEFORE quality evaluation updates previous_score
                    # This fixes the "record_baseline not called" bug where baseline was using current score
                    baseline_score_snapshot = previous_score
                    baseline_params_snapshot = dict(previous_params) if previous_params else {}
                    baseline_render_snapshot = session.final_render_path if session.iterations else None

                    # ====== PHASE 3: QUALITY EVALUATION (LLM-as-Judge) ======
                    # QW-1: Budget check before expensive vision evaluation
                    if not self._budget_tracker.can_afford_evaluation():
                        print(f"[Pipeline] BUDGET EXHAUSTED - using last available score", file=sys.stderr)
                        quality = QualityOutput(
                            overall_score=previous_score,  # Use last known score
                            passed=previous_score >= request.quality_threshold,
                            primary_issue="Budget exhausted - returning best available result",
                            issues=["Budget limit reached"],
                            suggestions=["Increase budget allocation or reduce quality requirements"],
                            vision_assessment="No vision evaluation - budget exhausted"
                        )
                        session.status = SessionStatus.BUDGET_EXHAUSTED if previous_score < request.quality_threshold else SessionStatus.PASSED
                        break

                    # ====== PHASE 3 TIER 1: Deterministic Render Checks ($0) ======
                    quality = None  # Sentinel: if set by Tier 1, skip LLM eval
                    if self._config.use_multi_grader_eval():
                        det_result = run_deterministic_checks(
                            render_path=execution.render_path,
                            script_path=script.script_path if script else None,
                            effect_type=request.effect_type.value,
                        )
                        print(f"[Pipeline] Tier 1: {det_result}", file=sys.stderr)
                        if not det_result.passed:
                            # Critical failure — short-circuit with score=0
                            quality = QualityOutput(
                                overall_score=0,
                                passed=False,
                                primary_issue=det_result.critical_issues[0] if det_result.critical_issues else "Deterministic check failed",
                                issues=det_result.critical_issues,
                                suggestions=["Fix critical render issue before re-evaluating"],
                                vision_assessment=f"Skipped (Tier 1 deterministic failure: {', '.join(det_result.critical_issues)})",
                            )
                            print(f"[Pipeline] Tier 1 CRITICAL FAIL — skipping LLM vision eval (saved ~$0.05)", file=sys.stderr)

                    # ====== PHASE 3 TIER 3: LLM Vision Evaluation ($0.01-0.05) ======
                    if quality is None:
                        print(f"[Pipeline] PHASE 3: Quality Analyst (LLM-as-Judge)", file=sys.stderr)
                        # Truncate description to avoid context bloat but keep enough for scene-aware eval
                        _desc_for_eval = request.description[:600] if request.description else ""
                        eval_prompt = f"""Evaluate the render quality strictly.

Render Path: {execution.render_path}
Effect Type: {request.effect_type.value}
Reference: {request.reference_path or "None"}
Quality Threshold: {request.quality_threshold}

Scene Description: {_desc_for_eval}

IMPORTANT: When calling analyze_with_vision and compare_to_reference, ALWAYS pass the
scene_description parameter with the scene description above. This ensures the vision
model evaluates the render against what was actually requested, not just the effect_type label.

Be a strict judge. Only pass renders that truly meet quality standards.
Provide detailed feedback for improvement."""

                        try:
                            eval_result = await self._run_agent(
                                self._quality_agent_standalone,
                                eval_prompt,
                                context=context,
                                session=iter_session,  # Phase 2B-1: Stateless iterations
                                hooks=quality_hooks,
                                max_turns=6,
                                run_config=self._build_run_config(
                                    session=session,
                                    request=request,
                                    iteration=iteration,
                                    phase="quality",
                                ),
                            )
                            # Structured output: QualityOutput
                            quality = eval_result.final_output
                        except LoopDetectedError as e:
                            print(f"[Pipeline] WARN: Quality Analyst loop: {e}", file=sys.stderr)
                            quality = QualityOutput(
                                overall_score=0,
                                passed=False,
                                primary_issue=f"Quality evaluation loop detected: {e}",
                                issues=[str(e)],
                                suggestions=["Simplify evaluation criteria"],
                                vision_assessment="Unable to complete evaluation due to loop",
                                reference_similarity=None
                            )
                        print(f"[Pipeline] Quality hooks stats: {quality_hooks.get_stats()}", file=sys.stderr)

                    # Artifact-First: Write quality evaluation to file
                    quality_artifact_path = artifact_mgr.write_quality_from_output(
                        iteration=iteration,
                        overall_score=quality.overall_score,
                        passed=quality.passed,
                        primary_issue=quality.primary_issue,
                        issues=quality.issues if hasattr(quality, 'issues') else [],
                        suggestions=quality.suggestions if hasattr(quality, 'suggestions') else [],
                        vision_assessment=quality.vision_assessment if hasattr(quality, 'vision_assessment') else "",
                        reference_similarity=quality.reference_similarity if hasattr(quality, 'reference_similarity') else None,
                        render_path=execution.render_path,
                    )
                    self._require_artifact_file(quality_artifact_path, "quality")
                    print(f"[Pipeline] Quality artifact: {quality_artifact_path}", file=sys.stderr)

                    previous_score = quality.overall_score
                    print(f"[Pipeline] Score: {quality.overall_score:.1f} | Passed: {quality.passed}", file=sys.stderr)
                    if quality.primary_issue:
                        print(f"[Pipeline] Issue: {quality.primary_issue[:60]}...", file=sys.stderr)

                    # ====== PHASE 3.5: CODE-GROUNDED QA FEEDBACK ======
                    # Pairs QA visual critique with actual script parameters
                    code_grounded_feedback = ""
                    if not quality.passed and script.script_path:
                        try:
                            code_grounded_feedback = create_code_grounded_feedback(
                                quality, script.script_path, context.truth_pack
                            )
                            print(f"[Pipeline] Code-grounded feedback: {len(code_grounded_feedback)} chars", file=sys.stderr)
                        except Exception as e:
                            print(f"[Pipeline] WARN: Code-grounded feedback failed: {e}", file=sys.stderr)

                    # Update session (quality best)
                    session.best_score = max(session.best_score, quality.overall_score)
                    if quality.overall_score == session.best_score:
                        session.best_iteration = iteration
                        session.final_render_path = execution.render_path

                    # Update session (SICA utility best)
                    iteration_duration = time.monotonic() - iteration_start_time
                    iteration_cost = max(0.0, self._budget_tracker.get_spent() - budget_spent_start)
                    utility_score = compute_sica_utility(
                        score=quality.overall_score,
                        cost_usd=iteration_cost,
                        time_seconds=iteration_duration
                    )
                    if utility_score >= session.best_utility_score:
                        session.best_utility_score = utility_score
                        session.best_utility_iteration = iteration
                        session.best_utility_quality_score = quality.overall_score
                        session.best_utility_cost_usd = iteration_cost
                        session.best_utility_time_seconds = iteration_duration
                        session.best_utility_script_path = script.script_path
                        session.best_utility_render_path = execution.render_path
                        print(
                            f"[Pipeline] Utility best: U={utility_score:.3f} "
                            f"(score={quality.overall_score:.1f}, cost=${iteration_cost:.2f}, "
                            f"time={iteration_duration:.1f}s) at iter {iteration}",
                            file=sys.stderr
                        )

                    # Artifact-First: Write scorecard combining quality + gates + metrics
                    # Extract critical issues for scorecard
                    critical_issues = [
                        issue for issue in (quality.issues if hasattr(quality, 'issues') else [])
                        if any(kw in issue.upper() for kw in ["ZERO_LIGHTS", "BLACK_SCREEN", "WHITE_SCREEN", "CLIPPING"])
                    ]
                    scorecard_artifact_path = artifact_mgr.write_scorecard_from_values(
                        iteration=iteration,
                        overall_score=quality.overall_score,
                        passed=quality.passed,
                        critical_issues=critical_issues,
                        warnings=quality.suggestions if hasattr(quality, 'suggestions') else [],
                        cache_size_mb=context.last_artifact_summary.cache_size_mb if getattr(context, 'last_artifact_summary', None) else 0.0,
                        render_count=context.last_artifact_summary.render_count if getattr(context, 'last_artifact_summary', None) else 0,
                        vdb_count=context.last_artifact_summary.vdb_count if getattr(context, 'last_artifact_summary', None) else 0,
                        iteration_cost_usd=iteration_cost,
                        iteration_time_seconds=iteration_duration,
                        utility_score=utility_score,
                    )
                    print(f"[Pipeline] Scorecard artifact: {scorecard_artifact_path}", file=sys.stderr)

                    # Extract current params from script (use parameters_set, the correct field on ScriptOutput)
                    current_params = script.parameters_set if hasattr(script, 'parameters_set') and script.parameters_set else {}
                    previous_params = current_params  # Save for next iteration's baseline

                    # Record result in SessionManager for iteration history tracking
                    session_mgr.record_result(
                        iteration=iteration,
                        params=current_params,
                        score=quality.overall_score,
                        issues=quality.issues if hasattr(quality, 'issues') else [],
                        primary_issue=quality.primary_issue,
                        technique=script.technique_used
                    )
                    print(f"[Pipeline] Result recorded: params_changed={len(session_mgr.iteration_history[-1].params_changed) if session_mgr.iteration_history else 0}", file=sys.stderr)

                    # Record iteration - create nested models from agent outputs
                    # Convert validation_errors to strings if they're dicts
                    validation_issues_str = []
                    if hasattr(script, 'validation_errors') and script.validation_errors:
                        for err in script.validation_errors:
                            if isinstance(err, dict):
                                # Format: "severity: message (category)"
                                sev = err.get('severity', 'info')
                                msg = err.get('message', str(err))
                                cat = err.get('category', '')
                                validation_issues_str.append(f"{sev}: {msg}" + (f" ({cat})" if cat else ""))
                            else:
                                validation_issues_str.append(str(err))
                    script_mod = ScriptModification(
                        script_path=script.script_path or "unknown",
                        modifications=script.parameters_set if hasattr(script, 'parameters_set') else {},
                        technique_name=script.technique_used,
                        validation_passed=script.validation_passed,
                        validation_issues=validation_issues_str,
                    )
                    blender_exec = BlenderExecution(
                        success=execution.success,
                        run_dir=str(Path(execution.render_path).parent) if execution.render_path else "",
                        render_path=execution.render_path,
                        vdb_files=[],
                        execution_time_seconds=execution.execution_time if hasattr(execution, 'execution_time') else None,
                        errors=[execution.error_message] if execution.error_message else [],
                    )
                    quality_metrics = QualityMetrics(
                        overall_score=quality.overall_score,
                        passed=quality.passed,
                        issues=quality.issues,
                        primary_issue=quality.primary_issue,
                        suggestions=quality.recommendations if hasattr(quality, 'recommendations') else [],
                    )
                    iter_result = IterationResult(
                        iteration=iteration,
                        script=script_mod,
                        execution=blender_exec,
                        quality=quality_metrics,
                        passed=quality.passed,
                        score=quality.overall_score,
                    )
                    session.iterations.append(iter_result)

                    # ====== PHASE 3.9: PIPELINE MONITOR (After Evaluation) ======
                    eval_alerts = self._pipeline_monitor.check_after_evaluation(
                        score=quality.overall_score,
                        primary_issue=quality.primary_issue,
                        params=session_mgr.baseline.params if session_mgr.baseline else {},
                        iteration=iteration,
                        budget_spent=self._budget_tracker.get_spent() - budget_spent_start,
                        vision_assessment=quality.vision_assessment,
                    )
                    for alert in eval_alerts:
                        print(f"[Monitor] {alert.level.value}: {alert.message}", file=sys.stderr)
                        if alert.alert_type == AlertType.CRITICAL_RENDER_ISSUE:
                            # Inject critical issue into quality feedback for downstream handling
                            if not quality.primary_issue or "BLACK_SCREEN" not in (quality.primary_issue or ""):
                                quality.primary_issue = alert.details.get("keywords", ["CRITICAL_RENDER"])[0]

                    # ====== PHASE 3.95: HITL CHECKPOINT (After Evaluation) ======
                    if self._hitl:
                        budget_status = self._budget_tracker.get_status()
                        hitl_checkpoint = self._hitl.check_post_eval(session, quality, budget_status)
                        if hitl_checkpoint:
                            if hitl_checkpoint.decision == CheckpointDecision.ABORT:
                                print(f"[Pipeline] HITL: Human chose ABORT", file=sys.stderr)
                                session.status = SessionStatus.PAUSED
                                break
                            elif hitl_checkpoint.decision == CheckpointDecision.SWITCH_TECHNIQUE:
                                session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
                            elif hitl_checkpoint.decision == CheckpointDecision.INCREASE_BUDGET:
                                # Bump vision budget by $5
                                self._budget_tracker.state.category_limits["vision"] = (
                                    self._budget_tracker.state.category_limits.get("vision", 10.0) + 5.0
                                )
                                print(f"[Pipeline] HITL: Budget increased by $5", file=sys.stderr)
                            elif hitl_checkpoint.decision is None:
                                # Non-interactive: save and pause
                                session.pending_checkpoint = hitl_checkpoint.to_dict()
                                session.status = SessionStatus.PAUSED
                                print(f"[Pipeline] HITL: Checkpoint saved, pausing for human decision", file=sys.stderr)
                                break
                            # Record in history
                            if hasattr(session, "hitl_history"):
                                session.hitl_history.append(hitl_checkpoint.to_dict())

                    # ====== PHASE 4+5: LEARNING + QUALITY GATE (Parallel or Sequential) ======
                    print(f"[Pipeline] PHASE 4+5: Learning + Quality Gate", file=sys.stderr)

                    # Try parallel Learning + Quality Gate first
                    learning: Optional[LearningOutput] = None
                    gate_decision: Optional[QualityDecision] = None

                    parallel_result = await self._run_parallel_learning_and_gate(
                        iteration=iteration,
                        request=request,
                        context=context,
                        session=session,
                        sdk_session=iter_session,  # Phase 2B-1: Stateless iterations
                        script=script,
                        execution=execution,
                        quality=quality,
                        quality_artifact_path=quality_artifact_path,
                        scorecard_artifact_path=scorecard_artifact_path,
                        baseline_score_snapshot=baseline_score_snapshot,
                        baseline_params_snapshot=baseline_params_snapshot,
                        baseline_render_snapshot=baseline_render_snapshot,
                        learning_hooks=learning_hooks,
                        session_mgr=session_mgr,
                        artifact_mgr=artifact_mgr,
                        previous_score=previous_score,
                    )

                    if parallel_result != (None, None):
                        # Parallel path succeeded - use pre-computed results
                        learning, gate_decision = parallel_result
                        print(f"[Pipeline] PARALLEL Learning+Gate completed", file=sys.stderr)
                        if learning:
                            print(f"[Pipeline] Learning: next_action={learning.next_action}", file=sys.stderr)
                        if gate_decision:
                            print(f"[Pipeline] Gate: passed={gate_decision.passed}, next={gate_decision.next_action}", file=sys.stderr)
                    else:
                        # Sequential fallback - parallel disabled or both failed
                        print(f"[Pipeline] SEQUENTIAL Learning+Gate (parallel disabled/failed)", file=sys.stderr)

                        # Sync baseline to ExperimentTracker (for Learning Agent's record_experiment_result)
                        # Uses the snapshot captured BEFORE quality evaluation updated previous_score
                        try:
                            baseline_params_json = json.dumps(baseline_params_snapshot)
                            baseline_scores_json = json.dumps({"overall": baseline_score_snapshot})
                            record_experiment_baseline(
                                params=baseline_params_json,
                                scores=baseline_scores_json,
                                render_path=baseline_render_snapshot or execution.render_path or "",
                                script_path=script.script_path or ""
                            )
                            print(f"[Pipeline] Baseline recorded: score={baseline_score_snapshot:.1f}", file=sys.stderr)
                        except Exception as e:
                            print(f"[Pipeline] WARN: Baseline sync failed: {e}", file=sys.stderr)

                        # Get context from SessionManager for informed decisions
                        learn_ctx = session_mgr.get_context_for_agents()

                        # Artifact-First: Reference artifacts instead of inline dumps
                        artifact_paths_section = artifact_mgr.get_artifact_paths_summary()
                        iteration_summary = artifact_mgr.get_iteration_summary(max_iterations=3)

                        # Generate deterministic parameter suggestions for Learning Agent (resume path)
                        _learn_det_suggestions_r = map_quality_issues_to_params(
                            issues=quality.issues,
                            primary_issue=quality.primary_issue,
                            effect_type=request.effect_type.value if hasattr(request.effect_type, 'value') else str(request.effect_type),
                        )
                        _learn_det_section_r = format_suggestions_for_prompt(_learn_det_suggestions_r)

                        learn_prompt = f"""Record the experiment results and suggest fixes.

## Current Iteration
Iteration: {iteration}
Script: {script.script_path}
Technique: {script.technique_used}
Render: {execution.render_path}
Score: {quality.overall_score:.1f}
Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}
Quality Artifact: {quality_artifact_path}
Scorecard Artifact: {scorecard_artifact_path}

{iteration_summary}

## Analysis Needed
1. Did the score improve? Delta: {learn_ctx.get('best_score', 0) - previous_score:+.1f}
2. Same issue {learn_ctx.get('consecutive_same_issue', 0)} times in a row
3. Techniques tried: {', '.join(learn_ctx.get('techniques_tried', []))}

{_learn_det_section_r}

## CRITICAL: Before Suggesting Modifications
FIRST call analyze_script_modifiable_patterns("{script.script_path}") to understand:
- What shader_node_inputs exist (these control visual appearance!)
- What Config class values exist (and if they're used)
- What settings assignments exist
Then provide parameter_modifications using EXACT identifiers from the analysis.
Refine the deterministic suggestions above using the ACTUAL identifiers you find in the script.

## Your Output
- If score improved significantly (delta >= 5), extract the pattern
- If same issue 3+ times, recommend 'switch_technique'
- If issues are parameter-related, provide parameter_modifications using EXACT patterns from script analysis
- Recommend: 'iterate' | 'switch_technique' | 'complete'"""

                        try:
                            learn_result = await self._run_agent(
                                self._learning_agent_standalone,
                                learn_prompt,
                                context=context,
                                session=iter_session,  # Phase 2B-1: Stateless iterations
                                hooks=learning_hooks,
                                max_turns=8,
                                run_config=self._build_run_config(
                                    session=session,
                                    request=request,
                                    iteration=iteration,
                                    phase="learning",
                                ),
                            )
                            # Structured output: LearningOutput
                            learning = learn_result.final_output
                        except LoopDetectedError as e:
                            print(f"[Pipeline] WARN: Learning Agent loop: {e}", file=sys.stderr)
                            learning = LearningOutput(
                                experiment_recorded=False,
                                pattern_extracted=False,
                                pattern_id=None,
                                next_action="iterate",  # Default to iterate on loop
                                suggested_modifications=["Learning agent hit loop - using default action"],
                                parameter_modifications={}
                            )
                        print(f"[Pipeline] Learning: next_action={learning.next_action}", file=sys.stderr)
                        print(f"[Pipeline] Learning hooks stats: {learning_hooks.get_stats()}", file=sys.stderr)

                    # ====== PHASE 4.5: PATTERN EXTRACTION ENFORCEMENT ======
                    # Phase 1.3: Orchestrator MUST handle extracted patterns
                    if learning.pattern_extracted and learning.pattern_id:
                        print(f"[Pipeline] Pattern extracted: {learning.pattern_id}", file=sys.stderr)
                        # Track patterns in session for future use
                        if not hasattr(session, 'extracted_patterns'):
                            session.extracted_patterns = []
                        session.extracted_patterns.append({
                            "pattern_id": learning.pattern_id,
                            "iteration": iteration,
                            "score_delta": quality.overall_score - baseline_score_snapshot if quality else 0,
                        })

                    # ====== PHASE 4.6: PATTERN OUTCOME REPORTING (Phase 2 Self-Learning) ======
                    # Report outcome for previously applied pattern with improvement score
                    if hasattr(context, 'last_applied_pattern_id') and context.last_applied_pattern_id:
                        try:
                            score_improvement = quality.overall_score - baseline_score_snapshot if quality else 0
                            pattern_success = score_improvement > 0
                            outcome_result = report_pattern_outcome(
                                pattern_id=context.last_applied_pattern_id,
                                success=pattern_success,
                                improvement=score_improvement,  # Include actual improvement value
                                notes=f"Issue: {quality.primary_issue or 'unknown'}, Score: {quality.overall_score:.1f}"
                            )
                            print(f"[Pipeline] Pattern outcome reported: {context.last_applied_pattern_id}", file=sys.stderr)
                            print(f"[Pipeline]   success={pattern_success}, improvement={score_improvement:+.1f}", file=sys.stderr)
                            context.last_applied_pattern_id = None  # Clear after reporting
                        except Exception as e:
                            print(f"[Pipeline] WARN: Pattern outcome report failed: {e}", file=sys.stderr)

                    # ====== QUALITY GATE (Coordinator Decision) ======
                    # Phase 2 Enhancement: Use Quality Gate Coordinator for intelligent decision
                    # Note: gate_decision may already be set from parallel path above

                    if gate_decision is None:
                        # Sequential path - run Quality Gate now (with Learning recommendation)
                        print(f"[Pipeline] PHASE 5: Quality Gate (Sequential)", file=sys.stderr)
                        ctx = session_mgr.get_context_for_agents()

                        # Artifact-First: Compact prompt with artifact references
                        gate_prompt = f"""Interpret quality evaluation results for iteration {iteration}.

## Key Metrics
Score: {quality.overall_score:.1f} / Threshold: {request.quality_threshold} / Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}

## Artifacts (read if needed)
Scorecard: {scorecard_artifact_path}
Quality: {quality_artifact_path}

## Context
Iteration: {iteration}/{request.max_iterations} | Same Issue: {ctx.get('consecutive_same_issue', 0)}x | Best: {session.best_score:.1f} | Escape: {ctx.get('escape_level', 0)}

## Learning Agent Recommendation
Action: {learning.next_action if learning else 'unknown'} | Reasoning: {learning.suggested_modifications[0][:60] if learning and learning.suggested_modifications else 'None'}...

Decide: Is quality gate PASSED? What is the next action?"""

                        try:
                            gate_result = await self._run_agent(
                                self._quality_gate_coordinator,
                                gate_prompt,
                                context=context,
                                session=iter_session,  # Phase 2B-1: Stateless iterations
                                max_turns=3,  # Quality gate should be very fast
                                run_config=self._build_run_config(
                                    session=session,
                                    request=request,
                                    iteration=iteration,
                                    phase="quality_gate",
                                ),
                            )
                            gate_decision = gate_result.final_output
                            print(f"[Pipeline] Quality Gate: passed={gate_decision.passed}, next={gate_decision.next_action}", file=sys.stderr)
                            print(f"[Pipeline] Escape Level: {gate_decision.escape_level}, Reasoning: {gate_decision.reasoning[:50]}...", file=sys.stderr)

                        except Exception as e:
                            print(f"[Pipeline] WARN: Quality Gate Coordinator failed: {e}", file=sys.stderr)
                            # Fall back to simple quality check
                            gate_decision = None

                    # Sync escape_level to session IMMEDIATELY after Coordinator returns (both paths)
                    # This ensures downstream logic has access to the current escape level
                    if gate_decision:
                        session.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)
                        context.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)

                    # Determine if passed based on Coordinator or simple check
                    if gate_decision:
                        is_passed = gate_decision.passed
                        next_action = gate_decision.next_action
                    else:
                        is_passed = quality.passed or (learning and learning.next_action == 'complete')
                        next_action = learning.next_action if learning else ('complete' if quality.passed else 'iterate')

                    if is_passed or next_action == 'complete':
                        print(f"\n[Pipeline] ✓ QUALITY GATE PASSED at iteration {iteration}", file=sys.stderr)
                        session.status = SessionStatus.PASSED
                        break

                    # Handle technique switching - either from Coordinator OR SessionManager detection
                    should_switch = (
                        next_action == 'switch_technique'
                        or (gate_decision and gate_decision.escape_level >= 2)
                        or (learning and learning.next_action == 'switch_technique')
                        or session_mgr.should_switch_technique()
                    )

                    # Level 4: Request human guidance via HITL (Phase 2B-5)
                    if (
                        session.stuck_state.escape_level == EscapeLevel.REQUEST_GUIDANCE
                        or (gate_decision and gate_decision.escape_level >= 4)
                    ):
                        print(
                            f"[Pipeline] ESCAPE LEVEL 4: Requesting human guidance. "
                            f"Tried {len(session.stuck_state.techniques_tried)} techniques, "
                            f"best score: {session.best_score:.1f}",
                            file=sys.stderr,
                        )
                        if self._hitl:
                            l4_checkpoint = self._hitl.check_escalation(session)
                            if l4_checkpoint and l4_checkpoint.decision:
                                # Interactive: decision already captured
                                if hasattr(session, "hitl_history"):
                                    session.hitl_history.append(l4_checkpoint.to_dict())
                                if l4_checkpoint.decision == CheckpointDecision.ABORT:
                                    session.status = SessionStatus.PAUSED
                                    break
                                elif l4_checkpoint.decision == CheckpointDecision.SWITCH_TECHNIQUE:
                                    session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
                                    # Fall through to should_switch handling below
                                elif l4_checkpoint.decision == CheckpointDecision.CONTINUE:
                                    pass  # Continue loop
                                # Other decisions: continue loop
                            elif l4_checkpoint:
                                # Non-interactive: save and pause
                                session.pending_checkpoint = l4_checkpoint.to_dict()
                                session.status = SessionStatus.PAUSED
                                break
                            else:
                                # HITL disabled at level 4 — old behavior
                                session.status = SessionStatus.PAUSED
                                break
                        else:
                            # No HITL handler — old behavior
                            session.status = SessionStatus.PAUSED
                            break

                    if should_switch:
                        consecutive = session_mgr.issue_tracker.consecutive_same_issue
                        print(f"[Pipeline] STUCK DETECTED: same issue {consecutive}x - re-running Research Agent", file=sys.stderr)

                        # Phase 1.4: Reset stuck state when switching techniques
                        # This prevents old failure streaks from affecting the new approach
                        session_mgr.reset_for_technique_switch(f"switch_from_iter{iteration}")
                        self._pipeline_monitor.reset()  # Phase 2A-4

                        # Re-run Research Agent to find NEW approaches
                        switch_prompt = f"""Find a DIFFERENT approach for {request.effect_type.value}.

## Current Status
We've tried {len(session_mgr.techniques_tried)} techniques: {', '.join(session_mgr.techniques_tried)}
Same issue persisted {consecutive} times: {session_mgr.issue_tracker.last_primary_issue}

## What We Need
1. A fundamentally different technique (not just param tweaks)
2. Alternative Blender approaches (e.g., shader-based vs simulation-based)
3. Reference to working examples if available

DO NOT suggest: {', '.join(session_mgr.techniques_tried)}"""

                        switch_research_hooks = create_research_hooks()
                        try:
                            research_result = await self._run_agent(
                                self._research_agent,
                                switch_prompt,
                                context=context,
                                session=iter_session,  # Phase 2B-1: Stateless iterations
                                hooks=switch_research_hooks,
                                max_turns=4,  # Bounded 3-stage: retrieve → follow-up → synthesize
                                run_config=self._build_run_config(
                                    session=session,
                                    request=request,
                                    iteration=iteration,
                                    phase="switch_research",
                                ),
                            )
                            # Phase 3: Structured output
                            switch_research: Optional[ResearchOutput] = research_result.final_output
                            if switch_research:
                                research_text = f"""## Research Summary (Technique Switch)
Recommended Approach: {switch_research.recommended_approach}
Key Parameters: {switch_research.key_parameters}
Warnings: {'; '.join(switch_research.warnings) if switch_research.warnings else 'None'}"""
                                # Add new alternatives
                                for approach in switch_research.alternative_approaches:
                                    if approach and approach not in session.alternative_approaches:
                                        session.alternative_approaches.append(approach)
                        except LoopDetectedError as e:
                            print(f"[Pipeline] WARN: Technique switch research loop: {e}", file=sys.stderr)
                            # Keep existing research_text if new research loops
                        except OutputGuardrailTripwireTriggered as e:
                            # S1-1 FIX: Fail-closed — abort technique switch, revert stuck state reset.
                            # The pipeline will continue with existing research_text and try
                            # the next iteration with param modifications instead of a switch.
                            print(
                                f"[Pipeline] FAIL-CLOSED: Technique-switch research guardrail tripped: {e}. "
                                f"Aborting technique switch; reverting to param modification path.",
                                file=sys.stderr,
                            )
                            session_mgr.issue_tracker.consecutive_same_issue = consecutive  # revert reset
                            should_switch = False  # prevent downstream switch logic from firing
                        else:
                            print(f"[Pipeline] New research: {research_text[:80]}...", file=sys.stderr)
                        print(f"[Pipeline] Switch research hooks stats: {switch_research_hooks.get_stats()}", file=sys.stderr)

                    # ====== END OF ITERATION: WRITE ITERATION ARTIFACT ======
                    # Artifact-First: Write iteration snapshot for cross-iteration reference
                    cache_path = str(Path(artifact_mgr.artifact_dir) / "cache") if artifact_mgr else None
                    iteration_artifact_path = artifact_mgr.write_iteration_from_values(
                        iteration=iteration,
                        script_path=script.script_path if script else "",
                        technique_used=script.technique_used if script else "",
                        score=quality.overall_score if quality else 0.0,
                        passed=quality.passed if quality else False,
                        primary_issue=quality.primary_issue if quality else None,
                        render_path=execution.render_path if execution else None,
                        cache_path=cache_path,
                        parameter_changes=script.parameters_set if script and hasattr(script, 'parameters_set') else {},
                        escape_level=gate_decision.escape_level if gate_decision else 0,
                    )
                    print(f"[Pipeline] Iteration artifact: {iteration_artifact_path}", file=sys.stderr)

                    # ====== END OF ITERATION: COMPACT SESSION HISTORY ======
                    # Phase 2B-1: No compaction needed in stateless mode
                    if not use_stateless:
                        if sdk_session and hasattr(sdk_session, 'run_compaction'):
                            try:
                                await sdk_session.run_compaction({"force": True})
                                print(f"[Pipeline] Session history compacted after iteration {iteration}", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] WARN: Session compaction failed: {e}", file=sys.stderr)
                    else:
                        print(f"[Pipeline] Stateless mode — no compaction needed", file=sys.stderr)

                # End of iteration loop
                if session.status != SessionStatus.PASSED:
                    session.status = SessionStatus.MAX_ITERATIONS
                    print(f"\n[Pipeline] ✗ Max iterations reached. Best: {session.best_score:.1f}", file=sys.stderr)

        except Exception as e:
            import traceback
            print(f"[Pipeline] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            session.status = SessionStatus.FAILED
            session.current_issues.append(f"Pipeline error: {str(e)}")

        # Save session state
        self._persistence.save_session(session)
        session.update_timestamp()

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"PIPELINE COMPLETE: {session.status.value}", file=sys.stderr)
        print(f"Best Score: {session.best_score:.1f} (iteration {session.best_iteration})", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)

        return session

    async def resume_session(
        self,
        session_id: str,
        max_iterations_override: Optional[int] = None,
    ) -> SessionState:
        """
        Resume a paused or incomplete session using the code-based pipeline.

        Delegates to create_asset_pipeline() with resume_session_id, which:
        - Loads the existing session
        - Skips Phase 0 (Research) and Phase 0.5 (Technique Selection)
        - Reconstructs loop state from the last completed iteration
        - Continues the iteration loop from where it left off

        Args:
            session_id: ID of session to resume
            max_iterations_override: Override max_iterations (useful for extending
                sessions that hit MAX_ITERATIONS status)

        Returns:
            Updated SessionState
        """
        if not self._initialized:
            await self.initialize()

        # Load session to get the original request
        session = self._persistence.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return await self.create_asset_pipeline(
            request=session.request,
            resume_session_id=session_id,
            max_iterations_override=max_iterations_override,
        )

    async def close(self) -> None:
        """Clean up resources.

        Note: With the function_tool-based DocsExpert, there are no MCP
        connections to close. All tools run in-process.
        """
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return self._budget_tracker.get_status()

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List saved sessions."""
        sessions = self._persistence.list_sessions()

        if status:
            sessions = [s for s in sessions if s.get('status') == status.value]

        return sessions[:limit]


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_orchestrator: Optional[BlenderVFXOrchestrator] = None


async def get_orchestrator() -> BlenderVFXOrchestrator:
    """Get the global orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BlenderVFXOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator


def is_orchestrator_initialized() -> bool:
    """
    Check if the orchestrator singleton is initialized WITHOUT creating it.

    Use this for status checks to avoid triggering initialization from within
    MCP tool handlers (which causes asyncio task group conflicts).
    """
    return _orchestrator is not None and _orchestrator.is_initialized


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_vfx_asset(
    asset_name: str,
    description: str,
    effect_type: str = "pyro",
    reference_path: Optional[str] = None,
    resolution: int = 96,
    frame_end: int = 50,
    quality_threshold: float = 60.0,
    max_iterations: int = 5,
    enable_diagnostics: bool = False,
    diagnostic_log: Optional[str] = None,
) -> SessionState:
    """
    Convenience function to create a VFX asset.

    Args:
        asset_name: Name for the asset
        description: Description of what to create
        effect_type: Type of effect (pyro, explosion, fire, etc.)
        reference_path: Optional reference image path
        resolution: Blender simulation resolution
        frame_end: Animation end frame
        quality_threshold: Minimum quality score to pass
        max_iterations: Maximum iteration attempts
        enable_diagnostics: Enable diagnostic hooks for local AI-parseable traces
        diagnostic_log: Path for diagnostic log file (default: traces/diagnostic_{asset_name}.jsonl)

    Returns:
        SessionState with results
    """
    try:
        from models.shared_context import EffectType
    except ImportError:
        from .models.shared_context import EffectType

    request = AssetRequest(
        asset_name=asset_name,
        description=description,
        effect_type=EffectType(effect_type),
        reference_path=reference_path,
        resolution=resolution,
        frame_end=frame_end,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
    )

    orchestrator = await get_orchestrator()

    # Enable diagnostic hooks if requested
    if enable_diagnostics:
        log_path = diagnostic_log or str(
            Path(__file__).parent / "traces" / f"diagnostic_{asset_name}.jsonl"
        )
        orchestrator._diagnostic_hooks = DiagnosticHooks(
            log_file=log_path,
            verbose=True,
            track_patterns=True,
        )
        print(f"[Orchestrator] Diagnostics enabled: {log_path}", file=sys.stderr)

    # Phase 2: Use the pipeline-based orchestration with Coordinators
    result = await orchestrator.create_asset_pipeline(request)

    # Print diagnostic summary if enabled
    if enable_diagnostics and orchestrator._diagnostic_hooks:
        orchestrator._diagnostic_hooks.print_summary()
        print(f"\n[Orchestrator] Flow tracker analysis:", file=sys.stderr)
        flow_analysis = orchestrator._flow_tracker.analyze()
        print(f"  Total flows: {flow_analysis['total_flows']}", file=sys.stderr)
        print(f"  Successful: {flow_analysis['successful']}", file=sys.stderr)
        print(f"  Breakdowns: {flow_analysis['breakdowns']}", file=sys.stderr)
        if flow_analysis['breakdown_details']:
            print(f"  Breakdown details:", file=sys.stderr)
            for bd in flow_analysis['breakdown_details'][:3]:
                print(f"    - {bd['coordinator']}: {bd['decision']} -> {bd['parameters']}", file=sys.stderr)

    return result


async def resume_vfx_session(
    session_id: str,
    max_iterations: Optional[int] = None,
) -> SessionState:
    """
    Convenience function to resume a VFX session.

    Args:
        session_id: ID of session to resume
        max_iterations: Override max iterations (useful for extending
            sessions that hit MAX_ITERATIONS status)

    Returns:
        Updated SessionState
    """
    orchestrator = await get_orchestrator()
    return await orchestrator.resume_session(
        session_id,
        max_iterations_override=max_iterations,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing BlenderVFXOrchestrator...")
        print("-" * 60)

        # Test initialization
        orchestrator = BlenderVFXOrchestrator()
        await orchestrator.initialize()
        print(f"Initialized: {orchestrator.is_initialized}")

        # Test budget status
        status = orchestrator.get_budget_status()
        print(f"Budget: ${status['total_spent']:.2f} / ${status['total_remaining'] + status['total_spent']:.2f}")

        # Test session listing
        sessions = orchestrator.list_sessions()
        print(f"Sessions: {len(sessions)}")

        await orchestrator.close()
        print("Closed orchestrator")

    asyncio.run(test())
