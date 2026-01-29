"""
Blender VFX Orchestrator using OpenAI Agents SDK.

Main orchestrator that coordinates specialized agents to autonomously
generate and iterate on VFX assets until quality thresholds are met.

Key capabilities:
- Multi-agent coordination via handoffs
- Quality gate enforcement
- Stuck detection and escape strategies
- Budget enforcement ($20/month limit)
- Session persistence across context limits
"""

from __future__ import annotations

import asyncio
import time
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from agents import Agent, ModelSettings, Runner, handoff, trace, RunContextWrapper, ItemHelpers, SQLiteSession, function_tool
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
    create_quality_analyst_hooks,
    create_learning_agent_hooks,
    create_api_spec_hooks,
    create_code_writer_hooks,
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
# Phase 7: Spec-First Pipeline guardrails (API Hallucination Prevention)
from guardrails.api_spec_guardrails import (
    validate_api_spec,
    validate_code_against_spec,
)
# Artifact Gates: Deterministic validation of execution outputs (bake/render gates)
from guardrails.artifact_gates import (
    validate_execution_artifacts,
    format_gate_failure_for_diagnosis,
)
from agents.extensions import handoff_filters
# Removed unused handoff prompt imports - standalone agents don't use handoffs
from openai.types.shared import Reasoning


# =============================================================================
# STRUCTURED OUTPUT MODELS (Pydantic)
# =============================================================================
# These enable type-safe data passing between agents in the pipeline.

class ResearchOutput(BaseModel):
    """Output from Research Agent - provides starting parameters for script generation.

    Phase 3: Structured schema for deterministic research outputs.
    All fields must be populated from actual tool results, not invented.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        """Normalize title-case keys to snake_case.

        LLMs sometimes return keys like "Recommended Approach" instead of
        "recommended_approach" despite the JSON schema specifying snake_case.
        """
        if isinstance(data, dict):
            return {
                key.lower().replace(" ", "_").replace("-", "_"): value
                for key, value in data.items()
            }
        return data

    recommended_approach: str = Field(description="Best approach for the effect type (from documentation)")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameter values (from patterns or docs)")
    api_modules: List[str] = Field(default_factory=list, description="Blender API modules to use (e.g., bpy.types.FluidDomainSettings)")
    code_patterns: List[Dict[str, str]] = Field(default_factory=list, description="Proven patterns from library: [{pattern_id, issue, code_snippet}]")
    warnings: List[str] = Field(default_factory=list, description="Potential pitfalls to avoid (from knowledge base)")
    alternative_approaches: List[str] = Field(default_factory=list, description="Backup approaches if primary fails")
    doc_refs: List[str] = Field(default_factory=list, description="Blender 5.0 documentation references used (URLs or section names)")


class ScriptOutput(BaseModel):
    """Output from Script Writer - path to generated/modified script."""
    script_path: str = Field(description="Absolute path to the generated script")
    technique_used: str = Field(description="Technique/approach used in script generation")
    parameters_set: Dict[str, Any] = Field(default_factory=dict, description="Key parameters configured")
    validation_passed: bool = Field(default=True, description="Whether script validation passed")
    validation_errors: List[Any] = Field(default_factory=list, description="Any validation errors (strings or structured dicts)")


class ExecutionOutput(BaseModel):
    """Output from Executor - render path and execution status."""
    success: bool = Field(description="Whether Blender execution succeeded")
    render_path: Optional[str] = Field(default=None, description="Path to rendered output")
    vdb_path: Optional[str] = Field(default=None, description="Path to VDB volume data")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_seconds: float = Field(default=0.0, description="Total execution time")


class QualityOutput(BaseModel):
    """Output from Quality Analyst - evaluation results with feedback."""
    overall_score: float = Field(description="Quality score 0-100")
    passed: bool = Field(description="Whether quality threshold was met")
    primary_issue: Optional[str] = Field(default=None, description="Most critical issue to fix")
    issues: List[str] = Field(default_factory=list, description="All identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")
    vision_assessment: str = Field(default="", description="Detailed visual quality description")
    reference_similarity: Optional[float] = Field(default=None, description="Similarity to reference (if available)")


class LearningOutput(BaseModel):
    """Output from Learning Agent - experiment recorded, next action suggested."""
    experiment_recorded: bool = Field(default=True, description="Whether experiment was logged")
    pattern_extracted: bool = Field(default=False, description="Whether a new pattern was extracted")
    pattern_id: Optional[str] = Field(default=None, description="ID of extracted pattern")
    next_action: str = Field(description="Recommended next action: 'iterate', 'switch_technique', 'complete'")
    suggested_modifications: List[str] = Field(default_factory=list, description="Text descriptions of changes for context")
    parameter_modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete parameter changes as {param_name: new_value}, e.g., {'temperature': 3.0, 'density': 5.0}"
    )


# =============================================================================
# COORDINATOR AGENT OUTPUT MODELS (Phase 2: Agents-as-Tools Pattern)
# =============================================================================
# These models define structured outputs for the Coordinator Agent's decisions.
# The Coordinator is called at specific decision points in the pipeline, NOT for
# every step. Python still controls the iteration loop.

class TechniqueDecision(BaseModel):
    """Output from Coordinator for initial technique selection."""
    selected_technique: str = Field(description="The technique to use (e.g., 'mantaflow_fire', 'shader_based_volume')")
    reasoning: str = Field(description="Why this technique was selected")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Starting parameters for the technique")
    alternative_techniques: List[str] = Field(default_factory=list, description="Backup techniques if primary fails")
    research_summary: str = Field(default="", description="Summary of research findings")


class ModificationDecision(BaseModel):
    """Output from Coordinator for parameter modification strategy."""
    action: str = Field(description="Action to take: 'modify_params', 'modify_code', 'switch_technique', 'continue'")
    parameter_changes: Dict[str, Any] = Field(default_factory=dict, description="Concrete parameter changes (for modify_params)")
    code_change_description: Optional[str] = Field(default=None, description="Description of structural code changes needed (for modify_code)")
    new_technique: Optional[str] = Field(default=None, description="New technique if switching")
    reasoning: str = Field(default="See code_change_description or parameter_changes", description="Why this modification strategy was chosen")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this decision (0.0-1.0)")


class QualityDecision(BaseModel):
    """Output from Coordinator for quality gate interpretation."""
    passed: bool = Field(description="Whether quality gate is passed")
    should_continue: bool = Field(description="Whether to continue iterating")
    next_action: str = Field(description="Next action: 'complete', 'iterate', 'switch_technique', 'request_guidance'")
    escape_level: int = Field(ge=0, le=4, description="Current escape velocity level (0-4)")
    reasoning: str = Field(description="Explanation of the quality decision")


def compute_sica_utility(score: float, cost_usd: float, time_seconds: float) -> float:
    """
    SICA utility function for selecting best iteration.

    U = 0.5 * score_norm + 0.25 * (1 - cost/10) + 0.25 * (1 - time/300)
    score_norm is score / 100.
    """
    score_norm = max(0.0, min(1.0, score / 100.0))
    cost_component = 1.0 - min(1.0, max(0.0, cost_usd) / 10.0)
    time_component = 1.0 - min(1.0, max(0.0, time_seconds) / 300.0)
    utility = 0.5 * score_norm + 0.25 * cost_component + 0.25 * time_component
    return max(0.0, min(1.0, utility))

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

# Dynamic instruction wrappers for standalone agents (SDK dynamic instructions pattern)
from tools.dynamic_instructions import (
    dynamic_script_writer_standalone_instructions,
    dynamic_quality_analyst_standalone_instructions,
    dynamic_learning_agent_standalone_instructions,
)

from specialized_agents import (
    create_script_writer,
    create_executor,
    create_quality_analyst,
    create_learning_agent,
)
# DocsExpert now uses in-process function_tools instead of MCP client connections.
# This fixes the anyio TaskGroup conflict that previously blocked MCP tool handlers.
from specialized_agents.docs_expert import create_docs_expert
# Spec-First Pipeline agents (Phase 7: API Hallucination Prevention)
# These agents replace direct Script Writer usage with a two-phase approach:
# 1. API Spec Agent: Creates verified APISpec from Blender 5.0 docs
# 2. Code Writer Agent: Writes code using ONLY verified APIs from spec
from specialized_agents.api_spec_agent import create_api_spec_agent
from specialized_agents.code_writer_agent import create_code_writer_agent, format_api_spec_for_prompt
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

    Auto-compaction is disabled; compaction is triggered manually at the end of
    each pipeline iteration via `await session.run_compaction({"force": True})`.
    This preserves full context within an iteration while compressing between them.

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
        # Disable auto-compaction — we trigger manually between iterations
        # to preserve full context within each iteration
        should_trigger_compaction=lambda _: False,
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
# ORCHESTRATOR INSTRUCTIONS
# =============================================================================

ORCHESTRATOR_INSTRUCTIONS = """## ROLE
Coordinate specialized agents for autonomous VFX asset generation through iterative improvement.

## STATE MACHINE (STRICT SEQUENCE - NO DEVIATION)
```
iter=1: [Research tools] → Script Writer → Executor → Quality → Learning → GATE
iter>1: [pre_iteration_check] → Script Writer → Executor → Quality → Learning → GATE

GATE: passed OR max_iter → END | else → Script Writer (with mods)
```

CRITICAL: After each handoff returns, IMMEDIATELY proceed to next step. NO extra research between steps.

## AGENTS (one handoff at a time)
- delegate_to_script_writer: generate/modify Blender scripts (UCB1 technique selection)
- delegate_to_executor: run scripts, parse errors
- delegate_to_quality_analyst: evaluate quality (vision + reference comparison)
- delegate_to_learning_agent: record experiments, suggest fixes
- delegate_to_docs_expert: search Blender docs (Phase 0 research, when stuck)

## PHASE 0: RESEARCH (iter=1 only)
Use tools DIRECTLY (no handoffs):
1. blender_doc_search_bundle(effect_type, description, intent="create {effect_type} effect")
2. search_code_patterns("initial generation", effect_type)

Then handoff to Script Writer with research findings.

## PHASE 1: PRE-ITERATION CHECK (iter>1 only)
pre_iteration_research(current_issue, current_approach, iteration_history, effect_type)
→ warning_level: none|early|stuck
→ escape_action: continue|check_knowledge_then_modify|switch_technique_or_mine_docs

## PHASE 2: SCRIPT
iter=1: delegate_to_script_writer with research findings
iter>1: delegate_to_script_writer with FULL visual context:
"VISION ANALYSIS: {vision_assessment}
PRIMARY ISSUE: {primary_issue}
VISUAL DETAILS: {what it looks like, what's wrong, target appearance}
SUGGESTED FIXES: {suggestions}
Current script: {path}, Score: {score}"

## PHASE 3: EXECUTE
delegate_to_executor with script_path
If failed → parse error, modify, retry

## PHASE 4: EVALUATE
delegate_to_quality_analyst with:
- render_path, effect_type, iteration
- Request: analyze_with_vision + find_reference_images + compare_to_reference

## PHASE 5: LEARN
IF score_delta >= 5:
  extract_successful_pattern(original, modified, issue, improvement, effect_type)
  record_code_pattern(issue, code_snippet, effect_type, improvement)

ALWAYS: delegate_to_learning_agent with iteration, issue, params, score_before/after, visual observations

GATE: passed (score>=60, no critical) → complete | else → check escape_level, loop

## ESCAPE VELOCITY (0-4)
L0 NORMAL: apply mods, continue
L1 KNOWLEDGE_CHECK: query KB first, if failure_rate>50% → L2
L2 SWITCH_TECHNIQUE: NEW script, DIFFERENT technique, mark current failed
L3 MINE_DOCS: semantic search for novel approaches not in library
L4 REQUEST_GUIDANCE: report exhausted, request human input

Step-down: 2 consecutive +5 score iterations → escape_level can decrease

## TOOLS
Research: blender_doc_search_bundle, find_alternative_approaches
Patterns: search_code_patterns, record_code_pattern, get_pattern_code, report_pattern_outcome
Distillation: extract_successful_pattern, apply_pattern_to_script, analyze_script_for_patterns
Escape: pre_iteration_research, evaluate_escape_velocity, search_alternative_approaches

## QUALITY GATES
PASS: score>=60 AND no critical issues (ZERO_LIGHTS, BLACK_SCREEN)

## EARLY WARNING
2+ iter same issue OR score plateau → escape_level 1-2
3+ iter same issue OR plateau → escape_level 3-4

## BUDGET
Monthly: $20 ($10 vision, $8 docs, $2 buffer)
80% → warn, reduce profile | 100% → stop, return best

## OUTPUT
Each iteration: iter#, score, improvement, primary_issue, escape_level, next_action
Complete: final score, iter count, output paths, techniques tried, cost
"""


# =============================================================================
# COORDINATOR AGENT INSTRUCTIONS (Phase 2: Agents-as-Tools Pattern)
# =============================================================================
# The Coordinator Agent is called at DECISION POINTS only, not for every step.
# Python controls the pipeline sequence; the Coordinator provides intelligent
# decision-making for: technique selection, modification strategy, quality gates.

COORDINATOR_INSTRUCTIONS = """## ROLE
You are the Decision Coordinator for VFX asset generation. You are called at specific decision points to make intelligent choices about the pipeline direction.

## YOUR TOOLS (Agents wrapped as tools)
You have access to specialized agents as tools:
- research_approach: Research best approach for effect type (Research Agent)
- validate_blender_api: Validate Blender 5.0 API calls (API Validator)
- generate_script: Generate or modify Blender Python script (Script Writer)
- execute_script: Run Blender script and return results (Executor)
- evaluate_render: Evaluate render quality with vision (Quality Analyst)
- record_experiment: Record experiment and suggest fixes (Learning Agent)

## DECISION TYPES

### 1. TECHNIQUE_SELECTION (Iteration 1)
You are asked to select the best technique for an effect type.
- Use research_approach tool to gather information
- Consider effect type, description, and available patterns
- Return TechniqueDecision with selected_technique, reasoning, key_parameters

### 2. MODIFICATION_STRATEGY (Iteration 2+)
You are asked how to fix quality issues.
- Analyze current quality feedback (score, issues, suggestions)
- Check iteration history (what's been tried, what worked/failed)
- Decide: modify_params OR switch_technique
- Return ModificationDecision with action, parameter_changes, reasoning

### 3. QUALITY_GATE (After each iteration)
You are asked to interpret quality evaluation results.
- Check if score >= threshold AND no critical issues
- Determine escape_level based on iteration history
- Return QualityDecision with passed, should_continue, next_action

## CRITICAL RULES
1. Use tools ONCE per decision - don't loop or retry
2. Make decisive choices - avoid hedging or "it depends"
3. If stuck (same issue 3+ times), recommend switch_technique
4. Trust the Quality Analyst's assessment - don't second-guess scores
5. Provide CONCRETE parameters when recommending modifications

## ESCAPE VELOCITY LEVELS
- L0 NORMAL: Standard parameter tweaks
- L1 KNOWLEDGE_CHECK: Query patterns before modifying
- L2 SWITCH_TECHNIQUE: Different approach needed
- L3 MINE_DOCS: Search docs for novel approaches
- L4 REQUEST_GUIDANCE: Human intervention needed

Increase level when: same issue 2+ times, score plateau, repeated failures
Decrease level when: 2 consecutive improvements of +5 score
"""


# =============================================================================
# AGENT TOOL WRAPPERS WITH TURN LIMITS
# =============================================================================
# SDK Update (v0.6.9+): agent.as_tool() now supports max_turns parameter.
# Reference: https://openai.github.io/openai-agents-python/tools
# Pattern: agent.as_tool(tool_name="name", tool_description="desc", max_turns=X)

def create_agent_tool_wrappers(
    research_agent: Agent,
    script_agent: Agent,
    executor_agent: Agent,
    quality_agent: Agent,
    learning_agent: Agent,
) -> list:
    """
    Create agent-as-tool wrappers with explicit turn limits using native SDK pattern.

    SDK Pattern (v0.6.9+): Use agent.as_tool(max_turns=X) for turn-limited sub-agents.
    This is cleaner than wrapping in @function_tool with Runner.run().

    Turn limits are set based on task complexity:
    - Research: 4 turns (query docs, analyze, synthesize)
    - Script: 6 turns (may need iteration on generation)
    - Executor: 3 turns (execute, parse errors, report)
    - Quality: 4 turns (evaluate, analyze issues, report)
    - Learning: 3 turns (record, query knowledge, suggest)

    Returns:
        List of Tool objects for coordinator agents
    """
    return [
        research_agent.as_tool(
            tool_name="research_approach",
            tool_description="Research best approach for VFX effect type. Use at iteration 1 to find optimal technique, parameters, and alternatives. Returns research summary.",
            max_turns=4,
        ),
        script_agent.as_tool(
            tool_name="generate_script",
            tool_description="Generate or modify Blender Python script for VFX effect. Provide effect_type, description, and technique. Returns script_path, technique_used, parameters.",
            max_turns=6,
        ),
        executor_agent.as_tool(
            tool_name="execute_script",
            tool_description="Execute Blender script and render VFX asset. Provide script_path. Returns success, render_path, vdb_path, execution_time.",
            max_turns=3,
        ),
        quality_agent.as_tool(
            tool_name="evaluate_render",
            tool_description="Evaluate render quality using vision and metrics. Provide render_path, effect_type. Returns overall_score, passed, issues, suggestions.",
            max_turns=4,
        ),
        learning_agent.as_tool(
            tool_name="record_experiment",
            tool_description="Record experiment results and suggest next action. Provide iteration, score, issues, params. Returns next_action, parameter_modifications.",
            max_turns=3,
        ),
    ]


def create_research_tool_wrapper(research_agent: Agent):
    """
    Create a single research_approach tool wrapper with turn limit using native as_tool.

    Used by lightweight coordinators that only need research capability.

    Args:
        research_agent: Research Agent instance

    Returns:
        Tool object for research capability (turn-limited)
    """
    return research_agent.as_tool(
        tool_name="research_approach",
        tool_description="Research best approach for effect type. Returns research summary.",
        max_turns=4,
    )


# =============================================================================
# COORDINATOR AGENT FACTORY
# =============================================================================

def create_coordinator_agent(
    research_agent: Agent,
    script_agent: Agent,
    executor_agent: Agent,
    quality_agent: Agent,
    learning_agent: Agent,
) -> Agent:
    """
    Create the Coordinator Agent with all sub-agents wrapped as tools.

    SDK Pattern (v0.6.9+): Uses agent.as_tool(max_turns=X) for turn-limited
    sub-agents. This is the "Manager" pattern from SDK documentation.

    Args:
        research_agent: Research Agent instance
        script_agent: Script Writer Agent instance
        executor_agent: Executor Agent instance
        quality_agent: Quality Analyst Agent instance
        learning_agent: Learning Agent Agent instance

    Returns:
        Coordinator Agent with all sub-agents as tools (turn-limited)
    """
    from specialized_agents.api_validator import get_api_validator_as_tool

    # Create turn-limited agent tools using native as_tool(max_turns=X)
    agent_tools = create_agent_tool_wrappers(
        research_agent=research_agent,
        script_agent=script_agent,
        executor_agent=executor_agent,
        quality_agent=quality_agent,
        learning_agent=learning_agent,
    )

    # API validator already provides its own as_tool wrapper
    agent_tools.append(get_api_validator_as_tool())

    # Also include direct research tools for the Coordinator to use
    research_tools = [
        blender_doc_search_bundle,
        find_alternative_approaches,
        search_code_patterns,
        list_patterns_by_effect,
        pre_iteration_research_tool,
        evaluate_escape_velocity,
    ]

    return Agent[SharedContext](
        name="VFX Coordinator",
        instructions=COORDINATOR_INSTRUCTIONS,
        model=os.getenv("COORDINATOR_MODEL", get_config().preset.default_model),
        model_settings=ModelSettings(verbosity="medium"),
        tools=agent_tools + research_tools,
    )


def create_technique_selection_coordinator(
    research_agent: Agent,
) -> Agent:
    """
    Create a lightweight Coordinator for technique selection decisions only.

    This is called at iteration 1 to select the initial technique.
    Uses TechniqueDecision structured output.
    Phase 3: Output guardrail validates TechniqueDecision structure.

    Uses turn-limited research_approach wrapper (max_turns=4) instead of
    agent.as_tool() which cannot enforce turn limits.
    """
    # Create turn-limited research wrapper
    research_tool = create_research_tool_wrapper(research_agent)

    # Get config-based settings for coordinator
    config = get_config()
    settings = config.get_agent_settings("technique_coordinator")
    model_settings_kwargs = settings.to_model_settings()
    model_settings_kwargs["verbosity"] = "high" if settings.verbose else "medium"

    return Agent[SharedContext](
        name="Technique Selector",
        instructions="""You select the best technique for VFX effect generation.

## Your Task
Given an effect type and description, research and select the optimal technique.

## Process
1. Use research_approach tool to gather information about the effect type
2. Consider available code patterns and Blender API capabilities
3. Select the technique with highest success probability

## Output
Return a TechniqueDecision with:
- selected_technique: Name of technique (e.g., 'mantaflow_fire', 'shader_volume')
- reasoning: Why this technique is best
- key_parameters: Starting parameter values
- alternative_techniques: Backup techniques if primary fails
- research_summary: Summary of findings""",
        model=settings.model,
        model_settings=ModelSettings(**model_settings_kwargs),
        tools=[
            research_tool,  # Turn-limited wrapper (max_turns=4)
            blender_doc_search_bundle,
            search_code_patterns,
            list_patterns_by_effect,
        ],
        output_type=AgentOutputSchema(TechniqueDecision, strict_json_schema=False),
        # Phase 3: Validate TechniqueDecision has required fields
        output_guardrails=[validate_technique_decision],
    )


def create_modification_coordinator() -> Agent:
    """
    Create a lightweight Coordinator for modification strategy decisions.

    This is called at iteration 2+ to decide how to fix quality issues.
    Uses ModificationDecision structured output.
    Phase 3: Output guardrail validates ModificationDecision structure.
    """
    # Get config-based settings for coordinator
    config = get_config()
    settings = config.get_agent_settings("modification_coordinator")
    model_settings_kwargs = settings.to_model_settings()
    model_settings_kwargs["verbosity"] = "high" if settings.verbose else "medium"

    return Agent[SharedContext](
        name="Modification Strategist",
        instructions="""You decide how to fix quality issues in VFX renders.

## Your Task
Given quality feedback and iteration history, decide the modification strategy.

## Input Context
- Current score and target threshold
- Primary issue identified
- Previous iterations and what was tried
- Consecutive same-issue count

## Decision Logic (CHOOSE THE RIGHT ACTION)

### modify_params — Change numeric values, booleans, enums
Use when the issue is about tuning: resolution too low, density wrong, wrong material property.
Examples: resolution_max=128, vorticity=2.0, flow_behavior='INFLOW'

### modify_code — Rewrite script to fix structural/logic issues
Use when the issue CANNOT be fixed by changing a parameter value. This rewrites the script.
Examples of when to use modify_code:
- "Execution failed: no Fluid modifier found during bake" → needs object context setup
- "Fluid clips through glass mesh" → needs collision effector on glass object
- Missing objects in scene (lights, cameras, effectors)
- Wrong object relationships (parenting, modifiers on wrong object)
- Incorrect bake/simulation setup order
- Missing material assignments or shader node connections

### switch_technique — Abandon current approach entirely
Use when same issue persists 3+ times despite param AND code fixes.
This generates a completely new script from scratch.

### continue — No changes needed
Use when score is already improving and no intervention needed.

## Output
Return a ModificationDecision with:
- action: 'modify_params' | 'modify_code' | 'switch_technique' | 'continue'
- parameter_changes: Dict of {param_name: new_value} (for modify_params only)
- code_change_description: What structural changes to make (for modify_code only)
- new_technique: New technique name (for switch_technique only)
- reasoning: Why this strategy
- confidence: 0.0-1.0 confidence level

## CRITICAL: modify_params vs modify_code
If the issue mentions "failed", "error", "missing", "not found", "clips through",
"no modifier", "no object", or similar STRUCTURAL problems → use modify_code.
Parameter tweaking CANNOT fix structural issues. Do NOT use modify_params for these.

## CRITICAL FORMAT RULES for modify_params (NO PROSE)
- parameter_changes keys MUST be Blender API paths, e.g.:
  - FluidDomainSettings.resolution_max
  - FluidFlowSettings.flow_behavior
- Values MUST be: numbers, booleans, short enum tokens, or "__DELETE__"
- NO nested dicts, NO sentences, NO explanations in values
- Use "__DELETE__" to remove an invalid line (e.g., flow.velocity_factor)
""",
        model=settings.model,
        model_settings=ModelSettings(**model_settings_kwargs),
        tools=[
            search_code_patterns,
            pre_iteration_research_tool,
            evaluate_escape_velocity,
        ],
        output_type=AgentOutputSchema(ModificationDecision, strict_json_schema=False),
        # Phase 3: Validate action is valid and params provided when needed
        output_guardrails=[validate_modification_decision],
    )


def create_quality_gate_coordinator() -> Agent:
    """
    Create a lightweight Coordinator for quality gate decisions.

    This is called after quality evaluation to interpret results.
    Uses QualityDecision structured output.
    Phase 3: Output guardrail validates QualityDecision structure and consistency.
    """
    # Get config-based settings for coordinator
    config = get_config()
    settings = config.get_agent_settings("quality_gate_coordinator")
    model_settings_kwargs = settings.to_model_settings()
    model_settings_kwargs["verbosity"] = "high" if settings.verbose else "medium"

    return Agent[SharedContext](
        name="Quality Gate Judge",
        instructions="""You interpret quality evaluation results and decide next action.

## Your Task
Determine if quality gate is passed and what action to take next.

## Input Context
- Quality score and threshold
- Issues identified (critical vs minor)
- Iteration count and history
- Escape velocity level

## Decision Logic
1. PASSED: score >= threshold AND no critical issues (ZERO_LIGHTS, BLACK_SCREEN)
2. If passed: next_action = 'complete'
3. If not passed but improving: next_action = 'iterate'
4. If stuck (same issue 3+ times): increase escape_level, next_action = 'switch_technique'
5. If escape_level >= 4: next_action = 'request_guidance'

## Critical Issues (auto-fail)
- ZERO_LIGHTS_ACTIVE
- BLACK_SCREEN
- WHITE_SCREEN
- CLIPPING_ARTIFACTS

## Output
Return a QualityDecision with:
- passed: Whether quality gate is passed
- should_continue: Whether to continue iterating
- next_action: 'complete' | 'iterate' | 'switch_technique' | 'request_guidance'
- escape_level: Current escape velocity (0-4)
- reasoning: Explanation of decision""",
        model=settings.model,
        model_settings=ModelSettings(**model_settings_kwargs),
        tools=[
            evaluate_escape_velocity,
        ],
        output_type=AgentOutputSchema(QualityDecision, strict_json_schema=False),
        # Phase 3: Validate decision consistency (e.g., passed=True must have next_action='complete')
        output_guardrails=[validate_quality_decision],
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
        self._orchestrator: Optional[Agent] = None
        self._script_writer: Optional[Agent] = None
        self._executor: Optional[Agent] = None
        self._quality_analyst: Optional[Agent] = None
        self._learning_agent: Optional[Agent] = None
        self._docs_expert: Optional[Agent] = None

        # Standalone agents for code-based orchestration (no handoffs)
        self._research_agent: Optional[Agent] = None
        self._script_agent_standalone: Optional[Agent] = None
        self._executor_agent_standalone: Optional[Agent] = None
        self._quality_agent_standalone: Optional[Agent] = None
        self._learning_agent_standalone: Optional[Agent] = None

        # Phase 2: Coordinator agents for decision points (agents-as-tools pattern)
        # These agents are called at specific decision points, NOT for every step
        self._technique_coordinator: Optional[Agent] = None
        self._modification_coordinator: Optional[Agent] = None
        self._quality_gate_coordinator: Optional[Agent] = None

        # Phase 7: Spec-First Pipeline agents (API Hallucination Prevention)
        # Two-phase approach: API Spec Agent → Code Writer Agent
        self._api_spec_agent: Optional[Agent] = None
        self._code_writer_agent: Optional[Agent] = None
        self._use_spec_first_pipeline: bool = True  # Feature flag for gradual rollout

        self._budget_tracker: BudgetTracker = get_budget_tracker()
        self._persistence: SessionPersistence = get_persistence()
        self._initialized = False

        # Config system for preset-based settings
        self._config: AgentConfigManager = get_config()
        print(f"[Orchestrator] Using preset: {self._config.preset.name} ({self._config.preset.description})", file=sys.stderr)

        # Local tracing for AI-parseable analysis
        # Tracks Coordinator → modify_script communication flow
        self._flow_tracker = CommunicationFlowTracker(
            log_file=str(Path(__file__).parent / "traces" / "communication_flow.jsonl")
        )
        self._diagnostic_hooks: Optional[DiagnosticHooks] = None

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

    async def _run_spec_first_pipeline(
        self,
        effect_type: str,
        technique: str,
        request: "AssetRequest",
        context: SharedContext,
        sdk_session: Optional[SessionABC] = None,
    ) -> ScriptOutput:
        """
        Run the Spec-First Pipeline for API hallucination prevention.

        This two-phase approach makes hallucinated attributes structurally impossible:
        1. API Spec Agent: Creates verified APISpec from Blender 5.0 docs
        2. Code Writer Agent: Writes code using ONLY verified APIs from spec

        The guardrails enforce that:
        - Every attribute in the spec has a valid doc_ref
        - Generated code only uses attributes from the spec

        Args:
            effect_type: VFX effect type (pyro, explosion, etc.)
            technique: Selected technique (mantaflow_smoke, etc.)
            request: Asset generation request
            context: Shared context for agents
            sdk_session: SDK session for conversation persistence

        Returns:
            ScriptOutput with the generated script path and metadata

        SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md
        """
        print(f"[Spec-First] Starting API Spec → Code Writer pipeline", file=sys.stderr)

        # ====== PHASE 1.0: DETERMINISTIC BUNDLE CALL ======
        # Call bundle programmatically BEFORE running the agent
        # This removes reliance on model compliance for bundle-first discipline
        print(f"[Spec-First] Phase 1.0: Deterministic bundle call", file=sys.stderr)

        try:
            # Use bundle_search_impl (direct callable) instead of blender_doc_search_bundle (FunctionTool)
            bundle_results = bundle_search_impl(
                effect_type=effect_type,
                description=request.description,
                intent=f"create {effect_type} effect with {technique}",
                domain="Mantaflow",
                max_results=6
            )
            print(f"[Spec-First] Bundle results loaded ({len(bundle_results)} chars)", file=sys.stderr)
        except Exception as e:
            print(f"[Spec-First] WARNING: Bundle call failed: {e}", file=sys.stderr)
            bundle_results = "{}"  # Empty JSON fallback

        # Create hooks for each agent
        api_spec_hooks = create_api_spec_hooks()
        code_writer_hooks = create_code_writer_hooks()

        # ====== PHASE 1.A: API SPEC AGENT ======
        # Creates verified API specification from Blender 5.0 docs
        # Bundle results are pre-loaded - agent only needs targeted searches for gaps
        print(f"[Spec-First] Phase 1.A: API Spec Agent (with pre-loaded bundle)", file=sys.stderr)

        spec_prompt = f"""Create a VERIFIED API specification for {effect_type} effect.

## Effect Type: {effect_type}
## Technique: {technique}
## Description: {request.description}

## PRE-LOADED BUNDLE RESULTS (from blender_doc_search_bundle)
The bundle search has already been executed. Use these results as your PRIMARY source:

```json
{bundle_results}
```

## YOUR TASK
1. Extract attributes and doc_refs from the bundle results above
2. ONLY use semantic_search_blender_docs for specific attributes NOT found in bundle (max 4 calls)
3. Construct doc_ref as: blender_python_reference_5_0/bpy.types.{{CLASS}}.html#{{ATTRIBUTE}}
4. Output the complete APISpec with all verified attributes

Key parameters needed for {effect_type}:
- Domain: resolution_max, domain_type, use_noise, noise_strength, vorticity
- Flow: flow_type, flow_behavior, temperature, density, velocity_normal
- Scene: frame_start, frame_end

Extract from bundle first, then fill gaps with targeted searches."""

        try:
            spec_result = await Runner.run(
                self._api_spec_agent,
                spec_prompt,
                context=context,
                session=sdk_session,
                hooks=api_spec_hooks,
                max_turns=8
            )
            api_spec: APISpec = spec_result.final_output
            print(f"[Spec-First] API Spec created: {len(api_spec.domain_attributes)} domain attrs, "
                  f"{len(api_spec.flow_attributes)} flow attrs, {len(api_spec.ops)} ops", file=sys.stderr)

            # Store the API spec in context for the Code Writer guardrail
            context.api_spec = api_spec

        except Exception as e:
            print(f"[Spec-First] ERROR: API Spec Agent failed: {e}", file=sys.stderr)
            # Fall back to the original Script Writer
            print(f"[Spec-First] Falling back to original Script Writer", file=sys.stderr)
            return await self._run_original_script_writer(
                effect_type, technique, request, context, sdk_session
            )

        print(f"[Spec-First] API Spec hooks stats: {api_spec_hooks.get_stats()}", file=sys.stderr)

        # ====== PHASE 1.B: CODE WRITER AGENT ======
        # Writes code using ONLY verified APIs from the spec
        print(f"[Spec-First] Phase 1.B: Code Writer Agent", file=sys.stderr)

        # Format the API spec for the Code Writer prompt
        spec_text = format_api_spec_for_prompt(api_spec)

        code_prompt = f"""Write a Blender Python script for {effect_type} effect using ONLY the verified APIs below.

{spec_text}

## Asset Parameters
- Asset Name: {request.asset_name}
- Effect Type: {effect_type}
- Technique: {technique}
- Description: {request.description}
- Resolution: {request.resolution}
- Frames: {request.frame_start}-{request.frame_end}

## CRITICAL RULES
1. ONLY use attributes listed in the APISpec above
2. Use variable names: dset for domain_settings, fset for flow_settings
3. Copy attribute names EXACTLY (case sensitive)
4. If an attribute you need is not in the spec, work without it

Call write_script to generate the script, then validate_script to check it.
Return the VerifiedScriptOutput with script_path and apis_used."""

        try:
            code_result = await Runner.run(
                self._code_writer_agent,
                code_prompt,
                context=context,
                session=sdk_session,
                hooks=code_writer_hooks,
                max_turns=8
            )
            verified_output: VerifiedScriptOutput = code_result.final_output

            # Convert VerifiedScriptOutput to ScriptOutput for pipeline compatibility
            script = ScriptOutput(
                script_path=verified_output.script_path,
                technique_used=verified_output.technique_used,
                parameters_set=verified_output.parameters_set,
                validation_passed=verified_output.validation_passed,
                validation_errors=verified_output.validation_errors,
            )
            print(f"[Spec-First] Script generated: {script.script_path}", file=sys.stderr)
            print(f"[Spec-First] APIs used: {verified_output.apis_used}", file=sys.stderr)

        except Exception as e:
            print(f"[Spec-First] ERROR: Code Writer Agent failed: {e}", file=sys.stderr)
            # Create error output
            script = ScriptOutput(
                script_path="",
                technique_used=technique,
                parameters_set={},
                validation_passed=False,
                validation_errors=[f"Spec-First pipeline failed: {e}"],
            )

        print(f"[Spec-First] Code Writer hooks stats: {code_writer_hooks.get_stats()}", file=sys.stderr)
        return script

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
            script_result = await Runner.run(
                self._script_agent_standalone,
                script_prompt,
                context=context,
                session=sdk_session,
                hooks=script_hooks,
                max_turns=15
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
        Initialize all specialized agents and create the orchestrator.

        Must be called before create_asset().

        All 5 specialized agents are initialized synchronously. DocsExpert uses
        in-process function_tools (extracted from blender-manual MCP server),
        which avoids the anyio TaskGroup conflicts that blocked MCP tool handlers.

        IMPORTANT: Sub-agents have handoffs back to the orchestrator to maintain
        the iteration loop. Without return handoffs, the run ends when any
        sub-agent outputs a message.
        """
        if self._initialized:
            return

        print("[Orchestrator] Initializing all 5 specialized agents...", file=sys.stderr)

        # Step 1: Create orchestrator first (without sub-agent handoffs yet)
        # This breaks the circular dependency
        orchestrator_tools = [
            # Proactive research tools (Strategy 3)
            pre_iteration_research_tool,
            evaluate_escape_velocity,
            search_alternative_approaches,
            # Semantic docs tools (Strategy 1)
            blender_doc_search_bundle,
            find_alternative_approaches,
            # Code pattern tools (Strategy 4)
            record_code_pattern,
            search_code_patterns,
            get_pattern_code,
            report_pattern_outcome,
            get_pattern_library_stats,
            list_patterns_by_effect,
            # Knowledge distillation tools (Strategy 2)
            extract_successful_pattern,
            apply_pattern_to_script,
            analyze_script_for_patterns,
            compare_scripts,
        ]

        # Create a temporary orchestrator (will be replaced with full version)
        temp_orchestrator = Agent[SharedContext](
            name="Blender VFX Orchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=os.getenv("ORCHESTRATOR_MODEL", get_config().preset.default_model),
            model_settings=ModelSettings(verbosity="medium"),
            tools=orchestrator_tools,
        )

        # =============================================================
        # DEPRECATED: Handoff-based agents (Steps 2-4)
        # =============================================================
        # These handoff-based agents are DEPRECATED as of v3.3.0.
        # Use create_asset_pipeline() for new asset generation.
        #
        # Kept ONLY for resume_session() backwards compatibility.
        # TODO: Create resume_session_pipeline() and remove this section.
        #
        # Step 2: Create sub-agents with handoffs back to the orchestrator
        # This allows the iteration loop to continue after each specialist finishes
        # IMPORTANT: Use remove_all_tools filter to strip reasoning items from history,
        # which prevents gpt-5.2 API error about "reasoning item without following item"
        return_to_orchestrator = handoff(
            temp_orchestrator,
            tool_name_override="return_to_orchestrator",
            tool_description_override="Return control to the Orchestrator to continue the VFX generation pipeline. ALWAYS use this when you have completed your task.",
            input_filter=handoff_filters.remove_all_tools
        )

        # Create specialized handoffs with explicit NEXT_ACTION directives
        # This tells the orchestrator exactly what to do next after each agent returns
        return_from_script_writer = handoff(
            temp_orchestrator,
            tool_name_override="script_complete_execute_next",
            tool_description_override="Script generation complete. Use this to signal the orchestrator to IMMEDIATELY delegate_to_executor to run the script. Do NOT research.",
            input_filter=handoff_filters.remove_all_tools
        )

        return_from_executor = handoff(
            temp_orchestrator,
            tool_name_override="execution_complete_evaluate_next",
            tool_description_override="Script execution complete. Use this to signal the orchestrator to IMMEDIATELY delegate_to_quality_analyst to evaluate the render. Do NOT research.",
            input_filter=handoff_filters.remove_all_tools
        )

        return_from_quality_analyst = handoff(
            temp_orchestrator,
            tool_name_override="evaluation_complete_learn_next",
            tool_description_override="Quality evaluation complete. Use this to signal the orchestrator to IMMEDIATELY delegate_to_learning_agent to record results. Do NOT research.",
            input_filter=handoff_filters.remove_all_tools
        )

        return_from_learning_agent = handoff(
            temp_orchestrator,
            tool_name_override="learning_complete_decide_next",
            tool_description_override="Learning recorded. Use this to signal the orchestrator to CHECK if quality passed. If passed, output final result. If not, delegate_to_script_writer with modifications.",
            input_filter=handoff_filters.remove_all_tools
        )

        # Create all 5 specialized agents with specific return handoffs
        # NOTE: Handoff agents use STATIC instructions (use_dynamic_instructions=False)
        # because we need to append handoff-specific text. Dynamic instructions are
        # used by the STANDALONE agents in create_asset_pipeline().
        base_sw = create_script_writer(use_dynamic_instructions=False)
        self._script_writer = base_sw.clone(
            handoffs=[return_from_script_writer],
            instructions=base_sw.instructions + """

CRITICAL: After generating and validating a script, use script_complete_execute_next to hand off.
This signals the orchestrator to IMMEDIATELY run the script - no more research needed."""
        )

        base_exec = create_executor()
        self._executor = base_exec.clone(
            handoffs=[return_from_executor],
            instructions=base_exec.instructions + """

CRITICAL: After executing a script (success or failure), use execution_complete_evaluate_next to hand off.
This signals the orchestrator to IMMEDIATELY evaluate the render - no more research needed."""
        )

        base_qa = create_quality_analyst(use_dynamic_instructions=False)
        self._quality_analyst = base_qa.clone(
            handoffs=[return_from_quality_analyst],
            instructions=base_qa.instructions + """

CRITICAL: After evaluating render quality, use evaluation_complete_learn_next to hand off.
This signals the orchestrator to IMMEDIATELY record learnings - no more research needed."""
        )

        base_la = create_learning_agent(use_dynamic_instructions=False)
        self._learning_agent = base_la.clone(
            handoffs=[return_from_learning_agent],
            instructions=base_la.instructions + """

CRITICAL: After recording experiments, use learning_complete_decide_next to hand off.
This signals the orchestrator to decide: if quality passed, finish. If not, modify script."""
        )

        base_docs = create_docs_expert()
        self._docs_expert = base_docs.clone(
            handoffs=[return_to_orchestrator],
            instructions=base_docs.instructions + """

CRITICAL: After searching documentation, use return_to_orchestrator to hand off."""
        )

        # Step 3: Build handoffs list with all 5 sub-agents
        # Use remove_all_tools filter on all handoffs to prevent reasoning item issues
        handoffs_list = [
            handoff(
                self._script_writer,
                tool_name_override="delegate_to_script_writer",
                tool_description_override="Delegate to Script Writer for generating or modifying Blender scripts",
                input_filter=handoff_filters.remove_all_tools
            ),
            handoff(
                self._executor,
                tool_name_override="delegate_to_executor",
                tool_description_override="Delegate to Executor for running Blender scripts and handling errors",
                input_filter=handoff_filters.remove_all_tools
            ),
            handoff(
                self._quality_analyst,
                tool_name_override="delegate_to_quality_analyst",
                tool_description_override="Delegate to Quality Analyst for evaluating render quality",
                input_filter=handoff_filters.remove_all_tools
            ),
            handoff(
                self._learning_agent,
                tool_name_override="delegate_to_learning_agent",
                tool_description_override="Delegate to Learning Agent for fix suggestions and experiment recording",
                input_filter=handoff_filters.remove_all_tools
            ),
            handoff(
                self._docs_expert,
                tool_name_override="delegate_to_docs_expert",
                tool_description_override="Delegate to Docs Expert for searching Blender documentation (use when stuck)",
                input_filter=handoff_filters.remove_all_tools
            ),
        ]

        # Step 4: Create the final orchestrator with handoffs to sub-agents
        self._orchestrator = Agent[SharedContext](
            name="Blender VFX Orchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=os.getenv("ORCHESTRATOR_MODEL", get_config().preset.default_model),
            model_settings=ModelSettings(verbosity="medium"),
            handoffs=handoffs_list,
            tools=orchestrator_tools,
        )

        # Step 5: Create STANDALONE agents for code-based orchestration
        # These have NO handoffs - Python controls the pipeline sequence directly
        # Using structured outputs (output_type) for type-safe data passing between agents
        research_model, research_settings = self._get_model_settings("research_agent")
        self._research_agent = Agent[SharedContext](
            name="Research Agent",
            instructions="""ROLE: Blender documentation research specialist.
INPUTS: effect_type, description.
TOOLS: blender_doc_search_bundle, search_code_patterns, list_patterns_by_effect.
TURNS: MAX 4.

## Tool Order
T1: blender_doc_search_bundle(effect_type, description, intent="create {effect_type} effect")
T2: list_patterns_by_effect(effect_type) OR search_code_patterns(issue="", effect_type=effect_type)
T3: Return ResearchOutput

## Output Contract (ResearchOutput)
- recommended_approach: str (from docs search - REQUIRED)
- key_parameters: dict (from patterns or docs)
- api_modules: list[str] (from doc bundle related_apis)
- code_patterns: list[{pattern_id, issue, code_snippet}] (from pattern search)
- warnings: list[str] (from doc bundle warnings or patterns)
- alternative_approaches: list[str] (optional; leave empty if not found)
- doc_refs: list[str] (Blender 5.0 doc references - REQUIRED)

STOP after T3. Do NOT retry tools. Return structured output only.""",
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

        # Executor with structured output
        base_executor = create_executor()
        executor_model, executor_settings = self._get_model_settings("executor")
        self._executor_agent_standalone = Agent[SharedContext](
            name="Executor",
            instructions=base_executor.instructions + """

## Output Requirements
After executing the script, return a structured ExecutionOutput with:
- success: Whether Blender executed without errors
- render_path: Path to the rendered output image/sequence
- vdb_path: Path to VDB volume data (if generated)
- error_message: Error details if execution failed
- execution_time_seconds: How long execution took""",
            model=executor_model,
            model_settings=executor_settings,
            output_type=AgentOutputSchema(ExecutionOutput, strict_json_schema=False),
            tools=base_executor.tools,
        )

        # Quality Analyst with structured output (LLM-as-judge pattern)
        # Physics observation happens through tools (observe_physics_anomaly, get_physics_patterns)
        # Phase 3: Input/output guardrails for validation
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
            tools=base_learning_standalone.tools,
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

        # =============================================================
        # Phase 7: Spec-First Pipeline Agents (API Hallucination Prevention)
        # =============================================================
        # Two-phase approach that makes API hallucinations structurally impossible:
        # 1. API Spec Agent: Creates verified APISpec from Blender 5.0 docs
        # 2. Code Writer Agent: Writes code using ONLY verified APIs from spec
        #
        # SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md

        if self._use_spec_first_pipeline:
            print("[Orchestrator] Creating Phase 7 Spec-First Pipeline agents...", file=sys.stderr)

            # API Spec Agent - creates verified API specifications
            # Has output guardrail that rejects specs with missing/invalid doc_refs
            self._api_spec_agent = create_api_spec_agent(
                model="gpt-5.2",
                use_high_reasoning=True,
            )

            # Code Writer Agent - writes code using ONLY verified APIs
            # Has output guardrail that validates code against the APISpec
            self._code_writer_agent = create_code_writer_agent(
                model="gpt-5.2",
                use_high_reasoning=True,
            )

            print("[Orchestrator] Spec-First Pipeline agents ready", file=sys.stderr)

        self._initialized = True
        agent_count = "5 agents + 5 standalone + 3 coordinators"
        if self._use_spec_first_pipeline:
            agent_count += " + 2 spec-first"
        print(f"[Orchestrator] Initialization complete ({agent_count} ready)", file=sys.stderr)

    async def create_asset(self, request: AssetRequest) -> SessionState:
        """
        DEPRECATED: Use create_asset_pipeline() instead.

        This method uses the handoff-based architecture which has been superseded
        by the code-based pipeline with Coordinator agents (Phase 2 of Architecture
        Optimization Plan).

        The handoff pattern is problematic because:
        1. Handoffs transfer control completely to sub-agents
        2. Relies on LLM instruction-following for workflow
        3. Cannot enforce mechanical guardrails

        Use create_asset_pipeline() for:
        - Deterministic Python-controlled workflow
        - Coordinator agents for intelligent decisions
        - Mechanical enforcement via hooks

        Args:
            request: Asset generation request parameters

        Returns:
            SessionState with final results
        """
        import warnings
        warnings.warn(
            "create_asset() is deprecated. Use create_asset_pipeline() instead. "
            "The handoff-based architecture has been superseded by the code-based "
            "pipeline with Coordinator agents (Phase 2).",
            DeprecationWarning,
            stacklevel=2
        )
        if not self._initialized:
            await self.initialize()

        # Check budget before starting
        if not self._budget_tracker.can_afford_evaluation():
            raise RuntimeError(
                f"Budget exhausted. Monthly limit: ${self._budget_tracker.monthly_limit}"
            )

        # Create session
        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Build initial prompt
        prompt = self._build_generation_prompt(request, context)

        # Run orchestrator (autonomous iteration loop)
        # gpt-5.2 with high reasoning effort needs more turns than default 10
        # Using trace() for end-to-end observability across iterations
        try:
            with trace(f"VFX Asset: {request.asset_name}"):
                result = await Runner.run(
                    self._orchestrator,
                    prompt,
                    context=context,  # Typed SharedContext for RunContextWrapper access
                    max_turns=25  # Increased for gpt-5.2 reasoning
                )

            # Parse and update session state from result
            session = self._parse_result(result, context)

        except Exception as e:
            # Handle errors gracefully
            context.session.status = SessionStatus.FAILED
            context.session.current_issues.append(f"Orchestration error: {str(e)}")
            session = context.session

        # Save session state
        self._persistence.save_session(session)

        return session

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

                if not is_resuming:
                    # ====== PHASE 0: RESEARCH (once at start) ======
                    print("[Pipeline] PHASE 0: Research", file=sys.stderr)
                    research_prompt = f"""Research the best approach for creating a {request.effect_type.value} VFX effect.

Effect Type: {request.effect_type.value}
Description: {request.description}
Reference: {request.reference_path or "None"}

Research documentation, patterns, and APIs to find the optimal starting approach."""

                    # Create research hooks for Phase 0
                    phase0_research_hooks = create_research_hooks()
                    research_output: Optional[ResearchOutput] = None
                    try:
                        research_result = await Runner.run(
                            self._research_agent,
                            research_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            hooks=phase0_research_hooks,
                            max_turns=4  # Phase 3: Aligned with prompt turn budget
                        )
                        # Phase 3: Structured output - ResearchOutput schema
                        research_output = research_result.final_output
                        if research_output:
                            # Build text summary from structured output for downstream use
                            research_text = f"""## Research Summary
Recommended Approach: {research_output.recommended_approach}
Key Parameters: {research_output.key_parameters}
API Modules: {', '.join(research_output.api_modules) if research_output.api_modules else 'None'}
Warnings: {'; '.join(research_output.warnings) if research_output.warnings else 'None'}
Doc Refs: {', '.join(research_output.doc_refs) if research_output.doc_refs else 'None'}"""
                        else:
                            research_text = "No research findings"
                    except LoopDetectedError as e:
                        # Research got stuck - use partial results
                        print(f"[Pipeline] WARN: Research loop detected: {e}", file=sys.stderr)
                        research_text = "Research incomplete due to loop - proceeding with default approach"
                    except OutputGuardrailTripwireTriggered as e:
                        # Research output failed validation (e.g., empty doc_refs) — non-fatal
                        print(f"[Pipeline] WARN: Research guardrail tripped: {e}", file=sys.stderr)
                        research_text = "Research output incomplete (guardrail) - proceeding with default approach"
                    except Exception as e:
                        # SDK wraps LoopDetectedError in UserError - check for it
                        if "Loop detected" in str(e) or "LoopDetectedError" in str(e):
                            print(f"[Pipeline] WARN: Research loop detected (wrapped): {e}", file=sys.stderr)
                            research_text = "Research incomplete due to loop - proceeding with default approach for this effect type"
                        else:
                            raise  # Re-raise other exceptions
                    print(f"[Pipeline] Research complete: {research_text[:80]}...", file=sys.stderr)
                    print(f"[Pipeline] Research hooks stats: {phase0_research_hooks.get_stats()}", file=sys.stderr)

                    # Store research in session for persistence and handoff to Script Writer
                    session.research_text = research_text

                    # Phase 3: Use structured output directly instead of regex parsing
                    if research_output and research_output.alternative_approaches:
                        for approach in research_output.alternative_approaches:
                            if approach and approach not in session.alternative_approaches:
                                session.alternative_approaches.append(approach)
                        print(f"[Pipeline] Found {len(session.alternative_approaches)} alternative approaches (structured)", file=sys.stderr)

                    # Artifact-First: Write research to file for downstream agents
                    research_artifact_path = None
                    if research_output:
                        research_artifact_path = artifact_mgr.write_research_from_output(
                            recommended_approach=research_output.recommended_approach,
                            key_parameters=research_output.key_parameters,
                            api_modules=research_output.api_modules,
                            warnings=research_output.warnings,
                            alternative_approaches=research_output.alternative_approaches,
                            doc_refs=research_output.doc_refs,
                        )
                        print(f"[Pipeline] Research artifact: {research_artifact_path}", file=sys.stderr)

                    # ====== PHASE 0.5: TECHNIQUE SELECTION (Coordinator Decision) ======
                    # Phase 2 Enhancement: Use Technique Coordinator to intelligently select
                    # the initial technique based on research findings.
                    print("[Pipeline] PHASE 0.5: Technique Selection (Coordinator)", file=sys.stderr)

                    # Artifact-First: Reference file instead of inline dump (compact prompt)
                    research_ref = f"Research artifact: {research_artifact_path}" if research_artifact_path else "No research artifact"
                    technique_prompt = f"""Select the best technique for creating a {request.effect_type.value} VFX effect.

## Research Summary
{research_ref}
Recommended: {research_output.recommended_approach if research_output else 'Default approach'}
Key params: {list(research_output.key_parameters.keys()) if research_output and research_output.key_parameters else 'None'}

## Effect Parameters
- Effect Type: {request.effect_type.value}
- Description: {request.description}
- Reference: {request.reference_path or "None"}
- Quality Threshold: {request.quality_threshold}

## Available Alternatives
{chr(10).join(f'- {a}' for a in session.alternative_approaches[:5]) if session.alternative_approaches else 'None identified'}

Select the optimal technique and provide starting parameters."""

                    try:
                        technique_result = await Runner.run(
                            self._technique_coordinator,
                            technique_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            max_turns=6  # Coordinators should be fast
                        )
                        selected_technique = technique_result.final_output
                        print(f"[Pipeline] Coordinator selected: {selected_technique.selected_technique}", file=sys.stderr)
                        print(f"[Pipeline] Reasoning: {selected_technique.reasoning[:60]}...", file=sys.stderr)

                        # Store in session for use by Script Writer
                        session.current_technique = selected_technique.selected_technique
                        if selected_technique.alternative_techniques:
                            for alt in selected_technique.alternative_techniques:
                                if alt not in session.alternative_approaches:
                                    session.alternative_approaches.append(alt)

                    except Exception as e:
                        print(f"[Pipeline] WARN: Technique Coordinator failed: {e}", file=sys.stderr)
                        # Fall back to default technique selection
                        selected_technique = None

                else:
                    # ====== RESUME: Skip Phase 0 and 0.5 ======
                    print("[Pipeline] RESUME: Skipping Phase 0/0.5 (already stored in session)", file=sys.stderr)
                    research_text = session.research_text or "Resumed session"
                    # selected_technique stays None — not needed for iteration 2+
                    # The guard `if selected_technique:` (Phase 1) handles this

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

                while iteration < effective_max_iterations:
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

                        # ====== PHASE 7: SPEC-FIRST PIPELINE (API Hallucination Prevention) ======
                        # Two-phase approach: API Spec Agent → Code Writer Agent
                        # This makes hallucinated attributes structurally impossible.
                        if self._use_spec_first_pipeline and self._api_spec_agent and self._code_writer_agent:
                            print(f"[Pipeline] Using SPEC-FIRST pipeline (Phase 7)", file=sys.stderr)
                            try:
                                script = await self._run_spec_first_pipeline(
                                    effect_type=request.effect_type.value,
                                    technique=technique_name,
                                    request=request,
                                    context=context,
                                    sdk_session=sdk_session,
                                )
                            except Exception as e:
                                print(f"[Pipeline] Spec-First pipeline failed: {e}", file=sys.stderr)
                                print(f"[Pipeline] Falling back to original Script Writer", file=sys.stderr)
                                # Fall back to original Script Writer
                                script = await self._run_original_script_writer(
                                    effect_type=request.effect_type.value,
                                    technique=technique_name,
                                    request=request,
                                    context=context,
                                    sdk_session=sdk_session,
                                )
                        else:
                            # Original Script Writer flow (deprecated but kept for compatibility)
                            print(f"[Pipeline] Using ORIGINAL Script Writer (Spec-First disabled)", file=sys.stderr)

                            # Build script prompt with Coordinator's technique selection
                            technique_guidance = ""
                            starting_params = {}
                            if selected_technique:
                                technique_guidance = f"""
## COORDINATOR SELECTED TECHNIQUE
Technique: {selected_technique.selected_technique}
Reasoning: {selected_technique.reasoning}
Starting Parameters: {json.dumps(selected_technique.key_parameters, indent=2) if selected_technique.key_parameters else 'None specified'}

YOU MUST USE THIS TECHNIQUE. The Coordinator has analyzed the research and selected this as optimal."""
                                starting_params = selected_technique.key_parameters or {}

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
                                script_result = await Runner.run(
                                    self._script_agent_standalone,
                                    script_prompt,
                                    context=context,
                                    session=sdk_session,  # Phase 4: SDK session for conversation persistence
                                    hooks=script_hooks,
                                    max_turns=15
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
                            and self._use_spec_first_pipeline
                        ):
                            consec = session_mgr.issue_tracker.consecutive_same_issue
                            print(
                                f"[Pipeline] EXECUTION FAILURE x{consec} -- forcing spec-first regeneration",
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
                                script = await self._run_spec_first_pipeline(
                                    effect_type=request.effect_type.value,
                                    technique=new_technique,
                                    request=request,
                                    context=context,
                                    sdk_session=sdk_session,
                                )
                                direct_modification_success = True
                                session_mgr.reset_for_technique_switch(new_technique)
                                print(f"[Pipeline] Spec-first regen succeeded: {script.script_path}", file=sys.stderr)
                            except Exception as e:
                                print(
                                    f"[Pipeline] Spec-first regen failed: {e}, falling back to Coordinator",
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

                                    modify_result_json = _modify_script_impl(
                                        script_path=previous_script.script_path,
                                        modifications=pattern_params,
                                        output_name=output_name
                                    )
                                    modify_result = json.loads(modify_result_json)

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
                            modify_result_json = _modify_script_impl(
                                script_path=previous_script.script_path,
                                modifications=learning.parameter_modifications,
                                output_name=output_name
                            )
                            modify_result = json.loads(modify_result_json)

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

                        # If no direct modifications (or they failed), consult Modification Coordinator
                        if not direct_modification_success:
                            # ====== PHASE 1.1: MODIFICATION STRATEGY (Coordinator Decision) ======
                            # Phase 2 Enhancement: Use Modification Coordinator to decide strategy
                            print(f"[Pipeline] PHASE 1.1: Modification Strategy (Coordinator)", file=sys.stderr)

                            mod_decision: Optional[ModificationDecision] = None
                            ctx = session_mgr.get_context_for_agents()
                            iteration_summary = session_mgr.get_iteration_summary()

                            mod_prompt = f"""Decide modification strategy for iteration {iteration}.

## Current State
- Current Score: {previous_score:.1f}
- Target Score: {request.quality_threshold}
- Primary Issue: {quality.primary_issue if quality else 'Unknown'}
- Consecutive Same Issue: {ctx.get('consecutive_same_issue', 0)}

## Quality Feedback
{quality.vision_assessment if quality else 'No assessment'}
Issues: {', '.join(quality.issues[:3]) if quality and quality.issues else 'None'}

## Iteration History
{iteration_summary}

## Techniques Tried
{', '.join(session_mgr.techniques_tried) if session_mgr.techniques_tried else 'None'}

## Available Alternatives
{', '.join(session.alternative_approaches[:3]) if session.alternative_approaches else 'None'}

Decide: modify_params, modify_code, OR switch_technique.
- modify_params: Provide CONCRETE parameter values for tuning issues.
- modify_code: Describe structural changes needed (missing objects, wrong setup, broken logic).
- switch_technique: When the current approach is fundamentally broken after 3+ attempts."""

                            try:
                                mod_result = await Runner.run(
                                    self._modification_coordinator,
                                    mod_prompt,
                                    context=context,
                                    session=sdk_session,  # Phase 4: SDK session for conversation persistence
                                    max_turns=4  # Coordinators should be fast
                                )
                                mod_decision = mod_result.final_output
                                print(f"[Pipeline] Coordinator decision: {mod_decision.action}", file=sys.stderr)
                                print(f"[Pipeline] Reasoning: {mod_decision.reasoning[:60]}...", file=sys.stderr)

                                # TECHNIQUE SWITCH: Route through spec-first pipeline for a fresh script
                                # This avoids the modify_script path which can only patch Config params
                                if mod_decision.action == 'switch_technique' and self._use_spec_first_pipeline:
                                    print(f"[Pipeline] TECHNIQUE SWITCH → Re-running spec-first pipeline", file=sys.stderr)

                                    # Pick new technique from alternatives
                                    new_technique = mod_decision.reasoning[:80] if mod_decision.reasoning else "alternative"
                                    untried = [a for a in session.alternative_approaches if a not in session_mgr.techniques_tried]
                                    if untried:
                                        new_technique = untried[0]

                                    try:
                                        script = await self._run_spec_first_pipeline(
                                            effect_type=request.effect_type.value,
                                            technique=new_technique,
                                            request=request,
                                            context=context,
                                            sdk_session=sdk_session,
                                        )
                                        direct_modification_success = True
                                        session_mgr.reset_for_technique_switch(new_technique)
                                    except Exception as e:
                                        print(f"[Pipeline] Spec-first switch failed: {e}, falling back to Script Writer", file=sys.stderr)
                                        # Fall through to existing Script Writer path

                                # MODIFY CODE: Route to Script Writer with structural fix instructions
                                # This uses write_script (full code gen) not modify_script (param swap)
                                if mod_decision.action == 'modify_code' and mod_decision.code_change_description and previous_script and previous_script.script_path:
                                    print(f"[Pipeline] MODIFY CODE → Structural script rewrite", file=sys.stderr)
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

## Instructions
1. Read the current script to understand the existing setup
2. Make the structural changes described above
3. Keep everything that's working — only fix what's broken
4. Use write_script to output the complete fixed script
5. Name the output: {request.asset_name}_script_iter{iteration}_codefix

## Common Structural Fixes
- Bake context: set domain as active_object + selected before bpy.ops.fluid.bake_data()
- Effectors: glass/solid meshes need Fluid modifier with fluid_type='EFFECTOR'
- Object context: bpy.ops calls need correct context override
- Missing depsgraph: call bpy.context.view_layer.depsgraph.update() after setup changes
- Material assignment: ensure materials are assigned to correct objects
- Frame range: align cache_frame_start/end with scene.frame_start/end"""

                                    try:
                                        code_fix_hooks = create_fallback_script_writer_hooks()
                                        code_fix_result = await Runner.run(
                                            self._script_agent_standalone,
                                            code_fix_prompt,
                                            context=context,
                                            session=sdk_session,
                                            hooks=code_fix_hooks,
                                            max_turns=8,
                                        )
                                        code_fix_output = code_fix_result.final_output
                                        if code_fix_output and code_fix_output.script_path:
                                            print(f"[Pipeline] Code fix SUCCESS: {code_fix_output.script_path}", file=sys.stderr)
                                            script = code_fix_output
                                            direct_modification_success = True
                                        else:
                                            print(f"[Pipeline] Code fix produced no script_path, falling back", file=sys.stderr)
                                    except Exception as e:
                                        print(f"[Pipeline] Code fix failed: {e}, falling back to Script Writer", file=sys.stderr)

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

                                    modify_result_json = _modify_script_impl(
                                        script_path=previous_script.script_path,
                                        modifications=mod_decision.parameter_changes,
                                        output_name=output_name
                                    )
                                    modify_result = json.loads(modify_result_json)

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

                        # If still no success, use Script Writer
                        if not direct_modification_success:
                            # Include quality feedback and learning suggestions
                            feedback_parts = []

                            # Add iteration history from SessionManager (prevents repeating failures)
                            if iteration_summary:
                                feedback_parts.append(iteration_summary)

                            # Add what we've learned about params
                            ctx = session_mgr.get_context_for_agents()
                            if ctx.get("params_that_helped"):
                                feedback_parts.append("## Parameters That Improved Scores")
                                for p, (v, d) in ctx["params_that_helped"].items():
                                    feedback_parts.append(f"- {p}={v} (+{d:.1f})")
                            if ctx.get("params_that_hurt"):
                                feedback_parts.append("## Parameters To AVOID (hurt scores)")
                                for p, (v, d) in ctx["params_that_hurt"].items():
                                    feedback_parts.append(f"- {p}={v} ({d:.1f})")
                            if ctx.get("stuck_issues"):
                                feedback_parts.append(f"## STUCK: These issues persist - try different approach")
                                feedback_parts.append(f"- {', '.join(ctx['stuck_issues'])}")

                            # Phase 2: Include knowledge base learnings from Phase 0.95
                            if kb_suggestions:
                                feedback_parts.append("## Knowledge Base Learnings (from previous experiments)")
                                for kb_entry in kb_suggestions[:3]:
                                    if isinstance(kb_entry, dict):
                                        kb_text = kb_entry.get('learning', kb_entry.get('text', str(kb_entry)))
                                    else:
                                        kb_text = str(kb_entry)
                                    feedback_parts.append(f"- {kb_text[:150]}...")

                            if quality:
                                feedback_parts.append(f"## Quality Assessment (Score: {quality.overall_score:.1f})")
                                feedback_parts.append(f"Vision Assessment: {quality.vision_assessment}")
                                if quality.primary_issue:
                                    feedback_parts.append(f"Primary Issue: {quality.primary_issue}")
                                if quality.suggestions:
                                    feedback_parts.append(f"Suggestions: {chr(10).join(f'- {s}' for s in quality.suggestions)}")
                            if learning and learning.suggested_modifications:
                                feedback_parts.append(f"## Learning Agent Recommendations")
                                feedback_parts.append(chr(10).join(f'- {m}' for m in learning.suggested_modifications))
                            # Also include concrete params as guidance for Script Writer
                            if learning and learning.parameter_modifications:
                                feedback_parts.append(f"## Concrete Parameter Changes")
                                for param, value in learning.parameter_modifications.items():
                                    feedback_parts.append(f"- {param}: {value}")

                            # Artifact-First: Compact references instead of inline dumps
                            research_ref = artifact_mgr.get_research_path()
                            prev_quality_ref = artifact_mgr.get_quality_path(iteration - 1) if iteration > 1 else None

                            # Build compact alternatives/techniques lists
                            untried = [a for a in session.alternative_approaches if a not in session.techniques_tried][:5]
                            untried_str = ", ".join(untried) if untried else "None"
                            tried_str = ", ".join(session.techniques_tried) if session.techniques_tried else "None"

                            # Recommend technique switch if stuck
                            switch_flag = ""
                            if ctx.get("stuck_issues") or ctx.get("consecutive_same_issue", 0) >= 2:
                                switch_flag = "**TECHNIQUE SWITCH RECOMMENDED** - current approach not resolving issues."

                            # Compact feedback from learning agent
                            feedback_str = quality.primary_issue or "No primary issue"
                            if learning and learning.parameter_modifications:
                                param_changes = ", ".join(f"{k}={v}" for k, v in learning.parameter_modifications.items())
                                feedback_str += f" | Suggested params: {param_changes}"

                            script_prompt = f"""Modify the existing script to fix quality issues.

## Script
Path: {previous_script.script_path if previous_script else 'Unknown'}
Technique: {previous_script.technique_used if previous_script else 'Unknown'}

## Artifacts (read if needed)
Research: {research_ref or 'None'}
Previous Quality: {prev_quality_ref or 'None'}

## Context
Untried approaches: {untried_str}
Already tried: {tried_str}
{switch_flag}

## Issue to Fix
{feedback_str}

## Target
Previous: {previous_score:.1f} | Target: {request.quality_threshold}

Fix the primary issue. Read artifacts for details if needed."""

                            # Create fresh hooks for this iteration (reset counters)
                            # Use fallback hooks for iteration 2+ since Research Agent already queried docs
                            # This prevents blocking modify_script when doc queries were done in iter 1
                            iter_script_hooks = create_fallback_script_writer_hooks()
                            try:
                                script_result = await Runner.run(
                                    self._script_agent_standalone,
                                    script_prompt,
                                    context=context,
                                    session=sdk_session,  # Phase 4: SDK session for conversation persistence
                                    hooks=iter_script_hooks,
                                    max_turns=15
                                )
                                # Structured output: ScriptOutput
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
                            except DocQueryRequiredError as e:
                                print(f"[Pipeline] ERROR: Script Writer missing doc query (iter {iteration}): {e}", file=sys.stderr)
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
                                    print(f"[Pipeline] WARN: Script Writer loop (iter {iteration}, wrapped): {e}", file=sys.stderr)
                                    script = ScriptOutput(
                                        script_path="",
                                        technique_used="loop_detected",
                                        parameters_set={},
                                        validation_passed=False,
                                        validation_errors=["Loop detected - agent made too many consecutive doc queries"]
                                    )
                                else:
                                    raise  # Re-raise if it's a different error
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

                    # ====== PHASE 2: EXECUTION ======
                    print(f"[Pipeline] PHASE 2: Executor", file=sys.stderr)
                    exec_prompt = f"""Execute the Blender script and render the VFX asset.

Script Path: {script.script_path}
Frames: {request.frame_start}-{request.frame_end}

Run the script and report results."""

                    # Executor doesn't need doc query enforcement, just loop detection
                    exec_hooks = EnforcementHooks(EnforcementConfig(
                        max_same_tool_calls=3,
                        max_turns=6,
                        require_doc_query_before=[],  # Executor doesn't write code
                        raise_on_doc_missing=False,
                    ))
                    try:
                        exec_result = await Runner.run(
                            self._executor_agent_standalone,
                            exec_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            hooks=exec_hooks,
                            max_turns=6
                        )
                        # Structured output: ExecutionOutput
                        execution: ExecutionOutput = exec_result.final_output
                    except LoopDetectedError as e:
                        print(f"[Pipeline] WARN: Executor loop: {e}", file=sys.stderr)
                        execution = ExecutionOutput(
                            success=False,
                            render_path=None,
                            vdb_path=None,
                            error_message=f"Executor loop detected: {e}",
                            execution_time_seconds=0.0
                        )
                    print(f"[Pipeline] Executor hooks stats: {exec_hooks.get_stats()}", file=sys.stderr)

                    if not execution.success or not execution.render_path:
                        error_msg = execution.error_message or "Unknown execution error"
                        print(f"[Pipeline] ERROR: Execution failed: {error_msg}", file=sys.stderr)

                        # ====== PHASE 2.5: ERROR RECOVERY (AI-driven script fix) ======
                        # Instead of blindly continuing to the next iteration (which would
                        # just tweak parameters), give the Script Writer a chance to fix
                        # the structural code issue using write_script (full code generation).
                        recovery_succeeded = False

                        # Only attempt recovery once per iteration to avoid infinite loops
                        if not getattr(context, '_recovery_attempted_this_iter', False):
                            context._recovery_attempted_this_iter = True
                            print(f"[Pipeline] PHASE 2.5: Error Recovery (AI-driven script fix)", file=sys.stderr)

                            recovery_prompt = f"""CRITICAL: The Blender script FAILED to execute. You must fix the script.

## Error Message
{error_msg}

## Failed Script
Path: {script.script_path}

## What Went Wrong
This is a STRUCTURAL code issue, not a parameter issue. Do NOT use modify_script
(which only changes parameter values). Instead:
1. Read the failing script to understand its structure
2. Identify the root cause of the error
3. Use write_script to generate a FIXED version of the complete script

## Common Structural Fixes
- Bake context: bpy.ops.fluid.bake_data() requires the domain object to be
  bpy.context.view_layer.objects.active AND selected
- Effector setup: Collision objects need a Fluid modifier with fluid_type='EFFECTOR'
- Object context: bpy.ops calls require correct context (active object, selection)
- Missing depsgraph update: call bpy.context.view_layer.depsgraph.update() after setup

## Effect Type: {request.effect_type.value}
## Technique: {script.technique_used}

Fix the script and output it with write_script. Keep the same technique name
but append '_errfix' to the output name."""

                            try:
                                recovery_hooks = create_fallback_script_writer_hooks()
                                recovery_result = await Runner.run(
                                    self._script_agent_standalone,
                                    recovery_prompt,
                                    context=context,
                                    session=sdk_session,
                                    hooks=recovery_hooks,
                                    max_turns=6,
                                )

                                recovery_script = recovery_result.final_output
                                if recovery_script and recovery_script.script_path:
                                    print(f"[Pipeline] Recovery produced: {recovery_script.script_path}", file=sys.stderr)

                                    # Re-execute the fixed script
                                    print(f"[Pipeline] PHASE 2.5b: Re-executing recovered script", file=sys.stderr)
                                    re_exec_hooks = create_quality_analyst_hooks()  # Fresh hooks
                                    reexec_prompt = f"""Execute this recovered Blender script:

Script: {recovery_script.script_path}
Effect: {request.effect_type.value}

Execute it and report results."""

                                    try:
                                        reexec_result = await Runner.run(
                                            self._executor_agent_standalone,
                                            reexec_prompt,
                                            context=context,
                                            session=sdk_session,
                                            hooks=re_exec_hooks,
                                            max_turns=4,
                                        )
                                        reexec_output = reexec_result.final_output
                                        if reexec_output and reexec_output.success and reexec_output.render_path:
                                            print(f"[Pipeline] Recovery SUCCEEDED - render: {reexec_output.render_path}", file=sys.stderr)
                                            # Replace the failed execution with recovered one
                                            execution = reexec_output
                                            script = recovery_script
                                            recovery_succeeded = True
                                        else:
                                            re_err = reexec_output.error_message if reexec_output else "No output"
                                            print(f"[Pipeline] Recovery re-execution also failed: {re_err}", file=sys.stderr)
                                    except Exception as e:
                                        print(f"[Pipeline] Recovery re-execution error: {e}", file=sys.stderr)
                                else:
                                    print(f"[Pipeline] Recovery produced no script_path", file=sys.stderr)
                            except Exception as e:
                                print(f"[Pipeline] Error recovery failed: {e}", file=sys.stderr)

                        # Reset recovery flag for next iteration
                        context._recovery_attempted_this_iter = False

                        if not recovery_succeeded:
                            # Recovery failed or wasn't attempted — record failure and continue
                            quality = QualityOutput(
                                overall_score=0,
                                passed=False,
                                primary_issue=f"Execution failed: {error_msg}",
                                issues=[error_msg],
                                suggestions=["Fix script errors and retry"]
                            )
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
                            print(f"[Pipeline] Execution failure recorded: consecutive_same_issue={session_mgr.issue_tracker.consecutive_same_issue}", file=sys.stderr)
                            continue

                        # Recovery succeeded — fall through to quality evaluation
                        print(f"[Pipeline] Continuing to quality evaluation with recovered script", file=sys.stderr)

                    print(f"[Pipeline] Render: {execution.render_path} ({execution.execution_time_seconds:.1f}s)", file=sys.stderr)

                    # ====== PHASE 2.7: ARTIFACT GATES (Deterministic Validation) ======
                    # Run BEFORE quality evaluation to catch structural failures (empty caches,
                    # missing renders) without involving LLM. Deterministic first, adaptive second.
                    print(f"[Pipeline] PHASE 2.7: Artifact Gates", file=sys.stderr)

                    # Derive output_dir from render_path (renders are in output_dir root)
                    artifact_output_dir = None
                    if execution.render_path:
                        artifact_output_dir = str(Path(execution.render_path).parent)

                    gates_passed, gate_results, artifact_summary = validate_execution_artifacts(
                        output_dir=artifact_output_dir,
                        run_dir=None,  # Not available in ExecutionOutput
                        effect_type=request.effect_type.value,
                        verbose=True,
                    )

                    if not gates_passed:
                        # Artifact gates failed - this is a structural issue, not a quality issue
                        # Route to failure path WITHOUT quality evaluation
                        failure_diagnosis = format_gate_failure_for_diagnosis(gate_results, artifact_summary)
                        print(f"[Pipeline] ARTIFACT GATE FAILED - skipping quality evaluation", file=sys.stderr)

                        # Record as execution failure with gate diagnostics
                        gate_issues = [g.reason for g in gate_results if not g.passed]
                        quality = QualityOutput(
                            overall_score=0,
                            passed=False,
                            primary_issue=f"Artifact gate failed: {gate_issues[0] if gate_issues else 'Unknown'}",
                            issues=gate_issues,
                            suggestions=[
                                "Check simulation settings (e.g., use_plane_init for liquid emitters)",
                                "Verify bake completed with non-empty cache",
                                "Ensure render output path is valid",
                            ],
                            vision_assessment=f"No vision evaluation - artifact gates failed. Cache: {artifact_summary.cache_size_mb:.2f}MB, Renders: {artifact_summary.render_count}",
                        )

                        # Update session tracking
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
                        print(f"[Pipeline] Artifact gate failure recorded: {gate_issues}", file=sys.stderr)

                        # Store gate diagnostics in context for potential use by diagnosis agent
                        context.last_gate_failure = failure_diagnosis
                        context.last_artifact_summary = artifact_summary

                        # Continue to next iteration (modification may fix the issue)
                        continue

                    print(f"[Pipeline] Artifact gates PASSED: cache={artifact_summary.cache_size_mb:.2f}MB, renders={artifact_summary.render_count}", file=sys.stderr)

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

                    print(f"[Pipeline] PHASE 3: Quality Analyst (LLM-as-Judge)", file=sys.stderr)
                    eval_prompt = f"""Evaluate the render quality strictly.

Render Path: {execution.render_path}
Effect Type: {request.effect_type.value}
Reference: {request.reference_path or "None"}
Quality Threshold: {request.quality_threshold}

Be a strict judge. Only pass renders that truly meet quality standards.
Provide detailed feedback for improvement."""

                    try:
                        eval_result = await Runner.run(
                            self._quality_agent_standalone,
                            eval_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            hooks=quality_hooks,
                            max_turns=6
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
                    print(f"[Pipeline] Quality artifact: {quality_artifact_path}", file=sys.stderr)

                    previous_score = quality.overall_score
                    print(f"[Pipeline] Score: {quality.overall_score:.1f} | Passed: {quality.passed}", file=sys.stderr)
                    if quality.primary_issue:
                        print(f"[Pipeline] Issue: {quality.primary_issue[:60]}...", file=sys.stderr)

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
                        cache_size_mb=artifact_summary.cache_size_mb if artifact_summary else 0.0,
                        render_count=artifact_summary.render_count if artifact_summary else 0,
                        vdb_count=artifact_summary.vdb_count if artifact_summary else 0,
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

                    # ====== PHASE 4: LEARNING ======
                    print(f"[Pipeline] PHASE 4: Learning Agent", file=sys.stderr)

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

## CRITICAL: Before Suggesting Modifications
FIRST call analyze_script_modifiable_patterns("{script.script_path}") to understand:
- What shader_node_inputs exist (these control visual appearance!)
- What Config class values exist (and if they're used)
- What settings assignments exist
Then provide parameter_modifications using EXACT identifiers from the analysis.

## Your Output
- If score improved significantly (delta >= 5), extract the pattern
- If same issue 3+ times, recommend 'switch_technique'
- If issues are parameter-related, provide parameter_modifications using EXACT patterns from script analysis
- Recommend: 'iterate' | 'switch_technique' | 'complete'"""

                    try:
                        learn_result = await Runner.run(
                            self._learning_agent_standalone,
                            learn_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            hooks=learning_hooks,
                            max_turns=8
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
                    print(f"[Pipeline] PHASE 5: Quality Gate (Coordinator)", file=sys.stderr)

                    gate_decision: Optional[QualityDecision] = None
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
Action: {learning.next_action} | Reasoning: {learning.suggested_modifications[0][:60] if learning.suggested_modifications else 'None'}...

Decide: Is quality gate PASSED? What is the next action?"""

                    try:
                        gate_result = await Runner.run(
                            self._quality_gate_coordinator,
                            gate_prompt,
                            context=context,
                            session=sdk_session,  # Phase 4: SDK session for conversation persistence
                            max_turns=3  # Quality gate should be very fast
                        )
                        gate_decision = gate_result.final_output
                        print(f"[Pipeline] Quality Gate: passed={gate_decision.passed}, next={gate_decision.next_action}", file=sys.stderr)
                        print(f"[Pipeline] Escape Level: {gate_decision.escape_level}, Reasoning: {gate_decision.reasoning[:50]}...", file=sys.stderr)

                        # Phase 1.2: Sync escape_level to session IMMEDIATELY after Coordinator returns
                        # This ensures downstream logic has access to the current escape level
                        session.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)
                        context.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)

                    except Exception as e:
                        print(f"[Pipeline] WARN: Quality Gate Coordinator failed: {e}", file=sys.stderr)
                        # Fall back to simple quality check
                        gate_decision = None

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
                    if should_switch:
                        consecutive = session_mgr.issue_tracker.consecutive_same_issue
                        print(f"[Pipeline] STUCK DETECTED: same issue {consecutive}x - re-running Research Agent", file=sys.stderr)

                        # Phase 1.4: Reset stuck state when switching techniques
                        # This prevents old failure streaks from affecting the new approach
                        session_mgr.reset_for_technique_switch(f"switch_from_iter{iteration}")

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
                            research_result = await Runner.run(
                                self._research_agent,
                                switch_prompt,
                                context=context,
                                session=sdk_session,  # Phase 4: SDK session for conversation persistence
                                hooks=switch_research_hooks,
                                max_turns=4  # Phase 3: Aligned with prompt turn budget
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
                            print(f"[Pipeline] WARN: Research Agent guardrail tripped (doc_refs empty): {e}", file=sys.stderr)
                            # Non-fatal: keep existing research_text, pipeline continues
                        print(f"[Pipeline] New research: {research_text[:80]}...", file=sys.stderr)
                        print(f"[Pipeline] Switch research hooks stats: {switch_research_hooks.get_stats()}", file=sys.stderr)
                        # Next iteration will get modified feedback to try different approach

                    # ====== END OF ITERATION: WRITE ITERATION ARTIFACT ======
                    # Artifact-First: Write iteration snapshot for cross-iteration reference
                    cache_path = str(Path(artifact_output_dir) / "cache") if artifact_output_dir else None
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
                    # Trigger manual compaction to summarize old conversation history.
                    # This keeps token usage bounded across iterations while preserving
                    # full context within each iteration.
                    if sdk_session and hasattr(sdk_session, 'run_compaction'):
                        try:
                            await sdk_session.run_compaction({"force": True})
                            print(f"[Pipeline] Session history compacted after iteration {iteration}", file=sys.stderr)
                        except Exception as e:
                            print(f"[Pipeline] WARN: Session compaction failed: {e}", file=sys.stderr)

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

    def _build_generation_prompt(
        self,
        request: AssetRequest,
        context: SharedContext
    ) -> str:
        """Build the initial prompt for asset generation."""
        return f"""Generate a VFX asset with the following specifications:

## Request
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}
- Description: {request.description}

## Parameters
- Resolution: {request.resolution}
- Frame Range: {request.frame_start} - {request.frame_end}
- Quality Threshold: {request.quality_threshold}
- Max Iterations: {request.max_iterations}

## Reference Materials
- Reference Image: {request.reference_path or "None provided"}
- Semantic Query: {request.semantic_query or request.description}

## Instructions
1. Start by generating a script using the Script Writer
2. Execute the script using the Executor
3. Evaluate quality using the Quality Analyst
4. If quality is below threshold, iterate with fixes
5. Continue until quality threshold met or max iterations reached

Session ID: {context.session.session_id}
Current Budget Used: ${self._budget_tracker.get_spent():.2f} / ${self._budget_tracker.monthly_limit}

Begin the asset generation process."""

    def _build_resumption_prompt(self, context: SharedContext) -> str:
        """Build the prompt for resuming a session."""
        session = context.session
        request = session.request

        # Get recent history
        recent_iterations = session.iterations[-3:] if session.iterations else []
        history_summary = "\n".join([
            f"  - Iteration {it.iteration}: score={it.score:.1f}, passed={it.passed}"
            for it in recent_iterations
        ])

        return f"""Resume VFX asset generation session.

## Session Info
- Session ID: {session.session_id}
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}

## Progress
- Current Iteration: {session.current_iteration}
- Best Score: {session.best_score:.1f} (iteration {session.best_iteration})
- Max Iterations: {request.max_iterations}

## Recent History
{history_summary or "  No iterations completed yet"}

## Current State
- Script Path: {session.current_script_path or "None"}
- Current Issues: {', '.join(session.current_issues) or "None"}

## Instructions
Continue from where you left off. Review the current state and proceed
with the next appropriate action.

Current Budget Used: ${self._budget_tracker.get_spent():.2f} / ${self._budget_tracker.monthly_limit}

Resume the asset generation process."""

    def _parse_result(
        self,
        result: Any,
        context: SharedContext
    ) -> SessionState:
        """Parse the orchestrator result and update session state."""
        session = context.session

        # Try to extract structured output
        if hasattr(result, 'final_output'):
            output = result.final_output

            # Try to parse as JSON
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                    if 'passed' in data:
                        session.status = SessionStatus.PASSED if data['passed'] else SessionStatus.IN_PROGRESS
                    if 'final_score' in data:
                        session.best_score = max(session.best_score, data['final_score'])
                    if 'final_render_path' in data:
                        session.final_render_path = data['final_render_path']
                except json.JSONDecodeError:
                    pass

        # Check for max iterations
        if session.current_iteration >= session.request.max_iterations:
            if session.status == SessionStatus.IN_PROGRESS:
                session.status = SessionStatus.MAX_ITERATIONS

        session.update_timestamp()
        return session

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
