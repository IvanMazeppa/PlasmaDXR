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
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, Runner, handoff, trace, RunContextWrapper, ItemHelpers

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
from agents.agent_output import AgentOutputSchema

# Enforcement Hooks for loop detection and doc query requirements
from hooks import (
    EnforcementHooks,
    EnforcementConfig,
    LoopDetectedError,
    DocQueryRequiredError,
    TurnBudgetExceededError,
)
from hooks.enforcement_hooks import (
    create_research_hooks,
    create_script_writer_hooks,
    create_quality_analyst_hooks,
    create_learning_agent_hooks,
)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions
from openai.types.shared import Reasoning


# =============================================================================
# STRUCTURED OUTPUT MODELS (Pydantic)
# =============================================================================
# These enable type-safe data passing between agents in the pipeline.

class ResearchOutput(BaseModel):
    """Output from Research Agent - provides starting parameters for script generation."""
    recommended_approach: str = Field(description="Best approach for the effect type")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameter values")
    api_modules: List[str] = Field(default_factory=list, description="Blender API modules to use")
    code_patterns: List[Dict[str, str]] = Field(default_factory=list, description="Proven patterns from library")
    warnings: List[str] = Field(default_factory=list, description="Potential pitfalls to avoid")
    alternative_approaches: List[str] = Field(default_factory=list, description="Backup approaches if primary fails")


class ScriptOutput(BaseModel):
    """Output from Script Writer - path to generated/modified script."""
    script_path: str = Field(description="Absolute path to the generated script")
    technique_used: str = Field(description="Technique/approach used in script generation")
    parameters_set: Dict[str, Any] = Field(default_factory=dict, description="Key parameters configured")
    validation_passed: bool = Field(default=True, description="Whether script validation passed")
    validation_errors: List[str] = Field(default_factory=list, description="Any validation errors")


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
    action: str = Field(description="Action to take: 'modify_params', 'switch_technique', 'continue'")
    parameter_changes: Dict[str, Any] = Field(default_factory=dict, description="Concrete parameter changes")
    new_technique: Optional[str] = Field(default=None, description="New technique if switching")
    reasoning: str = Field(description="Why this modification strategy was chosen")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this decision (0.0-1.0)")


class QualityDecision(BaseModel):
    """Output from Coordinator for quality gate interpretation."""
    passed: bool = Field(description="Whether quality gate is passed")
    should_continue: bool = Field(description="Whether to continue iterating")
    next_action: str = Field(description="Next action: 'complete', 'iterate', 'switch_technique', 'request_guidance'")
    escape_level: int = Field(ge=0, le=4, description="Current escape velocity level (0-4)")
    reasoning: str = Field(description="Explanation of the quality decision")

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

# Proactive research tools for Strategy 3: Early warning detection
from tools.proactive_research_tools import (
    pre_iteration_research,
    evaluate_escape_velocity,
    search_alternative_approaches,
)

# Semantic docs tools for Strategy 1: Vector Store for Blender Documentation
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    find_alternative_approaches,
    search_blender_api_by_intent,
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

from specialized_agents import (
    create_script_writer,
    create_executor,
    create_quality_analyst,
    create_learning_agent,
)
# DocsExpert now uses in-process function_tools instead of MCP client connections.
# This fixes the anyio TaskGroup conflict that previously blocked MCP tool handlers.
from specialized_agents.docs_expert import create_docs_expert
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
1. semantic_search_blender_docs("{effect_type} simulation best practices")
2. search_code_patterns("initial generation", effect_type)
3. search_blender_api_by_intent("create {effect_type} with density/emission", "fluid")

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
Research: semantic_search_blender_docs, find_alternative_approaches, search_blender_api_by_intent
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

    SDK Pattern: agent.as_tool() allows calling agents as utility functions
    while keeping control with the Coordinator. This is the "Manager" pattern
    from the SDK documentation.

    Args:
        research_agent: Research Agent instance
        script_agent: Script Writer Agent instance
        executor_agent: Executor Agent instance
        quality_agent: Quality Analyst Agent instance
        learning_agent: Learning Agent Agent instance

    Returns:
        Coordinator Agent with all sub-agents as tools
    """
    from specialized_agents.api_validator import get_api_validator_as_tool

    # Wrap each agent as a tool
    agent_tools = [
        research_agent.as_tool(
            tool_name="research_approach",
            tool_description=(
                "Research best approach for VFX effect type. Use at iteration 1 to find "
                "optimal technique, parameters, and alternatives. Returns research summary."
            ),
        ),
        get_api_validator_as_tool(),  # Already provides as_tool wrapper
        script_agent.as_tool(
            tool_name="generate_script",
            tool_description=(
                "Generate or modify Blender Python script for VFX effect. Provide effect_type, "
                "description, and technique. Returns script_path, technique_used, parameters."
            ),
        ),
        executor_agent.as_tool(
            tool_name="execute_script",
            tool_description=(
                "Execute Blender script and render VFX asset. Provide script_path. "
                "Returns success, render_path, vdb_path, execution_time."
            ),
        ),
        quality_agent.as_tool(
            tool_name="evaluate_render",
            tool_description=(
                "Evaluate render quality using vision and metrics. Provide render_path, "
                "effect_type. Returns overall_score, passed, issues, suggestions."
            ),
        ),
        learning_agent.as_tool(
            tool_name="record_experiment",
            tool_description=(
                "Record experiment results and suggest next action. Provide iteration, "
                "score, issues, params. Returns next_action, parameter_modifications."
            ),
        ),
    ]

    # Also include direct research tools for the Coordinator to use
    research_tools = [
        semantic_search_blender_docs,
        find_alternative_approaches,
        search_blender_api_by_intent,
        search_code_patterns,
        list_patterns_by_effect,
        pre_iteration_research,
        evaluate_escape_velocity,
    ]

    return Agent[SharedContext](
        name="VFX Coordinator",
        instructions=COORDINATOR_INSTRUCTIONS,
        model=os.getenv("COORDINATOR_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(verbosity="low"),
        tools=agent_tools + research_tools,
    )


def create_technique_selection_coordinator(
    research_agent: Agent,
) -> Agent:
    """
    Create a lightweight Coordinator for technique selection decisions only.

    This is called at iteration 1 to select the initial technique.
    Uses TechniqueDecision structured output.
    """
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
        model=os.getenv("COORDINATOR_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(verbosity="low"),
        tools=[
            research_agent.as_tool(
                tool_name="research_approach",
                tool_description="Research best approach for effect type",
            ),
            semantic_search_blender_docs,
            search_code_patterns,
            list_patterns_by_effect,
        ],
        output_type=AgentOutputSchema(TechniqueDecision, strict_json_schema=False),
    )


def create_modification_coordinator() -> Agent:
    """
    Create a lightweight Coordinator for modification strategy decisions.

    This is called at iteration 2+ to decide how to fix quality issues.
    Uses ModificationDecision structured output.
    """
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

## Decision Logic
1. If score is close to threshold (within 10): modify_params with targeted changes
2. If same issue 3+ times: switch_technique to fundamentally different approach
3. If score improving: continue with incremental param changes

## Output
Return a ModificationDecision with:
- action: 'modify_params' | 'switch_technique' | 'continue'
- parameter_changes: Dict of {param_name: new_value} - CONCRETE numbers
- new_technique: New technique name if switching
- reasoning: Why this strategy
- confidence: 0.0-1.0 confidence level""",
        model=os.getenv("COORDINATOR_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(verbosity="low"),
        tools=[
            search_code_patterns,
            pre_iteration_research,
            evaluate_escape_velocity,
        ],
        output_type=AgentOutputSchema(ModificationDecision, strict_json_schema=False),
    )


def create_quality_gate_coordinator() -> Agent:
    """
    Create a lightweight Coordinator for quality gate decisions.

    This is called after quality evaluation to interpret results.
    Uses QualityDecision structured output.
    """
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
        model=os.getenv("COORDINATOR_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(verbosity="low"),
        tools=[
            evaluate_escape_velocity,
        ],
        output_type=AgentOutputSchema(QualityDecision, strict_json_schema=False),
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

        self._budget_tracker: BudgetTracker = get_budget_tracker()
        self._persistence: SessionPersistence = get_persistence()
        self._initialized = False

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
            pre_iteration_research,
            evaluate_escape_velocity,
            search_alternative_approaches,
            # Semantic docs tools (Strategy 1)
            semantic_search_blender_docs,
            find_alternative_approaches,
            search_blender_api_by_intent,
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
            model=os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2"),
            model_settings=ModelSettings(verbosity="low"),
            tools=orchestrator_tools,
        )

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
            model=os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2"),
            model_settings=ModelSettings(verbosity="low"),
            handoffs=handoffs_list,
            tools=orchestrator_tools,
        )

        # Step 5: Create STANDALONE agents for code-based orchestration
        # These have NO handoffs - Python controls the pipeline sequence directly
        # Using structured outputs (output_type) for type-safe data passing between agents
        self._research_agent = Agent[SharedContext](
            name="Research Agent",
            instructions=prompt_with_handoff_instructions("""You are a Blender documentation research specialist.

Your job is to research the best approach for creating a VFX effect BEFORE any script generation.

## Research Steps
1. Search documentation for best practices for the effect type
2. Find proven code patterns from the pattern library
3. Search Blender API by intent to find the right modules/functions
4. Identify alternative approaches if the standard one is unclear

Use your tools to gather information, then return a summary of your findings including:
- recommended_approach: Best approach for the effect
- key_parameters: Important parameter settings
- warnings: Potential pitfalls to avoid"""),
            model=os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2"),
            model_settings=ModelSettings(verbosity="low"),
            # Note: Don't use output_type for tool-using agents - they output messages, not structured data
            tools=[
                semantic_search_blender_docs,
                find_alternative_approaches,
                search_blender_api_by_intent,
                search_code_patterns,
                list_patterns_by_effect,
            ],
        )

        # Script Writer with structured output
        # NOTE: Using static instructions for standalone agents (can't append to functions)
        # KB integration happens through tools (query_knowledge_base, get_physics_patterns)
        # No hardcoded physics rules - they emerge from experimentation
        base_script_writer_standalone = create_script_writer(use_dynamic_instructions=False)
        self._script_agent_standalone = Agent[SharedContext](
            name="Script Writer",
            instructions=prompt_with_handoff_instructions(base_script_writer_standalone.instructions + """

## EFFICIENCY REQUIREMENT - CRITICAL
You have LIMITED turns (max 10). Be efficient:
1. Call recommend_technique ONCE to pick approach
2. Call generate_script ONCE to create initial script
3. Call validate_script ONCE
4. If validation fails, call modify_script AT MOST 2 times
5. IMMEDIATELY return ScriptOutput - do NOT keep iterating

Total tool calls should be 4-6, not 10+. Return output even if imperfect.

## SELF-LEARNING NOTE
Physics rules are NOT hardcoded. They come from the knowledge base (via dynamic instructions).
If the KB has no rules for this effect type yet, use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from experiments.

## Output Requirements
After generating and validating the script, return a structured ScriptOutput with:
- script_path: Absolute path to the generated/modified script
- technique_used: The approach/technique used
- parameters_set: Key parameters configured in the script
- validation_passed: Whether validation succeeded
- validation_errors: Any validation errors encountered

IMPORTANT: Always return the script_path even if validation fails. Do NOT loop indefinitely."""),
            model=base_script_writer_standalone.model,
            model_settings=base_script_writer_standalone.model_settings,
            output_type=AgentOutputSchema(ScriptOutput, strict_json_schema=False),
            tools=base_script_writer_standalone.tools,
        )

        # Executor with structured output
        base_executor = create_executor()
        self._executor_agent_standalone = Agent[SharedContext](
            name="Executor",
            instructions=prompt_with_handoff_instructions(base_executor.instructions + """

## Output Requirements
After executing the script, return a structured ExecutionOutput with:
- success: Whether Blender executed without errors
- render_path: Path to the rendered output image/sequence
- vdb_path: Path to VDB volume data (if generated)
- error_message: Error details if execution failed
- execution_time_seconds: How long execution took"""),
            model=base_executor.model,
            model_settings=base_executor.model_settings,
            output_type=AgentOutputSchema(ExecutionOutput, strict_json_schema=False),
            tools=base_executor.tools,
        )

        # Quality Analyst with structured output (LLM-as-judge pattern)
        # NOTE: Using static instructions for standalone agents (can't append to functions)
        # Physics observation happens through tools (observe_physics_anomaly, get_physics_patterns)
        base_quality_standalone = create_quality_analyst(use_dynamic_instructions=False)
        self._quality_agent_standalone = Agent[SharedContext](
            name="Quality Analyst",
            instructions=prompt_with_handoff_instructions(base_quality_standalone.instructions + """

## SELF-LEARNING: Physics Observation
When you observe unexpected physical behavior, use observe_physics_anomaly() to record it.
The Learning Agent will correlate these observations with parameters to build knowledge.

DO NOT assume what physics should look like. OBSERVE and REPORT:
- What did you expect to see?
- What did you actually see?
- What parameters might be causing this?

## Output Requirements (LLM-as-Judge Pattern)
After evaluating render quality, return a structured QualityOutput with:
- overall_score: Quality score 0-100
- passed: Whether quality threshold was met
- primary_issue: The most critical issue to fix (if any)
- issues: List of all identified issues
- suggestions: Specific parameter changes to try
- vision_assessment: Detailed visual quality description
- reference_similarity: Similarity to reference image (if available)

Be a STRICT judge - only pass renders that truly meet quality standards."""),
            model=base_quality_standalone.model,
            model_settings=base_quality_standalone.model_settings,
            output_type=AgentOutputSchema(QualityOutput, strict_json_schema=False),
            tools=base_quality_standalone.tools,
        )

        # Learning Agent with structured output
        # NOTE: Using static instructions for standalone agents (can't append to functions)
        # Core of the self-learning system - processes physics observations via tools
        base_learning_standalone = create_learning_agent(use_dynamic_instructions=False)
        self._learning_agent_standalone = Agent[SharedContext](
            name="Learning Agent",
            instructions=prompt_with_handoff_instructions(base_learning_standalone.instructions + """

## CRITICAL: TURN BUDGET (MAX 8 TURNS - HARD LIMIT)
You MUST complete in 3-4 turns or the pipeline FAILS. Follow this EXACT sequence:

Turn 1: Query knowledge (query_knowledge_base) + check pending observations (get_pending_observations)
Turn 2: Record experiment (record_experiment_result) - CALL EXACTLY ONCE
        If pending physics observations: correlate_observation() for each
Turn 3: Return LearningOutput structured response

RULES:
- DO NOT make more than 4 tool calls total
- DO NOT call record_experiment_result more than ONCE (even if it returns an error)
- DO NOT call start_experiment_session or record_baseline (handled elsewhere)
- If record_experiment_result fails, STILL return LearningOutput (set experiment_recorded=False)
- NEVER retry failed tool calls

## SELF-LEARNING: Physics Observations
Check get_pending_observations() for any physics anomalies the Quality Analyst recorded.
For each pending observation, use correlate_observation() to record your analysis:
- What parameters likely caused this behavior?
- What's the recommended fix?
- How confident are you? (0.0-1.0)

This builds the knowledge base that dynamic instructions query!

## Output Requirements (RETURN AFTER 1 record_experiment_result call)
Return a structured LearningOutput with:
- experiment_recorded: True (you recorded it)
- pattern_extracted: bool
- pattern_id: str or None
- next_action: 'iterate' | 'switch_technique' | 'complete'
- suggested_modifications: List[str] - TEXT descriptions for context
- parameter_modifications: Dict[str, Any] - CONCRETE VALUES for direct script modification

CRITICAL FOR parameter_modifications:
Provide ACTUAL NUMBERS, not descriptions. Example:
  {"temperature": 3.0, "density": 5.0, "blackbody_intensity": 2.0, "beta": 0.0}

These values are applied DIRECTLY to the Blender script's Config class.
If quality issues relate to parameters, ALWAYS include concrete fixes in parameter_modifications."""),
            model=base_learning_standalone.model,
            model_settings=base_learning_standalone.model_settings,
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

        self._initialized = True
        print("[Orchestrator] Initialization complete (5 agents + 5 standalone + 3 coordinators ready)", file=sys.stderr)

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

    async def create_asset_pipeline(self, request: AssetRequest) -> SessionState:
        """
        CODE-BASED ORCHESTRATION: Python controls the pipeline sequence.

        This method implements deterministic workflow control instead of
        relying on LLM instruction-following. Each agent runs independently
        and Python manages the state transitions.

        Pipeline: Research → Script → Execute → Evaluate → Learn → (loop)

        Args:
            request: Asset generation request parameters

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

        # Create session
        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)
        session = context.session

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"PIPELINE ORCHESTRATION: {request.asset_name}", file=sys.stderr)
        print(f"Effect: {request.effect_type.value} | Max Iterations: {request.max_iterations}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)

        try:
            with trace(f"VFX Pipeline: {request.asset_name}"):
                # ====== PHASE 0: RESEARCH (once at start) ======
                print("[Pipeline] PHASE 0: Research", file=sys.stderr)
                research_prompt = f"""Research the best approach for creating a {request.effect_type.value} VFX effect.

Effect Type: {request.effect_type.value}
Description: {request.description}
Reference: {request.reference_path or "None"}

Research documentation, patterns, and APIs to find the optimal starting approach."""

                # Create research hooks for Phase 0
                phase0_research_hooks = create_research_hooks()
                try:
                    with trace(
                        "Phase 0: Research",
                        metadata={
                            "phase": "research",
                            "effect_type": request.effect_type.value,
                            "iteration": 0,
                        }
                    ):
                        research_result = await Runner.run(
                            self._research_agent,
                            research_prompt,
                            context=context,
                            hooks=phase0_research_hooks,
                            max_turns=8  # Limit research turns
                        )
                    # Research agent outputs text summary (no output_type for tool-using agents)
                    research_text = str(research_result.final_output) if research_result.final_output else "No research findings"
                except LoopDetectedError as e:
                    # Research got stuck - use partial results
                    print(f"[Pipeline] WARN: Research loop detected: {e}", file=sys.stderr)
                    research_text = "Research incomplete due to loop - proceeding with default approach"
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

                # Extract alternative approaches from research text (look for bullet points/lists)
                import re
                alt_pattern = r'(?:alternative|backup|other|also consider)[s]?[:\s]+([^\n]+(?:\n[-•*]\s*[^\n]+)*)'
                alt_matches = re.findall(alt_pattern, research_text.lower())
                if alt_matches:
                    for match in alt_matches:
                        approaches = re.split(r'[-•*\n,]', match)
                        for a in approaches:
                            a = a.strip()
                            if a and len(a) > 10 and a not in session.alternative_approaches:
                                session.alternative_approaches.append(a)
                    print(f"[Pipeline] Found {len(session.alternative_approaches)} alternative approaches", file=sys.stderr)

                # ====== PHASE 0.5: TECHNIQUE SELECTION (Coordinator Decision) ======
                # Phase 2 Enhancement: Use Technique Coordinator to intelligently select
                # the initial technique based on research findings.
                print("[Pipeline] PHASE 0.5: Technique Selection (Coordinator)", file=sys.stderr)
                technique_prompt = f"""Select the best technique for creating a {request.effect_type.value} VFX effect.

## Research Findings
{research_text[:2000]}

## Effect Parameters
- Effect Type: {request.effect_type.value}
- Description: {request.description}
- Reference: {request.reference_path or "None"}
- Quality Threshold: {request.quality_threshold}

## Available Alternatives
{chr(10).join(f'- {a}' for a in session.alternative_approaches[:5]) if session.alternative_approaches else 'None identified'}

Select the optimal technique and provide starting parameters."""

                selected_technique: Optional[TechniqueDecision] = None
                try:
                    with trace(
                        "Phase 0.5: Technique Selection",
                        metadata={
                            "phase": "technique_selection",
                            "effect_type": request.effect_type.value,
                            "coordinator": "technique",
                        }
                    ):
                        technique_result = await Runner.run(
                            self._technique_coordinator,
                            technique_prompt,
                            context=context,
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

                # ====== ITERATION LOOP ======
                iteration = 0
                previous_script: Optional[ScriptOutput] = None
                previous_score = 0.0
                previous_params: Dict[str, Any] = {}
                quality: Optional[QualityOutput] = None
                learning: Optional[LearningOutput] = None

                # Session Manager for deterministic experiment state tracking
                # This is Python-side logic, not LLM - ensures baseline is always recorded
                session_mgr = SessionManager(session_id=session.session_id)

                # Create enforcement hooks for each agent type
                # These prevent loops and enforce documentation-first patterns
                research_hooks = create_research_hooks()
                script_hooks = create_script_writer_hooks()
                quality_hooks = create_quality_analyst_hooks()
                learning_hooks = create_learning_agent_hooks()

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

                while iteration < request.max_iterations:
                    iteration += 1
                    session.current_iteration = iteration
                    print(f"\n[Pipeline] ====== ITERATION {iteration}/{request.max_iterations} ======", file=sys.stderr)

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

                    # ====== PHASE 1: SCRIPT GENERATION ======
                    print(f"[Pipeline] PHASE 1: Script Writer", file=sys.stderr)
                    if iteration == 1:
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
                            with trace(
                                f"Phase 1: Script Writer (iter {iteration})",
                                metadata={
                                    "phase": "script_writer",
                                    "effect_type": request.effect_type.value,
                                    "iteration": iteration,
                                    "is_initial": True,
                                }
                            ):
                                script_result = await Runner.run(
                                    self._script_agent_standalone,
                                    script_prompt,
                                    context=context,
                                    hooks=script_hooks,
                                    max_turns=10
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
                        print(f"[Pipeline] Script hooks stats: {script_hooks.get_stats()}", file=sys.stderr)
                    else:
                        # Track if we successfully modified the script directly
                        direct_modification_success = False
                        script: Optional[ScriptOutput] = None

                        # Check for concrete parameter modifications from Learning Agent
                        # If provided, apply directly without going through Script Writer interpretation
                        if learning and learning.parameter_modifications and previous_script and previous_script.script_path:
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
                                    key_parameters=learning.parameter_modifications,
                                    validation_passed=True,
                                    validation_errors=[],
                                    description=f"Parameter-modified from {previous_script.script_path}"
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

Decide: modify_params OR switch_technique. If modifying, provide CONCRETE parameter values."""

                            try:
                                with trace(
                                    f"Phase 1.1: Modification Strategy (iter {iteration})",
                                    metadata={
                                        "phase": "modification_strategy",
                                        "effect_type": request.effect_type.value,
                                        "iteration": iteration,
                                        "coordinator": "modification",
                                    }
                                ):
                                    mod_result = await Runner.run(
                                        self._modification_coordinator,
                                        mod_prompt,
                                        context=context,
                                        max_turns=4  # Coordinators should be fast
                                    )
                                mod_decision = mod_result.final_output
                                print(f"[Pipeline] Coordinator decision: {mod_decision.action}", file=sys.stderr)
                                print(f"[Pipeline] Reasoning: {mod_decision.reasoning[:60]}...", file=sys.stderr)

                                # If Coordinator provides parameter changes, try direct modification
                                if mod_decision.action == 'modify_params' and mod_decision.parameter_changes and previous_script and previous_script.script_path:
                                    print(f"[Pipeline] Coordinator provided params: {mod_decision.parameter_changes}", file=sys.stderr)
                                    output_name = f"{request.asset_name}_iter{iteration}_coordfix"

                                    modify_result_json = _modify_script_impl(
                                        script_path=previous_script.script_path,
                                        modifications=mod_decision.parameter_changes,
                                        output_name=output_name
                                    )
                                    modify_result = json.loads(modify_result_json)

                                    if modify_result.get("success") and modify_result.get("modified_path"):
                                        print(f"[Pipeline] Coordinator modification SUCCESS: {modify_result['modified_path']}", file=sys.stderr)
                                        script = ScriptOutput(
                                            script_path=modify_result["modified_path"],
                                            technique_used=previous_script.technique_used + " (coord-modified)",
                                            key_parameters=mod_decision.parameter_changes,
                                            validation_passed=True,
                                            validation_errors=[],
                                        )
                                        direct_modification_success = True

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

                            # Include research findings and alternatives for technique switching
                            research_section = ""
                            if session.research_text:
                                research_section = f"""## Research Findings (from Phase 0)
{session.research_text[:1500]}...
"""
                            alternatives_section = ""
                            if session.alternative_approaches:
                                untried = [a for a in session.alternative_approaches if a not in session.techniques_tried]
                                if untried:
                                    alternatives_section = f"""## Alternative Approaches NOT YET TRIED
{chr(10).join(f'- {a}' for a in untried[:5])}
"""
                            techniques_section = ""
                            if session.techniques_tried:
                                techniques_section = f"""## Techniques Already Tried (do NOT repeat)
{chr(10).join(f'- {t}' for t in session.techniques_tried)}
"""

                            # Recommend technique switch if stuck
                            switch_recommendation = ""
                            if ctx.get("stuck_issues") or ctx.get("consecutive_same_issue", 0) >= 2:
                                switch_recommendation = """
## IMPORTANT: TECHNIQUE SWITCH RECOMMENDED
The current approach is not resolving issues. You SHOULD try a DIFFERENT technique from the alternatives above.
Do NOT just tweak parameters - fundamentally change the approach.
"""

                            script_prompt = f"""Modify the existing script to fix quality issues.

## Current Script
Path: {previous_script.script_path if previous_script else 'Unknown'}
Technique: {previous_script.technique_used if previous_script else 'Unknown'}

{research_section}
{alternatives_section}
{techniques_section}
{switch_recommendation}
{chr(10).join(feedback_parts)}

## Target
Previous Score: {previous_score:.1f}
Target Score: {request.quality_threshold}

Modify the script to address the issues. Do NOT repeat parameter values that hurt scores.
If stuck on same issue, try a DIFFERENT TECHNIQUE from research alternatives.
Use patterns from library if available."""

                            # Create fresh hooks for this iteration (reset counters)
                            iter_script_hooks = create_script_writer_hooks()
                            try:
                                with trace(
                                    f"Phase 1: Script Writer (iter {iteration})",
                                    metadata={
                                        "phase": "script_writer",
                                        "effect_type": request.effect_type.value,
                                        "iteration": iteration,
                                        "is_modification": True,
                                        "previous_score": previous_score,
                                    }
                                ):
                                    script_result = await Runner.run(
                                        self._script_agent_standalone,
                                        script_prompt,
                                        context=context,
                                        hooks=iter_script_hooks,
                                        max_turns=10
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
                                        # Apply the correction
                                        corrected_content = corrected_content.replace(
                                            validation.api_call.split('.')[-1] if '.' in validation.api_call else validation.api_call,
                                            validation.correction.split('.')[-1] if '.' in validation.correction else validation.correction
                                        )

                                # Also apply known patterns directly
                                for pattern, fix in KNOWN_API_CHANGES.items():
                                    if pattern in corrected_content:
                                        corrected_content = corrected_content.replace(pattern, fix["correction"])
                                        print(f"[Pipeline]   Applied known fix: {pattern} → {fix['correction']}", file=sys.stderr)

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
                        with trace(
                            f"Phase 2: Executor (iter {iteration})",
                            metadata={
                                "phase": "executor",
                                "effect_type": request.effect_type.value,
                                "iteration": iteration,
                                "script_path": script.script_path,
                            }
                        ):
                            exec_result = await Runner.run(
                                self._executor_agent_standalone,
                                exec_prompt,
                                context=context,
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
                        print(f"[Pipeline] ERROR: Execution failed: {execution.error_message}", file=sys.stderr)
                        # Continue to next iteration with error context
                        quality = QualityOutput(
                            overall_score=0,
                            passed=False,
                            primary_issue=f"Execution failed: {execution.error_message}",
                            issues=[execution.error_message or "Unknown execution error"],
                            suggestions=["Fix script errors and retry"]
                        )
                        continue

                    print(f"[Pipeline] Render: {execution.render_path} ({execution.execution_time_seconds:.1f}s)", file=sys.stderr)

                    # ====== PHASE 3: QUALITY EVALUATION (LLM-as-Judge) ======
                    print(f"[Pipeline] PHASE 3: Quality Analyst (LLM-as-Judge)", file=sys.stderr)
                    eval_prompt = f"""Evaluate the render quality strictly.

Render Path: {execution.render_path}
Effect Type: {request.effect_type.value}
Reference: {request.reference_path or "None"}
Quality Threshold: {request.quality_threshold}

Be a strict judge. Only pass renders that truly meet quality standards.
Provide detailed feedback for improvement."""

                    try:
                        with trace(
                            f"Phase 3: Quality Analyst (iter {iteration})",
                            metadata={
                                "phase": "quality_analyst",
                                "effect_type": request.effect_type.value,
                                "iteration": iteration,
                                "render_path": execution.render_path,
                            }
                        ):
                            eval_result = await Runner.run(
                                self._quality_agent_standalone,
                                eval_prompt,
                                context=context,
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

                    previous_score = quality.overall_score
                    print(f"[Pipeline] Score: {quality.overall_score:.1f} | Passed: {quality.passed}", file=sys.stderr)
                    if quality.primary_issue:
                        print(f"[Pipeline] Issue: {quality.primary_issue[:60]}...", file=sys.stderr)

                    # Update session
                    session.best_score = max(session.best_score, quality.overall_score)
                    if quality.overall_score == session.best_score:
                        session.best_iteration = iteration
                        session.final_render_path = execution.render_path

                    # Extract current params from script
                    current_params = script.key_parameters if hasattr(script, 'key_parameters') and script.key_parameters else {}
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
                    script_mod = ScriptModification(
                        script_path=script.script_path or "unknown",
                        modifications=script.key_parameters if hasattr(script, 'key_parameters') else {},
                        technique_name=script.technique_used,
                        validation_passed=script.validation_passed,
                        validation_issues=script.validation_errors,
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

                    # Get context from SessionManager for informed decisions
                    learn_ctx = session_mgr.get_context_for_agents()
                    iteration_history = session_mgr.get_iteration_summary()

                    learn_prompt = f"""Record the experiment results.

## Current Iteration
Iteration: {iteration}
Script: {script.script_path}
Technique: {script.technique_used}
Render: {execution.render_path}
Score: {quality.overall_score:.1f}
Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}

{iteration_history}

## Analysis Needed
1. Did the score improve? Delta: {learn_ctx.get('best_score', 0) - previous_score:+.1f}
2. Same issue {learn_ctx.get('consecutive_same_issue', 0)} times in a row
3. Techniques tried: {', '.join(learn_ctx.get('techniques_tried', []))}

## Your Output
- If score improved significantly (delta >= 5), extract the pattern
- If same issue 3+ times, recommend 'switch_technique'
- If issues are parameter-related, provide parameter_modifications with CONCRETE values
- Recommend: 'iterate' | 'switch_technique' | 'complete'"""

                    try:
                        with trace(
                            f"Phase 4: Learning Agent (iter {iteration})",
                            metadata={
                                "phase": "learning_agent",
                                "effect_type": request.effect_type.value,
                                "iteration": iteration,
                                "score": quality.overall_score,
                                "passed": quality.passed,
                                "primary_issue": quality.primary_issue,
                            }
                        ):
                            learn_result = await Runner.run(
                                self._learning_agent_standalone,
                                learn_prompt,
                                context=context,
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

                    # ====== QUALITY GATE (Coordinator Decision) ======
                    # Phase 2 Enhancement: Use Quality Gate Coordinator for intelligent decision
                    print(f"[Pipeline] PHASE 5: Quality Gate (Coordinator)", file=sys.stderr)

                    gate_decision: Optional[QualityDecision] = None
                    ctx = session_mgr.get_context_for_agents()

                    gate_prompt = f"""Interpret quality evaluation results for iteration {iteration}.

## Quality Results
- Score: {quality.overall_score:.1f}
- Threshold: {request.quality_threshold}
- Passed (simple check): {quality.passed}
- Primary Issue: {quality.primary_issue or 'None'}
- All Issues: {', '.join(quality.issues[:5]) if quality.issues else 'None'}

## Iteration Context
- Current Iteration: {iteration}
- Max Iterations: {request.max_iterations}
- Consecutive Same Issue: {ctx.get('consecutive_same_issue', 0)}
- Best Score So Far: {session.best_score:.1f}
- Current Escape Level: {ctx.get('escape_level', 0)}

## Learning Agent Recommendation
Next Action: {learning.next_action}
Reasoning: {learning.suggested_modifications[0] if learning.suggested_modifications else 'No reasoning provided'}

Decide: Is quality gate PASSED? What is the next action?"""

                    try:
                        with trace(
                            f"Phase 5: Quality Gate (iter {iteration})",
                            metadata={
                                "phase": "quality_gate",
                                "effect_type": request.effect_type.value,
                                "iteration": iteration,
                                "coordinator": "quality_gate",
                                "score": quality.overall_score,
                            }
                        ):
                            gate_result = await Runner.run(
                                self._quality_gate_coordinator,
                                gate_prompt,
                                context=context,
                                max_turns=3  # Quality gate should be very fast
                            )
                        gate_decision = gate_result.final_output
                        print(f"[Pipeline] Quality Gate: passed={gate_decision.passed}, next={gate_decision.next_action}", file=sys.stderr)
                        print(f"[Pipeline] Escape Level: {gate_decision.escape_level}, Reasoning: {gate_decision.reasoning[:50]}...", file=sys.stderr)

                    except Exception as e:
                        print(f"[Pipeline] WARN: Quality Gate Coordinator failed: {e}", file=sys.stderr)
                        # Fall back to simple quality check
                        gate_decision = None

                    # Determine if passed based on Coordinator or simple check
                    if gate_decision:
                        is_passed = gate_decision.passed
                        next_action = gate_decision.next_action
                    else:
                        is_passed = quality.passed or learning.next_action == 'complete'
                        next_action = learning.next_action

                    if is_passed or next_action == 'complete':
                        print(f"\n[Pipeline] ✓ QUALITY GATE PASSED at iteration {iteration}", file=sys.stderr)
                        session.status = SessionStatus.PASSED
                        break

                    # Handle technique switching - either from Coordinator OR SessionManager detection
                    should_switch = (
                        next_action == 'switch_technique'
                        or (gate_decision and gate_decision.escape_level >= 2)
                        or learning.next_action == 'switch_technique'
                        or session_mgr.should_switch_technique()
                    )
                    if should_switch:
                        consecutive = session_mgr.issue_tracker.consecutive_same_issue
                        print(f"[Pipeline] STUCK DETECTED: same issue {consecutive}x - re-running Research Agent", file=sys.stderr)

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
                            with trace(
                                f"Technique Switch Research (iter {iteration})",
                                metadata={
                                    "phase": "technique_switch",
                                    "effect_type": request.effect_type.value,
                                    "iteration": iteration,
                                    "consecutive_same_issue": consecutive,
                                    "techniques_tried": list(session_mgr.techniques_tried),
                                }
                            ):
                                research_result = await Runner.run(
                                    self._research_agent,
                                    switch_prompt,
                                    context=context,
                                    hooks=switch_research_hooks,
                                    max_turns=6
                                )
                            research_text = str(research_result.final_output) if research_result.final_output else research_text
                        except LoopDetectedError as e:
                            print(f"[Pipeline] WARN: Technique switch research loop: {e}", file=sys.stderr)
                            # Keep existing research_text if new research loops
                        print(f"[Pipeline] New research: {research_text[:80]}...", file=sys.stderr)
                        print(f"[Pipeline] Switch research hooks stats: {switch_research_hooks.get_stats()}", file=sys.stderr)
                        # Next iteration will get modified feedback to try different approach

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

    async def resume_session(self, session_id: str) -> SessionState:
        """
        Resume a paused or incomplete session.

        Args:
            session_id: ID of session to resume

        Returns:
            Updated SessionState
        """
        if not self._initialized:
            await self.initialize()

        # Load session
        session = self._persistence.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status in [SessionStatus.PASSED, SessionStatus.CANCELLED]:
            return session  # Already complete

        # Create context from session
        context = SharedContext(session=session)

        # Build resumption prompt
        prompt = self._build_resumption_prompt(context)

        # Run orchestrator
        # gpt-5.2 with high reasoning effort needs more turns than default 10
        # Using trace() for end-to-end observability across resumed iterations
        try:
            with trace(f"VFX Resume: {session.session_id}"):
                result = await Runner.run(
                    self._orchestrator,
                    prompt,
                    context=context,  # Typed SharedContext for RunContextWrapper access
                    max_turns=25  # Increased for gpt-5.2 reasoning
                )

            session = self._parse_result(result, context)

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.current_issues.append(f"Resumption error: {str(e)}")

        # Save updated state
        self._persistence.save_session(session)

        return session

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
    # Phase 2: Use the pipeline-based orchestration with Coordinators
    return await orchestrator.create_asset_pipeline(request)


async def resume_vfx_session(session_id: str) -> SessionState:
    """
    Convenience function to resume a VFX session.

    Args:
        session_id: ID of session to resume

    Returns:
        Updated SessionState
    """
    orchestrator = await get_orchestrator()
    return await orchestrator.resume_session(session_id)


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
