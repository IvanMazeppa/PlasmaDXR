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
from agents.agent_output import AgentOutputSchema
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
    suggested_modifications: List[str] = Field(default_factory=list, description="Specific changes for next iteration")

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
from specialized_agents import (
    create_script_writer,
    create_executor,
    create_quality_analyst,
    create_learning_agent,
)
# DocsExpert now uses in-process function_tools instead of MCP client connections.
# This fixes the anyio TaskGroup conflict that previously blocked MCP tool handlers.
from specialized_agents.docs_expert import create_docs_expert
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
        self._script_writer = create_script_writer().clone(
            handoffs=[return_from_script_writer],
            instructions=create_script_writer().instructions + """

CRITICAL: After generating and validating a script, use script_complete_execute_next to hand off.
This signals the orchestrator to IMMEDIATELY run the script - no more research needed."""
        )

        self._executor = create_executor().clone(
            handoffs=[return_from_executor],
            instructions=create_executor().instructions + """

CRITICAL: After executing a script (success or failure), use execution_complete_evaluate_next to hand off.
This signals the orchestrator to IMMEDIATELY evaluate the render - no more research needed."""
        )

        self._quality_analyst = create_quality_analyst().clone(
            handoffs=[return_from_quality_analyst],
            instructions=create_quality_analyst().instructions + """

CRITICAL: After evaluating render quality, use evaluation_complete_learn_next to hand off.
This signals the orchestrator to IMMEDIATELY record learnings - no more research needed."""
        )

        self._learning_agent = create_learning_agent().clone(
            handoffs=[return_from_learning_agent],
            instructions=create_learning_agent().instructions + """

CRITICAL: After recording experiments, use learning_complete_decide_next to hand off.
This signals the orchestrator to decide: if quality passed, finish. If not, modify script."""
        )

        self._docs_expert = create_docs_expert().clone(
            handoffs=[return_to_orchestrator],
            instructions=create_docs_expert().instructions + """

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
        base_script_writer = create_script_writer()
        self._script_agent_standalone = Agent[SharedContext](
            name="Script Writer",
            instructions=prompt_with_handoff_instructions(base_script_writer.instructions + """

## EFFICIENCY REQUIREMENT - CRITICAL
You have LIMITED turns (max 10). Be efficient:
1. Call recommend_technique ONCE to pick approach
2. Call generate_script ONCE to create initial script
3. Call validate_script ONCE
4. If validation fails, call modify_script AT MOST 2 times
5. IMMEDIATELY return ScriptOutput - do NOT keep iterating

Total tool calls should be 4-6, not 10+. Return output even if imperfect.

## Domain Physics Context
- sun/star/nebula effects: SPACE environment → set gravity=0, buoyancy=0 (beta=0)
- explosion/fire/smoke effects: EARTH environment → normal gravity/buoyancy OK
Apply correct physics parameters based on effect type.

## Output Requirements
After generating and validating the script, return a structured ScriptOutput with:
- script_path: Absolute path to the generated/modified script
- technique_used: The approach/technique used
- parameters_set: Key parameters configured in the script
- validation_passed: Whether validation succeeded
- validation_errors: Any validation errors encountered

IMPORTANT: Always return the script_path even if validation fails. Do NOT loop indefinitely."""),
            model=base_script_writer.model,
            model_settings=base_script_writer.model_settings,
            output_type=AgentOutputSchema(ScriptOutput, strict_json_schema=False),
            tools=base_script_writer.tools,
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
        base_quality = create_quality_analyst()
        self._quality_agent_standalone = Agent[SharedContext](
            name="Quality Analyst",
            instructions=prompt_with_handoff_instructions(base_quality.instructions + """

## Domain Physics Awareness
Check for physics anomalies based on effect type:
- sun/star/nebula: Should be STATIC or have internal motion only (no drift/rise)
  If flame/volume moves upward → wrong: buoyancy should be 0 for space
- explosion/fire: Can have upward motion (buoyancy is expected on Earth)

Flag physics violations as HIGH PRIORITY issues.

## Output Requirements (LLM-as-Judge Pattern)
After evaluating render quality, return a structured QualityOutput with:
- overall_score: Quality score 0-100
- passed: Whether quality threshold was met
- primary_issue: The most critical issue to fix (if any)
- issues: List of all identified issues (include physics violations)
- suggestions: Specific parameter changes to try
- vision_assessment: Detailed visual quality description
- reference_similarity: Similarity to reference image (if available)

Be a STRICT judge - only pass renders that truly meet quality standards."""),
            model=base_quality.model,
            model_settings=base_quality.model_settings,
            output_type=AgentOutputSchema(QualityOutput, strict_json_schema=False),
            tools=base_quality.tools,
        )

        # Learning Agent with structured output
        base_learning = create_learning_agent()
        self._learning_agent_standalone = Agent[SharedContext](
            name="Learning Agent",
            instructions=prompt_with_handoff_instructions(base_learning.instructions + """

## CRITICAL: TURN BUDGET (MAX 8 TURNS - HARD LIMIT)
You MUST complete in 3-4 turns or the pipeline FAILS. Follow this EXACT sequence:

Turn 1: Query knowledge (query_knowledge_base) - MAX 2 parallel calls
Turn 2: Record experiment (record_experiment_result) - CALL EXACTLY ONCE
Turn 3: Return LearningOutput structured response

RULES:
- DO NOT make more than 3 tool calls total
- DO NOT call record_experiment_result more than ONCE (even if it returns an error)
- DO NOT call start_experiment_session or record_baseline (handled elsewhere)
- If record_experiment_result fails, STILL return LearningOutput (set experiment_recorded=False)
- NEVER retry failed tool calls

## Domain Physics Context
- sun/star/nebula effects: SPACE environment → gravity=0, buoyancy=0 (beta=0)
- explosion/fire/smoke effects: EARTH environment → normal gravity/buoyancy
If a sun/star is drifting upward, the fix is: beta=0 (disable buoyancy)

## Output Requirements (RETURN AFTER 1 record_experiment_result call)
Return a structured LearningOutput with:
- experiment_recorded: True (you recorded it)
- pattern_extracted: bool
- pattern_id: str or None
- next_action: 'iterate' | 'switch_technique' | 'complete'
- suggested_modifications: List of specific parameter changes (include beta=0 for space effects)"""),
            model=base_learning.model,
            model_settings=base_learning.model_settings,
            output_type=AgentOutputSchema(LearningOutput, strict_json_schema=False),
            tools=base_learning.tools,
        )

        self._initialized = True
        print("[Orchestrator] Initialization complete (5 agents + 5 standalone ready)", file=sys.stderr)

    async def create_asset(self, request: AssetRequest) -> SessionState:
        """
        Main entry point for asset generation.

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

                research_result = await Runner.run(
                    self._research_agent,
                    research_prompt,
                    context=context,
                    max_turns=8  # Limit research turns
                )
                # Research agent outputs text summary (no output_type for tool-using agents)
                research_text = str(research_result.final_output) if research_result.final_output else "No research findings"
                print(f"[Pipeline] Research complete: {research_text[:80]}...", file=sys.stderr)

                # ====== ITERATION LOOP ======
                iteration = 0
                previous_script: Optional[ScriptOutput] = None
                previous_score = 0.0
                quality: Optional[QualityOutput] = None
                learning: Optional[LearningOutput] = None

                while iteration < request.max_iterations:
                    iteration += 1
                    session.current_iteration = iteration
                    print(f"\n[Pipeline] ====== ITERATION {iteration}/{request.max_iterations} ======", file=sys.stderr)

                    # ====== PHASE 1: SCRIPT GENERATION ======
                    print(f"[Pipeline] PHASE 1: Script Writer", file=sys.stderr)
                    if iteration == 1:
                        script_prompt = f"""Generate a Blender Python script for {request.effect_type.value} VFX.

## Research Findings
{research_text}

## Parameters
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}
- Description: {request.description}
- Resolution: {request.resolution}
- Frames: {request.frame_start}-{request.frame_end}

Generate a complete, validated script. Return the script_path in your output."""
                    else:
                        # Include quality feedback and learning suggestions
                        feedback_parts = []
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

                        script_prompt = f"""Modify the existing script to fix quality issues.

## Current Script
Path: {previous_script.script_path if previous_script else 'Unknown'}
Technique: {previous_script.technique_used if previous_script else 'Unknown'}

{chr(10).join(feedback_parts)}

## Target
Previous Score: {previous_score:.1f}
Target Score: {request.quality_threshold}

Modify the script to address the issues. Use patterns from library if available."""

                    script_result = await Runner.run(
                        self._script_agent_standalone,
                        script_prompt,
                        context=context,
                        max_turns=10
                    )
                    # Structured output: ScriptOutput
                    script: ScriptOutput = script_result.final_output

                    # Always capture script_path if available (for subsequent iterations)
                    if script.script_path:
                        session.current_script_path = script.script_path
                        previous_script = script
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

                    # ====== PHASE 2: EXECUTION ======
                    print(f"[Pipeline] PHASE 2: Executor", file=sys.stderr)
                    exec_prompt = f"""Execute the Blender script and render the VFX asset.

Script Path: {script.script_path}
Frames: {request.frame_start}-{request.frame_end}

Run the script and report results."""

                    exec_result = await Runner.run(
                        self._executor_agent_standalone,
                        exec_prompt,
                        context=context,
                        max_turns=6
                    )
                    # Structured output: ExecutionOutput
                    execution: ExecutionOutput = exec_result.final_output

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

                    eval_result = await Runner.run(
                        self._quality_agent_standalone,
                        eval_prompt,
                        context=context,
                        max_turns=6
                    )
                    # Structured output: QualityOutput
                    quality = eval_result.final_output

                    previous_score = quality.overall_score
                    print(f"[Pipeline] Score: {quality.overall_score:.1f} | Passed: {quality.passed}", file=sys.stderr)
                    if quality.primary_issue:
                        print(f"[Pipeline] Issue: {quality.primary_issue[:60]}...", file=sys.stderr)

                    # Update session
                    session.best_score = max(session.best_score, quality.overall_score)
                    if quality.overall_score == session.best_score:
                        session.best_iteration = iteration
                        session.final_render_path = execution.render_path

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
                    learn_prompt = f"""Record the experiment results.

Iteration: {iteration}
Script: {script.script_path}
Technique: {script.technique_used}
Render: {execution.render_path}
Score: {quality.overall_score:.1f}
Passed: {quality.passed}
Primary Issue: {quality.primary_issue or 'None'}

If this iteration improved significantly, extract the pattern.
Recommend next action: 'iterate' (continue improving), 'switch_technique' (try different approach), or 'complete' (quality achieved)."""

                    learn_result = await Runner.run(
                        self._learning_agent_standalone,
                        learn_prompt,
                        context=context,
                        max_turns=8
                    )
                    # Structured output: LearningOutput
                    learning = learn_result.final_output
                    print(f"[Pipeline] Learning: next_action={learning.next_action}", file=sys.stderr)

                    # ====== QUALITY GATE ======
                    if quality.passed or learning.next_action == 'complete':
                        print(f"\n[Pipeline] ✓ QUALITY GATE PASSED at iteration {iteration}", file=sys.stderr)
                        session.status = SessionStatus.PASSED
                        break

                    # Handle technique switching recommendation
                    if learning.next_action == 'switch_technique':
                        print(f"[Pipeline] Learning Agent recommends switching technique", file=sys.stderr)
                        # Research agent already found alternatives in research_text
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
    return await orchestrator.create_asset(request)


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
