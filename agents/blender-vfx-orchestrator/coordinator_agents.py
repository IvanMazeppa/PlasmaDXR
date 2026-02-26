"""
Coordinator agent factories for the VFX pipeline.

These lightweight agents are called at DECISION POINTS only, not for every step.
Python controls the pipeline sequence; coordinators provide intelligent
decision-making for: technique selection, modification strategy, quality gates.

Extracted from orchestrator.py during Phase 2C decomposition.
"""

from __future__ import annotations

from agents import Agent, ModelSettings
from agents.agent_output import AgentOutputSchema

from config import get_config
from models.shared_context import SharedContext
from models.pipeline_models import (
    TechniqueDecision,
    ModificationDecision,
    QualityDecision,
)

# Tools used by coordinators
from tools.semantic_docs_tools import blender_doc_search_bundle
from tools.code_pattern_tools import search_code_patterns, list_patterns_by_effect
from tools.proactive_research_tools import (
    pre_iteration_research as pre_iteration_research_tool,
    evaluate_escape_velocity,
)
from tools.artifact_tools import read_artifact

# Coordinator guardrails
from guardrails import (
    validate_technique_decision,
    validate_modification_decision,
    validate_quality_decision,
)


# =============================================================================
# COORDINATOR AGENT INSTRUCTIONS
# =============================================================================

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
# FACTORY FUNCTIONS
# =============================================================================


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
- Use read_artifact() to get full details from scorecard/quality artifact paths

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
            read_artifact,  # Phase 2B-8: Read scorecard/quality artifacts for full details
        ],
        output_type=AgentOutputSchema(QualityDecision, strict_json_schema=False),
        # Phase 3: Validate decision consistency (e.g., passed=True must have next_action='complete')
        output_guardrails=[validate_quality_decision],
        # Phase 2A-2 NOTE: StopAtTools not used here because the agent needs to
        # synthesize evaluate_escape_velocity results into QualityDecision structured
        # output. stop_on_first_tool would bypass the structured output processing.
    )
