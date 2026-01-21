"""
Dynamic Instructions Module for Self-Learning VFX Orchestrator.

This module implements the SDK pattern of dynamic instructions - functions that
generate agent instructions at runtime based on accumulated knowledge from the
knowledge base.

KEY PRINCIPLE: No hardcoded rules. Rules emerge from experimentation and are
injected only when validated (success_rate > threshold).

Usage:
    from tools.dynamic_instructions import dynamic_script_writer_instructions

    agent = Agent[SharedContext](
        name="Script Writer",
        instructions=dynamic_script_writer_instructions,  # Function, not string!
        ...
    )
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper
    from models.shared_context import SharedContext

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
if str(ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_ROOT))

# Lazy import to avoid circular dependencies
_knowledge_base = None


def _get_knowledge_base():
    """Lazy import of knowledge base query function."""
    global _knowledge_base
    if _knowledge_base is None:
        try:
            from tools.experiment_tracker_tools import _query_knowledge_base_impl
            _knowledge_base = _query_knowledge_base_impl
        except ImportError:
            # Fallback: return empty results
            _knowledge_base = lambda q: json.dumps({"results": []})
    return _knowledge_base


def query_validated_learnings(
    effect_type: str,
    category: str = "physics",
    min_success_rate: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Query knowledge base for validated learnings.

    Only returns learnings that have been tested and have a success rate
    above the threshold. This prevents injecting untested rules.

    Args:
        effect_type: Type of effect (sun, explosion, fire, etc.)
        category: Category of learning (physics, visual, parameter)
        min_success_rate: Minimum success rate to include (0.0-1.0)

    Returns:
        List of validated learnings with their success rates
    """
    kb_query = _get_knowledge_base()

    try:
        result = kb_query(f"{category} rules for {effect_type}")
        data = json.loads(result) if isinstance(result, str) else result
        results = data.get("results", [])

        # Filter by success rate
        validated = []
        for learning in results:
            success_rate = learning.get("success_rate", 0)
            if success_rate >= min_success_rate:
                validated.append(learning)

        return validated
    except Exception:
        return []


def format_learnings_as_instructions(
    learnings: List[Dict[str, Any]],
    header: str = "Validated Rules (learned from experiments)"
) -> str:
    """
    Format learnings as instruction text.

    Args:
        learnings: List of learning dicts with 'rule' and 'success_rate' keys
        header: Section header

    Returns:
        Formatted instruction string
    """
    if not learnings:
        return ""

    lines = [f"\n## {header}"]
    for learning in learnings:
        rule = learning.get("rule", learning.get("description", "Unknown rule"))
        success_rate = learning.get("success_rate", 0)
        effect_type = learning.get("effect_type", "")

        if effect_type:
            lines.append(f"- [{effect_type}] {rule} (success: {success_rate:.0%})")
        else:
            lines.append(f"- {rule} (success: {success_rate:.0%})")

    return "\n".join(lines)


# =============================================================================
# BASE INSTRUCTIONS (Generic, no hardcoded rules)
# =============================================================================

SCRIPT_WRITER_BASE_INSTRUCTIONS = """## ROLE
Generate/modify Blender 5.0 Mantaflow scripts. Research-first approach for unknown issues.

## TURN BUDGET: MAX 5 TURNS
T1: recommend_technique OR analyze feedback
T2: generate_script OR modify_script (SINGLE CALL with ALL changes batched)
T3: validate_script
T4: Return ScriptOutput

CRITICAL: Batch ALL parameter changes into ONE modify_script call. Never call modify_script multiple times.

## NEW SCRIPT WORKFLOW
1. recommend_technique(effect_type, description)
2. generate_script(effect_type, description, output_name, technique_name=recommended)
3. validate_script(script_path)
4. Return {script_path, technique_used, parameters, validation}

## MODIFICATION WORKFLOW
1. Parse quality feedback -> identify ALL visual issues
2. Research unknown issues: search_blender_api_by_intent(issue, "fluid")
3. Map issues -> parameters (batch ALL into single dict)
4. modify_script(script_path, modifications_json, output_name) - ONCE
5. validate_script -> Return

## RESEARCH-FIRST APPROACH
Unknown issue? DON'T guess. Search first:
- search_blender_api_by_intent("what causes upward motion", "fluid")
- semantic_search_blender_docs("emitter position fluid simulation")

## PARAMETER REFERENCE (Blender 5.0 ranges)
vorticity: 0-4, flame_vorticity: 0-2, flame_smoke: 0-8, burning_rate: 0.01-4
resolution_max: 32-512, alpha: -5 to 5, beta: -5 to 5, temperature: -10 to 10
fuel_amount: 0-10, velocity_normal: -100 to 100

## PATH HANDLING
Use EXACT paths from tool responses. Never modify/prefix paths.

## OUTPUT
Return ScriptOutput with: script_path, technique_used, parameters, validation, warnings
"""


QUALITY_ANALYST_BASE_INSTRUCTIONS = """## ROLE
Evaluate VFX render quality. Be a STRICT judge - only pass renders that truly meet standards.

## TURN BUDGET: MAX 3 TURNS
T1: analyze_with_vision + find_reference_images (parallel)
T2: compare_to_reference (if reference available)
T3: Return QualityOutput

## EVALUATION CRITERIA
1. Visual Quality: Does it look like the intended effect?
2. Technical Quality: No artifacts, proper lighting, correct scale
3. Animation Quality: Appropriate motion for the effect type
4. Reference Match: How close to reference image (if available)?

## ANOMALY DETECTION
When you observe unexpected behavior:
1. Use observe_physics_anomaly() to record it
2. Include: effect_type, what you expected, what you saw, suspected parameters
3. This feeds the learning system - your observations improve future iterations

## SCORING
- 0-20: Critical failures (black screen, no render, crashes)
- 20-40: Major issues (wrong effect type, severe artifacts)
- 40-60: Moderate issues (visual problems, wrong motion)
- 60-80: Minor issues (tweaking needed)
- 80-100: Production quality

## OUTPUT
Return QualityOutput with:
- overall_score: 0-100
- passed: score >= threshold AND no critical issues
- primary_issue: Most important thing to fix
- issues: All identified problems
- suggestions: Specific parameter changes to try
- vision_assessment: Detailed visual description
"""


LEARNING_AGENT_BASE_INSTRUCTIONS = """## ROLE
Record experiments, extract patterns, suggest fixes from accumulated knowledge.

## TURN BUDGET: MAX 3 TURNS
T1: query_knowledge_base + search_code_patterns (parallel)
T2: record_experiment_result (ONCE)
T3: Return LearningOutput

CRITICAL: Call record_experiment_result ONCE. Never retry on error.

## BEFORE CHANGES
1. search_code_patterns(issue) -> find known fixes
2. query_knowledge_base(issue) -> check past learnings
3. get_parameter_knowledge(param) -> accumulated wisdom

## AFTER EXPERIMENT
record_experiment_result() with:
- issue_addressed, parameters_changed, score_before/after
- success: score_delta > 0
- observed_effects: [{metric, direction, magnitude, expected, side_effect}]
- learnings: what we learned

IF score_delta >= 5 (success):
-> extract_successful_pattern(script_path, issue_type, params_changed, improvement)
-> record_code_pattern(pattern_name, issue_type, code_snippet, params)

## PHYSICS OBSERVATIONS
When Quality Analyst reports physics anomalies via observe_physics_anomaly():
1. Correlate with current parameters
2. Record the {params} -> {behavior} relationship
3. If pattern emerges, add as learning with appropriate success_rate

## KNOWLEDGE STRUCTURE
- Rules: "adjust X when changing Y" (with success_rate)
- Warnings: "X without Y causes Z" (with confidence)
- Patterns: reusable fixes with tracked success rates

## OUTPUT (LearningOutput)
- experiment_recorded: bool
- pattern_extracted: bool (True if successful pattern found)
- pattern_id: str or None
- next_action: "iterate" | "switch_technique" | "complete"
- suggested_modifications: List[str] - TEXT descriptions for context (e.g., "reduce emission intensity")
- parameter_modifications: Dict[str, Any] - CONCRETE VALUES for direct application:
  Example: {"temperature": 3.0, "density": 5.0, "blackbody_intensity": 2.0}
  CRITICAL: Include ACTUAL NUMBERS, not descriptions. These are applied directly to scripts.

## PARAMETER RECOMMENDATIONS
When recommending parameter changes, ALWAYS provide:
1. suggested_modifications: Human-readable explanation of what/why
2. parameter_modifications: Machine-readable {param: value} dict with EXACT values

Common Blender Mantaflow parameters:
- temperature: -10.0 to 10.0 (flow temperature)
- density: 0.0 to 10.0 (volume density)
- fuel_amount: 0.0 to 10.0 (fuel for fire)
- burning_rate: 0.01 to 4.0 (combustion speed)
- flame_smoke: 0.0 to 8.0 (smoke from flames)
- flame_vorticity: 0.0 to 2.0 (flame turbulence)
- beta: -5.0 to 5.0 (density buoyancy - use 0.0 for space)
- alpha: -5.0 to 5.0 (thermal buoyancy - use 0.0 for space)
- resolution_max: 32 to 512 (simulation resolution)

Shader parameters (Principled Volume):
- blackbody_intensity: 0.0 to 20.0 (emission brightness)
- emission_strength: 0.0 to 100.0 (glow intensity)
"""


# =============================================================================
# DYNAMIC INSTRUCTION FUNCTIONS
# =============================================================================

def dynamic_script_writer_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Script Writer instructions dynamically based on accumulated knowledge.

    This function is called at the START of each agent run, allowing us to inject
    learnings that have been validated through experimentation.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string with validated learnings injected
    """
    base = SCRIPT_WRITER_BASE_INSTRUCTIONS

    # Try to get effect type from context
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context.session, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception:
        pass

    # Query validated learnings
    learnings = []

    if effect_type:
        # Effect-specific learnings
        learnings.extend(query_validated_learnings(effect_type, "physics", 0.7))
        learnings.extend(query_validated_learnings(effect_type, "visual", 0.7))

    # General learnings (apply to all effects)
    general_learnings = query_validated_learnings("general", "physics", 0.8)
    learnings.extend(general_learnings)

    # Add learnings to instructions
    if learnings:
        base += format_learnings_as_instructions(
            learnings,
            f"Validated Rules for {effect_type or 'VFX'} (learned from experiments)"
        )
    else:
        base += """

## Physics Rules
No validated rules yet for this effect type. Use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from your experiments.

IMPORTANT: Do NOT assume physics settings. Let the simulation run with defaults,
then observe what happens. If something is wrong, the Quality Analyst will flag it
and the Learning Agent will record the relationship.
"""

    return base


def dynamic_quality_analyst_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Quality Analyst instructions dynamically.

    Includes known physics behaviors to watch for based on past observations.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string
    """
    base = QUALITY_ANALYST_BASE_INSTRUCTIONS

    # Try to get effect type from context
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context.session, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception:
        pass

    # Query known physics behaviors to watch for
    if effect_type:
        physics_learnings = query_validated_learnings(effect_type, "physics", 0.5)

        if physics_learnings:
            base += f"\n\n## Known Behaviors for {effect_type} (from past experiments)"
            for learning in physics_learnings:
                rule = learning.get("rule", "")
                success_rate = learning.get("success_rate", 0)
                if success_rate < 0.5:
                    # This is a known problem to watch for
                    base += f"\n- WATCH FOR: {rule}"
                else:
                    # This is expected good behavior
                    base += f"\n- EXPECTED: {rule}"

    return base


def dynamic_learning_agent_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Learning Agent instructions dynamically.

    Includes current knowledge stats and areas needing more data.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string
    """
    base = LEARNING_AGENT_BASE_INSTRUCTIONS

    # Try to get effect type from context
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception:
        pass

    # Add context about current knowledge state
    if effect_type:
        learnings = query_validated_learnings(effect_type, "physics", 0.0)  # All learnings

        validated = [l for l in learnings if l.get("success_rate", 0) >= 0.7]
        needs_data = [l for l in learnings if 0.3 <= l.get("success_rate", 0) < 0.7]

        base += f"\n\n## Knowledge State for {effect_type}"
        base += f"\n- Validated rules: {len(validated)}"
        base += f"\n- Needs more data: {len(needs_data)}"

        if needs_data:
            base += "\n\n### Rules Needing Validation (run more experiments)"
            for learning in needs_data[:3]:  # Top 3
                rule = learning.get("rule", "")
                success_rate = learning.get("success_rate", 0)
                base += f"\n- {rule} (current: {success_rate:.0%})"

    return base


# =============================================================================
# STATIC FALLBACK INSTRUCTIONS (for when context is unavailable)
# =============================================================================

def get_script_writer_instructions_static() -> str:
    """Get static Script Writer instructions (no dynamic KB query)."""
    return SCRIPT_WRITER_BASE_INSTRUCTIONS + """

## Physics Rules
No validated rules available. Use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from experiments.
"""


def get_quality_analyst_instructions_static() -> str:
    """Get static Quality Analyst instructions."""
    return QUALITY_ANALYST_BASE_INSTRUCTIONS


def get_learning_agent_instructions_static() -> str:
    """Get static Learning Agent instructions."""
    return LEARNING_AGENT_BASE_INSTRUCTIONS
