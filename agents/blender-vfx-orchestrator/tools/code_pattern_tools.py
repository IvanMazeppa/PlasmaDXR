"""
Code Pattern Function Tools for Blender VFX Orchestrator.

Implements Strategy 4: Code Pattern Memory - Agent-facing tools.

These tools provide agents with the ability to:
- Record successful code patterns from experiments
- Search for patterns that might fix an issue
- Apply stored patterns to scripts
- Track pattern effectiveness

Unlike parameter-based knowledge that tracks metadata, these tools work
with actual Python code that can be retrieved and applied.
"""

from __future__ import annotations

import json
import sys
from typing import Optional, List

from agents import function_tool, RunContextWrapper

# Import SharedContext directly (not under TYPE_CHECKING) because
# @function_tool decorator evaluates type hints at runtime
from models.shared_context import SharedContext

from utils.code_pattern_memory import get_pattern_memory, CodePattern


@function_tool
def record_code_pattern(
    wrapper: RunContextWrapper[SharedContext],
    issue: str,
    code_snippet: str,
    improvement: float,
    effect_type: str = "",
    experiment_id: str = "",
    pattern_name: str = "",
    context_before: str = "",
    context_after: str = ""
) -> str:
    """
    Record a successful code pattern after an experiment improves quality.

    Call this AFTER a script modification achieves measurable improvement.
    The pattern will be stored and can be retrieved later for similar issues.

    WHEN TO USE:
    - After any modification achieves >= 5 point improvement
    - When you discover a novel fix that might help future scripts
    - After successfully resolving a persistent issue

    NOTE: With RunContextWrapper, effect_type and experiment_id are auto-populated from context.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        issue: Description of the issue this pattern fixes
               Example: "smoke dissipates too quickly"
        code_snippet: The actual Python code that fixed the issue
                     Example: "domain.dissolve_speed = 5\\ndomain.flame_smoke = 3.0"
        improvement: Score improvement achieved (e.g., 12.5 for +12.5 points)
        effect_type: Type of effect (auto-populated from context if empty)
        experiment_id: ID of the experiment (auto-populated from session_id if empty)
        pattern_name: Optional human-readable name (auto-generated if empty)
        context_before: Code context before the change (for understanding)
        context_after: Code context after the change (for understanding)

    Returns:
        JSON with:
        - success: Whether pattern was recorded
        - pattern_id: Unique ID for retrieving this pattern
        - is_update: Whether this updated an existing similar pattern
        - message: Description of what happened
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        if not effect_type and session.request:
            effect_type = session.request.effect_type.value
        if not experiment_id:
            experiment_id = session.session_id
        print(f"[record_code_pattern] Using context: effect_type={effect_type}, experiment_id={experiment_id}", file=sys.stderr)

    memory = get_pattern_memory()

    try:
        # Check if similar pattern exists before recording
        existing = memory._find_similar(code_snippet)
        is_update = existing is not None

        pattern_id = memory.record_successful_pattern(
            issue=issue,
            code_snippet=code_snippet,
            effect_type=effect_type,
            improvement=improvement,
            experiment_id=experiment_id,
            name=pattern_name if pattern_name else None,
            context_before=context_before,
            context_after=context_after
        )

        pattern = memory.get_pattern(pattern_id)

        return json.dumps({
            "success": True,
            "pattern_id": pattern_id,
            "pattern_name": pattern.name if pattern else "unknown",
            "is_update": is_update,
            "confidence": pattern.confidence if pattern else 50,
            "usage_count": pattern.usage_count if pattern else 1,
            "message": f"{'Updated existing' if is_update else 'Created new'} pattern: {pattern_id}"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to record pattern: {e}"
        })


@function_tool
def search_code_patterns(
    wrapper: RunContextWrapper[SharedContext],
    issue: str,
    effect_type: str = "",
    min_confidence: float = 30.0,
    max_results: int = 5
) -> str:
    """
    Search for code patterns that might fix an issue.

    Use this BEFORE attempting to fix an issue to see if there's
    already a proven solution in the pattern library.

    WHEN TO USE:
    - At the start of each iteration, search for relevant patterns
    - When stuck on a persistent issue
    - Before trying a novel approach (check if it's already been tried)

    NOTE: With RunContextWrapper, effect_type is auto-populated from context if not provided.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        issue: Description of the issue to fix
               Example: "smoke lacks density", "explosion too dim"
        effect_type: Optional filter by effect type (auto-populated from context)
        min_confidence: Minimum confidence threshold (0-100, default 30)
        max_results: Maximum patterns to return (default 5)

    Returns:
        JSON with:
        - patterns_found: Number of matching patterns
        - patterns: List of patterns, each with:
            - pattern_id: ID for applying this pattern
            - name: Human-readable name
            - code_snippet: The actual Python code
            - confidence: How likely to help (0-100)
            - average_improvement: Historical improvement score
            - usage_count: How many times this has been used
            - success_rate: Historical success rate (0-1)
        - recommendation: Which pattern to try first (if any)
    """
    # Auto-populate effect_type from context if not provided
    context = wrapper.context
    if not effect_type and context and hasattr(context, 'session'):
        session = context.session
        if session.request:
            effect_type = session.request.effect_type.value
            print(f"[search_code_patterns] Auto-populated effect_type={effect_type} from context", file=sys.stderr)

    memory = get_pattern_memory()

    try:
        patterns = memory.retrieve_patterns_for_issue(
            issue=issue,
            effect_type=effect_type if effect_type else None,
            min_confidence=min_confidence,
            max_results=max_results
        )

        if not patterns:
            return json.dumps({
                "patterns_found": 0,
                "patterns": [],
                "recommendation": "No matching patterns found. Try a novel approach and record it if successful."
            })

        pattern_data = []
        for p in patterns:
            pattern_data.append({
                "pattern_id": p.pattern_id,
                "name": p.name,
                "issue_category": p.issue_category,
                "code_snippet": p.code_snippet,
                "confidence": p.confidence,
                "average_improvement": p.average_improvement,
                "usage_count": p.usage_count,
                "success_rate": p.success_rate,
                "effect_types": p.effect_types,
                "parameters_affected": p.parameters_affected
            })

        # Recommendation based on confidence
        best = patterns[0]
        if best.confidence >= 70:
            recommendation = f"High confidence: Apply pattern '{best.name}' (confidence: {best.confidence:.0f}%)"
        elif best.confidence >= 50:
            recommendation = f"Moderate confidence: Consider pattern '{best.name}' (confidence: {best.confidence:.0f}%)"
        else:
            recommendation = f"Low confidence: Pattern '{best.name}' available but may not help (confidence: {best.confidence:.0f}%)"

        return json.dumps({
            "patterns_found": len(patterns),
            "patterns": pattern_data,
            "recommendation": recommendation
        })

    except Exception as e:
        return json.dumps({
            "patterns_found": 0,
            "patterns": [],
            "error": str(e),
            "recommendation": "Error searching patterns. Proceed with novel approach."
        })


@function_tool
def get_pattern_code(pattern_id: str) -> str:
    """
    Get the full code and details for a specific pattern.

    Use after search_code_patterns to get complete code for application.

    Args:
        pattern_id: The pattern ID from search results

    Returns:
        JSON with:
        - found: Whether pattern exists
        - pattern_id: The pattern ID
        - name: Human-readable name
        - code_snippet: The full Python code to apply
        - context_before: Code context before the change
        - context_after: Code context after the change
        - parameters_affected: List of parameters this modifies
        - blender_apis_used: Blender APIs referenced
        - usage_instructions: How to apply this pattern
    """
    memory = get_pattern_memory()

    pattern = memory.get_pattern(pattern_id)

    if not pattern:
        return json.dumps({
            "found": False,
            "error": f"Pattern {pattern_id} not found"
        })

    return json.dumps({
        "found": True,
        "pattern_id": pattern.pattern_id,
        "name": pattern.name,
        "issue_category": pattern.issue_category,
        "code_snippet": pattern.code_snippet,
        "context_before": pattern.context_before,
        "context_after": pattern.context_after,
        "parameters_affected": pattern.parameters_affected,
        "blender_apis_used": pattern.blender_apis_used,
        "confidence": pattern.confidence,
        "average_improvement": pattern.average_improvement,
        "usage_count": pattern.usage_count,
        "success_rate": pattern.success_rate,
        "usage_instructions": f"""
To apply this pattern:
1. Locate the relevant section in your script (usually domain settings)
2. Add or modify the following code:

{pattern.code_snippet}

Parameters affected: {', '.join(pattern.parameters_affected)}
Expected improvement: ~{pattern.average_improvement:.1f} points
"""
    })


def _report_pattern_outcome_impl(
    pattern_id: str,
    success: bool,
    improvement: float = 0.0,
    notes: str = ""
) -> str:
    """
    Direct callable implementation of report_pattern_outcome.

    For use by orchestrator without @function_tool wrapper.
    """
    memory = get_pattern_memory()

    pattern_before = memory.get_pattern(pattern_id)
    if not pattern_before:
        return json.dumps({
            "recorded": False,
            "error": f"Pattern {pattern_id} not found"
        })

    result = memory.record_pattern_outcome(
        pattern_id=pattern_id,
        success=success,
        improvement=improvement
    )

    pattern_after = memory.get_pattern(pattern_id)

    return json.dumps({
        "recorded": result,
        "pattern_id": pattern_id,
        "outcome": "success" if success else "failure",
        "improvement": improvement,
        "new_confidence": pattern_after.confidence if pattern_after else 0,
        "new_success_rate": pattern_after.success_rate if pattern_after else 0,
        "total_uses": pattern_after.usage_count if pattern_after else 0,
        "notes": notes,
        "message": f"Pattern outcome recorded. New confidence: {pattern_after.confidence:.0f}%" if pattern_after else "Error updating pattern"
    })


@function_tool
def report_pattern_outcome(
    pattern_id: str,
    success: bool,
    improvement: float = 0.0,
    notes: str = ""
) -> str:
    """
    Report the outcome after applying a pattern.

    IMPORTANT: Always call this after applying a pattern from the library.
    This feedback improves future pattern recommendations.

    Args:
        pattern_id: The pattern that was applied
        success: Whether the pattern helped (True) or not (False)
        improvement: Score improvement achieved (if successful)
        notes: Optional notes about what happened

    Returns:
        JSON with:
        - recorded: Whether outcome was recorded
        - pattern_id: The pattern ID
        - new_confidence: Updated confidence score
        - new_success_rate: Updated success rate
    """
    return _report_pattern_outcome_impl(pattern_id, success, improvement, notes)


@function_tool
def get_pattern_library_stats() -> str:
    """
    Get statistics about the code pattern library.

    Use to understand what patterns are available and their effectiveness.

    Returns:
        JSON with:
        - total_patterns: Total number of stored patterns
        - categories: Count of patterns per issue category
        - effect_types: Count of patterns per effect type
        - average_confidence: Mean confidence across all patterns
        - high_confidence_patterns: Count of patterns with >= 70% confidence
        - total_usage: Total times patterns have been applied
    """
    memory = get_pattern_memory()

    try:
        stats = memory.get_statistics()
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "total_patterns": 0
        })


@function_tool
def list_patterns_by_effect(
    wrapper: RunContextWrapper[SharedContext],
    effect_type: str = "",
    min_confidence: float = 0.0
) -> str:
    """
    List all patterns available for a specific effect type.

    Use to see what proven fixes exist for a type of effect.

    NOTE: With RunContextWrapper, effect_type is auto-populated from context if not provided.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        effect_type: Effect type to filter by (auto-populated from context if empty)
        min_confidence: Minimum confidence threshold (default 0)

    Returns:
        JSON with:
        - effect_type: The effect type queried
        - pattern_count: Number of patterns found
        - patterns: List of pattern summaries with id, name, issue, confidence
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if not effect_type and context and hasattr(context, 'session'):
        session = context.session
        if session.request:
            effect_type = session.request.effect_type.value
            print(f"[list_patterns_by_effect] Auto-populated effect_type={effect_type} from context", file=sys.stderr)

    memory = get_pattern_memory()

    try:
        patterns = memory.list_patterns(
            effect_type=effect_type,
            min_confidence=min_confidence
        )

        return json.dumps({
            "effect_type": effect_type,
            "pattern_count": len(patterns),
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "issue_category": p.issue_category,
                    "confidence": p.confidence,
                    "average_improvement": p.average_improvement,
                    "usage_count": p.usage_count
                }
                for p in patterns
            ]
        })

    except Exception as e:
        return json.dumps({
            "effect_type": effect_type,
            "pattern_count": 0,
            "patterns": [],
            "error": str(e)
        })


# =============================================================================
# DIRECT CALLABLE VERSIONS (for non-agent code)
# =============================================================================

def record_pattern_impl(
    issue: str,
    code_snippet: str,
    effect_type: str,
    improvement: float,
    experiment_id: str
) -> str:
    """Direct callable version for recording patterns from non-agent code."""
    memory = get_pattern_memory()

    pattern_id = memory.record_successful_pattern(
        issue=issue,
        code_snippet=code_snippet,
        effect_type=effect_type,
        improvement=improvement,
        experiment_id=experiment_id
    )

    return json.dumps({
        "success": True,
        "pattern_id": pattern_id
    })


def search_patterns_impl(
    issue: str,
    effect_type: str = "",
    min_confidence: float = 30.0
) -> List[CodePattern]:
    """Direct callable version for searching patterns from non-agent code."""
    memory = get_pattern_memory()

    return memory.retrieve_patterns_for_issue(
        issue=issue,
        effect_type=effect_type if effect_type else None,
        min_confidence=min_confidence
    )
