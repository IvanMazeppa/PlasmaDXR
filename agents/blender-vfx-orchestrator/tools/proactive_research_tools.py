"""
Proactive Research Tools for Blender VFX Orchestrator.

Implements Strategy 3: Proactive Documentation Mining.

These tools enable the orchestrator to detect early warning signs of
being stuck and proactively search for alternative approaches BEFORE
wasting iterations on failing strategies.

Key capabilities:
- Early warning detection (same issue 2x, score plateau)
- Proactive documentation search when warnings triggered
- Alternative approach discovery
- Escape velocity evaluation
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from agents import function_tool, RunContextWrapper

# Import SharedContext directly (not under TYPE_CHECKING) because
# @function_tool decorator evaluates type hints at runtime
from models.shared_context import SharedContext

# Use in-process function implementations instead of MCP subprocess
# This avoids the anyio TaskGroup conflicts with nested MCP calls
# Import the _impl versions which are directly callable (not FunctionTool wrapped)
from shared import search_semantic_impl


# =============================================================================
# EARLY WARNING DETECTION
# =============================================================================

def analyze_iteration_history(
    iterations: List[Dict[str, Any]],
    current_issue: str
) -> Dict[str, Any]:
    """
    Analyze iteration history for early warning signs.

    Args:
        iterations: List of iteration results with score, issue, technique
        current_issue: The current primary issue

    Returns:
        Dict with warning_level and analysis details
    """
    if len(iterations) < 2:
        return {
            "warning_level": "none",
            "same_issue_count": 0,
            "plateau_count": 0,
            "analysis": "Insufficient history for warning detection"
        }

    # Check for same issue persisting
    recent_issues = [
        it.get("primary_issue", it.get("issue", ""))
        for it in iterations[-3:]
    ]
    same_issue_count = sum(1 for issue in recent_issues if issue == current_issue)

    # Check for score plateau (< 3 point change between iterations)
    recent_scores = [it.get("score", 0) for it in iterations[-3:]]
    plateau_count = 0
    for i in range(1, len(recent_scores)):
        if abs(recent_scores[i] - recent_scores[i-1]) < 3.0:
            plateau_count += 1

    # Determine warning level
    if same_issue_count >= 3 or plateau_count >= 3:
        warning_level = "stuck"
    elif same_issue_count >= 2 or plateau_count >= 2:
        warning_level = "early"
    else:
        warning_level = "none"

    # Get techniques already tried
    techniques_tried = list(set(
        it.get("technique", it.get("technique_name", ""))
        for it in iterations
        if it.get("technique") or it.get("technique_name")
    ))

    return {
        "warning_level": warning_level,
        "same_issue_count": same_issue_count,
        "plateau_count": plateau_count,
        "recent_scores": recent_scores,
        "techniques_tried": techniques_tried,
        "analysis": _generate_analysis_text(
            warning_level, same_issue_count, plateau_count, current_issue
        )
    }


def _generate_analysis_text(
    warning_level: str,
    same_issue_count: int,
    plateau_count: int,
    current_issue: str
) -> str:
    """Generate human-readable analysis text."""
    if warning_level == "none":
        return "Normal progress - no warning signs detected"

    parts = []
    if same_issue_count >= 2:
        parts.append(f"Same issue '{current_issue}' persisting for {same_issue_count} iterations")
    if plateau_count >= 2:
        parts.append(f"Score plateau detected ({plateau_count} iterations with <3 point change)")

    if warning_level == "stuck":
        return f"STUCK: {'; '.join(parts)}. Recommend switching technique or mining documentation."
    else:
        return f"EARLY WARNING: {'; '.join(parts)}. Recommend checking knowledge base for alternatives."


# =============================================================================
# PROACTIVE RESEARCH TOOL
# =============================================================================

@function_tool
async def pre_iteration_research(
    wrapper: RunContextWrapper[SharedContext],
    current_issue: str,
    current_approach: str,
    iteration_history: str = "",
    effect_type: str = ""
) -> str:
    """
    Proactively research alternatives BEFORE attempting another iteration.

    Call this at the START of each iteration to detect early warning signs.
    If warnings are detected, this tool searches for alternative approaches
    so you can try something different instead of repeating failures.

    WHEN TO USE:
    - Call at the beginning of every iteration after the first
    - If warning_level is "early" or "stuck", follow the recommendations
    - If warning_level is "none", proceed with normal modification

    NOTE: With RunContextWrapper, iteration_history and effect_type can be
    auto-populated from context.session if not provided explicitly.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        current_issue: The primary issue being addressed (e.g., "smoke too thin")
        current_approach: Description of current approach (e.g., "increasing flame_smoke parameter")
        iteration_history: JSON array of iteration results (optional - auto-populated from context)
        effect_type: Type of effect (optional - auto-populated from context)

    Returns:
        JSON with:
        - warning_level: "none", "early", or "stuck"
        - should_research: bool - whether to search for alternatives
        - same_issue_count: How many iterations with same issue
        - plateau_count: How many iterations with score plateau
        - analysis: Human-readable analysis
        - recommendation: What to do next
        - alternative_queries: Suggested doc searches if should_research
        - escape_action: Specific action to take

    Example:
        pre_iteration_research(
            current_issue="smoke lacks density",
            current_approach="increasing flame_smoke parameter"
        )
        # iteration_history and effect_type auto-populated from context
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        if not iteration_history and session.iterations:
            # Build iteration history from session state
            iterations = [
                {
                    "score": it.score,
                    "issue": it.primary_issue,
                    "technique": it.technique_used
                }
                for it in session.iterations
            ]
        else:
            try:
                iterations = json.loads(iteration_history) if iteration_history else []
            except json.JSONDecodeError:
                iterations = []

        if not effect_type and session.request:
            effect_type = session.request.effect_type.value

        # Log context usage for debugging
        print(f"[pre_iteration_research] Using context: session={session.session_id}, iterations={len(iterations)}", file=sys.stderr)
    else:
        try:
            iterations = json.loads(iteration_history) if iteration_history else []
        except json.JSONDecodeError:
            iterations = []

    # Analyze history for warning signs
    analysis = analyze_iteration_history(iterations, current_issue)

    warning_level = analysis["warning_level"]
    should_research = warning_level in ("early", "stuck")

    # Build response
    result = {
        "warning_level": warning_level,
        "should_research": should_research,
        "same_issue_count": analysis["same_issue_count"],
        "plateau_count": analysis["plateau_count"],
        "techniques_tried": analysis.get("techniques_tried", []),
        "analysis": analysis["analysis"],
    }

    if warning_level == "none":
        result["recommendation"] = "Proceed with planned modification"
        result["escape_action"] = "continue"
        return json.dumps(result)

    # Generate alternative search queries
    alternative_queries = _generate_search_queries(
        current_issue, current_approach, effect_type
    )
    result["alternative_queries"] = alternative_queries

    # Determine escape action and recommendation
    if warning_level == "early":
        result["escape_action"] = "check_knowledge_then_modify"
        result["recommendation"] = (
            f"Early warning: {current_issue} persisting. "
            f"Before modifying, query knowledge base with: '{alternative_queries[0]}'. "
            f"If no good alternatives found, consider switching technique."
        )
    else:  # stuck
        result["escape_action"] = "switch_technique_or_mine_docs"
        result["recommendation"] = (
            f"STUCK on '{current_issue}'. Do NOT continue with '{current_approach}'. "
            f"Options: 1) Generate new script with different technique, "
            f"2) Search Blender docs for alternative approaches using queries: {alternative_queries[:2]}"
        )

    return json.dumps(result)


def _generate_search_queries(
    issue: str,
    approach: str,
    effect_type: str
) -> List[str]:
    """Generate search queries for finding alternatives."""
    queries = []

    # Issue-focused query
    queries.append(f"Blender {effect_type} fix {issue} alternative methods")

    # Approach-negation query
    queries.append(f"Blender {effect_type} {issue} without {approach}")

    # Effect-type specific query
    queries.append(f"Mantaflow {effect_type} simulation {issue} solutions")

    # General alternative query
    queries.append(f"Blender 5.0 {issue} different approach")

    return queries


# =============================================================================
# ESCAPE VELOCITY EVALUATION
# =============================================================================

@function_tool
async def evaluate_escape_velocity(
    wrapper: RunContextWrapper[SharedContext],
    current_issue: str,
    current_score: float = 0.0,
    iteration_history: str = "",
    techniques_available: str = ""
) -> str:
    """
    Evaluate current stuck state and determine escape action.

    Use this when pre_iteration_research indicates warning signs.
    This tool provides specific guidance on what escape action to take.

    ESCAPE LEVELS:
    - Level 0 (NORMAL): Continue with planned modification
    - Level 1 (KNOWLEDGE_CHECK): Query knowledge base for alternatives first
    - Level 2 (SWITCH_TECHNIQUE): Don't modify - generate new script with different technique
    - Level 3 (MINE_DOCS): Search Blender documentation for novel approaches
    - Level 4 (REQUEST_GUIDANCE): Report stuck status, request human input

    NOTE: With RunContextWrapper, most args can be auto-populated from context.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        current_issue: Current primary issue being addressed
        current_score: Current quality score (optional - auto-populated from context)
        iteration_history: JSON array of {score, issue, technique} (optional - auto-populated)
        techniques_available: JSON array of available technique names (optional)

    Returns:
        JSON with:
        - escape_level: 0-4 numeric level
        - escape_level_name: Human-readable level name
        - action: Specific action to take
        - reason: Why this action was chosen
        - technique_suggestion: If switching, which technique to try
        - search_queries: If mining docs, what to search for
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        if not iteration_history and session.iterations:
            history = [
                {
                    "score": it.score,
                    "issue": it.primary_issue,
                    "technique": it.technique_used
                }
                for it in session.iterations
            ]
        else:
            try:
                history = json.loads(iteration_history) if iteration_history else []
            except json.JSONDecodeError:
                history = []

        if current_score == 0.0:
            current_score = session.best_score

        # Log context usage
        print(f"[evaluate_escape_velocity] Using context: escape_level={session.stuck_state.escape_level if session.stuck_state else 'N/A'}", file=sys.stderr)
    else:
        try:
            history = json.loads(iteration_history) if iteration_history else []
        except json.JSONDecodeError:
            history = []

    try:
        techniques = json.loads(techniques_available) if techniques_available else []
    except json.JSONDecodeError:
        techniques = []

    # Analyze history
    analysis = analyze_iteration_history(history, current_issue)

    # Get tried techniques
    tried = set(analysis.get("techniques_tried", []))
    untried = [t for t in techniques if t not in tried]

    # Determine escape level
    same_issue = analysis["same_issue_count"]
    plateau = analysis["plateau_count"]

    if same_issue >= 4 or plateau >= 4:
        escape_level = 4
    elif same_issue >= 3 or plateau >= 3:
        escape_level = 3
    elif same_issue >= 2:
        escape_level = 2
    elif plateau >= 2:
        escape_level = 1
    else:
        escape_level = 0

    level_names = {
        0: "NORMAL",
        1: "KNOWLEDGE_CHECK",
        2: "SWITCH_TECHNIQUE",
        3: "MINE_DOCS",
        4: "REQUEST_GUIDANCE"
    }

    result = {
        "escape_level": escape_level,
        "escape_level_name": level_names[escape_level],
        "current_score": current_score,
        "same_issue_count": same_issue,
        "plateau_count": plateau,
    }

    # Level-specific guidance
    if escape_level == 0:
        result["action"] = "continue"
        result["reason"] = "Normal progress"

    elif escape_level == 1:
        result["action"] = "check_knowledge"
        result["reason"] = f"Score plateau detected ({plateau} iterations with minimal change)"
        result["knowledge_queries"] = [
            f"alternative fixes for {current_issue}",
            f"different approach to {current_issue}"
        ]

    elif escape_level == 2:
        result["action"] = "switch_technique"
        result["reason"] = f"Same issue '{current_issue}' persisting for {same_issue} iterations"
        if untried:
            result["technique_suggestion"] = untried[0]
            result["untried_techniques"] = untried
        else:
            # All techniques tried, escalate to level 3
            result["action"] = "mine_docs"
            result["reason"] += " - all known techniques already tried"
            escape_level = 3
            result["escape_level"] = 3
            result["escape_level_name"] = "MINE_DOCS"

    if escape_level == 3:
        result["action"] = "mine_docs"
        result["reason"] = result.get("reason", f"Current approaches exhausted after {same_issue} iterations")
        result["search_queries"] = [
            f"Blender 5.0 {current_issue} alternative solutions",
            f"Mantaflow {current_issue} workaround",
            f"Blender fluid simulation {current_issue} fix"
        ]

    elif escape_level == 4:
        result["action"] = "request_guidance"
        result["reason"] = "Exhausted autonomous options"
        result["summary"] = {
            "iterations": len(history),
            "techniques_tried": list(tried),
            "best_score": max((h.get("score", 0) for h in history), default=0),
            "persistent_issue": current_issue
        }
        result["message"] = (
            f"I've tried {len(tried)} different techniques over {len(history)} iterations "
            f"but cannot resolve '{current_issue}'. Best score achieved: {result['summary']['best_score']:.1f}. "
            f"Please provide guidance or accept current best result."
        )

    return json.dumps(result)


# =============================================================================
# DOCUMENTATION SEARCH (Bridges to DocsExpert)
# =============================================================================

@function_tool
async def search_alternative_approaches(
    wrapper: RunContextWrapper[SharedContext],
    issue: str,
    current_approach: str,
    effect_type: str = "",
    exclude_techniques: str = "[]"
) -> str:
    """
    Search Blender documentation for alternative approaches to solve an issue.

    Use when escape level >= 2 (SWITCH_TECHNIQUE) to find genuinely
    different approaches rather than parameter variations.

    NOTE: With RunContextWrapper, effect_type and exclude_techniques can be
    auto-populated from context.session.stuck_state.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        issue: The problem to solve (e.g., "smoke lacks density")
        current_approach: What you've been trying (to exclude from results)
        effect_type: Type of effect (optional - auto-populated from context)
        exclude_techniques: JSON array of technique names to exclude (optional)

    Returns:
        JSON with:
        - alternatives_found: Number of alternatives discovered
        - alternatives: List of alternative approaches, each with:
            - approach_name: Short description
            - description: How this approach works
            - relevant_apis: Blender API functions involved
            - trade_offs: Pros/cons vs current approach
            - confidence: Estimated likelihood of helping (0-100)
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        if not effect_type and session.request:
            effect_type = session.request.effect_type.value

        if exclude_techniques == "[]" and session.stuck_state:
            exclude = list(session.stuck_state.techniques_tried)
        else:
            try:
                exclude = json.loads(exclude_techniques) if exclude_techniques else []
            except json.JSONDecodeError:
                exclude = []

        print(f"[search_alternative_approaches] Using context: effect_type={effect_type}, exclude={exclude}", file=sys.stderr)
    else:
        try:
            exclude = json.loads(exclude_techniques) if exclude_techniques else []
        except json.JSONDecodeError:
            exclude = []

    # Build search query that excludes current approach
    exclude_text = " NOT " + " NOT ".join(exclude) if exclude else ""
    queries = [
        f"Blender {effect_type} {issue} alternative solution{exclude_text}",
        f"Mantaflow simulation {issue} different method",
        f"Blender 5.0 fluid {issue} workaround",
    ]

    # Search using in-process function tools (not MCP subprocess)
    # This avoids anyio TaskGroup conflicts with nested MCP calls
    try:
        all_results = []
        for query in queries[:2]:  # Limit to 2 queries for efficiency
            # Call in-process search_semantic_impl function directly (not FunctionTool)
            result_str = search_semantic_impl(query=query, limit=3)
            try:
                data = json.loads(result_str)
                if "results" in data:
                    all_results.extend(data["results"])
            except json.JSONDecodeError:
                pass

        # Process and deduplicate results
        alternatives = _process_search_results(all_results, current_approach, exclude)

        return json.dumps({
            "issue": issue,
            "alternatives_found": len(alternatives),
            "alternatives": alternatives,
            "queries_used": queries[:2]
        })

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "alternatives_found": 0,
            "alternatives": [],
            "fallback_suggestion": (
                f"Manual search recommended: Look in Blender docs for "
                f"'{effect_type} {issue}' alternatives"
            )
        })


def _process_search_results(
    results: List[Dict],
    current_approach: str,
    exclude: List[str]
) -> List[Dict]:
    """Process and filter search results into alternatives."""
    alternatives = []
    seen_titles = set()

    for result in results:
        title = result.get("title", result.get("name", ""))

        # Skip duplicates
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        # Skip if matches current approach or excluded techniques
        if any(exc.lower() in title.lower() for exc in exclude):
            continue
        if current_approach.lower() in title.lower():
            continue

        # Extract useful information
        alternative = {
            "approach_name": title,
            "description": result.get("content", result.get("description", ""))[:500],
            "source": result.get("url", result.get("path", "")),
            "confidence": 50,  # Base confidence, could be enhanced with relevance scoring
        }

        # Try to extract API references
        content = alternative["description"].lower()
        apis = []
        if "bpy.types" in content:
            apis.append("bpy.types")
        if "bpy.ops" in content:
            apis.append("bpy.ops")
        alternative["relevant_apis"] = apis

        alternatives.append(alternative)

        if len(alternatives) >= 5:
            break

    return alternatives
