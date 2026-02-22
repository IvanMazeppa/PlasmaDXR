"""Conditional tool visibility callbacks for is_enabled SDK feature.

Phase 2A-1: Uses the Agents SDK `is_enabled` parameter on FunctionTool to
dynamically hide expensive tools from the LLM when they can't/shouldn't be used.

When is_enabled returns False, the tool is completely invisible to the LLM --
it never appears in the tool list, so the model cannot attempt to call it.
This saves tokens (no tool schema in prompt) and prevents wasted API calls.

Kill switch: Set ENABLE_CONDITIONAL_TOOLS=false to disable all gating
(all callbacks return True, restoring pre-Phase-2A behavior).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import Agent
    from agents.run_context import RunContextWrapper
    from models.shared_context import SharedContext

from utils.budget_tracker import get_budget_tracker

# Master kill switch -- set to "false" to disable all conditional tool gating.
# Useful for debugging or rollback if gating causes unexpected behavior.
ENABLE_CONDITIONAL_TOOLS = os.getenv(
    "ENABLE_CONDITIONAL_TOOLS", "true"
).lower() in ("1", "true", "yes")


def budget_allows_vision(ctx: "RunContextWrapper[SharedContext]", agent: "Agent") -> bool:
    """Hide vision tools when vision budget is exhausted.

    Affects: analyze_with_vision, compare_to_reference, evaluate_render,
             compare_renders, diagnose_issues, analyze_temporal_quality
    """
    if not ENABLE_CONDITIONAL_TOOLS:
        return True
    return get_budget_tracker().can_afford_vision()


def budget_allows_docs(ctx: "RunContextWrapper[SharedContext]", agent: "Agent") -> bool:
    """Hide doc search tools when docs budget is exhausted.

    Affects: semantic_search_blender_docs, search_blender_api_by_intent,
             find_alternative_approaches
    """
    if not ENABLE_CONDITIONAL_TOOLS:
        return True
    return get_budget_tracker().can_afford_docs()


def learning_tool_for_iteration(ctx: "RunContextWrapper[SharedContext]", agent: "Agent") -> bool:
    """Only show learning/mining tools after iteration 1.

    Rationale: On iteration 1, there is no prior data to learn from.
    Hiding these tools reduces prompt size and prevents the LLM from
    wasting turns on empty queries.
    """
    if not ENABLE_CONDITIONAL_TOOLS:
        return True
    try:
        return ctx.context.session.current_iteration > 1
    except (AttributeError, TypeError):
        return True  # Default to visible if context unavailable


def budget_allows_evaluation(ctx: "RunContextWrapper[SharedContext]", agent: "Agent") -> bool:
    """Hide evaluation tools when budget can't afford a full evaluation.

    This is a convenience alias that checks can_afford_evaluation(),
    which internally checks the vision budget (evaluations use vision).
    """
    if not ENABLE_CONDITIONAL_TOOLS:
        return True
    return get_budget_tracker().can_afford_evaluation()
