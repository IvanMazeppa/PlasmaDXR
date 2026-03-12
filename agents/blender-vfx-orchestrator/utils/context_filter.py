"""Phase 2B-2: Per-agent call_model_input_filter for context boundaries.

Each agent sees only the context it needs — system prompt + current-phase
inputs.  This prevents context bloat (F7), reduces cost, and stops agents
from being confused by irrelevant history from earlier pipeline phases.

The filter dispatches by ``data.agent.name`` so a single RunConfig instance
can be shared across all agents in the pipeline.

SDK signature (v0.10.5):
    ``Callable[[CallModelData[Any]], ModelInputData | Awaitable[ModelInputData]]``

IMPORTANT: With reasoning models (GPT-5.4, o3, etc.), conversation items
form required groups:
  - ``reasoning`` must be followed by its ``function_call``
  - ``function_call`` must be followed by its ``function_call_output``
  - ``function_call_output`` requires its ``function_call`` to be present

Naively slicing by position splits these groups → 400 errors.
This filter identifies "turn boundaries" (user messages) and only cuts
at those points, guaranteeing complete groups.
"""

from __future__ import annotations

import sys
from typing import Any

from agents.run_config import CallModelData, ModelInputData


# ---------------------------------------------------------------------------
# Agent-name → keep-last-N-turns mapping
# ---------------------------------------------------------------------------
# A "turn" is a user message plus all model responses/tool calls that follow
# it until the next user message.  This is safer than counting raw items
# because it can never split a reasoning/function_call/output group.

_KEEP_LAST_TURNS: dict[str, int] = {
    # Bounded single-shot agents — keep only the current turn
    "Research Agent": 1,
    "Quality Analyst": 1,
    "Documentation Expert (Standalone)": 1,
    # Decision-making coordinators — current turn + 1 previous
    "Technique Selection Coordinator": 2,
    "Modification Strategy Coordinator": 2,
    "Quality Gate Coordinator": 2,
    # Iterative agents — need a few turns of tool-call context
    "Script Writer": 3,
    "Learning Agent": 3,
}

# Fallback for unknown agents (generous, avoids breaking anything)
_DEFAULT_KEEP_LAST_TURNS = 5


def _get_item_role(item: Any) -> str | None:
    """Extract the role/type of a conversation item."""
    if isinstance(item, dict):
        return item.get("role") or item.get("type")
    return getattr(item, "role", None) or getattr(item, "type", None)


def _is_turn_start(item: Any) -> bool:
    """Return True if this item starts a new conversational turn.

    User and system messages are turn boundaries.  Everything else
    (reasoning, function_call, function_call_output, assistant) belongs
    to the preceding turn.
    """
    role = _get_item_role(item)
    return role in ("user", "system")


def _split_into_turns(items: list) -> list[list]:
    """Split items into turns, where each turn starts with a user/system message.

    Items before the first user/system message form a "preamble" turn.
    This keeps reasoning/function_call/output groups intact because they
    are never split across turns.
    """
    turns: list[list] = []
    current: list = []

    for item in items:
        if _is_turn_start(item) and current:
            turns.append(current)
            current = []
        current.append(item)

    if current:
        turns.append(current)

    return turns


def vfx_context_filter(data: CallModelData[Any]) -> ModelInputData:
    """Trim conversation history per-agent before each LLM call.

    Strategy:
    - Split items into turns (user/system message + all responses that follow).
    - Always preserve the first turn (system prompt / instructions).
    - Keep the last N turns (agent-specific).
    - This guarantees reasoning/function_call/output groups stay intact.

    This is deterministic and $0 — it runs as plain Python before the
    API call.
    """
    items = data.model_data.input
    agent_name = getattr(data.agent, "name", None) or ""
    keep_turns = _KEEP_LAST_TURNS.get(agent_name, _DEFAULT_KEEP_LAST_TURNS)

    # Split into turns
    turns = _split_into_turns(items)

    # Nothing to trim if we have few enough turns
    if len(turns) <= keep_turns + 1:  # +1 for the head turn we always keep
        return ModelInputData(
            input=items,
            instructions=data.model_data.instructions,
        )

    # Always keep first turn (system prompt / instructions injection)
    head_turn = turns[0]
    # Keep the last N turns
    tail_turns = turns[-keep_turns:] if keep_turns > 0 else []

    # Deduplicate: if head turn is also in tail, drop head
    if tail_turns and turns[0] is tail_turns[0]:
        final_items = []
        for turn in tail_turns:
            final_items.extend(turn)
    else:
        final_items = list(head_turn)
        for turn in tail_turns:
            final_items.extend(turn)

    trimmed_count = len(items) - len(final_items)
    if trimmed_count > 0:
        print(
            f"[ContextFilter] {agent_name}: trimmed {trimmed_count} items "
            f"({len(items)} → {len(final_items)}, "
            f"{len(turns)} turns → {min(len(turns), keep_turns + 1)})",
            file=sys.stderr,
        )

    return ModelInputData(
        input=final_items,
        instructions=data.model_data.instructions,
    )
