"""Tests for Phase 2B-2: call_model_input_filter context boundaries.

The filter works on *turns* (user/system message + all responses that follow),
not raw items.  This guarantees reasoning/function_call/output groups are
never split.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.context_filter import (
    vfx_context_filter,
    _KEEP_LAST_TURNS,
    _DEFAULT_KEEP_LAST_TURNS,
    _is_turn_start,
    _split_into_turns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeModelInputData:
    input: List[Any]
    instructions: Optional[str] = None


@dataclass
class FakeCallModelData:
    model_data: FakeModelInputData
    agent: Any = None
    context: Any = None


def _make_agent(name: str):
    agent = MagicMock()
    agent.name = name
    return agent


def _call_filter(agent_name: str, items: list, instructions: str = "sys") -> Any:
    data = FakeCallModelData(
        model_data=FakeModelInputData(input=items, instructions=instructions),
        agent=_make_agent(agent_name),
    )
    return vfx_context_filter(data)


def _make_simple_turns(n_turns: int) -> list:
    """Create n turns, each with a user message + assistant response."""
    items = []
    for i in range(n_turns):
        items.append({"role": "user", "content": f"prompt_{i}"})
        items.append({"role": "assistant", "content": f"response_{i}"})
    return items


def _make_reasoning_turn() -> list:
    """Create a turn with reasoning + function_call + output (GPT-5.4 pattern)."""
    return [
        {"role": "user", "content": "do something"},
        {"type": "reasoning", "id": "rs_1"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "status": "completed"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]


def _make_multi_tool_reasoning_turn() -> list:
    """Turn with reasoning + 2 parallel function calls + outputs."""
    return [
        {"role": "user", "content": "research this"},
        {"type": "reasoning", "id": "rs_2"},
        {"type": "function_call", "id": "fc_2a", "call_id": "call_2a", "status": "completed"},
        {"type": "function_call", "id": "fc_2b", "call_id": "call_2b", "status": "completed"},
        {"type": "function_call_output", "call_id": "call_2a", "output": "result_a"},
        {"type": "function_call_output", "call_id": "call_2b", "output": "result_b"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTurnDetection(unittest.TestCase):
    """Verify turn boundary detection."""

    def test_user_is_turn_start(self):
        self.assertTrue(_is_turn_start({"role": "user"}))

    def test_system_is_turn_start(self):
        self.assertTrue(_is_turn_start({"role": "system"}))

    def test_assistant_is_not_turn_start(self):
        self.assertFalse(_is_turn_start({"role": "assistant"}))

    def test_function_call_is_not_turn_start(self):
        self.assertFalse(_is_turn_start({"type": "function_call"}))

    def test_reasoning_is_not_turn_start(self):
        self.assertFalse(_is_turn_start({"type": "reasoning"}))

    def test_function_call_output_is_not_turn_start(self):
        self.assertFalse(_is_turn_start({"type": "function_call_output"}))


class TestSplitIntoTurns(unittest.TestCase):
    """Verify turn splitting logic."""

    def test_simple_two_turns(self):
        items = _make_simple_turns(2)
        turns = _split_into_turns(items)
        self.assertEqual(len(turns), 2)
        self.assertEqual(len(turns[0]), 2)  # user + assistant
        self.assertEqual(len(turns[1]), 2)

    def test_reasoning_turn_stays_together(self):
        items = _make_reasoning_turn()
        turns = _split_into_turns(items)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0]), 4)  # user + reasoning + fc + fco

    def test_multi_tool_turn_stays_together(self):
        items = _make_multi_tool_reasoning_turn()
        turns = _split_into_turns(items)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0]), 6)

    def test_mixed_turns(self):
        items = (
            [{"role": "system", "content": "instructions"}]
            + _make_reasoning_turn()
            + [{"role": "user", "content": "followup"},
               {"role": "assistant", "content": "answer"}]
        )
        turns = _split_into_turns(items)
        self.assertEqual(len(turns), 3)  # system, reasoning turn, followup

    def test_empty_items(self):
        self.assertEqual(_split_into_turns([]), [])

    def test_preamble_without_user(self):
        """Items before any user/system message form their own turn."""
        items = [
            {"type": "reasoning", "id": "rs_x"},
            {"type": "function_call", "id": "fc_x"},
        ]
        turns = _split_into_turns(items)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0]), 2)


class TestBasicFiltering(unittest.TestCase):
    """Verify basic pass-through and trimming."""

    def test_short_history_passthrough(self):
        """Few turns should pass through unchanged."""
        items = _make_simple_turns(2)  # 2 turns, 4 items
        result = _call_filter("Research Agent", items)
        self.assertEqual(len(result.input), 4)  # keep_turns=1, but only 2 turns total ≤ 1+1

    def test_instructions_preserved(self):
        items = _make_simple_turns(5)
        result = _call_filter("Research Agent", items, instructions="custom")
        self.assertEqual(result.instructions, "custom")

    def test_empty_items(self):
        result = _call_filter("Research Agent", [])
        self.assertEqual(len(result.input), 0)

    def test_single_item(self):
        result = _call_filter("Script Writer", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(result.input), 1)


class TestResearchAgentFilter(unittest.TestCase):
    """Research Agent: keep_turns=1 (system turn + current turn only)."""

    def test_trims_to_system_plus_last(self):
        # 5 turns: system + 4 user turns
        items = [{"role": "system", "content": "sys"}]
        for i in range(4):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"role": "assistant", "content": f"a{i}"})
        result = _call_filter("Research Agent", items)
        # head turn (system) + last 1 turn = 3 items
        self.assertEqual(len(result.input), 3)
        self.assertEqual(result.input[0]["content"], "sys")
        self.assertEqual(result.input[1]["content"], "q3")

    def test_preserves_first_item(self):
        items = [{"role": "system", "content": "instructions"}]
        for i in range(5):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"role": "assistant", "content": f"a{i}"})
        result = _call_filter("Research Agent", items)
        self.assertEqual(result.input[0]["content"], "instructions")


class TestQualityAnalystFilter(unittest.TestCase):
    """Quality Analyst: keep_turns=1 — no previous QA results leak in."""

    def test_trims_old_evaluations(self):
        items = [{"role": "system", "content": "sys"}]
        for i in range(4):
            items.extend(_make_reasoning_turn())
        result = _call_filter("Quality Analyst", items)
        # system turn + last reasoning turn only
        self.assertEqual(result.input[0]["content"], "sys")


class TestScriptWriterFilter(unittest.TestCase):
    """Script Writer: keep_turns=3 — needs tool-call context."""

    def test_keeps_3_turns_plus_head(self):
        items = [{"role": "system", "content": "sys"}]
        for i in range(6):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"role": "assistant", "content": f"a{i}"})
        result = _call_filter("Script Writer", items)
        # head(1 item) + last 3 turns(6 items) = 7 items
        self.assertEqual(len(result.input), 7)

    def test_short_history_passthrough(self):
        items = _make_simple_turns(3)
        result = _call_filter("Script Writer", items)
        self.assertEqual(len(result.input), 6)  # 3 turns ≤ 3+1, no trim


class TestCoordinatorFilter(unittest.TestCase):
    """Coordinators: keep_turns=2."""

    def test_technique_coordinator(self):
        items = [{"role": "system", "content": "sys"}]
        for i in range(5):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"role": "assistant", "content": f"a{i}"})
        result = _call_filter("Technique Selection Coordinator", items)
        # head(1) + last 2 turns(4) = 5
        self.assertEqual(len(result.input), 5)


class TestUnknownAgentFilter(unittest.TestCase):
    """Unknown agents get generous default."""

    def test_unknown_agent_uses_default(self):
        items = [{"role": "system", "content": "sys"}]
        for i in range(20):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"role": "assistant", "content": f"a{i}"})
        result = _call_filter("Some Unknown Agent", items)
        # head(1) + last 5 turns(10) = 11
        expected = 1 + _DEFAULT_KEEP_LAST_TURNS * 2
        self.assertEqual(len(result.input), expected)

    def test_no_name_agent(self):
        items = _make_simple_turns(20)
        agent = MagicMock(spec=[])
        data = FakeCallModelData(
            model_data=FakeModelInputData(input=items, instructions=None),
            agent=agent,
        )
        result = vfx_context_filter(data)
        # Uses default keep_turns
        self.assertGreater(len(result.input), 0)
        self.assertLess(len(result.input), len(items))


class TestReasoningModelSafety(unittest.TestCase):
    """The critical test: reasoning/function_call groups must never be split."""

    def test_reasoning_group_never_split(self):
        """A reasoning turn trimmed as a whole — no orphaned items."""
        items = [{"role": "system", "content": "sys"}]
        # 4 reasoning turns
        for _ in range(4):
            items.extend(_make_reasoning_turn())
        result = _call_filter("Research Agent", items)
        # Check no orphaned function_call_output
        result_types = [i.get("type") or i.get("role") for i in result.input]
        for i, t in enumerate(result_types):
            if t == "function_call_output":
                # Its function_call must be present
                call_id = result.input[i].get("call_id")
                fc_ids = [
                    r.get("call_id") for r in result.input
                    if (r.get("type") or r.get("role")) == "function_call"
                ]
                self.assertIn(call_id, fc_ids,
                              f"Orphaned function_call_output: {call_id}")

    def test_multi_tool_parallel_calls_preserved(self):
        """Parallel tool calls in one turn stay together."""
        items = [{"role": "system", "content": "sys"}]
        for _ in range(3):
            items.extend(_make_multi_tool_reasoning_turn())
        result = _call_filter("Research Agent", items)
        result_types = [i.get("type") or i.get("role") for i in result.input]
        # Count: each kept turn should have matching fc/fco pairs
        fc_count = result_types.count("function_call")
        fco_count = result_types.count("function_call_output")
        self.assertEqual(fc_count, fco_count)

    def test_reasoning_always_has_function_call(self):
        """Every reasoning item in output must be followed by function_call."""
        items = [{"role": "system", "content": "sys"}]
        for _ in range(5):
            items.extend(_make_reasoning_turn())
        result = _call_filter("Quality Analyst", items)
        result_types = [i.get("type") or i.get("role") for i in result.input]
        for i, t in enumerate(result_types):
            if t == "reasoning":
                self.assertGreater(len(result_types), i + 1,
                                   "reasoning is last item — missing function_call")
                self.assertEqual(result_types[i + 1], "function_call",
                                 "reasoning not followed by function_call")

    def test_docs_expert_with_reasoning_model(self):
        """Reproduce original bug: DocsExpert trimming splits reasoning group."""
        # DocsExpert keep_turns=1
        # System turn, then a turn with reasoning+fc+fco, then another turn
        items = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "search docs"},
            {"type": "reasoning", "id": "rs_1"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "status": "completed"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            {"role": "user", "content": "search more"},
            {"type": "reasoning", "id": "rs_2"},
            {"type": "function_call", "id": "fc_2", "call_id": "call_2", "status": "completed"},
            {"type": "function_call_output", "call_id": "call_2", "output": "result2"},
        ]
        result = _call_filter("Documentation Expert (Standalone)", items)
        result_types = [i.get("type") or i.get("role") for i in result.input]
        # No orphaned items
        for i, t in enumerate(result_types):
            if t == "reasoning":
                self.assertEqual(result_types[i + 1], "function_call")
            if t == "function_call_output":
                call_id = result.input[i].get("call_id")
                fc_ids = [
                    r.get("call_id") for r in result.input
                    if (r.get("type") or r.get("role")) == "function_call"
                ]
                self.assertIn(call_id, fc_ids)


class TestAllKnownAgentsCovered(unittest.TestCase):
    """Sanity check: all agents produce valid output."""

    def test_all_agents_produce_valid_output(self):
        for agent_name in _KEEP_LAST_TURNS:
            with self.subTest(agent=agent_name):
                items = [{"role": "system", "content": "sys"}]
                for i in range(10):
                    items.extend(_make_reasoning_turn())
                result = _call_filter(agent_name, items)
                self.assertGreater(len(result.input), 0)
                # Verify no orphaned function_call_outputs
                result_types = [i.get("type") or i.get("role") for i in result.input]
                for j, t in enumerate(result_types):
                    if t == "function_call_output":
                        call_id = result.input[j].get("call_id")
                        fc_ids = [
                            r.get("call_id") for r in result.input
                            if (r.get("type") or r.get("role")) == "function_call"
                        ]
                        self.assertIn(call_id, fc_ids,
                                      f"{agent_name}: orphaned fco {call_id}")


if __name__ == "__main__":
    unittest.main()
