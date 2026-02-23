"""
Tests for Phase 2B-5: HITL Framework.

Tests checkpoint models, autonomy gating, trigger conditions,
interactive prompt mocking, pending resume, and config integration.
"""

import sys
import os
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hitl_handler import (
    CheckpointType,
    CheckpointDecision,
    HITLCheckpoint,
    HITLHandler,
)
from models.shared_context import (
    SessionState,
    AssetRequest,
    EscapeLevel,
    StuckDetectionState,
    IterationResult,
    ScriptModification,
    BlenderExecution,
    QualityMetrics,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_session(**overrides) -> SessionState:
    """Create a minimal SessionState for testing."""
    defaults = dict(
        session_id="test_session",
        request=AssetRequest(
            asset_name="test_asset",
            description="test",
        ),
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def _make_iteration(score: float, iteration: int = 1) -> IterationResult:
    """Create a minimal IterationResult."""
    return IterationResult(
        iteration=iteration,
        script=ScriptModification(script_path="test.py"),
        execution=BlenderExecution(success=True, run_dir="/tmp"),
        quality=QualityMetrics(overall_score=score, passed=score >= 60),
        passed=score >= 60,
        score=score,
    )


@dataclass
class _FakeQuality:
    """Minimal quality stand-in for check_post_eval."""
    overall_score: float = 50.0
    passed: bool = False
    primary_issue: Optional[str] = None
    issues: List[str] = field(default_factory=list)


# =============================================================================
# CHECKPOINT MODEL TESTS
# =============================================================================

class TestHITLCheckpoint(unittest.TestCase):

    def test_fields(self):
        cp = HITLCheckpoint(
            checkpoint_type=CheckpointType.BUDGET_WARNING,
            reason="test",
            context={"iteration": 3},
        )
        self.assertEqual(cp.checkpoint_type, CheckpointType.BUDGET_WARNING)
        self.assertIsNone(cp.decision)
        self.assertIsNone(cp.human_notes)

    def test_serialization_roundtrip(self):
        cp = HITLCheckpoint(
            checkpoint_type=CheckpointType.ESCALATION,
            reason="L4",
            context={"best_score": 42.0},
            decision=CheckpointDecision.SWITCH_TECHNIQUE,
        )
        d = cp.to_dict()
        self.assertEqual(d["checkpoint_type"], "escalation")
        self.assertEqual(d["decision"], "switch")

        cp2 = HITLCheckpoint.from_dict(d)
        self.assertEqual(cp2.checkpoint_type, CheckpointType.ESCALATION)
        self.assertEqual(cp2.decision, CheckpointDecision.SWITCH_TECHNIQUE)

    def test_type_enum_values(self):
        self.assertEqual(CheckpointType.STALL_DETECTION.value, "stall_detection")
        self.assertEqual(CheckpointType.BUDGET_WARNING.value, "budget_warning")
        self.assertEqual(CheckpointType.CRITICAL_ISSUE.value, "critical_issue")
        self.assertEqual(CheckpointType.ESCALATION.value, "escalation")
        self.assertEqual(CheckpointType.QUALITY_PLATEAU.value, "quality_plateau")


# =============================================================================
# AUTONOMY GATING TESTS
# =============================================================================

class TestAutonomyGating(unittest.TestCase):
    """Verify that autonomy levels correctly gate checkpoint firing."""

    def _budget_over_80(self) -> Dict[str, Any]:
        return {"total_spent": 17.0, "total_remaining": 3.0}

    def _budget_ok(self) -> Dict[str, Any]:
        return {"total_spent": 5.0, "total_remaining": 15.0}

    def _session_with_critical(self) -> SessionState:
        """Session with 2 consecutive zero-score iterations."""
        session = _make_session()
        session.iterations.append(_make_iteration(0, iteration=1))
        session.iterations.append(_make_iteration(0, iteration=2))
        session.current_iteration = 2
        session.stuck_state.same_issue_count = 2
        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        return session

    def _session_with_stall(self) -> SessionState:
        """Session stuck on same issue 3x at escape level >= 2."""
        session = _make_session()
        session.stuck_state.same_issue_count = 3
        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        session.current_iteration = 3
        return session

    def test_level0_fires_all(self):
        """Level 0 (guided): all checkpoint types fire."""
        handler = HITLHandler(autonomy_level=0, interactive=False, enabled=True)
        session = self._session_with_stall()
        quality = _FakeQuality(overall_score=30)
        cp = handler.check_post_eval(session, quality, self._budget_over_80())
        # Should fire (stall or budget — whichever is checked first that matches)
        self.assertIsNotNone(cp)

    def test_level1_fires_budget_and_stall(self):
        """Level 1 (supervised): budget + stall checkpoints fire."""
        handler = HITLHandler(autonomy_level=1, interactive=False, enabled=True)

        # Budget checkpoint
        session = _make_session()
        quality = _FakeQuality()
        cp = handler.check_post_eval(session, quality, self._budget_over_80())
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.BUDGET_WARNING)

    def test_level2_only_critical(self):
        """Level 2 (semi-autonomous): only critical checkpoints fire."""
        handler = HITLHandler(autonomy_level=2, interactive=False, enabled=True)

        # Budget should NOT fire at level 2
        session = _make_session()
        quality = _FakeQuality()
        cp = handler.check_post_eval(session, quality, self._budget_over_80())
        self.assertIsNone(cp)

        # Critical SHOULD fire at level 2
        session = self._session_with_critical()
        quality = _FakeQuality(overall_score=0)
        cp = handler.check_post_eval(session, quality, self._budget_ok())
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.CRITICAL_ISSUE)

    def test_level3_only_l4_escalation(self):
        """Level 3 (autonomous): no post-eval checkpoints, only escalation."""
        handler = HITLHandler(autonomy_level=3, interactive=False, enabled=True)

        # Critical should NOT fire at level 3
        session = self._session_with_critical()
        quality = _FakeQuality(overall_score=0)
        cp = handler.check_post_eval(session, quality, self._budget_ok())
        self.assertIsNone(cp)

        # But escalation SHOULD fire at level 3
        session.stuck_state.escape_level = EscapeLevel.REQUEST_GUIDANCE
        cp = handler.check_escalation(session)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.ESCALATION)

    def test_level4_fires_nothing(self):
        """Level 4 (full autonomous): nothing fires."""
        handler = HITLHandler(autonomy_level=4, interactive=False, enabled=True)

        session = self._session_with_critical()
        quality = _FakeQuality(overall_score=0)
        cp = handler.check_post_eval(session, quality, self._budget_over_80())
        self.assertIsNone(cp)

        # Even escalation should NOT fire at level 4
        cp = handler.check_escalation(session)
        self.assertIsNone(cp)


# =============================================================================
# BUDGET CHECKPOINT TESTS
# =============================================================================

class TestBudgetCheckpoint(unittest.TestCase):

    def test_over_80_triggers(self):
        handler = HITLHandler(autonomy_level=0, interactive=False, enabled=True)
        session = _make_session()
        quality = _FakeQuality()
        budget = {"total_spent": 17.0, "total_remaining": 3.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.BUDGET_WARNING)

    def test_under_80_no_trigger(self):
        handler = HITLHandler(autonomy_level=0, interactive=False, enabled=True)
        session = _make_session()
        quality = _FakeQuality()
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNone(cp)


# =============================================================================
# STALL CHECKPOINT TESTS
# =============================================================================

class TestStallCheckpoint(unittest.TestCase):

    def test_3_same_issue_triggers(self):
        handler = HITLHandler(autonomy_level=1, interactive=False, enabled=True)
        session = _make_session()
        session.stuck_state.same_issue_count = 3
        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        session.current_iteration = 3
        quality = _FakeQuality()
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.STALL_DETECTION)

    def test_1_issue_no_trigger(self):
        handler = HITLHandler(autonomy_level=1, interactive=False, enabled=True)
        session = _make_session()
        session.stuck_state.same_issue_count = 1
        session.stuck_state.escape_level = EscapeLevel.NORMAL
        quality = _FakeQuality()
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNone(cp)


# =============================================================================
# CRITICAL CHECKPOINT TESTS
# =============================================================================

class TestCriticalCheckpoint(unittest.TestCase):

    def test_2_consecutive_zero_triggers(self):
        handler = HITLHandler(autonomy_level=2, interactive=False, enabled=True)
        session = _make_session()
        session.iterations.append(_make_iteration(0, iteration=1))
        session.iterations.append(_make_iteration(0, iteration=2))
        session.current_iteration = 2
        quality = _FakeQuality(overall_score=0)
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.CRITICAL_ISSUE)

    def test_single_zero_no_trigger(self):
        handler = HITLHandler(autonomy_level=2, interactive=False, enabled=True)
        session = _make_session()
        session.iterations.append(_make_iteration(0, iteration=1))
        session.current_iteration = 1
        quality = _FakeQuality(overall_score=0)
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNone(cp)


# =============================================================================
# ESCALATION TESTS
# =============================================================================

class TestEscalation(unittest.TestCase):

    def test_l4_creates_checkpoint(self):
        handler = HITLHandler(autonomy_level=3, interactive=False, enabled=True)
        session = _make_session()
        session.stuck_state.escape_level = EscapeLevel.REQUEST_GUIDANCE
        session.stuck_state.techniques_tried = ["fire_mantaflow", "fire_shader"]
        session.best_score = 35.0
        cp = handler.check_escalation(session)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.checkpoint_type, CheckpointType.ESCALATION)
        self.assertIn("35.0", cp.reason)

    @patch("builtins.input", return_value="s")
    def test_interactive_decision_applied(self, mock_input):
        handler = HITLHandler(autonomy_level=1, interactive=True, enabled=True)
        session = _make_session()
        session.stuck_state.escape_level = EscapeLevel.REQUEST_GUIDANCE
        session.best_score = 30.0
        cp = handler.check_escalation(session)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.decision, CheckpointDecision.SWITCH_TECHNIQUE)


# =============================================================================
# INTERACTIVE PROMPT TESTS
# =============================================================================

class TestInteractivePrompt(unittest.TestCase):

    @patch("builtins.input", return_value="c")
    def test_continue_choice(self, mock_input):
        handler = HITLHandler(autonomy_level=0, interactive=True, enabled=True)
        session = _make_session()
        session.stuck_state.same_issue_count = 3
        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        session.current_iteration = 3
        quality = _FakeQuality()
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.decision, CheckpointDecision.CONTINUE)

    @patch("builtins.input", return_value="q")
    def test_abort_choice(self, mock_input):
        handler = HITLHandler(autonomy_level=0, interactive=True, enabled=True)
        session = _make_session()
        session.stuck_state.same_issue_count = 3
        session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        session.current_iteration = 3
        quality = _FakeQuality()
        budget = {"total_spent": 5.0, "total_remaining": 15.0}
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.decision, CheckpointDecision.ABORT)


# =============================================================================
# PENDING RESUME TESTS
# =============================================================================

class TestPendingResume(unittest.TestCase):

    def test_pending_checkpoint_consumed(self):
        handler = HITLHandler(autonomy_level=1, interactive=False, enabled=True)
        session = _make_session()
        # Simulate a saved non-interactive checkpoint with decision already set (JSON edit)
        session.pending_checkpoint = HITLCheckpoint(
            checkpoint_type=CheckpointType.STALL_DETECTION,
            reason="test",
            context={},
            decision=CheckpointDecision.CONTINUE,
        ).to_dict()
        cp = handler.check_pending(session)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.decision, CheckpointDecision.CONTINUE)
        # History should be recorded
        self.assertEqual(len(session.hitl_history), 1)

    @patch("builtins.input", return_value="q")
    def test_pending_interactive_prompt(self, mock_input):
        handler = HITLHandler(autonomy_level=1, interactive=True, enabled=True)
        session = _make_session()
        # Pending checkpoint without decision
        session.pending_checkpoint = HITLCheckpoint(
            checkpoint_type=CheckpointType.BUDGET_WARNING,
            reason="test",
            context={},
        ).to_dict()
        cp = handler.check_pending(session)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.decision, CheckpointDecision.ABORT)


# =============================================================================
# CONFIG INTEGRATION TESTS
# =============================================================================

class TestConfig(unittest.TestCase):

    def test_preset_flags_loaded(self):
        """Verify PresetConfig loads HITL flags from dict."""
        from config.agent_config import PresetConfig
        pc = PresetConfig.from_dict("test", {
            "hitl_enabled": True,
            "hitl_autonomy_level": 3,
            "hitl_interactive": False,
        })
        self.assertTrue(pc.hitl_enabled)
        self.assertEqual(pc.hitl_autonomy_level, 3)
        self.assertFalse(pc.hitl_interactive)

    def test_preset_defaults(self):
        """Verify defaults when HITL flags not provided."""
        from config.agent_config import PresetConfig
        pc = PresetConfig.from_dict("test", {})
        self.assertTrue(pc.hitl_enabled)
        self.assertEqual(pc.hitl_autonomy_level, 1)
        self.assertTrue(pc.hitl_interactive)

    def test_config_manager_accessors(self):
        """Verify AgentConfigManager accessor methods."""
        from config.agent_config import AgentConfigManager, PresetConfig
        preset = PresetConfig(
            name="test",
            hitl_enabled=True,
            hitl_autonomy_level=2,
            hitl_interactive=False,
        )
        mgr = AgentConfigManager(preset=preset)
        self.assertTrue(mgr.use_hitl())
        self.assertEqual(mgr.get_hitl_autonomy_level(), 2)
        self.assertFalse(mgr.is_hitl_interactive())


# =============================================================================
# SESSION STATE TESTS
# =============================================================================

class TestSessionStateHITL(unittest.TestCase):

    def test_new_fields_default(self):
        """New HITL fields have safe defaults."""
        session = _make_session()
        self.assertIsNone(session.pending_checkpoint)
        self.assertEqual(session.hitl_history, [])
        self.assertEqual(session.autonomy_level, 1)

    def test_serialization_roundtrip(self):
        """HITL fields survive JSON roundtrip."""
        session = _make_session()
        session.pending_checkpoint = {"checkpoint_type": "budget_warning", "reason": "test", "context": {}}
        session.hitl_history = [{"checkpoint_type": "stall_detection", "decision": "continue"}]
        session.autonomy_level = 3

        d = session.model_dump()
        session2 = SessionState(**d)
        self.assertEqual(session2.pending_checkpoint["checkpoint_type"], "budget_warning")
        self.assertEqual(len(session2.hitl_history), 1)
        self.assertEqual(session2.autonomy_level, 3)


# =============================================================================
# DISABLED HANDLER TESTS
# =============================================================================

class TestDisabledHandler(unittest.TestCase):

    def test_disabled_returns_none(self):
        handler = HITLHandler(autonomy_level=0, interactive=True, enabled=False)
        session = _make_session()
        session.stuck_state.same_issue_count = 5
        session.stuck_state.escape_level = EscapeLevel.REQUEST_GUIDANCE
        session.iterations.append(_make_iteration(0, iteration=1))
        session.iterations.append(_make_iteration(0, iteration=2))
        quality = _FakeQuality(overall_score=0)
        budget = {"total_spent": 19.0, "total_remaining": 1.0}
        # Post-eval: disabled handler returns None
        cp = handler.check_post_eval(session, quality, budget)
        self.assertIsNone(cp)
        # Escalation: disabled handler returns None
        cp = handler.check_escalation(session)
        self.assertIsNone(cp)


if __name__ == "__main__":
    unittest.main()
