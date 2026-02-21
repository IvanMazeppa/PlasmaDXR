"""Tests for escape velocity stuck detection and technique switching."""
import sys
from pathlib import Path

# Make models package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.shared_context import (
    StuckDetectionState,
    EscapeLevel,
    SessionState,
    AssetRequest,
    EffectType,
)


class TestStuckDetectionEscalation:
    """Test that stuck detection escalates correctly."""

    def test_normal_starts_at_level_0(self):
        state = StuckDetectionState()
        assert state.escape_level == EscapeLevel.NORMAL

    def test_same_issue_2x_escalates_to_level_2(self):
        state = StuckDetectionState()
        state.update_from_iteration(30.0, "too_dark", "mantaflow_smoke")
        state.update_from_iteration(32.0, "too_dark", "mantaflow_smoke")
        assert state.escape_level == EscapeLevel.SWITCH_TECHNIQUE

    def test_same_issue_3x_escalates_to_level_3(self):
        state = StuckDetectionState()
        state.update_from_iteration(30.0, "too_dark", "mantaflow_smoke")
        state.update_from_iteration(32.0, "too_dark", "mantaflow_smoke")
        state.update_from_iteration(31.0, "too_dark", "mantaflow_smoke")
        assert state.escape_level == EscapeLevel.MINE_DOCS

    def test_same_issue_4x_escalates_to_level_4(self):
        state = StuckDetectionState()
        for i in range(4):
            state.update_from_iteration(30.0 + i, "too_dark", "mantaflow_smoke")
        assert state.escape_level == EscapeLevel.REQUEST_GUIDANCE

    def test_plateau_2x_escalates_to_level_1(self):
        state = StuckDetectionState()
        state.update_from_iteration(50.0, "low_density", "mantaflow_smoke")
        state.update_from_iteration(51.0, "different_issue", "mantaflow_smoke")  # <3 point change
        state.update_from_iteration(51.5, "another_issue", "mantaflow_smoke")  # <3 point change
        assert state.escape_level == EscapeLevel.KNOWLEDGE_CHECK

    def test_different_issues_dont_escalate(self):
        state = StuckDetectionState()
        state.update_from_iteration(30.0, "too_dark", "mantaflow_smoke")
        state.update_from_iteration(40.0, "no_smoke", "mantaflow_smoke")
        state.update_from_iteration(50.0, "clipping", "mantaflow_smoke")
        assert state.escape_level == EscapeLevel.NORMAL


class TestGetUntriedTechniques:
    """Test that untried techniques are correctly identified."""

    def test_excludes_tried_techniques(self):
        state = StuckDetectionState()
        state.techniques_tried = ["mantaflow_smoke", "shader_volume"]
        available = ["mantaflow_smoke", "shader_volume", "mantaflow_fire", "particle_emitter"]
        untried = state.get_untried_techniques(available)
        assert "mantaflow_smoke" not in untried
        assert "shader_volume" not in untried
        assert "mantaflow_fire" in untried

    def test_returns_non_failed_when_all_tried(self):
        state = StuckDetectionState()
        state.techniques_tried = ["a", "b", "c"]
        state.techniques_failed = ["a"]
        untried = state.get_untried_techniques(["a", "b", "c"])
        assert "a" not in untried
        assert "b" in untried
        assert "c" in untried

    def test_empty_available_returns_empty(self):
        state = StuckDetectionState()
        assert state.get_untried_techniques([]) == []


class TestResetForNewTechnique:
    """Test that reset clears counters correctly."""

    def test_resets_counters(self):
        state = StuckDetectionState()
        state.same_issue_count = 3
        state.plateau_count = 2
        state.progress_count = 1
        state.reset_for_new_technique()
        assert state.same_issue_count == 0
        assert state.plateau_count == 0
        assert state.progress_count == 0

    def test_preserves_escape_level(self):
        """Escape level is not reset -- it steps down naturally with progress."""
        state = StuckDetectionState()
        state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        state.reset_for_new_technique()
        assert state.escape_level == EscapeLevel.SWITCH_TECHNIQUE

    def test_preserves_techniques_tried(self):
        state = StuckDetectionState()
        state.techniques_tried = ["a", "b"]
        state.reset_for_new_technique()
        assert state.techniques_tried == ["a", "b"]


class TestStepDown:
    """Test that escape level steps down with progress."""

    def test_steps_down_after_consecutive_progress(self):
        state = StuckDetectionState()
        # Escalate to level 2
        state.update_from_iteration(30.0, "too_dark")
        state.update_from_iteration(31.0, "too_dark")
        assert state.escape_level == EscapeLevel.SWITCH_TECHNIQUE

        # Two consecutive progress iterations (score +5)
        state.update_from_iteration(40.0, "better_but_dim")
        state.update_from_iteration(50.0, "almost_there")
        # Should step down since significant progress was made
        assert state.escape_level.value < EscapeLevel.SWITCH_TECHNIQUE.value


class TestEscapeActions:
    """Test escape action recommendations."""

    def test_normal_action(self):
        state = StuckDetectionState()
        assert state.get_escape_action() == "proceed_with_modification"

    def test_level_2_action(self):
        state = StuckDetectionState()
        state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
        assert state.get_escape_action() == "generate_new_script_different_technique"

    def test_level_4_action(self):
        state = StuckDetectionState()
        state.escape_level = EscapeLevel.REQUEST_GUIDANCE
        assert state.get_escape_action() == "report_stuck_request_guidance"
