"""Tests for conditional tool visibility callbacks (Phase 2A-1)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make packages importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.shared_context import (
    SharedContext,
    SessionState,
    AssetRequest,
    EffectType,
)


class TestBudgetAllowsVision:
    """Test budget_allows_vision callback."""

    def test_returns_false_when_vision_budget_exhausted(self):
        from utils.tool_visibility import budget_allows_vision

        mock_tracker = MagicMock()
        mock_tracker.can_afford_vision.return_value = False

        with patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker):
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_vision(ctx, agent) is False

    def test_returns_true_when_vision_budget_available(self):
        from utils.tool_visibility import budget_allows_vision

        mock_tracker = MagicMock()
        mock_tracker.can_afford_vision.return_value = True

        with patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker):
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_vision(ctx, agent) is True


class TestBudgetAllowsDocs:
    """Test budget_allows_docs callback."""

    def test_returns_false_when_docs_budget_exhausted(self):
        from utils.tool_visibility import budget_allows_docs

        mock_tracker = MagicMock()
        mock_tracker.can_afford_docs.return_value = False

        with patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker):
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_docs(ctx, agent) is False

    def test_returns_true_when_docs_budget_available(self):
        from utils.tool_visibility import budget_allows_docs

        mock_tracker = MagicMock()
        mock_tracker.can_afford_docs.return_value = True

        with patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker):
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_docs(ctx, agent) is True


class TestBudgetAllowsEvaluation:
    """Test budget_allows_evaluation callback."""

    def test_returns_false_when_evaluation_unaffordable(self):
        from utils.tool_visibility import budget_allows_evaluation

        mock_tracker = MagicMock()
        mock_tracker.can_afford_evaluation.return_value = False

        with patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker):
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_evaluation(ctx, agent) is False


class TestLearningToolForIteration:
    """Test iteration-gated tool visibility."""

    def _make_ctx(self, iteration: int) -> MagicMock:
        """Create a mock RunContextWrapper with session.current_iteration set."""
        ctx = MagicMock()
        ctx.context.session.current_iteration = iteration
        return ctx

    def test_hidden_on_iteration_0(self):
        from utils.tool_visibility import learning_tool_for_iteration

        ctx = self._make_ctx(0)
        agent = MagicMock()
        assert learning_tool_for_iteration(ctx, agent) is False

    def test_hidden_on_iteration_1(self):
        from utils.tool_visibility import learning_tool_for_iteration

        ctx = self._make_ctx(1)
        agent = MagicMock()
        assert learning_tool_for_iteration(ctx, agent) is False

    def test_visible_on_iteration_2(self):
        from utils.tool_visibility import learning_tool_for_iteration

        ctx = self._make_ctx(2)
        agent = MagicMock()
        assert learning_tool_for_iteration(ctx, agent) is True

    def test_visible_on_iteration_5(self):
        from utils.tool_visibility import learning_tool_for_iteration

        ctx = self._make_ctx(5)
        agent = MagicMock()
        assert learning_tool_for_iteration(ctx, agent) is True

    def test_defaults_to_visible_when_context_unavailable(self):
        from utils.tool_visibility import learning_tool_for_iteration

        ctx = MagicMock()
        ctx.context = None  # Simulate missing context
        agent = MagicMock()
        # Should not raise, should return True
        assert learning_tool_for_iteration(ctx, agent) is True


class TestKillSwitch:
    """Test ENABLE_CONDITIONAL_TOOLS kill switch."""

    def test_kill_switch_disables_vision_gating(self):
        """When kill switch is off, budget_allows_vision always returns True."""
        mock_tracker = MagicMock()
        mock_tracker.can_afford_vision.return_value = False

        with (
            patch("utils.tool_visibility.ENABLE_CONDITIONAL_TOOLS", False),
            patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker),
        ):
            from utils.tool_visibility import budget_allows_vision
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_vision(ctx, agent) is True

    def test_kill_switch_disables_docs_gating(self):
        """When kill switch is off, budget_allows_docs always returns True."""
        mock_tracker = MagicMock()
        mock_tracker.can_afford_docs.return_value = False

        with (
            patch("utils.tool_visibility.ENABLE_CONDITIONAL_TOOLS", False),
            patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker),
        ):
            from utils.tool_visibility import budget_allows_docs
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_docs(ctx, agent) is True

    def test_kill_switch_disables_iteration_gating(self):
        """When kill switch is off, learning_tool_for_iteration always returns True."""
        with patch("utils.tool_visibility.ENABLE_CONDITIONAL_TOOLS", False):
            from utils.tool_visibility import learning_tool_for_iteration
            ctx = MagicMock()
            ctx.context.session.current_iteration = 0
            agent = MagicMock()
            assert learning_tool_for_iteration(ctx, agent) is True

    def test_kill_switch_disables_evaluation_gating(self):
        """When kill switch is off, budget_allows_evaluation always returns True."""
        mock_tracker = MagicMock()
        mock_tracker.can_afford_evaluation.return_value = False

        with (
            patch("utils.tool_visibility.ENABLE_CONDITIONAL_TOOLS", False),
            patch("utils.tool_visibility.get_budget_tracker", return_value=mock_tracker),
        ):
            from utils.tool_visibility import budget_allows_evaluation
            ctx = MagicMock()
            agent = MagicMock()
            assert budget_allows_evaluation(ctx, agent) is True
