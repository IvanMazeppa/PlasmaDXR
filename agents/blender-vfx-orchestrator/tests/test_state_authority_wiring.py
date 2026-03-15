"""
Tests for Phase 1 wiring fix: state authority.

Verifies that:
1. session.record_iteration() updates StuckDetectionState
2. plateau_count and same_issue_count are computed correctly
3. choose_repair_intent() receives deterministic values
4. Quality Gate cannot overwrite deterministic escape_level
"""

import pytest
from models.shared_context import (
    SessionState, EscapeLevel, IterationResult, QualityMetrics,
    ScriptModification, BlenderExecution,
    AssetRequest, EffectType,
)
from models.pipeline_models import (
    QualityOutput,
    RepairIntent,
)
from phases.repair_routing import choose_repair_intent


def _make_session() -> SessionState:
    """Create a minimal SessionState for testing."""
    request = AssetRequest(
        asset_name="test",
        description="test effect",
        effect_type=EffectType.FIRE,
    )
    return SessionState(
        session_id="test_session_001",
        asset_name="test",
        effect_type="fire",
        request=request,
    )


def _make_iter_result(
    iteration: int,
    score: float,
    primary_issue: str = "too dark",
    passed: bool = False,
    technique: str = "mantaflow_gas",
) -> IterationResult:
    """Helper to create minimal IterationResult for testing."""
    return IterationResult(
        iteration=max(iteration, 1),  # IterationResult requires >= 1
        script=ScriptModification(
            script_path=f"/tmp/test_iter{iteration}.py",
            technique_name=technique,
        ),
        execution=BlenderExecution(
            success=True,
            run_dir=f"/tmp/run_{iteration}",
            render_path=f"/tmp/run_{iteration}/render.png",
        ),
        quality=QualityMetrics(
            overall_score=score,
            passed=passed,
            primary_issue=primary_issue,
        ),
        passed=passed,
        score=score,
    )


def _make_quality_output(
    score: float,
    passed: bool = False,
    primary_issue: str = "too dark",
) -> QualityOutput:
    """Helper to create QualityOutput for repair routing."""
    return QualityOutput(
        overall_score=score,
        passed=passed,
        primary_issue=primary_issue,
        issues=[primary_issue] if primary_issue else [],
    )


class TestRecordIterationActivatesStuckState:
    """Verify record_iteration() computes plateau/same_issue/escape correctly."""

    def test_same_issue_count_increments(self):
        """3 iterations with same issue → same_issue_count == 3."""
        session = _make_session()

        for i in range(3):
            session.record_iteration(
                _make_iter_result(i, score=40.0, primary_issue="too dark")
            )

        assert session.stuck_state.same_issue_count == 3

    def test_same_issue_resets_on_new_issue(self):
        """Different issue resets same_issue_count."""
        session = _make_session()

        session.record_iteration(
            _make_iter_result(0, score=40.0, primary_issue="too dark")
        )
        session.record_iteration(
            _make_iter_result(1, score=40.0, primary_issue="too dark")
        )
        session.record_iteration(
            _make_iter_result(2, score=35.0, primary_issue="wrong shape")
        )

        assert session.stuck_state.same_issue_count == 1
        assert session.stuck_state.last_primary_issue == "wrong shape"

    def test_plateau_increments_on_small_delta(self):
        """Scores within ±5 count as plateau."""
        session = _make_session()

        session.record_iteration(
            _make_iter_result(0, score=40.0, primary_issue="too dark")
        )
        session.record_iteration(
            _make_iter_result(1, score=41.0, primary_issue="too dark")
        )
        session.record_iteration(
            _make_iter_result(2, score=40.5, primary_issue="too dark")
        )

        assert session.stuck_state.plateau_count >= 2

    def test_escape_level_escalates(self):
        """Same issue 3x → escape_level reaches SWITCH_TECHNIQUE."""
        session = _make_session()

        for i in range(3):
            escape = session.record_iteration(
                _make_iter_result(i, score=40.0, primary_issue="too dark")
            )

        assert session.stuck_state.escape_level.value >= EscapeLevel.SWITCH_TECHNIQUE.value

    def test_best_score_tracked(self):
        """Best score is correctly tracked through record_iteration."""
        session = _make_session()

        session.record_iteration(_make_iter_result(0, score=30.0))
        session.record_iteration(_make_iter_result(1, score=50.0))
        session.record_iteration(_make_iter_result(2, score=45.0))

        assert session.best_score == 50.0
        assert session.best_iteration == 1


class TestRepairRoutingReceivesCorrectInputs:
    """Verify choose_repair_intent uses deterministic stuck-state values."""

    def test_plateau_triggers_modify_code(self):
        """plateau_count >= 2 → modify_code."""
        quality = _make_quality_output(score=40.0, primary_issue="too dark")
        intent = choose_repair_intent(
            quality=quality,
            code_grounded_feedback="",
            plateau_count=2,
            same_issue_count=2,
            iteration=3,
            escape_level=0,
        )
        assert intent.mode == "modify_code"
        assert intent.trigger == "plateau"

    def test_escape_level_2_triggers_switch(self):
        """escape_level >= 2 → switch_technique."""
        quality = _make_quality_output(score=40.0, primary_issue="too dark")
        intent = choose_repair_intent(
            quality=quality,
            code_grounded_feedback="",
            plateau_count=0,
            same_issue_count=0,
            iteration=3,
            escape_level=2,
        )
        assert intent.mode == "switch_technique"

    def test_zero_plateau_does_not_trigger_modify_code(self):
        """plateau_count=0 (the old bug) → should NOT trigger modify_code from plateau."""
        quality = _make_quality_output(score=40.0, primary_issue="too dark")
        intent = choose_repair_intent(
            quality=quality,
            code_grounded_feedback="",
            plateau_count=0,
            same_issue_count=0,
            iteration=2,
            escape_level=0,
        )
        # Should be modify_params (default) not modify_code
        assert intent.mode != "modify_code" or intent.trigger != "plateau"


class TestEndToEndStateToRouting:
    """Integration test: record_iteration → stuck_state → repair_intent."""

    def test_three_plateau_iterations_trigger_modify_code(self):
        """Full flow: 3 iterations with plateau → repair routing gets modify_code."""
        session = _make_session()

        # Simulate 3 plateau iterations
        for i in range(3):
            session.record_iteration(
                _make_iter_result(i, score=40.0 + (i * 0.5), primary_issue="too dark")
            )

        # Now call repair routing with deterministic stuck-state
        quality = _make_quality_output(score=41.0, primary_issue="too dark")
        intent = choose_repair_intent(
            quality=quality,
            code_grounded_feedback="",
            plateau_count=session.stuck_state.plateau_count,
            same_issue_count=session.stuck_state.same_issue_count,
            iteration=3,
            escape_level=int(session.stuck_state.escape_level.value),
        )

        # With plateau_count >= 2, should get modify_code
        assert intent.mode in ("modify_code", "switch_technique"), (
            f"Expected modify_code or switch_technique, got {intent.mode} "
            f"(plateau={session.stuck_state.plateau_count}, "
            f"same_issue={session.stuck_state.same_issue_count}, "
            f"escape={session.stuck_state.escape_level})"
        )
