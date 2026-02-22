"""
Tests for Phase 2A-4: PipelineMonitor (Deterministic Monitoring Layer).

Tests the 7 detection signals:
1. Parameter oscillation
2. Stuck loop (same primary issue)
3. Score plateau
4. Wrong feedback cascade
5. Budget overrun
6. Script quality (line count)
7. Critical render issues
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Path setup (same as conftest.py)
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from tools.pipeline_monitor import (
    PipelineMonitor,
    AlertLevel,
    AlertType,
    MonitorAlert,
    OSCILLATION_WINDOW,
    PLATEAU_DELTA_THRESHOLD,
    PLATEAU_WINDOW,
    CASCADE_DROP_THRESHOLD,
    STUCK_LOOP_THRESHOLD,
    MIN_SCRIPT_LINES,
)


class TestParameterOscillation(unittest.TestCase):
    """Test oscillation detection: same param changes direction 2x in 3 iterations."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_oscillating_values_detected(self):
        """Values [50, 2500, 10] → assert is_oscillating == True."""
        # Feed 3 iterations with oscillating energy
        self.monitor.check_after_evaluation(
            score=30, primary_issue=None, params={"energy": 50}, iteration=1
        )
        self.monitor.check_after_evaluation(
            score=35, primary_issue=None, params={"energy": 2500}, iteration=2
        )
        alerts = self.monitor.check_after_evaluation(
            score=25, primary_issue=None, params={"energy": 10}, iteration=3
        )
        osc_alerts = [a for a in alerts if a.alert_type == AlertType.PARAMETER_OSCILLATION]
        self.assertEqual(len(osc_alerts), 1)
        self.assertEqual(osc_alerts[0].details["param"], "energy")
        self.assertEqual(osc_alerts[0].action, "clamp_to_midpoint")

    def test_monotonic_values_no_oscillation(self):
        """Values [50, 100, 150] → assert is_oscillating == False."""
        self.monitor.check_after_evaluation(
            score=30, primary_issue=None, params={"energy": 50}, iteration=1
        )
        self.monitor.check_after_evaluation(
            score=40, primary_issue=None, params={"energy": 100}, iteration=2
        )
        alerts = self.monitor.check_after_evaluation(
            score=50, primary_issue=None, params={"energy": 150}, iteration=3
        )
        osc_alerts = [a for a in alerts if a.alert_type == AlertType.PARAMETER_OSCILLATION]
        self.assertEqual(len(osc_alerts), 0)

    def test_insufficient_history_no_alert(self):
        """Only 2 iterations → no oscillation detection possible."""
        self.monitor.check_after_evaluation(
            score=30, primary_issue=None, params={"energy": 50}, iteration=1
        )
        alerts = self.monitor.check_after_evaluation(
            score=35, primary_issue=None, params={"energy": 2500}, iteration=2
        )
        osc_alerts = [a for a in alerts if a.alert_type == AlertType.PARAMETER_OSCILLATION]
        self.assertEqual(len(osc_alerts), 0)

    def test_oscillating_params_in_status_report(self):
        """Status report lists oscillating params."""
        for i, val in enumerate([50, 2500, 10], 1):
            self.monitor.check_after_evaluation(
                score=30, primary_issue=None, params={"energy": val}, iteration=i
            )
        report = self.monitor.get_status_report()
        self.assertIn("energy", report["oscillating_params"])


class TestStuckLoop(unittest.TestCase):
    """Test stuck loop detection: 3x same primary_issue → force technique switch."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_stuck_loop_detected(self):
        """Same issue 3x → CRITICAL stuck_loop alert."""
        issue = "overexposed highlights"
        for i in range(1, 4):
            alerts = self.monitor.check_after_evaluation(
                score=30, primary_issue=issue, params={}, iteration=i
            )
        stuck_alerts = [a for a in alerts if a.alert_type == AlertType.STUCK_LOOP]
        self.assertEqual(len(stuck_alerts), 1)
        self.assertEqual(stuck_alerts[0].level, AlertLevel.CRITICAL)
        self.assertEqual(stuck_alerts[0].action, "force_technique_switch")

    def test_different_issues_no_stuck(self):
        """Different issues each iteration → no stuck alert."""
        for i, issue in enumerate(["dark", "overexposed", "clipped"], 1):
            alerts = self.monitor.check_after_evaluation(
                score=30, primary_issue=issue, params={}, iteration=i
            )
        stuck_alerts = [a for a in alerts if a.alert_type == AlertType.STUCK_LOOP]
        self.assertEqual(len(stuck_alerts), 0)

    def test_two_same_not_stuck(self):
        """Same issue only 2x → not yet stuck."""
        for i in range(1, 3):
            alerts = self.monitor.check_after_evaluation(
                score=30, primary_issue="blob shape", params={}, iteration=i
            )
        stuck_alerts = [a for a in alerts if a.alert_type == AlertType.STUCK_LOOP]
        self.assertEqual(len(stuck_alerts), 0)


class TestScorePlateau(unittest.TestCase):
    """Test score plateau detection: abs(delta) < 3.0 for 3 iterations."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_plateau_detected(self):
        """Scores [50, 51, 50.5] → plateau detected."""
        for i, score in enumerate([50.0, 51.0, 50.5], 1):
            alerts = self.monitor.check_after_evaluation(
                score=score, primary_issue=None, params={}, iteration=i
            )
        plateau_alerts = [a for a in alerts if a.alert_type == AlertType.SCORE_PLATEAU]
        self.assertEqual(len(plateau_alerts), 1)
        self.assertEqual(plateau_alerts[0].action, "escalate_escape_velocity")

    def test_improving_scores_no_plateau(self):
        """Scores [50, 60, 70] → no plateau."""
        for i, score in enumerate([50.0, 60.0, 70.0], 1):
            alerts = self.monitor.check_after_evaluation(
                score=score, primary_issue=None, params={}, iteration=i
            )
        plateau_alerts = [a for a in alerts if a.alert_type == AlertType.SCORE_PLATEAU]
        self.assertEqual(len(plateau_alerts), 0)

    def test_insufficient_history_no_plateau(self):
        """Only 2 data points → no plateau detection."""
        for i, score in enumerate([50.0, 50.5], 1):
            alerts = self.monitor.check_after_evaluation(
                score=score, primary_issue=None, params={}, iteration=i
            )
        plateau_alerts = [a for a in alerts if a.alert_type == AlertType.SCORE_PLATEAU]
        self.assertEqual(len(plateau_alerts), 0)


class TestWrongFeedbackCascade(unittest.TestCase):
    """Test cascade detection: score drops >5 after applying QA suggestion."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_cascade_detected(self):
        """Score drops 8 points after QA suggestion → WARNING cascade alert."""
        # Iteration 1: establish baseline
        self.monitor.check_after_evaluation(
            score=60.0, primary_issue="dim lighting", params={}, iteration=1
        )
        # Record QA suggestion before iteration 2
        self.monitor.record_qa_suggestion("Increase light energy to 500", current_score=60.0)
        # Iteration 2: score dropped after applying suggestion
        alerts = self.monitor.check_after_evaluation(
            score=52.0, primary_issue="overexposed", params={"energy": 500}, iteration=2
        )
        cascade_alerts = [a for a in alerts if a.alert_type == AlertType.WRONG_FEEDBACK_CASCADE]
        self.assertEqual(len(cascade_alerts), 1)
        self.assertEqual(cascade_alerts[0].action, "revert_params")
        self.assertAlmostEqual(cascade_alerts[0].details["drop"], 8.0, places=1)

    def test_no_cascade_when_score_improves(self):
        """Score improves after QA suggestion → no cascade."""
        self.monitor.check_after_evaluation(
            score=50.0, primary_issue=None, params={}, iteration=1
        )
        self.monitor.record_qa_suggestion("Add more detail", current_score=50.0)
        alerts = self.monitor.check_after_evaluation(
            score=60.0, primary_issue=None, params={}, iteration=2
        )
        cascade_alerts = [a for a in alerts if a.alert_type == AlertType.WRONG_FEEDBACK_CASCADE]
        self.assertEqual(len(cascade_alerts), 0)

    def test_small_drop_no_cascade(self):
        """Score drops only 3 points → below threshold, no cascade."""
        self.monitor.check_after_evaluation(
            score=60.0, primary_issue=None, params={}, iteration=1
        )
        self.monitor.record_qa_suggestion("Minor tweak", current_score=60.0)
        alerts = self.monitor.check_after_evaluation(
            score=57.0, primary_issue=None, params={}, iteration=2
        )
        cascade_alerts = [a for a in alerts if a.alert_type == AlertType.WRONG_FEEDBACK_CASCADE]
        self.assertEqual(len(cascade_alerts), 0)


class TestBudgetOverrun(unittest.TestCase):
    """Test budget overrun detection."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_budget_overrun_detected(self):
        """$0.60 spent on $0.50 budget → CRITICAL alert."""
        alerts = self.monitor.check_after_evaluation(
            score=50.0, primary_issue=None, params={}, iteration=1,
            budget_spent=0.60,
        )
        budget_alerts = [a for a in alerts if a.alert_type == AlertType.BUDGET_OVERRUN]
        self.assertEqual(len(budget_alerts), 1)
        self.assertEqual(budget_alerts[0].level, AlertLevel.CRITICAL)
        self.assertEqual(budget_alerts[0].action, "disable_expensive_tools")

    def test_within_budget_no_alert(self):
        """$0.30 spent → no alert."""
        alerts = self.monitor.check_after_evaluation(
            score=50.0, primary_issue=None, params={}, iteration=1,
            budget_spent=0.30,
        )
        budget_alerts = [a for a in alerts if a.alert_type == AlertType.BUDGET_OVERRUN]
        self.assertEqual(len(budget_alerts), 0)


class TestScriptQuality(unittest.TestCase):
    """Test script line count check in check_after_generation()."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def _write_script(self, line_count: int) -> str:
        lines = [f"# Line {i}" for i in range(line_count)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("\n".join(lines))
            return f.name

    def test_short_script_warns(self):
        """Script < 500 lines → WARNING alert."""
        path = self._write_script(200)
        try:
            alerts = self.monitor.check_after_generation(path, iteration=1)
            script_alerts = [a for a in alerts if a.alert_type == AlertType.SCRIPT_TOO_SHORT]
            self.assertEqual(len(script_alerts), 1)
            self.assertEqual(script_alerts[0].level, AlertLevel.WARNING)
        finally:
            os.unlink(path)

    def test_long_script_no_alert(self):
        """Script >= 500 lines → no alert."""
        path = self._write_script(600)
        try:
            alerts = self.monitor.check_after_generation(path, iteration=1)
            script_alerts = [a for a in alerts if a.alert_type == AlertType.SCRIPT_TOO_SHORT]
            self.assertEqual(len(script_alerts), 0)
        finally:
            os.unlink(path)


class TestCriticalRender(unittest.TestCase):
    """Test critical render issue detection."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_black_screen_detected(self):
        """BLACK_SCREEN in vision assessment → CRITICAL alert."""
        alerts = self.monitor.check_after_evaluation(
            score=0.0, primary_issue="BLACK_SCREEN",
            params={}, iteration=1, vision_assessment="Render is BLACK_SCREEN"
        )
        render_alerts = [a for a in alerts if a.alert_type == AlertType.CRITICAL_RENDER_ISSUE]
        self.assertEqual(len(render_alerts), 1)
        self.assertEqual(render_alerts[0].level, AlertLevel.CRITICAL)
        self.assertEqual(render_alerts[0].action, "skip_normal_iteration")

    def test_normal_render_no_alert(self):
        """Normal assessment → no critical alert."""
        alerts = self.monitor.check_after_evaluation(
            score=65.0, primary_issue="minor noise",
            params={}, iteration=1, vision_assessment="Good volumetric density"
        )
        render_alerts = [a for a in alerts if a.alert_type == AlertType.CRITICAL_RENDER_ISSUE]
        self.assertEqual(len(render_alerts), 0)


class TestStatusReport(unittest.TestCase):
    """Test get_status_report() output."""

    def setUp(self):
        self.monitor = PipelineMonitor()

    def test_status_report_structure(self):
        """Status report has all expected fields."""
        self.monitor.check_after_evaluation(
            score=50.0, primary_issue=None, params={"energy": 100}, iteration=1
        )
        report = self.monitor.get_status_report()
        self.assertIn("enabled", report)
        self.assertIn("iterations_tracked", report)
        self.assertIn("score_history", report)
        self.assertIn("score_trend", report)
        self.assertIn("params_tracked", report)
        self.assertIn("oscillating_params", report)
        self.assertIn("total_alerts", report)
        self.assertIn("alerts_by_level", report)
        self.assertIn("recent_alerts", report)

    def test_score_trend_improving(self):
        """Improving scores → trend = 'improving'."""
        for i, score in enumerate([30.0, 45.0, 60.0], 1):
            self.monitor.check_after_evaluation(
                score=score, primary_issue=None, params={}, iteration=i
            )
        report = self.monitor.get_status_report()
        self.assertEqual(report["score_trend"], "improving")

    def test_score_trend_declining(self):
        """Declining scores → trend = 'declining'."""
        for i, score in enumerate([60.0, 45.0, 30.0], 1):
            self.monitor.check_after_evaluation(
                score=score, primary_issue=None, params={}, iteration=i
            )
        report = self.monitor.get_status_report()
        self.assertEqual(report["score_trend"], "declining")


class TestIntegration(unittest.TestCase):
    """Integration test: mock a 3-iteration loop."""

    def test_three_iteration_monitoring(self):
        """Feed 3 iterations with varied signals → verify comprehensive report."""
        monitor = PipelineMonitor()

        # Iteration 1: generate + evaluate
        gen_alerts = monitor.check_after_generation("/tmp/fake_script.py", iteration=1)
        # File doesn't exist, so no script alert (handled gracefully)
        eval_alerts_1 = monitor.check_after_evaluation(
            score=40.0, primary_issue="dark scene",
            params={"energy": 50, "density": 5.0}, iteration=1,
            budget_spent=0.15,
        )

        # Iteration 2: score improves slightly
        eval_alerts_2 = monitor.check_after_evaluation(
            score=42.0, primary_issue="dark scene",
            params={"energy": 100, "density": 5.0}, iteration=2,
            budget_spent=0.30,
        )

        # Iteration 3: same issue again → stuck after 3x
        eval_alerts_3 = monitor.check_after_evaluation(
            score=41.5, primary_issue="dark scene",
            params={"energy": 80, "density": 5.0}, iteration=3,
            budget_spent=0.45,
        )

        # Should detect: stuck loop (3x "dark scene") and plateau
        stuck_alerts = [a for a in eval_alerts_3 if a.alert_type == AlertType.STUCK_LOOP]
        self.assertEqual(len(stuck_alerts), 1)

        # Verify status report
        report = monitor.get_status_report()
        self.assertEqual(report["iterations_tracked"], 3)
        self.assertEqual(len(report["score_history"]), 3)
        self.assertGreater(report["total_alerts"], 0)


class TestKillSwitch(unittest.TestCase):
    """Test ENABLE_PIPELINE_MONITOR kill switch."""

    def test_kill_switch_disables_generation_checks(self):
        """When disabled, check_after_generation returns empty."""
        with patch("tools.pipeline_monitor.ENABLE_PIPELINE_MONITOR", False):
            monitor = PipelineMonitor()
            alerts = monitor.check_after_generation("/tmp/fake.py", iteration=1)
            self.assertEqual(len(alerts), 0)

    def test_kill_switch_disables_evaluation_checks(self):
        """When disabled, check_after_evaluation returns empty."""
        with patch("tools.pipeline_monitor.ENABLE_PIPELINE_MONITOR", False):
            monitor = PipelineMonitor()
            alerts = monitor.check_after_evaluation(
                score=0.0, primary_issue="BLACK_SCREEN",
                params={}, iteration=1, budget_spent=10.0,
            )
            self.assertEqual(len(alerts), 0)


class TestReset(unittest.TestCase):
    """Test monitor reset (e.g., technique switch)."""

    def test_reset_clears_state(self):
        """After reset, no historical alerts or data."""
        monitor = PipelineMonitor()
        for i in range(3):
            monitor.check_after_evaluation(
                score=50.0, primary_issue="stuck", params={"x": i}, iteration=i + 1
            )
        monitor.reset()
        report = monitor.get_status_report()
        self.assertEqual(report["iterations_tracked"], 0)
        self.assertEqual(report["total_alerts"], 0)
        self.assertEqual(len(report["score_history"]), 0)


if __name__ == "__main__":
    unittest.main()
