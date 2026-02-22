"""
Phase 2A-4: PipelineMonitor — Deterministic Monitoring Layer.

A pure Python class (NOT an LLM agent) that runs at orchestrator checkpoints
between pipeline phases. Detects parameter oscillation, stuck loops, wrong
feedback cascades, budget overruns, and script quality issues.

Cost: $0 (pure Python, no LLM calls).

Usage:
    monitor = PipelineMonitor()

    # After script generation:
    alerts = monitor.check_after_generation(script_path, iteration)

    # After quality evaluation:
    alerts = monitor.check_after_evaluation(
        score=quality.overall_score,
        primary_issue=quality.primary_issue,
        params=current_params,
        iteration=iteration,
        budget_spent=budget_tracker.total_spent(),
    )

    # Status report for logging:
    report = monitor.get_status_report()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Kill switch
ENABLE_PIPELINE_MONITOR = os.getenv("ENABLE_PIPELINE_MONITOR", "true").lower() != "false"


# =============================================================================
# ALERT TYPES
# =============================================================================

class AlertLevel(Enum):
    """Severity level for monitor alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of issues the monitor can detect."""
    PARAMETER_OSCILLATION = "parameter_oscillation"
    STUCK_LOOP = "stuck_loop"
    SCORE_PLATEAU = "score_plateau"
    WRONG_FEEDBACK_CASCADE = "wrong_feedback_cascade"
    BUDGET_OVERRUN = "budget_overrun"
    SCRIPT_TOO_SHORT = "script_too_short"
    CRITICAL_RENDER_ISSUE = "critical_render_issue"


@dataclass
class MonitorAlert:
    """A single alert from the pipeline monitor."""
    level: AlertLevel
    alert_type: AlertType
    message: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "type": self.alert_type.value,
            "message": self.message,
            "action": self.action,
            "details": self.details,
        }


# =============================================================================
# THRESHOLDS
# =============================================================================

# Parameter oscillation: direction reversal in recent window
# With 3 values [a, b, c], one direction change (up→down or down→up) = oscillation
OSCILLATION_WINDOW = 3
OSCILLATION_DIRECTION_CHANGES = 1

# Stuck loop: 3x same primary_issue
STUCK_LOOP_THRESHOLD = 3

# Score plateau: abs(score_delta) < 3.0 for 3 iterations
PLATEAU_DELTA_THRESHOLD = 3.0
PLATEAU_WINDOW = 3

# Wrong feedback cascade: score drops >5 after applying QA suggestion
CASCADE_DROP_THRESHOLD = 5.0

# Budget: per-run budget limit
PER_RUN_BUDGET = float(os.getenv("PER_RUN_BUDGET", "0.50"))

# Script quality: minimum lines
MIN_SCRIPT_LINES = 500

# Critical render keywords
CRITICAL_RENDER_KEYWORDS = [
    "BLACK_SCREEN",
    "WHITE_SCREEN",
    "ZERO_LIGHTS",
]


# =============================================================================
# PIPELINE MONITOR
# =============================================================================

class PipelineMonitor:
    """
    Deterministic monitoring layer for the VFX pipeline.

    Runs at orchestrator checkpoints between pipeline phases. All detection
    is pure Python — no LLM calls, $0 cost.
    """

    def __init__(self) -> None:
        # Score history: [(iteration, score)]
        self._score_history: List[Tuple[int, float]] = []

        # Parameter history: {param_name: [(iteration, value)]}
        self._param_history: Dict[str, List[Tuple[int, Any]]] = {}

        # Issue history: [(iteration, primary_issue)]
        self._issue_history: List[Tuple[int, Optional[str]]] = []

        # QA suggestion tracking for cascade detection
        self._last_qa_suggestion: Optional[str] = None
        self._score_before_suggestion: Optional[float] = None

        # Alert history for status report
        self._alerts: List[MonitorAlert] = []

    # =========================================================================
    # PUBLIC CHECKPOINTS
    # =========================================================================

    def check_after_generation(
        self,
        script_path: str,
        iteration: int,
    ) -> List[MonitorAlert]:
        """
        Run checks after script generation (PHASE 1).

        Checks:
        - Script line count (too short = too basic)

        Returns list of alerts (empty = all clear).
        """
        if not ENABLE_PIPELINE_MONITOR:
            return []

        alerts: List[MonitorAlert] = []

        # Script quality check
        try:
            text = Path(script_path).read_text()
            line_count = len(text.splitlines())
            if line_count < MIN_SCRIPT_LINES:
                alerts.append(MonitorAlert(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.SCRIPT_TOO_SHORT,
                    message=f"Script is {line_count} lines (< {MIN_SCRIPT_LINES}). "
                            f"May lack full scene setup.",
                    action="inject_warning",
                    details={"line_count": line_count, "threshold": MIN_SCRIPT_LINES},
                ))
        except Exception:
            pass  # File issues are handled by other pipeline stages

        self._alerts.extend(alerts)
        return alerts

    def check_after_evaluation(
        self,
        score: float,
        primary_issue: Optional[str],
        params: Dict[str, Any],
        iteration: int,
        budget_spent: float = 0.0,
        qa_suggestion: Optional[str] = None,
        vision_assessment: str = "",
    ) -> List[MonitorAlert]:
        """
        Run checks after quality evaluation (PHASE 3).

        Checks:
        - Parameter oscillation
        - Stuck loop (same primary issue)
        - Score plateau
        - Wrong feedback cascade
        - Budget overrun
        - Critical render issues

        Returns list of alerts (empty = all clear).
        """
        if not ENABLE_PIPELINE_MONITOR:
            return []

        alerts: List[MonitorAlert] = []

        # Record state
        self._score_history.append((iteration, score))
        self._issue_history.append((iteration, primary_issue))
        for param, value in params.items():
            if param not in self._param_history:
                self._param_history[param] = []
            self._param_history[param].append((iteration, value))

        # 1. Parameter oscillation
        alerts.extend(self._check_oscillation(params, iteration))

        # 2. Stuck loop
        alerts.extend(self._check_stuck_loop(primary_issue))

        # 3. Score plateau
        alerts.extend(self._check_plateau())

        # 4. Wrong feedback cascade
        alerts.extend(self._check_cascade(score))

        # 5. Budget overrun
        alerts.extend(self._check_budget(budget_spent))

        # 6. Critical render issues
        alerts.extend(self._check_critical_render(vision_assessment, primary_issue))

        # Track QA suggestion for next cascade check
        if qa_suggestion:
            self._last_qa_suggestion = qa_suggestion
            self._score_before_suggestion = score

        self._alerts.extend(alerts)
        return alerts

    def record_qa_suggestion(self, suggestion: str, current_score: float) -> None:
        """Record that a QA suggestion is about to be applied.

        Call this before applying the suggestion so we can detect cascades
        on the next evaluation.
        """
        self._last_qa_suggestion = suggestion
        self._score_before_suggestion = current_score

    def get_status_report(self) -> Dict[str, Any]:
        """Get a summary of monitor state for logging/artifacts."""
        scores = [s for _, s in self._score_history]
        return {
            "enabled": ENABLE_PIPELINE_MONITOR,
            "iterations_tracked": len(self._score_history),
            "score_history": scores,
            "score_trend": self._compute_trend(scores),
            "params_tracked": list(self._param_history.keys()),
            "oscillating_params": self._get_oscillating_params(),
            "total_alerts": len(self._alerts),
            "alerts_by_level": {
                "INFO": sum(1 for a in self._alerts if a.level == AlertLevel.INFO),
                "WARNING": sum(1 for a in self._alerts if a.level == AlertLevel.WARNING),
                "CRITICAL": sum(1 for a in self._alerts if a.level == AlertLevel.CRITICAL),
            },
            "recent_alerts": [a.to_dict() for a in self._alerts[-5:]],
        }

    def reset(self) -> None:
        """Reset all monitor state (e.g., when switching techniques)."""
        self._score_history.clear()
        self._param_history.clear()
        self._issue_history.clear()
        self._last_qa_suggestion = None
        self._score_before_suggestion = None
        self._alerts.clear()

    # =========================================================================
    # INTERNAL DETECTION METHODS
    # =========================================================================

    def _check_oscillation(
        self, params: Dict[str, Any], iteration: int
    ) -> List[MonitorAlert]:
        """Detect parameters that oscillate (change direction 2+ times in 3 iterations)."""
        alerts = []
        for param, history in self._param_history.items():
            if len(history) < OSCILLATION_WINDOW:
                continue

            recent = history[-OSCILLATION_WINDOW:]
            values = [v for _, v in recent]

            if not all(isinstance(v, (int, float)) for v in values):
                continue

            direction_changes = 0
            for i in range(1, len(values) - 1):
                prev_dir = values[i] - values[i - 1]
                next_dir = values[i + 1] - values[i]
                if prev_dir != 0 and next_dir != 0:
                    if (prev_dir > 0) != (next_dir > 0):
                        direction_changes += 1

            if direction_changes >= OSCILLATION_DIRECTION_CHANGES:
                midpoint = sum(values) / len(values)
                alerts.append(MonitorAlert(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.PARAMETER_OSCILLATION,
                    message=f"Parameter '{param}' is oscillating: {values}. "
                            f"Direction changed {direction_changes}x in {OSCILLATION_WINDOW} iterations.",
                    action="clamp_to_midpoint",
                    details={
                        "param": param,
                        "values": values,
                        "midpoint": round(midpoint, 2),
                        "direction_changes": direction_changes,
                    },
                ))

        return alerts

    def _check_stuck_loop(self, primary_issue: Optional[str]) -> List[MonitorAlert]:
        """Detect when the same primary issue repeats 3+ times."""
        if not primary_issue:
            return []

        # Count consecutive same issue from the end
        consecutive = 0
        for _, issue in reversed(self._issue_history):
            if issue == primary_issue:
                consecutive += 1
            else:
                break

        if consecutive >= STUCK_LOOP_THRESHOLD:
            return [MonitorAlert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.STUCK_LOOP,
                message=f"Same primary issue '{primary_issue}' for {consecutive} consecutive iterations. "
                        f"Pipeline is stuck.",
                action="force_technique_switch",
                details={
                    "issue": primary_issue,
                    "consecutive_count": consecutive,
                },
            )]
        return []

    def _check_plateau(self) -> List[MonitorAlert]:
        """Detect score plateau (abs(delta) < 3.0 for 3 iterations)."""
        if len(self._score_history) < PLATEAU_WINDOW:
            return []

        recent_scores = [s for _, s in self._score_history[-PLATEAU_WINDOW:]]
        deltas = [
            abs(recent_scores[i] - recent_scores[i - 1])
            for i in range(1, len(recent_scores))
        ]

        if all(d < PLATEAU_DELTA_THRESHOLD for d in deltas):
            return [MonitorAlert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.SCORE_PLATEAU,
                message=f"Score plateau detected: deltas {[round(d, 1) for d in deltas]} "
                        f"all < {PLATEAU_DELTA_THRESHOLD} over {PLATEAU_WINDOW} iterations.",
                action="escalate_escape_velocity",
                details={
                    "scores": recent_scores,
                    "deltas": [round(d, 1) for d in deltas],
                },
            )]
        return []

    def _check_cascade(self, current_score: float) -> List[MonitorAlert]:
        """Detect wrong feedback cascade (score drops >5 after QA suggestion)."""
        if self._score_before_suggestion is None or self._last_qa_suggestion is None:
            return []

        drop = self._score_before_suggestion - current_score
        if drop > CASCADE_DROP_THRESHOLD:
            alert = MonitorAlert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.WRONG_FEEDBACK_CASCADE,
                message=f"Score dropped {drop:.1f} points after applying QA suggestion: "
                        f"'{self._last_qa_suggestion[:80]}'. Likely wrong feedback.",
                action="revert_params",
                details={
                    "score_before": self._score_before_suggestion,
                    "score_after": current_score,
                    "drop": round(drop, 1),
                    "suggestion": self._last_qa_suggestion,
                },
            )
            # Reset cascade tracking after detection
            self._last_qa_suggestion = None
            self._score_before_suggestion = None
            return [alert]

        # Clear suggestion tracking if no cascade detected
        self._last_qa_suggestion = None
        self._score_before_suggestion = None
        return []

    def _check_budget(self, budget_spent: float) -> List[MonitorAlert]:
        """Check if per-run budget has been exceeded."""
        if budget_spent > PER_RUN_BUDGET:
            return [MonitorAlert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.BUDGET_OVERRUN,
                message=f"Budget overrun: ${budget_spent:.2f} spent vs "
                        f"${PER_RUN_BUDGET:.2f} per-run limit.",
                action="disable_expensive_tools",
                details={
                    "spent": round(budget_spent, 4),
                    "limit": PER_RUN_BUDGET,
                },
            )]
        return []

    def _check_critical_render(
        self, vision_assessment: str, primary_issue: Optional[str]
    ) -> List[MonitorAlert]:
        """Check for critical render failures in evaluation text."""
        combined = f"{vision_assessment} {primary_issue or ''}"
        found = [kw for kw in CRITICAL_RENDER_KEYWORDS if kw in combined]
        if found:
            return [MonitorAlert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.CRITICAL_RENDER_ISSUE,
                message=f"Critical render failure: {', '.join(found)}. "
                        f"Skip normal iteration — requires targeted fix.",
                action="skip_normal_iteration",
                details={"keywords": found},
            )]
        return []

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_oscillating_params(self) -> List[str]:
        """Return list of currently oscillating parameter names."""
        oscillating = []
        for param, history in self._param_history.items():
            if len(history) < OSCILLATION_WINDOW:
                continue
            recent = history[-OSCILLATION_WINDOW:]
            values = [v for _, v in recent]
            if not all(isinstance(v, (int, float)) for v in values):
                continue
            direction_changes = 0
            for i in range(1, len(values) - 1):
                prev_dir = values[i] - values[i - 1]
                next_dir = values[i + 1] - values[i]
                if prev_dir != 0 and next_dir != 0 and (prev_dir > 0) != (next_dir > 0):
                    direction_changes += 1
            if direction_changes >= OSCILLATION_DIRECTION_CHANGES:
                oscillating.append(param)
        return oscillating

    @staticmethod
    def _compute_trend(scores: List[float]) -> str:
        """Compute a simple trend label from score history."""
        if len(scores) < 2:
            return "insufficient_data"
        recent = scores[-3:] if len(scores) >= 3 else scores
        avg_delta = sum(recent[i] - recent[i - 1] for i in range(1, len(recent))) / (len(recent) - 1)
        if avg_delta > 3.0:
            return "improving"
        elif avg_delta < -3.0:
            return "declining"
        else:
            return "plateau"
