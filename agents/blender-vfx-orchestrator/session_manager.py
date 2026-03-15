"""
Session Manager for Experiment State Tracking.

This module provides Python-side (not LLM) state management for the VFX
orchestrator's experiment loop. It ensures:
1. Baselines are always recorded before experiments
2. Iteration history is tracked and available to agents
3. Parameter changes are correlated with score changes

Key insight: This is deterministic Python code, not LLM prompts.
It removes the burden of state tracking from agents, reducing turn budget
and eliminating the "forgot to call record_baseline" failure mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from models.shared_context import SessionState


@dataclass
class ExperimentState:
    """Snapshot of experiment state at a point in time."""
    params: Dict[str, Any]
    score: float
    script_path: str
    render_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "score": self.score,
            "script_path": self.script_path,
            "render_path": self.render_path,
            "timestamp": self.timestamp,
        }


@dataclass
class IterationRecord:
    """Record of a single iteration's results."""
    iteration: int
    params: Dict[str, Any]
    score: float
    delta: float  # Change from previous iteration
    issues: List[str]
    primary_issue: Optional[str] = None
    technique: Optional[str] = None
    params_changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # {param: (old, new)}

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "params": self.params,
            "score": self.score,
            "delta": self.delta,
            "issues": self.issues,
            "primary_issue": self.primary_issue,
            "technique": self.technique,
            "params_changed": {k: list(v) for k, v in self.params_changed.items()},
        }


@dataclass
class IssueTracker:
    """Tracks issue frequency to detect when we're stuck."""
    issue_counts: Dict[str, int] = field(default_factory=dict)
    issue_first_seen: Dict[str, int] = field(default_factory=dict)  # issue → iteration
    consecutive_same_issue: int = 0
    last_primary_issue: Optional[str] = None

    def record_issue(self, issue: str, iteration: int) -> None:
        """Record an issue occurrence."""
        # Normalize issue text for matching
        normalized = self._normalize_issue(issue)

        if normalized not in self.issue_counts:
            self.issue_counts[normalized] = 0
            self.issue_first_seen[normalized] = iteration
        self.issue_counts[normalized] += 1

    def record_primary_issue(self, issue: str, iteration: int) -> int:
        """Record primary issue and return consecutive count."""
        if issue == self.last_primary_issue:
            self.consecutive_same_issue += 1
        else:
            self.consecutive_same_issue = 1
            self.last_primary_issue = issue
        self.record_issue(issue, iteration)
        return self.consecutive_same_issue

    def get_stuck_issues(self, min_count: int = 3) -> List[str]:
        """Get issues that have appeared multiple times without resolution."""
        return [issue for issue, count in self.issue_counts.items() if count >= min_count]

    def _normalize_issue(self, issue: str) -> str:
        """Normalize issue text for comparison."""
        # Extract key terms, ignore minor wording differences
        key_terms = [
            "overexposed", "clipped", "static", "no animation", "no movement",
            "blob", "not spherical", "no detail", "no texture", "no surface",
            "no corona", "no limb", "monochrome", "white", "dark", "black"
        ]
        found = [term for term in key_terms if term.lower() in issue.lower()]
        return "|".join(sorted(found)) if found else issue[:50].lower()

    def reset(self) -> None:
        """
        Reset issue tracker state.

        QW-3: Call this when switching techniques to prevent old failure
        streaks from affecting the new approach.
        """
        self.issue_counts.clear()
        self.issue_first_seen.clear()
        self.consecutive_same_issue = 0
        self.last_primary_issue = None


class SessionManager:
    """
    Manages experiment state without LLM intervention.

    This class provides deterministic tracking of:
    - Baseline state before each experiment
    - Results after each iteration
    - Iteration history for context
    - Parameter values that have been tried
    - Issues that persist across iterations

    Usage in orchestrator:
        session_mgr = SessionManager()

        # Before iteration 2+:
        session_mgr.record_baseline(params, score, script_path)

        # After quality evaluation:
        session_mgr.record_result(iteration, params, score, issues, technique)

        # When building agent prompts:
        history = session_mgr.get_iteration_summary()
        context = session_mgr.get_context_for_agents()
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.baseline: Optional[ExperimentState] = None
        self.iteration_history: List[IterationRecord] = []
        self.issue_tracker = IssueTracker()
        self.params_tried: Dict[str, List[Any]] = {}  # param → list of values tried
        self.params_that_helped: Dict[str, List[Tuple[Any, float]]] = {}  # param → [(value, score_delta)]
        self.params_that_hurt: Dict[str, List[Tuple[Any, float]]] = {}
        self.techniques_tried: List[str] = []
        self.best_score: float = 0.0
        self.best_iteration: int = 0
        self.best_params: Dict[str, Any] = {}

    def record_baseline(
        self,
        params: Dict[str, Any],
        score: float,
        script_path: str,
        render_path: Optional[str] = None
    ) -> None:
        """
        Record baseline state before an experiment.

        Must be called before record_result() for meaningful comparisons.

        Args:
            params: Current parameter values
            score: Current quality score
            script_path: Path to current script
            render_path: Path to current render (if available)
        """
        self.baseline = ExperimentState(
            params=params.copy(),
            score=score,
            script_path=script_path,
            render_path=render_path
        )

        # Track all param values we've seen
        for param, value in params.items():
            if param not in self.params_tried:
                self.params_tried[param] = []
            if value not in self.params_tried[param]:
                self.params_tried[param].append(value)

    def record_result(
        self,
        iteration: int,
        params: Dict[str, Any],
        score: float,
        issues: List[str],
        primary_issue: Optional[str] = None,
        technique: Optional[str] = None
    ) -> IterationRecord:
        """
        Record the result of an iteration.

        Args:
            iteration: Current iteration number
            params: Parameter values used
            score: Quality score achieved
            issues: List of issues identified
            primary_issue: Most critical issue
            technique: Technique name used

        Returns:
            IterationRecord with computed delta and change tracking
        """
        # Compute delta from baseline
        delta = 0.0
        params_changed: Dict[str, Tuple[Any, Any]] = {}

        if self.baseline:
            delta = score - self.baseline.score

            # Track which params changed and correlate with score
            for param, new_value in params.items():
                old_value = self.baseline.params.get(param)
                if old_value != new_value:
                    params_changed[param] = (old_value, new_value)

                    # Track whether this change helped or hurt
                    if delta > 0:
                        if param not in self.params_that_helped:
                            self.params_that_helped[param] = []
                        self.params_that_helped[param].append((new_value, delta))
                    elif delta < 0:
                        if param not in self.params_that_hurt:
                            self.params_that_hurt[param] = []
                        self.params_that_hurt[param].append((new_value, delta))

        # Track technique
        if technique and technique not in self.techniques_tried:
            self.techniques_tried.append(technique)

        # Track issues
        for issue in issues:
            self.issue_tracker.record_issue(issue, iteration)
        if primary_issue:
            self.issue_tracker.record_primary_issue(primary_issue, iteration)

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = iteration
            self.best_params = params.copy()

        # Track param values
        for param, value in params.items():
            if param not in self.params_tried:
                self.params_tried[param] = []
            if value not in self.params_tried[param]:
                self.params_tried[param].append(value)

        # Create record
        record = IterationRecord(
            iteration=iteration,
            params=params.copy(),
            score=score,
            delta=delta,
            issues=issues,
            primary_issue=primary_issue,
            technique=technique,
            params_changed=params_changed
        )

        self.iteration_history.append(record)
        return record

    def get_iteration_summary(self, max_iterations: int = 5) -> str:
        """
        Get a text summary of iteration history for agent prompts.

        Args:
            max_iterations: Maximum iterations to include (most recent)

        Returns:
            Formatted string suitable for inclusion in prompts
        """
        if not self.iteration_history:
            return "## Iteration History\nNo previous iterations."

        lines = ["## Iteration History"]

        # Show recent iterations
        recent = self.iteration_history[-max_iterations:]
        for rec in recent:
            delta_str = f"+{rec.delta:.1f}" if rec.delta >= 0 else f"{rec.delta:.1f}"
            lines.append(f"- Iter {rec.iteration}: score={rec.score:.1f} ({delta_str})")

            if rec.params_changed:
                changes = [f"{k}: {v[0]}→{v[1]}" for k, v in list(rec.params_changed.items())[:3]]
                lines.append(f"  Changed: {', '.join(changes)}")

            if rec.primary_issue:
                lines.append(f"  Issue: {rec.primary_issue[:60]}...")

        # Show stuck issues
        stuck = self.issue_tracker.get_stuck_issues()
        if stuck:
            lines.append(f"\n**Persistent Issues (appeared 3+ times):** {', '.join(stuck)}")

        # Show best so far
        lines.append(f"\n**Best:** {self.best_score:.1f} (iteration {self.best_iteration})")

        return "\n".join(lines)

    def get_context_for_agents(self) -> Dict[str, Any]:
        """
        Get structured context for agent prompts.

        Returns:
            Dict with iteration_summary, stuck_issues, params_to_avoid, etc.
        """
        return {
            "iteration_summary": self.get_iteration_summary(),
            "current_iteration": len(self.iteration_history) + 1,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "stuck_issues": self.issue_tracker.get_stuck_issues(),
            "consecutive_same_issue": self.issue_tracker.consecutive_same_issue,
            "techniques_tried": self.techniques_tried,
            "params_that_helped": {k: v[-1] for k, v in self.params_that_helped.items()} if self.params_that_helped else {},
            "params_that_hurt": {k: v[-1] for k, v in self.params_that_hurt.items()} if self.params_that_hurt else {},
            # Phase 1 wiring fix: expose deterministic stuck-state to agent prompts
            # so Quality Gate and other agents see real escape/plateau values.
            "escape_level": 0,  # Default; orchestrator overwrites from session.stuck_state
            "plateau_count": 0,
        }

    def get_iteration_history_json(self) -> str:
        """
        Get iteration history as JSON for pre_iteration_research.

        Returns a JSON array with score, issue (primary_issue), and technique
        for each iteration - the format expected by pre_iteration_research_direct().

        Returns:
            JSON string of iteration history array
        """
        history = [
            {
                "score": rec.score,
                "issue": rec.primary_issue or "",
                "primary_issue": rec.primary_issue or "",  # Alias for compatibility
                "technique": rec.technique or "",
                "technique_name": rec.technique or "",  # Alias for compatibility
            }
            for rec in self.iteration_history
        ]
        return json.dumps(history)

    def should_switch_technique(self, threshold: int = 3) -> bool:
        """
        Determine if we should try a different technique.

        Returns True if:
        - Same primary issue persists for `threshold` iterations
        - We have untried alternative techniques available
        """
        return self.issue_tracker.consecutive_same_issue >= threshold

    def reset_for_technique_switch(self, new_technique: str) -> None:
        """
        Reset stuck state when switching to a new technique.

        QW-3: Prevents old failure streaks from affecting the new approach.
        Call this when escape_level triggers a technique switch.

        Args:
            new_technique: Name of the new technique being tried
        """
        self.issue_tracker.reset()
        if new_technique and new_technique not in self.techniques_tried:
            self.techniques_tried.append(new_technique)
        # Note: We keep params_that_helped/hurt - those might still be useful

    def get_params_to_avoid(self) -> Dict[str, List[Any]]:
        """Get parameter values that consistently hurt scores."""
        avoid = {}
        for param, outcomes in self.params_that_hurt.items():
            # Values that hurt more than they helped
            hurt_values = [v for v, delta in outcomes if delta < -2.0]
            if hurt_values:
                avoid[param] = hurt_values
        return avoid

    def get_suggested_params(self) -> Dict[str, Any]:
        """Get parameter values that consistently helped."""
        suggested = {}
        for param, outcomes in self.params_that_helped.items():
            # Get the value that helped the most
            if outcomes:
                best_value, best_delta = max(outcomes, key=lambda x: x[1])
                if best_delta > 2.0:  # Only suggest if significant improvement
                    suggested[param] = best_value
        return suggested

    def get_baseline_for_experiment_tracker(self) -> Dict[str, Any]:
        """
        Get baseline data formatted for experiment_tracker_tools.record_baseline().

        Returns:
            Dict with params, scores, render_path, script_path as JSON-ready strings
        """
        if not self.baseline:
            return {}

        return {
            "params": json.dumps(self.baseline.params),
            "scores": json.dumps({"overall": self.baseline.score}),
            "render_path": self.baseline.render_path or "",
            "script_path": self.baseline.script_path,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "iteration_history": [r.to_dict() for r in self.iteration_history],
            "params_tried": self.params_tried,
            "params_that_helped": {k: list(v) for k, v in self.params_that_helped.items()},
            "params_that_hurt": {k: list(v) for k, v in self.params_that_hurt.items()},
            "techniques_tried": self.techniques_tried,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "best_params": self.best_params,
            "issue_tracker": {
                "issue_counts": self.issue_tracker.issue_counts,
                "consecutive_same_issue": self.issue_tracker.consecutive_same_issue,
                "last_primary_issue": self.issue_tracker.last_primary_issue,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionManager":
        """Deserialize from dictionary."""
        mgr = cls(session_id=data.get("session_id"))
        mgr.params_tried = data.get("params_tried", {})
        mgr.params_that_helped = {k: [tuple(x) for x in v] for k, v in data.get("params_that_helped", {}).items()}
        mgr.params_that_hurt = {k: [tuple(x) for x in v] for k, v in data.get("params_that_hurt", {}).items()}
        mgr.techniques_tried = data.get("techniques_tried", [])
        mgr.best_score = data.get("best_score", 0.0)
        mgr.best_iteration = data.get("best_iteration", 0)
        mgr.best_params = data.get("best_params", {})

        if data.get("baseline"):
            b = data["baseline"]
            mgr.baseline = ExperimentState(
                params=b["params"],
                score=b["score"],
                script_path=b["script_path"],
                render_path=b.get("render_path"),
                timestamp=b.get("timestamp", "")
            )

        for r in data.get("iteration_history", []):
            mgr.iteration_history.append(IterationRecord(
                iteration=r["iteration"],
                params=r["params"],
                score=r["score"],
                delta=r["delta"],
                issues=r["issues"],
                primary_issue=r.get("primary_issue"),
                technique=r.get("technique"),
                params_changed={k: tuple(v) for k, v in r.get("params_changed", {}).items()}
            ))

        if data.get("issue_tracker"):
            it = data["issue_tracker"]
            mgr.issue_tracker.issue_counts = it.get("issue_counts", {})
            mgr.issue_tracker.consecutive_same_issue = it.get("consecutive_same_issue", 0)
            mgr.issue_tracker.last_primary_issue = it.get("last_primary_issue")

        return mgr

    @classmethod
    def from_session_state(cls, session: "SessionState") -> "SessionManager":
        """
        Reconstruct a SessionManager by replaying iteration history from a SessionState.

        This properly rebuilds all derived state (issue_tracker, params_that_helped/hurt,
        best_score, etc.) by running each iteration through record_result(), avoiding
        duplication of derivation logic.

        Args:
            session: Loaded SessionState with iterations to replay

        Returns:
            SessionManager with fully reconstructed state
        """
        mgr = cls(session_id=session.session_id)

        # Record initial baseline (before any iterations)
        mgr.record_baseline(params={}, score=0.0, script_path="", render_path=None)

        for i, iter_result in enumerate(session.iterations):
            params = iter_result.script.modifications or {}

            # Record baseline for iteration 2+ (mirrors create_asset_pipeline baseline logic)
            if i > 0:
                prev = session.iterations[i - 1]
                mgr.record_baseline(
                    params=prev.script.modifications or {},
                    score=prev.score,
                    script_path=prev.script.script_path,
                    render_path=prev.execution.render_path,
                )

            mgr.record_result(
                iteration=iter_result.iteration,
                params=params,
                score=iter_result.score,
                issues=iter_result.quality.issues if iter_result.quality else [],
                primary_issue=iter_result.quality.primary_issue if iter_result.quality else None,
                technique=iter_result.script.technique_name,
            )

        return mgr
