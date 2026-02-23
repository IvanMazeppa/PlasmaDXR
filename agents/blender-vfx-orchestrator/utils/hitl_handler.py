"""
Human-in-the-Loop (HITL) Framework — Phase 2B-5.

Pipeline-level checkpoints that pause execution when the system needs
human guidance. Two modes:
  - Interactive: CLI input() prompt, blocks until human responds.
  - Non-interactive: Saves checkpoint to session state, sets PAUSED.

Autonomy levels gate which checkpoints fire:
  0 = Guided:         All checkpoints
  1 = Supervised:     Budget + critical + stall
  2 = Semi-autonomous: Only critical checkpoints
  3 = Autonomous:     Only L4 escalation
  4 = Full autonomous: No checkpoints (not even L4)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CheckpointType(str, Enum):
    STALL_DETECTION = "stall_detection"
    BUDGET_WARNING = "budget_warning"
    CRITICAL_ISSUE = "critical_issue"
    ESCALATION = "escalation"
    QUALITY_PLATEAU = "quality_plateau"


class CheckpointDecision(str, Enum):
    CONTINUE = "continue"
    SWITCH_TECHNIQUE = "switch"
    ADJUST_PARAMS = "adjust"
    ABORT = "abort"
    INCREASE_BUDGET = "increase"


# Map single-character shortcuts to decisions for CLI prompt
_DECISION_SHORTCUTS = {
    "c": CheckpointDecision.CONTINUE,
    "s": CheckpointDecision.SWITCH_TECHNIQUE,
    "a": CheckpointDecision.ADJUST_PARAMS,
    "q": CheckpointDecision.ABORT,
    "i": CheckpointDecision.INCREASE_BUDGET,
}


# =============================================================================
# CHECKPOINT DATA
# =============================================================================

@dataclass
class HITLCheckpoint:
    """A frozen snapshot of a pipeline checkpoint awaiting human decision."""

    checkpoint_type: CheckpointType
    reason: str
    context: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    decision: Optional[CheckpointDecision] = None
    human_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoint_type"] = self.checkpoint_type.value
        if self.decision is not None:
            d["decision"] = self.decision.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HITLCheckpoint":
        return cls(
            checkpoint_type=CheckpointType(data["checkpoint_type"]),
            reason=data["reason"],
            context=data.get("context", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            decision=CheckpointDecision(data["decision"]) if data.get("decision") else None,
            human_notes=data.get("human_notes"),
        )


# =============================================================================
# HITL HANDLER
# =============================================================================

class HITLHandler:
    """
    Pipeline-level HITL checkpoint manager.

    Checks conditions after evaluation and at escalation points,
    optionally prompting a human for decisions.
    """

    def __init__(
        self,
        autonomy_level: int = 1,
        interactive: bool = True,
        enabled: bool = True,
    ):
        self._autonomy_level = max(0, min(4, autonomy_level))
        self._interactive = interactive
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_post_eval(
        self,
        session: Any,
        quality: Any,
        budget: Dict[str, Any],
    ) -> Optional[HITLCheckpoint]:
        """
        After quality evaluation — check for budget warnings, stall, critical issues.

        Returns a checkpoint if one should fire, None otherwise.
        """
        if not self._enabled:
            return None

        # Level 4 = full autonomous, never fire post-eval checkpoints
        if self._autonomy_level >= 4:
            return None

        # Build context dict shared by all checkpoint types
        ctx = self._build_context(session, quality, budget)

        # Check in priority order (most severe first)
        if self._should_checkpoint_critical(quality, session):
            if self._autonomy_level <= 2:  # Levels 0, 1, 2
                return self._create_checkpoint(
                    CheckpointType.CRITICAL_ISSUE,
                    f"Score=0 for {self._consecutive_zero_count(session)} consecutive iterations",
                    ctx,
                )

        if self._should_checkpoint_stall(session):
            if self._autonomy_level <= 1:  # Levels 0, 1
                return self._create_checkpoint(
                    CheckpointType.STALL_DETECTION,
                    f"Same issue {session.stuck_state.same_issue_count}x at escape level {session.stuck_state.escape_level.value}",
                    ctx,
                )

        if self._should_checkpoint_budget(budget):
            if self._autonomy_level <= 1:  # Levels 0, 1
                return self._create_checkpoint(
                    CheckpointType.BUDGET_WARNING,
                    f"Budget >80% spent: ${budget.get('total_spent', 0):.2f}/${budget.get('total_remaining', 0) + budget.get('total_spent', 0):.2f}",
                    ctx,
                )

        if self._should_checkpoint_plateau(session):
            if self._autonomy_level <= 0:  # Level 0 only
                return self._create_checkpoint(
                    CheckpointType.QUALITY_PLATEAU,
                    f"Score plateau for {session.stuck_state.plateau_count} iterations",
                    ctx,
                )

        return None

    def check_escalation(self, session: Any) -> Optional[HITLCheckpoint]:
        """
        Escape L4 handler — replaces bare PAUSED+break.

        Always triggers regardless of autonomy (except level 4).
        """
        if not self._enabled:
            return None

        if self._autonomy_level >= 4:
            return None

        ctx = {
            "iteration": session.current_iteration,
            "best_score": session.best_score,
            "techniques_tried": list(session.stuck_state.techniques_tried),
            "escape_level": session.stuck_state.escape_level.value,
            "same_issue_count": session.stuck_state.same_issue_count,
        }
        return self._create_checkpoint(
            CheckpointType.ESCALATION,
            f"Escape Level 4: tried {len(session.stuck_state.techniques_tried)} techniques, best score {session.best_score:.1f}",
            ctx,
        )

    def check_pending(self, session: Any) -> Optional[HITLCheckpoint]:
        """
        On session resume — check for pending checkpoint needing decision.

        If interactive, prompts for decision now. Otherwise returns the
        checkpoint as-is (caller should check if decision is set).
        """
        if not session.pending_checkpoint:
            return None

        checkpoint = HITLCheckpoint.from_dict(session.pending_checkpoint)
        if checkpoint.decision is not None:
            # Decision was already provided (e.g., via JSON edit)
            self._record_history(session, checkpoint)
            return checkpoint

        if self._interactive:
            checkpoint.decision = self._prompt_interactive(checkpoint)
            self._record_history(session, checkpoint)

        return checkpoint

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_checkpoint(
        self,
        cp_type: CheckpointType,
        reason: str,
        context: Dict[str, Any],
    ) -> HITLCheckpoint:
        checkpoint = HITLCheckpoint(
            checkpoint_type=cp_type,
            reason=reason,
            context=context,
        )
        print(f"[Pipeline] HITL: {cp_type.value} — {reason}", file=sys.stderr)

        if self._interactive:
            checkpoint.decision = self._prompt_interactive(checkpoint)

        return checkpoint

    def _prompt_interactive(self, checkpoint: HITLCheckpoint) -> CheckpointDecision:
        """CLI prompt: display checkpoint info, get decision via input()."""
        print(
            f"\n{'='*60}\n"
            f"  HITL CHECKPOINT: {checkpoint.checkpoint_type.value.upper()}\n"
            f"{'='*60}\n"
            f"  Reason: {checkpoint.reason}\n",
            file=sys.stderr,
        )
        for key, val in checkpoint.context.items():
            print(f"  {key}: {val}", file=sys.stderr)

        print(
            f"\n  Options:\n"
            f"    [c] Continue as-is\n"
            f"    [s] Switch technique\n"
            f"    [a] Adjust parameters\n"
            f"    [q] Abort session\n"
            f"    [i] Increase budget\n",
            file=sys.stderr,
        )

        while True:
            try:
                choice = input("  Decision [c/s/a/q/i]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  (EOF/interrupt — defaulting to abort)", file=sys.stderr)
                return CheckpointDecision.ABORT

            if choice in _DECISION_SHORTCUTS:
                decision = _DECISION_SHORTCUTS[choice]
                print(f"  -> {decision.value}", file=sys.stderr)
                return decision

            print(f"  Invalid choice '{choice}'. Use c/s/a/q/i.", file=sys.stderr)

    def _should_checkpoint_budget(self, budget: Dict[str, Any]) -> bool:
        """True if budget > 80% spent."""
        total_spent = budget.get("total_spent", 0)
        total_limit = total_spent + budget.get("total_remaining", float("inf"))
        if total_limit <= 0:
            return False
        return (total_spent / total_limit) > 0.8

    def _should_checkpoint_stall(self, session: Any) -> bool:
        """True if same issue 3+ times at escape level >= 2."""
        return (
            session.stuck_state.same_issue_count >= 3
            and session.stuck_state.escape_level.value >= 2
        )

    def _should_checkpoint_critical(self, quality: Any, session: Any) -> bool:
        """True if score=0 for 2+ consecutive iterations."""
        return self._consecutive_zero_count(session) >= 2

    def _should_checkpoint_plateau(self, session: Any) -> bool:
        """True if score plateau for 3+ iterations."""
        return session.stuck_state.plateau_count >= 3

    def _consecutive_zero_count(self, session: Any) -> int:
        """Count trailing iterations with score=0."""
        count = 0
        for it in reversed(session.iterations):
            if it.score <= 0:
                count += 1
            else:
                break
        return count

    def _build_context(
        self,
        session: Any,
        quality: Any,
        budget: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "iteration": session.current_iteration,
            "score": quality.overall_score if hasattr(quality, "overall_score") else 0,
            "best_score": session.best_score,
            "primary_issue": getattr(quality, "primary_issue", None),
            "escape_level": session.stuck_state.escape_level.value,
            "techniques_tried": list(session.stuck_state.techniques_tried),
            "budget_spent": budget.get("total_spent", 0),
            "budget_remaining": budget.get("total_remaining", 0),
        }

    def _record_history(self, session: Any, checkpoint: HITLCheckpoint) -> None:
        """Append resolved checkpoint to session's hitl_history."""
        if hasattr(session, "hitl_history"):
            session.hitl_history.append(checkpoint.to_dict())
