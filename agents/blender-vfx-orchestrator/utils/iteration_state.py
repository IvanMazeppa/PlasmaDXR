"""Phase 2B-1: Iteration state snapshot for stateless iterations.

Provides a read-only snapshot of cross-iteration state that can be
formatted into compact prompt context (~100 tokens). This is a safety
net — prompts already embed most context via f-strings, but this
provides a consistent format for optional enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from models.shared_context import SessionState


@dataclass(frozen=True)
class IterationSnapshot:
    """Read-only snapshot of cross-iteration state."""

    iteration: int
    max_iterations: int
    effect_type: str
    technique: str
    best_score: float
    best_iteration: int
    previous_score: float
    previous_primary_issue: Optional[str]
    escape_level: int
    same_issue_count: int
    plateau_count: int
    techniques_tried: List[str]
    techniques_failed: List[str]
    iteration_scores: List[float]

    @classmethod
    def from_session(
        cls,
        session: "SessionState",
        previous_score: float = 0.0,
        previous_primary_issue: Optional[str] = None,
    ) -> "IterationSnapshot":
        """Build snapshot from SessionState."""
        stuck = session.stuck_state
        scores = [r.score for r in session.iterations]
        return cls(
            iteration=session.current_iteration,
            max_iterations=session.request.max_iterations,
            effect_type=(
                session.request.effect_type.value
                if hasattr(session.request.effect_type, "value")
                else str(session.request.effect_type)
            ),
            technique=session.current_technique or "",
            best_score=session.best_score,
            best_iteration=session.best_iteration,
            previous_score=previous_score,
            previous_primary_issue=previous_primary_issue,
            escape_level=stuck.escape_level.value if hasattr(stuck.escape_level, "value") else int(stuck.escape_level),
            same_issue_count=stuck.same_issue_count,
            plateau_count=stuck.plateau_count,
            techniques_tried=list(stuck.techniques_tried),
            techniques_failed=list(stuck.techniques_failed),
            iteration_scores=scores,
        )

    def format_for_prompt(self, max_history: int = 5) -> str:
        """Compact cross-iteration context for agent prompts (~100 tokens)."""
        scores = self.iteration_scores[-max_history:]
        score_str = ", ".join(f"{s:.1f}" for s in scores) if scores else "none"
        issue_str = self.previous_primary_issue or "none"

        lines = [
            "## Cross-Iteration Context",
            f"Iteration: {self.iteration}/{self.max_iterations} | Best: {self.best_score:.1f} (iter {self.best_iteration}) | Previous: {self.previous_score:.1f}",
            f"Escape Level: {self.escape_level} | Same Issue: {self.same_issue_count}x | Plateau: {self.plateau_count}x",
            f"Previous Issue: {issue_str}",
            f"Score Trend: [{score_str}]",
        ]
        if self.techniques_tried:
            lines.append(f"Tried: {', '.join(self.techniques_tried)}")
        if self.techniques_failed:
            lines.append(f"Failed: {', '.join(self.techniques_failed)}")
        return "\n".join(lines)
