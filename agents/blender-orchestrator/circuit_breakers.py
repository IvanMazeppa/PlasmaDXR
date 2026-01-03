"""
Circuit Breakers for Blender VFX Orchestrator

Prevents runaway loops and resource exhaustion by enforcing hard limits
on iterations, time, and failure thresholds.

Circuit breakers are checked before each iteration. When triggered:
1. Log: "CIRCUIT BREAKER: {breaker_name} triggered"
2. Save session state via iteration-controller
3. Report final status with best score and output paths
4. DO NOT continue iterating
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

logger = logging.getLogger("blender-orchestrator.circuit_breakers")


@dataclass
class ConvergenceCriteria:
    """
    Circuit breaker configuration for autonomous iteration loops.

    Any breaker triggers immediate stop to prevent resource exhaustion.

    Attributes:
        max_iterations: Hard limit on improvement attempts per session (default: 10)
        max_wall_time_minutes: Total session duration limit in minutes (default: 30)
        no_improvement_threshold: Stop if best_score unchanged for N iterations (default: 3)
        quality_floor: Minimum quality score before pause (default: 40)
        quality_floor_consecutive: Number of consecutive low scores to trigger (default: 2)
        blender_failure_limit: Stop after N consecutive Blender failures (default: 2)
    """

    max_iterations: int = 10
    max_wall_time_minutes: int = 30
    no_improvement_threshold: int = 3
    quality_floor: int = 40
    quality_floor_consecutive: int = 2
    blender_failure_limit: int = 2

    def should_stop(
        self,
        current_iteration: int,
        start_time: datetime,
        best_score: float,
        score_history: List[float],
        consecutive_blender_failures: int = 0,
    ) -> Tuple[bool, str]:
        """
        Check if any circuit breaker should trigger a stop.

        Args:
            current_iteration: Current iteration number (1-based)
            start_time: When the session started
            best_score: Current best score achieved
            score_history: List of scores from all iterations
            consecutive_blender_failures: Number of Blender failures in a row

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        # Check max iterations
        if current_iteration >= self.max_iterations:
            return True, f"MAX_ITERATIONS reached ({self.max_iterations})"

        # Check wall time
        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        if elapsed_minutes > self.max_wall_time_minutes:
            return True, f"MAX_WALL_TIME exceeded ({elapsed_minutes:.1f} > {self.max_wall_time_minutes} min)"

        # Check for no improvement plateau
        if len(score_history) >= self.no_improvement_threshold:
            recent_scores = score_history[-self.no_improvement_threshold:]
            if all(s <= best_score for s in recent_scores):
                # Check if best_score hasn't improved
                if len(score_history) > self.no_improvement_threshold:
                    earlier_best = max(score_history[:-self.no_improvement_threshold])
                    if best_score <= earlier_best:
                        return True, f"NO_IMPROVEMENT for {self.no_improvement_threshold} iterations (score plateau at {best_score:.1f})"

        # Check quality floor (consecutive low scores)
        if len(score_history) >= self.quality_floor_consecutive:
            recent_scores = score_history[-self.quality_floor_consecutive:]
            if all(s < self.quality_floor for s in recent_scores):
                return True, f"QUALITY_FLOOR breached ({self.quality_floor_consecutive} consecutive scores < {self.quality_floor})"

        # Check Blender failures
        if consecutive_blender_failures >= self.blender_failure_limit:
            return True, f"BLENDER_FAILURE_LIMIT reached ({consecutive_blender_failures} consecutive failures)"

        return False, ""

    def should_pause(
        self,
        current_score: float,
        trust_score: float,
        parameters_at_limits: bool = False,
        conflicting_suggestions: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if session should pause for human review (not stop).

        Pause triggers are less severe than stop triggers and allow resumption
        after human review.

        Args:
            current_score: Most recent quality score
            trust_score: Current trust score (0.0 - 1.0)
            parameters_at_limits: True if parameters pushed > 90% of API limits
            conflicting_suggestions: True if knowledge base has conflicts

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        # Low trust + low quality = pause for review
        if trust_score < 0.3 and current_score < 60:
            return True, f"LOW_TRUST ({trust_score:.2f}) with LOW_QUALITY ({current_score:.1f})"

        # Parameters at API limits
        if parameters_at_limits:
            return True, "PARAMETERS_AT_LIMITS (> 90% of max range)"

        # Conflicting knowledge base suggestions
        if conflicting_suggestions:
            return True, "CONFLICTING_SUGGESTIONS from knowledge base"

        return False, ""


class CircuitBreakerState:
    """
    Tracks state needed for circuit breaker evaluation.

    Updated by the orchestrator after each iteration.
    """

    def __init__(self):
        self.start_time: datetime = datetime.now()
        self.score_history: List[float] = []
        self.consecutive_blender_failures: int = 0
        self.parameters_at_limits: bool = False
        self.conflicting_suggestions: bool = False

    def record_iteration(self, score: float, blender_succeeded: bool = True) -> None:
        """
        Record the result of an iteration.

        Args:
            score: VFX quality score from this iteration
            blender_succeeded: Whether Blender execution succeeded
        """
        self.score_history.append(score)

        if blender_succeeded:
            self.consecutive_blender_failures = 0
        else:
            self.consecutive_blender_failures += 1

    def reset_blender_failures(self) -> None:
        """Reset Blender failure counter (e.g., after successful execution)."""
        self.consecutive_blender_failures = 0

    def set_parameters_at_limits(self, at_limits: bool) -> None:
        """Update whether parameters are at API limits."""
        self.parameters_at_limits = at_limits

    def set_conflicting_suggestions(self, conflicting: bool) -> None:
        """Update whether knowledge base has conflicts."""
        self.conflicting_suggestions = conflicting

    @property
    def current_iteration(self) -> int:
        """Get current iteration number (1-based)."""
        return len(self.score_history) + 1

    @property
    def best_score(self) -> float:
        """Get best score achieved so far."""
        return max(self.score_history) if self.score_history else 0.0

    @property
    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (datetime.now() - self.start_time).total_seconds() / 60


def check_circuit_breakers(
    criteria: ConvergenceCriteria,
    state: CircuitBreakerState,
    trust_score: float = 0.5,
) -> Tuple[str, Optional[str]]:
    """
    Check all circuit breakers and return status.

    Args:
        criteria: Circuit breaker configuration
        state: Current circuit breaker state
        trust_score: Current trust score (0.0 - 1.0)

    Returns:
        Tuple of (status: "continue" | "stop" | "pause", reason: Optional[str])
    """
    # Check stop conditions first (more severe)
    should_stop, stop_reason = criteria.should_stop(
        current_iteration=state.current_iteration,
        start_time=state.start_time,
        best_score=state.best_score,
        score_history=state.score_history,
        consecutive_blender_failures=state.consecutive_blender_failures,
    )

    if should_stop:
        logger.warning(f"CIRCUIT BREAKER: {stop_reason}")
        return "stop", stop_reason

    # Check pause conditions
    current_score = state.score_history[-1] if state.score_history else 0.0
    should_pause, pause_reason = criteria.should_pause(
        current_score=current_score,
        trust_score=trust_score,
        parameters_at_limits=state.parameters_at_limits,
        conflicting_suggestions=state.conflicting_suggestions,
    )

    if should_pause:
        logger.warning(f"CIRCUIT BREAKER PAUSE: {pause_reason}")
        return "pause", pause_reason

    return "continue", None
