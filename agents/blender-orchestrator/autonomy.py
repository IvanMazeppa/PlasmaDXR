"""
Adaptive Autonomy System

Manages trust score, autonomy levels, and decision permissions.
The orchestrator earns more autonomy with good performance and loses it with failures.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("blender-orchestrator.autonomy")


class AutonomyLevel(Enum):
    """Autonomy levels from most restrictive to most autonomous."""
    SUPERVISED = "supervised"   # Ask before everything
    GUIDED = "guided"           # Ask before technique changes
    AUTONOMOUS = "autonomous"   # Ask only for session end
    TRUSTED = "trusted"         # Fully autonomous


class DecisionType(Enum):
    """Types of decisions the orchestrator can make."""
    ITERATE = "iterate"                     # Continue with adjusted params
    CHANGE_PARAMS = "change_params"         # Modify specific parameters
    CHANGE_TECHNIQUE = "change_technique"   # Switch to different technique
    END_SESSION_PASS = "end_session_pass"   # End session, asset passed
    END_SESSION_FAIL = "end_session_fail"   # End session, asset failed
    RETRY_EXECUTION = "retry_execution"     # Retry failed Blender run
    ERROR_RECOVERY = "error_recovery"       # Handle unexpected error


class TrustEvent(Enum):
    """Events that affect trust score."""
    QUALITY_IMPROVED = "quality_improved"
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    ASSET_COMPLETED = "asset_completed"
    HUMAN_APPROVAL = "human_approval"
    QUALITY_DEGRADED = "quality_degraded"
    EXECUTION_ERROR = "execution_error"
    TECHNIQUE_CHANGE = "technique_change"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    HUMAN_OVERRIDE = "human_override"
    CRITICAL_FAILURE = "critical_failure"


@dataclass
class TrustAdjustments:
    """Configuration for trust score adjustments."""
    quality_improved: float = 0.05
    quality_threshold_met: float = 0.10
    asset_completed: float = 0.15
    human_approval: float = 0.02
    quality_degraded: float = -0.05
    execution_error: float = -0.10
    technique_change: float = -0.03
    token_limit_exceeded: float = -0.15
    human_override: float = -0.10
    critical_failure: float = -0.20

    def get_adjustment(self, event: TrustEvent) -> float:
        """Get the adjustment value for an event."""
        return getattr(self, event.value, 0.0)


@dataclass
class AutonomyConfig:
    """Configuration for the autonomy system."""
    initial_trust_score: float = 0.2
    override: Optional[str] = None  # Force specific level
    min_trust: float = 0.0
    max_trust: float = 1.0
    daily_decay: float = 0.02
    adjustments: TrustAdjustments = field(default_factory=TrustAdjustments)

    # Level thresholds
    supervised_max: float = 0.3
    guided_max: float = 0.6
    autonomous_max: float = 0.8

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "AutonomyConfig":
        """Create config from YAML dict."""
        autonomy = config.get("autonomy", {})

        adjustments_dict = autonomy.get("trust_adjustments", {})
        adjustments = TrustAdjustments(
            quality_improved=adjustments_dict.get("quality_improved", 0.05),
            quality_threshold_met=adjustments_dict.get("quality_threshold_met", 0.10),
            asset_completed=adjustments_dict.get("asset_completed", 0.15),
            human_approval=adjustments_dict.get("human_approval", 0.02),
            quality_degraded=adjustments_dict.get("quality_degraded", -0.05),
            execution_error=adjustments_dict.get("execution_error", -0.10),
            technique_change=adjustments_dict.get("technique_change", -0.03),
            token_limit_exceeded=adjustments_dict.get("token_limit_exceeded", -0.15),
            human_override=adjustments_dict.get("human_override", -0.10),
            critical_failure=adjustments_dict.get("critical_failure", -0.20),
        )

        levels = autonomy.get("levels", {})

        return cls(
            initial_trust_score=autonomy.get("initial_trust_score", 0.2),
            override=autonomy.get("override"),
            min_trust=autonomy.get("min_trust", 0.0),
            max_trust=autonomy.get("max_trust", 1.0),
            daily_decay=autonomy.get("daily_decay", 0.02),
            adjustments=adjustments,
            supervised_max=levels.get("supervised", {}).get("max_trust", 0.3),
            guided_max=levels.get("guided", {}).get("max_trust", 0.6),
            autonomous_max=levels.get("autonomous", {}).get("max_trust", 0.8),
        )


@dataclass
class TrustState:
    """Persistent trust state."""
    score: float
    level: AutonomyLevel
    last_updated: datetime
    history: list  # List of (event, adjustment, timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "score": self.score,
            "level": self.level.value,
            "last_updated": self.last_updated.isoformat(),
            "history": [
                {
                    "event": h[0].value if isinstance(h[0], TrustEvent) else h[0],
                    "adjustment": h[1],
                    "timestamp": h[2].isoformat()
                }
                for h in self.history[-50:]  # Keep last 50 events
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustState":
        """Create from dict."""
        history = []
        for h in data.get("history", []):
            try:
                event = TrustEvent(h["event"])
            except ValueError:
                event = h["event"]
            history.append((
                event,
                h["adjustment"],
                datetime.fromisoformat(h["timestamp"])
            ))

        return cls(
            score=data.get("score", 0.2),
            level=AutonomyLevel(data.get("level", "supervised")),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
            history=history
        )


class AutonomyController:
    """
    Controls the adaptive autonomy system.

    The orchestrator's autonomy level is determined by its trust score,
    which increases with successful operations and decreases with failures.
    """

    # Permission matrix: which decisions require approval at each level
    PERMISSION_MATRIX = {
        AutonomyLevel.SUPERVISED: {
            DecisionType.ITERATE: "ask",
            DecisionType.CHANGE_PARAMS: "ask",
            DecisionType.CHANGE_TECHNIQUE: "ask",
            DecisionType.END_SESSION_PASS: "ask",
            DecisionType.END_SESSION_FAIL: "ask",
            DecisionType.RETRY_EXECUTION: "ask",
            DecisionType.ERROR_RECOVERY: "ask",
        },
        AutonomyLevel.GUIDED: {
            DecisionType.ITERATE: "allow",
            DecisionType.CHANGE_PARAMS: "ask",
            DecisionType.CHANGE_TECHNIQUE: "ask",
            DecisionType.END_SESSION_PASS: "ask",
            DecisionType.END_SESSION_FAIL: "ask",
            DecisionType.RETRY_EXECUTION: "allow",
            DecisionType.ERROR_RECOVERY: "ask",
        },
        AutonomyLevel.AUTONOMOUS: {
            DecisionType.ITERATE: "allow",
            DecisionType.CHANGE_PARAMS: "allow",
            DecisionType.CHANGE_TECHNIQUE: "ask",
            DecisionType.END_SESSION_PASS: "ask",
            DecisionType.END_SESSION_FAIL: "ask",
            DecisionType.RETRY_EXECUTION: "allow",
            DecisionType.ERROR_RECOVERY: "ask",
        },
        AutonomyLevel.TRUSTED: {
            DecisionType.ITERATE: "allow",
            DecisionType.CHANGE_PARAMS: "allow",
            DecisionType.CHANGE_TECHNIQUE: "allow",
            DecisionType.END_SESSION_PASS: "notify",  # Just notify, don't ask
            DecisionType.END_SESSION_FAIL: "ask",     # Always ask on failure
            DecisionType.RETRY_EXECUTION: "allow",
            DecisionType.ERROR_RECOVERY: "allow",
        },
    }

    def __init__(self, config: AutonomyConfig, state_path: Optional[Path] = None):
        """
        Initialize autonomy controller.

        Args:
            config: Autonomy configuration
            state_path: Path to persist trust state
        """
        self.config = config
        self.state_path = state_path

        # Load or initialize state
        if state_path and state_path.exists():
            self.state = self._load_state()
        else:
            self.state = TrustState(
                score=config.initial_trust_score,
                level=self._score_to_level(config.initial_trust_score),
                last_updated=datetime.now(),
                history=[]
            )

        logger.info(
            f"Autonomy controller initialized: "
            f"trust={self.state.score:.2f}, level={self.state.level.value}"
        )

    def _load_state(self) -> TrustState:
        """Load trust state from file."""
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                state = TrustState.from_dict(data.get("trust_state", {}))

                # Apply daily decay
                days_since_update = (datetime.now() - state.last_updated).days
                if days_since_update > 0:
                    decay = self.config.daily_decay * days_since_update
                    state.score = max(self.config.min_trust, state.score - decay)
                    state.level = self._score_to_level(state.score)
                    logger.info(f"Applied {days_since_update} days of trust decay: -{decay:.2f}")

                return state
        except Exception as e:
            logger.warning(f"Failed to load trust state: {e}")
            return TrustState(
                score=self.config.initial_trust_score,
                level=self._score_to_level(self.config.initial_trust_score),
                last_updated=datetime.now(),
                history=[]
            )

    def _save_state(self) -> None:
        """Save trust state to file."""
        if not self.state_path:
            return

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing data or create new
            if self.state_path.exists():
                with open(self.state_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            data["trust_state"] = self.state.to_dict()

            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trust state: {e}")

    def _score_to_level(self, score: float) -> AutonomyLevel:
        """Convert trust score to autonomy level."""
        if score < self.config.supervised_max:
            return AutonomyLevel.SUPERVISED
        elif score < self.config.guided_max:
            return AutonomyLevel.GUIDED
        elif score < self.config.autonomous_max:
            return AutonomyLevel.AUTONOMOUS
        else:
            return AutonomyLevel.TRUSTED

    def get_level(self) -> AutonomyLevel:
        """
        Get current autonomy level.

        Returns:
            Current autonomy level (respects override if set)
        """
        if self.config.override:
            try:
                return AutonomyLevel(self.config.override)
            except ValueError:
                logger.warning(f"Invalid autonomy override: {self.config.override}")

        return self.state.level

    def get_trust_score(self) -> float:
        """Get current trust score."""
        return self.state.score

    def set_trust_score(self, score: float) -> None:
        """Manually set trust score (for testing/override)."""
        self.state.score = max(self.config.min_trust, min(self.config.max_trust, score))
        self.state.level = self._score_to_level(self.state.score)
        self.state.last_updated = datetime.now()
        self._save_state()

        logger.info(f"Trust score manually set to {self.state.score:.2f} ({self.state.level.value})")

    def set_override(self, level: Optional[str]) -> None:
        """Set or clear autonomy level override."""
        self.config.override = level
        logger.info(f"Autonomy override set to: {level}")

    def record_event(self, event: TrustEvent) -> float:
        """
        Record an event and adjust trust score.

        Args:
            event: The event that occurred

        Returns:
            New trust score
        """
        adjustment = self.config.adjustments.get_adjustment(event)
        old_score = self.state.score
        old_level = self.state.level

        # Apply adjustment with bounds
        self.state.score = max(
            self.config.min_trust,
            min(self.config.max_trust, self.state.score + adjustment)
        )
        self.state.level = self._score_to_level(self.state.score)
        self.state.last_updated = datetime.now()
        self.state.history.append((event, adjustment, datetime.now()))

        # Log significant changes
        if adjustment != 0:
            direction = "+" if adjustment > 0 else ""
            logger.info(
                f"Trust adjusted: {old_score:.2f} {direction}{adjustment:.2f} = {self.state.score:.2f} "
                f"(event: {event.value})"
            )

        if old_level != self.state.level:
            logger.info(f"Autonomy level changed: {old_level.value} -> {self.state.level.value}")

        self._save_state()
        return self.state.score

    def can_proceed(self, decision: DecisionType) -> str:
        """
        Check if a decision can proceed without approval.

        Args:
            decision: Type of decision to make

        Returns:
            "allow" - proceed autonomously
            "ask" - ask for human approval
            "notify" - proceed but notify human
            "deny" - not allowed at any level
        """
        level = self.get_level()
        return self.PERMISSION_MATRIX.get(level, {}).get(decision, "ask")

    def get_status(self) -> Dict[str, Any]:
        """Get full autonomy status."""
        return {
            "trust_score": self.state.score,
            "autonomy_level": self.get_level().value,
            "override_active": self.config.override is not None,
            "override_level": self.config.override,
            "last_updated": self.state.last_updated.isoformat(),
            "recent_events": [
                {
                    "event": h[0].value if isinstance(h[0], TrustEvent) else str(h[0]),
                    "adjustment": h[1],
                    "timestamp": h[2].isoformat()
                }
                for h in self.state.history[-10:]
            ]
        }

    def format_approval_request(
        self,
        decision: DecisionType,
        session_id: str,
        iteration: int,
        context: Dict[str, Any]
    ) -> str:
        """
        Format an approval request for the human.

        Args:
            decision: Type of decision
            session_id: Current session ID
            iteration: Current iteration number
            context: Additional context (scores, proposed action, etc.)

        Returns:
            Formatted approval request string
        """
        level = self.get_level()
        tokens_used = context.get("tokens_used", 0)
        tokens_limit = context.get("tokens_limit", 100000)

        request = f"""
{'=' * 79}
APPROVAL REQUESTED - Blender VFX Orchestrator
{'=' * 79}

Session: {session_id}
Iteration: {iteration}/{context.get('max_iterations', 10)}
Trust Level: {level.value.upper()} ({self.state.score:.2f})
Tokens Used: {tokens_used:,} / {tokens_limit:,}

DECISION: {decision.value.replace('_', ' ').title()}

"""
        # Add decision-specific context
        if decision == DecisionType.CHANGE_TECHNIQUE:
            current = context.get("current_technique", "unknown")
            proposed = context.get("proposed_technique", "unknown")
            scores = context.get("recent_scores", [])
            request += f"""Current State:
- Current Technique: {current}
- Recent Scores: {' -> '.join(f'{s:.1f}' for s in scores)}
- Proposed Technique: {proposed}
- Rationale: {context.get('rationale', 'Plateau detected')}

"""
        elif decision == DecisionType.END_SESSION_PASS:
            request += f"""Final State:
- Best Score: {context.get('best_score', 0):.1f}
- Best Iteration: {context.get('best_iteration', 0)}
- Output Path: {context.get('output_path', 'unknown')}

"""
        elif decision == DecisionType.END_SESSION_FAIL:
            request += f"""Failure Details:
- Current Score: {context.get('current_score', 0):.1f}
- Reason: {context.get('failure_reason', 'Unknown')}
- Attempts: {iteration}

"""

        request += """Options:
[1] APPROVE - Proceed with decision
[2] MODIFY  - Adjust the proposed action
[3] OVERRIDE - Take manual control
[4] ABORT   - End session, save current best

Your choice: """

        return request
