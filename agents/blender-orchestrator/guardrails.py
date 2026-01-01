"""
Token Usage Guardrails

Tracks token usage and prevents runaway API costs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("blender-orchestrator.guardrails")


@dataclass
class TokenPricing:
    """Token pricing configuration (Claude Opus 4.5)."""
    input_per_million: float = 15.00   # $15/1M input tokens
    output_per_million: float = 75.00  # $75/1M output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return input_cost + output_cost


@dataclass
class TokenLimits:
    """Token usage limits."""
    per_session: int = 100000
    per_iteration: int = 20000
    per_tool_call: int = 5000
    absolute_max: int = 150000


@dataclass
class CostLimits:
    """Cost limits in USD."""
    per_session: float = 5.00
    per_day: float = 20.00
    per_week: float = 75.00


@dataclass
class GuardrailConfig:
    """Full guardrail configuration."""
    token_limits: TokenLimits = field(default_factory=TokenLimits)
    cost_limits: CostLimits = field(default_factory=CostLimits)
    pricing: TokenPricing = field(default_factory=TokenPricing)
    warning_threshold: float = 0.80
    pause_threshold: float = 0.95
    hard_stop_threshold: float = 1.00

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "GuardrailConfig":
        """Create config from YAML dict."""
        guardrails = config.get("guardrails", {})
        tokens = guardrails.get("token_limits", {})
        costs = guardrails.get("cost_limits", {})
        pricing = guardrails.get("pricing", {})
        thresholds = guardrails.get("thresholds", {})

        return cls(
            token_limits=TokenLimits(
                per_session=tokens.get("per_session", 100000),
                per_iteration=tokens.get("per_iteration", 20000),
                per_tool_call=tokens.get("per_tool_call", 5000),
                absolute_max=tokens.get("absolute_max", 150000),
            ),
            cost_limits=CostLimits(
                per_session=costs.get("per_session_usd", 5.00),
                per_day=costs.get("per_day_usd", 20.00),
                per_week=costs.get("per_week_usd", 75.00),
            ),
            pricing=TokenPricing(
                input_per_million=pricing.get("input_per_million", 15.00),
                output_per_million=pricing.get("output_per_million", 75.00),
            ),
            warning_threshold=thresholds.get("warning", 0.80),
            pause_threshold=thresholds.get("pause", 0.95),
            hard_stop_threshold=thresholds.get("hard_stop", 1.00),
        )


@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation
        }


@dataclass
class SessionUsage:
    """Token usage for entire session."""
    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    iterations: int = 0
    operations: List[TokenUsage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "iterations": self.iterations,
            "started_at": self.started_at.isoformat(),
            "operations": [op.to_dict() for op in self.operations[-50:]]  # Keep last 50
        }


@dataclass
class DailyUsage:
    """Token usage for a single day."""
    date: date
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    sessions_count: int = 0
    sessions: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "sessions_count": self.sessions_count,
            "sessions": self.sessions[-20:]  # Keep last 20
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailyUsage":
        return cls(
            date=date.fromisoformat(data["date"]),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            sessions_count=data.get("sessions_count", 0),
            sessions=data.get("sessions", [])
        )


class GuardrailAction:
    """Result of guardrail check."""
    ALLOW = "allow"      # Proceed normally
    WARN = "warn"        # Proceed with warning
    PAUSE = "pause"      # Ask for approval
    STOP = "stop"        # Hard stop


class TokenGuardrails:
    """
    Manages token usage tracking and cost guardrails.

    Prevents runaway API costs by tracking usage at multiple levels
    (operation, iteration, session, daily) and enforcing limits.
    """

    def __init__(
        self,
        config: GuardrailConfig,
        session_id: str,
        stats_path: Optional[Path] = None
    ):
        """
        Initialize guardrails.

        Args:
            config: Guardrail configuration
            session_id: Current session ID
            stats_path: Path to daily stats file
        """
        self.config = config
        self.stats_path = stats_path

        # Current session usage
        self.session = SessionUsage(session_id=session_id)

        # Load daily stats
        self.daily = self._load_daily_stats()

        logger.info(
            f"Guardrails initialized: "
            f"session_limit={config.token_limits.per_session:,}, "
            f"daily_cost_limit=${config.cost_limits.per_day:.2f}"
        )

    def _load_daily_stats(self) -> DailyUsage:
        """Load or create daily usage stats."""
        today = date.today()

        if self.stats_path and self.stats_path.exists():
            try:
                with open(self.stats_path, "r") as f:
                    data = json.load(f)
                    if data.get("date") == today.isoformat():
                        return DailyUsage.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load daily stats: {e}")

        return DailyUsage(date=today)

    def _save_daily_stats(self) -> None:
        """Save daily usage stats."""
        if not self.stats_path:
            return

        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w") as f:
                json.dump(self.daily.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        operation: str = ""
    ) -> None:
        """
        Record token usage for an operation.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            operation: Description of the operation
        """
        # Create usage record
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation
        )

        # Calculate cost
        cost = self.config.pricing.calculate_cost(input_tokens, output_tokens)

        # Update session
        self.session.input_tokens += input_tokens
        self.session.output_tokens += output_tokens
        self.session.cost_usd += cost
        self.session.operations.append(usage)

        # Update daily
        self.daily.input_tokens += input_tokens
        self.daily.output_tokens += output_tokens
        self.daily.cost_usd += cost

        logger.debug(
            f"Token usage: +{usage.total_tokens:,} tokens, "
            f"+${cost:.4f} ({operation})"
        )

        self._save_daily_stats()

    def record_iteration(self) -> None:
        """Record that an iteration was completed."""
        self.session.iterations += 1

    def check_limits(self) -> tuple[str, str]:
        """
        Check all limits and return action to take.

        Returns:
            Tuple of (action, message)
            action: GuardrailAction constant
            message: Explanation of the limit status
        """
        checks = [
            self._check_session_tokens(),
            self._check_session_cost(),
            self._check_daily_cost(),
            self._check_absolute_limit(),
        ]

        # Return the most restrictive action
        for action, message in sorted(checks, key=lambda x: self._action_priority(x[0])):
            if action != GuardrailAction.ALLOW:
                return action, message

        return GuardrailAction.ALLOW, "All limits OK"

    def _action_priority(self, action: str) -> int:
        """Priority for sorting actions (lower = more restrictive)."""
        priorities = {
            GuardrailAction.STOP: 0,
            GuardrailAction.PAUSE: 1,
            GuardrailAction.WARN: 2,
            GuardrailAction.ALLOW: 3,
        }
        return priorities.get(action, 3)

    def _check_session_tokens(self) -> tuple[str, str]:
        """Check session token limit."""
        limit = self.config.token_limits.per_session
        used = self.session.total_tokens
        ratio = used / limit

        if ratio >= self.config.hard_stop_threshold:
            return (
                GuardrailAction.STOP,
                f"Session token limit exceeded: {used:,}/{limit:,} ({ratio:.0%})"
            )
        elif ratio >= self.config.pause_threshold:
            return (
                GuardrailAction.PAUSE,
                f"Session token limit nearly reached: {used:,}/{limit:,} ({ratio:.0%})"
            )
        elif ratio >= self.config.warning_threshold:
            return (
                GuardrailAction.WARN,
                f"Session token limit warning: {used:,}/{limit:,} ({ratio:.0%})"
            )

        return GuardrailAction.ALLOW, ""

    def _check_session_cost(self) -> tuple[str, str]:
        """Check session cost limit."""
        limit = self.config.cost_limits.per_session
        used = self.session.cost_usd
        ratio = used / limit

        if ratio >= self.config.hard_stop_threshold:
            return (
                GuardrailAction.STOP,
                f"Session cost limit exceeded: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )
        elif ratio >= self.config.pause_threshold:
            return (
                GuardrailAction.PAUSE,
                f"Session cost limit nearly reached: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )
        elif ratio >= self.config.warning_threshold:
            return (
                GuardrailAction.WARN,
                f"Session cost limit warning: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )

        return GuardrailAction.ALLOW, ""

    def _check_daily_cost(self) -> tuple[str, str]:
        """Check daily cost limit."""
        limit = self.config.cost_limits.per_day
        used = self.daily.cost_usd
        ratio = used / limit

        if ratio >= self.config.hard_stop_threshold:
            return (
                GuardrailAction.STOP,
                f"Daily cost limit exceeded: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )
        elif ratio >= self.config.pause_threshold:
            return (
                GuardrailAction.PAUSE,
                f"Daily cost limit nearly reached: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )
        elif ratio >= self.config.warning_threshold:
            return (
                GuardrailAction.WARN,
                f"Daily cost limit warning: ${used:.2f}/${limit:.2f} ({ratio:.0%})"
            )

        return GuardrailAction.ALLOW, ""

    def _check_absolute_limit(self) -> tuple[str, str]:
        """Check absolute token limit."""
        limit = self.config.token_limits.absolute_max
        used = self.session.total_tokens

        if used >= limit:
            return (
                GuardrailAction.STOP,
                f"Absolute token limit exceeded: {used:,}/{limit:,}"
            )

        return GuardrailAction.ALLOW, ""

    def can_proceed(self, estimated_tokens: int = 0) -> tuple[bool, str]:
        """
        Check if we can proceed with an operation.

        Args:
            estimated_tokens: Estimated tokens for the operation

        Returns:
            Tuple of (can_proceed, message)
        """
        action, message = self.check_limits()

        if action == GuardrailAction.STOP:
            return False, message

        # Check if estimated tokens would exceed limit
        if estimated_tokens > 0:
            projected = self.session.total_tokens + estimated_tokens
            if projected > self.config.token_limits.per_session:
                return False, f"Estimated tokens ({estimated_tokens:,}) would exceed session limit"

        return True, message

    def get_status(self) -> Dict[str, Any]:
        """Get full guardrail status."""
        action, message = self.check_limits()

        return {
            "action": action,
            "message": message,
            "session": {
                "id": self.session.session_id,
                "tokens_used": self.session.total_tokens,
                "tokens_limit": self.config.token_limits.per_session,
                "tokens_percent": (self.session.total_tokens / self.config.token_limits.per_session) * 100,
                "cost_usd": self.session.cost_usd,
                "cost_limit": self.config.cost_limits.per_session,
                "cost_percent": (self.session.cost_usd / self.config.cost_limits.per_session) * 100,
                "iterations": self.session.iterations,
            },
            "daily": {
                "date": self.daily.date.isoformat(),
                "tokens_used": self.daily.total_tokens,
                "cost_usd": self.daily.cost_usd,
                "cost_limit": self.config.cost_limits.per_day,
                "cost_percent": (self.daily.cost_usd / self.config.cost_limits.per_day) * 100,
                "sessions_count": self.daily.sessions_count,
            },
        }

    def format_status_line(self) -> str:
        """Format a compact status line for logging."""
        status = self.get_status()
        session = status["session"]
        daily = status["daily"]

        return (
            f"Tokens: {session['tokens_used']:,}/{session['tokens_limit']:,} ({session['tokens_percent']:.0f}%) | "
            f"Session: ${session['cost_usd']:.2f}/${session['cost_limit']:.2f} | "
            f"Daily: ${daily['cost_usd']:.2f}/${daily['cost_limit']:.2f}"
        )

    def finalize_session(self) -> None:
        """Mark session as complete in daily stats."""
        if self.session.session_id not in self.daily.sessions:
            self.daily.sessions.append(self.session.session_id)
            self.daily.sessions_count += 1
            self._save_daily_stats()
