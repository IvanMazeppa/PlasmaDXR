#!/usr/bin/env python3
"""
Workflow Tracer for Blender VFX Orchestrator

Provides detailed tracing of workflow state transitions for debugging:
- State transition logging with timestamps
- Tool call traces with parameters and results
- Circuit breaker trigger events
- Knowledge base consultations
- Quality evaluation decisions

This is especially useful for debugging SKILL.md's loop control behavior.
The trace format matches what SKILL.md describes for iteration tracking.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("blender-orchestrator.workflow_tracer")


class TraceEventType(Enum):
    """Types of events that can be traced."""
    STATE_TRANSITION = "state_transition"
    TOOL_CALL = "tool_call"
    CIRCUIT_BREAKER = "circuit_breaker"
    KNOWLEDGE_BASE = "knowledge_base"
    QUALITY_DECISION = "quality_decision"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    ERROR = "error"
    HUMAN_INTERVENTION = "human_intervention"


@dataclass
class TraceEvent:
    """A single trace event in the workflow."""
    timestamp: str
    event_type: TraceEventType
    state: str
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "state": self.state,
            "data": self.data,
            "latency_ms": self.latency_ms,
        }


@dataclass
class IterationSummary:
    """Summary of a single iteration (matches SKILL.md tracking)."""
    iteration_number: int
    start_time: str
    end_time: Optional[str] = None
    vfx_score: Optional[float] = None
    issues_found: List[str] = field(default_factory=list)
    parameters_changed: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker_status: str = "ok"  # ok, warning, triggered
    knowledge_base_consulted: bool = False
    blender_success: bool = True
    outcome: str = "pending"  # pending, improved, regressed, stagnant

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "vfx_score": self.vfx_score,
            "issues_found": self.issues_found,
            "parameters_changed": self.parameters_changed,
            "circuit_breaker_status": self.circuit_breaker_status,
            "knowledge_base_consulted": self.knowledge_base_consulted,
            "blender_success": self.blender_success,
            "outcome": self.outcome,
        }


class WorkflowTracer:
    """
    Traces workflow execution for debugging and analysis.

    Matches the iteration tracking described in SKILL.md:
    - current_iteration, max_iterations, best_score, best_iteration
    - iterations_without_improvement, consecutive_blender_failures
    - Circuit breaker checks before each iteration

    Usage:
        tracer = WorkflowTracer("sun_v1_20260103")

        # Trace iteration start (matches SKILL.md requirement)
        tracer.trace_iteration_start(1)

        # Trace state transitions
        tracer.trace_state_transition("GENERATE_SCRIPT", "EXECUTE_BLENDER")

        # Trace tool calls
        tracer.trace_tool_call("generate_script", params, result, latency_ms)

        # Trace circuit breaker checks (critical for debugging SKILL.md behavior)
        tracer.trace_circuit_breaker_check("MAX_ITERATIONS", triggered=False)

        # Save trace for analysis
        tracer.save()
    """

    def __init__(
        self,
        session_id: str,
        trace_dir: Optional[Path] = None,
        max_events: int = 10000,
    ):
        """
        Initialize workflow tracer.

        Args:
            session_id: Unique session identifier
            trace_dir: Directory for trace files
            max_events: Maximum events to keep in memory
        """
        self.session_id = session_id
        self.trace_dir = trace_dir or Path("build/orchestrator_state/traces")
        self.max_events = max_events

        # Trace state
        self.events: List[TraceEvent] = []
        self.iterations: List[IterationSummary] = []
        self.current_iteration: Optional[IterationSummary] = None

        # Tracking variables (match SKILL.md)
        self.iteration_count = 0
        self.max_iterations = 10
        self.best_score = 0.0
        self.best_iteration = 0
        self.iterations_without_improvement = 0
        self.consecutive_blender_failures = 0
        self.consecutive_low_scores = 0

        # Current state
        self._current_state = "SESSION_START"
        self._start_time = datetime.now()

        logger.info(f"WorkflowTracer initialized for session: {session_id}")

    def _add_event(self, event: TraceEvent) -> None:
        """Add event to trace, respecting max limit."""
        self.events.append(event)
        if len(self.events) > self.max_events:
            # Remove oldest 10% when limit reached
            self.events = self.events[self.max_events // 10:]

    def trace_iteration_start(self, iteration_number: int) -> None:
        """
        Trace the start of an iteration.

        SKILL.md requires: "At the START of each iteration, you MUST:
        1. Increment iteration counter
        2. Log: === ITERATION {current_iteration} of 10 ===
        3. Check circuit breakers BEFORE proceeding"
        """
        timestamp = datetime.now().isoformat()
        self.iteration_count = iteration_number

        # Create iteration summary
        self.current_iteration = IterationSummary(
            iteration_number=iteration_number,
            start_time=timestamp,
        )
        self.iterations.append(self.current_iteration)

        # Add trace event
        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.ITERATION_START,
            state=self._current_state,
            data={
                "iteration": iteration_number,
                "max_iterations": self.max_iterations,
                "best_score": self.best_score,
                "best_iteration": self.best_iteration,
                "iterations_without_improvement": self.iterations_without_improvement,
                "consecutive_blender_failures": self.consecutive_blender_failures,
            }
        )
        self._add_event(event)

        logger.info(f"=== ITERATION {iteration_number} of {self.max_iterations} ===")

    def trace_iteration_end(
        self,
        vfx_score: float,
        issues: List[str],
        blender_success: bool = True,
    ) -> None:
        """
        Trace the end of an iteration with results.

        Updates tracking variables per SKILL.md.
        """
        timestamp = datetime.now().isoformat()

        # Update tracking variables
        if vfx_score > self.best_score:
            self.best_score = vfx_score
            self.best_iteration = self.iteration_count
            self.iterations_without_improvement = 0
            outcome = "improved"
        else:
            self.iterations_without_improvement += 1
            outcome = "regressed" if vfx_score < self.best_score else "stagnant"

        if blender_success:
            self.consecutive_blender_failures = 0
        else:
            self.consecutive_blender_failures += 1

        if vfx_score < 40:
            self.consecutive_low_scores += 1
        else:
            self.consecutive_low_scores = 0

        # Update current iteration
        if self.current_iteration:
            self.current_iteration.end_time = timestamp
            self.current_iteration.vfx_score = vfx_score
            self.current_iteration.issues_found = issues
            self.current_iteration.blender_success = blender_success
            self.current_iteration.outcome = outcome

        # Add trace event
        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.ITERATION_END,
            state=self._current_state,
            data={
                "iteration": self.iteration_count,
                "vfx_score": vfx_score,
                "issues_count": len(issues),
                "outcome": outcome,
                "best_score": self.best_score,
                "iterations_without_improvement": self.iterations_without_improvement,
            }
        )
        self._add_event(event)

        logger.info(
            f"Iteration {self.iteration_count} ended: "
            f"score={vfx_score:.1f}, outcome={outcome}"
        )

    def trace_state_transition(
        self,
        from_state: str,
        to_state: str,
        reason: str = "",
    ) -> None:
        """Trace a state machine transition."""
        timestamp = datetime.now().isoformat()

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.STATE_TRANSITION,
            state=from_state,
            data={
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            }
        )
        self._add_event(event)
        self._current_state = to_state

        logger.debug(f"State: {from_state} -> {to_state} ({reason})")

    def trace_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Trace an MCP tool call (matches plan trace format)."""
        timestamp = datetime.now().isoformat()

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.TOOL_CALL,
            state=self._current_state,
            data={
                "tool_called": tool_name,
                "tool_params": self._sanitize_params(params),
                "tool_result": {"status": "success" if success else "failed"},
                "success": success,
            },
            latency_ms=latency_ms,
        )
        self._add_event(event)

    def trace_circuit_breaker_check(
        self,
        breaker_name: str,
        triggered: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trace a circuit breaker check.

        SKILL.md defines 5 circuit breakers:
        - MAX_ITERATIONS: current_iteration >= 10
        - MAX_WALL_TIME: elapsed > 30 minutes
        - NO_IMPROVEMENT: iterations_without_improvement >= 3
        - QUALITY_FLOOR: consecutive_low_scores >= 2 (score < 40)
        - BLENDER_FAILURES: consecutive_blender_failures >= 2
        """
        timestamp = datetime.now().isoformat()

        status = "triggered" if triggered else "ok"
        if self.current_iteration:
            self.current_iteration.circuit_breaker_status = status

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.CIRCUIT_BREAKER,
            state=self._current_state,
            data={
                "breaker_name": breaker_name,
                "triggered": triggered,
                "details": details or {},
                "current_values": {
                    "iteration": self.iteration_count,
                    "max_iterations": self.max_iterations,
                    "best_score": self.best_score,
                    "iterations_without_improvement": self.iterations_without_improvement,
                    "consecutive_blender_failures": self.consecutive_blender_failures,
                    "consecutive_low_scores": self.consecutive_low_scores,
                }
            }
        )
        self._add_event(event)

        if triggered:
            logger.warning(f"CIRCUIT BREAKER: {breaker_name} triggered")
        else:
            logger.debug(f"Circuit breaker check: {breaker_name} = OK")

    def trace_knowledge_base_consultation(
        self,
        query_type: str,  # "warnings" or "suggestions"
        parameter: str,
        result: Optional[Dict[str, Any]],
        applied: bool = False,
    ) -> None:
        """Trace knowledge base consultation (mandatory per SKILL.md)."""
        timestamp = datetime.now().isoformat()

        if self.current_iteration:
            self.current_iteration.knowledge_base_consulted = True

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.KNOWLEDGE_BASE,
            state=self._current_state,
            data={
                "query_type": query_type,
                "parameter": parameter,
                "has_result": result is not None,
                "applied": applied,
            }
        )
        self._add_event(event)

        logger.debug(f"Knowledge base consulted: {query_type} for {parameter}")

    def trace_quality_decision(
        self,
        vfx_score: float,
        decision: str,  # "accept", "iterate", "reject"
        issues: List[str],
        lpips_score: Optional[float] = None,
        clip_score: Optional[float] = None,
    ) -> None:
        """
        Trace a quality evaluation decision.

        Per SKILL.md decision tree:
        - VFX >= 60: ACCEPT
        - VFX 40-59: ITERATE (check LPIPS/CLIP)
        - VFX < 40: REJECT immediately
        """
        timestamp = datetime.now().isoformat()

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.QUALITY_DECISION,
            state=self._current_state,
            data={
                "vfx_score": vfx_score,
                "decision": decision,
                "issues_count": len(issues),
                "lpips_score": lpips_score,
                "clip_score": clip_score,
            }
        )
        self._add_event(event)

        logger.info(f"Quality decision: {decision} (VFX={vfx_score:.1f})")

    def trace_error(
        self,
        error_type: str,
        error_message: str,
        recoverable: bool = True,
    ) -> None:
        """Trace an error event."""
        timestamp = datetime.now().isoformat()

        event = TraceEvent(
            timestamp=timestamp,
            event_type=TraceEventType.ERROR,
            state=self._current_state,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "recoverable": recoverable,
            }
        )
        self._add_event(event)

        if recoverable:
            logger.warning(f"Recoverable error: {error_type} - {error_message}")
        else:
            logger.error(f"Fatal error: {error_type} - {error_message}")

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize params for logging (truncate large values)."""
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            elif isinstance(value, dict) and len(str(value)) > 500:
                sanitized[key] = {"_truncated": True, "_keys": list(value.keys())}
            else:
                sanitized[key] = value
        return sanitized

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of the trace for analysis."""
        elapsed = (datetime.now() - self._start_time).total_seconds()

        # Count events by type
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Circuit breaker summary
        cb_events = [e for e in self.events if e.event_type == TraceEventType.CIRCUIT_BREAKER]
        cb_triggered = [e for e in cb_events if e.data.get("triggered")]

        return {
            "session_id": self.session_id,
            "elapsed_seconds": elapsed,
            "iterations_completed": len(self.iterations),
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "circuit_breakers_triggered": len(cb_triggered),
            "final_state": self._current_state,
            "tracking_variables": {
                "iterations_without_improvement": self.iterations_without_improvement,
                "consecutive_blender_failures": self.consecutive_blender_failures,
                "consecutive_low_scores": self.consecutive_low_scores,
            }
        }

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get history of all iterations."""
        return [it.to_dict() for it in self.iterations]

    def get_circuit_breaker_events(self) -> List[Dict[str, Any]]:
        """Get all circuit breaker events for analysis."""
        cb_events = [
            e for e in self.events
            if e.event_type == TraceEventType.CIRCUIT_BREAKER
        ]
        return [e.to_dict() for e in cb_events]

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save trace to JSON file.

        Args:
            filepath: Optional specific path

        Returns:
            Path where trace was saved
        """
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.trace_dir / f"trace_{self.session_id}_{timestamp}.json"

        trace_data = {
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_trace_summary(),
            "iterations": self.get_iteration_history(),
            "traces": [e.to_dict() for e in self.events],
        }

        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Workflow trace saved to {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        summary = self.get_trace_summary()

        print("\n" + "=" * 60)
        print(f"WORKFLOW TRACE SUMMARY: {self.session_id}")
        print("=" * 60)
        print(f"Duration: {summary['elapsed_seconds']:.1f}s")
        print(f"Iterations: {summary['iterations_completed']}")
        print(f"Best Score: {summary['best_score']:.1f} (iteration {summary['best_iteration']})")
        print(f"Final State: {summary['final_state']}")
        print(f"Total Events: {summary['total_events']}")

        print("\nEvent Breakdown:")
        for event_type, count in summary['event_counts'].items():
            print(f"  {event_type}: {count}")

        if summary['circuit_breakers_triggered'] > 0:
            print(f"\n⚠️  Circuit Breakers Triggered: {summary['circuit_breakers_triggered']}")

        print("\nTracking Variables:")
        for var, value in summary['tracking_variables'].items():
            print(f"  {var}: {value}")

        print("=" * 60 + "\n")


# Singleton instance for global access
_tracer_instance: Optional[WorkflowTracer] = None


def get_tracer(session_id: Optional[str] = None) -> WorkflowTracer:
    """Get or create the global tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _tracer_instance = WorkflowTracer(session_id)
    return _tracer_instance


def reset_tracer() -> None:
    """Reset the global tracer instance."""
    global _tracer_instance
    _tracer_instance = None


if __name__ == "__main__":
    # Quick test
    tracer = WorkflowTracer("test_session")

    # Simulate a workflow
    tracer.trace_iteration_start(1)
    tracer.trace_state_transition("SESSION_START", "GENERATE_SCRIPT", "initial")
    tracer.trace_tool_call("generate_script", {"effect_type": "pyro"}, {"path": "/test"}, 1500)
    tracer.trace_circuit_breaker_check("MAX_ITERATIONS", triggered=False)
    tracer.trace_knowledge_base_consultation("warnings", "turbulence", None)
    tracer.trace_quality_decision(55.0, "iterate", ["too dark", "needs structure"])
    tracer.trace_iteration_end(55.0, ["too dark", "needs structure"])

    tracer.trace_iteration_start(2)
    tracer.trace_circuit_breaker_check("NO_IMPROVEMENT", triggered=False)
    tracer.trace_iteration_end(62.0, [])

    tracer.print_summary()
    print(f"\nSaved to: {tracer.save()}")
