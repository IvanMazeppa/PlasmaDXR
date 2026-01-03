#!/usr/bin/env python3
"""
Formal State Machine for Blender VFX Orchestrator

Defines the workflow states and valid transitions for the autonomous
asset generation loop. This ensures the orchestrator follows a
predictable, debuggable path through the workflow.

States match the stages described in SKILL.md.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("blender-orchestrator.state_machine")


class WorkflowState(Enum):
    """
    Formal workflow states for the VFX asset generation pipeline.

    Matches the stages in SKILL.md:
    1. Initialize Session (SESSION_START)
    2. Generate Script (GENERATE_SCRIPT)
    3. Execute Blender (EXECUTE_BLENDER)
    4. Evaluate Quality (EVALUATE_QUALITY)
    5. Decide Next Action (DECIDE_NEXT_ACTION)
    6. Record Learning (RECORD_LEARNING)
    """
    # Primary workflow states
    SESSION_START = "session_start"
    GENERATE_SCRIPT = "generate_script"
    VALIDATE_SCRIPT = "validate_script"  # NEW: Pre-execution validation
    EXECUTE_BLENDER = "execute_blender"
    EVALUATE_QUALITY = "evaluate_quality"
    DECIDE_NEXT_ACTION = "decide_next_action"
    RECORD_LEARNING = "record_learning"
    CHECK_CONVERGENCE = "check_convergence"  # NEW: Circuit breaker check
    SESSION_END = "session_end"

    # Error/Recovery states
    ERROR_RECOVERY = "error_recovery"
    AWAITING_APPROVAL = "awaiting_approval"

    # Special states
    PAUSED = "paused"
    RESUMING = "resuming"


# Valid state transitions
TRANSITIONS: Dict[WorkflowState, List[WorkflowState]] = {
    # Start can only go to script generation
    WorkflowState.SESSION_START: [
        WorkflowState.GENERATE_SCRIPT,
    ],

    # After generating, validate before executing
    WorkflowState.GENERATE_SCRIPT: [
        WorkflowState.VALIDATE_SCRIPT,
        WorkflowState.ERROR_RECOVERY,
    ],

    # Validation can pass (execute) or fail (regenerate with context)
    WorkflowState.VALIDATE_SCRIPT: [
        WorkflowState.EXECUTE_BLENDER,
        WorkflowState.GENERATE_SCRIPT,  # Loop back if invalid
        WorkflowState.ERROR_RECOVERY,
    ],

    # Execution leads to quality evaluation
    WorkflowState.EXECUTE_BLENDER: [
        WorkflowState.EVALUATE_QUALITY,
        WorkflowState.ERROR_RECOVERY,
    ],

    # After evaluation, decide what to do
    WorkflowState.EVALUATE_QUALITY: [
        WorkflowState.DECIDE_NEXT_ACTION,
        WorkflowState.ERROR_RECOVERY,
    ],

    # Decision point: iterate, end, or await approval
    WorkflowState.DECIDE_NEXT_ACTION: [
        WorkflowState.RECORD_LEARNING,
        WorkflowState.GENERATE_SCRIPT,  # Direct retry without learning
        WorkflowState.SESSION_END,  # Quality passed
        WorkflowState.AWAITING_APPROVAL,  # Need human input
    ],

    # After recording learning, check if we should continue
    WorkflowState.RECORD_LEARNING: [
        WorkflowState.CHECK_CONVERGENCE,
    ],

    # Convergence check (circuit breakers)
    WorkflowState.CHECK_CONVERGENCE: [
        WorkflowState.GENERATE_SCRIPT,  # Continue iterating
        WorkflowState.SESSION_END,  # Circuit breaker triggered
        WorkflowState.PAUSED,  # Pause for human review
    ],

    # Error recovery can retry or end
    WorkflowState.ERROR_RECOVERY: [
        WorkflowState.GENERATE_SCRIPT,  # Retry
        WorkflowState.SESSION_END,  # Give up
    ],

    # Human approval can resume or end
    WorkflowState.AWAITING_APPROVAL: [
        WorkflowState.GENERATE_SCRIPT,  # Approved, continue
        WorkflowState.SESSION_END,  # Rejected, stop
    ],

    # Paused sessions can resume or end
    WorkflowState.PAUSED: [
        WorkflowState.RESUMING,
        WorkflowState.SESSION_END,
    ],

    # Resuming rejoins the main flow
    WorkflowState.RESUMING: [
        WorkflowState.GENERATE_SCRIPT,
        WorkflowState.EXECUTE_BLENDER,
        WorkflowState.CHECK_CONVERGENCE,
    ],

    # End state has no transitions
    WorkflowState.SESSION_END: [],
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: WorkflowState
    to_state: WorkflowState
    timestamp: str
    reason: str
    metadata: Dict[str, Any]


class WorkflowStateMachine:
    """
    Formal state machine for the VFX workflow.

    Enforces valid transitions and provides debugging/analysis capabilities.

    Usage:
        sm = WorkflowStateMachine()

        # Transition with validation
        success = sm.transition(WorkflowState.GENERATE_SCRIPT, "Starting workflow")

        # Check what's possible
        if sm.can_transition(WorkflowState.EXECUTE_BLENDER):
            sm.transition(WorkflowState.EXECUTE_BLENDER, "Script ready")

        # Get current state
        print(f"Current state: {sm.current_state}")
    """

    def __init__(self, initial_state: WorkflowState = WorkflowState.SESSION_START):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state (default: SESSION_START)
        """
        self._current_state = initial_state
        self._history: List[StateTransition] = []
        self._transition_callbacks: Dict[WorkflowState, List[Callable]] = {}

        # Record initial state
        self._history.append(StateTransition(
            from_state=initial_state,
            to_state=initial_state,
            timestamp=datetime.now().isoformat(),
            reason="initial_state",
            metadata={}
        ))

        logger.info(f"State machine initialized in state: {initial_state.value}")

    @property
    def current_state(self) -> WorkflowState:
        """Get current workflow state."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """Check if in a terminal state (no more transitions possible)."""
        return len(TRANSITIONS.get(self._current_state, [])) == 0

    @property
    def is_error_state(self) -> bool:
        """Check if in an error state."""
        return self._current_state == WorkflowState.ERROR_RECOVERY

    @property
    def is_awaiting_human(self) -> bool:
        """Check if awaiting human input."""
        return self._current_state in [
            WorkflowState.AWAITING_APPROVAL,
            WorkflowState.PAUSED,
        ]

    def can_transition(self, to_state: WorkflowState) -> bool:
        """
        Check if a transition to the target state is valid.

        Args:
            to_state: Target state

        Returns:
            True if transition is valid
        """
        valid_transitions = TRANSITIONS.get(self._current_state, [])
        return to_state in valid_transitions

    def get_valid_transitions(self) -> List[WorkflowState]:
        """Get list of valid transitions from current state."""
        return TRANSITIONS.get(self._current_state, [])

    def transition(
        self,
        to_state: WorkflowState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Transition to a new state.

        Args:
            to_state: Target state
            reason: Reason for transition (for debugging)
            metadata: Additional context
            force: If True, skip validation (use with caution)

        Returns:
            True if transition succeeded

        Raises:
            ValueError: If transition is invalid and force=False
        """
        if not force and not self.can_transition(to_state):
            valid = [s.value for s in self.get_valid_transitions()]
            error_msg = (
                f"Invalid transition: {self._current_state.value} -> {to_state.value}. "
                f"Valid transitions: {valid}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Record transition
        transition = StateTransition(
            from_state=self._current_state,
            to_state=to_state,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            metadata=metadata or {}
        )
        self._history.append(transition)

        # Execute callbacks
        callbacks = self._transition_callbacks.get(to_state, [])
        for callback in callbacks:
            try:
                callback(transition)
            except Exception as e:
                logger.warning(f"Transition callback failed: {e}")

        # Update state
        old_state = self._current_state
        self._current_state = to_state

        logger.info(f"State: {old_state.value} -> {to_state.value} ({reason})")
        return True

    def on_enter(self, state: WorkflowState, callback: Callable[[StateTransition], None]) -> None:
        """
        Register callback to run when entering a state.

        Args:
            state: State to watch
            callback: Function to call with StateTransition
        """
        if state not in self._transition_callbacks:
            self._transition_callbacks[state] = []
        self._transition_callbacks[state].append(callback)

    def reset(self, to_state: WorkflowState = WorkflowState.SESSION_START) -> None:
        """
        Reset state machine to initial state.

        Args:
            to_state: State to reset to
        """
        self._current_state = to_state
        self._history = [StateTransition(
            from_state=to_state,
            to_state=to_state,
            timestamp=datetime.now().isoformat(),
            reason="reset",
            metadata={}
        )]
        logger.info(f"State machine reset to: {to_state.value}")

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transition history."""
        history = self._history[-limit:]
        return [
            {
                "from": t.from_state.value,
                "to": t.to_state.value,
                "timestamp": t.timestamp,
                "reason": t.reason,
                "metadata": t.metadata,
            }
            for t in history
        ]

    def get_state_counts(self) -> Dict[str, int]:
        """Get count of times each state was entered."""
        counts: Dict[str, int] = {}
        for t in self._history:
            state = t.to_state.value
            counts[state] = counts.get(state, 0) + 1
        return counts

    def get_loop_count(self) -> int:
        """Get number of iteration loops (GENERATE_SCRIPT entries)."""
        return sum(
            1 for t in self._history
            if t.to_state == WorkflowState.GENERATE_SCRIPT
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get state machine summary."""
        return {
            "current_state": self._current_state.value,
            "is_terminal": self.is_terminal,
            "is_error_state": self.is_error_state,
            "is_awaiting_human": self.is_awaiting_human,
            "valid_transitions": [s.value for s in self.get_valid_transitions()],
            "transition_count": len(self._history),
            "loop_count": self.get_loop_count(),
            "state_counts": self.get_state_counts(),
        }

    def print_graph(self) -> None:
        """Print ASCII representation of state machine."""
        print("\n" + "=" * 60)
        print("WORKFLOW STATE MACHINE")
        print("=" * 60)

        for state, transitions in TRANSITIONS.items():
            if transitions:
                targets = ", ".join(t.value for t in transitions)
                print(f"  {state.value}")
                print(f"    -> {targets}")
            else:
                print(f"  {state.value} [TERMINAL]")

        print("-" * 60)
        print(f"Current: {self._current_state.value}")
        if self.is_terminal:
            print("  (TERMINAL STATE)")
        else:
            valid = [s.value for s in self.get_valid_transitions()]
            print(f"  Valid next: {valid}")
        print("=" * 60 + "\n")


def create_workflow_graph_mermaid() -> str:
    """Generate Mermaid diagram of the state machine."""
    lines = ["stateDiagram-v2"]

    for state, transitions in TRANSITIONS.items():
        for target in transitions:
            lines.append(f"    {state.value} --> {target.value}")

    # Mark terminal state
    lines.append(f"    {WorkflowState.SESSION_END.value} --> [*]")

    return "\n".join(lines)


# Singleton instance
_state_machine_instance: Optional[WorkflowStateMachine] = None


def get_state_machine() -> WorkflowStateMachine:
    """Get or create the global state machine instance."""
    global _state_machine_instance
    if _state_machine_instance is None:
        _state_machine_instance = WorkflowStateMachine()
    return _state_machine_instance


def reset_state_machine() -> None:
    """Reset the global state machine instance."""
    global _state_machine_instance
    _state_machine_instance = None


if __name__ == "__main__":
    # Test the state machine
    sm = WorkflowStateMachine()

    # Print the graph
    sm.print_graph()

    # Simulate a workflow
    print("Simulating workflow...\n")

    sm.transition(WorkflowState.GENERATE_SCRIPT, "Start asset generation")
    sm.transition(WorkflowState.VALIDATE_SCRIPT, "Script generated")
    sm.transition(WorkflowState.EXECUTE_BLENDER, "Script valid")
    sm.transition(WorkflowState.EVALUATE_QUALITY, "Blender completed")
    sm.transition(WorkflowState.DECIDE_NEXT_ACTION, "Quality evaluated")
    sm.transition(WorkflowState.RECORD_LEARNING, "Needs improvement")
    sm.transition(WorkflowState.CHECK_CONVERGENCE, "Learning recorded")

    # Try invalid transition (should fail)
    try:
        sm.transition(WorkflowState.SESSION_START, "Invalid!")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Continue valid path
    sm.transition(WorkflowState.GENERATE_SCRIPT, "Continue iteration")

    print("\nSummary:")
    for key, value in sm.get_summary().items():
        print(f"  {key}: {value}")

    print("\nHistory:")
    for entry in sm.get_history(limit=5):
        print(f"  {entry['from']} -> {entry['to']}: {entry['reason']}")

    print("\nMermaid Diagram:")
    print(create_workflow_graph_mermaid())
