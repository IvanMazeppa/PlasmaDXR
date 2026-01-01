"""
Workflow State Machine

Manages the asset generation workflow stages and transitions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("blender-orchestrator.workflow")


class WorkflowStage(Enum):
    """Stages in the asset generation workflow."""
    SESSION_START = "session_start"
    GENERATE_SCRIPT = "generate_script"
    EXECUTE_BLENDER = "execute_blender"
    EVALUATE_QUALITY = "evaluate_quality"
    DECIDE_NEXT_ACTION = "decide_next_action"
    RECORD_LEARNING = "record_learning"
    SESSION_END = "session_end"
    ERROR_RECOVERY = "error_recovery"
    AWAITING_APPROVAL = "awaiting_approval"


class WorkflowOutcome(Enum):
    """Possible outcomes of a workflow stage."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    SKIP = "skip"
    AWAITING_INPUT = "awaiting_input"


@dataclass
class StageResult:
    """Result of executing a workflow stage."""
    stage: WorkflowStage
    outcome: WorkflowOutcome
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used
        }


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    max_iterations: int = 10
    max_retries: int = 3
    max_technique_changes: int = 2

    # Timeouts in seconds
    blender_timeout: int = 900
    evaluation_timeout: int = 120
    mcp_timeout: int = 60

    # Feature flags
    validate_api: bool = True
    use_knowledge_base: bool = True
    record_learnings: bool = True

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "WorkflowConfig":
        """Create config from YAML dict."""
        workflow = config.get("workflow", {})
        timeouts = workflow.get("timeouts", {})

        return cls(
            max_iterations=workflow.get("max_iterations", 10),
            max_retries=workflow.get("max_retries", 3),
            max_technique_changes=workflow.get("max_technique_changes", 2),
            blender_timeout=timeouts.get("blender_execution", 900),
            evaluation_timeout=timeouts.get("evaluation", 120),
            mcp_timeout=timeouts.get("mcp_tool_call", 60),
            validate_api=workflow.get("validate_api", True),
            use_knowledge_base=workflow.get("use_knowledge_base", True),
            record_learnings=workflow.get("record_learnings", True),
        )


@dataclass
class QualityConfig:
    """Quality threshold configuration."""
    vfx_pass_threshold: float = 70.0
    vfx_good_threshold: float = 80.0
    vfx_excellent_threshold: float = 90.0
    plateau_threshold: float = 5.0
    plateau_iterations: int = 3
    temporal_threshold: float = 0.70

    # Effect-specific overrides
    effect_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "QualityConfig":
        """Create config from YAML dict."""
        quality = config.get("quality", {})
        plateau = quality.get("plateau_detection", {})

        return cls(
            vfx_pass_threshold=quality.get("vfx_pass_threshold", 70.0),
            vfx_good_threshold=quality.get("vfx_good_threshold", 80.0),
            vfx_excellent_threshold=quality.get("vfx_excellent_threshold", 90.0),
            plateau_threshold=plateau.get("threshold", 5.0),
            plateau_iterations=plateau.get("iterations", 3),
            temporal_threshold=quality.get("temporal_threshold", 0.70),
            effect_overrides=quality.get("effect_overrides", {}),
        )

    def get_pass_threshold(self, effect_type: str) -> float:
        """Get pass threshold for an effect type."""
        if effect_type in self.effect_overrides:
            return self.effect_overrides[effect_type].get(
                "vfx_pass_threshold", self.vfx_pass_threshold
            )
        return self.vfx_pass_threshold


# State transition table
TRANSITIONS = {
    WorkflowStage.SESSION_START: {
        WorkflowOutcome.SUCCESS: WorkflowStage.GENERATE_SCRIPT,
        WorkflowOutcome.FAILURE: WorkflowStage.SESSION_END,
    },
    WorkflowStage.GENERATE_SCRIPT: {
        WorkflowOutcome.SUCCESS: WorkflowStage.EXECUTE_BLENDER,
        WorkflowOutcome.FAILURE: WorkflowStage.ERROR_RECOVERY,
        WorkflowOutcome.RETRY: WorkflowStage.GENERATE_SCRIPT,
    },
    WorkflowStage.EXECUTE_BLENDER: {
        WorkflowOutcome.SUCCESS: WorkflowStage.EVALUATE_QUALITY,
        WorkflowOutcome.FAILURE: WorkflowStage.ERROR_RECOVERY,
        WorkflowOutcome.RETRY: WorkflowStage.EXECUTE_BLENDER,
    },
    WorkflowStage.EVALUATE_QUALITY: {
        WorkflowOutcome.SUCCESS: WorkflowStage.DECIDE_NEXT_ACTION,
        WorkflowOutcome.FAILURE: WorkflowStage.ERROR_RECOVERY,
        WorkflowOutcome.SKIP: WorkflowStage.DECIDE_NEXT_ACTION,
    },
    WorkflowStage.DECIDE_NEXT_ACTION: {
        WorkflowOutcome.SUCCESS: WorkflowStage.RECORD_LEARNING,
        WorkflowOutcome.AWAITING_INPUT: WorkflowStage.AWAITING_APPROVAL,
    },
    WorkflowStage.RECORD_LEARNING: {
        WorkflowOutcome.SUCCESS: WorkflowStage.GENERATE_SCRIPT,  # Continue iterating
        WorkflowOutcome.FAILURE: WorkflowStage.GENERATE_SCRIPT,  # Continue anyway
    },
    WorkflowStage.ERROR_RECOVERY: {
        WorkflowOutcome.SUCCESS: WorkflowStage.GENERATE_SCRIPT,  # Retry from script
        WorkflowOutcome.FAILURE: WorkflowStage.SESSION_END,
        WorkflowOutcome.AWAITING_INPUT: WorkflowStage.AWAITING_APPROVAL,
    },
    WorkflowStage.AWAITING_APPROVAL: {
        WorkflowOutcome.SUCCESS: None,  # Dynamic based on pending decision
        WorkflowOutcome.FAILURE: WorkflowStage.SESSION_END,
    },
    WorkflowStage.SESSION_END: {
        # Terminal state
    },
}


class WorkflowStateMachine:
    """
    Manages workflow state transitions for asset generation.

    Tracks current stage, iteration count, retry counts, and
    provides methods to advance through the workflow.
    """

    def __init__(
        self,
        workflow_config: WorkflowConfig,
        quality_config: QualityConfig,
        effect_type: str = "pyro"
    ):
        """
        Initialize workflow state machine.

        Args:
            workflow_config: Workflow configuration
            quality_config: Quality threshold configuration
            effect_type: Effect type for quality thresholds
        """
        self.workflow_config = workflow_config
        self.quality_config = quality_config
        self.effect_type = effect_type

        # Current state
        self.current_stage = WorkflowStage.SESSION_START
        self.iteration = 0
        self.retry_count = 0
        self.technique_changes = 0

        # Quality tracking
        self.scores: List[float] = []
        self.best_score = 0.0
        self.best_iteration = 0

        # Stage history
        self.history: List[StageResult] = []

        # Pending decision (for AWAITING_APPROVAL state)
        self.pending_decision: Optional[Dict[str, Any]] = None
        self.resume_stage: Optional[WorkflowStage] = None

        logger.info(f"Workflow initialized: effect_type={effect_type}")

    def get_stage(self) -> WorkflowStage:
        """Get current workflow stage."""
        return self.current_stage

    def advance(self, result: StageResult) -> WorkflowStage:
        """
        Advance to next stage based on result.

        Args:
            result: Result of the current stage

        Returns:
            New workflow stage
        """
        self.history.append(result)
        old_stage = self.current_stage

        # Handle special cases
        if result.stage == WorkflowStage.AWAITING_APPROVAL:
            if result.outcome == WorkflowOutcome.SUCCESS and self.resume_stage:
                self.current_stage = self.resume_stage
                self.resume_stage = None
                self.pending_decision = None
            else:
                self.current_stage = TRANSITIONS.get(
                    result.stage, {}
                ).get(result.outcome, WorkflowStage.SESSION_END)
        else:
            # Look up transition
            transitions = TRANSITIONS.get(result.stage, {})
            next_stage = transitions.get(result.outcome, WorkflowStage.SESSION_END)

            if next_stage:
                self.current_stage = next_stage
            else:
                logger.warning(
                    f"No transition for {result.stage.value}:{result.outcome.value}, "
                    f"defaulting to SESSION_END"
                )
                self.current_stage = WorkflowStage.SESSION_END

        # Update counters based on stage
        if result.stage == WorkflowStage.GENERATE_SCRIPT:
            if result.outcome == WorkflowOutcome.SUCCESS:
                self.iteration += 1
                self.retry_count = 0
            elif result.outcome == WorkflowOutcome.RETRY:
                self.retry_count += 1

        elif result.stage == WorkflowStage.EXECUTE_BLENDER:
            if result.outcome == WorkflowOutcome.RETRY:
                self.retry_count += 1

        elif result.stage == WorkflowStage.EVALUATE_QUALITY:
            if result.outcome == WorkflowOutcome.SUCCESS:
                score = result.data.get("score", 0.0)
                self.scores.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_iteration = self.iteration

        logger.info(
            f"Workflow transition: {old_stage.value} -> {self.current_stage.value} "
            f"(iteration={self.iteration}, outcome={result.outcome.value})"
        )

        return self.current_stage

    def set_pending_approval(
        self,
        decision: Dict[str, Any],
        resume_stage: WorkflowStage
    ) -> None:
        """
        Set pending approval and transition to AWAITING_APPROVAL.

        Args:
            decision: Decision details for approval request
            resume_stage: Stage to resume after approval
        """
        self.pending_decision = decision
        self.resume_stage = resume_stage
        self.current_stage = WorkflowStage.AWAITING_APPROVAL

        logger.info(f"Awaiting approval for: {decision.get('type', 'unknown')}")

    def record_technique_change(self) -> bool:
        """
        Record a technique change and check if limit exceeded.

        Returns:
            True if technique change allowed, False if limit exceeded
        """
        self.technique_changes += 1
        return self.technique_changes <= self.workflow_config.max_technique_changes

    def check_iteration_limit(self) -> bool:
        """Check if iteration limit has been reached."""
        return self.iteration >= self.workflow_config.max_iterations

    def check_retry_limit(self) -> bool:
        """Check if retry limit has been reached."""
        return self.retry_count >= self.workflow_config.max_retries

    def check_quality_passed(self) -> bool:
        """Check if quality threshold has been met."""
        if not self.scores:
            return False
        threshold = self.quality_config.get_pass_threshold(self.effect_type)
        return self.scores[-1] >= threshold

    def check_quality_excellent(self) -> bool:
        """Check if excellent quality has been achieved."""
        if not self.scores:
            return False
        return self.scores[-1] >= self.quality_config.vfx_excellent_threshold

    def check_plateau(self) -> bool:
        """
        Check if quality scores have plateaued.

        Returns:
            True if scores stable for plateau_iterations with change < plateau_threshold
        """
        n = self.quality_config.plateau_iterations
        if len(self.scores) < n:
            return False

        recent = self.scores[-n:]
        score_range = max(recent) - min(recent)

        return score_range < self.quality_config.plateau_threshold

    def check_quality_degraded(self) -> bool:
        """Check if quality has degraded from previous iteration."""
        if len(self.scores) < 2:
            return False
        return self.scores[-1] < self.scores[-2]

    def get_pass_threshold(self) -> float:
        """Get quality pass threshold for current effect type."""
        return self.quality_config.get_pass_threshold(self.effect_type)

    def get_status(self) -> Dict[str, Any]:
        """Get full workflow status."""
        return {
            "stage": self.current_stage.value,
            "iteration": self.iteration,
            "max_iterations": self.workflow_config.max_iterations,
            "retry_count": self.retry_count,
            "max_retries": self.workflow_config.max_retries,
            "technique_changes": self.technique_changes,
            "max_technique_changes": self.workflow_config.max_technique_changes,
            "scores": self.scores,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "pass_threshold": self.get_pass_threshold(),
            "quality_passed": self.check_quality_passed(),
            "plateau_detected": self.check_plateau(),
            "pending_decision": self.pending_decision,
        }

    def format_progress(self) -> str:
        """Format a compact progress string."""
        score_str = f"{self.scores[-1]:.1f}" if self.scores else "N/A"
        best_str = f"{self.best_score:.1f}" if self.best_score > 0 else "N/A"
        threshold = self.get_pass_threshold()

        return (
            f"Iter {self.iteration}/{self.workflow_config.max_iterations} | "
            f"Score: {score_str} (best: {best_str}, target: {threshold:.0f}) | "
            f"Stage: {self.current_stage.value}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for persistence."""
        return {
            "current_stage": self.current_stage.value,
            "iteration": self.iteration,
            "retry_count": self.retry_count,
            "technique_changes": self.technique_changes,
            "scores": self.scores,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "effect_type": self.effect_type,
            "pending_decision": self.pending_decision,
            "resume_stage": self.resume_stage.value if self.resume_stage else None,
            "history": [h.to_dict() for h in self.history[-20:]],  # Keep last 20
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        workflow_config: WorkflowConfig,
        quality_config: QualityConfig
    ) -> "WorkflowStateMachine":
        """Restore state machine from dict."""
        sm = cls(
            workflow_config=workflow_config,
            quality_config=quality_config,
            effect_type=data.get("effect_type", "pyro")
        )

        sm.current_stage = WorkflowStage(data.get("current_stage", "session_start"))
        sm.iteration = data.get("iteration", 0)
        sm.retry_count = data.get("retry_count", 0)
        sm.technique_changes = data.get("technique_changes", 0)
        sm.scores = data.get("scores", [])
        sm.best_score = data.get("best_score", 0.0)
        sm.best_iteration = data.get("best_iteration", 0)
        sm.pending_decision = data.get("pending_decision")

        if data.get("resume_stage"):
            sm.resume_stage = WorkflowStage(data["resume_stage"])

        return sm
