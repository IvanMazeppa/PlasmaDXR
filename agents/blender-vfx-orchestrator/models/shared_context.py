"""
Pydantic Models for the Blender VFX Orchestrator.

Provides type-safe, validated data structures for:
- SharedContext: State shared across all agents in an iteration
- IterationResult: Result of a single VFX iteration
- AssetRequest: User request for asset generation
- SessionState: Persistent session state for resumption

These models ensure consistency across agent handoffs and enable
automatic validation of inputs/outputs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class EffectType(str, Enum):
    """Supported VFX effect types."""
    # Pyro effects
    PYRO = "pyro"
    EXPLOSION = "explosion"
    FIRE = "fire"
    SMOKE = "smoke"
    # Celestial effects
    NEBULA = "nebula"
    SUN = "sun"
    STAR = "star"
    SUPERNOVA = "supernova"
    # Water/liquid effects
    WATER = "water"
    WATER_SPLASH = "water_splash"
    OCEAN = "ocean"
    WATERFALL = "waterfall"
    RAIN = "rain"


class SessionStatus(str, Enum):
    """Session lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class IssueSeverity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EscapeLevel(int, Enum):
    """
    Escape velocity levels for stuck detection.

    Higher levels indicate more aggressive exploration strategies.
    """
    NORMAL = 0           # Standard modification
    KNOWLEDGE_CHECK = 1  # Query knowledge base for alternatives
    SWITCH_TECHNIQUE = 2 # Force different technique entirely
    MINE_DOCS = 3        # Search documentation for novel approaches
    REQUEST_GUIDANCE = 4 # Admit defeat, request human help


# =============================================================================
# STUCK DETECTION STATE
# =============================================================================

class StuckDetectionState(BaseModel):
    """
    Track stuck indicators for escape velocity mechanism (Strategy 5).

    Monitors iteration history to detect when the agent is stuck
    and determines the appropriate escape action. Supports step-down
    capability when progress resumes.

    Key features:
    - Tracks same-issue persistence and score plateaus
    - Distinguishes between "tried" and "failed" techniques
    - Allows escape level to step down when progress resumes
    - Records documentation queries to avoid duplicates
    """
    same_issue_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive iterations with same primary issue"
    )
    last_primary_issue: str = Field(
        default="",
        description="Primary issue from last iteration"
    )
    plateau_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive iterations with <3 point score change"
    )
    last_score: float = Field(
        default=0.0,
        description="Score from last iteration"
    )
    escape_level: EscapeLevel = Field(
        default=EscapeLevel.NORMAL,
        description="Current escape velocity level"
    )
    techniques_tried: List[str] = Field(
        default_factory=list,
        description="Techniques already attempted this session"
    )
    techniques_failed: List[str] = Field(
        default_factory=list,
        description="Techniques that failed to make progress (avoid retrying)"
    )
    research_queries_used: List[str] = Field(
        default_factory=list,
        description="Documentation queries already executed"
    )
    progress_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive iterations with significant progress (for step-down)"
    )
    peak_escape_level: EscapeLevel = Field(
        default=EscapeLevel.NORMAL,
        description="Highest escape level reached this session"
    )

    def update_from_iteration(
        self,
        score: float,
        primary_issue: str,
        technique_used: str = ""
    ) -> EscapeLevel:
        """
        Update state from iteration result and compute new escape level.

        Supports step-down: if significant progress is made (score +5 or new issue),
        the escape level can step down after 2 consecutive progress iterations.

        Args:
            score: Quality score from this iteration
            primary_issue: Primary issue identified (empty if passed)
            technique_used: Technique name if available

        Returns:
            New escape level after update
        """
        # Track technique
        if technique_used and technique_used not in self.techniques_tried:
            self.techniques_tried.append(technique_used)

        # Check same issue persisting
        issue_changed = primary_issue != self.last_primary_issue
        if primary_issue and not issue_changed:
            self.same_issue_count += 1
        else:
            self.same_issue_count = 1 if primary_issue else 0
            self.last_primary_issue = primary_issue

        # Check score plateau (< 3 point change)
        score_delta = score - self.last_score  # Signed delta
        significant_progress = score_delta >= 5.0  # Positive progress of 5+ points

        if self.last_score > 0 and abs(score_delta) < 3.0:
            self.plateau_count += 1
            self.progress_count = 0
        else:
            self.plateau_count = 0
            # Track progress for step-down
            if significant_progress or issue_changed:
                self.progress_count += 1
            else:
                self.progress_count = 0

        self.last_score = score

        # Compute new escape level
        new_level = self._compute_escape_level()

        # Step-down logic: if we have consecutive progress, allow level to decrease
        if self.progress_count >= 2 and new_level < self.escape_level:
            # Allow step-down, but record peak for logging
            if self.escape_level > self.peak_escape_level:
                self.peak_escape_level = self.escape_level
            self.escape_level = new_level
        else:
            # Only escalate, never step down without progress
            if new_level > self.escape_level:
                self.escape_level = new_level
                if self.escape_level > self.peak_escape_level:
                    self.peak_escape_level = self.escape_level

        return self.escape_level

    def _compute_escape_level(self) -> EscapeLevel:
        """Compute escape level from current stuck indicators."""
        # Level 4: Exhausted all options
        if self.same_issue_count >= 4 or self.plateau_count >= 4:
            return EscapeLevel.REQUEST_GUIDANCE

        # Level 3: Need novel approaches
        if self.same_issue_count >= 3 or self.plateau_count >= 3:
            return EscapeLevel.MINE_DOCS

        # Level 2: Current technique exhausted
        if self.same_issue_count >= 2:
            return EscapeLevel.SWITCH_TECHNIQUE

        # Level 1: Early warning, check knowledge
        if self.plateau_count >= 2:
            return EscapeLevel.KNOWLEDGE_CHECK

        return EscapeLevel.NORMAL

    def mark_technique_failed(self, technique: str) -> None:
        """
        Mark a technique as failed (did not make progress).

        Failed techniques should be avoided when switching techniques.

        Args:
            technique: Name of the technique that failed
        """
        if technique and technique not in self.techniques_failed:
            self.techniques_failed.append(technique)

    def get_untried_techniques(self, available: List[str]) -> List[str]:
        """
        Get techniques that haven't been tried or have failed.

        Prioritizes untried techniques, excludes failed ones.

        Args:
            available: List of available technique names

        Returns:
            List of techniques to try, prioritizing untried over failed
        """
        untried = [t for t in available if t not in self.techniques_tried]
        if untried:
            return untried

        # All tried - return any that didn't explicitly fail
        non_failed = [t for t in available if t not in self.techniques_failed]
        return non_failed

    def reset_for_new_technique(self) -> None:
        """
        Reset stuck counters when switching to a new technique.

        Call this when switching techniques at Level 2+ to give the
        new technique a fair chance before escalating further.
        """
        self.same_issue_count = 0
        self.plateau_count = 0
        self.progress_count = 0
        # Note: Don't reset escape_level - let it naturally step down if progress is made

    def should_research_before_iteration(self) -> bool:
        """Check if proactive research is recommended."""
        return self.escape_level >= EscapeLevel.KNOWLEDGE_CHECK

    def get_escape_action(self) -> str:
        """Get recommended action for current escape level."""
        actions = {
            EscapeLevel.NORMAL: "proceed_with_modification",
            EscapeLevel.KNOWLEDGE_CHECK: "query_knowledge_base",
            EscapeLevel.SWITCH_TECHNIQUE: "generate_new_script_different_technique",
            EscapeLevel.MINE_DOCS: "search_documentation_for_alternatives",
            EscapeLevel.REQUEST_GUIDANCE: "report_stuck_request_guidance",
        }
        return actions[self.escape_level]

    def reset(self) -> None:
        """Reset stuck detection state (e.g., after technique switch)."""
        self.same_issue_count = 0
        self.plateau_count = 0
        # Keep techniques_tried and research_queries_used for session history


# =============================================================================
# QUALITY METRICS
# =============================================================================

class QualityMetrics(BaseModel):
    """
    Quality evaluation results from asset-evaluator.

    Contains scores from the consolidated v2 evaluation system.
    """
    overall_score: float = Field(
        ge=0.0, le=100.0,
        description="Combined quality score (0-100)"
    )
    passed: bool = Field(
        description="Whether the render passed quality thresholds"
    )

    # Individual metric scores (optional - depends on evaluation profile)
    lpips_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="LPIPS perceptual similarity (lower = better)"
    )
    siglip_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="SigLIP 2 semantic similarity"
    )
    topiq_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=100.0,
        description="TOPIQ aesthetic quality"
    )
    structural_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="DINOv2 structural similarity"
    )
    feature_cv: Optional[float] = Field(
        default=None,
        description="Feature size coefficient of variation"
    )

    # Diagnostic information
    issues: List[str] = Field(
        default_factory=list,
        description="Identified quality issues"
    )
    primary_issue: Optional[str] = Field(
        default=None,
        description="Most critical issue to address"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Recommended parameter changes"
    )

    # Metadata
    profile_used: str = Field(
        default="standard",
        description="Evaluation profile (quick/standard/comprehensive)"
    )
    evaluation_time_seconds: Optional[float] = Field(
        default=None,
        description="Time taken for evaluation"
    )


# =============================================================================
# SCRIPT MODIFICATION
# =============================================================================

class ScriptModification(BaseModel):
    """
    Modifications applied to a Blender script.

    Tracks parameter changes across iterations.
    """
    script_path: str = Field(
        description="Path to the Blender script"
    )
    modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter changes applied"
    )
    technique_name: Optional[str] = Field(
        default=None,
        description="Technique used (from script-generator catalog)"
    )
    validation_passed: bool = Field(
        default=True,
        description="Whether script validation passed"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="Validation warnings/errors if any"
    )


# =============================================================================
# BLENDER EXECUTION
# =============================================================================

class BlenderExecution(BaseModel):
    """
    Results from Blender script execution.

    Contains paths to outputs and any errors encountered.
    """
    success: bool = Field(
        description="Whether execution completed successfully"
    )
    run_dir: str = Field(
        description="Directory containing run outputs"
    )
    render_path: Optional[str] = Field(
        default=None,
        description="Path to rendered frame (if available)"
    )
    vdb_files: List[str] = Field(
        default_factory=list,
        description="List of VDB files generated"
    )
    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total execution time"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages if execution failed"
    )
    stdout_log: Optional[str] = Field(
        default=None,
        description="Path to stdout log file"
    )


# =============================================================================
# EXPERIMENT OUTCOME
# =============================================================================

class ExperimentOutcome(BaseModel):
    """
    Outcome of an experiment iteration for learning.

    Used to record what worked and what didn't for future iterations.
    """
    hypothesis: str = Field(
        description="What we were testing"
    )
    issue_addressed: str = Field(
        description="The problem we tried to fix"
    )
    success: bool = Field(
        description="Whether the experiment achieved its goal"
    )
    score_improvement: float = Field(
        description="Score change (positive = improvement)"
    )
    observed_effects: List[str] = Field(
        default_factory=list,
        description="What changed as a result"
    )
    learnings: List[str] = Field(
        default_factory=list,
        description="What we learned from this experiment"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Gotchas discovered"
    )


# =============================================================================
# ITERATION RESULT
# =============================================================================

class IterationResult(BaseModel):
    """
    Complete result of a single VFX iteration.

    Contains all artifacts and metrics from script generation through evaluation.
    """
    iteration: int = Field(
        ge=1,
        description="Iteration number (1-indexed)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When this iteration completed"
    )

    # Components
    script: ScriptModification = Field(
        description="Script generation/modification details"
    )
    execution: BlenderExecution = Field(
        description="Blender execution results"
    )
    quality: QualityMetrics = Field(
        description="Quality evaluation metrics"
    )
    experiment: Optional[ExperimentOutcome] = Field(
        default=None,
        description="Experiment tracking if recorded"
    )

    # Summary
    passed: bool = Field(
        description="Whether this iteration passed quality thresholds"
    )
    score: float = Field(
        ge=0.0, le=100.0,
        description="Overall quality score"
    )
    improvement: float = Field(
        default=0.0,
        description="Score improvement from previous iteration"
    )


# =============================================================================
# ASSET REQUEST
# =============================================================================

class AssetRequest(BaseModel):
    """
    User request for VFX asset generation.

    Captures all parameters needed to start an asset generation session.
    """
    asset_name: str = Field(
        description="Name for the asset (used for file naming)"
    )
    description: str = Field(
        description="Description of what to create"
    )
    effect_type: EffectType = Field(
        default=EffectType.PYRO,
        description="Type of VFX effect"
    )

    # Reference materials
    reference_path: Optional[str] = Field(
        default=None,
        description="Path to reference image for comparison"
    )
    semantic_query: Optional[str] = Field(
        default=None,
        description="Text description for semantic evaluation"
    )

    # Generation parameters
    resolution: int = Field(
        default=96,
        ge=32, le=512,
        description="Blender simulation resolution"
    )
    frame_start: int = Field(
        default=1,
        ge=1,
        description="Animation start frame"
    )
    frame_end: int = Field(
        default=50,
        ge=1,
        description="Animation end frame"
    )

    # Quality thresholds
    quality_threshold: float = Field(
        default=60.0,
        ge=0.0, le=100.0,
        description="Minimum quality score to pass"
    )
    max_iterations: int = Field(
        default=5,
        ge=1, le=20,
        description="Maximum iteration attempts"
    )

    # Optional overrides
    technique_name: Optional[str] = Field(
        default=None,
        description="Force specific technique from catalog"
    )
    initial_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override initial parameters"
    )


# =============================================================================
# SESSION STATE
# =============================================================================

class SessionState(BaseModel):
    """
    Persistent session state for resumption after context limits.

    Saved to JSON after each iteration to enable recovery.
    """
    session_id: str = Field(
        description="Unique session identifier"
    )
    status: SessionStatus = Field(
        default=SessionStatus.IN_PROGRESS,
        description="Current session status"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Session creation timestamp"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Last update timestamp"
    )

    # Request that started this session
    request: AssetRequest = Field(
        description="Original asset request"
    )

    # Progress tracking
    current_iteration: int = Field(
        default=0,
        ge=0,
        description="Current iteration number"
    )
    best_score: float = Field(
        default=0.0,
        ge=0.0, le=100.0,
        description="Best quality score achieved"
    )
    best_iteration: int = Field(
        default=0,
        ge=0,
        description="Iteration that achieved best score"
    )

    # Iteration history
    iterations: List[IterationResult] = Field(
        default_factory=list,
        description="Results from all iterations"
    )

    # Current state for resumption
    current_script_path: Optional[str] = Field(
        default=None,
        description="Path to current script"
    )
    current_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current parameter values"
    )
    current_issues: List[str] = Field(
        default_factory=list,
        description="Issues identified in last iteration"
    )

    # Stuck detection for escape velocity
    stuck_state: StuckDetectionState = Field(
        default_factory=StuckDetectionState,
        description="Stuck detection state for escape velocity"
    )

    # Research tracking for technique switching
    research_text: str = Field(
        default="",
        description="Research findings text for Script Writer context"
    )
    alternative_approaches: List[str] = Field(
        default_factory=list,
        description="Alternative approaches from Research Agent"
    )
    techniques_tried: List[str] = Field(
        default_factory=list,
        description="Techniques already attempted in this session"
    )
    current_technique: Optional[str] = Field(
        default=None,
        description="Currently active technique"
    )

    # Final outputs
    final_render_path: Optional[str] = Field(
        default=None,
        description="Path to final render (if passed)"
    )
    final_vdb_dir: Optional[str] = Field(
        default=None,
        description="Directory with final VDB files"
    )

    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.now().isoformat()

    def record_iteration(self, result: IterationResult) -> EscapeLevel:
        """
        Record a completed iteration and update best score.

        Also updates stuck detection state and returns the new escape level.

        Returns:
            Current escape level after this iteration
        """
        self.iterations.append(result)
        self.current_iteration = result.iteration

        if result.score > self.best_score:
            self.best_score = result.score
            self.best_iteration = result.iteration

        if result.passed:
            self.status = SessionStatus.PASSED
            self.final_render_path = result.execution.render_path
            if result.execution.vdb_files:
                self.final_vdb_dir = str(result.execution.run_dir)

        # Update stuck detection state
        primary_issue = result.quality.primary_issue or ""
        technique = result.script.technique_name or ""
        escape_level = self.stuck_state.update_from_iteration(
            score=result.score,
            primary_issue=primary_issue,
            technique_used=technique
        )

        self.update_timestamp()
        return escape_level


# =============================================================================
# SHARED CONTEXT
# =============================================================================

class SharedContext(BaseModel):
    """
    State shared across all agents during an iteration.

    Passed between agents during handoffs to maintain context.
    This is the primary data structure for agent coordination.
    """
    session: SessionState = Field(
        description="Persistent session state"
    )

    # Current iteration working data
    current_script: Optional[ScriptModification] = Field(
        default=None,
        description="Script being worked on"
    )
    current_execution: Optional[BlenderExecution] = Field(
        default=None,
        description="Latest execution results"
    )
    current_quality: Optional[QualityMetrics] = Field(
        default=None,
        description="Latest quality evaluation"
    )

    # Decision context
    should_continue: bool = Field(
        default=True,
        description="Whether to continue iterating"
    )
    next_action: str = Field(
        default="generate_script",
        description="Next action to take"
    )
    blocking_issues: List[str] = Field(
        default_factory=list,
        description="Issues preventing progress"
    )

    # Agent communication
    handoff_reason: Optional[str] = Field(
        default=None,
        description="Why the current agent is handing off"
    )
    handoff_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data for the next agent"
    )

    # Budget tracking
    iteration_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="API cost for this iteration"
    )
    session_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total API cost for the session"
    )

    @classmethod
    def from_request(cls, request: AssetRequest, session_id: str) -> "SharedContext":
        """Create a new SharedContext from an asset request."""
        session = SessionState(
            session_id=session_id,
            request=request,
        )
        return cls(session=session)

    def complete_iteration(self) -> Optional[IterationResult]:
        """
        Compile current iteration data into an IterationResult.

        Returns None if any required components are missing.
        """
        if not all([self.current_script, self.current_execution, self.current_quality]):
            return None

        result = IterationResult(
            iteration=self.session.current_iteration + 1,
            script=self.current_script,
            execution=self.current_execution,
            quality=self.current_quality,
            passed=self.current_quality.passed,
            score=self.current_quality.overall_score,
            improvement=self.current_quality.overall_score - self.session.best_score,
        )

        # Record in session
        self.session.record_iteration(result)
        self.session_cost += self.iteration_cost

        # Reset working data for next iteration
        self.current_script = None
        self.current_execution = None
        self.current_quality = None
        self.iteration_cost = 0.0

        return result
