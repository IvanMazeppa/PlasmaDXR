"""
Session State Persistence

Manages saving and loading orchestrator session state for resumability.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .autonomy import AutonomyConfig, AutonomyController, TrustState
    from .guardrails import GuardrailConfig, SessionUsage, TokenGuardrails
    from .workflow import (
        QualityConfig,
        WorkflowConfig,
        WorkflowStage,
        WorkflowStateMachine,
    )
except ImportError:
    from autonomy import AutonomyConfig, AutonomyController, TrustState
    from guardrails import GuardrailConfig, SessionUsage, TokenGuardrails
    from workflow import (
        QualityConfig,
        WorkflowConfig,
        WorkflowStage,
        WorkflowStateMachine,
    )

logger = logging.getLogger("blender-orchestrator.state")


@dataclass
class AssetRequest:
    """Request for asset generation."""
    asset_name: str
    effect_type: str
    description: str
    reference_path: Optional[str] = None
    semantic_query: Optional[str] = None
    resolution: int = 96
    frame_end: int = 50
    technique_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "effect_type": self.effect_type,
            "description": self.description,
            "reference_path": self.reference_path,
            "semantic_query": self.semantic_query,
            "resolution": self.resolution,
            "frame_end": self.frame_end,
            "technique_name": self.technique_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetRequest":
        return cls(
            asset_name=data["asset_name"],
            effect_type=data["effect_type"],
            description=data["description"],
            reference_path=data.get("reference_path"),
            semantic_query=data.get("semantic_query"),
            resolution=data.get("resolution", 96),
            frame_end=data.get("frame_end", 50),
            technique_name=data.get("technique_name"),
        )


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    iteration: int
    score: Optional[float]
    action: str
    technique: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    script_path: Optional[str] = None
    render_path: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "score": self.score,
            "action": self.action,
            "technique": self.technique,
            "parameters": self.parameters,
            "script_path": self.script_path,
            "render_path": self.render_path,
            "issues": self.issues,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        return cls(
            iteration=data["iteration"],
            score=data.get("score"),
            action=data["action"],
            technique=data.get("technique"),
            parameters=data.get("parameters", {}),
            script_path=data.get("script_path"),
            render_path=data.get("render_path"),
            issues=data.get("issues", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
        )


@dataclass
class SessionState:
    """Complete session state for persistence."""
    session_id: str
    request: AssetRequest
    status: str  # "in_progress", "completed", "failed", "paused"

    # Current state
    current_technique: Optional[str] = None
    current_parameters: Dict[str, Any] = field(default_factory=dict)
    current_script_path: Optional[str] = None

    # Best result
    best_score: float = 0.0
    best_iteration: int = 0
    best_render_path: Optional[str] = None
    best_vdb_path: Optional[str] = None

    # History
    iterations: List[IterationRecord] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Workflow state (stored separately)
    workflow_data: Dict[str, Any] = field(default_factory=dict)

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request": self.request.to_dict(),
            "status": self.status,
            "current_technique": self.current_technique,
            "current_parameters": self.current_parameters,
            "current_script_path": self.current_script_path,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "best_render_path": self.best_render_path,
            "best_vdb_path": self.best_vdb_path,
            "iterations": [i.to_dict() for i in self.iterations],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "workflow_data": self.workflow_data,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            request=AssetRequest.from_dict(data["request"]),
            status=data.get("status", "in_progress"),
            current_technique=data.get("current_technique"),
            current_parameters=data.get("current_parameters", {}),
            current_script_path=data.get("current_script_path"),
            best_score=data.get("best_score", 0.0),
            best_iteration=data.get("best_iteration", 0),
            best_render_path=data.get("best_render_path"),
            best_vdb_path=data.get("best_vdb_path"),
            iterations=[IterationRecord.from_dict(i) for i in data.get("iterations", [])],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            workflow_data=data.get("workflow_data", {}),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
        )


class SessionManager:
    """
    Manages session state persistence and recovery.

    Handles saving/loading session state, listing sessions,
    and cleaning up old sessions.
    """

    def __init__(
        self,
        state_directory: Path,
        workflow_config: WorkflowConfig,
        quality_config: QualityConfig,
        autonomy_config: AutonomyConfig,
        guardrail_config: GuardrailConfig,
        retention_days: int = 30
    ):
        """
        Initialize session manager.

        Args:
            state_directory: Directory for state files
            workflow_config: Workflow configuration
            quality_config: Quality configuration
            autonomy_config: Autonomy configuration
            guardrail_config: Guardrail configuration
            retention_days: Days to keep completed sessions
        """
        self.state_directory = Path(state_directory)
        self.state_directory.mkdir(parents=True, exist_ok=True)

        self.workflow_config = workflow_config
        self.quality_config = quality_config
        self.autonomy_config = autonomy_config
        self.guardrail_config = guardrail_config
        self.retention_days = retention_days

        logger.info(f"SessionManager initialized: {self.state_directory}")

    def _session_file(self, session_id: str) -> Path:
        """Get path to session state file."""
        return self.state_directory / f"session_{session_id}.json"

    def create_session(self, request: AssetRequest) -> str:
        """
        Create a new session.

        Args:
            request: Asset generation request

        Returns:
            Session ID
        """
        # Generate session ID from asset name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{request.asset_name}_{timestamp}"

        state = SessionState(
            session_id=session_id,
            request=request,
            status="in_progress"
        )

        self.save_session(state)
        logger.info(f"Created session: {session_id}")

        return session_id

    def save_session(self, state: SessionState) -> None:
        """Save session state to file."""
        state.updated_at = datetime.now()
        file_path = self._session_file(state.session_id)

        try:
            with open(file_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug(f"Saved session: {state.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session {state.session_id}: {e}")

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state from file."""
        file_path = self._session_file(session_id)

        if not file_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            state = SessionState.from_dict(data)
            logger.info(f"Loaded session: {session_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(
        self,
        status_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List available sessions.

        Args:
            status_filter: Filter by status ("in_progress", "completed", etc.)
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        sessions = []

        for file_path in sorted(
            self.state_directory.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        ):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if status_filter and data.get("status") != status_filter:
                    continue

                sessions.append({
                    "session_id": data.get("session_id"),
                    "asset_name": data.get("request", {}).get("asset_name"),
                    "effect_type": data.get("request", {}).get("effect_type"),
                    "status": data.get("status"),
                    "best_score": data.get("best_score", 0.0),
                    "iterations": len(data.get("iterations", [])),
                    "updated_at": data.get("updated_at"),
                })

                if len(sessions) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to read session file {file_path}: {e}")

        return sessions

    def get_resumable_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions that can be resumed."""
        return self.list_sessions(status_filter="in_progress")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session state file."""
        file_path = self._session_file(session_id)

        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
                return False

        return False

    def cleanup_old_sessions(self) -> int:
        """
        Clean up old completed sessions.

        Returns:
            Number of sessions deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (self.retention_days * 86400)

        for file_path in self.state_directory.glob("session_*.json"):
            try:
                if file_path.stat().st_mtime < cutoff:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Only delete completed/failed sessions
                    if data.get("status") in ["completed", "failed"]:
                        file_path.unlink()
                        deleted += 1
                        logger.debug(f"Cleaned up old session: {file_path.name}")

            except Exception as e:
                logger.warning(f"Error during cleanup of {file_path}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old sessions")

        return deleted

    def create_workflow(
        self,
        state: SessionState
    ) -> WorkflowStateMachine:
        """
        Create or restore workflow state machine for a session.

        Args:
            state: Session state

        Returns:
            Workflow state machine
        """
        if state.workflow_data:
            # Restore from saved state
            return WorkflowStateMachine.from_dict(
                state.workflow_data,
                self.workflow_config,
                self.quality_config
            )
        else:
            # Create new
            return WorkflowStateMachine(
                workflow_config=self.workflow_config,
                quality_config=self.quality_config,
                effect_type=state.request.effect_type
            )

    def create_autonomy_controller(self) -> AutonomyController:
        """Create autonomy controller with shared state file."""
        state_path = self.state_directory / "autonomy_state.json"
        return AutonomyController(
            config=self.autonomy_config,
            state_path=state_path
        )

    def create_guardrails(self, session_id: str) -> TokenGuardrails:
        """Create token guardrails for a session."""
        stats_path = self.state_directory / "daily_stats.json"
        return TokenGuardrails(
            config=self.guardrail_config,
            session_id=session_id,
            stats_path=stats_path
        )

    def update_session_from_workflow(
        self,
        state: SessionState,
        workflow: WorkflowStateMachine
    ) -> None:
        """Update session state from workflow state machine."""
        state.workflow_data = workflow.to_dict()
        state.best_score = workflow.best_score
        state.best_iteration = workflow.best_iteration

    def update_session_from_guardrails(
        self,
        state: SessionState,
        guardrails: TokenGuardrails
    ) -> None:
        """Update session state from guardrails."""
        state.total_input_tokens = guardrails.session.input_tokens
        state.total_output_tokens = guardrails.session.output_tokens
        state.total_cost_usd = guardrails.session.cost_usd

    def record_iteration(
        self,
        state: SessionState,
        iteration: int,
        score: Optional[float],
        action: str,
        technique: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        script_path: Optional[str] = None,
        render_path: Optional[str] = None,
        issues: Optional[List[str]] = None
    ) -> None:
        """Record an iteration to session state."""
        record = IterationRecord(
            iteration=iteration,
            score=score,
            action=action,
            technique=technique,
            parameters=parameters or {},
            script_path=script_path,
            render_path=render_path,
            issues=issues or []
        )

        state.iterations.append(record)
        state.current_technique = technique
        state.current_parameters = parameters or {}
        state.current_script_path = script_path

        if score is not None and score > state.best_score:
            state.best_score = score
            state.best_iteration = iteration
            state.best_render_path = render_path

    def finalize_session(
        self,
        state: SessionState,
        status: str,
        vdb_path: Optional[str] = None
    ) -> None:
        """Finalize session with final status."""
        state.status = status
        if vdb_path:
            state.best_vdb_path = vdb_path
        self.save_session(state)

        logger.info(
            f"Session finalized: {state.session_id} "
            f"status={status}, best_score={state.best_score:.1f}"
        )
