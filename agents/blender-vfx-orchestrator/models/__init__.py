"""
Pydantic models for structured agent inputs/outputs.

Provides type-safe data structures for:
- SharedContext: State shared across all agents
- IterationResult: Result of a single iteration
- AssetRequest: User request for asset generation
- SessionState: Persistent session state
- StuckDetectionState: Escape velocity tracking (Strategy 3)
"""

from .shared_context import (
    SharedContext,
    IterationResult,
    AssetRequest,
    SessionState,
    QualityMetrics,
    ScriptModification,
    BlenderExecution,
    ExperimentOutcome,
    # Enums
    EffectType,
    SessionStatus,
    IssueSeverity,
    EscapeLevel,
    # Strategy 3: Stuck detection
    StuckDetectionState,
)

__all__ = [
    "SharedContext",
    "IterationResult",
    "AssetRequest",
    "SessionState",
    "QualityMetrics",
    "ScriptModification",
    "BlenderExecution",
    "ExperimentOutcome",
    # Enums
    "EffectType",
    "SessionStatus",
    "IssueSeverity",
    "EscapeLevel",
    # Strategy 3: Stuck detection
    "StuckDetectionState",
]
