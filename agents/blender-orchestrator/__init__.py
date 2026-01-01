"""
Blender VFX Orchestrator

Autonomous Claude Agent SDK agent for Blender VFX asset generation
with adaptive autonomy and token guardrails.
"""

from .autonomy import (
    AutonomyConfig,
    AutonomyController,
    AutonomyLevel,
    DecisionType,
    TrustAdjustments,
    TrustEvent,
)
from .guardrails import (
    CostLimits,
    GuardrailAction,
    GuardrailConfig,
    TokenGuardrails,
    TokenLimits,
    TokenPricing,
)
from .orchestrator import BlenderOrchestratorAgent
from .state import AssetRequest, SessionManager, SessionState
from .workflow import (
    QualityConfig,
    StageResult,
    WorkflowConfig,
    WorkflowOutcome,
    WorkflowStage,
    WorkflowStateMachine,
)

__all__ = [
    # Main agent
    "BlenderOrchestratorAgent",
    # Autonomy
    "AutonomyConfig",
    "AutonomyController",
    "AutonomyLevel",
    "DecisionType",
    "TrustAdjustments",
    "TrustEvent",
    # Guardrails
    "CostLimits",
    "GuardrailAction",
    "GuardrailConfig",
    "TokenGuardrails",
    "TokenLimits",
    "TokenPricing",
    # State
    "AssetRequest",
    "SessionManager",
    "SessionState",
    # Workflow
    "QualityConfig",
    "StageResult",
    "WorkflowConfig",
    "WorkflowOutcome",
    "WorkflowStage",
    "WorkflowStateMachine",
]

__version__ = "0.1.0"
