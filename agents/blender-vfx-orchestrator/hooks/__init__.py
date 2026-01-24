"""
Hooks package for OpenAI Agents SDK RunHooks implementations.

This package provides:
- Enforcement hooks that ensure agents follow required patterns
- Diagnostic hooks for analyzing agent behavior and detecting repetition patterns
"""

from .enforcement_hooks import (
    EnforcementHooks,
    LoopDetectedError,
    DocQueryRequiredError,
    TurnBudgetExceededError,
    EnforcementConfig,
)

from .diagnostic_hooks import (
    DiagnosticHooks,
    PromptFingerprint,
    ToolCallPattern,
    OutputPattern,
    CommunicationFlowTracker,
)

__all__ = [
    # Enforcement
    "EnforcementHooks",
    "LoopDetectedError",
    "DocQueryRequiredError",
    "TurnBudgetExceededError",
    "EnforcementConfig",
    # Diagnostics
    "DiagnosticHooks",
    "PromptFingerprint",
    "ToolCallPattern",
    "OutputPattern",
    "CommunicationFlowTracker",
]
