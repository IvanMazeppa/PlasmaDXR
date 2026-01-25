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
    # Specialized hook factories
    create_research_hooks,
    create_script_writer_hooks,
    create_fallback_script_writer_hooks,
    create_quality_analyst_hooks,
    create_learning_agent_hooks,
    create_api_spec_hooks,
    create_code_writer_hooks,
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
    # Specialized hook factories
    "create_research_hooks",
    "create_script_writer_hooks",
    "create_fallback_script_writer_hooks",
    "create_quality_analyst_hooks",
    "create_learning_agent_hooks",
    "create_api_spec_hooks",
    "create_code_writer_hooks",
    # Diagnostics
    "DiagnosticHooks",
    "PromptFingerprint",
    "ToolCallPattern",
    "OutputPattern",
    "CommunicationFlowTracker",
]
