"""
Utility modules for the orchestrator.

- mcp_connection_pool: Persistent MCP server connections
- budget_tracker: API cost tracking
- session_persistence: JSON-based state saving
- retry: Async retry with exponential backoff
- code_pattern_memory: Successful code pattern storage (Strategy 4)
"""

from .mcp_connection_pool import MCPConnectionPool, get_connection_pool
from .budget_tracker import BudgetTracker, get_budget_tracker
from .session_persistence import (
    SessionPersistence,
    get_persistence,
    generate_session_id,
    create_session_from_request,
    resume_or_create_session,
)
from .retry import retry_with_backoff, estimate_query_complexity, get_reasoning_settings
from .code_pattern_memory import (
    CodePattern,
    CodePatternMemory,
    get_pattern_memory,
)
from .artifact_manager import (
    ArtifactManager,
    get_artifact_manager,
    ResearchArtifact,
    QualityArtifact,
    IterationArtifact,
    ScorecardArtifact,
    ManifestArtifact,
)

__all__ = [
    "MCPConnectionPool",
    "get_connection_pool",
    "BudgetTracker",
    "get_budget_tracker",
    "SessionPersistence",
    "get_persistence",
    "generate_session_id",
    "create_session_from_request",
    "resume_or_create_session",
    "retry_with_backoff",
    "estimate_query_complexity",
    "get_reasoning_settings",
    # Code Pattern Memory (Strategy 4)
    "CodePattern",
    "CodePatternMemory",
    "get_pattern_memory",
    # Artifact Manager (Artifact-First Handoffs)
    "ArtifactManager",
    "get_artifact_manager",
    "ResearchArtifact",
    "QualityArtifact",
    "IterationArtifact",
    "ScorecardArtifact",
    "ManifestArtifact",
]
