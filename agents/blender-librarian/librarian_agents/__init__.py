"""
OpenAI Agents SDK-based agents for blender-librarian.

Phase 9 implementation providing multi-agent orchestration with:
- Native MCP support for blender-manual integration
- Agent handoffs for specialized tasks
- SQLite-backed session persistence for cross-session learning
- Pydantic models for type-safe structured outputs
"""

from .doc_expert import DocExpertAgent, create_doc_expert, create_doc_expert_pooled
from .vision_expert import VisionExpertAgent, create_vision_expert
from .librarian_orchestrator import LibrarianOrchestrator
from .session_manager import SessionManager, SessionContext

# Pydantic models for structured agent outputs
from .models import (
    ModificationAdvice,
    RenderDiagnosis,
    AgentSearchResult,
    PlaybookEntry,
    BudgetStatus,
)

__all__ = [
    # Agents
    "DocExpertAgent",
    "create_doc_expert",
    "create_doc_expert_pooled",
    "VisionExpertAgent",
    "create_vision_expert",
    "LibrarianOrchestrator",
    # Session management
    "SessionManager",
    "SessionContext",
    # Pydantic models
    "ModificationAdvice",
    "RenderDiagnosis",
    "AgentSearchResult",
    "PlaybookEntry",
    "BudgetStatus",
]
