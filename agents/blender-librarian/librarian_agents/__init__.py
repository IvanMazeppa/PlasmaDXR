"""
OpenAI Agents SDK-based agents for blender-librarian.

Phase 9 implementation providing multi-agent orchestration with:
- Native MCP support for blender-manual integration
- Agent handoffs for specialized tasks
- SQLite-backed session persistence for cross-session learning
"""

from .doc_expert import DocExpertAgent, create_doc_expert
from .vision_expert import VisionExpertAgent, create_vision_expert
from .librarian_orchestrator import LibrarianOrchestrator
from .session_manager import SessionManager, SessionContext

__all__ = [
    "DocExpertAgent",
    "create_doc_expert",
    "VisionExpertAgent",
    "create_vision_expert",
    "LibrarianOrchestrator",
    "SessionManager",
    "SessionContext",
]
