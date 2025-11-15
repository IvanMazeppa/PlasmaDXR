"""
GraphState definition for Log Analysis RAG workflow
Based on NVIDIA BAT.AI architecture with PlasmaDX-specific extensions
"""

from typing import List, TypedDict, Optional


class GraphState(TypedDict):
    """State that flows through the LangGraph workflow"""

    # Input
    path: str  # Log file path or directory
    question: str  # User's diagnostic question

    # Retrieved documents
    documents: List[str]  # Relevant log entries/PIX data

    # Generation
    generation: str  # LLM's diagnostic response

    # PlasmaDX-specific extensions
    confidence: float  # Confidence score (0.0-1.0) for diagnosis
    recommended_specialist: Optional[str]  # Which specialist agent to assign
    file_line_refs: List[str]  # file:line references for code locations
    artifact_paths: List[str]  # Related screenshot/PIX/buffer paths
