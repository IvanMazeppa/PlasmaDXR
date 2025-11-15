"""
LangGraph StateGraph Workflow - NVIDIA Multi-Agent RAG
Self-correcting diagnostic workflow with hallucination detection
"""

from langgraph.graph import StateGraph, END

from .state import GraphState
from ..nodes import (
    retrieve_node,
    rerank_node,
    grade_documents_node,
    generate_node,
    grade_generation_node,
    transform_query_node
)
from ..edges import decide_to_generate, check_hallucination


def create_workflow() -> StateGraph:
    """
    Create the complete LangGraph StateGraph workflow

    Workflow:
    1. retrieve → Hybrid retrieval (BM25 + FAISS)
    2. rerank → NVIDIA reranker (top 20 → top 10)
    3. grade_documents → Binary relevance scoring
    4. [CONDITIONAL] decide_to_generate:
       - If relevant → generate
       - If not → transform_query (self-correction)
    5. generate → Diagnostic response (DiagnosisOutput)
    6. grade_generation → Hallucination detection
    7. [CONDITIONAL] check_hallucination:
       - If grounded → END
       - If hallucinated → transform_query (self-correction)
    8. transform_query → Rewrite question, loop back to retrieve

    Returns:
        Compiled StateGraph workflow
    """
    # Initialize workflow
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("transform_query", transform_query_node)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges (sequential flow)
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")

    # Conditional edge: decide_to_generate
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )

    # Continue from generate to hallucination check
    workflow.add_edge("generate", "grade_generation")

    # Conditional edge: check_hallucination
    workflow.add_conditional_edges(
        "grade_generation",
        check_hallucination,
        {
            "END": END,
            "transform_query": "transform_query"
        }
    )

    # Self-correction loop: transform_query → retrieve
    workflow.add_edge("transform_query", "retrieve")

    # Compile workflow
    return workflow.compile()


# Create singleton workflow instance
_workflow = None


def get_workflow() -> StateGraph:
    """
    Get or create compiled workflow (singleton pattern)
    """
    global _workflow

    if _workflow is None:
        print("Compiling LangGraph workflow...")
        _workflow = create_workflow()
        print("✅ Workflow compiled")

    return _workflow
