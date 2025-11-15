"""
LangGraph conditional edges for routing decisions
"""

from ..graph.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    """
    Conditional edge: Route to generate or transform_query

    Decision logic:
    - If documents are relevant → "generate"
    - If no relevant documents → "transform_query" (self-correction loop)

    Args:
        state: Current graph state after grade_documents

    Returns:
        Next node name: "generate" or "transform_query"
    """
    documents = state.get("documents", [])

    if len(documents) > 0:
        print("  → Route: GENERATE (documents are relevant)")
        return "generate"
    else:
        print("  → Route: TRANSFORM_QUERY (no relevant documents, retry)")
        return "transform_query"


def check_hallucination(state: GraphState) -> str:
    """
    Conditional edge: Check if generation is grounded or hallucinated

    Decision logic:
    - If generation is grounded → "END" (success)
    - If generation hallucinated → "transform_query" (self-correction loop)

    Args:
        state: Current graph state after grade_generation

    Returns:
        Next node name: "END" or "transform_query"
    """
    hallucination_check = state.get("hallucination_check", "yes")  # Default: assume OK

    if hallucination_check == "yes":
        print("  → Route: END (diagnosis is grounded)")
        return "END"
    else:
        print("  → Route: TRANSFORM_QUERY (hallucination detected, retry)")
        return "transform_query"


__all__ = ["decide_to_generate", "check_hallucination"]
