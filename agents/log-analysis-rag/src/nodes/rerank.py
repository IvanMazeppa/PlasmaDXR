"""
Rerank Node - NVIDIA nv-rerankqa-mistral-4b-v3
Reranks retrieved documents to prioritize most relevant ones
"""

from typing import List
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

from ..graph.state import GraphState


def rerank_node(state: GraphState) -> GraphState:
    """
    Rerank documents using NVIDIA's reranking model

    Takes top 20 from hybrid retrieval and reranks to get top 10 most relevant

    Args:
        state: Current graph state with documents from retrieval

    Returns:
        Updated state with reranked documents
    """
    print("---RERANK DOCUMENTS---")

    question = state["question"]
    documents = state["documents"]

    # Convert document strings back to Document objects if needed
    if documents and isinstance(documents[0], str):
        # Documents were serialized as strings, reconstruct
        docs = [Document(page_content=doc) for doc in documents]
    else:
        docs = documents

    # Initialize NVIDIA reranker
    # Model: nv-rerankqa-mistral-4b-v3 (optimized for Q&A reranking)
    reranker = NVIDIARerank(
        model="nvidia/nv-rerankqa-mistral-4b-v3",
        top_n=10  # Rerank top 20 → top 10
    )

    # Rerank documents based on relevance to question
    try:
        reranked_docs = reranker.compress_documents(
            documents=docs,
            query=question
        )

        print(f"Reranked {len(documents)} → {len(reranked_docs)} documents")

        # Update state with reranked documents (convert back to strings for serialization)
        return {
            **state,
            "documents": [doc.page_content for doc in reranked_docs]
        }

    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        print("Falling back to original retrieval order")

        # Fallback: just take top 10 from original retrieval
        top_10 = documents[:10] if len(documents) > 10 else documents

        return {
            **state,
            "documents": top_10
        }
