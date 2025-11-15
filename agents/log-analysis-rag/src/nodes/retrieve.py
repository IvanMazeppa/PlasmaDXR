"""
Retrieve Node - Hybrid Retrieval (BM25 + FAISS)
Retrieves relevant documents using ensemble retriever
"""

from typing import List
from ..graph.state import GraphState
from ..tools.hybrid_retriever import PlasmaDXHybridRetriever


# Initialize hybrid retriever (loaded once, reused across calls)
_retriever = None


def get_retriever() -> PlasmaDXHybridRetriever:
    """
    Get or initialize hybrid retriever (singleton pattern)
    """
    global _retriever

    if _retriever is None:
        print("Initializing hybrid retriever...")

        _retriever = PlasmaDXHybridRetriever(
            log_dirs=[
                "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/logs",
                "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/PIX/buffer_dumps"
            ],
            embedding_model="nvidia/nv-embedqa-e5-v5",
            top_k=20  # Retrieve top 20 for reranking
        )

        # Load documents and initialize retrievers
        _retriever.load_documents()

        print(f"✅ Retriever initialized with {len(_retriever.documents)} documents")

    return _retriever


def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve relevant documents using hybrid retrieval

    Uses ensemble of BM25 (keyword) + FAISS (semantic)
    Retrieves top 20 documents for reranking

    Args:
        state: Current graph state with question

    Returns:
        Updated state with retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")

    question = state["question"]

    # Get hybrid retriever
    retriever = get_retriever()

    # Retrieve documents
    try:
        docs = retriever.retrieve(question)

        print(f"Retrieved {len(docs)} documents")

        # Extract document content for state (serialize as strings)
        documents = [doc.page_content for doc in docs]

        return {
            **state,
            "documents": documents
        }

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")

        return {
            **state,
            "documents": []  # Empty on error
        }
