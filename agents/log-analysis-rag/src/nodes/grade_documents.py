"""
Grade Documents Node - Binary Relevance Scoring
Filters documents by relevance to diagnostic question with confidence scoring
"""

from typing import List
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from ..graph.state import GraphState
from ..models.binary_score_models import GradeDocuments


def grade_documents_node(state: GraphState) -> GraphState:
    """
    Grade each document for relevance to the diagnostic question

    Uses binary scoring ('yes'/'no') with confidence and reasoning
    Filters out irrelevant documents before generation

    Args:
        state: Current graph state with reranked documents

    Returns:
        Updated state with only relevant documents
    """
    print("---GRADE DOCUMENTS FOR RELEVANCE---")

    question = state["question"]
    documents = state["documents"]

    # Initialize NVIDIA LLM for grading
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",  # Fast, accurate for grading
        temperature=0.0  # Deterministic grading
    )

    # Create structured output chain
    grader = llm.with_structured_output(GradeDocuments)

    # Grade each document
    relevant_docs = []
    total_confidence = 0.0

    for i, doc_content in enumerate(documents):
        print(f"\nGrading document {i+1}/{len(documents)}...")

        # Create grading prompt
        grading_prompt = f"""You are grading a log entry or diagnostic artifact for relevance to a PlasmaDX rendering issue.

Question: {question}

Document:
{doc_content[:500]}...

Does this document contain information relevant to answering the question?

Provide:
- binary_score: 'yes' if relevant, 'no' if not
- confidence: 0.0-1.0 (how confident you are)
- reasoning: Brief explanation of why it is/isn't relevant
"""

        try:
            grade: GradeDocuments = grader.invoke(grading_prompt)

            print(f"  Score: {grade.binary_score} (confidence: {grade.confidence:.2f})")
            if grade.reasoning:
                print(f"  Reason: {grade.reasoning}")

            # Keep document if relevant (and confidence > 0.5)
            if grade.binary_score == "yes" and grade.confidence >= 0.5:
                relevant_docs.append(doc_content)
                total_confidence += grade.confidence

        except Exception as e:
            print(f"  ⚠️ Grading failed: {e}")
            # On error, keep document (conservative approach)
            relevant_docs.append(doc_content)
            total_confidence += 0.6  # Assume moderate confidence

    # Calculate average confidence
    avg_confidence = (total_confidence / len(relevant_docs)) if relevant_docs else 0.0

    print(f"\n✅ Kept {len(relevant_docs)}/{len(documents)} documents")
    print(f"   Average confidence: {avg_confidence:.2f}")

    # Update state
    return {
        **state,
        "documents": relevant_docs,
        "confidence": avg_confidence  # Store avg confidence for later use
    }
