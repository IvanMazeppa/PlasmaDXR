"""
Grade Generation Node - Hallucination Detection
Validates that diagnostic response is grounded in provided documents
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from ..graph.state import GraphState
from ..models.binary_score_models import GradeGeneration


def grade_generation_node(state: GraphState) -> GraphState:
    """
    Grade diagnostic generation for hallucination

    Checks if the diagnosis is:
    - Grounded in provided documents (not hallucinated)
    - Answers the original question
    - Based on actual evidence vs speculation

    Args:
        state: Current graph state with generation

    Returns:
        Updated state with generation quality score
        State key 'hallucination_check' = 'yes' (good) or 'no' (hallucinated)
    """
    print("---GRADE GENERATION FOR HALLUCINATION---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Initialize NVIDIA LLM for grading
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",  # Fast, accurate for grading
        temperature=0.0  # Deterministic grading
    )

    # Create structured output chain
    grader = llm.with_structured_output(GradeGeneration)

    # Build context
    context = "\n\n---\n\n".join(documents[:10])

    # Hallucination detection prompt
    hallucination_prompt = f"""You are validating a diagnostic response for hallucination.

**Original Question:**
{question}

**Provided Evidence (Logs/PIX/Buffers):**
{context}

**Generated Diagnosis:**
{generation}

**Your Task:**
Determine if the diagnosis is grounded in the provided evidence or if it contains hallucinations/speculation.

**Grading Criteria:**
- 'yes' (grounded): Diagnosis cites specific log lines, metrics, or artifacts from evidence
- 'no' (hallucinated): Diagnosis includes claims not supported by evidence

**Examples of HALLUCINATION:**
- "DLSS-SR broke because of buffer dumping changes" (when evidence shows different timeline)
- "FPS is 45" (when logs show 120 FPS)
- "Shader compilation failed" (when evidence shows successful compilation)

**Examples of GROUNDED:**
- "Log shows DLSS-SR error at line 228" (cites specific log line)
- "PIX capture shows 0 active lights" (references actual PIX data)
- "Shader recompilation took 2.1ms per frame" (cites actual metrics)

Provide:
- binary_score: 'yes' if grounded, 'no' if hallucinated
- confidence: 0.0-1.0 (how confident you are in this grading)
"""

    try:
        grade: GradeGeneration = grader.invoke(hallucination_prompt)

        print(f"  Hallucination check: {grade.binary_score} (confidence: {grade.confidence:.2f})")

        if grade.binary_score == "no":
            print("  ⚠️ HALLUCINATION DETECTED - Diagnosis not grounded in evidence")
        else:
            print("  ✅ Diagnosis is grounded in provided evidence")

        # Update state with hallucination check result
        return {
            **state,
            "hallucination_check": grade.binary_score  # 'yes' = grounded, 'no' = hallucinated
        }

    except Exception as e:
        print(f"  ❌ Grading failed: {e}")
        print("  Assuming diagnosis is grounded (conservative approach)")

        return {
            **state,
            "hallucination_check": "yes"  # Conservative: assume OK on error
        }
