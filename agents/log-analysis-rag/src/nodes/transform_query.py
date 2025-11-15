"""
Transform Query Node - Self-Correction Loop
Rewrites diagnostic question when documents are irrelevant or generation hallucinated
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

from ..graph.state import GraphState


def transform_query_node(state: GraphState) -> GraphState:
    """
    Transform/rewrite diagnostic question for better retrieval

    Triggered when:
    - Documents were irrelevant (failed grade_documents)
    - Generation hallucinated (failed grade_generation)

    Strategy:
    - Expand acronyms (RTXDI → "RTX Direct Illumination weighted reservoir sampling")
    - Add context (DLSS → "DLSS Super Resolution upscaling")
    - Rephrase for semantic search

    Args:
        state: Current graph state with original question

    Returns:
        Updated state with transformed question
    """
    print("---TRANSFORM QUERY (SELF-CORRECTION)---")

    original_question = state["question"]

    # Initialize NVIDIA LLM for query transformation
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",  # Fast, good at rewriting
        temperature=0.3  # Slightly creative for query expansion
    )

    # Query transformation prompt
    transform_prompt = f"""You are rewriting a PlasmaDX diagnostic question to improve retrieval from logs/PIX/buffers.

**Original Question:**
{original_question}

**Why Rewrite is Needed:**
The original question didn't retrieve relevant documents. Possible reasons:
- Too vague or generic
- Uses acronyms not in logs
- Missing technical context

**Rewriting Strategy:**
1. **Expand acronyms**: RTXDI → "RTX Direct Illumination weighted reservoir sampling"
2. **Add technical context**: DLSS → "DLSS Super Resolution upscaling (NGX SDK)"
3. **Include file/component names**: "shadows" → "PCSS soft shadows (shadow_systems.cpp)"
4. **Add error patterns**: "broken" → "error, failed, warning, crash"

**Examples:**

Original: "Why is DLSS broken?"
Rewritten: "DLSS Super Resolution (NGX SDK) initialization failed or not supported error in DLSSSystem.cpp"

Original: "RTXDI looks wrong"
Rewritten: "RTX Direct Illumination (RTXDI) weighted reservoir sampling visual artifacts patchwork pattern temporal accumulation M5"

Original: "Shadows are flickering"
Rewritten: "PCSS soft shadows temporal filtering instability flickering shadow rays shadow_systems ParticleRenderer_Gaussian"

**Your Task:**
Rewrite the original question to:
- Be more specific and technical
- Expand acronyms and add context
- Include likely file/component names
- Use error pattern keywords

Provide ONLY the rewritten question (no explanation).
"""

    try:
        rewritten = llm.invoke(transform_prompt).content.strip()

        print(f"\n  Original: {original_question}")
        print(f"  Rewritten: {rewritten}")

        # Update state with transformed question
        return {
            **state,
            "question": rewritten
        }

    except Exception as e:
        print(f"  ❌ Query transformation failed: {e}")
        print("  Keeping original question")

        return state  # No change on error
