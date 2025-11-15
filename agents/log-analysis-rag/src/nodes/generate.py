"""
Generate Node - Diagnostic Response Generation
Produces structured diagnostic output from relevant documents
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

from ..graph.state import GraphState
from ..models.binary_score_models import DiagnosisOutput


def generate_node(state: GraphState) -> GraphState:
    """
    Generate diagnostic response from relevant documents

    Uses structured output (DiagnosisOutput) to ensure:
    - Clear diagnostic explanation
    - Confidence score
    - Recommended specialist agent (if escalation needed)
    - File:line references for code locations
    - Related artifact paths (screenshots, PIX, buffers)

    Args:
        state: Current graph state with relevant documents

    Returns:
        Updated state with diagnostic generation
    """
    print("---GENERATE DIAGNOSTIC RESPONSE---")

    question = state["question"]
    documents = state["documents"]

    # Initialize NVIDIA LLM for generation
    llm = ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",  # Most capable model for complex diagnostics
        temperature=0.2  # Slightly creative but mostly factual
    )

    # Create structured output chain
    diagnostic_chain = llm.with_structured_output(DiagnosisOutput)

    # Build context from documents
    context = "\n\n---\n\n".join(documents[:10])  # Max 10 docs to avoid token limits

    # Diagnostic prompt
    diagnostic_prompt = f"""You are a PlasmaDX rendering diagnostics expert analyzing logs, PIX captures, and buffer dumps.

**Diagnostic Question:**
{question}

**Relevant Evidence (Logs/PIX/Buffers):**
{context}

**Your Task:**
Provide a clear, actionable diagnostic explanation based on ONLY the evidence provided.

**Output Requirements:**
1. **diagnosis**: Concise explanation of the issue (2-4 sentences)
   - What is happening (symptom)
   - Why it's happening (root cause from evidence)
   - What to fix (actionable next step)

2. **confidence**: 0.0-1.0 confidence in this diagnosis
   - 0.9-1.0: Strong evidence (explicit errors, clear metrics)
   - 0.7-0.9: Good evidence (correlations, known patterns)
   - 0.5-0.7: Moderate evidence (hypotheses, indirect signals)
   - <0.5: Weak evidence (insufficient data)

3. **recommended_specialist**: Which specialist agent should handle this (or null if no escalation)
   - "rt-lighting-engineer": RayQuery lighting issues
   - "rt-shadow-engineer": Shadow ray problems
   - "sampling-and-distribution": RTXDI/ReSTIR issues
   - "dxr-image-quality-analyst": Visual quality/LPIPS problems
   - "pix-debugger": GPU buffer/resource issues
   - "material-system-engineer": Particle struct/material issues
   - null: Can be handled without specialist

4. **file_line_refs**: List of code locations mentioned (format: "file.cpp:123")
   - Extract from log entries like "Application.cpp:1880"
   - Extract from shader references like "particle_gaussian_raytrace.hlsl"

5. **artifact_paths**: Related files mentioned (screenshots, PIX captures, buffer dumps)
   - Extract from log paths like "logs/PlasmaDX-Clean_20251114_161707.log"
   - Extract PIX paths like "PIX/Captures/RTXDI_6.wpix"

**Critical Rules:**
- Base diagnosis ONLY on provided evidence
- If evidence is unclear, lower confidence score
- Use brutal honesty (per CLAUDE.md feedback philosophy)
- Cite specific log lines or metrics when possible
"""

    try:
        diagnosis: DiagnosisOutput = diagnostic_chain.invoke(diagnostic_prompt)

        print(f"\n✅ Diagnosis generated")
        print(f"   Confidence: {diagnosis.confidence:.2f}")
        print(f"   Specialist: {diagnosis.recommended_specialist or 'None'}")
        print(f"   File refs: {len(diagnosis.file_line_refs)}")
        print(f"   Artifacts: {len(diagnosis.artifact_paths)}")

        # Update state with generation
        return {
            **state,
            "generation": diagnosis.diagnosis,
            "confidence": diagnosis.confidence,
            "recommended_specialist": diagnosis.recommended_specialist,
            "file_line_refs": diagnosis.file_line_refs,
            "artifact_paths": diagnosis.artifact_paths
        }

    except Exception as e:
        print(f"❌ Generation failed: {e}")

        # Fallback: simple text generation without structure
        fallback_response = f"Unable to generate structured diagnosis. Error: {str(e)}\n\nQuestion: {question}\n\nAvailable evidence: {len(documents)} documents"

        return {
            **state,
            "generation": fallback_response,
            "confidence": 0.0,
            "recommended_specialist": None,
            "file_line_refs": [],
            "artifact_paths": []
        }
