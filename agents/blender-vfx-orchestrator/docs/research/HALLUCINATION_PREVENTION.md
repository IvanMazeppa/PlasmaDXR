# Agent Hallucination Prevention

**Source:** [LLM-based Agents Suffer from Hallucinations: A Survey](https://arxiv.org/html/2509.18970v1)
**Relevance:** Directly addresses why ScriptWriter hallucinates Blender API attributes

---

## Hallucination Taxonomy for Agents

### 1. Execution Hallucinations (YOUR MAIN PROBLEM)

Agents claim task completion without actual performance. Includes:

| Subtype | Description | Example in Your System |
|---------|-------------|------------------------|
| **Tool Selection Errors** | Agent calls non-existent tools | N/A (tools are fixed) |
| **Tool Calling Mistakes** | Incorrect parameters to valid tools | `noise_scale = 2.0` (float instead of int) |
| **API Fabrication** | Agent invents API attributes | `velocity_multi` instead of `velocity_factor` |

### 2. Reasoning Hallucinations

Agents generate "seemingly plausible plans which are logically flawed."

| Subtype | Description | Example |
|---------|-------------|---------|
| Goal understanding errors | Vague specifications misinterpreted | "Add turbulence" → wrong attribute |
| Intention decomposition failures | Omissions, redundancy, disorder | Missing `domain_type = 'GAS'` |
| Planning generation errors | Steps that don't achieve goal | Setting attribute on wrong object |

### 3. Memorization Hallucinations

Agents rely on "outdated, fabricated, or conflated memories."

**Critical for Your System:** LLM training data contains Blender 2.8-4.x APIs. When it can't find exact match, it fabricates plausible-sounding attributes.

### 4. Communication Hallucinations (Multi-Agent)

In multi-agent systems, agents exchange "inaccurate, misleading, or fabricated" information.

**Risk:** If Coordinator passes hallucinated API path to ScriptWriter, the hallucination propagates.

---

## Self-Correction Strategies

### 1. Self-Questioning (Recommended for API Validation)

Agent poses verification questions about its own reasoning:

```
Before using attribute X:
1. "Have I verified X exists in Blender 5.0 docs?"
2. "What is the expected type of X?"
3. "What class does X belong to?"
```

**Implementation:** Add to ScriptWriter system prompt:
```
Before writing any bpy attribute access:
- STOP and ask: "Have I verified this attribute exists?"
- If not verified via semantic_search_blender_docs(), DO NOT USE IT
```

### 2. Self-Reflection

Enable agents to revisit and critique their own outputs:

```python
# After script generation, have ScriptWriter review:
"Review the script I just generated. For each bpy.* call:
1. Is this a real Blender 5.0 API?
2. What is my confidence (0-100)?
3. If confidence < 80, mark for verification"
```

### 3. Self-Consistency

Generate multiple candidate outputs and aggregate via voting:

```python
# Generate 3 script variations
scripts = [await generate_script(prompt) for _ in range(3)]

# Extract API calls from each
api_calls = [extract_api_calls(s) for s in scripts]

# Keep only APIs that appear in 2+ scripts
consensus_apis = get_majority_vote(api_calls)
```

---

## Grounding Techniques

### 1. External Knowledge Guidance

**Your Current Approach:** Vector store search (semantic_search_blender_docs)

**Problem:** Agent can ignore search results or not search at all.

**Solution:** Make search results authoritative:
```
CRITICAL: The semantic_search results are the ONLY source of truth.
If an attribute is NOT in the search results, it DOES NOT EXIST.
Never infer or guess attributes based on naming patterns.
```

### 2. Rule-Based Constraints

Add hard rules that override LLM "creativity":

```python
BLENDER_5_RULES = {
    "FluidDomainSettings": {
        "banned": ["resolution_divisions", "time_scale", "velocity_multi"],
        "required_type": {"noise_scale": int, "resolution_max": int},
    }
}
```

### 3. World Model Grounding

Provide foundational knowledge about Blender's structure:

```
BLENDER 5.0 STRUCTURE:
- All fluid settings are in FluidDomainSettings, NOT FluidSettings
- Noise attributes: noise_scale (int), noise_strength (float)
- Velocity attributes: velocity_factor (float 0-1000), NOT velocity_multi
```

---

## Multi-Agent Approaches

### 1. Communication Protocol Standardization

Use structured formats (JSON/Pydantic) over natural language:

```python
class APISpec(BaseModel):
    """Structured API specification passed between agents."""
    domain_attributes: List[VerifiedAttribute]
    flow_attributes: List[VerifiedAttribute]

class VerifiedAttribute(BaseModel):
    name: str
    type: Literal["int", "float", "bool", "str", "enum"]
    verified_source: str  # Doc URL or "HALLUCINATION_WARNING"
```

### 2. Collaborative Verification

Multiple agents verify each other's outputs:

```
Pipeline with verification:
1. ScriptWriter generates script
2. APIValidator extracts API calls
3. DocsExpert verifies each call against vector store
4. If ANY call unverified → reject script, provide corrections
```

### 3. Cross-Agent Checking

```python
# In Coordinator, before passing to Executor:
api_calls = extract_api_calls(script)
for call in api_calls:
    verified = await docs_expert.verify(call)
    if not verified:
        return f"REJECTED: Unverified API call: {call}"
```

---

## Implementation Recommendations

### Immediate (P0)

1. **Tool-level guardrail** on `modify_script` and `generate_script`:
   - Extract all bpy.* calls
   - Check against KNOWN_API_CHANGES
   - Block execution if unverified calls found

2. **Output guardrail** on ScriptWriter:
   - Parse generated script
   - Verify ALL API calls against vector store
   - Tripwire if >0 unverified calls

### Short-term (P1)

3. **Self-questioning prompt injection**:
   - Before each API usage, agent must state verification status
   - Format: `# VERIFIED: noise_scale (int) - source: FluidDomainSettings docs`

4. **Confidence scoring**:
   - ScriptWriter outputs confidence per API call
   - Low confidence triggers automatic doc search

### Medium-term (P2)

5. **Multi-agent verification chain**:
   - Dedicated VerificationAgent between ScriptWriter and Executor
   - Hard gate: no unverified scripts reach Executor
