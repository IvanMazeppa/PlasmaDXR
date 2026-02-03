# Hybrid Search & Hallucination Fix Analysis

**Date:** 2026-02-02
**Status:** Implementation in Progress
**Author:** Claude (Analysis conducted with Ben)

---

## Executive Summary

The blender-vfx-orchestrator has a critical API hallucination problem where the LLM generates plausible-sounding but non-existent Blender 5.0 attributes. This document analyzes the root causes and proposes a hybrid search solution combining semantic and keyword retrieval with mandatory verification gates.

---

## Problem Analysis

### Current Defense Layers (5 Layers, All Failing)

| Layer | Implementation | Current State | Issue |
|-------|----------------|---------------|-------|
| **1. Vector Store Search** | `semantic_docs_tools.py` | Implemented | Results can be ignored by LLM |
| **2. Static Whitelist** | `api_spec_guardrails.py` | 100+ entries | Only catches known patterns |
| **3. Dynamic Instructions** | `dynamic_instructions.py` | Implemented | Instructions don't force compliance |
| **4. Enforcement Hooks** | `enforcement_hooks.py` | **DISABLED** | Doc query requirement was turned off |
| **5. API Fixer** | `blender_api_fixer.py` | 50+ regex rules | Reactive, not preventive |

### Evidence from Error Logs

Found **50+ unique AttributeErrors** in `build/blender_cli_logs/`:

| Generated (WRONG) | Correct | Frequency |
|-------------------|---------|-----------|
| `resolution_divisions` | `resolution_max` | 5+ occurrences |
| `use_adaptive_time_steps` | `use_adaptive_timesteps` | 3+ occurrences |
| `use_dissolve` | `use_dissolve_smoke` | 2+ occurrences |
| `bake_frame_start` | N/A (doesn't exist) | 2+ occurrences |
| `timesteps_maximum` | N/A (hallucinated) | 2+ occurrences |
| `absolute_density` | `density` + `use_absolute` | 2+ occurrences |
| `velocity_factor_normal` | `velocity_normal` | 2+ occurrences |
| `cache_format` | `cache_data_format` | 1+ occurrences |
| `adaptive_domain` | `use_adaptive_domain` | 2+ occurrences |
| `noise_res_factor` | N/A (removed in 5.0) | 1+ occurrences |

These are **semantically plausible fabrications**, not typos.

### Root Cause Analysis

#### 1. Enforcement Hooks Were Disabled

From `hooks/enforcement_hooks.py` (lines 503-507):
```python
require_doc_query_before=[
    # NOTE: write_script and generate_script NO LONGER require doc query
    # because Research Agent already did the documentation research
    # and the API Validator (Phase 1.5) catches API issues
],
```

**CRITICAL FLAW:** This assumes:
1. Research Agent's general docs lookup covers specific API attributes (FALSE)
2. API Validator catches all API issues (FALSE - it assumes valid by default)

#### 2. API Validator Assumes Valid By Default

From `specialized_agents/api_validator.py`:
```python
for call in api_calls:
    validation = APICallValidation(
        api_call=call,
        is_valid=True,        # ← DEFAULT IS TRUE!
        confidence=0.5,
    )
```

Any attribute NOT in `KNOWN_API_CHANGES` dictionary is assumed valid.

#### 3. Single-Retrieval Weakness

Current vector search returns results but doesn't:
- Cross-verify with exact keyword matching
- Gate generation based on verification confidence
- Force the LLM to cite specific search results

---

## Proposed Solution: Hybrid Search with Mandatory Verification

### Architecture: Triple-Verification Pipeline

```
API Call from LLM
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Vector Store Semantic Search                       │
│  Query: "FluidDomainSettings.resolution_max Blender 5.0"    │
│  Returns: Ranked results with similarity scores              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Keyword/BM25 Search                                │
│  Query: Exact attribute name with quotes                     │
│  Returns: Exact matches from API reference                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Reciprocal Rank Fusion (RRF)                      │
│  Combines semantic + keyword results                         │
│  GATE: If attribute not in TOP 3 of fused results → REJECT  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  VERIFIED ✓ / REJECTED ✗
```

### Reciprocal Rank Fusion (RRF) Formula

```
RRF_score(d) = Σ (1 / (k + rank_i(d)))
```

Where:
- `k` = 60 (constant to prevent over-weighting top results)
- `rank_i(d)` = rank of document d in result list i

---

## Implementation Plan

### Priority 0 (Immediate - Low Effort, High Impact)

#### P0.1: Re-enable Doc Query Enforcement

**File:** `hooks/enforcement_hooks.py`

```python
require_doc_query_before=[
    "write_script",
    "generate_script",
    "modify_script",
]
```

#### P0.2: Add Negative Examples to Script Writer

**File:** `specialized_agents/script_writer.py`

Add few-shot examples showing correct behavior when attributes don't exist:

```python
FEW_SHOT_NEGATIVE_EXAMPLES = """
<example type="hallucination_prevention">
User: Set resolution_divisions to 128
❌ WRONG: domain_settings.resolution_divisions = 128
✅ CORRECT: I searched for "resolution_divisions" and it does NOT exist.
   The correct attribute is "resolution_max" (found in bpy.types.FluidDomainSettings)
   domain_settings.resolution_max = 128
</example>
...
"""
```

### Priority 1 (Medium Effort, High Impact)

#### P1.1: Implement Hybrid Search Verification Function

**New function in:** `tools/semantic_docs_tools.py`

```python
async def verify_api_attribute(
    attribute: str,
    parent_class: str,
    similarity_threshold: float = 0.75
) -> dict:
    """
    Hybrid verification using RRF fusion.
    """
    # Stage 1: Semantic search
    semantic_results = _search_vector_store(...)

    # Stage 2: Exact keyword search
    keyword_results = _search_vector_store(exact_query, ...)

    # Stage 3: RRF Fusion
    fused = reciprocal_rank_fusion([semantic_results, keyword_results])

    # Gate decision
    return {"exists": bool, "confidence": float, "correct_name": str | None}
```

#### P1.2: Add Pre-Generation Verification Gate

**New phase in:** `orchestrator.py`

Insert between Technique Selection and Script Writing:

```python
async def pre_generation_verification(
    effect_type: str,
    technique: str,
    proposed_attributes: List[str]
) -> Dict:
    """GATE: Verify all proposed attributes BEFORE code generation."""
```

### Priority 2 (Medium Effort, High Impact)

#### P2.1: Structured Output with VerifiedAttribute Schema

Force Script Writer to use Pydantic models that validate attributes:

```python
class VerifiedAttribute(BaseModel):
    name: str
    parent_class: str
    verification_status: Literal["verified", "rejected"]
    verification_source: str  # doc path that confirms existence

class ScriptOutput(BaseModel):
    verified_attributes: List[VerifiedAttribute]
    script_code: str
```

### Priority 3 (High Effort, Medium Impact)

#### P3.1: Self-Consistency Sampling

Generate 3 script variations and keep only API calls that appear in 2+ versions:

```python
async def generate_with_consistency_check(prompt: str, num_samples: int = 3):
    scripts = [await generate_script(prompt) for _ in range(num_samples)]
    consensus_apis = find_consensus_api_calls(scripts)
    return regenerate_with_constraints(prompt, allowed_apis=consensus_apis)
```

---

## Implementation Status

| Item | Status | Date | Notes |
|------|--------|------|-------|
| P0.1: Re-enable enforcement hooks | ✅ DONE | 2026-02-02 | `create_fallback_script_writer_hooks()` now requires doc query |
| P0.2: Add negative examples | ✅ DONE | 2026-02-02 | Added 9 more examples (7-15) covering real AttributeErrors |
| P1.1: Hybrid search verification | PLANNED | | |
| P1.2: Pre-generation gate | PLANNED | | |
| P2.1: Structured VerifiedAttribute | PLANNED | | |
| P3.1: Self-consistency sampling | PLANNED | | |

### P0 Implementation Details

#### P0.1: Enforcement Hook Fix
**File:** `hooks/enforcement_hooks.py`
**Function:** `create_fallback_script_writer_hooks()`
**Change:**
- Added `require_doc_query_before=["write_script", "modify_script"]`
- Changed `raise_on_doc_missing=True` (was False)
- Added comprehensive docstring explaining the hallucination problem

#### P0.2: Negative Examples Added
**File:** `tools/dynamic_instructions.py`
**Added Examples 7-15:**
- Example 7: `adaptive_domain` → `use_adaptive_domain`
- Example 8: `cache_format` → `cache_data_format`
- Example 9: `timesteps_maximum` (HALLUCINATED - doesn't exist)
- Example 10: `bake_frame_start` (doesn't exist)
- Example 11: `use_high_resolution` (doesn't exist)
- Example 12: `use_caching` (REMOVED in 5.0)
- Example 13: `noise_res_factor` (REMOVED in 5.0)
- Example 14: `velocity_factor_normal` → separate attrs
- Example 15: `Scene.node_tree` access pattern

---

## Testing Plan

### P0 Validation

After implementing P0 changes:

1. Run `test_candle_3iter.py` - should see doc query calls before write_script
2. Check traces for `doc_query_made: True` before script generation
3. Monitor for reduction in hallucinated attributes

### P1 Validation

After implementing hybrid search:

1. Test with known-bad attributes (resolution_divisions, use_dissolve)
2. Verify RRF fusion returns correct suggestions
3. Measure verification accuracy vs. ground truth

---

## Budget Considerations

| Operation | Cost per call | Calls per iteration |
|-----------|---------------|---------------------|
| Vector store search | ~$0.001 | 3-5 |
| Hybrid verification | ~$0.002 | 5-10 (doubled) |
| Script generation | ~$0.05 | 1 |

**Estimated increase:** ~$0.01 per iteration (acceptable within $20/month budget)

---

## References

- `docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md` - Original audit
- `docs/research/PRACTICAL_HALLUCINATION_FIXES.md` - Technique research
- `docs/SDK_ENFORCEMENT_PROTOCOL.md` - SDK patterns
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Fusion algorithm
