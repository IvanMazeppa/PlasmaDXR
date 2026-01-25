# CRITICAL AUDIT: LLM Attribute Hallucination in Script Writer

> Consolidated issue list: `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

**Date:** 2026-01-25
**Status:** CRITICAL - Blocking all progress
**Author:** Claude (Audit conducted with Ben)

---

## Executive Summary

The Script Writer agent is **hallucinating Blender API attribute names** instead of using them correctly. This is not a typo problem - it's a fundamental breakdown where the LLM generates plausible-sounding but non-existent attributes, causing 100% script failure rate.

**Evidence of hallucination (not typos):**
| Generated (WRONG) | Correct | Analysis |
|-------------------|---------|----------|
| `use_adaptive_time_steps` | `use_adaptive_timesteps` | Added underscore that doesn't exist |
| `resolution_divisions` | `resolution_max` | Completely different word |
| `use_dissolve` | `use_dissolve_smoke` | Truncated/simplified name |
| `absolute_density` | `density` or `use_absolute` | Combined two concepts incorrectly |
| `timesteps_per_frame` | `timesteps_maximum` | **NEW 2026-01-25** - doesn't exist |
| `time_scale` | `timesteps_maximum` or `cfl_condition` | **NEW 2026-01-25** - doesn't exist |
| `noise_scale = 1.0` | `noise_scale = 1` (int) | **NEW 2026-01-25** - type error |

These aren't typos - they're **semantically plausible fabrications**.

---

## Current Architecture

### Pipeline Flow
```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 0: Research Agent                                     │
│  Tools: blender_doc_search_bundle, list_patterns_by_effect  │
│  Output: Research summary with approaches                    │
│  ✓ DOES query documentation                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 0.5: Technique Selector (Coordinator)                │
│  Input: Research findings                                    │
│  Output: Selected technique + parameters                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Script Writer Agent                    ← PROBLEM  │
│  Tools: write_script, validate_script, modify_script,       │
│         semantic_search_blender_docs,                        │
│         search_blender_api_by_intent                         │
│  Output: Generated Python script                             │
│  ✗ HAS doc tools but DOESN'T USE THEM for code generation   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1.5: API Validator                        ← PROBLEM  │
│  Check: KNOWN_API_CHANGES dictionary only                    │
│  ✗ ASSUMES VALID if not in known-bad list                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Executor                                           │
│  Action: Run script in Blender                               │
│  Result: AttributeError (script fails)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## The Core Problem

### Problem 1: Script Writer Has Tools But Doesn't Use Them

**File:** `specialized_agents/script_writer.py` (lines 129-140)

```python
tools=[
    write_script,         # Save LLM-generated code
    validate_script,      # Check script (BROKEN - see Problem 2)
    modify_script,        # Modify existing scripts
    record_technique_outcome,
    # Documentation search tools ARE AVAILABLE:
    semantic_search_blender_docs,      # ← HAS THIS
    search_blender_api_by_intent,      # ← HAS THIS
]
```

The agent HAS documentation tools but the **instructions don't REQUIRE their use** before writing code.

**Evidence from trace:**
```
Tool calls: ['write_script', 'validate_script']
doc_query_made: False  ← NO DOC QUERY DURING SCRIPT WRITING
```

### Problem 2: Doc Query Enforcement Was DISABLED

**File:** `hooks/enforcement_hooks.py` (lines 503-507)

```python
require_doc_query_before=[
    # NOTE: write_script and generate_script NO LONGER require doc query
    # because Research Agent already did the documentation research
    # and the API Validator (Phase 1.5) catches API issues
],
```

**CRITICAL FLAW:** This assumes:
1. Research Agent's general docs lookup covers specific API attributes (FALSE)
2. API Validator catches API issues (FALSE - see Problem 3)

### Problem 3: API Validator Assumes Valid By Default

**File:** `specialized_agents/api_validator.py` (lines 673-694)

```python
for call in api_calls:
    validation = APICallValidation(
        api_call=call,
        is_valid=True,        # ← DEFAULT IS TRUE!
        confidence=0.5,       # Low confidence but still "valid"
    )

    # Only checks known issues in static dictionary
    for pattern, fix in KNOWN_API_CHANGES.items():
        if pattern in call:
            # Only HERE does it mark as invalid
```

**Result:** Any attribute NOT in `KNOWN_API_CHANGES` is assumed valid, including hallucinated ones.

### Problem 4: validate_script() Is Essentially Useless

**File:** `tools/script_generator_tools.py` (lines 303-374)

The `validate_script` function:
1. Checks Python syntax (AST parsing) ✓
2. Extracts some parameters ✓
3. Does NOT validate Blender API attribute names ✗
4. Does NOT query documentation ✗
5. Returns `valid: true` for scripts with hallucinated attributes ✗

**Evidence from trace:**
```json
{
  "valid": true,
  "issues": [],
  "extracted_params": {"vorticity": 0.2, "temperature": 1.0, ...},
  "error_count": 0,
  "warning_count": 0
}
```

This passed validation despite containing `absolute_density` which doesn't exist.

---

## Why Is The LLM Hallucinating?

### Hypothesis 1: Training Data Mismatch
The LLM (gpt-5-mini in quick_test preset) was trained on Blender 2.8x-4.x documentation. Blender 5.0 changed many API names. The LLM is generating what it "remembers" rather than what exists.

### Hypothesis 2: Semantic Plausibility Over Accuracy
The LLM is generating attribute names that "make sense":
- `use_adaptive_time_steps` - sounds right (time + steps)
- `absolute_density` - sounds right (absolute + density)
- `resolution_divisions` - sounds right (resolution divided)

But Blender's actual naming conventions don't follow this pattern.

### Hypothesis 3: Research Findings Not Connected to Code Generation
The Research Agent queries documentation and returns a **summary**, not exact API specifications. The Script Writer then generates code from:
1. The summary (general approach, not exact attributes)
2. Its training data (outdated)
3. Its "best guess" at attribute names

**The actual documentation is never consulted during code generation.**

---

## What Should Happen vs What Happens

### Expected Flow (What We Designed)
```
Research Agent → General approach + doc references
    ↓
Script Writer → Uses doc tools to verify EACH attribute before writing
    ↓
API Validator → Catches anything missed
    ↓
Executor → Runs validated script
```

### Actual Flow (What's Happening)
```
Research Agent → General approach (good)
    ↓
Script Writer → Writes code from memory, IGNORES doc tools
    ↓
API Validator → "Looks good to me!" (only checks known-bad list)
    ↓
Executor → AttributeError (script fails)
```

---

## Configuration Analysis

### Script Writer Instructions
**File:** `tools/dynamic_instructions.py`

The instructions include:
- A typo warning table (reactive, not preventive)
- Error patterns to avoid (reactive, not preventive)
- NO requirement to query docs before using an attribute

**Missing:** "Before using ANY bpy.types attribute, verify it exists using semantic_search_blender_docs"

### Model Settings
**File:** `config/presets.yaml`

```yaml
quick_test:
  description: "Fast iteration with minimal cost"
  agents:
    script_writer:
      model: "gpt-5-mini"  # ← Smaller model, more prone to hallucination?
      reasoning_effort: "medium"
```

**Question:** Would a larger model (gpt-5.2) hallucinate less? Or is this a prompt/tool usage issue?

### Guardrails
**File:** `guardrails/script_guardrails.py`

Current guardrails:
- `require_research_context` - Checks research was done (passes)
- `validate_effect_type` - Checks effect type is valid (passes)
- `validate_script_output` - Checks output has required fields (passes)

**Missing:** No guardrail verifies API attribute accuracy.

---

## Evidence Trail

### Test Run 1: `use_adaptive_time_steps`
```
Script Writer generates: dsettings.use_adaptive_time_steps = False
Correct attribute: use_adaptive_timesteps
validate_script returns: valid: true
API Validator: 11/11 valid (not in KNOWN_API_CHANGES)
Executor: AttributeError
```

### Test Run 2: `use_dissolve`
```
Script Writer generates: dsettings.use_dissolve = DISSOLVE_ENABLED
Correct attribute: use_dissolve_smoke
validate_script returns: valid: true
API Validator: 9/9 valid
Executor: AttributeError
```

### Test Run 3: `absolute_density`
```
Script Writer generates: flow_settings.absolute_density = False
Correct attribute: density (or use_absolute for the flag)
validate_script returns: valid: true
API Validator: 19/19 valid
Executor: AttributeError
```

**Pattern:** Every script passes validation but fails execution.

---

## Potential Solutions

### Solution 1: Re-enable Doc Query Enforcement (Quick)
**File:** `hooks/enforcement_hooks.py`

```python
require_doc_query_before=[
    "write_script",   # REQUIRE doc lookup before writing
    "modify_script",  # REQUIRE doc lookup before modifying
],
```

**Pros:** Immediate enforcement
**Cons:** May slow down, doesn't guarantee correct usage

### Solution 2: Flip Validation Default (Medium)
**File:** `specialized_agents/api_validator.py`

Change from "assume valid" to "flag unknown":
```python
validation = APICallValidation(
    api_call=call,
    is_valid=False,       # ← DEFAULT IS NOW FALSE
    confidence=0.0,       # Unknown = unverified
    notes="UNVERIFIED - not in known API list"
)
```

**Pros:** Surfaces all unchecked attributes
**Cons:** May generate many false positives

### Solution 3: Add Attribute Verification Tool (Comprehensive)
Create a new tool that MUST be called for each attribute:
```python
@function_tool
async def verify_blender_attribute(
    object_type: str,  # e.g., "FluidDomainSettings"
    attribute: str,    # e.g., "use_adaptive_timesteps"
) -> str:
    """Verify an attribute exists in Blender 5.0 API before using it."""
    result = await semantic_search_blender_docs(
        f"{object_type} {attribute} Blender 5.0"
    )
    # Return whether attribute exists and exact name if different
```

**Pros:** Precise verification
**Cons:** Significant implementation effort

### Solution 4: Build Blender 5.0 API Whitelist (Comprehensive)
Extract all valid attributes from Blender 5.0 docs and validate against whitelist.

**Pros:** Fast, accurate validation
**Cons:** Large maintenance burden as Blender updates

### Solution 5: Change Script Writer Instructions (Quick)
Add explicit requirement:
```
CRITICAL: Before writing ANY code that accesses bpy.types attributes:
1. Call semantic_search_blender_docs with the exact attribute name
2. Verify the attribute EXISTS in Blender 5.0
3. Use the EXACT name from documentation, not from memory
```

**Pros:** Direct instruction
**Cons:** LLM may still ignore/forget

---

## Questions Requiring Investigation

1. **Why doesn't the LLM use its available tools?**
   - Is it a prompt issue?
   - Is it a model capability issue (gpt-5-mini vs gpt-5.2)?
   - Is it a tool description issue?

2. **Is the Research Agent output too high-level?**
   - Does it provide exact attribute names or just general approaches?
   - Should it return structured API specifications?

3. **Should we use a different model for Script Writer?**
   - Would gpt-5.2 hallucinate less?
   - Would it use tools more reliably?

4. **Is there an SDK pattern we're missing?**
   - Does the OpenAI Agents SDK have patterns for "verify before act"?
   - Can guardrails intercept and verify code content?

---

## Recommended Next Steps

1. **Immediate:** Add explicit instruction to Script Writer requiring doc verification
2. **Short-term:** Re-enable `require_doc_query_before` enforcement
3. **Medium-term:** Create `verify_blender_attribute` tool with mandatory usage
4. **Long-term:** Build comprehensive Blender 5.0 API validation layer

---

## Fixes Applied (2026-01-25 Phase 3)

### API Validator Updates
**File:** `specialized_agents/api_validator.py`

Added to `KNOWN_API_CHANGES` dictionary:

```python
# LLM HALLUCINATED ATTRIBUTES (Phase 3: 2026-01-25)
".timesteps_per_frame": {
    "correction": ".timesteps_maximum",
    "reason": "HALLUCINATED: timesteps_per_frame does not exist."
},
".time_scale": {
    "correction": "# DELETE - use cfl_condition or timesteps_maximum",
    "reason": "HALLUCINATED: FluidDomainSettings.time_scale does not exist."
},

# TYPE WARNINGS (Phase 3: 2026-01-25)
"noise_scale = 1.0": {
    "correction": "noise_scale = 1  # Must be int, not float",
    "reason": "TYPE ERROR: noise_scale expects int, not float."
},
```

### VERSION_TRUTH.md Updates
Added to invalid attributes table:
- `timesteps_per_frame` → `timesteps_maximum`
- `time_scale` → `timesteps_maximum` or `cfl_condition`

Added new TYPE REQUIREMENTS section documenting integer-only attributes.

### SDK Reference (Mandatory)
Per SDK_ENFORCEMENT_PROTOCOL.md, all fixes must reference OpenAI Agents SDK patterns:
- SDK docs: https://github.com/openai/openai-agents-python/tree/main/docs
- Pattern used: `KNOWN_API_CHANGES` dict for auto-correction in validation phase
- RunHooks enforce doc queries before script writing (existing infrastructure)

---

## Appendix: Files Involved

| File | Role | Issue |
|------|------|-------|
| `specialized_agents/script_writer.py` | Creates Script Writer agent | Has tools, doesn't use them |
| `tools/dynamic_instructions.py` | Script Writer instructions | No verification requirement |
| `hooks/enforcement_hooks.py` | Enforcement configuration | Doc query disabled |
| `specialized_agents/api_validator.py` | API validation | Assumes valid by default |
| `tools/script_generator_tools.py` | `validate_script` function | Doesn't validate API names |
| `guardrails/script_guardrails.py` | Input/output guardrails | No API accuracy check |
| `orchestrator.py` | Pipeline orchestration | Trusts upstream validation |

---

## Conclusion

This is not a typo problem. This is a **validation architecture failure** where:
1. The system was designed to validate, but validation was disabled/neutered
2. The LLM has tools but isn't required to use them
3. The assumption "Research Agent covers it" is false for specific API attributes
4. The assumption "API Validator catches it" is false for unknown attributes

The result is 100% script failure rate due to hallucinated API attributes.

**This requires architectural intervention, not more whack-a-mole fixes.**
