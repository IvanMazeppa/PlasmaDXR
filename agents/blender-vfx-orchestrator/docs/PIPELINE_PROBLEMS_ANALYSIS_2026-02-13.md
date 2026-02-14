# Pipeline Problems Analysis — 2026-02-13

**Purpose:** Comprehensive analysis of current pipeline failures, their relationship to GPT-5.3-XHigh's P0-P06 changes, and fundamental architectural concerns. Intended as input for the next round of deep analysis.

**Author:** Claude Opus 4.6 (hands-on debugging across 4 sessions, ~20 test runs)

---

## Executive Summary

The pipeline has two categories of problems:

1. **Fundamental (pre-existing):** LLM hallucination of Blender APIs is the #1 systemic risk. Every agent in the pipeline — Research, Script Writer, Code Writer, API Spec — generates Blender attribute names from training data. Training data is Blender 4.x. We run Blender 5.0. Dozens of attributes changed names. This single fact has caused more wasted iterations, false failures, and debugging hours than all other issues combined.

2. **Introduced by GPT-5.3 P0-P06 changes:** The strict doc-grounding guardrails are conceptually correct but reject valid pipeline output. The research guardrail demands `bpy.types.*` formatted doc_refs, but the vector store returns filenames. The `validate_code_against_spec` guardrail tripped on the second test run, aborting a valid 846-line script. The DIAGNOSE/FIX phases crash on SharedContext Pydantic errors. The `_apply_script_modifications` method awaits a FunctionTool object instead of calling the underlying function.

**The pipeline cannot currently complete a single clean iteration without hitting at least one of these failures.**

---

## Part 1: The Hallucination Problem (CRITICAL — READ THIS FIRST)

### The Core Issue

LLMs are trained on Blender 4.x documentation and StackOverflow answers. Blender 5.0 renamed, removed, or restructured many attributes. When an LLM writes `resolution_divisions`, `use_adaptive_time_steps`, `timesteps_per_frame`, `ShaderNodeMixRGB`, `BLENDER_EEVEE_NEXT`, or `Subsurface Color`, it is hallucinating — these don't exist in Blender 5.0 and cause silent failures or crashes.

### What We've Built to Combat This

| Layer | Mechanism | Status |
|-------|-----------|--------|
| **Instructions** | CLAUDE.md and dynamic_instructions.py list wrong→correct mappings | Working but LLMs ignore instructions when training data is strong |
| **API Fixer** | `blender_api_fixer.py` — regex-based find-and-replace of known bad patterns | Working, catches ~15 known patterns |
| **API Validator** | `api_validator.py` — checks generated code against known attribute list | Working but the known list is incomplete |
| **API Spec Agent** | Searches vector store docs to build verified attribute spec | Working but slow (2+ min), and doc_refs are filenames not API paths |
| **Vector Store** | OpenAI vector store with Blender 5.0 Python reference docs | Working but search returns filenames, not `bpy.types.X` paths |
| **Enforcement Hooks** | Require doc query before write_script | Working but blocks error recovery |

### Why This Is Still Not Solved

1. **The API Fixer only catches KNOWN patterns.** Every new script introduces novel hallucinations we haven't seen before. The fixer has 15 patterns; there are hundreds of possible wrong attributes.

2. **The API Spec Agent verifies attributes BEFORE script generation**, but the Code Writer still uses training-data attributes not in the spec. The spec says "use `resolution_max`" but the LLM writes `resolution_divisions` anyway because it's deeply embedded in training data.

3. **Instructions are ignored when training signal is strong.** We added a prominent `## CRITICAL: primitive_cube_add SCALE MATH — DO NOT HALVE DIMENSIONS` section. The LLM generated `* 0.5` on every single scale value anyway. This is not a prompt engineering problem — it's a fundamental limitation of instruction-following vs. training data.

4. **The vector store returns filenames, not structured API paths.** When the Research Agent queries docs, it gets back references like `b26_blender_python_reference_5_0_genindex-all_html_f_1.md`. These ARE valid docs containing the right information, but GPT-5.3's strict guardrail rejects them because they don't match `bpy.types.*` format.

### The Scale Math Example (Illustrative)

This is the most persistent bug across 10+ test runs:

```python
# LLM ALWAYS writes this (from Blender 4.x training data):
floor.scale = (CAB_W * 0.5, CAB_D * 0.5, PANEL_T * 0.5)

# Correct for Blender's primitive_cube_add(size=1.0):
floor.scale = (CAB_W, CAB_D, PANEL_T)
```

With `size=1.0`, `obj.scale = (W, D, H)` gives visual dimensions (W, D, H). The `* 0.5` halves everything, creating "exploded geometry" — gaps between walls, floors, ceilings, emitters outside domains. We added an instruction. LLM ignored it. We then added a **deterministic API fixer** (`_fix_cube_scale_halving`) that strips `* 0.5` from scale tuples. This is the only reliable approach: don't ask the LLM to do it right — fix it after the fact.

### Recommendation: Consider a Fundamentally Different Approach

The current approach is:
1. Tell the LLM what's correct (instructions)
2. Let the LLM generate code
3. Check for known-bad patterns (API fixer, validator)
4. Hope we caught everything

This is a losing game because hallucinations are UNBOUNDED. We can't enumerate every wrong attribute name. Possible alternatives:

**A. Whitelist-only code generation:** Instead of blacklisting bad attributes, ONLY allow attributes that appear in a verified whitelist. Any attribute not in the whitelist is rejected. This inverts the burden — instead of catching infinite bad patterns, we define the finite set of good patterns.

**B. Template-based generation:** Don't generate Blender API code from scratch. Provide the LLM with working template scripts for each effect type (fire, water, smoke, etc.) and have it ONLY modify parameters. The API calls are pre-written and verified. The LLM fills in values, not code.

**C. Two-phase with strict compilation:** Generate code, then compile/execute in a Blender sandbox that reports ALL attribute errors. Parse errors, fix, repeat. This already partially exists (the executor) but the feedback loop is too slow (full bake + render per attempt).

**D. AST-level validation:** Parse the generated Python code's AST, extract all `bpy.*` attribute accesses, and validate each one against the Blender 5.0 API reference. Reject the script if ANY unverified attribute is found. This is more robust than regex-based checking.

---

## Part 2: GPT-5.3 P0-P06 Changes — Impact Assessment

### What GPT-5.3 Changed (from DEEP_ANALYSIS_INVESTIGATION_2026-02-12.md)

| Change | Files Modified | Status |
|--------|---------------|--------|
| P0-1: DIAGNOSE/FIX phases | orchestrator.py | **CRASHES** — assigns to non-existent SharedContext fields |
| P0-2: Strict doc grounding | research_guardrails.py, semantic_docs_tools.py | **REJECTS VALID OUTPUT** — vector store returns filenames, not API paths |
| P0-3: Spec-First re-enabled | orchestrator.py | **WORKS** when guardrails don't trip; generates better code |
| P0-4: `_apply_script_modifications` | orchestrator.py | **CRASHES** — awaits FunctionTool object (not callable) |
| P0-5: Strict grounding abort | orchestrator.py | **TOO AGGRESSIVE** — RuntimeError aborts entire run |
| P0-6: Artifact manager additions | utils/artifact_manager.py | **WORKS** — DiagnosisArtifact and FixArtifact dataclasses |

### Detailed Bugs in GPT-5.3's Changes

#### Bug 1: SharedContext Pydantic Crash (P0-1)
**File:** `orchestrator.py:3268, 3594`
```python
context.last_execution_diagnosis = execute_diagnosis  # Line 3268
context.pending_gate_fix_instructions = gate_fix_steps  # Line 3594
```
`SharedContext` is a strict Pydantic `BaseModel`. These fields didn't exist. **Fixed by Claude** — added `last_execution_diagnosis` and `pending_gate_fix_instructions` as `Optional[Any]` fields.

#### Bug 2: FunctionTool Await (P0-4)
**File:** `orchestrator.py:838`
```python
result_raw = await modify_script(
    script_path=script_path,
    modifications_json=json.dumps(modifications),
    output_name=output_name,
)
```
`modify_script` is decorated with `@function_tool` which returns a `FunctionTool` object — NOT an awaitable function. This crashes at runtime on iteration 2+. **Fixed by Claude** — changed to call `_modify_script_impl()` directly (sync function).

**IMPORTANT NOTE:** This fix is the OPPOSITE of what GPT-5.3's Finding #2 recommended. Finding #2 said "direct `_impl` calls bypass hooks/guardrails" and recommended routing everything through the tool wrapper. But the tool wrapper (`@function_tool`) creates an SDK tool descriptor, not a callable function. You CANNOT `await` it from Python code. The correct solution is either:
- Call `_impl` directly (current fix — fast but bypasses hooks)
- Run the agent with the tool via `Runner.run()` (correct but adds an entire agent invocation per modification)

#### Bug 3: Research Guardrail Too Strict (P0-2)
**File:** `research_guardrails.py`
The guardrail requires doc_refs to match patterns like `bpy.types.FluidDomainSettings` or `bpy.ops.fluid.bake_all`. But the vector store returns filenames like `b26_blender_python_reference_5_0_genindex-all_html_f_1.md`. These ARE the docs — they contain the API reference content — but the guardrail rejects them because the filename format doesn't match.

**Result:** Research Agent output is REJECTED on every single run. The pipeline proceeds without research findings, degrading script quality.

#### Bug 4: Code Writer Guardrail Intermittent Failure
The `validate_code_against_spec` guardrail tripped on the second test run (score 0.0, script never executed). On the first test run it passed the same guardrail. The guardrail only checks for deprecated attributes — so either the second run's script contained a deprecated attribute that the first didn't, or there's a non-deterministic element.

### What GPT-5.3 Got Right

1. **Spec-First pipeline re-enablement (P0-3):** When it works, it produces significantly better code. The API Spec Agent verified 25-30 attributes across 2 test runs. Code Writer generated 846-line scripts with zero deprecated attributes.

2. **`time_scale` un-banned:** `time_scale` IS a valid Blender 5.0 attribute. It was incorrectly in the hallucination blacklist.

3. **DIAGNOSE/FIX conceptual design (P0-1):** The idea of structured diagnosis artifacts before fix attempts is sound. The implementation just had the Pydantic field bug.

4. **Artifact manager additions (P0-6):** DiagnosisArtifact and FixArtifact dataclasses are clean and useful.

---

## Part 3: Current Test Results Summary

### Test Run 1 (with GPT-5.3 changes, pre-Claude fixes)
- Research: REJECTED by guardrail (filename doc_refs)
- API Spec: PASSED (25 attrs, 5 ops)
- Code Writer: 846-line script, PASSED guardrail
- Execution: Blender exit_code=0, BUT pipeline reports "Unknown execution error" (executor false-positive)
- DIAGNOSE phase: CRASHED on `context.last_execution_diagnosis` (Pydantic)
- Score: 0.0

### Test Run 2 (with Claude's SharedContext + FunctionTool + scale fixes)
- Research: REJECTED by guardrail (same issue)
- API Spec: PASSED (30 attrs, 2 ops)
- Code Writer: Generated script, then error recovery produced 755-line script
- Execution: Blender exit_code=0, render produced (`kitchen_leak_beauty_frame_40.png`)
- Recovery pipeline WORKED — produced render + cache (1.54MB, 100 VDBs)
- Artifact gates: ALL PASSED
- Quality score: **28.0/100** — "water effect not readable, no convincing spray/puddle"
- Scale `* 0.5` still present in ALL generated scripts (instruction ignored)

### Test Run 3 (with scale halving API fixer added)
- Research: REJECTED by guardrail (same issue)
- API Spec: PASSED (30 attrs, 2 ops)
- Code Writer: Generated script, REJECTED by `validate_code_against_spec` guardrail
- Score: 0.0 (script never executed)

---

## Part 4: Executor False-Positive ("Unknown Execution Error")

This is a pre-existing bug, NOT from GPT-5.3's changes. The Executor agent receives `execute_blender_script` tool results showing `success=true, exit_code=0`, but its LLM response is parsed as a failure. The pipeline then enters error recovery unnecessarily.

**Root cause:** The Executor agent's structured output format doesn't always match the pipeline's parsing expectations. When the agent describes the output verbosely (e.g., listing files found), the pipeline's regex/JSON extraction fails to find the success indicator.

**Impact:** Wastes ~3 minutes on error recovery for scripts that already succeeded. On test run 2, the "recovery" script was essentially a rewrite that also worked — but it burned time and API budget.

---

## Part 5: Prioritized Fix List

### P0: Must Fix (Pipeline Cannot Function)

1. **Research guardrail doc_ref validation** — Either relax the regex to accept vector store filenames, or normalize doc_refs from filenames to API paths before validation. Currently blocks ALL research output.

2. **`validate_code_against_spec` intermittent failure** — Investigate what triggered the guardrail on test run 3. If it's a false positive, relax it. A guardrail that randomly blocks valid scripts is worse than no guardrail.

3. **Executor false-positive** — The executor reports failure when Blender succeeds. This has been partially mitigated (render discovery override) but the root cause remains: the Executor agent's LLM output parsing is unreliable.

### P1: Important (Quality Impact)

4. **Scale `* 0.5` halving** — API fixer added and tested (strips `* 0.5` from all-three-component scale tuples). Needs integration testing to confirm it fixes exploded geometry.

5. **Vector store query length limit** — The prompt description (4699 chars) exceeds the 4096 char limit for vector store searches. Need to truncate or summarize before querying.

### P2: Architectural (Long-Term)

6. **Fundamental hallucination mitigation** — See Part 1 recommendations. The current blacklist approach cannot scale. Consider whitelist-only, template-based, or AST-level validation.

7. **Spec-First pipeline stability** — When it works, it's great. But it adds 2-3 minutes of API Spec Agent work and the strict grounding guardrails create new failure modes. Need to decide: keep spec-first with relaxed guardrails, or disable it and strengthen the API fixer?

---

## Part 6: Key Lessons for Any Future Changes

1. **NEVER trust LLM training data for Blender API names.** Every attribute must be verified against the Blender 5.0 Python reference docs or runtime introspection. Training data from Blender 4.x and older is ACTIVELY HARMFUL.

2. **Deterministic fixes > LLM instructions.** When you need the LLM to NOT do something (like `* 0.5` on scale), adding instructions is unreliable. A post-processing fixer that deterministically corrects the pattern is the only reliable approach.

3. **Guardrails that reject valid output are worse than no guardrails.** A guardrail should catch real problems, not create false failures. The research guardrail rejecting valid doc_refs and the code guardrail randomly blocking valid scripts are net-negative.

4. **Test every code change against the actual pipeline.** GPT-5.3's `await modify_script(...)` bug would have been caught by a single test run. The Pydantic field assignments would have been caught by checking `SharedContext.__fields__`.

5. **The OpenAI Agents SDK `@function_tool` decorator returns a FunctionTool object, not the original function.** You cannot call it directly from Python code. You must either call the `_impl` function or use `Runner.run()` with an agent that has the tool.

---

## Part 7: Proposed Solution — Structured Attribute Registry (Claude + Ben)

### The Core Insight

Vector stores are great for DISCOVERY ("what Mantaflow settings affect spray quality?") but terrible for ENFORCEMENT ("is `resolution_max` a valid attribute?"). We've been using a discovery tool for an enforcement job. The result: the LLM reads the right answer from the vector store, then hallucinates a different name anyway because training data signal is stronger than retrieved context.

### Proposed Architecture: Batch-Convert Blender 5.0 Docs to Machine-Readable Registry

**Step 1: Batch-convert the full Blender 5.0 Python reference to structured JSON/YAML.**

The downloaded docs already exist locally (they were the original KB before vector stores). Convert every `bpy.types.*` and `bpy.ops.*` page to a structured format:

```json
{
  "bpy.types.FluidDomainSettings": {
    "attributes": {
      "resolution_max": {"type": "int", "range": [1, 10000], "default": 64},
      "domain_type": {"type": "enum", "values": ["GAS", "LIQUID"], "default": "GAS"},
      "cache_type": {"type": "enum", "values": ["REPLAY", "MODULAR", "ALL"], "default": "REPLAY"},
      "use_adaptive_timesteps": {"type": "bool", "default": false},
      "timesteps_max": {"type": "int", "range": [1, 100], "default": 4}
    },
    "deprecated_from_4x": {
      "resolution_divisions": "Use resolution_max",
      "use_adaptive_time_steps": "Use use_adaptive_timesteps",
      "timesteps_per_frame": "Use timesteps_max"
    }
  }
}
```

**Step 2: Use the registry for ENFORCEMENT (deterministic, no LLM).**

The API Validator reads this JSON directly. No vector search, no LLM interpretation. For every `bpy.types.X.attr` in the generated script:
- Is `X` a valid type? Yes/No.
- Is `attr` a valid attribute of `X`? Yes/No.
- If No: is it in `deprecated_from_4x`? → Auto-fix to the correct name.
- If not in deprecated list either: REJECT with specific error.

This is a dict lookup. Zero tokens. Zero hallucination. Milliseconds.

**Step 3: Keep vector store for DISCOVERY only.**

The vector store remains useful for the Research Agent: "what settings make liquid spray look realistic?" But it no longer has any role in attribute name validation. The structured registry handles that deterministically.

### Why This Is Better Than the Current Approach

| Current | Proposed |
|---------|----------|
| Blacklist ~15 known-bad attributes | Whitelist ALL valid attributes (thousands) |
| LLM reads docs, then hallucinates anyway | Dict lookup, no LLM in validation path |
| New hallucinations require manual blacklist updates | Unknown attributes auto-rejected |
| Vector store search: ~2s + LLM interpretation | JSON lookup: <1ms, deterministic |
| API Spec Agent: 2-3 min per run, $0.05+ | Zero cost, zero time for validation |

### Implementation Effort

1. **Batch conversion script** — Parse the HTML Blender Python reference docs into structured JSON. The docs have consistent HTML structure (class pages list attributes with types, defaults, ranges). A Python script with BeautifulSoup can do this. Estimated: 2-4 hours for the converter, produces ~2-5MB of JSON.

2. **Registry-based validator** — Replace the current API Validator's approach with a JSON lookup. Estimated: 1-2 hours.

3. **API Fixer integration** — The deprecated_from_4x mapping in the registry replaces the hardcoded BLENDER_50_FIXES list. Auto-fix becomes comprehensive instead of partial.

4. **Decide on API Spec Agent** — With a structured registry, the API Spec Agent may be redundant for attribute verification. It could be repurposed or removed, saving 2-3 min per run.

### Ben's Context

Ben notes the agent peaked with the wineglass test ~2 weeks ago and has regressed over 89+ scripts. The instability from hallucinated attributes prevents working on the main goals: autonomy, self-learning, and self-improvement. The blacklist approach (adding hallucinated attributes one at a time) is impractical and token-wasteful. Ben is willing to make sweeping changes and spend time batch-converting the KB.

### Open Question for GPT-5.3-XHigh

Should the structured registry REPLACE the API Spec Agent entirely, or should they coexist? The registry handles "is this attribute valid?" deterministically. The API Spec Agent handles "which attributes should I use for this effect type?" with LLM reasoning. These are different questions. But the API Spec Agent currently takes 2-3 minutes and its strict grounding guardrails create failure modes. Is the discovery value worth the cost and risk?

### Brainstorming Session Proposal

Ben proposes a three-way session: Claude (hands-on debugging knowledge), GPT-5.3-XHigh (architectural reasoning, high coding benchmarks), and Ben (pragmatic judgment). Goal: design an enforcement system that makes it IMPOSSIBLE for hallucinated attributes to reach Blender execution, while keeping the pipeline fast and the agents focused on creative decisions (scene design, parameter tuning) rather than fighting API correctness.

---

## Appendix: Files Modified Across All Sessions

### By Claude (this session + previous 3):
- `tools/dynamic_instructions.py` — Scale math gotcha, modifier guard, scene design requirements, banned API wrappers, lighting table, Brick Texture fixes
- `tools/blender_api_fixer.py` — MixRGB socket fixer, Brick Texture renames, liquid domain regex, particle_radius fixer, **scale halving fixer (NEW)**, indent fix, syntax checker
- `models/shared_context.py` — Added `last_execution_diagnosis`, `pending_gate_fix_instructions` fields
- `orchestrator.py` — Import fix (`_modify_script_impl`), error recovery hooks, render discovery, dirname path fix
- `hooks/enforcement_hooks.py` — `create_error_recovery_hooks()`, relaxed consecutive tool limits
- `specialized_agents/executor.py` — Rewritten EXECUTOR_INSTRUCTIONS

### By GPT-5.3 (P0-1 through P0-6):
- `orchestrator.py` — Spec-first re-enabled, `_apply_script_modifications`, DIAGNOSE/FIX phases, strict grounding abort
- `guardrails/research_guardrails.py` — Strict doc_ref validation
- `guardrails/script_guardrails.py` — Removed `time_scale` from hallucination patterns
- `tools/script_generator_tools.py` — Hallucination scan in `_modify_script_impl`
- `tools/semantic_docs_tools.py` — Doc_ref normalization/validation
- `utils/artifact_manager.py` — DiagnosisArtifact, FixArtifact dataclasses
- `specialized_agents/api_validator.py` — Removed `time_scale` from KNOWN_API_CHANGES
