# Architecture Failure Analysis (2026-01-25)

## Executive Summary
The workflow is failing for architectural (not incidental) reasons. The system allows
the Script Writer to generate Blender API attribute names without mandatory,
attribute‑level verification. Validation is non‑authoritative (unknown attributes
are treated as valid), and doc search results are not wired into code generation.
This creates a feedback loop where scripts pass “validation” but crash in Blender,
producing zero learning signal and repeated AttributeErrors.

## Evidence (Trace + Audit)

### Latest Practical Test (Verbose, gpt-5-mini low reasoning)
Key new observations from the 2026-01-25 verbose run:

1) **Doc search still returns manual‑only content (no API refs).**
```325:330:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
{"event_type": "trace_start", "name": "VFX Pipeline: test_smoke_e2e", "group_id": "test_smoke_e2e_20260125_061226"}
{"event_type": "span_end", "span_type": "function", "name": "blender_doc_search_bundle",
 "span_data": {"output": "{\"effect_type\": \"pyro\", ... \"DocType: manual\" ... \"related_apis\": [], ... }"}}
# ... truncated for brevity
```
This confirms the search bundle is returning **manual pages** and no API references,
so attribute names still aren’t grounded.

2) **Script Writer still does not call doc tools** (only `write_script` + `validate_script`).
```353:358:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
{"event_type": "span_start", "span_type": "function", "name": "write_script", ...}
{"event_type": "span_end", "span_type": "function", "name": "write_script", ...}
{"event_type": "span_start", "span_type": "function", "name": "validate_script", ...}
{"event_type": "span_end", "span_type": "function", "name": "validate_script", ...}
# ... truncated for brevity
```
No `semantic_search_blender_docs` or `search_blender_api_by_intent` calls appear
inside the Script Writer span, which means doc usage is still optional in practice.

3) **Learning Agent still can’t record experiments** (baseline missing).
```391:392:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
{"event_type": "span_end", "span_type": "function", "name": "record_experiment_result",
 "span_data": {"output": "{\"experiment_id\": \"\", \"success\": false, \"error\": \"No baseline recorded. Call record_baseline() first.\", ...}"}}
# ... truncated for brevity
```
This prevents learning outcomes from being persisted across iterations.

4) **Quality analysis still uses gpt‑5.2** even under quick_test preset.
```381:382:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
{"event_type": "span_end", "span_type": "function", "name": "analyze_with_vision",
 "span_data": {"output": "{... \"vision_model\": \"gpt-5.2\", ... }"}}
# ... truncated for brevity
```
This undermines the “low‑cost gpt‑5‑mini” objective unless explicitly overridden.

5) **Test harness crash (non‑orchestrator issue).**
During the run, `test_e2e_orchestrator.py` crashed with
`AttributeError: 'IterationResult' object has no attribute 'primary_issue'`.
This is a test script mismatch, not a core pipeline error, but it masks results.

6) **Console‑only warnings (not in trace JSONL).**
The test run also emitted runtime warnings that indicate pipeline contract drift:
- `BudgetTracker` missing `get_remaining` in `check_budget_before_quality`
- Baseline sync failure: `_record_baseline_impl()` got unexpected `effect_type`
- Pattern outcome report failed: `'FunctionTool' object is not callable`

### Status Update (Completed)
As of 2026-01-26, items 1-5 above are fixed in code:
- blender_doc_search_bundle now forces API-store queries and preserves API doc refs.
- Script Writer hooks now require a doc query before write_script/modify_script.
- Baseline sync now calls record_baseline with script_path and no extra args.
- Vision analysis model set to gpt-5-mini (configurable via VISION_MODEL).
- test_e2e_orchestrator now reads primary_issue from iteration quality.

Also fixed: budget guardrail uses BudgetTracker.get_spent/remaining to avoid get_remaining errors.

### Latest Practical Test (Post-fix, 2026-01-25 07:00)
Run: `ORCHESTRATOR_PRESET=quick_test ORCHESTRATOR_MODEL=gpt-5-mini VERBOSE_TRACING=1 python3 test_e2e_orchestrator.py`

Improvements vs the 05:00 run:

1) **Doc bundle now uses API-first queries and returns API doc refs.**
```445:446:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
{"event_type": "span_end", "span_type": "function", "name": "blender_doc_search_bundle", ... "queries_used": ["bpy.types.FluidDomainSettings", "bpy.types.FluidFlowSettings", "bpy.ops.fluid.bake", ...], ... "doc_refs": ["blender_python_reference_5_0/info_overview.html#class-registration-", "blender_python_reference_5_0/gpu.types.html#gpu.types.GPUFrameBuffer.read_depth", ...]}
# ... truncated for brevity
```

2) **Doc-query enforcement now blocks Script Writer from calling write_script without a doc lookup.**
This is a correctness improvement (prevents ungrounded code), but the Script Writer
still does not call doc tools, so the pipeline halts at Phase 1 with DocQueryRequiredError.

3) **Test harness no longer crashes.**
The run terminates cleanly with `Status: failed` instead of throwing
`IterationResult.primary_issue` errors.

Not yet verified (blocked by early stop):
- Baseline sync + experiment recording
- Vision model usage in Quality Analyst
- Budget guardrail behavior during quality eval

### Attribute Hallucination (Trace)
Multiple runs fail with `AttributeError` for plausible but nonexistent properties:

```108:112:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
Traceback (most recent call last):
  File "/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/test_smoke_e2e_v1.py", line 106, in create_domain
    dsettings.resolution_divisions = DOMAIN_SETTINGS['resolution_divisions']
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FluidDomainSettings' object has no attribute 'resolution_divisions'
```

```172:176:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
Traceback (most recent call last):
  File "/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/test_smoke_e2e_v1.py", line 89, in <module>
    dsettings.use_adaptive_time_steps = False
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FluidDomainSettings' object has no attribute 'use_adaptive_time_steps'
```

```304:308:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
Traceback (most recent call last):
  File "/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/test_smoke_e2e_v1.py", line 126, in <module>
    flow_settings.absolute_density = False
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FluidFlowSettings' object has no attribute 'absolute_density'
```

### Doc Search Returns Manual Only (Trace)
The research doc search returns manual pages, but `related_apis` is empty and
no API references are cited, so exact attribute names never become ground truth:

```261:262:agents/blender-vfx-orchestrator/traces/e2e_test_verbose.jsonl
"results": [{"content": "# Settings… DocType: manual …"}],
"code_snippets": [], "related_apis": [], "doc_refs": ["blender_manual_html/physics/fluid/type/flow.html#settings-", …]
```

### Audit Findings (Design Gaps)
The audit doc confirms critical enforcement and validation gaps:

- Script Writer has doc tools but **no requirement to use them** for code generation.
```75:97:agents/blender-vfx-orchestrator/docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md
The agent HAS documentation tools but the instructions don't REQUIRE their use
before writing code.
Tool calls: ['write_script', 'validate_script']
doc_query_made: False
```

- Doc query enforcement was disabled with the assumption that research/validator
cover it (they do not).
```99:113:agents/blender-vfx-orchestrator/docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md
require_doc_query_before=[
    # NOTE: write_script and generate_script NO LONGER require doc query
]
This assumes… Research Agent’s docs lookup covers specific API attributes (FALSE)
```

- API validator defaults to “valid” for unknown attributes.
```115:133:agents/blender-vfx-orchestrator/docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md
validation = APICallValidation(
    is_valid=True,        # DEFAULT IS TRUE!
)
Only checks known issues in static dictionary
```

- `validate_script` does not verify Blender API names at all.
```135:144:agents/blender-vfx-orchestrator/docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md
validate_script … Does NOT validate Blender API attribute names
Returns valid: true for scripts with hallucinated attributes
```

## Root Causes (Systemic)
1. **No authoritative API truth source.** Research outputs are high‑level
   summaries; the Script Writer still “guesses” attribute names. The API store
   exists but isn’t used to validate attributes in the script generation path.
2. **Validation is permissive by default.** The validator only catches known
   bad patterns; unknowns are treated as valid.
3. **Enforcement is opt‑out.** Doc query is not required before `write_script`
   or `modify_script`, so the Script Writer bypasses documentation.
4. **Feedback loop is weak.** Execution failures do not force an API‑level
   correction or prevent further attempts with invalid attributes.

## SDK Alignment (Required for Reliability)
The Agents SDK encourages explicit enforcement using output guardrails and
RunHooks, plus tracing to audit tool usage:

- Use a single `trace()` with `group_id` for run‑wide correlation.
```55:68:agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md
with trace("Workflow Name", group_id=session_id):
Do NOT nest trace() calls. Use one outer trace per pipeline run.
```

- Enforce rules with guardrails and RunHooks (not just prompt text).
```105:142:agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md
Input/Output Guardrails … RunHooks (Lifecycle Callbacks)
```

- Tool guardrails don’t apply to `agent.as_tool()`, so cross‑tool enforcement
  must live in RunHooks.
```215:226:agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md
Tool guardrails apply only to function tools … Use RunHooks for cross-tool enforcement
```

## Proposed Architecture (Fix the Design, Not Just Symptoms)

### A. Insert an “API Spec” stage (Spec‑First Pipeline)
**Goal:** Do not allow code generation until every attribute is explicitly
verified.

**New pipeline:**
1) Research (approach + doc refs)
2) **API Spec Builder** (structured list of attributes + doc refs)
3) Script Writer (only uses approved attributes)
4) API Validator (strict: unknown == invalid)
5) Executor

**Key rule:** Script Writer can only use attributes present in the API spec.

### B. Attribute Verification Gate (Deterministic)
Create a deterministic validator against a generated API index:
- Build `blender_api_index.json` from the API docs store
- Extract `bpy.types.<Class>.<attr>` and `bpy.ops.<op>`
- Validation becomes a fast lookup (no LLM guesses)
- If missing, trigger doc search and update index or block execution

### C. Doc‑Coverage Guardrail
Add an output guardrail that enforces **attribute‑level citations**:
- Parse Script Writer output to extract `bpy.types` and `bpy.ops` usage
- Require a `doc_refs` list where **each attribute** has a matching API doc path
- Tripwire on any unverified attribute (per SDK guardrail pattern)

### D. Enforced Doc Query Before Code
Use RunHooks to block `write_script`/`modify_script` unless a doc query was made
in the same run. This enforces SDK‑aligned tooling behavior instead of relying on
prompts alone. Tool guardrails do not apply to `agent.as_tool()`, so this must be
RunHooks‑based.

### E. Strict API Validator (Unknown == Invalid)
Flip the validation default:
- If an attribute is not in the API index, mark **invalid**
- Provide “unverified attribute” warnings to force doc search and correction
- Only permit execution when **all attributes are verified**

### F. AttributeError Feedback → Correction Memory
When execution fails:
- Extract the unknown attribute from the exception
- Search the API store for the closest match
- Write a correction record to the KB (or `KNOWN_API_CHANGES` if high‑confidence)
- Prevent repeating the same invalid attribute next iteration

---

## Proposed Redesign (End‑to‑End)

### New Primary Contract
The system’s ground truth must be **API‑verified attributes**, not LLM memory.

**Artifact flow:**
1. **ResearchOutput** (approach + doc refs)
2. **APISpec** (exact attributes + doc paths + confidence)
3. **ScriptOutput** (uses only attributes in APISpec)
4. **ValidationReport** (strict: unknown => invalid)
5. **ExecutionOutput**

### APISpec Example (structure)
```json
{
  "object": "FluidDomainSettings",
  "attributes": [
    {"name": "resolution_max", "doc_path": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#bpy.types.FluidDomainSettings.resolution_max"},
    {"name": "use_adaptive_timesteps", "doc_path": "…"}
  ]
}
```

---

## Implementation Plan

### Phase 0 (Immediate / Low Risk)
- **Re‑enable doc query enforcement** for `write_script`/`modify_script`
- **Add doc‑coverage guardrail** on Script Writer output
- **Change validator default to invalid** for unknown attributes
- **Update doc search routing** to prioritize API store for attribute queries

### Phase 1 (1–2 days)
- **Generate API index** from the vector store (API docs only)
- **Add APISpec stage** (struct output + doc refs)
- **Update Script Writer** to only use APISpec attributes

### Phase 2 (1 week)
- **Attribute correction tool** + KB memory
- **Automated regression tests** for known attribute errors
- **RunHooks audit** to ensure doc query usage is visible in traces

---

## Success Criteria
- 2‑iteration shakedown completes **without AttributeError**
- Every attribute used in scripts is tied to a **DocPath** from the API store
- API Validator blocks **all unknown attributes** before execution
- Trace shows doc‑query tool usage before `write_script`

---

## Risks / Tradeoffs
- **More tool calls** may increase cost; mitigate by using API index lookups
  rather than repeated vector store calls.
- **False negatives** if API index coverage is incomplete; mitigate with
  controlled fallback to vector store search.
- **Slower iteration** in quick_test; mitigate by skipping execution when
  doc coverage is incomplete rather than running a known‑bad script.

---

## Bottom Line
The system must treat documentation as the **authoritative source of API truth**.
Without a mandatory attribute‑verification gate, the workflow will keep generating
valid‑looking but invalid code and will never converge. The proposed redesign
forces doc‑grounded attributes at every stage and aligns enforcement with SDK
guardrails/RunHooks patterns.
