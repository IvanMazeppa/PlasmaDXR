# LLM Primer - Blender VFX Orchestrator

**Date:** 2026-01-26  
**Audience:** New LLM/agent onboarding  
**Purpose:** One-stop, accurate, up-to-date context for this system

---

## 1) Non-Negotiables (Read First)

1. **Agents SDK is ground truth.**  
   Follow `docs/SDK_ENFORCEMENT_PROTOCOL.md` and OpenAI Agents SDK docs.
2. **Training data is outdated.**  
   Always trust `docs/VERSION_TRUTH.md` over memory.
3. **Spec-first pipeline is mandatory.**  
   No direct attribute use without verification.

---

## 2) What This System Does

The Blender VFX Orchestrator is a multi-agent system that generates Blender 5.0
Python scripts for volumetric effects (smoke, fire, pyro, etc.), executes them,
evaluates visual quality, and learns from results.

**Goal:** Produce valid Blender scripts with minimal hallucination and iterate
until quality gates pass.

---

## 3) High-Level Architecture

**Pipeline (simplified):**

User Request  
-> Research Agent  
-> Technique Selector (Coordinator)  
-> Spec-First Script Generation (API Spec Agent -> Code Writer)  
-> API Validator  
-> Executor (runs Blender)  
-> Quality Analyst  
-> Learning Agent  
-> Quality Gate Judge (Coordinator)

**Core idea:** The Script Writer cannot invent attributes because it must consume
a verified APISpec.

---

## 4) Key Agents and Responsibilities

**Coordinators**
- `TechniqueSelector`: chooses initial approach
- `ModificationStrategist`: decides modification strategy (iter 2+)
- `QualityGateJudge`: decides next action after evaluation

**Specialized Agents**
- `Research Agent`: doc search + pattern lookup
- `API Spec Agent`: builds verified APISpec (attributes + ops)
- `Code Writer`: writes script using only the spec
- `API Validator`: catches known bad API usage
- `Executor`: runs Blender script
- `Quality Analyst`: evaluates output (vision + metrics)
- `Learning Agent`: records experiments + patterns

---

## 5) Spec-First Pipeline (Critical)

**Files:**
- `specialized_agents/api_spec_agent.py`
- `models/api_spec.py`
- `guardrails/api_spec_guardrails.py`

**Flow:**
1. API Spec Agent queries docs for each attribute and op.
2. Output guardrail validates every doc_ref.
3. Code Writer uses only the APISpec (enforced by guardrail).

**Enum values:**
- `enum_values` must be extracted when `value_type="enum"`.
- Guardrail fallback map exists (temporary) to prevent enum hallucination.

---

## 6) Enforcement and Guardrails

**RunHooks** (`hooks/enforcement_hooks.py`):
- Loop detection
- Doc-query requirement
- Turn budgets
- Doc query statistics (counts)

**Guardrails**
- `validate_api_spec` ensures doc_refs
- `validate_code_against_spec` ensures only spec APIs are used
- Enum values are validated in `validate_code_against_spec`

**Current safety net:**  
`VALID_ENUM_VALUES` in `guardrails/api_spec_guardrails.py` is a TEMPORARY
fallback until enum extraction is reliable.

---

## 7) Tracing and Observability

**Tracing**
- Use `trace()` with `group_id=session_id`.
- Verbose traces go to `traces/*.jsonl`.

**Common commands**
```bash
VERBOSE_TRACING=1 VERBOSE_TRACE_FILE="traces/e2e_test_v10_$(date +%Y%m%d_%H%M%S).jsonl" \
python test_e2e_orchestrator.py
```

**Known trace issues**
- Some runs produce multiple `trace_start` lines; fix by ensuring a single
  outer trace wrapper per pipeline.

---

## 8) Current State (as of 2026-01-26)

**Recent runs:**
- **v9**: Spec-first passed; executor failed due to enum value hallucination.
- **v10**: API Spec Agent hit loop detection and fell back to old Script Writer;
  executor failed due to missing camera.

**Improvements:**
- Spec-first guardrails are working.
- Loop detection prevents doc-query spam.
- Enum validation guardrail added (needs verification).

---

## 9) Active Problems (Blockers)

See `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`. Summary:

1. **Spec-first doc search stabilization** (loop detection still triggers).
2. **Enum validation verification** (needs successful spec-first run).
3. **Render pipeline missing camera** (causes runtime failures).
4. **Doc search precision** (search results still noisy).
5. **API validator too permissive** (unknown attrs not strict).
6. **Pattern library can reintroduce outdated APIs**.

---

## 10) What To Do Next (Priority Order)

1. **Stabilize API Spec Agent search**
   - Reduce per-turn search volume
   - Enforce "Turn 4 output only"
2. **Fix render camera**
   - Create or set active camera before render
3. **Verify enum guardrail**
   - Run v10 again after spec-first succeeds
4. **Improve doc search precision**
   - Prefer API docs over manual/genindex hits

---

## 11) Key Files (Map)

**Pipeline**
- `orchestrator.py`

**Agents**
- `specialized_agents/api_spec_agent.py`
- `specialized_agents/code_writer_agent.py`
- `specialized_agents/api_validator.py`
- `specialized_agents/executor.py`

**Models**
- `models/api_spec.py`
- `models/shared_context.py`

**Guardrails**
- `guardrails/api_spec_guardrails.py`
- `guardrails/script_guardrails.py`

**Hooks**
- `hooks/enforcement_hooks.py`

**Docs (source of truth)**
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`
- `docs/VERSION_TRUTH.md`
- `docs/SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md`
- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

---

## 12) Rules for New LLMs

1. Read `VERSION_TRUTH.md` first.
2. Follow `SDK_ENFORCEMENT_PROTOCOL.md`.
3. Do not invent Blender attributes or enums.
4. Prefer spec-first pipeline at all times.
5. If unsure, search docs and record doc_refs.

---

## 13) Quick Status Check (Checklist)

- [ ] API Spec Agent search volume within limits
- [ ] Enum validation verified in spec-first run
- [ ] Executor renders without missing camera
- [ ] Quality Analyst and Learning Agent reached
- [ ] Traces correlate with group_id

---

*End of Primer*
