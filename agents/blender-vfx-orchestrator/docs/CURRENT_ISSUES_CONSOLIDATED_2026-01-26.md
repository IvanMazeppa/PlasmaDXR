# Current Issues Consolidated (2026-01-26)

This document consolidates **current problems and blockers** across recent issue
documents to make triage easier. Use this as the primary “what’s broken” list.

## Sources Consolidated
- `docs/ARCHITECTURE_FAILURE_ANALYSIS_2026-01-25.md`
- `docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md`
- `docs/TRACING_SHAKEDOWN_TASKLIST_2026-01-25.md`
- `docs/SHAKEDOWN_TEST_FINDINGS_2026-01-24.md`
- `docs/REMEDIATION_PLAN_2026-01-23.md`

---

## Architecture Problems (Systemic)
1) **No authoritative API truth source.**
   Doc search results are not attribute-level, so scripts still guess API names.
2) **Validation is permissive.**
   API validator allows unknown attributes (examples continue to reach Blender).
3) **Enforcement was opt-out.**
   Doc-query enforcement now exists, but doc search quality is still weak.
4) **Feedback loop is weak.**
   Execution errors do not force attribute-level corrections or KB updates.

---

## Latest Run Findings (2026-01-25 21:05)
Trace: `traces/e2e_test_verbose_water_20260125_210557.jsonl`

- **Doc queries executed** before `write_script` (enforcement working).
- **Doc grounding still weak**: API search returns unrelated docs (e.g., Material preview).
- **Runtime failure:** `FluidDomainSettings.bake_frame_start` does not exist → AttributeError.
- **Result:** Both iterations fail; no quality eval or learning recording reached.

---

## Current Blockers (Must Fix)
1) **`bake_frame_start` / cache frame API mismatch**
   - Scripts still emit invalid attributes for bake frame ranges.
   - Add to `KNOWN_API_CHANGES` + `BLENDER_50_FIXES` and ensure Script Writer guidance forbids it.

2) **Doc search precision**
   - `search_blender_api_by_intent` returns unrelated APIs for fluid-domain intents.
   - API doc hits are mostly genindex snippets, not attribute-level sources.

3) **API validator too permissive**
   - Mark unknown attributes invalid (strict mode) before execution.

4) **Pattern library propagates outdated API**
   - Pattern application can reintroduce removed attributes.
   - Needs validation on pattern application step.

---

## High Priority (Next)
1) **Doc-coverage guardrail**
   - Require each attribute used in script to have a doc ref from API store.

2) **APISpec stage (spec-first)**
   - Build a minimal attribute list from docs before code generation.

3) **Learning agent baseline + experiment recording**
   - Baseline sync was fixed but not verified in a complete run.

---

## Fixed / Verified
- **Doc-query enforcement** for Script Writer (RunHooks).
- **Budget guardrail** uses `get_spent()` + `monthly_limit` (no `get_remaining()`).
- **Vision model default** for evaluation set to `gpt-5-mini`.
- **Test harness crash** (`IterationResult.primary_issue`) resolved.

---

## Still Unverified (Needs a successful full run)
- Baseline sync + experiment recording
- Quality Analyst run using `gpt-5-mini`
- Budget guardrail behavior under near-exhaustion

---

## Suggested Next Test (Once blockers fixed)
Run a short 2-iteration test and confirm:
- No AttributeError in Blender
- Doc refs match attributes used in script
- Learning Agent records experiment successfully

