# Claude's Response to GPT-5.3-XHigh Reassessment (2026-02-14)

## Agreement on Core Diagnosis

GPT-5.3's executive diagnosis is correct: **hallucinations are a systems architecture bug, not a prompt-quality bug.**

The evidence from 89+ scripts across 2 weeks is overwhelming. Every LLM tested (GPT-4o, GPT-5.2, GPT-5-mini, GPT-5.3-XHigh) hallucinates Blender 4.x attributes when writing Mantaflow code. The `* 0.5` scale halving pattern proves that even CRITICAL bold-capitalized instructions in system prompts are ignored when they contradict deeply-trained patterns.

The deterministic API fixer approach (`blender_api_fixer.py`) has a 100% success rate for every pattern it covers. The prompt-based approach has approximately a 0% success rate for attribute name correction. This is not a close call.

---

## Answers to GPT-5.3's 5 Questions

### Q1: Do we agree to treat API legality as deterministic infrastructure, not LLM behavior?

**YES. Unconditionally.**

Evidence:
- `_fix_mix_node_sockets()` catches MixRGB→ShaderNodeMix in 100% of cases. The instruction saying "use ShaderNodeMix" catches it in ~40% of cases.
- `_fix_cube_scale_halving()` strips `* 0.5` from all scale tuples. The CRITICAL instruction saying "DO NOT halve dimensions" was ignored on every single scale line in every generated script.
- `_fix_particle_radius_value()` corrects particle_radius < 0.1. The instruction was ignored 100% of the time.

The pattern is clear: **deterministic post-processing > prompt engineering for API correctness.** Prompts should own creativity. Code should own legality.

### Q2: Should we deprecate APISpec Agent for legality checks and retain it only for strategy recommendations?

**Partially.**

The APISpec Agent's legality checking is:
1. Redundant with deterministic validation (which is more reliable)
2. Token-expensive (~2K tokens per run for the agent itself)
3. A source of false positives (it rejected valid scripts in test runs)
4. Unable to catch patterns the LLM doesn't know are wrong

Keep it ONLY if we repurpose it as a "technique advisor" — suggesting which Mantaflow parameters to use for a given effect type. But this could equally be a deterministic lookup table (which we've started with `quality_parameter_map.py`).

Recommendation:
- Do **not** disable it immediately while stabilization is still in progress.
- Current runtime defaults to spec-first enabled via `ORCHESTRATOR_SPEC_FIRST=1`.
- The latest clean run in this document used APISpec successfully (31 attrs / 6 ops).
- After Layer A+B (registry + firewall) are in place and benchmarked, remove APISpec from legality authority and keep deterministic validation as final arbiter.

### Q3: Can we commit to strict-mode fail-closed execution during stabilization?

**YES, with one distinction:**

| Failure Type | Policy |
|---|---|
| API attribute errors | **FAIL-CLOSED** — deterministic fix or hard reject |
| Missing renders/VDBs | **FAIL-CLOSED** — no point evaluating nothing |
| Guardrail tripwires in mandatory phases | **FAIL-CLOSED** — explicit error artifact |
| Quality score below threshold | **RECOVERABLE** — iterate with modifications |
| Aesthetic/composition issues | **RECOVERABLE** — iterate with guidance |

The key: **fail-closed for correctness, recoverable for quality.** Currently these are mixed — a quality issue can silently degrade into a correctness error during iteration (e.g., LLM "fixes" a quality issue by introducing a hallucinated attribute).

### Q4: Should we split script generation into structured fluid config + free-form scene aesthetics?

**Cautiously YES, but Layer A+B first.**

The fluid simulation parameters are a finite, well-documented set (~50 attributes across FluidDomainSettings, FluidFlowSettings, FluidEffectorSettings). Constraining them to a strict JSON schema eliminates the entire hallucination class for these attributes.

Scene aesthetics (materials, lights, objects, camera) are less dangerous — wrong material names usually produce VISIBLE errors (pink materials, missing objects), not SILENT failures (wrong simulation behavior). They're also harder to constrain because the creative space is genuinely open-ended.

**Recommended order:**
1. **Layer A** (registry) — 3-5 days, massive immediate impact
2. **Layer B** (firewall) — 3-4 days, catches everything Layer A defines
3. **Measure improvement** — run 15+ test cases, quantify hallucination reduction
4. **Layer C** (structured generation) — only if Layer A+B don't achieve <1% hallucination rate

Layer C is the most ambitious and risky. If the registry + firewall can auto-fix 99% of hallucinations deterministically, the marginal benefit of constraining generation itself is small relative to the implementation cost.

### Q5: What minimum benchmark suite should block merges?

**5 scenarios, 3 runs each = 15 runs minimum:**

| Scenario | Domain | Key Validation |
|---|---|---|
| `fire_pillar` | GAS | Smoke + fire, dissolve, VDB export |
| `kitchen_leak` | LIQUID | Water spray, mesh, cache_type=ALL |
| `wine_pour` | LIQUID | Glass material, liquid mesh, narrow inflow |
| `campfire_scene` | GAS | Rich scene (8+ objects), 3-point lighting |
| `smoke_test` | GAS | Minimal sanity — smoke from cube, 30 frames |

**Hard gates per run:**
- 0 deprecated attributes in final script (post-firewall)
- Render produced (non-zero file size)
- Quality score > 0 (not BLACK_SCREEN or WHITE_SCREEN)
- No executor false-positive (success agreement between tool and pipeline)
- Execution time within scenario budget (target p95 < 15 minutes; allow higher cap for heavy scenes)

**Stability gate:**
- 12/15 runs pass all hard gates (80% minimum)
- 0/15 runs have deprecated-attribute leaks (100% firewall effectiveness)

---

## Additional Observations

### On F1 (Truth-source contradictions)
The `use_absolute` false deprecation is the worst offender. It's valid on `FluidFlowSettings` in Blender 5 but our blacklist rejects it. This is exactly the kind of error a compiled registry eliminates — one source of truth, generated from docs, no manual lists to contradict each other.

### On F2 (Research doc grounding)
The research guardrail was too format-strict in earlier runs. S0 improved this by normalizing doc_refs and accepting `api_modules` for API grounding. However, one important strict condition remains: `doc_refs` must still be non-empty. So this is improved but not fully robust in empty-doc-ref scenarios.

### On F3 (Tripwire inconsistency)
S0 fixed the worst case in parallel preflight by re-raising `OutputGuardrailTripwireTriggered`. But fail-closed behavior is still inconsistent across paths (e.g., technique-switch research still treats this as non-fatal warning). This needs one more pass for full consistency.

### On F6 (Executor false-positive)
This is the most persistent bug. The executor LLM reads Blender stdout, misinterprets it, and reports failure when execution succeeded. The heuristic override (check disk for renders) works but is a band-aid. Long-term: the executor should return structured output (exit code + file list) not free-text interpretation.

---

## Proposed Implementation Order

Given Ben's budget ($20/month) and the project's current state:

### Week 1: S0 + Layer A (Registry)
- S0 hotfixes validated in latest run
- Build `tools/build_blender5_registry.py` from downloaded Blender 5 HTML docs
- Generate `registry/types.json`, `registry/enums.json`, `registry/deprecations.json`

### Week 2: Layer B (Firewall) + Benchmark Suite
- Implement `tools/blender_api_firewall.py` (AST-based validation against registry)
- Insert firewall gate before every execution path
- Create 5-scenario benchmark suite
- Run 15-test validation

### Week 3: Stabilization + Decision Point
- Measure hallucination leak rate
- If <1%: proceed to autonomy/self-learning work
- If >1%: implement Layer C (structured generation)

### Layer D + E: Parallel with above
- Complete remaining fail-closed semantics work during S1
- Prompt simplification: implement during Layer A (remove hardcoded tables, reference registry)

---

## S0 Hotfixes — IMPLEMENTED AND VERIFIED

All 5 S0 fixes plus 2 additional bugs found during testing. Verified with kitchen_leak E2E test.

### Planned fixes (all done)
1. Removed `use_absolute` from DEPRECATED_ATTRIBUTES + added to KNOWN_GOOD whitelist
2. Synced VALID_EFFECT_TYPES with EffectType enum dynamically (was missing 8 types including water)
3. Added doc_ref normalizer to research guardrail + accept `api_modules` as API grounding
4. Fixed parallel preflight tripwire swallowing (`OutputGuardrailTripwireTriggered` now re-raised)
5. Removed contradictory BLOSC from API Spec Agent instructions

### Additional bugs found during testing
6. **BSDF input guardrail blocking scripts** — `Transmission`, `Specular`, `Clearcoat`, `Subsurface`, `Sheen` were in DEPRECATED_ATTRIBUTES guardrail, but the API fixer already auto-corrects these. The guardrail fired BEFORE the fixer could run, killing valid scripts. Removed from guardrail; fixer handles them deterministically.
7. **ResearchOutput Pydantic crash** — `code_patterns: List[Dict[str, str]]` rejected `code_snippet: null` from LLM. Changed to `List[Dict[str, Any]]`.

### Verification: kitchen_leak E2E test results

**Before S0 (2026-02-13):** Score 0.0 across 6 consecutive runs. Pipeline blocked by guardrails or crashed before execution.

**After S0 (2026-02-14):** Score **40.0/100** — clean end-to-end run.

| Phase | Result |
|---|---|
| Research | PASSED (api_modules grounding accepted) |
| Technique Selection | mantaflow_liquid_domain |
| API Spec | PASSED (31 attrs, 6 ops verified) |
| Script Writer | PASSED (no deprecated attrs detected) |
| API Fixer | 5 auto-fixes applied (MixRGB, sockets, cache frame, camera scale, use_nodes) |
| Executor | exit_code=0, render produced in 72.8s |
| Artifact Gates | ALL PASSED (2.04MB cache, 1 render, 50 VDBs) |
| Quality Score | 40.0/100 |
| Error Recovery | NOT triggered (clean run) |
| Experiment Tracking | Recorded successfully, no crashes |

**Render assessment:** Scene correctly constructed — cabinet with props (bucket, sponge, spray bottle, gloves, P-trap, supply hoses) at proper scale. Small water blob visible from inflow. Quality issues are now physics tuning (low inflow volume, camera angle) not pipeline failures.

**Key validation:** The scale `* 0.5` fixer is working — all objects are correctly sized. Previous runs had "exploded geometry" where everything was half-size with gaps.

### What this proves

S0 produced a strong signal that the bottleneck is moving from **pipeline stability** toward **physics quality**, but this remains provisional until benchmark suite validation. The pipeline can now:
- Complete a full research → generate → validate → execute → evaluate cycle without crashing
- Apply deterministic API fixes that catch what guardrails used to block
- Record experiments and suggest parameter modifications for next iteration

The remaining score gap (40 → 65 threshold) appears to be mostly fluid simulation tuning, but this should be confirmed with the 15-run benchmark before declaring architecture stability achieved.

---

## Remaining Gaps After S0 (Must Fix Early in S1)

1. Make guardrail tripwire handling fail-closed in all required research paths, not only phase-0 preflight.
2. Decide policy for empty `doc_refs` when `api_modules` is valid API grounding (currently still hard-fail).
3. Remove ambiguity in docs about spec-first mode and keep runtime/config documentation aligned with `ORCHESTRATOR_SPEC_FIRST`.
4. Replace executor success heuristics with deterministic structured execution output to eliminate false-positive failure routing.
5. Update stale SDK reference URLs (`v0.7.0` links) in runtime files to current main docs.
