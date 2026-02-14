# Deep Analysis Investigation: Blender VFX Orchestrator

Date: 2026-02-12
Scope: `agents/blender-vfx-orchestrator/`
Author: Codex investigation (workflow + implementation + methodology)

## Executive Summary
The orchestrator has a solid architecture on paper, but enforcement and execution paths have drifted from the documented state machine. The largest risks are: execution failures not entering a real DIAGNOSE/FIX path, enforcement bypass through direct script mutation calls, weak doc-grounding validation, and observability gaps that leave run outcomes as `unknown` with `quality_score: null`.

## Investigation Method
- Compared required behavior in:
  - `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md`
  - `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md`
  - `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md`
- Reviewed implementation in:
  - `agents/blender-vfx-orchestrator/orchestrator.py`
  - `agents/blender-vfx-orchestrator/guardrails/*.py`
  - `agents/blender-vfx-orchestrator/hooks/enforcement_hooks.py`
  - `agents/blender-vfx-orchestrator/tools/script_generator_tools.py`
  - `agents/blender-vfx-orchestrator/utils/artifact_manager.py`
- Analyzed traces and summaries:
  - `agents/blender-vfx-orchestrator/traces/phase4_gate_shakedown_20260126_184218.jsonl`
  - `agents/blender-vfx-orchestrator/traces/phase4_gate_shakedown_20260126_184218_summary.json`
  - `agents/blender-vfx-orchestrator/traces/shakedown_2iter_20260129_195522.jsonl`
  - `agents/blender-vfx-orchestrator/traces/shakedown_2iter_20260129_195522_summary.json`

## Findings

### 1) High: EXECUTE failure path does not implement DIAGNOSE -> FIX -> EXECUTE
Evidence:
- Required routing is explicit in `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md:37` and `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md:59`.
- Runtime path in `agents/blender-vfx-orchestrator/orchestrator.py:3330-3370` only attempts a re-exec; if recovery fails it records score `0` and `continue`s.
Impact:
- Repeated execution failures consume iterations without root-cause correction.
- State-machine contract is violated.
Recommendation:
- Add explicit DIAGNOSE and FIX phases before any second EXECUTE attempt.
- Require a concrete fix artifact (diff or regenerated script) before retry.

### 2) High: Direct script mutation bypasses RunHooks/guardrails/doc-query enforcement
Evidence:
- Direct calls to `_modify_script_impl(...)` in `agents/blender-vfx-orchestrator/orchestrator.py:2535`, `agents/blender-vfx-orchestrator/orchestrator.py:2571`, and `agents/blender-vfx-orchestrator/orchestrator.py:2610`.
- Enforcement is configured around tool names (`write_script`, `generate_script`, `modify_script`) in `agents/blender-vfx-orchestrator/hooks/enforcement_hooks.py:133-137`.
- `_modify_script_impl` is a raw function (`agents/blender-vfx-orchestrator/tools/script_generator_tools.py:868`) and therefore does not trigger tool-level hook checks.
Impact:
- Policy can be bypassed exactly where high-risk script edits occur.
- API hallucinations can be reintroduced outside guarded tool flow.
Recommendation:
- Route all modifications through `modify_script` tool wrapper only.
- Block direct `_impl` calls from orchestrator path (lint/test guard).

### 3) High: Doc-grounding integrity is incomplete (blank/null `doc_path` accepted)
Evidence:
- `agents/blender-vfx-orchestrator/traces/phase4_gate_shakedown_20260126_184218.jsonl:7` includes results with `"doc_path": ""`.
- `agents/blender-vfx-orchestrator/traces/phase4_gate_shakedown_20260126_184218.jsonl:24` and `:62` include `"doc_path": null`.
- Research guardrail only checks `doc_refs` is non-empty, not that refs map to valid doc paths: `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py:56-57`.
- Roadmap requires real DocPath fidelity: `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md:53-55`.
Impact:
- "Doc-grounded" outputs can pass with unusable references.
Recommendation:
- Add strict `DocPath` validation (`non-empty`, stable path format) in research output guardrail and bundle tooling.

### 4) High: Quality gate execution/telemetry is inconsistent in observed shakedown runs
Evidence:
- `agents/blender-vfx-orchestrator/traces/shakedown_2iter_20260129_195522_summary.json:8-9` => `"status": "unknown"`, `"quality_score": null`.
- `agents/blender-vfx-orchestrator/traces/phase4_gate_shakedown_20260126_184218_summary.json:8-9` => same pattern.
- No quality-analysis events found in `agents/blender-vfx-orchestrator/traces/shakedown_2iter_20260129_195522.jsonl` for `Quality Analyst`/quality evaluation strings.
- Manual requires quality gates as non-negotiable: `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md:87-94`.
Impact:
- Impossible to trust pass/fail status for these runs.
- Regression triage is delayed because core decision metric is absent.
Recommendation:
- Treat missing quality artifact as hard failure.
- Add trace assertions: each iteration must emit a quality artifact + scorecard.

### 5) High: Effect-type guardrail is out of sync with canonical effect enum
Evidence:
- Guardrail allowlist omits valid enum values such as `sun`, `star`, `supernova`, `water`, etc.: `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py:34-45`.
- Canonical enum includes them: `agents/blender-vfx-orchestrator/models/shared_context.py:26-43`.
- Guardrail trips when no listed token found: `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py:243-261`.
Impact:
- Valid requests can be blocked depending on prompt wording.
- Behavior differs from model schema and MCP API contract.
Recommendation:
- Generate guardrail effect list from `EffectType` enum at runtime.

### 6) Medium: Artifact manifest is specified but not produced
Evidence:
- Manifest is required by architecture docs: `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md:74-81` and `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md:70-73`.
- Manifest writer exists: `agents/blender-vfx-orchestrator/utils/artifact_manager.py:294-323`.
- Orchestrator writes scorecard/iteration but not manifest: `agents/blender-vfx-orchestrator/orchestrator.py:3568-3581`, `agents/blender-vfx-orchestrator/orchestrator.py:3948-3960`.
- On-disk check found zero manifest artifacts under `agents/blender-vfx-orchestrator/sessions/artifacts/`.
Impact:
- Missing end-to-end run provenance (inputs/outputs/decisions/env).
Recommendation:
- Emit `manifest.json` once per iteration and once final run-level summary.

### 7) Medium: Artifact-gate failures do not force deterministic remediation
Evidence:
- On gate failure, pipeline records issues and continues next loop: `agents/blender-vfx-orchestrator/orchestrator.py:3407-3448`.
- Documented failure routing expects fix path before next execute: `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md:36-38`.
Impact:
- Can loop with repeated structural failures (empty cache/wrong renders) without hard correction.
Recommendation:
- Gate failure should require an explicit fix plan artifact before loop continuation.

### 8) Medium: "Mandatory" pre-iteration research is conditional and can be skipped
Evidence:
- Research call is gated behind `iteration > 1 and previous_script and previous_script.script_path`: `agents/blender-vfx-orchestrator/orchestrator.py:2237-2256`.
- Manual states this step as mandatory before each modification iteration: `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md:98-103`.
- In one two-iteration shakedown summary, only one Research Agent call is observed: `agents/blender-vfx-orchestrator/traces/shakedown_2iter_20260129_195522_summary.json` (`agents.Research Agent.calls = 1`).
Impact:
- Iteration-2 decisions may proceed without fresh risk analysis.
Recommendation:
- Move pre-iteration research into a dedicated phase that always runs for `iteration > 1`.

### 9) Medium (Historical at investigation time): SDK version truth was inconsistent across docs vs runtime dependency
Evidence:
- Runtime dependency pins `openai-agents==0.8.3`: `agents/blender-vfx-orchestrator/requirements.txt:10`.
- At investigation time, truth docs still stated SDK v0.7.0: `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md:36`, `agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md:7`.
Impact:
- Engineering decisions may follow stale behavior assumptions.
Recommendation:
- Update all SDK truth docs to match installed version and revalidate claimed API patterns.

Resolution update (2026-02-13):
- `docs/VERSION_TRUTH.md` and `docs/SDK_ENFORCEMENT_PROTOCOL.md` were updated to SDK `v0.8.3`.

### 10) Medium: Regex-centric script mutation remains brittle for structural fixes
Evidence:
- `_modify_script_impl` performs broad regex replacements, including chained attribute patterns: `agents/blender-vfx-orchestrator/tools/script_generator_tools.py:948-1077`.
- Roadmap itself flags direct-modify routing and contract stability as unresolved risks: `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md:74-77` and `:101-103`.
Impact:
- Complex code changes can silently fail or partially apply.
Recommendation:
- Replace regex-heavy mutation for non-trivial changes with regenerate-from-spec or AST-safe editing.

### 11) ~~Critical~~ RESOLVED: Hallucination-prevention path is hard-disabled in runtime
**STATUS (2026-02-14):** FIXED. Spec-first now defaults ON via `ORCHESTRATOR_SPEC_FIRST=1` env var (constructor reads `os.getenv("ORCHESTRATOR_SPEC_FIRST", "1")`). The 2026-02-14 kitchen_leak E2E test confirmed spec-first ran successfully (31 attrs, 6 ops verified).

Evidence (at time of investigation):
- Spec-first path was disabled by code in constructor: `agents/blender-vfx-orchestrator/orchestrator.py:733` (`self._use_spec_first_pipeline: bool = False`).
- With this flag off, all generation/modification routes fell back to the original Script Writer path.
- Recent `kitchen_leak` runs aligned with this behavior pattern and did not use spec-first protection.
Impact (historical):
- The system path designed to structurally constrain API hallucinations was unavailable in production runs.
Recommendation (DONE):
- ~~Re-enable spec-first behind an explicit runtime flag~~ Done: `ORCHESTRATOR_SPEC_FIRST=1` is now the default.
- Trace field `generation_mode: spec_first|legacy` is logged at startup.

### 12) Critical: Doc retrieval is polluted and accepted without strong grounding checks
Evidence:
- `kitchen_leak` API query includes mixed and legacy-prone tokens (`...time_scale...`) and returns unrelated chunks (`gpu.html`, quickstart) with `doc_path: null`: `agents/blender-vfx-orchestrator/traces/kitchen_leak_20260212_035933.jsonl:47`.
- Research guardrail only enforces non-empty `doc_refs`, not valid API doc paths: `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py:56-57`.
- Doc tool may emit sentinel instead of real sources: `doc_refs = ["doc_search_empty"]` in `agents/blender-vfx-orchestrator/tools/semantic_docs_tools.py:650`.
- Session evidence shows Docs Expert recommendations can include deprecated names like `resolution_divisions`: `agents/blender-vfx-orchestrator/build/orchestrator_state/session_fuel_ignition_20260203_234101_20260203_234102.json:46`.
Impact:
- "Doc-grounded" outputs can be grounded to irrelevant or non-authoritative chunks.
- Hallucinated fields propagate into downstream prompts and parameter suggestions.
Recommendation:
- Require `doc_refs` entries to match strict API/manual path patterns (no sentinels, no filenames, no null path).
- For attribute-level grounding, require at least one `bpy.types.FluidDomainSettings` or `bpy.types.FluidFlowSettings` doc path per generated fluid attribute.
- Reject research outputs whose top-k results are change-log/index/tutorial-only when API intent is requested.

### 13) High: Internal API truth is contradictory across prompts, validators, and docs
Evidence:
- At investigation time, dynamic instructions suggested `time_scale` directly (`agents/blender-vfx-orchestrator/tools/dynamic_instructions.py:660-661`) while other project files treated it as deprecated hallucination.
- At investigation time, hallucination scanner blacklists marked `time_scale` as removed in Blender 5.0 (resolved in P0 implementation).
- Historical spec-first trace produced invalid enum `fset.flow_behavior = 'FLOW'`: `agents/blender-vfx-orchestrator/traces/e2e_test_v9_20260126_024559.jsonl:79`, and executor failure confirms valid enum set is `('INFLOW', 'OUTFLOW', 'GEOMETRY')`: `agents/blender-vfx-orchestrator/traces/e2e_test_v9_20260126_024559.jsonl:93`.
- Blender 5 API docs show `FluidDomainSettings.time_scale` exists and `FluidFlowSettings.flow_behavior` valid enums are `INFLOW/OUTFLOW/GEOMETRY` (Blender API pages: `bpy.types.FluidDomainSettings.html`, `bpy.types.FluidFlowSettings.html`).
Impact:
- The system cannot consistently decide whether a token is valid, invalid, or deprecated.
- This creates both false negatives (hallucinations pass) and false positives (valid settings blocked/rewritten).
Recommendation:
- Build a single generated "Blender 5 truth registry" from API docs and consume it from:
  - prompt templates,
  - guardrails,
  - api validator,
  - deterministic parameter maps.
- Replace hardcoded deprecated-token blacklists with generated allow/deny sets from the registry.
- Add a CI consistency check that fails when any instruction/guardrail references attributes not in the registry.

### 14) High (Historical at investigation time): Direct mutation path bypassed hallucination scanner
Evidence:
- At investigation time, orchestrator called `_modify_script_impl(...)` directly at multiple points.
- At investigation time, `_modify_script_impl` did not run explicit post-modification hallucination scanning.
Impact:
- Hallucinated attributes could be reintroduced during "fix" iterations even if initial generation was clean.
Resolution update (2026-02-13):
- Direct calls were centralized into `_apply_script_modifications(...)` and `_modify_script_impl` now performs `_scan_for_hallucinated_api(...)` before writing output.
Recommendation:
- Keep centralized modification path and hallucination scanning as required.
- Add regression tests to fail if any pipeline path bypasses `_apply_script_modifications(...)`.

### 15) Medium: `hasattr(...)` masking allows silent API drift
Evidence:
- Generated scripts heavily use guarded assignments (`if hasattr(ds, '...')`) in lieu of strict attribute guarantees (see `kitchen_leak` write output: `agents/blender-vfx-orchestrator/traces/kitchen_leak_20260212_035933.jsonl:55`).
- This pattern can silently no-op when attribute names are wrong, producing low-quality or empty simulations without immediate hard failures.
Impact:
- Hallucinations shift from explicit crashes to silent behavior regressions.
- Token spend increases due iterative retries on low-quality outcomes.
Recommendation:
- Disallow `hasattr` guards for required simulation-critical attributes in generated scripts.
- Require explicit validation artifact listing: "required attributes present and set" before execution.

## Root-Cause Themes
- Contract drift: design docs and runtime behavior diverged.
- Enforcement holes: safeguards exist, but critical paths bypass them.
- Observability gaps: key artifacts/metrics are not consistently emitted.
- Truth-source fragmentation: prompts, validators, and docs tool outputs disagree on Blender 5 API reality.

## Prioritized Remediation Plan

### P0 (Immediate)
1. Re-enable spec-first behind runtime flag and add trace-visible `generation_mode`.
2. Enforce no direct `_impl` modifications from orchestrator; use guarded `modify_script` only.
3. Implement explicit DIAGNOSE -> FIX phase for EXECUTE and gate failures.
4. Make missing quality artifact a hard failure.
5. Enforce strict doc grounding:
   - no null/empty/sentinel doc refs,
   - API-intent results must include valid `bpy.types.*` or `bpy.ops.*` paths.
6. Remove contradictory deprecated guidance from prompts and dynamic mappings.

### P1 (Short Term)
1. Generate and adopt a single Blender 5 API truth registry for prompts/guardrails/validators.
2. Unify effect-type guardrail with `EffectType` enum.
3. Emit run manifest artifacts and add tests for manifest existence.
4. Update SDK truth docs to pinned runtime version.
5. Add strict retrieval quality gates (minimum relevant API hits before script writing).

### P2 (Hardening)
1. Reduce regex mutation scope and adopt structured patching/regeneration for code fixes.
2. Add state-machine conformance tests asserting documented routing.
3. Add telemetry conformance tests for quality/scorecard/manifest per iteration.
4. Add hallucination-rate telemetry (`deprecated_token_hits`, `invalid_enum_hits`, `doc_ref_quality_score`) per iteration.

## Suggested Verification Tests
- State-machine test: injected EXECUTE failure must produce DIAGNOSE and FIX artifacts before re-exec.
- Enforcement test: any attempt to call `_modify_script_impl` from pipeline path fails CI.
- Doc fidelity test: reject research output containing blank/null `doc_path`.
- Artifact test: each iteration must produce `quality`, `scorecard`, `iteration`, and `manifest` artifacts.
- Hallucination regression suite (must fail on):
  - `resolution_divisions`, `use_adaptive_time_steps`, `timesteps_per_frame`, `timesteps_maximum`, `velocity_multi`,
  - invalid enums such as `flow_behavior='FLOW'`.
- Prompt lint test: fail if prompt/instruction files contain deprecated tokens not present in Blender 5 truth registry.
- Retrieval quality test: API-intent doc searches must return >=2 results with valid API `doc_path` anchors before code generation.

## P0 Implementation Status (2026-02-13)
- [x] `P0-1` Re-enable spec-first behind runtime flag and add trace-visible generation mode.
  - Implemented: `ORCHESTRATOR_SPEC_FIRST` controls pipeline mode (default enabled); `generation_mode` now emitted in `RunConfig.trace_metadata`.
  - Code: `agents/blender-vfx-orchestrator/orchestrator.py`.
- [x] `P0-2` Enforce no direct `_impl` modifications from orchestrator.
  - Implemented: removed scattered direct `_modify_script_impl(...)` pipeline calls and centralized modifications via guarded `_apply_script_modifications(...)`; `_modify_script_impl` now includes hallucination scanning before write.
  - Code: `agents/blender-vfx-orchestrator/orchestrator.py`.
- [x] `P0-3` Implement explicit DIAGNOSE -> FIX phase for EXECUTE and gate failures.
  - Implemented: execute failures now emit diagnosis + fix artifacts before recovery/re-exec; artifact-gate failures now emit diagnosis + fix artifacts before next-iteration remediation.
  - Code: `agents/blender-vfx-orchestrator/orchestrator.py`, `agents/blender-vfx-orchestrator/utils/artifact_manager.py`.
- [x] `P0-4` Make missing quality artifact a hard failure.
  - Implemented: quality artifact file existence/non-empty checks now raise hard runtime failure via `_require_artifact_file(...)`.
  - Code: `agents/blender-vfx-orchestrator/orchestrator.py`.
- [x] `P0-5` Enforce strict doc grounding (`doc_refs` validity and API-intent source quality).
  - Implemented: `doc_refs` now reject sentinels/filenames/nulls; research output guardrail requires strict doc path format and at least one API doc ref (`bpy.types.*`/`bpy.ops.*`); ungrounded research now aborts run.
  - Code: `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py`, `agents/blender-vfx-orchestrator/tools/semantic_docs_tools.py`, `agents/blender-vfx-orchestrator/orchestrator.py`.
- [x] `P0-6` Remove contradictory deprecated guidance from prompts and dynamic mappings.
  - Implemented: removed incorrect deprecation handling for `time_scale` from script guardrails and API validator to align with Blender 5 API ground truth.
  - Code: `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py`, `agents/blender-vfx-orchestrator/tools/script_generator_tools.py`, `agents/blender-vfx-orchestrator/specialized_agents/api_validator.py`.
