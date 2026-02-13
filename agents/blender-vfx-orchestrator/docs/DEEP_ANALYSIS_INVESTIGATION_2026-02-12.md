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

### 9) Medium: SDK version truth is inconsistent across docs vs runtime dependency
Evidence:
- Runtime dependency pins `openai-agents==0.8.3`: `agents/blender-vfx-orchestrator/requirements.txt:10`.
- Truth docs still state SDK v0.7.0: `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md:36`, `agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md:7`.
Impact:
- Engineering decisions may follow stale behavior assumptions.
Recommendation:
- Update all SDK truth docs to match installed version and revalidate claimed API patterns.

### 10) Medium: Regex-centric script mutation remains brittle for structural fixes
Evidence:
- `_modify_script_impl` performs broad regex replacements, including chained attribute patterns: `agents/blender-vfx-orchestrator/tools/script_generator_tools.py:948-1077`.
- Roadmap itself flags direct-modify routing and contract stability as unresolved risks: `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md:74-77` and `:101-103`.
Impact:
- Complex code changes can silently fail or partially apply.
Recommendation:
- Replace regex-heavy mutation for non-trivial changes with regenerate-from-spec or AST-safe editing.

## Root-Cause Themes
- Contract drift: design docs and runtime behavior diverged.
- Enforcement holes: safeguards exist, but critical paths bypass them.
- Observability gaps: key artifacts/metrics are not consistently emitted.

## Prioritized Remediation Plan

### P0 (Immediate)
1. Enforce no direct `_impl` modifications from orchestrator.
2. Implement explicit DIAGNOSE -> FIX phase for EXECUTE and gate failures.
3. Make missing quality artifact a hard failure.
4. Enforce strict `DocPath` validation.

### P1 (Short Term)
1. Unify effect-type guardrail with `EffectType` enum.
2. Emit run manifest artifacts and add tests for manifest existence.
3. Update SDK truth docs to pinned runtime version.

### P2 (Hardening)
1. Reduce regex mutation scope and adopt structured patching/regeneration for code fixes.
2. Add state-machine conformance tests asserting documented routing.
3. Add telemetry conformance tests for quality/scorecard/manifest per iteration.

## Suggested Verification Tests
- State-machine test: injected EXECUTE failure must produce DIAGNOSE and FIX artifacts before re-exec.
- Enforcement test: any attempt to call `_modify_script_impl` from pipeline path fails CI.
- Doc fidelity test: reject research output containing blank/null `doc_path`.
- Artifact test: each iteration must produce `quality`, `scorecard`, `iteration`, and `manifest` artifacts.
