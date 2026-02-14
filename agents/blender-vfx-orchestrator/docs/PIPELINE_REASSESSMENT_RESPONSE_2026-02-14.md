# Pipeline Reassessment Response - 2026-02-14

## Purpose
This document is a top-to-bottom reassessment of the Blender VFX orchestrator after P0 rollout, with an emphasis on why Blender 4.x/deprecated API hallucinations are still leaking into execution and how to redesign for stability before P1.

It is written as a response package you can pass directly to Claude Opus for joint planning.

---

## Inputs Reviewed

### Project analysis docs
- `agents/blender-vfx-orchestrator/docs/PIPELINE_PROBLEMS_ANALYSIS_2026-02-13.md`
- `agents/blender-vfx-orchestrator/docs/DEEP_ANALYSIS_INVESTIGATION_2026-02-12.md`

### Runtime implementation (line-level validation)
- `agents/blender-vfx-orchestrator/orchestrator.py`
- `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py`
- `agents/blender-vfx-orchestrator/guardrails/api_spec_guardrails.py`
- `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py`
- `agents/blender-vfx-orchestrator/tools/semantic_docs_tools.py`
- `agents/blender-vfx-orchestrator/specialized_agents/api_spec_agent.py`
- `agents/blender-vfx-orchestrator/specialized_agents/code_writer_agent.py`
- `agents/blender-vfx-orchestrator/models/shared_context.py`

### Ground-truth API checks (Blender MCP docs)
- `bpy.types.FluidFlowSettings.html`
- `bpy.types.FluidDomainSettings.html`

### External reliability references
- OpenAI Structured Outputs announcement: https://openai.com/index/introducing-structured-outputs-in-the-api/
- OpenAI Agents SDK guardrails docs (raw): https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/guardrails.md
- OpenAI Agents SDK tools docs (raw): https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/tools.md
- PICARD constrained decoding paper: https://arxiv.org/abs/2109.05093

---

## Executive Diagnosis
Your core diagnosis is correct: hallucinations are not a prompt-quality bug, they are a systems architecture bug.

P0 improved parts of enforcement, but current behavior still allows correctness drift because:

1. There are multiple contradictory truth sources (prompts, blacklists, hardcoded allowlists, vector chunks, docs metadata assumptions) that disagree.
2. Enforcement is not fail-closed end-to-end (parallel exception swallowing, fallback-to-legacy paths, inconsistent tripwire handling).
3. API correctness still depends heavily on LLM compliance, with regex patching after the fact.
4. Doc-grounding is strict in shape but weak in semantics (good refs can be rejected; bad reasoning can still pass).

Result: the system can create complex scenes, but cannot reliably enforce Blender 5 API validity under iteration pressure.

---

## High-Confidence Findings

### F1) Truth-source contradictions are still present

#### Evidence
- `use_absolute` is marked deprecated in `DEPRECATED_ATTRIBUTES`:
  - `agents/blender-vfx-orchestrator/guardrails/api_spec_guardrails.py:66`
- Blender API ground truth shows `FluidFlowSettings.use_absolute` is valid:
  - `bpy.types.FluidFlowSettings.html` (MCP read)
- `API_SPEC_AGENT_INSTRUCTIONS` still lists `openvdb_cache_compress_type` enum including `BLOSC`:
  - `agents/blender-vfx-orchestrator/specialized_agents/api_spec_agent.py:138`
- Guardrails and fixer paths treat `BLOSC` as removed:
  - `agents/blender-vfx-orchestrator/guardrails/api_spec_guardrails.py:63-85`
  - Blender ground truth confirms only `ZIP` and `NONE` in current docs.

#### Impact
- False positives reject valid scripts.
- Prompt layer and guardrail layer can force opposite actions.
- LLM receives conflicting constraints and regresses to memorized priors.

---

### F2) Research doc grounding is format-strict but pipeline-shape fragile

#### Evidence
- Research guardrail requires very specific path formats:
  - `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py:21-29`
- `.md` refs and sentinel values are always rejected:
  - `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py:38-42`
  - `agents/blender-vfx-orchestrator/tools/semantic_docs_tools.py:163-167`
- `semantic_docs_tools` also depends on `DocPath:` headers in chunk text:
  - `agents/blender-vfx-orchestrator/tools/semantic_docs_tools.py:129-134`
- Existing tests still use old doc_ref formats that no longer satisfy strict checks:
  - `agents/blender-vfx-orchestrator/test_parallel_phases.py:113`
  - `agents/blender-vfx-orchestrator/test_parallel_phases.py:193`
  - `agents/blender-vfx-orchestrator/test_parallel_phases.py:442`

#### Impact
- Valid research gets discarded for shape mismatch.
- Strict mode increases brittleness without guaranteeing better semantic grounding.

---

### F3) Tripwire handling is inconsistent in parallel paths

#### Evidence
- Parallel preflight gathers with `return_exceptions=True`:
  - `agents/blender-vfx-orchestrator/orchestrator.py:982`
- Exceptions become `None` results and are only logged:
  - `agents/blender-vfx-orchestrator/orchestrator.py:986-1005`
- But phase-0 caller expects guardrail tripwire abort:
  - `agents/blender-vfx-orchestrator/orchestrator.py:2108-2112`
- Technique-switch research treats guardrail tripwire as non-fatal and continues:
  - `agents/blender-vfx-orchestrator/orchestrator.py:4091-4094`

#### Impact
- Same class of guardrail failure can abort or silently degrade depending on call path.
- Reliability and debugging become non-deterministic.

---

### F4) Spec-first still has non-fail-closed fallback behavior

#### Evidence
- In spec-first pipeline, API spec or code writer errors fall back (or degrade) instead of hard-stop:
  - `agents/blender-vfx-orchestrator/orchestrator.py:1506-1512`
  - `agents/blender-vfx-orchestrator/orchestrator.py:1570-1582`
- `_apply_script_modifications` calls `_modify_script_impl` directly:
  - `agents/blender-vfx-orchestrator/orchestrator.py:819-869`
- This centralization is good, but still bypasses tool-run semantics/hook lifecycle.

#### Impact
- Guardrail discipline can be bypassed by fallback and non-tool mutation paths.
- Hallucination defenses are not guaranteed under failure conditions.

---

### F5) Script guardrails and effect taxonomy are out of sync

#### Evidence
- Script guardrail valid effect types are limited:
  - `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py:34-45`
- Canonical enum includes more production effect types:
  - `agents/blender-vfx-orchestrator/models/shared_context.py:26-43`

#### Impact
- Valid requests can fail for the wrong reason.
- Prompting/guardrail mismatch increases pointless retries.

---

### F6) Executor layer still needs heuristic override for false failure

#### Evidence
- Main execution and recovery both run render-discovery overrides to patch agent false negatives:
  - `agents/blender-vfx-orchestrator/orchestrator.py:3214-3244`
  - `agents/blender-vfx-orchestrator/orchestrator.py:3422-3446`

#### Impact
- Pipeline success still depends on repair heuristics rather than trustworthy execution outputs.
- Token/time waste remains high during failure routing.

---

### F7) Documentation drift remains non-trivial

#### Evidence
- Multiple files still reference SDK v0.7.0 docs links:
  - `agents/blender-vfx-orchestrator/orchestrator.py:1418`
  - `agents/blender-vfx-orchestrator/specialized_agents/code_writer_agent.py:12`
  - `agents/blender-vfx-orchestrator/specialized_agents/api_spec_agent.py:13`
  - `agents/blender-vfx-orchestrator/guardrails/api_spec_guardrails.py:8`
- Runtime package in project venv is `openai-agents==0.8.3`, and `Agent.as_tool` signature includes `max_turns`.

#### Impact
- Engineers and LLMs receive conflicting "truth" about SDK behavior.
- This contributes directly to wrong implementation recommendations.

---

## Why P0 Did Not Deliver Major Stability Gains

P0 added important controls, but most are still prompt-centric or regex-centric.

Current stack still works as:
1. Ask LLM to obey Blender 5 constraints.
2. Run mixed guardrails/regex checks.
3. Execute and repair after failures.

This architecture cannot eliminate hallucinations because the allowed output space is effectively open-ended until late in the pipeline.

The missing piece is a deterministic, centralized API truth compiler and fail-closed validator that runs before execution in every path.

---

## Proposed New Direction: Deterministic Correctness Core

### Principle
Move API correctness out of prompts and into a compiled Blender 5 registry + structural validators.

Prompts should handle creative decisions.
Deterministic code should own API legality.

---

## Architecture Proposal (Stability-First)

### Layer A: Blender 5 Truth Registry (single source of truth)

Build `tools/build_blender5_registry.py` that compiles API docs into machine-readable artifacts:
- `registry/types.json`
- `registry/operators.json`
- `registry/enums.json`
- `registry/deprecations.json`
- `registry/version_meta.json`

Each attribute entry should include:
- object type (`FluidDomainSettings`, `FluidFlowSettings`, etc.)
- canonical name
- type, enum values, default/range when available
- source doc ref
- optional migration alias(es) for known 4.x names

This replaces hardcoded contradictory lists spread across:
- `api_spec_guardrails.py`
- `api_validator.py`
- `dynamic_instructions.py`
- `blender_api_fixer.py`
- agent prompt blocks

---

### Layer B: AST Validator + Auto-fix (pre-execution firewall)

Add `tools/blender_api_firewall.py`:
- Parse script AST.
- Extract `dset.*`, `fset.*`, and key `bpy.ops.*` usage.
- Validate each token against registry.
- Apply deterministic alias rewrites where unambiguous.
- Reject on unknown/ambiguous attributes.

Outputs:
- `firewall_report.json` with:
  - illegal attrs
  - auto-fixes applied
  - unresolved blockers
  - source doc refs for each decision

Policy:
- Unknown critical API tokens => fail closed.
- No execution without `firewall_report.status == pass`.

---

### Layer C: Structured Generation Contract

Do not ask Code Writer to emit unconstrained free-form correctness for fluid APIs.

Instead:
1. Generate a structured scene/spec object (strict schema with enums).
2. Deterministically transpile this to Python template code.

Use schema enums from registry so deprecated names are structurally unrepresentable.

This follows constrained decoding principles:
- OpenAI Structured Outputs + strict JSON Schema
- Grammar-constrained decoding pattern (PICARD-style concept)

---

### Layer D: Fail-Closed Pipeline Semantics

Unify behavior:
- Any guardrail tripwire in required phases returns explicit phase failure artifact.
- No silent `None` substitution from `return_exceptions=True` for mandatory phases.
- Fallback-to-legacy path disabled by default in production mode.

Introduce runtime mode:
- `ORCHESTRATOR_STRICT_STABILITY=1`
  - disables legacy fallback
  - enforces firewall pass
  - enforces quality artifact presence

---

### Layer E: Prompt Simplification

Current instructions are long and internally contradictory.

Refactor prompts to:
- remove full deprecated tables duplicated across agents
- reference only generated registry snippets relevant to this request
- avoid multi-page "do not do X" blocks that fight model priors

Rule:
- Prompts should not be truth stores.
- Prompts should reference truth stores.

---

## Revised Roadmap (Before P1 Autonomy Work)

### S0 - Stabilization Hotfix (2-3 days)
1. Remove false deprecation of `use_absolute`.
2. Align effect-type guardrail with `EffectType` enum dynamically.
3. Normalize doc_ref handling with adapter (accept legacy shapes, canonicalize, then enforce).
4. Make mandatory-phase tripwires deterministic (no swallow in parallel preflight).
5. Remove contradictory `BLOSC` guidance from prompts where it conflicts with registry truth.

### S1 - Truth Compiler Sprint (5-7 days)
1. Build and version Blender 5 registry artifacts.
2. Add CI check: any deprecated token in prompt/guardrail not in registry => fail.
3. Replace hardcoded deprecated lists with generated data.
4. Generate docs summary from registry automatically.

### S2 - Firewall Integration (4-6 days)
1. Implement AST firewall and auto-fixer.
2. Insert firewall gate before every execution path (initial + recovery + modification).
3. Emit per-iteration legality telemetry.
4. Add regression corpus from failed runs (wine_pour, kitchen_leak, oil_pour, etc.).

### S3 - Structured Spec Generation (5-8 days)
1. Move fluid API generation to strict structured output.
2. Deterministic transpiler to Python script template.
3. Keep LLM free-form only for high-level scene design notes and non-critical aesthetics.

### S4 - Re-enable autonomy/self-learning milestones
Only after stability metrics (below) are achieved for 100+ runs.

---

## Acceptance Criteria (Hard Gates)

### Correctness
- `invalid_api_attribute_rate <= 1%` across 100 run corpus.
- `deprecated_attribute_hits == 0` after firewall for 100 runs.
- `guardrail_false_positive_rate < 2%`.

### Reliability
- `phase_abort_reason` always explicit (no `unknown` terminal status).
- `quality_score` always emitted (or explicit hard-failure artifact).
- `execution_success_disagreement` (tool says success but pipeline says fail) < 2%.

### Efficiency
- reduce average iterations-to-first-valid-render by 30% from current baseline.
- reduce retries caused by API errors by 70%.

---

## Suggested Experiments Before Committing Full Rewrite

### Experiment 1: Registry + Firewall only (minimal architecture change)
- Keep current agents.
- Add registry-backed pre-execution firewall.
- Measure hallucination leak reduction.

### Experiment 2: Structured scene spec + transpiler for fluid core only
- Restrict only fluid settings path to schema.
- Keep scene decoration free-form.
- Compare quality and pass rate.

### Experiment 3: Strict-mode fail-closed
- Disable legacy fallback paths for a controlled batch.
- Measure clarity of failures and remediation speed.

---

## Immediate Documentation Cleanup Targets

High-impact docs/files still carrying stale/conflicting behavior should be updated in lockstep:

1. Update SDK references currently pinned to v0.7.0 docs URLs:
   - `agents/blender-vfx-orchestrator/orchestrator.py`
   - `agents/blender-vfx-orchestrator/specialized_agents/code_writer_agent.py`
   - `agents/blender-vfx-orchestrator/specialized_agents/api_spec_agent.py`
   - `agents/blender-vfx-orchestrator/guardrails/api_spec_guardrails.py`
2. Add one canonical "runtime truth generated from registry" doc and link all prompt docs to it.
3. Mark old prompt tables as informative only, not enforcement.

---

## Questions For Claude (Proposed Joint Session Focus)

1. Do we agree to treat API legality as deterministic infrastructure, not LLM behavior?
2. Should we deprecate APISpec Agent for legality checks and retain it only for strategy recommendations?
3. Can we commit to strict-mode fail-closed execution during stabilization (temporary slower dev velocity, faster long-term)?
4. Should we split script generation into:
   - structured fluid config (strict schema),
   - free-form scene aesthetics (non-critical)?
5. What minimum benchmark suite should block merges (which scenarios + how many runs)?

---

## Final Position
You are very close on capability, but currently above the instability threshold needed for autonomy/self-improvement work.

The next major gain is not more prompt engineering. It is a deterministic correctness core:
- compiled Blender 5 truth registry,
- AST firewall,
- fail-closed phase semantics,
- simplified prompt roles.

Once that is in place, autonomy and self-learning loops can optimize quality instead of spending most tokens recovering from API validity errors.
