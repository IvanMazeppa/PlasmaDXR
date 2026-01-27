# Hallucination Stabilization Plan (Phase A)

**Date:** 2026-01-26  
**Owner:** blender-vfx-orchestrator  
**Status:** Drafted for implementation

## Purpose

Stop API hallucinations from reaching Blender **without reducing autonomy**.  
This plan hardens validation, enforces structured modifications, and makes RAG authoritative.

## Scope

- Keep the existing multi-agent architecture.
- Add guardrails and structured contracts, not a rebuild.
- Prioritize low-risk changes that immediately stop invalid API usage.

## Evidence

- Shakedown (water spill) failed due to invalid attribute usage (`flow.velocity_factor` on `bpy.types.Object`).
- Spec-first still falls back due to API Spec Agent max-turns.
- Coordinator modifications often apply **zero changes** due to prose/unsupported keys.

## Phase A (P0) Objectives

1) **Block invalid APIs before execution**
2) **Make doc results authoritative (no guessing)**
3) **Force structured, patchable modification outputs**

## Phase A Tasks

### A1 — Tool-level guardrails on script tools

- Add input/output guardrails to `write_script` and `modify_script`
- Reject known hallucinations and invalid attribute patterns
- Fail fast if banned API usage appears in generated code

### A2 — ScriptWriter output guardrail

- Parse generated script after `write_script`
- Tripwire if unverified/known-hallucinated APIs are present
- Prevents invalid scripts from reaching Executor

### A3 — Enforce structured modifications

- Coordinator outputs **flat, patchable keys** only  
  Example: `{"FluidFlowSettings.velocity_factor": 0.7}`
- Add **delete sentinel** for removals (e.g., `"__DELETE__"`)
- Update guardrails to reject prose/nested output

### A4 — Self-questioning + refusal examples

- Add prompt section requiring verification for each attribute
- Add few-shot “I cannot use X” examples (velocity_multi, resolution_divisions)
- Explicit: “If not in docs, do not use”

### A5 — Utility-based selection (SICA)

- Compute utility per iteration: `U = 0.5*score_norm + 0.25*(1 - cost/10) + 0.25*(1 - time/300)`
- Track best utility iteration and keep that script as the preferred output

## Success Criteria

- No Blender executions with known hallucinations
- Coordinator changes applied (non-empty `changes_made`)
- Spec-first/ScriptWriter outputs pass guardrails without fallback

## References

- `docs/research/HALLUCINATION_PREVENTION.md`
- `docs/research/PRACTICAL_HALLUCINATION_FIXES.md`
- `docs/research/OPENAI_GUARDRAILS.md`
- `docs/research/MULTI_AGENT_PATTERNS.md`
