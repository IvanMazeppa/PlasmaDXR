# Multi‑Agent Architecture Addendum (2026‑01‑29)

This addendum proposes a concrete, implementable multi‑agent architecture for the Blender VFX orchestrator. It is designed to stabilize autonomous operation, reduce regressions, and make improvements measurable.

## Goals
- **Determinism for critical steps** (bake/render/export) while still allowing agent creativity.
- **Observable, replayable runs** with reliable artifacts and checkpoints.
- **Self‑improvement** through explicit evaluation and learnings storage.

## Core Roles
1) **Coordinator (Orchestrator)**
   - Owns the state machine and sequencing.
   - Decides which agent runs next and why.
   - Enforces guardrails and gating criteria.

2) **Script Writer**
   - Generates Blender Python scripts from specs.
   - Emits a **script contract** (expected outputs, cache range, render count, bake settings).

3) **Executor**
   - Executes Blender headless runs.
   - Never edits scripts except through the API fixer (policy‑enforced).

4) **Quality Analyst**
   - Evaluates outputs against simple gates and optional ML metrics.
   - Produces a **scorecard** and “pass/fail” gating result.

5) **Docs / Research Agent**
   - Resolves API questions from MCP (Blender manual + API).
   - Emits a **verified API note** with exact property names and usage.

## State Machine (Minimal)
```
PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE

If FAIL in VALIDATE: return to GENERATE with fixes.
If FAIL in EXECUTE: route to DIAGNOSE → FIX → EXECUTE.
If FAIL in EVALUATE: route to IMPROVE → GENERATE.
```

### State Definitions
- **PLAN**: Define constraints, scene objectives, and required outputs.
- **GENERATE**: Produce script + metadata.
- **VALIDATE**: Run API fixer + static checks (known bad patterns).
- **EXECUTE**: Headless Blender run; emit artifacts.
- **EVALUATE**: Gate on outputs + quality metrics.
- **DECIDE**: Accept, refine, or revert.

## Guardrails & Gating
- **Render loop limits**: cap to representative frames.
- **Bake gates**: ensure cache size > threshold; reject empty caches.
- **Output artifact gates**: require blend + log + render set.
- **Write protection**: never overwrite golden outputs unless pass criteria is met.

### Suggested Gate Checklist
- Cache directory exists AND size >= minimum threshold.
- Exactly N renders created (3 by default).
- No Blender fatal errors in logs.
- Optional: asset evaluation score >= threshold.

## Artifact‑First Handoffs
Every agent handoff should be a **file reference** instead of inline dumps.

Examples:
- Script Writer → `assets/blender_scripts/generated/<run_id>.py`
- Executor → `build/blender_cli_logs/<run_id>/`
- Analyst → `build/vfx_output/<run_id>/scorecard.json`

## Autonomy & Learning Loop
- Store run outcomes in a **learning registry** (issues, fixes, wins).
- Promote recurring fixes into the API fixer (or a dedicated “auto‑repair” layer).
- Use a “failure taxonomy” to route to the correct specialist agent.

## Operational Telemetry
- **Per‑run manifest** (json): inputs, outputs, environment, decisions.
- **Cost + duration counters** for each phase.
- **Diff‑aware fixes**: record what was changed and why.

## Minimal Implementation Plan
1) **Add a run manifest** written by Coordinator.
2) **Add validation gates** (cache size, render count, error‑free logs).
3) **Formalize artifacts** and update handoffs to file references.
4) **Enable learning registry** (append‑only)
5) **Add a critic loop** to reject low‑quality runs.

## Design Principles
- **Deterministic first, adaptive second**: the pipeline must be repeatable.
- **Fix once, capture forever**: any reliable fix becomes a rule.
- **Reduce context load**: logs and scripts should be files, not chat history.

## Integration Notes
- The API fixer is now a critical “policy enforcement” layer.
- The Executor should never bypass the fixer.
- All bake/render settings should be centralized in the config layer and validated pre‑execution.

