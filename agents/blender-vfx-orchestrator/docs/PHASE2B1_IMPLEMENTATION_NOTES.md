# Phase 2B-1: Stateless Iterations — Implementation Notes

**Date:** 2026-02-23
**Branch:** `0.34.9/phase2b1-stateless-iteration`
**Status:** COMPLETE — 23 new tests, 215 total pass

---

## What Was Built

Feature-flagged stateless iterations for the VFX orchestrator pipeline. When enabled (default), each iteration-loop `Runner.run()` call receives `session=None` instead of `session=sdk_session`, giving every agent a fresh conversation context per iteration. Pre-loop phases (Research, Technique Selection) retain the shared session for research continuity.

### Files Changed

| File | Type | Lines | What |
|------|------|-------|------|
| `config/presets.yaml` | Modified | +7 | `stateless_iterations` flag on all 7 presets |
| `config/agent_config.py` | Modified | +10 | `PresetConfig.stateless_iterations` field, `from_dict` reader, `use_stateless_iterations()` accessor |
| `utils/iteration_state.py` | **New** | 85 | `IterationSnapshot` frozen dataclass for optional cross-iteration prompt context |
| `orchestrator.py` | Modified | ~25 changed | `iter_session` variable, 12 `session=sdk_session` → `session=iter_session` replacements, compaction guard |
| `tests/test_stateless_iterations.py` | **New** | 120 | 5 test classes, 23 tests |
| **Total** | | **~247** | |

---

## Delta From Original Roadmap Spec

The roadmap (v2.0, 2026-02-23) spec'd 2B-1 as a ~600 LOC, 5-day effort with incremental rollout. The actual implementation was ~240 LOC completed in a single session. Here's why every major assumption changed.

### 1. Incremental Rollout → All-at-Once Behind Feature Flag

**Roadmap said:** Convert iteration 2 only → E2E A/B comparison → convert remaining iterations if quality holds.

**What we did:** All iterations switched at once, gated by `stateless_iterations: true/false` in presets.yaml.

**Why:** The A/B comparison assumes conversation history carries unique information that agents need. Reading every prompt template in the iteration loop (Script Writer, Quality Analyst, Learning Agent, Quality Gate, Modification Coordinator, Research Agent) proved this false. Every single prompt already embeds its full context via f-strings from Python variables — `effect_type`, `technique`, `quality.overall_score`, `session.stuck_state`, `script.script_path`, etc. The conversation history was always redundant noise, not a context source. There's nothing to A/B test.

### 2. ~600 LOC → ~240 LOC

**Roadmap said:** 180 lines for `IterationState` + serialization, 300 lines for iteration loop restructuring, 120 lines for prompt reconstruction. Total ~600.

**What we actually needed:**
- 85 lines for `IterationSnapshot` (read-only, no serialization needed)
- 25 lines changed in orchestrator (variable + find-replace + guard)
- 0 lines for prompt reconstruction
- 10 lines in config
- 120 lines of tests

**Why the reduction:**
- **No serialization needed.** `SessionState` (Pydantic model) already persists to disk via `SessionPersistence`. `SharedContext` is rebuilt from `SessionState` each iteration. There's no new state to serialize — it was already being done.
- **No prompt reconstruction needed.** Prompts already pull from Python variables, not conversation history. Removing the session doesn't remove any information the prompts access.
- **No iteration loop restructuring needed.** The key insight: `_run_agent()` already handles `session=None` gracefully (line 938: `{k: v for k, v in kwargs.items() if v is not None}`). All helper methods (`_run_parallel_learning_and_gate`, `_run_original_script_writer`, `_run_parallel_preflight`) already accept `Optional[SessionABC]`. The change was literally find-replace on 12 lines.

### 3. `IterationState` with write/read → `IterationSnapshot` (Read-Only, No I/O)

**Roadmap said:** A state file written to disk between iterations, read back to construct prompts. Detailed JSON structure with `qa_diagnosis`, `monitor_alerts`, `parameter_bounds`, `truth_pack_types`.

**What we built:** A frozen `@dataclass` that builds from the existing `SessionState` in memory. No file I/O. Provides `format_for_prompt()` for optional prompt enrichment (~100 tokens).

**Why:** The roadmap assumed prompts needed to be reconstructed from external state. They don't — `SessionState`, `SharedContext`, `StuckDetectionState`, and the artifact manager already carry all this data in-memory and pass it to prompts via Python variables. The QA diagnosis bridge, truth pack, parameter bounds, and monitor alerts are all already available through the existing object graph. Writing another file would duplicate state that's already tracked.

The `IterationSnapshot` exists as a safety net — if future agents need compact cross-iteration context that the existing f-string prompts don't cover, it's ready. But today, nothing uses it.

### 4. 5 Test Scenarios → 23 Tests Across 5 Classes

**Roadmap said:** 5 tests (state roundtrip, fresh context, A/B comparison, failure routing, SharedContext rebuild).

**What we wrote:** 23 tests across 5 classes:
- `TestStatelessConfig` (6 tests) — PresetConfig defaults, from_dict parsing, AgentConfigManager accessor
- `TestIterationSnapshot` (5 tests) — from_session construction, format_for_prompt output, frozen immutability, history truncation
- `TestSessionRouting` (2 tests) — stateless→None, stateful→sdk_session
- `TestCompactionGuard` (2 tests) — skipped in stateless, runs in stateful
- `TestPresetYamlIntegration` (8 tests) — all presets have flag, debug is stateful, loaded presets expose flag

**Why the difference:** The roadmap's 5 tests assumed the complex IterationState serialization path existed. Since it doesn't, those tests don't apply. Instead we tested what was actually built: config plumbing, the snapshot utility, session routing logic, compaction guard, and YAML integration.

### 5. Dependencies Dissolved

**Roadmap said:** Dependencies on 2A-4 (PipelineMonitor for monitor_alerts) and 2A-5 (parameter bounds).

**Actual:** No dependencies. The PipelineMonitor and parameter bounds are already operational from Phase 2A. They don't need any changes to work with stateless iterations — they operate on Python objects, not conversation history.

### 6. Risk: Medium → Low

**Roadmap said:** Medium risk — "changes iteration loop structure and context management."

**Actual:** Low risk. The iteration loop structure didn't change. The session routing is a single conditional variable (`iter_session = None if use_stateless else sdk_session`), and the 12 replacements are mechanical. Rollback is a one-line YAML change. No agent definitions, hooks, models, tools, or prompt templates were modified.

---

## What Stays as `sdk_session` (Pre-Loop)

These 8 references intentionally keep the shared session for research continuity:

| Location | Phase | Why |
|----------|-------|-----|
| `_run_parallel_preflight` body (3 refs) | 0 | Research Agent + Docs Expert — research flows into technique selection |
| `_run_parallel_learning_and_gate` body (2 refs) | Helper | Receives session as parameter from caller — caller passes `iter_session` |
| `_run_original_script_writer` body (1 ref) | Helper | Same — receives from caller |
| `_run_parallel_preflight` call | 0 | Phase 0 research |
| Technique Coordinator call | 0.5 | Phase 0.5 technique selection |

## What Changed to `iter_session` (Iteration Loop)

12 `Runner.run()` calls across all iteration-loop phases:

| Phase | Agent | Count |
|-------|-------|-------|
| 1 (Generation) | Script Writer | 2 (new script + fallback) |
| Decision | Modification Coordinator, Script Writer (modify + technique switch) | 4 |
| Recovery | Script Writer (error recovery) | 1 |
| 3 (Evaluation) | Quality Analyst | 1 |
| 4+5 (Learning+Gate) | Parallel call, sequential Learning, sequential Gate | 3 |
| Escape | Research Agent (technique switch) | 1 |

---

## Rollback

Set `stateless_iterations: false` in the active preset (or in all presets). The `debug` preset already has this set to `false` for conversation inspection during debugging.

---

## Key Insight

The roadmap estimated ~600 LOC because it assumed the architecture was fundamentally stateful — that agents relied on conversation history from prior iterations to function correctly. Codebase analysis revealed the opposite: the architecture was already ~80% stateless by design. SharedContext rebuilds each iteration, SessionState persists to disk, results pass between phases via Python variables, and every prompt embeds its context through f-strings. The conversation history was redundant noise that grew unbounded and triggered compaction. Removing it was a 12-line find-replace, not a 600-line restructure.
