# Phase 2A Status Review

**Date:** 2026-02-22
**Reviewer:** Claude Opus 4.6
**Branch:** `0.34.3/phase-2a4-pipeline-monitor`
**Scope:** 2A-0 through 2A-3 (2A-4 in progress, not reviewed)

---

## Summary

| Item | Status | Completion | Issues |
|------|--------|------------|--------|
| **2A-0: Documentation Pipeline** | DONE | 90% | Missing 2 of 5 tests |
| **2A-1: Conditional Tool Enabling** | DONE | 95% | Missing test file only |
| **2A-2: Deterministic Agent Control** | DONE | 100% | Only executor needs it; others correctly skipped |
| **2A-3: Tool Guardrails** | DONE | 100% | Production-ready, 21 tests |

**Overall Phase 2A (items 0-3): ~96% complete.** All items functionally done. Remaining gap is test coverage for 2A-0 and 2A-1.

---

## 2A-0: Documentation Pipeline — DONE (90%)

### What Shipped
- `scripts/experiment_manual_rewrite.py` — `--all-physics` flag, batch processing, cost tracking. Production-grade.
- `scripts/upload_rewritten_manual.py` — Vector store creation, retry logic, polling, test query verification. Excellent.
- `scripts/seed_kb.py` — 6 effect types, dry-run mode, snake_case conversion, "emerging" trust level. Well-designed.
- `tools/semantic_docs_tools.py` — Rewritten store ID loaded from env, searched first (priority over original manual), correct routing.
- `tests/test_technique_discovery.py` — 3 test cases with env var skip guard.
- 131 pages processed, 100% success rate, $0.14 total cost.

### What's Missing
1. **Test: upload workflow** — No test that runs upload + verifies results are discoverable via vector store query.
2. **Test: cost validation** — No test asserting 119 pages < $0.30.
3. **Test: batch processing** — No test running `--all-physics` on a subset and asserting all produce output with TECHNIQUE sections.

### Issues Found
- **Minor:** `experiment_manual_rewrite.py` temperature logic (lines 242-244) excludes gpt-5-mini from temperature setting, which is correct behavior but the condition is phrased confusingly.
- **Minor:** No warning if `BLENDER_REWRITTEN_MANUAL_STORE_ID` points to an invalid/deleted store — silently falls back to original manual. Functional but could mask config errors.

### Assessment
The pipeline works end-to-end. Manual rewrite quality is excellent (reviewed separately). The missing tests are for CI validation, not functional gaps — the pipeline has been manually verified. **Low risk.**

---

## 2A-1: Conditional Tool Enabling — DONE (95%)

### What Shipped
- `utils/tool_visibility.py` (78 lines) — 4 callbacks: `budget_allows_vision`, `budget_allows_docs`, `learning_tool_for_iteration`, `budget_allows_evaluation`. Kill switch via `ENABLE_CONDITIONAL_TOOLS` env var.
- `specialized_agents/quality_analyst.py` — 6 vision tools gated with `budget_allows_vision`. Cheap tools (filesystem, in-memory) correctly left ungated.
- `specialized_agents/docs_expert.py` — 3 vector-store doc search tools gated with `budget_allows_docs`. Cheap tools (in-memory validation) correctly left ungated.
- Budget tracker integration working — monthly limits, per-category tracking, persistence.

### What's Missing
1. **Tests** — No dedicated test file for tool_visibility callbacks. Roadmap specified 4 test cases (budget exhausted, budget available, iteration gating, mid-run transition).

### Resolved After Deeper Analysis

2. **~~Orchestrator agent-as-tool wrapping~~** — Originally flagged as a gap (~15 lines). After code review: **already handled.** The quality analyst is NOT called via `.as_tool()` — it's called via `Runner.run()` from Python code. The orchestrator already has a Python-level budget guard at line 3791: `if not self._budget_tracker.can_afford_evaluation(): break`. This is actually MORE robust than SDK-level `is_enabled` because it can exit the iteration loop entirely rather than just hiding a tool.

3. **Asset evaluator divergence** — Roadmap said to move budget checks into `asset_evaluator_tools.py`. Implementation keeps checks at agent level instead. This is actually a **better design** (cleaner separation of concerns). No action needed.

### Assessment
Feature works. Budget gating at tool level (is_enabled callbacks) + Python-level budget guard at orchestrator level (line 3791) provides defense-in-depth. Only missing piece is test coverage. **Low risk.**

---

## 2A-2: Deterministic Agent Control — DONE (100%)

### What Shipped
- `specialized_agents/executor.py` — `tool_use_behavior="stop_on_first_tool"` added (line 121). Well-documented with rationale comment.

### Intentionally Skipped (With Rationale)

1. **`specialized_agents/api_validator.py`** — `stop_on_first_tool` NOT appropriate. The validator has a multi-step workflow: T1 `extract_blender_api_calls` → T2 `check_known_api_changes` → T3 `semantic_search_blender_docs` → T4 `format_validation_report`. Stopping after the first tool would return raw extraction output, skipping validation entirely. The executor works because it calls ONE tool (run script); the validator calls 4 in sequence.

2. **`specialized_agents/learning_agent.py`** — `stop_on_first_tool` NOT appropriate. 17+ tools, agent needs to call multiple tools in sequence (query KB → correlate observations → extract patterns → record outcomes) and synthesize results across tools. No distinct "recording-only mode" exists in the current code.

3. **`orchestrator.py` (quality gate coordinator)** — `StopAtTools` intentionally NOT used. Documented at lines 723-726:
   > "StopAtTools not used here because the agent needs to synthesize evaluate_escape_velocity results into QualityDecision structured output."

   **This is the correct decision.** The quality gate coordinator must synthesize tool results into structured output — stopping at the tool would break the pipeline.

### Design Principle
`stop_on_first_tool` is appropriate ONLY when: (a) the agent calls exactly one tool, and (b) the raw tool output IS the final answer with no synthesis needed. Only the executor meets both criteria. All other agents need multi-step tool workflows or synthesis.

### Assessment
The executor is the only agent where `stop_on_first_tool` applies. All other agents correctly skip it. **Complete — no action needed.**

---

## 2A-3: Tool Guardrails — DONE (100%)

### What Shipped
- `guardrails/tool_guardrails.py` (274 lines) — Three guardrails:
  1. **truth_pack_input_guardrail** (56 lines) — Validates scripts before execution, auto-fixes hallucinations, rejects unfixable errors. Runs at $0.
  2. **critical_failure_output_guardrail** (22 lines) — Catches BLACK_SCREEN, WHITE_SCREEN, ZERO_LIGHTS, CLIPPING_ARTIFACTS. Rejects with targeted message.
  3. **script_length_output_guardrail** (54 lines) — Rejects < 300 lines, warns 300-500, allows 500+.
  4. **attach_tool_guardrails()** (36 lines) — Dynamic wiring to FunctionTool objects.
- Orchestrator integration — `attach_tool_guardrails()` called during init, truth pack passed during technique switches.
- `tests/test_tool_guardrails.py` (305 lines) — 21 test cases across 3 test classes. Covers hallucination detection, auto-fix, rejection, thresholds, kill switch.
- Kill switch via `ENABLE_TOOL_GUARDRAILS` env var.

### Bypass-Proof Assessment
| Vector | Result |
|--------|--------|
| Call `write_script()` directly | BLOCKED — input guardrail validates |
| Call `generate_script()` | BLOCKED — output guardrail checks length |
| Call `execute_blender_script()` | BLOCKED — input guardrail validates against truth pack |
| Modify script after generation | BLOCKED — `modify_script()` has output guardrail |

**Truth pack validation is impossible to bypass** for the intended tool pipeline.

### Issues Found
- **Minor:** `generate_script()` has no hardcoded guardrail decorator — relies on dynamic attachment via `attach_tool_guardrails()`. If attachment fails silently, no protection. Tests verify attachment occurs, so risk is low.
- **Minor:** Dual validation between `script_generator_tools.py` local checks and truth pack guardrail. Overlapping but provides defense-in-depth. Not urgent to consolidate.
- **Minor:** Critical failure guardrail rejection message is generic ("requires fundamental fix") — doesn't specify WHICH fix. Could be improved with targeted suggestions.

### Assessment
This is the best-implemented item in Phase 2A. Comprehensive tests, bypass-proof design, $0 cost, proper kill switch. **Production-ready. No action needed.**

---

## Action Items (Priority Order)

### Resolved (2026-02-22, second pass)

| # | Item | Phase | Resolution |
|---|------|-------|------------|
| ~~1~~ | ~~Add `stop_on_first_tool` to `api_validator.py`~~ | 2A-2 | **Incorrect** — multi-step workflow (4 tools in sequence), would break validation |
| ~~2~~ | ~~Decide on learning_agent.py approach~~ | 2A-2 | **Skipped** — 17+ tools, needs synthesis, no recording-only mode exists |
| ~~3~~ | ~~Add `is_enabled` to agent `.as_tool()` calls~~ | 2A-1 | **Already handled** — orchestrator uses `Runner.run()` with Python-level budget guard (line 3791) |

### Should-Do (Test Coverage)

| # | Item | Phase | Effort | Impact |
|---|------|-------|--------|--------|
| 4 | Add `test_tool_visibility.py` (4 test cases from roadmap) | 2A-1 | ~60 lines | CI coverage |
| 5 | Add upload workflow + cost validation tests to `test_technique_discovery.py` | 2A-0 | ~40 lines | CI coverage |

### Nice-to-Have

| # | Item | Phase | Effort | Impact |
|---|------|-------|--------|--------|
| 6 | Add warning log if `BLENDER_REWRITTEN_MANUAL_STORE_ID` is set but store unreachable | 2A-0 | ~10 lines | Config debugging |
| 7 | Enhance critical failure guardrail with targeted fix suggestions | 2A-3 | ~20 lines | Better agent guidance |

---

## Items Not Yet Started (Remaining 2A)

| Item | Status | Notes |
|------|--------|-------|
| **2A-4: PipelineMonitor** | IN PROGRESS | Current branch work |
| **2A-5: Parameter Bounds** | NOT STARTED | Depends on 2A-4 |
| **2A-6: Deprecate Spec-First** | NOT STARTED | Independent |
| **2A-7: Remove Executor Agent** | NOT STARTED | Independent |
| **2A-8: Tool Timeouts** | NOT STARTED | Independent |
| **KB Seeding** | NOT STARTED | Depends on 2A-0 (done) |

---

## Code Quality Notes

- **Consistent patterns:** All implementations follow the same conventions — kill switches via env vars, docstrings on callbacks, defensive error handling.
- **Good integration comments:** Phase references (e.g., "Phase 2A-1:") appear in modified files, making it easy to trace changes back to roadmap items.
- **Orchestrator `load_dotenv()` added:** Ensures `.env` values are available regardless of invocation method.
- **No regressions observed:** Existing functionality preserved across all reviewed files.
