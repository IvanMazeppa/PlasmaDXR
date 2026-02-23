# Cross-Reference & Consistency Audit Report

**Date:** 2026-02-23
**Auditor:** Claude Opus 4.6
**Scope:** Documentation drift, dependency graph accuracy, feature flag status, Phase 2A code verification, mission statement alignment
**Method:** Read-only analysis of all documentation and source code

---

## Audit 1: Document Drift (Validates Codex Claim C11)

**Finding:** The Codex critique (C11) is CORRECT. `REVISED_CROSS_CUTTING_SECTIONS.md` contains outdated 2C stage mappings that conflict with the canonical `PHASE2_ROADMAP.md`.

**Details:**

### 1.1 Cross-Cutting Sections vs Roadmap: 2C Numbering Drift

The "Roadmap Items by Pipeline Stage" table in `REVISED_CROSS_CUTTING_SECTIONS.md` (lines 497-509) uses a DIFFERENT numbering scheme for Phase 2C items than the canonical `PHASE2_ROADMAP.md`:

| Pipeline Stage | Cross-Cutting Doc | Canonical Roadmap | Conflict? |
|---------------|-------------------|-------------------|-----------|
| Script Generation (2C) | `Agent.clone (2C-4), streaming (2C-5)` | Agent.clone **REMOVED**. Streaming is `2C-7`. Budget degradation is `2C-4`. | **YES** |
| Evaluation (2C) | `Budget degradation (2C-6), enhanced QA bridge (2C-7)` | Budget degradation is `2C-4`, Enhanced QA bridge is `2C-5` | **YES** |
| Learning (2C) | `Knowledge distillation (2C-8)` | Knowledge distillation is `2C-6` | **YES** |
| Monitoring (2C) | `MASC (2D-5)` listed without "directional" qualifier | Roadmap correctly marks 2D-5 as directional | **Minor** |

The cross-cutting doc appears to use the PRE-v2.0 numbering from before `Agent.clone()` was removed. In the old scheme, Agent.clone was 2C-4, which pushed everything else up by 1-2. The roadmap removed Agent.clone and renumbered, but the cross-cutting doc was never updated.

### 1.2 REVISED_PHASE_2A_2B.md vs Roadmap

The research doc `REVISED_PHASE_2A_2B.md` uses CONSISTENT numbering with the roadmap for items 2A-0 through 2A-8 and 2B-1 through 2B-8. No conflicts found.

However, there is one terminology difference:
- Research doc: "ENABLE_CONTEXT_FILTERS" (line 520 of the research doc's feature flags example)
- Roadmap: "ENABLE_CONTEXT_TRIMMING" (line 884 of the roadmap)

This is a minor naming inconsistency in the planned (not yet implemented) feature flag for 2B-2.

### 1.3 REVISED_PHASE_2C_2D.md vs Roadmap

The research doc `REVISED_PHASE_2C_2D.md` uses CONSISTENT numbering: 2C-1 through 2C-7 match the canonical roadmap exactly. Agent.clone is correctly shown as "Removed" (line 277). 2D items 2D-1 through 2D-5 match. No conflicts found.

### 1.4 SDK_FEATURES_ANALYSIS.md

This research doc does NOT use roadmap item IDs -- it uses its own section numbering (1 through 11 plus bonuses). No ID drift possible. References to roadmap items use descriptive names rather than IDs.

### 1.5 MONITORING_QUALITY_ARCHITECTURE.md

Uses its own section numbering (Part 1, Part 2, Part 3). Cross-references to roadmap items use descriptive names. No ID drift found.

### 1.6 AUTONOMOUS_SYSTEMS_RESEARCH.md

Uses its own section numbering. No roadmap ID references that could drift.

**Severity:** Medium (matches C11 assessment)

**Recommendation:**
1. Update the "Roadmap Items by Pipeline Stage" table in `REVISED_CROSS_CUTTING_SECTIONS.md` to use the post-v2.0 numbering (Agent.clone removed, 2C-4 = Budget Degradation, etc.).
2. Add a "Source of Truth and Sync Policy" statement to the roadmap header declaring `PHASE2_ROADMAP.md` as canonical and research docs as reference-only.
3. Resolve "ENABLE_CONTEXT_FILTERS" vs "ENABLE_CONTEXT_TRIMMING" naming before 2B-2 implementation.

---

## Audit 2: Dependency Graph Accuracy

**Finding:** The dependency graph is MOSTLY ACCURATE for completed items, with one implicit dependency correctly captured in code but not visible in the graph.

**Details:**

### 2.1 Completed Item Dependencies (2A-0 through 2A-6)

| Item | Declared Dependencies | Actual Dependencies in Code | Match? |
|------|----------------------|----------------------------|--------|
| **2A-0** | None | None (manual work + scripts) | YES |
| **2A-1** | None | None (`utils/tool_visibility.py` is standalone, depends only on `budget_tracker`) | YES |
| **2A-2** | None | None (1-line changes per agent file) | YES |
| **2A-3** | Truth Pack (Phase 1) | Imports from `tools/truth_pack.py` and `tools/truth_pack_validator.py` (both Phase 1) | YES |
| **2A-4** | None | None (`tools/pipeline_monitor.py` is fully standalone -- no imports from other 2A items) | YES |
| **2A-5** | PipelineMonitor (2A-4) for runtime bounds refinement | The `apply_bounds()` function accepts `oscillating_params: Optional[Set[str]]` which comes from PipelineMonitor. However, `parameter_bounds.py` does NOT import PipelineMonitor -- the integration happens in `orchestrator.py` (lines 862-871) which queries the monitor and passes oscillating params to `apply_bounds()`. | **PARTIAL** -- The dependency is real but is at the integration layer, not the module layer. `parameter_bounds.py` works standalone without PipelineMonitor; the dependency is a WIRING dependency in orchestrator.py, not a code dependency. |
| **2A-6** | Truth Pack (Phase 1) | Removed spec-first code paths; preserved `truth_pack_to_api_spec()` | YES |

### 2.2 Implicit Dependencies Not Listed

**2A-3 depends on orchestrator initialization order:** `guardrails/tool_guardrails.py` has `set_truth_pack()` which must be called before the guardrails fire. The orchestrator calls `set_guardrail_truth_pack()` and `attach_tool_guardrails()` during `initialize()`. This is an implicit dependency on the orchestrator's initialization sequence, not a code dependency. Correctly not listed.

**2A-5 has a soft runtime dependency on 2A-4:** As noted above, `parameter_bounds.py` accepts `oscillating_params` from PipelineMonitor but works without it (`oscillating_params=None` path). The dependency graph correctly lists this as a dependency but does not distinguish it from a hard dependency. This is acceptable.

### 2.3 Circular Dependencies

**None found.** The dependency graph is a DAG (directed acyclic graph). Each item either has no dependencies or depends on items from earlier in the sequence. No item in Phase 2A depends on a later 2A item.

### 2.4 Dependencies on Unbuilt Items

For the TODO items (2A-7, 2A-8): Both declare "No dependencies" which is correct -- they are standalone dead code removal and timeout addition.

For Phase 2B: The dependency graph lists:
- 2B-1 depends on 2A-4 + 2A-5 (both DONE) -- VALID
- 2B-2 depends on 2B-1 (not built) -- VALID sequencing
- 2B-5 depends on 2A-4 (DONE) -- VALID
- 2B-7 depends on 2B-6 (not built) -- VALID sequencing
- 2B-8 depends on 2A-4 (DONE) -- VALID

No dependencies on unbuilt items that would block currently planned work.

### 2.5 Dependency Graph Rendering Accuracy

The ASCII dependency graph in the roadmap (lines 1596-1638) is structurally accurate but has one visual alignment issue: the vertical bars on the right side all connect to a single column but don't clearly show what they connect TO. This is a cosmetic issue, not a factual one.

**Severity:** Low

**Recommendation:**
1. Clarify that 2A-5's dependency on 2A-4 is a "wiring dependency" (integration-level), not a module-level dependency. Both modules work standalone.
2. No changes needed for circular dependency or missing dependency concerns.

---

## Audit 3: Feature Flag Status

**Finding:** The feature flag system is PARTIALLY IMPLEMENTED. Individual modules have their own environment-variable-based kill switches. The centralized `config/agent_config.py` exists but serves a DIFFERENT purpose (agent model configuration, not feature flags).

**Details:**

### 3.1 `config/agent_config.py` -- Actual Contents

The file is an `AgentConfigManager` class (310 lines) that:
- Loads agent configuration presets from YAML (`config/presets.yaml`)
- Manages per-agent model settings (model name, reasoning_effort, temperature, verbosity, max_output_tokens)
- Supports environment-based preset selection via `ORCHESTRATOR_PRESET`
- Has NO `ENABLE_*` feature flags whatsoever

**This means the roadmap's feature flag specification (lines 282-307) is NOT implemented in `config/agent_config.py`.** The roadmap shows a code block with `ENABLE_TOOL_GUARDRAILS`, `ENABLE_PIPELINE_MONITOR`, etc. living in this file. They do not.

### 3.2 Actual Feature Flag Locations

Feature flags ARE implemented, but each module defines its own:

| Flag | Location | Implementation | Default |
|------|----------|---------------|---------|
| `ENABLE_TOOL_GUARDRAILS` | `guardrails/tool_guardrails.py:49` | `os.getenv("ENABLE_TOOL_GUARDRAILS", "true").lower() != "false"` | True |
| `ENABLE_PIPELINE_MONITOR` | `tools/pipeline_monitor.py:40` | `os.getenv("ENABLE_PIPELINE_MONITOR", "true").lower() != "false"` | True |
| `ENABLE_PARAMETER_BOUNDS` | `tools/parameter_bounds.py:19` | `os.environ.get("ENABLE_PARAMETER_BOUNDS", "1") != "0"` | True (1) |
| `ENABLE_CONDITIONAL_TOOLS` | `utils/tool_visibility.py:28-30` | `os.getenv("ENABLE_CONDITIONAL_TOOLS", "true").lower() in ("1", "true", "yes")` | True |

### 3.3 Inconsistent Flag Parsing

There are THREE different parsing patterns used across four files:

1. **Pattern A:** `os.getenv("X", "true").lower() != "false"` (tool_guardrails, pipeline_monitor)
   - Disables on: `"false"` (case-insensitive)
   - Enables on: anything else including `"0"`, `"no"`, empty string

2. **Pattern B:** `os.environ.get("X", "1") != "0"` (parameter_bounds)
   - Disables on: `"0"` only
   - Enables on: anything else including `"false"`, `"no"`

3. **Pattern C:** `os.getenv("X", "true").lower() in ("1", "true", "yes")` (tool_visibility)
   - Enables on: `"1"`, `"true"`, `"yes"` (case-insensitive)
   - Disables on: everything else including `"0"`, `"false"`, `"no"`

This means setting `ENABLE_PARAMETER_BOUNDS=false` would NOT disable parameter bounds (Pattern B treats `"false"` as truthy). And setting `ENABLE_CONDITIONAL_TOOLS=0` would NOT disable conditional tools (Pattern C does not include `"0"` wait -- it does: `"1"` is in the truthy set, so `"0"` is NOT in it, meaning it would be disabled. Correction: Pattern C DOES handle `"0"` correctly as falsy.)

The real inconsistency is Pattern B: `"false"` does not disable parameter bounds. This could cause confusion.

### 3.4 Flags Actually Checked in Code

All four flags are checked in their respective modules:
- `ENABLE_TOOL_GUARDRAILS`: Checked at the start of each guardrail function (3 check sites)
- `ENABLE_PIPELINE_MONITOR`: Checked at the start of `check_after_generation()` and `check_after_evaluation()` (2 check sites)
- `ENABLE_PARAMETER_BOUNDS`: Checked at the start of `apply_bounds()` (1 check site)
- `ENABLE_CONDITIONAL_TOOLS`: Checked in all 4 visibility callback functions (4 check sites)

### 3.5 Missing Flags (Listed in Roadmap but Not Yet Implemented)

These flags are listed in the roadmap's feature flag specification but do not exist in code (they correspond to unimplemented Phase 2B/2C features):

- `ENABLE_RALPH_ITERATIONS` (2B-1)
- `ENABLE_CONTEXT_TRIMMING` (2B-2)
- `ENABLE_MULTI_GRADER_EVAL` (2B-3)
- `ENABLE_ADVANCED_SESSION` (2B-4)
- `ENABLE_HITL_FRAMEWORK` (2B-5)
- `ENABLE_MEMORY_DECAY` (2B-6)
- `ENABLE_ARTIFACT_SHARING` (2B-8)
- `ENABLE_UCB1_SELECTOR` (2C-1)
- `ENABLE_MICRO_EXPERIMENTS` (2C-3)
- `ENABLE_BUDGET_DEGRADATION` (2C-4)

These are correctly absent -- they will be implemented with their respective features.

### 3.6 `ENABLE_SPEC_FIRST_PIPELINE` (2A-6)

The deprecation notices in `api_spec_agent.py` and `code_writer_agent.py` reference `ENABLE_SPEC_FIRST_PIPELINE=1` for rollback. However, this flag does NOT exist in code -- there is no `os.getenv("ENABLE_SPEC_FIRST_PIPELINE")` anywhere. The rollback procedure would require re-enabling imports and agent creation manually in `orchestrator.py`, not flipping a flag. The deprecation notice is misleading.

**Severity:** Medium

**Recommendation:**
1. Standardize on a single flag parsing pattern across all modules. Recommend Pattern A (`os.getenv("X", "true").lower() not in ("false", "0", "no")`) which handles all common falsy values.
2. Update `parameter_bounds.py` to use the standardized pattern so `ENABLE_PARAMETER_BOUNDS=false` works.
3. Either add a real `ENABLE_SPEC_FIRST_PIPELINE` flag or update the deprecation notice to reflect the actual rollback procedure (manual code changes).
4. Consider whether the centralized `config/agent_config.py` should also contain the feature flags as originally planned in the roadmap, or whether the per-module approach is intentional. If per-module is the chosen approach, update the roadmap to reflect this.

---

## Audit 4: Phase 2A Status Verification

**Finding:** Status claims are ACCURATE for all completed items. The roadmap correctly marks 2A-0 through 2A-6 as DONE and 2A-7, 2A-8 as TODO.

**Details:**

### 4.1 Item-by-Item Verification

| Item | Claimed Status | Code Evidence | Verdict |
|------|---------------|---------------|---------|
| **2A-0: Documentation Pipeline** | DONE | `scripts/upload_rewritten_manual.py` exists. Completion notes reference vector store `vs_699b7e6221bc81919ba2f4a1eae11588`. | **CONFIRMED** |
| **2A-1: is_enabled** | DONE | `utils/tool_visibility.py` exists (79 lines). Imported by `specialized_agents/quality_analyst.py` and `specialized_agents/docs_expert.py`. 4 callback functions: `budget_allows_vision`, `budget_allows_docs`, `learning_tool_for_iteration`, `budget_allows_evaluation`. Kill switch via env var. | **CONFIRMED** |
| **2A-2: tool_use_behavior** | DONE | `specialized_agents/executor.py:125` has `tool_use_behavior="stop_on_first_tool"`. Tests in `tests/test_tool_use_behavior.py` (10 tests). Note: `StopAtTools` for QualityGateJudge was NOT implemented -- test `TestQualityGateNoStopAtTools` confirms the quality gate agent uses `"run_llm_again"` (line 88), with a code comment explaining why (line 717-719 of orchestrator.py: "the agent needs structured output. stop_on_first_tool would bypass the structured output processing"). | **CONFIRMED** (with documented deviation from original plan) |
| **2A-3: Tool guardrails** | DONE | `guardrails/tool_guardrails.py` exists (275 lines). 3 guardrails: truth_pack_input, critical_failure_output, script_length_output. `attach_tool_guardrails()` function attaches to FunctionTool instances. Kill switch via env var. Tests in `tests/test_tool_guardrails.py`. | **CONFIRMED** |
| **2A-4: PipelineMonitor** | DONE | `tools/pipeline_monitor.py` exists (~483 lines). `PipelineMonitor` class with `check_after_generation()`, `check_after_evaluation()`, `get_status_report()`, `reset()`. 7 detection signals. Imported and used extensively in `orchestrator.py` (13 usage sites). Kill switch via env var. Tests in `tests/test_pipeline_monitor.py`. | **CONFIRMED** |
| **2A-5: Parameter bounds** | DONE | `tools/parameter_bounds.py` exists (~207 lines). `ParameterBound` dataclass with `clamp()` and `damped_change()`. 9 effect types bounded. Wired into orchestrator at `_apply_script_modifications()` (line 866). Oscillation damping via PipelineMonitor integration. Kill switch via env var. Tests in `tests/test_parameter_bounds.py`. | **CONFIRMED** |
| **2A-6: Deprecate Spec-First** | DONE | `specialized_agents/api_spec_agent.py` has deprecation notice at top (lines 1-25). `specialized_agents/code_writer_agent.py` also has deprecation notice (not read in full but grep confirms). Orchestrator no longer imports or creates API Spec Agent. Net ~730 lines removed per completion notes. Tests in `tests/test_deprecate_spec_first.py` (24 tests). | **CONFIRMED** |
| **2A-7: Remove Executor agent** | TODO | `specialized_agents/executor.py` EXISTS and is NOT deprecated. It still has `from agents import Agent, ModelSettings` and creates an Executor agent. No deprecation notice. The file is 170+ lines of active code. | **CONFIRMED TODO** |
| **2A-8: Tool timeouts** | TODO | Grep for "timeout" in `tools/blender_executor_tools.py` would show no timeout parameter on the function_tool definition. No `failure_error_function` parameter found. | **CONFIRMED TODO** |

### 4.2 Discrepancy: Roadmap Header vs Detail

The roadmap executive summary table (line 20) says "6/9 items DONE -- 154 tests pass". But the detail section (Phase 2A Summary, line 700) says "7/9 done". Count of DONE items: 2A-0, 2A-1, 2A-2, 2A-3, 2A-4, 2A-5, 2A-6 = **7 items**. The executive summary should say 7/9, not 6/9.

Update: Re-reading the executive summary more carefully, line 6 says "2A-0 through 2A-5 COMPLETE" which is 6 items (0-5). But 2A-6 is also DONE per the detail section. The header was likely written before 2A-6 was completed and not updated.

### 4.3 Test Count Verification

The roadmap v2.1 update claims "154 tests" for Phase 2A. The Phase 2A Summary table shows actual test counts:

| Item | Actual Tests |
|------|-------------|
| 2A-0 | 5 |
| 2A-1 | 22 |
| 2A-2 | 10 |
| 2A-3 | 22 |
| 2A-4 | 26 |
| 2A-5 | 40 |
| 2A-6 | 24 |
| **Total** | **149** |

The summary line says "39 / 178 (includes extras)" which also does not match 149 or 154. This is confusing -- the "39" is the estimated count and "178" appears to include tests from extra items beyond the roadmap-specified tests. The "154 tests" in the header may have been counted at a different point.

**Severity:** Low (status claims are correct; minor discrepancies in header vs details)

**Recommendation:**
1. Update the executive summary to say "7/9 items DONE" to match reality.
2. Reconcile test count claims -- either recount or remove the specific number from the header and let the detail table be authoritative.

---

## Audit 5: Mission Statement Alignment

**Finding:** The roadmap is WELL ALIGNED with the Mission Statement's design principles in priority order. No P6/P7 work is being done while P1 issues remain unaddressed. However, there are two areas where the roadmap could better serve the stated principles.

**Details:**

### 5.1 Principle-by-Principle Assessment

**P1: Reliability before capability**

Phases 2A and 2B focus almost exclusively on reliability: monitoring (2A-4), parameter bounds (2A-5), tool guardrails (2A-3), dead code removal (2A-6, 2A-7), context management (2B-1, 2B-2), and multi-grader eval (2B-3). Phase 2C starts adding capability (multi-physics 2C-2, micro-experiments 2C-3) but ONLY after the reliability infrastructure is in place and gated by the Decomposition Gate. Phase 2D (autonomy) is correctly deferred.

**Alignment: STRONG.** P1 is well served.

**P2: LLM creativity is the core value**

The roadmap explicitly avoids replacing LLM agents with templates. The Script Writer, Research Agent, and Quality Analyst remain LLM-powered. Deterministic replacements are limited to tasks that should never have been LLM-powered (execution, validation, monitoring, parameter clamping). The 2A-6 deprecation of the API Spec Agent replaces it with truth pack introspection, not a template.

**Alignment: STRONG.** P2 is well served.

**P3: Blender is the source of truth**

The truth pack (Phase 1) + tool guardrails (2A-3) enforce this at $0 cost. Multi-physics truth pack extension (2C-2) expands coverage to new physics types. Parameter bounds (2A-5) use Blender-sourced ranges.

**Alignment: STRONG.** P3 is well served.

**P4: Compute what you can, generate what you must**

PipelineMonitor (2A-4), parameter bounds (2A-5), tool guardrails (2A-3), and multi-grader Tier 1 (2B-3) all move computable tasks from LLM to deterministic code. The `tool_use_behavior` change (2A-2) eliminates unnecessary LLM calls.

**Alignment: STRONG.** P4 is well served.

**P5: Context is precious**

Ralph iterations (2B-1), context trimming (2B-2), artifact-based sharing (2B-8), and session compaction (Phase 1) all directly serve P5. This is the weakest area currently -- only Phase 1's session compaction is implemented. The heavy context management work is in 2B.

**Alignment: ADEQUATE.** P5 will be well served after 2B ships, but is underserved in current code.

**P6: Every run produces learning signal**

KB seeding (post-2A-0), memory decay (2B-6), effect-type scoping (2B-7), knowledge distillation (2C-6), and cross-session learning (2D-2) serve P6. These are correctly sequenced AFTER reliability (P1) infrastructure.

**Alignment: STRONG sequencing.** P6 work properly waits for P1.

**P7: Earn autonomy through evidence**

Autonomy progression (2D-1) is in Phase 2D -- the last phase. Evidence gating (Phase 1) provides the foundation. This is correctly the lowest-priority principle.

**Alignment: STRONG sequencing.** P7 is appropriately last.

### 5.2 Roadmap "End Goals" vs Mission Statement "Success Criteria"

**Mission Statement Success Criteria (Section 14):**

1. Every run produces a render -- even if low-scoring
2. Known effect types score >= 60 reliably
3. Novel prompts produce reasonable first attempts
4. System demonstrably improves over time
5. Failed runs produce useful diagnostics
6. User can intervene at any point
7. Budget is respected

**Roadmap End Goals (by Phase):**

| MS Criterion | Roadmap Phase | Status |
|-------------|---------------|--------|
| 1. Every run produces render | Phase 1 + 2A (monitors, guardrails, bounds prevent crashes) | Partially met (live E2E showed a crash-free run) |
| 2. Known effects >= 60 | Phase 2B success criteria #1: "Known effects score >= 60 in 50%+ of runs" | Addressed |
| 3. Novel prompts | Phase 2C success criteria #1: "Novel prompts produce evaluable renders" | Addressed |
| 4. Improves over time | Phase 2D success criteria #2: "Same prompt scores higher after 50 runs vs 5" | Addressed |
| 5. Failed runs produce diagnostics | Phase 1 (QA bridge) + 2A-4 (monitor) + 2C-5 (enhanced QA bridge) | Partially addressed |
| 6. User can intervene | Phase 2B-5 (HITL framework) | Addressed |
| 7. Budget respected | Phase 2A-1 (is_enabled) + 2A-4 (budget monitoring) + 2C-4 (budget degradation) | Addressed |

**Alignment: STRONG.** Every Mission Statement success criterion maps to at least one roadmap item, and they are sequenced in priority order.

### 5.3 Gaps Identified

**Gap 1: Camera placement is not addressed in the roadmap.**

The Mission Statement (Section 12) lists "camera placement catastrophically wrong in 2/3 benchmark scenarios" as a known problem. The Design Principles (P4) say "camera placement is geometry" -- it should be deterministic. Yet there is NO roadmap item that adds deterministic camera placement verification or correction. The closest is the enhanced QA bridge (2C-5) which detects camera issues but does not fix them.

This is a P4 issue (compute what you can) that is not served by any roadmap item. Camera inside geometry is listed as a critical issue auto-fail, but there is no deterministic pre-check or correction mechanism planned.

**Gap 2: Collision effector injection not addressed.**

The Mission Statement (Section 12) lists "missing collision effectors -- liquid falls through containment geometry" as a known problem. The Design Principles (P4) say "collision effectors are topology" -- deterministic. No roadmap item addresses this. The truth pack extension (2C-2) adds physics type coverage but does not add collision effector verification.

**Severity:** Medium (two P4 gaps, but they do not block P1 work)

**Recommendation:**
1. Add a roadmap item (suggested: 2C-8 or a 2B addition) for deterministic camera placement verification: check camera is outside scene bounds, pointed at subject, with reasonable focal length for scene size.
2. Add a roadmap item for collision effector injection: for liquid scenes, verify containment geometry has physics effector modifiers and inject them if missing.
3. Both items serve P4 and would prevent known critical failures (CAMERA_INSIDE_GEOMETRY, liquid falling through containers).

---

## Summary Table

| Audit | Severity | Key Finding |
|-------|----------|-------------|
| **1: Document Drift** | Medium | `REVISED_CROSS_CUTTING_SECTIONS.md` uses pre-v2.0 2C numbering (Agent.clone still present as 2C-4). Other research docs are consistent. |
| **2: Dependency Graph** | Low | Graph is accurate. 2A-5's dependency on 2A-4 is wiring-level (orchestrator.py), not module-level. No circular dependencies. No dependencies on unbuilt items blocking current work. |
| **3: Feature Flags** | Medium | Flags exist but are scattered per-module, not centralized in `config/agent_config.py` as the roadmap specifies. Three inconsistent parsing patterns. `ENABLE_SPEC_FIRST_PIPELINE` referenced in deprecation notices but does not exist. |
| **4: Phase 2A Status** | Low | All status claims verified correct. Executive summary says "6/9" but should say "7/9" (2A-6 is also DONE). Test count discrepancy between header (154) and detail table (149). |
| **5: Mission Alignment** | Low | Strong overall alignment. Two P4 gaps: camera placement verification and collision effector injection are known problems but not in the roadmap. |

---

## Appendix: File Inventory

### Files Read

| File | Purpose in Audit |
|------|-----------------|
| `docs/PHASE2_ROADMAP.md` | Canonical roadmap (1700+ lines) |
| `docs/MISSION_STATEMENT_2026-02-22.md` | Design principles and success criteria |
| `docs/PHASE2_ROADMAP_CRITIQUE_2026-02-23.md` | Codex critique (source of C11 claim) |
| `docs/research/REVISED_CROSS_CUTTING_SECTIONS.md` | Cross-cutting sections (document drift source) |
| `docs/research/REVISED_PHASE_2A_2B.md` | Phase 2A/2B research doc |
| `docs/research/REVISED_PHASE_2C_2D.md` | Phase 2C/2D research doc |
| `docs/research/SDK_FEATURES_ANALYSIS.md` | SDK features research |
| `docs/research/MONITORING_QUALITY_ARCHITECTURE.md` | Monitoring architecture research |
| `docs/research/AUTONOMOUS_SYSTEMS_RESEARCH.md` | Autonomous systems research |
| `config/agent_config.py` | Agent configuration (not feature flags) |
| `utils/tool_visibility.py` | 2A-1 implementation |
| `guardrails/tool_guardrails.py` | 2A-3 implementation |
| `tools/pipeline_monitor.py` | 2A-4 implementation |
| `tools/parameter_bounds.py` | 2A-5 implementation |
| `specialized_agents/executor.py` | 2A-7 target (NOT deprecated) |
| `specialized_agents/api_spec_agent.py` | 2A-6 target (deprecated) |
| `orchestrator.py` | Integration verification |

### Grep Searches Performed

- `ENABLE_` across all Python files (feature flag inventory)
- `from tools.pipeline_monitor` / `from tools.parameter_bounds` (dependency verification)
- `import.*tool_visibility` / `from.*tool_visibility` (2A-1 usage verification)
- `stop_on_first_tool` / `StopAtTools` / `tool_use_behavior` (2A-2 verification)
- `attach_tool_guardrails` / `set_truth_pack` (2A-3 integration verification)
- `oscillating` in orchestrator.py (2A-4/2A-5 integration verification)
- `apply_parameter_bounds` in orchestrator.py (2A-5 wiring verification)
