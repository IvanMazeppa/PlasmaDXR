# Critique Investigation Report: Phase 2 Roadmap Design Audit (C1-C12)

**Date:** 2026-02-23
**Investigator:** Claude Opus 4.6
**Scope:** Codebase-grounded verification of all 12 claims from `PHASE2_ROADMAP_CRITIQUE_2026-02-23.md`
**Method:** Read-only analysis of source code, documentation, and test files. No files modified.

---

## Executive Summary

Of the 12 claims made in the Codex critique, **5 are real issues** that warrant action, **4 are partially valid** (the concern is legitimate but the roadmap already partially addresses it), and **3 are not real issues** at this project's scale. The critique consistently assumes a multi-person team with concurrent workstreams; this is a solo developer (Ben) working sequentially with AI assistance. Several claims that would be critical in a team setting are low-severity here.

| Verdict | Count |
|---------|-------|
| REAL ISSUE | 5 (C2, C6, C10, C11, C12) |
| PARTIALLY VALID | 4 (C1, C4, C7, C8) |
| NOT A REAL ISSUE | 3 (C3, C5, C9) |

---

## C1: Phase scope too broad for reliability-first principle

**Codex Severity:** Critical
**Verdict:** PARTIALLY VALID
**Actual Severity:** Medium
**Evidence:**

The roadmap covers 4 sub-phases (2A through 2D) with 26 work items total. However, the critique's framing misrepresents the actual execution model:

1. **7 of 9 Phase 2A items are already DONE** (2A-0 through 2A-6). Each was shipped on its own branch with its own test suite: `0.34.0/phase-2-implementation` (2A-0, 2A-1, 2A-2), `0.34.2/phase-2a3-tool-guardrails` (2A-3), `0.34.3/phase-2a4-pipeline-monitor` (2A-4), `0.34.4/phase-2a5-parameter-bounds` (2A-5), `0.34.5/phase-2a6-deprecate-spec-first` (2A-6). Total: 154 tests passing.

2. **Items are NOT cross-coupled in implementation.** Code analysis shows:
   - 2A-1 (`utils/tool_visibility.py`): standalone, env-var kill switch, 0 imports from other 2A items
   - 2A-2: single-line `tool_use_behavior` additions to agent definitions
   - 2A-3 (`guardrails/tool_guardrails.py`): depends only on Phase 1 truth pack
   - 2A-4 (`tools/pipeline_monitor.py`): standalone class, 0 imports from 2A-1/2/3
   - 2A-5 (`tools/parameter_bounds.py`): reads PipelineMonitor's oscillation data but functions independently

3. **The roadmap already sequences 2B-1 (Ralph) in isolation** (Week 2, dedicated sprint, own branch). This is the single riskiest item and the roadmap explicitly addresses it.

4. **Phase 2D-3/4/5 are explicitly marked "Directional"** with the note "No line estimate or timeline assigned" and "This item will be designed after Phase 2C ships."

**Where the critique IS valid:** The roadmap presents 2B-2 through 2B-8 as a single "Week 3" block. Seven items in one week is aggressive even for a solo developer. If any of those items interact unexpectedly, debugging is harder because multiple things changed.

**Current State:** 2A is 7/9 done. Remaining: 2A-7 (trivial dead code removal), 2A-8 (trivial timeout addition). The "breadth" problem is theoretical for 2A, potentially real for 2B.

**Recommendation:** Add a `Class` column (Committed/Experimental/Directional) to 2B items as the critique suggests, but the urgency is medium, not critical. The solo-developer context means "too many concurrent paths" is less dangerous than in a team.

---

## C2: Gate criteria not statistically robust (5-run windows too small)

**Codex Severity:** Critical
**Verdict:** REAL ISSUE
**Actual Severity:** High
**Evidence:**

The 5-run gate checks exist ONLY in the roadmap documentation, specifically in the "E2E Validation Gates Between Phases" table (lines 252-261 of `PHASE2_ROADMAP.md`):

```
Phase 2A -> 2B: "Run 5 iterations on fire + liquid"
Phase 2B -> Decomposition: "Run 5 iterations with Ralph"
```

**There is NO gate-checking code implemented anywhere.** Search of `orchestrator.py` (3,901 lines), `session_manager.py` (528 lines), and all tools found zero references to phase gate logic, rolling windows, or promotion/demotion thresholds based on run counts.

The `SessionManager` tracks iteration history within a single session but does NOT aggregate across sessions. The only cross-session tracking is in the experiment tracker tools (KB queries), which are not structured as gate checks.

For 2D-1 (Autonomy Progression), the roadmap specifies level transitions at 10 runs. This is documented but not implemented. At 10 runs, any statistical claim about pass rates is fragile. The critique's suggestion of 20-30 runs for stability gates is sound.

The `PipelineMonitor` (2A-4, implemented) does per-iteration checks (oscillation, stuck loops, budget) but these are within-run safety nets, NOT cross-run statistical gates.

**Current State:** Gates are documentation-only. No code exists.

**Recommendation:** The critique's two-stage gate proposal (smoke gate at 5 runs, stability gate at 20 runs) is sound. When gate code is written (likely during 2D-1), use confidence bounds rather than raw averages. For now, document this explicitly as a known gap in the roadmap.

---

## C3: Dependency graph lacks critical-path pressure model

**Codex Severity:** Critical
**Verdict:** NOT A REAL ISSUE
**Actual Severity:** Low
**Evidence:**

The dependency graph in `PHASE2_ROADMAP.md` (lines 1594-1638) is detailed and accurate. Parallelism opportunities are listed (lines 1640-1643). The Implementation Sequence (lines 1748-1800) provides a week-by-week plan.

The critique's concern about "teams starting many threads in parallel" and "merge debt" is inapplicable:

1. **This is a solo developer** (Ben) working with AI assistance. There are no concurrent workstreams competing for reviewer bandwidth.
2. **Each item ships on its own git branch** (verified: 6 branches already pushed for 2A items).
3. **Items within a week are worked sequentially**, not in parallel. "Parallel" in this roadmap means "no hard dependency between these items" rather than "work on them simultaneously."

The implicit dependencies I found in code are accurately captured in the roadmap:
- `parameter_bounds.py` reads from `pipeline_monitor.get_status_report()` (2A-5 depends on 2A-4) -- documented
- `tool_guardrails.py` imports from `truth_pack.py` (2A-3 depends on Phase 1) -- documented
- `dynamic_instructions.py` queries KB via `experiment_tracker_tools` -- documented as 2B-6/2B-7 dependency

**Current State:** Dependency graph is accurate. No hidden couplings found beyond what's documented.

**Recommendation:** A critical-path overlay adds no value for a solo developer. Skip this. If the project ever adds contributors, revisit.

---

## C4: Definition-of-done inconsistent across items

**Codex Severity:** High
**Verdict:** PARTIALLY VALID
**Actual Severity:** Medium
**Evidence:**

Comparing test specifications across items:

**Well-specified (2A items, all have actual tests):**
- 2A-3 (Tool Guardrails): 5 specified tests, 22 actual tests in `tests/test_tool_guardrails.py`
- 2A-4 (PipelineMonitor): 5 specified tests, 26 actual tests in `tests/test_pipeline_monitor.py`
- 2A-5 (Parameter Bounds): 5 specified tests, 40 actual tests in `tests/test_parameter_bounds.py`

**Adequately specified (2B items):**
- 2B-1 (Ralph): 5 specific tests listed including "State roundtrip", "Fresh context", "A/B comparison"
- 2B-3 (Multi-Grader): 5 specific tests including "Black screen" and "Cost tracking"

**Under-specified (2D directional items):**
- 2D-3 (Prompt Versioning): No tests listed. Says "This item will be designed after Phase 2C ships."
- 2D-4 (Adaptive Replanning): No tests listed. Same caveat.
- 2D-5 (Full MASC): No tests listed. Same caveat.

**However:** The 2D directional items explicitly state they have no line estimates, timelines, or implementation plans. They are labeled "Directional" precisely because they are not yet scoped. The roadmap itself says: "Items 2D-3, 2D-4, and 2D-5 are directional goals -- their exact design depends on what Phase 2A-C reveals."

The roadmap DOES have a Testing Strategy section (lines 227-273) with a standardized template and three test categories. Every concrete item (2A through 2B, plus 2C-1 through 2C-7 and 2D-1/2D-2) has inline `**Tests:**` blocks.

**Where the critique IS valid:** The standard template proposed by the critique (Objective / Primary metric / Guardrail metric / Exit test / Rollback trigger / Owner) is better than the current format. Currently, "Rollback" and "Tests" exist but "Primary metric" and "Guardrail metric" are inconsistent. Some items have them (e.g., 2B-3: "cost tracking: 5 iterations with 2 critical failures -> assert 2 vision calls skipped"), others don't.

**Current State:** 10 test files exist in `tests/` with 154+ tests for Phase 2A. Concrete items have tests. Directional items don't, by design.

**Recommendation:** Adopt the 6-field template for 2B+ items. The gap between 2A's detailed completion notes and 2B's planned specifications is acceptable because 2B hasn't started yet.

---

## C5: Directional 2D items too execution-adjacent

**Codex Severity:** High
**Verdict:** NOT A REAL ISSUE
**Actual Severity:** Low
**Evidence:**

Reading 2D-3, 2D-4, and 2D-5 in the roadmap:

**2D-3 (Prompt Versioning):**
- Has "Research support" reference (Autonomy Research section 1.1)
- Has "Design direction" paragraph listing open questions
- Ends with: "This item will be designed after Phase 2C ships. No line estimate or timeline assigned."
- NO file paths listed. NO implementation table.

**2D-4 (Adaptive Replanning):**
- Has "Research support" reference (Autonomy Research section 2.3)
- Has "Design direction" paragraph with questions to resolve
- Ends with: "This item will be designed after Phase 2C ships. No line estimate or timeline assigned."
- NO file paths listed. NO implementation table.

**2D-5 (Full MASC):**
- Has "Research support" reference
- Has "Design direction" paragraph
- Same caveat about designing after 2C
- NO file paths listed. NO implementation table.

The critique claims these items "still have file-level implementation references." This is **incorrect for 2D-3/4/5 specifically.** They have zero file paths. The only 2D items with implementation tables are 2D-1 (Autonomy Progression, labeled "Concrete -- ready to implement") and 2D-2 (Cross-Session Learning, also labeled "Concrete").

The risk of "premature implementation" is negligible because:
1. There is no code to implement against -- no file paths, no line estimates
2. 2D items are gated behind 2C completion, which is itself gated behind the Decomposition Gate
3. This is a solo developer who works sequentially, not a team that might speculatively start coding

**Checked for any 2D code already started:** No files exist for `utils/autonomy_tracker.py`, no 2D-related imports in `orchestrator.py`, no 2D feature flags in the codebase.

**Current State:** Directional items are correctly minimal. No leakage.

**Recommendation:** No action needed. The roadmap already handles this correctly.

---

## C6: Budget model under-specifies worst-case envelopes

**Codex Severity:** High
**Verdict:** REAL ISSUE
**Actual Severity:** High
**Evidence:**

The Budget Impact Analysis (roadmap lines 1730-1745) provides only average-case estimates:
- "Current cost per 3-iteration run: ~$0.30-0.70"
- "Projected cost per 3-iteration run after Phase 2B: $0.15-0.35"

No P90, no worst-case, no per-run cap enforcement.

**What exists in code:**

1. **Monthly limit:** `BudgetTracker` in `utils/budget_tracker.py` tracks `monthly_limit` (default $20) and `total_spent`. Has `can_afford_evaluation()` check.

2. **Per-run budget:** `PipelineMonitor` (2A-4, implemented) has a `PER_RUN_BUDGET = 0.50` constant (line 104 of `tools/pipeline_monitor.py`). When `budget_spent > PER_RUN_BUDGET`, it generates a CRITICAL alert. However, this is advisory -- it produces an alert but does NOT force-stop the pipeline.

3. **Per-day cap:** Does NOT exist. No daily spending cap anywhere in the codebase.

4. **Budget guardrail:** `guardrails/quality_guardrails.py` has `check_budget_before_quality()` which checks remaining budget before allowing vision evaluation. This is a tool-level gate that blocks expensive operations.

5. **is_enabled gating (2A-1):** `utils/tool_visibility.py` hides expensive tools when budget is low. However, the threshold is coarse: `budget_allows_vision` checks `can_afford_evaluation()` and `budget_allows_docs` checks `remaining > 2.0`.

**What's missing:**
- No per-day spend cap
- No distribution analysis (P90/worst-case)
- PipelineMonitor's per-run budget alert is advisory, not blocking
- No auto-disable mechanism when per-run budget is exceeded (the alert says "disable expensive tools" but this is a recommendation in the alert text, not automatic)

The $20/month hard budget with $0.30-0.70/run means 29-67 runs/month. A worst-case run (5 iterations, all with vision evaluation, plus error recovery loops) could conceivably cost $1-2, consuming 5-10% of the monthly budget in a single run.

**Current State:** Basic monthly budget tracking exists. Per-run budget alerting exists but is advisory. No per-day cap. No worst-case analysis.

**Recommendation:** The critique's three-cost reporting (median/P90/worst-case) is the right approach. Add a hard per-run cap that force-stops the pipeline (not just alerts) when exceeded. Add a per-day cap as a safety net. This is a real risk at $20/month with no guardrails against cost spikes.

---

## C7: KB seeding trust bootstrapping is risky

**Codex Severity:** High
**Verdict:** PARTIALLY VALID
**Actual Severity:** Medium-Low
**Evidence:**

The KB Seeding Strategy (roadmap lines 714-781) already addresses the core concern:

1. **Seeded entries start at "emerging" trust level** -- explicitly stated. They are NOT at "trusted" level.

2. **"Emerging" entries are NOT injected into Script Writer prompts.** The `dynamic_instructions.py` code (line 762) calls `query_validated_learnings(effect_type, "physics", 0.8)` -- requiring 80% success rate minimum. Seeded entries with no production usage have 0% success rate and will NOT be injected.

3. **Who receives seeded entries:** Only the Research Agent and Technique Selector see them (as exploration hints via KB queries). The Script Writer, which generates actual code, does NOT see them until they earn trust.

4. **The KB was wiped** (Phase 1, confirmed in MEMORY.md: "KB wiped -- 18 items backed up + wiped. Clean slate."). So there is no contamination from stale data.

5. **Evidence gating already exists** (Phase 1): `min_confidence 50`, `min_success_rate 0.8`, decay. These thresholds filter seeded entries out of prescriptive use.

**Where the critique IS valid:** The critique's suggestion of separating `reference_seed` (never prescriptive) from `execution_validated` (eligible for injection) is a cleaner formalization of what the current code already does implicitly through trust levels and success rate thresholds. Making this explicit would be clearer.

Also, the critique correctly identifies that seeded patterns could dominate prompt context for the Research Agent, reducing exploration diversity. If 6 effect types each seed 5 entries, that's 30 KB entries that the Research Agent will see before any production evidence exists. This could steer initial technique selection.

**Current State:** The architecture handles this well through evidence gating. The risk of "confidently wrong" guidance is low because seeded entries cannot reach the Script Writer without production validation.

**Recommendation:** Add a `source` field distinction (as planned) and consider limiting how many seeded entries the Research Agent sees per query. The critique's lane separation is good practice but not urgent.

---

## C8: Multi-grader plan can become control-loop instability source

**Codex Severity:** High
**Verdict:** PARTIALLY VALID
**Actual Severity:** Medium
**Evidence:**

**2B-3 (Multi-Grader) is NOT implemented yet.** No file `tools/deterministic_quality_checks.py` exists (confirmed by grep -- it's referenced in 5 docs but no Python file exists). The current evaluation pipeline uses a single grader: `tools/asset_evaluator_tools.py` with LLM vision only.

The PLANNED 3-tier design (roadmap lines 892-924):
- Tier 1: Deterministic (histogram, lights, camera) -- $0, hard gates
- Tier 2: ML metrics (CLIP, LPIPS, TOPIQ) -- $0 local
- Tier 3: LLM vision -- $0.01-0.05

**Design analysis of the proposed approach:**

1. **Short-circuit logic is sound:** Critical Tier 1 failure (BLACK_SCREEN) immediately sets score=0 and skips Tiers 2+3. This prevents conflicting signals -- if the render is blank, no ML or vision analysis runs.

2. **Weighting is fixed:** "ML (60%) + Vision (40%)" is a static formula. No dynamic weight adjustment. This eliminates the oscillation risk that the critique warns about.

3. **The critique's concern about "conflicting optimization pressure" is theoretically valid** but mitigated by the design:
   - Tier 1 is pass/fail (not scored), so no conflict with Tiers 2/3
   - Tiers 2 and 3 produce a single blended score, not competing recommendations
   - The Modification Strategist agent receives one combined score and one primary issue, not separate signals from each grader

4. **What IS missing:** No evaluator governance rules, no conflict resolution policy, no drift detection between tier outputs. If Tier 2 says "excellent" (90) but Tier 3 says "poor" (30), the blended score would be 66 (passing!). The roadmap does not specify how to handle this divergence.

**Current State:** Single-grader only (LLM vision). Multi-grader is planned for 2B-3 with fixed weights. No evaluator governance.

**Recommendation:** The critique's suggestion of frozen weighting per phase window is already satisfied (the weights are static). Add a divergence alert: if |Tier2 - Tier3| > 30, flag for review. Add a "no action" outcome when confidence is low. These are ~20 lines of code when 2B-3 is implemented.

---

## C9: Feature-flag defaults create high blast radius

**Codex Severity:** Medium-High
**Verdict:** NOT A REAL ISSUE
**Actual Severity:** Low
**Evidence:**

The critique claims feature flags defaulting to `True` creates "combinatorial behavior changes from one deployment." Here is what actually exists:

**Feature flags in code (NOT in a centralized `config/agent_config.py`):**

The `config/agent_config.py` file is the AGENT CONFIG MANAGER -- it manages model presets (which LLM model to use, reasoning effort, temperature). It does NOT contain feature flags. Feature flags are implemented as environment variables in individual modules:

| Flag | File | Default | Kill Switch |
|------|------|---------|-------------|
| `ENABLE_CONDITIONAL_TOOLS` | `utils/tool_visibility.py:28` | `"true"` (env var) | `ENABLE_CONDITIONAL_TOOLS=false` |
| `ENABLE_TOOL_GUARDRAILS` | `guardrails/tool_guardrails.py:49` | `"true"` (env var) | `ENABLE_TOOL_GUARDRAILS=false` |
| `ENABLE_PARAMETER_BOUNDS` | `tools/parameter_bounds.py:19` | `"1"` (env var) | `ENABLE_PARAMETER_BOUNDS=0` |
| `ENABLE_PIPELINE_MONITOR` | (referenced in roadmap, env var) | Default on | env var disable |

**Total active flags: 3-4.** Not "multiple default-on flags increasing combinatorial behavior." Each flag was added one at a time on its own branch. There was never a deployment where multiple new flags activated simultaneously.

The roadmap's feature flag table (lines 283-307) lists 14 planned flags, but only 3-4 currently exist. The rest are for 2B and 2C items that haven't been implemented yet.

**Critical context:** This is a solo developer running the system locally, not a production deployment with rollout stages. "Staging profile" and "phased defaults" add process overhead with no benefit for a local development workflow. Setting an env var to `0` and restarting is the entire rollback procedure, and it works.

**Current State:** 3-4 flags exist, each with env var kill switch, each shipped individually on its own branch with its own tests.

**Recommendation:** No action needed at current scale. If the project ever deploys as a service, revisit with staged rollout. For now, the env-var kill switches are sufficient.

---

## C10: Decomposition gate effort underestimated

**Codex Severity:** Medium
**Verdict:** REAL ISSUE
**Actual Severity:** Medium-High
**Evidence:**

**Orchestrator measurements:**
- `orchestrator.py`: **3,901 lines** (the roadmap says ~4,544 but this was before 2A-6 removed ~730 lines)
- **19 methods** in `BlenderVFXOrchestrator` class
- **119 references to `self._`** instance variables, indicating heavy shared state
- **10 Pydantic output models** defined at module level (used across phases)
- **~15 top-level imports** from other project modules (truth pack, guardrails, tools, hooks, etc.)

**Hidden complexity factors:**

1. **Shared state is pervasive.** The `BlenderVFXOrchestrator.__init__()` creates 10+ instance variables (`_research_agent`, `_script_agent_standalone`, `_quality_agent_standalone`, `_learning_agent_standalone`, `_technique_coordinator`, `_modification_coordinator`, `_quality_gate_coordinator`, `_budget_tracker`, `_persistence`, `_pipeline_monitor`, `_config`, `_flow_tracker`). Each phase module would need access to these shared agents and trackers.

2. **The `_apply_script_modifications()` method** (lines 836-905) is called from multiple phases (modification, error recovery, technique switch, fallback). Extracting it requires deciding which module owns it or creating a shared utility.

3. **The `_run_agent()` wrapper** (lines 917-939) is used by every phase. It handles RunConfig injection, hook setup, and session management. This is cross-cutting infrastructure.

4. **The `_run_parallel_preflight()` method** spans research and docs phases, handling fan-out/fan-in concurrency. This doesn't fit cleanly into a single phase module.

5. **The roadmap estimates "~200 lines" for decomposition** and calls it "Low risk." This is almost certainly wrong. Moving 3,401 lines (3,901 minus ~500 for the coordinator shell) into 7 phase modules requires:
   - Defining interfaces for each module (which agents they receive, what state they return)
   - Handling the shared agent instances (dependency injection or module-level globals)
   - Ensuring error handling chains work across module boundaries
   - Updating all 10 test files that may import from orchestrator.py

**The roadmap's own completion notes from 2A-6** are instructive: estimated ~128 lines, actual result was ~730 lines net removed. The 1.5x multiplier they added based on Phase 1 experience may not be sufficient for a decomposition task.

**Current State:** Orchestrator is a 3,901-line monolith with 19 methods, 10+ shared instance variables, and 119 self._ references. The proposed module structure (7 phase files) is reasonable but the effort estimate is too low.

**Recommendation:** Increase the estimate to 400-600 lines of restructuring work (2-3x the current estimate). Adopt the critique's staged migration approach: extract one module at a time, starting with the simplest (execution.py), and validate E2E after each extraction. Do NOT attempt a single big-bang decomposition.

---

## C11: Internal doc drift already visible

**Codex Severity:** Medium
**Verdict:** REAL ISSUE
**Actual Severity:** Medium
**Evidence:**

Comparing `REVISED_CROSS_CUTTING_SECTIONS.md` with `PHASE2_ROADMAP.md`:

**Confirmed drift in the Pipeline Stage table:**

| Difference | Cross-Cutting Doc | Roadmap |
|-----------|-------------------|---------|
| Script Generation - 2C items | Lists `Agent.clone (2C-4), streaming (2C-5)` | Lists `Streaming (2C-7, nice-to-have)`. Agent.clone was CUT. |
| Evaluation - 2C items | Lists `Budget degradation (2C-6), enhanced QA bridge (2C-7)` | Lists `Budget degradation (2C-4), enhanced QA bridge (2C-5)`. Numbers shifted. |
| Learning - 2C items | Lists `Knowledge distillation (2C-8)` | Lists `Knowledge distillation (2C-6)`. Number shifted. |
| Monitoring - 2D items | Lists `MASC (2D-5)` | Same -- this is consistent |

The root cause is clear: the cross-cutting doc was written when 2C had 8 items including Agent.clone (2C-4) and different numbering. The roadmap was later revised to cut Agent.clone and renumber 2C items. The cross-cutting doc was not updated.

**Additional drift found:**
- Cross-cutting doc (line 504): references `Agent.clone (2C-4)` -- this was explicitly removed from the roadmap (lines 1429-1431): "Moved to Future Ideas."
- Cross-cutting doc (line 428): references `Knowledge Distillation (2C-8)` -- roadmap has it as `2C-6`
- Cross-cutting doc Rollback Strategy branch names use `0.33.2/` prefix (line 151), while actual branches use `0.34.X/` (confirmed from git log and roadmap line 316)

**Other research docs checked:**
- `REVISED_PHASE_2A_2B.md` and `REVISED_PHASE_2C_2D.md` are referenced in the critique but were not investigated for drift (they are research inputs, not operational docs)

**Current State:** The cross-cutting doc has stale 2C numbering, references a removed feature (Agent.clone), and uses wrong branch name prefixes. The main `PHASE2_ROADMAP.md` is the corrected version.

**Recommendation:** Either update `REVISED_CROSS_CUTTING_SECTIONS.md` to match current numbering or add a prominent header declaring it superseded by `PHASE2_ROADMAP.md`. The roadmap already states it is v2.1 with updates, but the research docs do not carry a "superseded" notice. The critique's suggestion of a consistency lint script is overkill for a solo project; a manual header notice suffices.

---

## C12: No compact risk register with owner-level accountability

**Codex Severity:** Medium
**Verdict:** REAL ISSUE
**Actual Severity:** Medium
**Evidence:**

Searched `PHASE2_ROADMAP.md` for "risk" -- found 33 occurrences. Risk information is distributed across:

1. **Phase-level risk ratings:** Each phase header has a risk line (e.g., "Risk: Low -- all changes are additive", "Risk: Medium -- changes iteration loop structure").

2. **Per-item risk notes:** Individual items have `**Risk:** Low/Medium/High` at the bottom.

3. **Rollback Strategy section** (lines 276-332): Covers feature flags, git branches, rollback decision criteria.

4. **Conflict Resolutions section** (lines 1647-1691): Addresses 6 design conflicts with explicit resolutions.

5. **Design Principles** (lines 67-79): P1 says "Reliability before capability."

**What does NOT exist:**
- A single consolidated risk register table
- Leading indicators for each risk
- Trigger thresholds
- Mitigation actions specific to each risk
- Owner assignments (moot for solo developer, but useful for AI assistants)
- Review cadence

The risks ARE addressed, but they're spread across ~1,800 lines of document. There is no "one place that says: what can break us this month."

The closest thing to a consolidated view is the Phase 2A/2B/2C/2D risk ratings, but these are per-phase, not per-risk.

**Missing risks not mentioned anywhere in the roadmap:**
- "What if the OpenAI Agents SDK v0.9.3 breaks backward compatibility in a minor update?"
- "What if Blender 5.0 manual content degrades rewritten doc quality?"
- "What if the $20/month budget proves insufficient for meaningful E2E testing?"

**Current State:** Risks are distributed, not consolidated. No dedicated risk register.

**Recommendation:** Create a one-page risk register as the critique suggests. For a solo developer project, "Owner" can be simplified to "Mitigation agent" (which AI tool or process handles it). A 10-row table with Risk/Indicator/Threshold/Action would take 30 minutes to write and provide significant value for planning.

---

## Summary Table

| Claim | Codex Severity | Verdict | Actual Severity | Action Needed |
|-------|---------------|---------|-----------------|---------------|
| C1: Scope too broad | Critical | PARTIALLY VALID | Medium | Add work class labels for 2B items |
| C2: 5-run gates too small | Critical | REAL ISSUE | High | Design proper gate statistics when implementing 2D-1 |
| C3: No critical-path model | Critical | NOT A REAL ISSUE | Low | Skip -- solo developer |
| C4: Inconsistent DoD | High | PARTIALLY VALID | Medium | Adopt 6-field template for 2B+ items |
| C5: Directional items leaky | High | NOT A REAL ISSUE | Low | No action -- already well-handled |
| C6: Budget worst-case missing | High | REAL ISSUE | High | Add hard per-run cap and per-day cap |
| C7: KB seeding trust risk | High | PARTIALLY VALID | Medium-Low | Add source field distinction |
| C8: Multi-grader instability | High | PARTIALLY VALID | Medium | Add divergence alert when implementing 2B-3 |
| C9: Flag blast radius | Medium-High | NOT A REAL ISSUE | Low | No action at current scale |
| C10: Decomposition underestimated | Medium | REAL ISSUE | Medium-High | Increase estimate to 400-600 lines, staged migration |
| C11: Doc drift visible | Medium | REAL ISSUE | Medium | Update or mark cross-cutting doc as superseded |
| C12: No risk register | Medium | REAL ISSUE | Medium | Create one-page risk register |

---

## Recommended Priority for Addressing Findings

**Before 2B starts (immediate):**
1. C11: Mark `REVISED_CROSS_CUTTING_SECTIONS.md` as superseded (5 minutes)
2. C6: Add hard per-run cost cap to PipelineMonitor (convert advisory alert to force-stop) (~30 lines)

**When implementing 2B items:**
3. C10: Use staged migration for decomposition gate, increase estimate
4. C8: Add divergence alert to multi-grader design (~20 lines)
5. C4: Adopt standard DoD template for 2B items

**When implementing 2D items:**
6. C2: Design proper gate statistics with confidence bounds
7. C12: Create risk register

**No action needed:**
- C3, C5, C9: Not applicable at project scale
