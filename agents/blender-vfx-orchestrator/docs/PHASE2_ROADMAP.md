# Phase 2+ Roadmap: From Reliability to Full Autonomy

**Version:** 2.1
**Date:** 2026-02-23
**Authors:** Integration Architect (Claude Opus 4.6), synthesizing research from SDK Specialist, Codebase Analyst, Autonomy Researcher, Monitoring Architect, and Critique Reviewer
**Status:** Phase 2A COMPLETE — 2A-0 through 2A-8 ALL DONE
**Revision note:** v2.0 addresses all findings from the Phase 2 Roadmap Critique: adds documentation pipeline (2A-0), testing strategy, rollback strategy, orchestrator decomposition gate, KB seeding, Ralph isolation, 1.5x line estimates, Agent.clone() removal, Phase 2D marked directional.
**v2.1 update (2026-02-23):** Progress tracking added. 2A-0 through 2A-5 complete (154 tests, 6 branches merged). Actual line counts noted where they differ from estimates.

---

## Executive Summary

Phase 1 (Reliability) shipped 1,574 lines across 11 files: truth pack, QA feedback bridge, KB wipe, session compaction, escape velocity testing, and doc search fix. The pipeline runs end-to-end without crashing. Scripts still have quality issues, but the infrastructure is solid.

Phase 2+ takes the system from "produces evaluable renders" to "reliably produces quality renders that improve over time." It is organized into four sub-phases plus infrastructure gates:

| Phase | Focus | Estimated Effort | Status |
|-------|-------|-----------------|--------|
| **2A: Quick Wins** | Documentation pipeline, cost savings, safety nets, monitoring | ~1,431 lines | **9/9 COMPLETE — 192 tests pass** |
| **2B: Core Architecture** | Stateless iterations, context management, multi-grader eval, HITL | ~2,027 lines | **3/8 COMPLETE (2B-1, 2B-3, 2B-5)** — 269 tests pass |
| **2C: Advanced Capabilities** | Multi-physics, experimentation, technique diversity | ~1,595 lines (core) | Weeks 6-9 |
| **2D: Full Autonomy (Directional)** | Autonomy tracking, cross-session learning | ~405 lines (concrete) + TBD | Week 10+ |

Plus infrastructure: KB Seeding (~130 lines after 2A-0), Orchestrator Decomposition Gate (~200 lines refactoring in Week 5).

**Total estimated concrete code: ~5,800 lines.** All estimates include a 1.5x multiplier based on Phase 1 experience (estimated 1,200 lines, shipped 1,574). Directional items (2D-3, 2D-4, 2D-5) will be scoped after Phase 2C ships.

**Critical addition (v2.0):** Phase 2A now starts with **2A-0: Documentation Pipeline** — processing the Blender manual into LLM-optimized format. This is the highest-ROI item in the roadmap: without it, the self-learning pipeline has nothing to learn from and the agent repeats the same 3-4 techniques for every prompt.

**Expected outcomes by end of Phase 2B:**
- Every run produces a render (reliability target maintained)
- Known effects (fire, liquid, smoke) score >= 60 in 50%+ of runs
- Cost per 3-iteration run: $0.15-0.35 (down from $0.30-0.70)
- No parameter oscillation across iterations
- Context stays within budget per agent per iteration

---

## Foundation: What Phase 1 Delivered

| Deliverable | File | Impact |
|------------|------|--------|
| Truth Pack (bl_rna introspection) | `tools/truth_pack.py` (743 LOC) | Eliminates #1 failure mode (hallucinated attributes) at $0 |
| Truth Pack Validator | `tools/truth_pack_validator.py` (187 LOC) | Validate-fix-write cycle for scripts |
| QA Feedback Bridge | `tools/qa_diagnosis_bridge.py` (254 LOC) | Pairs visual critique with script code |
| KB Wipe Script | `scripts/wipe_kb.py` (99 LOC) | Clean slate, 18 items backed up |
| Escape Velocity Tests | `tests/test_escape_velocity.py` (144 LOC) | 16 tests, L4→PAUSED verified |
| Session Compaction | `orchestrator.py` modification | Auto-triggers at ~25K tokens |
| Evidence Gating | Multiple files | min_confidence 50, min_success_rate 0.8, decay |
| Doc Search Fix (quotas + routing + technique discovery) | `tools/semantic_docs_tools.py` | Unblocks technique diversity — research agent can now discover manual content |

**Live E2E test result** (trace `63eb4ef1`): Pipeline completed without crash. 21KB fire mantaflow script generated. Execution failed on `use_nodes` deprecation and `collection.get(int)` bugs — both now fixed by truth pack patterns #13 and #15.

**Doc Search Fix details:** Three bugs were discovered in `semantic_docs_tools.py` that explain why the research agent could not discover new techniques — the root cause of "technique monotony" (Mission Statement §12):

| Bug | Impact | Fix |
|-----|--------|-----|
| API results filled shared quota before manual queries ran | Manual/tutorial content completely blocked | Separate quotas: `manual_quota = max(max_results // 2, 3)`, `api_quota = max(max_results // 2, 3)` |
| Manual queries not explicitly routed to manual vector store | Queries hit API store by default, missing conceptual content | Explicit intent routing: `intent="manual"` for conceptual queries, `intent="api"` for attribute lookups |
| No technique discovery queries | Agent never asked "what alternatives exist?" | Added technique discovery queries: "how to create X effect", "alternative methods techniques for X", "physics simulation types Blender" |

Results are now interleaved: manual first (conceptual understanding, techniques), then API (attribute names, types). This fix is the prerequisite that makes the entire Research → Learning pipeline functional.

---

## Design Principles (Ranked — Higher Overrides Lower)

These principles resolve conflicts between competing approaches. Every work item references which principles it serves.

| Rank | Principle | Shorthand |
|------|-----------|-----------|
| **P1** | Reliability before capability | Don't add features that destabilize |
| **P2** | LLM creativity is the core value | Don't replace creative agents with templates |
| **P3** | Blender is the source of truth for its own API | Runtime introspection > training data |
| **P4** | Compute what you can, generate what you must | Deterministic where possible, LLM where valuable |
| **P5** | Context is precious — every token must earn its place | Dense output, artifact-based sharing |
| **P6** | Every run produces learning signal | Even failures have value if captured |
| **P7** | Earn autonomy through evidence | Trust levels based on outcomes, not calendar time |

---

## Architecture: The Research → Learning Pipeline

This diagram shows the end-to-end flow that Phase 2 builds and optimizes. Every box exists in the codebase or this roadmap.

```
                     DOCUMENTATION SOURCES
                     =====================

BLENDER MANUAL                              BLENDER PYTHON API
(LLM-optimized rewrite)                    (Runtime introspection)
  │                                           │
  │ "HOW / WHY / WHEN"                        │ "WHAT EXISTS"
  │ Techniques, workflows,                    │ Attribute names, types,
  │ parameter relationships,                  │ valid ranges, defaults
  │ conceptual understanding                  │
  │                                           │
  │  ┌─ 2A-0: Manual Rewrite ──────┐         │  ┌─ Phase 1: Truth Pack ──────┐
  │  │ gpt-5-mini offline rewrite  │         │  │ bl_rna.properties          │
  │  │ 64% compression, $0.19      │         │  │ introspection, $0.00       │
  │  └─────────────────────────────┘         │  └────────────────────────────┘
  │                                           │
  ▼                                           ▼
┌─────────────────────────────────────────────────┐
│                RESEARCH AGENT                    │
│  Discovers techniques, understands approaches,   │
│  knows what's possible for the current prompt    │
│                                                  │
│  Queries: manual (how/why) + API (what exists)   │
│  Output: Dense, structured research summary      │
│  + full_report_path artifact                     │
│                                                  │
│  Fixed in Phase 1: doc search quotas + routing   │
│  Seeded in 2A-0: KB bootstrap for 6 effect types │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│            TECHNIQUE SELECTION                     │
│  UCB1 (2C-1) as default algorithm:                │
│    - Always try untried techniques first           │
│    - Balance exploitation (best avg score)          │
│    vs exploration (underexplored techniques)        │
│  LLM override when research identifies specific    │
│  reason to deviate from UCB1 recommendation        │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│            SCRIPT GENERATION                       │
│  Script Writer creates complete Blender Python     │
│  Context: research summary + truth pack +          │
│           dynamic instructions (trusted patterns)  │
│                                                    │
│  Truth Pack Validation (2A-3 tool guardrail):      │
│  Every attribute access checked against bl_rna     │
│  before script reaches Blender. $0 cost.           │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│       EXECUTION → EVALUATION                       │
│                                                    │
│  Execution: Blender headless subprocess            │
│  Evaluation: 3-tier pipeline (2B-3)                │
│    Tier 1: Deterministic (histogram, lights) — $0  │
│    Tier 2: ML metrics (CLIP, LPIPS, TOPIQ) — $0   │
│    Tier 3: LLM vision (if Tiers 1-2 pass) — $0.05 │
│                                                    │
│  PipelineMonitor (2A-4) checks after each tier:    │
│    oscillation, stuck loops, cascades, budget       │
│                                                    │
│  QA Diagnosis Bridge pairs visual critique          │
│  with script code for root cause identification     │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│            LEARNING AGENT                          │
│  Records what worked and what failed:              │
│    - Technique + effect type + quality score       │
│    - Parameter values that produced this score     │
│    - Root cause of failures (from QA bridge)       │
│                                                    │
│  Knowledge Distillation (2C-6):                    │
│    If score >= 60, extract code patterns            │
│    (lighting setup, materials, camera, physics)     │
│    as named, reusable patterns with metadata        │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│            KNOWLEDGE BASE                          │
│  Evidence-gated entries (Phase 1):                 │
│    Untrusted → Emerging → Trusted → Deprecated     │
│                                                    │
│  Effect-type scoped (2B-7):                        │
│    Patterns tracked per effect type, not globally   │
│                                                    │
│  Ebbinghaus memory decay (2B-6):                   │
│    Unreinforced entries decay exponentially          │
│    retention < 0.3 → excluded from queries          │
│    retention < 0.1 → archived                       │
│                                                    │
│  KB seeding (post-2A-0):                           │
│    Bootstrap with 6 effect types from rewritten     │
│    manual. Start at "emerging" trust level.         │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│         DYNAMIC INSTRUCTIONS                       │
│  Injects trusted patterns into agent prompts:      │
│    - Script Writer: known-good code patterns        │
│    - Research Agent: validated technique context     │
│    - Modification Strategist: parameter bounds      │
│                                                    │
│  Filtered by:                                      │
│    - Effect type (2B-7)                             │
│    - Retention score (2B-6)                         │
│    - Trust level (Phase 1 evidence gating)          │
│    - Prompt version (2D-3, directional)             │
└─────────────────────┬──────────────────────────────┘
                      │
                      │  feeds back to
                      │
                      └──────────────► RESEARCH AGENT (top of diagram)
```

### Roadmap Items by Pipeline Stage

| Pipeline Stage | Phase 1 (Done) | Phase 2A | Phase 2B | Phase 2C | Phase 2D |
|---------------|----------------|----------|----------|----------|----------|
| Documentation | Truth pack, doc search fix | Manual rewrite (2A-0), KB seeding | | | |
| Research | | KB seeding | | | |
| Technique Selection | | | | UCB1 (2C-1) | Adaptive replanning (2D-4, directional) |
| Script Generation | Truth pack validation | Tool guardrails (2A-3) | Context trimming (2B-2), artifact sharing (2B-8) | Streaming (2C-7, nice-to-have) | Prompt versioning (2D-3, directional) |
| Execution | | Deprecate Spec-First (2A-6), remove Executor (2A-7), timeouts (2A-8) | | | |
| Evaluation | QA bridge | | Multi-grader (2B-3) | Budget degradation (2C-4), enhanced QA bridge (2C-5) | |
| Learning | Evidence gating, KB wipe | | Memory decay (2B-6), effect-type gating (2B-7) | Knowledge distillation (2C-6) | Cross-session transfer (2D-2) |
| Monitoring | Escape velocity | PipelineMonitor (2A-4), parameter bounds (2A-5) | HITL (2B-5) | | Autonomy tracking (2D-1), MASC (2D-5, directional) |
| Context Management | Session compaction | is_enabled (2A-1), stop_on_first_tool (2A-2) | Ralph iterations (2B-1), AdvancedSQLiteSession (2B-4) | | |

---

## Testing Strategy

**Applies to:** All phases (2A through 2D)

### Philosophy

Every work item ships with tests. No exceptions. Phase 1 established the standard: 16 escape velocity tests across 6 test classes in 144 lines. That's the baseline — every Phase 2 work item meets or exceeds it.

Three test categories, in order of priority:

| Category | What It Covers | When Required |
|----------|---------------|---------------|
| **Unit tests** | New classes, functions, and data structures in isolation | Every work item that adds new code |
| **Integration tests** | Pipeline behavior when new code interacts with existing components | Every work item that modifies `orchestrator.py` or agent behavior |
| **Regression tests** | Existing behavior preserved after code removal or replacement | Every work item that deprecates, removes, or replaces existing code |

### Test Location and Naming

All tests live in `tests/`. Naming convention:

```
tests/test_<module_name>.py              # Unit tests for a specific module
tests/test_<module_name>_integration.py  # Integration tests
```

### E2E Validation Gates Between Phases

| Gate | Trigger | Validation |
|------|---------|------------|
| **Phase 1 → 2A** | Phase 1 merged (DONE) | 3 effect types produce evaluable renders. Verified via trace `63eb4ef1`. |
| **Phase 2A → 2B** | All 2A items pass unit + integration tests | Run 5 iterations on fire + liquid. Verify: no parameter oscillation, budget < $0.30/run, zero hallucinated attributes reach Blender. |
| **Phase 2B → Decomposition Gate** | 2B-1 (Ralph) passes E2E | Run 5 iterations with Ralph. Compare iteration 3-4 quality against old baseline. Quality must not regress. |
| **Decomposition Gate → 2C** | Orchestrator decomposed into phase modules | All existing tests pass against decomposed code. No new functionality — pure refactor. |
| **Phase 2C → 2D** | 2C items pass tests + E2E | Novel prompt (rigid body) produces evaluable render. UCB1 selects different techniques. Micro-experiment completes in <30 seconds. |

### Test Execution

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific phase
python -m pytest tests/test_pipeline_monitor.py tests/test_parameter_bounds.py tests/test_tool_guardrails.py -v  # Phase 2A

# Run E2E validation
python test_e2e_orchestrator.py --effects fire,liquid,smoke --iterations 5
```

---

## Rollback Strategy

**Applies to:** All phases (2A through 2D)

### Feature Flags

Every major behavioral change ships behind a feature flag. Flags live in `config/agent_config.py` with a consistent naming convention: `ENABLE_<FEATURE_NAME>`. All flags default to `True` (new behavior active). Setting a flag to `False` reverts to previous behavior.

```python
# config/agent_config.py

# Phase 2A flags
ENABLE_TOOL_GUARDRAILS = True       # 2A-3: Set False → revert to pipeline-level validation
ENABLE_PIPELINE_MONITOR = True      # 2A-4: Set False → skip monitoring checkpoints
ENABLE_PARAMETER_BOUNDS = True      # 2A-5: Set False → no parameter clamping
ENABLE_CONDITIONAL_TOOLS = True     # 2A-1: Set False → all tools always visible

# Phase 2B flags
ENABLE_RALPH_ITERATIONS = True      # 2B-1: Set False → revert to accumulated context
ENABLE_CONTEXT_TRIMMING = True      # 2B-2: Set False → no per-agent input filtering
ENABLE_MULTI_GRADER_EVAL = True     # 2B-3: Set False → skip Tier 1 deterministic checks
ENABLE_ADVANCED_SESSION = True      # 2B-4: Set False → use SQLiteSession
ENABLE_HITL_FRAMEWORK = True        # 2B-5: Set False → no native approval gates
ENABLE_MEMORY_DECAY = True          # 2B-6: Set False → no Ebbinghaus decay on KB entries
ENABLE_ARTIFACT_SHARING = True      # 2B-8: Set False → inline results (old behavior)

# Phase 2C flags
ENABLE_UCB1_SELECTOR = True         # 2C-1: Set False → LLM-only technique selection
ENABLE_MICRO_EXPERIMENTS = True     # 2C-3: Set False → no sandbox experiments
ENABLE_BUDGET_DEGRADATION = True    # 2C-4: Set False → binary budget check
```

**Implementation cost:** ~3 lines per feature (flag check + conditional). Total: ~45 lines across all phases.

### Git Branch Strategy

| Branch | Base | Content | Status |
|--------|------|---------|--------|
| `0.34.0/phase-2-implementation` | main | 2A-0, 2A-1, 2A-2 | Pushed |
| `0.34.2/phase-2a3-tool-guardrails` | above | 2A-3 | Pushed |
| `0.34.3/phase-2a4-pipeline-monitor` | above | 2A-4 | Pushed |
| `0.34.4/phase-2a5-parameter-bounds` | above | 2A-5 | Pushed (current) |
| `0.34.X/phase-2aY-*` | above | 2A-6, 2A-7, 2A-8 | Pending |
| TBD | phase2a complete | 2B items | Pending |

**Critical: Ralph (2B-1) gets its own branch.** It's the highest-risk change — restructuring the core iteration loop.

### Rollback Decision Criteria

| Signal | Threshold | Action |
|--------|-----------|--------|
| E2E pass rate drops | Below previous phase's baseline | Disable flag, investigate |
| Cost per run increases | >50% above projected savings | Disable flag, investigate |
| New crash type introduced | Any crash not seen before the change | Disable flag, investigate |
| Quality scores regress | Average score drops >10 points across 5 runs | Disable flag, investigate |

---

## Phase 2A: Quick Wins

**Goal:** Enable technique discovery, immediate cost savings, deterministic safety nets, monitoring infrastructure.
**Risk:** Low — all changes are additive or replace redundant code.
**Principles served:** P1 (reliability), P3 (Blender is truth), P4 (compute what you can), P5 (context precious).

### 2A-0: Documentation Pipeline (LLM-Optimized Manual) — DONE (2026-02-23)

**What:** Process the physics section of the Blender manual (119 pages) through an LLM rewrite pipeline, upload to the vector store, and validate that the research agent can discover new techniques.

**Why it matters:** This is the **highest-ROI item in the entire roadmap**. Without it, the self-learning pipeline has nothing to learn from. The research agent defaults to LLM training data — which contains 3-4 well-known Mantaflow techniques. Every downstream item that depends on technique diversity (UCB1, micro-experiments, multi-physics, cross-session learning) is building on sand without this.

**Evidence: Manual rewrite experiment (already run)**

| Page | Original Words | Rewritten Words | Compression | Cost |
|------|---------------|----------------|-------------|------|
| Domain Settings | 4,041 | 898 | 78% smaller | $0.003 |
| Flow | 2,387 | 919 | 61% smaller | $0.002 |
| Gas (index) | 37 | 281 | Expanded (sparse) | $0.001 |
| Noise | 756 | 451 | 40% smaller | $0.001 |
| Cache | 1,751 | 723 | 59% smaller | $0.002 |
| **Total** | **8,972** | **3,272** | **64% avg** | **$0.008** |

The rewritten output produces dense, structured content with explicit TECHNIQUE and GOTCHAS sections — exactly what the research agent needs.

**Prerequisite fix (already done):** Three bugs in `semantic_docs_tools.py` were fixed (see Foundation section above).

**Research support:**
- Critique §1: "This is the highest ROI work item in the entire Phase 2 plan"
- Mission Statement §4: "The manual should be optimized for LLM consumption"
- Mission Statement §12: "Technique monotony — System defaults to Mantaflow for everything"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `scripts/experiment_manual_rewrite.py` | Extend with `--all-physics` flag for batch processing + `--upload` flag | ~75 |
| New: `scripts/upload_rewritten_manual.py` | Upload rewritten pages to OpenAI vector store, verify upload, report stats | ~120 |
| `tools/semantic_docs_tools.py` | Update vector store IDs to include rewritten manual store | ~15 |
| New: `tests/test_technique_discovery.py` | Validate research agent discovers techniques it previously couldn't | ~30 |

**Workflow:**
1. Run `python scripts/experiment_manual_rewrite.py --all-physics --output rewritten_manual/` (~$0.19)
2. Review a sample of output for quality
3. Run `python scripts/upload_rewritten_manual.py --input rewritten_manual/`
4. Update `semantic_docs_tools.py` with new vector store ID
5. Run validation test

**Tests:**
1. Batch processing: run `--all-physics` on 5 pages → assert all produce non-empty output with TECHNIQUE section
2. Upload validation: upload 5 pages → assert vector store query returns results from rewritten content
3. Technique discovery: query "alternative fire techniques" → assert results include at least 2 techniques beyond mantaflow_gas
4. No regression: existing API reference queries still return relevant results
5. Cost check: processing 119 pages costs < $0.30

**Rollback:** Revert vector store ID to original manual store in `semantic_docs_tools.py`.

**Estimated effort:** ~240 lines new code + $0.19 processing cost | **Dependencies:** None | **Risk:** Low

**Completion notes:** ~130 physics/rendering pages rewritten to LLM-optimized format. Uploaded to separate vector store `vs_699b7e6221bc81919ba2f4a1eae11588`. Technique discovery tests pass. Manual work by Ben (scripts + upload + validation). Branch: `0.34.0/phase-2-implementation`. Also included: 7 new truth pack patterns (forcefield_add, openvdb_data_depth, cache formats, subframes), recovery path validation, false positive fix for bpy.ops.render.render, unfixable-error commenting.

---

### 2A-1: Conditional Tool Enabling (`is_enabled`) — DONE (2026-02-22)

**What:** Hide expensive tools from agents when budget is low, instead of letting agents try to use them and hitting guardrail errors.

**Why it matters:** Currently all 20+ tools are visible to every agent regardless of budget state. Vision evaluation tools remain visible when budget is exhausted, leading to guardrail-blocked calls that waste turns.

**Research support:**
- SDK Analysis §4: "Trivial to implement, immediate budget savings"
- Codebase Analysis §7.9, §7.10: "Learning Agent has 20+ tools — use is_enabled"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `specialized_agents/quality_analyst.py` | Add `is_enabled=budget_allows_vision` to vision evaluation tool | ~8 |
| `specialized_agents/docs_expert.py` | Add `is_enabled=budget_allows_docs` to doc search tools | ~8 |
| `orchestrator.py` (coordinator creation) | Wrap expensive agent-as-tool calls with `is_enabled` callbacks | ~15 |
| `tools/asset_evaluator_tools.py` | Move budget check from guardrail to `is_enabled` | ~15 |
| New: `utils/tool_visibility.py` | Callback functions for budget/phase/escape-level gating | ~60 |

**Callback examples:**
```python
def budget_allows_vision(ctx: RunContextWrapper, agent) -> bool:
    return ctx.context.budget_tracker.can_afford_evaluation()

def budget_allows_docs(ctx: RunContextWrapper, agent) -> bool:
    return ctx.context.budget_tracker.get_remaining() > 2.0

def learning_tool_for_iteration(ctx: RunContextWrapper, agent) -> bool:
    return ctx.context.session.current_iteration > 1
```

**Tests:**
1. Budget exhausted: set budget to $0 → assert vision tool not in agent.tools list
2. Budget available: set budget to $10 → assert vision tool IS in agent.tools list
3. Iteration gating: iteration=1 → assert mine_docs_for_patterns is hidden; iteration=3 → assert visible
4. Transition: exhaust budget mid-run → assert next agent call has tool hidden

**Rollback:** Set `ENABLE_CONDITIONAL_TOOLS = False` in `config/agent_config.py`.

**Estimated effort:** ~106 lines | **Dependencies:** None | **Risk:** Trivial

**Completion notes:** `utils/tool_visibility.py` created (~90 lines). 22 tests in `tests/test_tool_visibility.py`. Kill switch: `ENABLE_CONDITIONAL_TOOLS=0`. Commit: `a2506f3`. Branch: `0.34.0/phase-2-implementation`.

---

### 2A-2: Deterministic Agent Control (`tool_use_behavior`) — DONE (2026-02-22)

**What:** Eliminate unnecessary LLM post-processing calls for agents whose tool output IS the final answer.

**Why it matters:** The Executor agent calls `execute_blender_script`, then the LLM processes the result to produce a "response" — pure waste. Each unnecessary LLM call costs ~$0.01-0.02. Eliminating 4-5 saves $0.05-0.10/run.

**Research support:**
- SDK Analysis §5: "Saves ~$0.05-0.10/run from a single-line change"
- Codebase Analysis §6.5, Autonomy Research §3.1

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `specialized_agents/executor.py` | Add `tool_use_behavior="stop_on_first_tool"` | 1 |
| `specialized_agents/api_validator.py` | Add `tool_use_behavior="stop_on_first_tool"` | 1 |
| `specialized_agents/learning_agent.py` | Add `tool_use_behavior="stop_on_first_tool"` for recording-only mode | 1 |
| `orchestrator.py` (quality gate coordinator) | Add `StopAtTools(stop_at_tool_names=["make_quality_decision"])` | 5 |

**Tests:**
1. Executor: call with valid script → assert only 1 LLM call made
2. API Validator: call with script → assert only 1 LLM call
3. Quality Gate: call with score data → assert stops at make_quality_decision
4. Cost comparison: run 3 iterations with vs without → assert measurable cost reduction

**Rollback:** Remove `tool_use_behavior` parameter from agent definitions.

**Estimated effort:** ~8 lines | **Dependencies:** None | **Risk:** Trivial
**Monthly savings:** ~$2-5 at current run volume

**Completion notes:** `stop_on_first_tool` applied to Executor, API Validator. 10 tests in `tests/test_tool_use_behavior.py`. Commit: `b589d5e`. Branch: `0.34.0/phase-2-implementation`.

---

### 2A-3: Tool Guardrails for Truth Pack Enforcement — DONE (2026-02-22)

**What:** Move truth pack validation from an orchestrator pipeline step INTO the tool itself using `ToolInputGuardrail`. Every time any agent calls `execute_blender_script`, the script is automatically validated first. Tool guardrails make validation **impossible to bypass**.

**Why it matters:** Currently truth pack validation runs as a separate Phase 1.5 orchestrated by Python code. An agent could theoretically bypass it during error recovery (Phase 2.5-2.7).

**Research support:**
- SDK Analysis §6: "Deterministic validation as a tool wrapper — replaces an entire pipeline step with a zero-cost guardrail."
- Codebase Analysis §7.8, §3.3

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `guardrails/tool_guardrails.py` | Truth pack input guardrail, execution output guardrail, script length output guardrail | ~150 |
| `tools/blender_executor_tools.py` | Wrap `execute_blender_script` with truth pack input guardrail | ~8 |
| `tools/asset_evaluator_tools.py` | Wrap `evaluate_render` with output guardrail for critical failures | ~8 |
| `tools/script_generator_tools.py` | Wrap `generate_script` with output guardrail for script length check | ~8 |

**Tests:**
1. Hallucinated attribute: script with `resolution_divisions` → assert guardrail catches and auto-fixes to `resolution_max`
2. Clean script: script with all valid attributes → assert guardrail allows execution
3. Error recovery path: trigger Phase 2.5 recovery → assert recovery script is also validated
4. Script length warning: 200-line script → assert output guardrail warns "scene too basic"
5. Critical failure output: BLACK_SCREEN → assert output guardrail rejects with targeted message

**Rollback:** Set `ENABLE_TOOL_GUARDRAILS = False` in `config/agent_config.py`.

**Estimated effort:** ~174 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

**Completion notes:** `guardrails/tool_guardrails.py` (~275 actual lines — 1.6x estimate). 3 guardrails: truth_pack_input, critical_failure_output, script_length_output. Attaches via FunctionTool mutation. 22 tests in `tests/test_tool_guardrails.py`. SDK uses `ToolGuardrailFunctionOutput` with `behavior` dict (not `tripwire_triggered` like agent guardrails). Kill switch: `ENABLE_TOOL_GUARDRAILS=0`. Commit: `55756a2`. Branch: `0.34.2/phase-2a3-tool-guardrails`.

---

### 2A-4: PipelineMonitor (Deterministic Monitoring Layer) — DONE (2026-02-22)

**What:** A Python class (NOT an LLM agent) that runs at orchestrator checkpoints between pipeline phases. Detects parameter oscillation, stuck loops, wrong feedback cascades, budget overruns, and script quality issues.

**Why it matters:** Ben explicitly wants a monitoring layer (Interview Q10, Q16). The system's known failure modes are all detectable deterministically at $0 cost.

**Research support:**
- Autonomy Research §2.4 (MASC), Monitoring Architecture §1.1-§1.3, Mission Statement §9

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/pipeline_monitor.py` | `PipelineMonitor` class with `check_after_generation()`, `check_after_evaluation()`, `get_status_report()` | ~375 |
| `orchestrator.py` (iteration loop) | Add monitor checkpoint calls after generation and after evaluation | ~60 |
| `orchestrator.py` (iteration loop) | Wire monitor alerts into modification pipeline | ~45 |

**Signals monitored:**

| Signal | Threshold | Action |
|--------|-----------|--------|
| Parameter oscillation | Same param changes direction 2x in 3 iterations | Clamp to bounded midpoint |
| Stuck loop (same error) | 3x same `primary_issue` | Force technique switch (escape L2) |
| Score plateau | `abs(score_delta) < 3.0` for 3 iterations | Escalate escape velocity |
| Wrong feedback cascade | Score drops >5 after applying QA suggestion | Revert params, warn |
| Budget overrun | >$0.50 per run | Disable expensive tools |
| Script quality | <500 lines | Inject "scene too basic" warning |
| Critical render issue | BLACK_SCREEN, WHITE_SCREEN, ZERO_LIGHTS | Skip normal iteration, targeted fix |

**Tests:**
1. Oscillation detection: feed values [50, 2500, 10] → assert `is_oscillating() == True`
2. No false positive: feed values [50, 100, 150] → assert `is_oscillating() == False`
3. Cascade detection: score drops 8 points after QA suggestion → assert WARNING alert with revert action
4. Budget alert: $0.60 spent on $0.50 budget → assert CRITICAL alert
5. Integration: mock a 3-iteration loop → verify monitor produces status artifact at each checkpoint

**Rollback:** Set `ENABLE_PIPELINE_MONITOR = False` in `config/agent_config.py`.

**Estimated effort:** ~480 lines | **Dependencies:** None | **Risk:** Low

**Completion notes:** `tools/pipeline_monitor.py` (~370 actual lines). 7 detection signals. Oscillation threshold tuned: `OSCILLATION_DIRECTION_CHANGES=1` (not 2 — only 1 direction change possible in a 3-element window). 26 tests in `tests/test_pipeline_monitor.py`. Kill switch: `ENABLE_PIPELINE_MONITOR=0`. Commit: `75178c8`. Branch: `0.34.3/phase-2a4-pipeline-monitor`.

---

### 2A-5: Parameter Bounds and Damped Convergence — DONE (2026-02-23)

**What:** Per-effect-type parameter bounds that prevent overcorrection. All parameter modifications are clamped to safe ranges and limited to a maximum step size per iteration.

**Why it matters:** Light energy oscillation (50 → 2500 → 10) is a known top failure mode (Mission Statement §12).

**Research support:**
- Monitoring Architecture §3.3, Autonomy Research §5.3, Mission Statement §12

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/parameter_bounds.py` | `ParameterBound` dataclass, `PARAMETER_BOUNDS` dict per effect type, `clamp()` and `damped_change()` methods | ~180 |
| `orchestrator.py` (modification phase) | Apply bounds before passing parameter changes to Script Writer | ~45 |

**Example bounds:**
```python
PARAMETER_BOUNDS = {
    "fire": {
        "energy": ParameterBound("energy", 20.0, 200.0, step_size=50.0),
        "density": ParameterBound("density", 1.0, 15.0, step_size=3.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 0.5, 10.0, step_size=2.0),
    },
    "liquid": {
        "energy": ParameterBound("energy", 30.0, 300.0, step_size=80.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 5.0, step_size=1.0),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
}
```

**Tests:**
1. Clamp: propose energy=5000 with max=200 → assert clamped to 200
2. Damped change: current=50, target=200, step_size=50 → assert result=100
3. Within bounds: propose energy=150 within [20, 200] → assert unchanged
4. Effect type routing: fire bounds applied for fire, NOT applied for liquid
5. Integration with monitor: oscillating parameter gets tighter bounds

**Rollback:** Set `ENABLE_PARAMETER_BOUNDS = False` in `config/agent_config.py`.

**Estimated effort:** ~225 lines | **Dependencies:** PipelineMonitor (2A-4) for runtime bounds refinement | **Risk:** Low

**Completion notes:** `tools/parameter_bounds.py` (~175 lines). 9 effect types bounded (fire, explosion, smoke, pyro, liquid, water, water_splash, nebula, sun). Oscillating params get halved step via PipelineMonitor integration. Wired into `_apply_script_modifications()` covering all 4 call paths. 40 tests in `tests/test_parameter_bounds.py`. Kill switch: `ENABLE_PARAMETER_BOUNDS=0`. Commit: `4dd3335`. Branch: `0.34.4/phase-2a5-parameter-bounds`.

---

### 2A-6: Deprecate Spec-First Pipeline — DONE

**What:** Remove the LLM-powered API Spec Agent from the pipeline. The Truth Pack provides the same data deterministically at $0.

**Why it matters:** The API Spec Agent costs ~2 LLM calls per run. The Truth Pack provides the same data from Blender introspection. Running both is redundant.

**Research support:**
- Codebase Analysis §7.4, §10

**Completion notes:**
- **Branch:** `0.34.5/phase-2a6-deprecate-spec-first`
- Removed 4 dead methods (584 lines): `_run_api_spec_only`, `_run_parallel_technique_and_spec`, `_run_spec_first_pipeline`, `_run_spec_first_modification`
- Removed 3 instance variables: `self._use_spec_first_pipeline`, `self._api_spec_agent`, `self._code_writer_agent`
- Removed API Spec Agent + Code Writer Agent creation from `initialize()`
- Cleaned 7 call sites in pipeline: Phase 0.5, Phase 1, execution failure escalation, technique switch, modify code, fallback modification, error recovery
- Added deprecation notices to `api_spec_agent.py` and `code_writer_agent.py`
- **Net reduction:** ~730 lines from `orchestrator.py` (943 deleted, 211 inserted)
- **Tests:** 24 tests in `tests/test_deprecate_spec_first.py`
- **Preserved:** `truth_pack_to_api_spec()`, `models/api_spec.py`, `context.truth_pack`, `context.api_spec`

**What's preserved:** `models/api_spec.py` Pydantic models stay — `truth_pack_to_api_spec()` populates them from truth pack data, maintaining backward compatibility.

**Tests (24 total):**
1. `TestTruthPackToApiSpec` (11 tests): truth_pack_to_api_spec() produces valid APISpec from truth pack
2. `TestSpecFirstRemoval` (7 tests): All spec-first attributes, methods, and calls removed from orchestrator
3. `TestScriptWriterContext` (4 tests): Script Writer still receives truth pack data via context
4. `TestDeprecationNotices` (2 tests): Deprecation notices present on old modules

**Rollback:** Re-enable imports and agent creation in orchestrator `initialize()`. Deprecated files are preserved with instructions.

**Estimated effort:** ~128 lines | **Actual:** ~730 lines net removed | **Dependencies:** Truth Pack (Phase 1, done) | **Risk:** Low-Medium
**Savings:** ~$0.04-0.10/run (2 LLM calls eliminated)

---

### 2A-7: Remove Dead Executor Agent

**What:** Phase 2 (Execution) calls `_execute_blender_script_impl()` directly — deterministic, no LLM. The Executor agent (170 LOC) is created in `initialize()` but never used. Remove it.

**Research support:**
- Codebase Analysis §7.3: "Dead code."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (initialize) | Remove Executor agent creation | ~30 |
| `specialized_agents/executor.py` | Add deprecation notice or delete | ~8 |

**Tests:**
1. E2E pipeline: run full pipeline → assert execution still works via direct call
2. No import: assert `orchestrator.py` does not import from `specialized_agents/executor.py`

**Rollback:** Not needed — dead code removal. File is in git history.

**Estimated effort:** ~38 lines | **Dependencies:** None | **Risk:** Trivial

---

### 2A-8: Function Tool Timeouts

**What:** Add `timeout` parameter to Blender execution tools to prevent hung pipelines.

**Research support:**
- SDK Analysis §B1

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/blender_executor_tools.py` | Add `timeout=300.0` to `execute_blender_script` function_tool | 2 |
| `tools/blender_executor_tools.py` | Add `failure_error_function=blender_error_handler` for actionable error messages | ~30 |

**Tests:**
1. Timeout fires: mock hung Blender subprocess → assert timeout error within 300s
2. Normal execution: run valid script → assert completes under timeout
3. Error handler: simulate MemoryError → assert actionable message

**Rollback:** Remove `timeout` parameter.

**Estimated effort:** ~32 lines | **Dependencies:** None | **Risk:** Trivial

---

### Phase 2A Summary

| Item | Est. Lines | Actual Lines | Tests (est/actual) | Status |
|------|-----------|-------------|-------------------|--------|
| **2A-0: Documentation Pipeline** | **~240** | ~130 pages rewritten + scripts | **5 / 5** | **DONE** |
| 2A-1: is_enabled | ~106 | ~90 | 4 / 22 | **DONE** |
| 2A-2: tool_use_behavior | ~8 | ~8 | 4 / 10 | **DONE** |
| 2A-3: Tool guardrails | ~174 | ~275 | 5 / 22 | **DONE** |
| 2A-4: PipelineMonitor | ~480 | ~370 | 5 / 26 | **DONE** |
| 2A-5: Parameter bounds | ~225 | ~175 + 45 wiring | 5 / 40 | **DONE** |
| 2A-6: Deprecate Spec-First | ~128 | ~730 net removed + 24 test lines | 3 / 24 | **DONE** |
| 2A-7: Remove Executor agent | ~38 | ~35 removed + 2 test | 2 / 2 | **DONE** |
| 2A-8: Tool timeouts | ~32 | ~50 | 3 / 14 | **DONE** |
| **Total** | **~1,431** | | **55 / 192** (includes extras) | **9/9 COMPLETE** |

**Note:** Line estimates include a 1.5x multiplier based on Phase 1 experience.

**Phase 2A success criteria:**
1. Research agent discovers >= 3 new techniques per effect type from rewritten manual (2A-0)
2. No parameter oscillation across a 5-iteration run (monitor catches and clamps)
3. Budget per 3-iteration run drops below $0.30 (from ~$0.50)
4. All scripts validated by tool guardrails — zero hallucinated attributes reach Blender
5. PipelineMonitor produces status artifacts for every iteration
6. Dead code removed (Executor agent, Spec-First pipeline)

---

## KB Seeding Strategy

**Position:** Runs immediately after 2A-0 (Manual Rewrite) completes.
**Principle served:** P6 (every run produces learning signal) — but for the KB, we need initial signal to learn FROM.

### Context

Phase 1 wiped the KB (18 stale entries removed). The manual rewrite (2A-0) produces LLM-optimized documentation. Between these two events, the KB is empty and the documentation is fresh. This is the optimal moment to bootstrap the KB with high-quality technique knowledge.

### Seeding Workflow

```
Manual Rewrite Complete (2A-0)
        │
        ▼
Run Research Agent against 6 common effect types:
  1. fire (mantaflow gas — burning)
  2. smoke (mantaflow gas — non-burning)
  3. liquid (mantaflow liquid — pour/splash)
  4. rigid body (destruction/shattering)
  5. particles (debris/sparks/rain)
  6. cloth (fabric/flag/curtain)
        │
        ▼
For each effect type:
  - Research agent queries rewritten manual + API reference
  - Discovers available techniques, parameter relationships, workflows
  - Records findings as KB entries with:
    - effect_type: scoped to the queried type
    - trust_level: "emerging" (not yet production-validated)
    - source: "manual_seed_2026-02-XX"
        │
        ▼
Validate seeded KB:
  - Query KB for each effect type → verify non-empty results
  - Verify technique diversity: >= 2 distinct techniques per effect type
  - Verify no hallucinated attributes (truth pack cross-check)
```

### Cost

| Item | Cost |
|------|------|
| 6 research agent runs (gpt-5-mini) | ~$0.30-0.60 |
| Manual rewrite processing (already done in 2A-0) | $0.19 |
| **Total seeding cost** | **~$0.50-0.80** |

### What Gets Seeded vs What Gets Earned

| Seeded (bootstrap) | Earned (production evidence) |
|--------------------|-----------------------------|
| Technique names and descriptions | Quality scores per technique |
| Parameter relationships from manual | Optimal parameter values |
| Known gotchas and warnings | Specific code patterns that work |
| Alternative approaches per effect type | Evidence-gated trust levels |

Seeded entries start at "emerging" trust level. They are NOT injected into the Script Writer's dynamic instructions until they earn "trusted" status through 3+ successful production uses.

### Implementation

| File | Change | Lines |
|------|--------|-------|
| New: `scripts/seed_kb.py` | Run research agent against 6 effect types, store findings in KB | ~100 |
| `tools/experiment_tracker_tools.py` | Add `seed_technique_entry()` with "manual_seed" source tag | ~30 |

**Estimated effort:** ~130 lines + ~$0.50-0.80 API cost

---

## Phase 2B: Core Architecture

**Goal:** Stateless iterations, precision context management, multi-grader evaluation, HITL framework.
**Risk:** Medium — changes iteration loop structure and context management.
**Principles served:** P1 (reliability), P5 (context precious), P6 (learning signal), P7 (evidence).

**Schedule note:** 2B-1 (Ralph-Style Stateless Iterations) gets its own dedicated sprint in Week 2 with isolated E2E testing. All other 2B items are Week 3+. Ralph is the highest-risk change in Phase 2 and must not be bundled with other work.

### 2B-1: Ralph-Style Stateless Iterations — DONE

**Status:** COMPLETE | **Branch:** `0.34.9/phase2b1-stateless-iteration` | **Tests:** 23 new (215 total) | **Actual lines:** ~240

**What shipped:** Feature-flagged stateless iterations. In stateless mode (`stateless_iterations: true`, default), iteration-loop `Runner.run()` calls get `session=None` instead of `session=sdk_session`. Pre-loop phases (0, 0.5) keep `sdk_session` for research continuity. Compaction is skipped in stateless mode.

**What changed from the original spec:** See `docs/PHASE2B1_IMPLEMENTATION_NOTES.md` for full delta analysis. Key finding: the architecture was already 80% stateless — SharedContext rebuilds each iteration, SessionState persists to disk, results pass via Python variables, prompts embed context via f-strings. The roadmap's 5-day incremental rollout and ~600 LOC estimate assumed conversation history carried unique context. It didn't.

**Implementation approach:** All-at-once behind feature flag (not gradual iteration-by-iteration). The `IterationSnapshot` utility provides optional prompt enrichment but was NOT required — existing prompts already embed all needed context.

| File | Change | Lines |
|------|--------|-------|
| `config/presets.yaml` | `stateless_iterations` flag on all 7 presets | ~7 |
| `config/agent_config.py` | `PresetConfig.stateless_iterations` field + accessor | ~10 |
| New: `utils/iteration_state.py` | `IterationSnapshot` frozen dataclass (safety net for prompt enrichment) | ~85 |
| `orchestrator.py` | `iter_session` variable, 12 session replacements, compaction guard | ~25 changed |
| New: `tests/test_stateless_iterations.py` | 5 test classes, 23 tests | ~120 |

**Rollback:** Set `stateless_iterations: false` in the active preset. All existing behavior preserved.

**Estimated effort (roadmap):** ~600 lines | **Actual effort:** ~240 lines | **Risk realized:** Low (not Medium)

---

### 2B-2: Context Trimming (`call_model_input_filter`)

**What:** Per-agent input filters that trim conversation history before each LLM call. Each agent type gets only the context it needs.

**Why it matters:** Even within a single iteration, agents accumulate tool call outputs. The Script Writer doesn't need the Learning Agent's experiment recording. The Quality Analyst shouldn't see its own previous evaluations (prevents self-reinforcing bias).

**Research support:**
- SDK Analysis §2, Monitoring Architecture §2.2, Codebase Analysis §7.5

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/context_filters.py` | Per-agent `call_model_input_filter` functions | ~120 |
| `orchestrator.py` (agent creation) | Add filters to `RunConfig` in `_run_agent()` | ~23 |

**Per-agent strategy:**

| Agent | Keep | Drop | Token Budget |
|-------|------|------|-------------|
| ScriptWriter | System + truth pack + current prompt | All previous iterations | ~8K |
| QualityAnalyst | System + current render/script path | Previous evaluations | ~4K |
| LearningAgent | System + last 2 iteration summaries | Old research, old scripts | ~6K |
| ModificationStrategist | System + current prompt | Previous modification history | ~4K |
| QualityGateJudge | System + current prompt | Everything else | ~2K |

**Tests:**
1. ScriptWriter filter: inject 10 messages → assert filter keeps only system + last message
2. QualityAnalyst isolation: inject previous QA results → assert filter strips them
3. Token budget: assert each agent's filtered input is within its budget
4. No data loss: assert current iteration's critical data survives filtering

**Rollback:** Set `ENABLE_CONTEXT_TRIMMING = False` in `config/agent_config.py`.

**Estimated effort:** ~143 lines | **Dependencies:** 2B-1 (Ralph reduces baseline context) | **Risk:** Low

---

### 2B-3: Multi-Grader Quality Evaluation

**What:** Three-tier evaluation pipeline: deterministic checks ($0) → ML metrics ($0 local) → LLM vision ($0.01-0.05). Short-circuits on critical failures.

**Why it matters:** Currently the full quality evaluation runs on every render, even obviously failed ones. Deterministic pre-checks catch critical failures for free.

**Research support:**
- Autonomy Research §5.1, Monitoring Architecture §3.2, Mission Statement §4

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/deterministic_quality_checks.py` | Tier 1: render exists? correct size? not blank? lights in script? camera bounds? | ~270 |
| `orchestrator.py` (Phase 3) | Insert Tier 1 before ML/vision. Short-circuit on critical issues. | ~75 |
| `tools/asset_evaluator_tools.py` | Scoring weights: Tier 1 (pass/fail) + ML (60%) + Vision (40%) | ~60 |

**Short-circuit logic:**
```
Tier 1 critical issue → score=0, skip Tiers 2+3 (save $0.05)
Tier 2 score > 80, no issues → skip Tier 3 (save $0.05)
Tier 3 always runs on iteration 1 (baseline assessment)
```

**Tests:**
1. Black screen: blank render → assert Tier 1 catches, Tiers 2+3 skipped, score=0
2. Good render: quality render → assert all 3 tiers run, combined score reflects weighted average
3. High ML score: Tier 2 score=85 → assert Tier 3 skipped (except iteration 1)
4. Cost tracking: 5 iterations with 2 critical failures → assert 2 vision calls skipped
5. Scoring weights: ML=70, Vision=50 → assert combined = 70*0.6 + 50*0.4 = 62.0

**Rollback:** Set `ENABLE_MULTI_GRADER_EVAL = False` in `config/agent_config.py`.

**Estimated effort:** ~405 lines | **Dependencies:** None | **Risk:** Low

---

### 2B-4: AdvancedSQLiteSession (Token Tracking + Branching)

**What:** Replace `SQLiteSession` + `OpenAIResponsesCompactionSession` with `AdvancedSQLiteSession` for built-in token tracking, conversation branching, and keyword search.

**Why it matters:** Token tracking replaces the brittle character-count heuristic. Branching enables trying different techniques from the same starting point — critical for escape velocity L2+.

**Research support:**
- SDK Analysis §1, Codebase Analysis §6.1, §7.7

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py:367-401` | Replace `get_or_create_sdk_session()` with `AdvancedSQLiteSession` | ~45 |
| `orchestrator.py` (technique switch) | `branch_id = await session.create_branch_from_turn(...)` | ~60 |
| `orchestrator.py` (budget tracking) | Replace custom budget tracking with `session.store_run_usage(result)` | ~75 |
| `session_manager.py` | Add `branches: Dict[str, BranchInfo]` to SessionState | ~30 |

**Tests:**
1. Session swap: replace SQLiteSession → assert pipeline still runs end-to-end
2. Token tracking: 3 iterations → assert `get_session_usage()` matches actual API usage within 15%
3. Branching: create branch at turn 2 → assert original preserved, branch starts from turn 2
4. Compaction: verify compaction wrapping still works with AdvancedSQLiteSession

**Rollback:** Set `ENABLE_ADVANCED_SESSION = False` in `config/agent_config.py`.

**Estimated effort:** ~210 lines | **Dependencies:** None (parallel with 2B-1) | **Risk:** Medium

---

### 2B-5: HITL Framework — **DONE**

**What:** Pipeline-level HITL checkpoints that pause execution when the system needs human guidance. Two modes: interactive (CLI `input()`) and non-interactive (save checkpoint to session, PAUSED). Autonomy levels (0-4) gate which checkpoints fire.

**Why it matters:** Ben explicitly wants this (Mission Statement §9). Previously Level 4 escape velocity set `session.status = PAUSED` with no structured pause/resume flow.

**SDK reality:** `needs_approval` is MCP-only (`HostedMCPTool`). `RunResult` has no `interruptions` field. Implemented as pure Python pipeline-level checkpoints per Architecture doc recommendation.

**Implementation (actual):**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/hitl_handler.py` | Core HITL logic: CheckpointType/Decision enums, HITLCheckpoint dataclass, HITLHandler with autonomy gating | ~230 |
| `models/shared_context.py` | `pending_checkpoint`, `hitl_history`, `autonomy_level` on SessionState | ~15 |
| `config/agent_config.py` | `hitl_enabled`, `hitl_autonomy_level`, `hitl_interactive` + accessors | ~25 |
| `config/presets.yaml` | HITL flags on all 7 presets | ~21 |
| `orchestrator.py` | 4 integration points: import+init, resume, post-eval (Phase 3.95), escape L4 | ~60 |
| New: `tests/test_hitl_framework.py` | 26 tests: models, autonomy gating, triggers, interactive mock, resume, config | ~480 |

**HITL triggers:**

| Checkpoint | Trigger | Autonomy Levels |
|-----------|---------|-----------------|
| Quality plateau | Score plateau 3+ iterations | 0 only |
| Stall detection | Same issue 3x at escape >= 2 | 0, 1 |
| Budget warning | >80% budget spent | 0, 1 |
| Critical issue | Score=0 for 2+ consecutive iterations | 0, 1, 2 |
| Escalation (L4) | Escape velocity Level 4 | 0, 1, 2, 3 |

**Tests:** 26 pass — checkpoint models, autonomy gating (5 levels), budget/stall/critical triggers, escalation, interactive prompt mocking, pending resume, config integration, disabled handler.

**Rollback:** Set `hitl_enabled: false` in the active preset.

---

### 2B-6: Ebbinghaus Memory Decay

**What:** Knowledge base entries decay exponentially if not reinforced by successful outcomes. Entries below retention threshold are archived (not deleted).

**Why it matters:** Without decay, stale patterns from early (low-quality) runs will pollute the KB within weeks.

**Research support:**
- Autonomy Research §1.3 (SAGE), Mission Statement §7

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `utils/code_pattern_memory.py` | Add `compute_retention(entry, now)` | ~45 |
| `tools/code_pattern_tools.py` | Filter patterns by retention during `search_patterns()` | ~23 |
| `tools/dynamic_instructions.py` | Filter KB entries by retention during injection | ~23 |
| `tools/experiment_tracker_tools.py` | Add `reinforce_entry(entry_id)` | ~30 |

**Decay function:**
```python
def compute_retention(entry, now):
    days_since = (now - entry.last_reinforced).days
    strength = entry.success_count * entry.avg_quality_score / 100
    return math.exp(-days_since / max(strength, 0.1))
```

**Thresholds:** retention < 0.3 → excluded from queries. retention < 0.1 → archived.

**Tests:**
1. Fresh entry: created today → assert retention = 1.0, included in queries
2. Stale entry: last reinforced 60 days ago, low strength → assert retention < 0.3, excluded
3. Reinforcement: use pattern successfully → assert `last_reinforced` resets, retention ~1.0
4. Archive threshold: retention < 0.1 → assert entry moved to archive, not deleted

**Rollback:** Set `ENABLE_MEMORY_DECAY = False` in `config/agent_config.py`.

**Estimated effort:** ~121 lines | **Dependencies:** KB wipe (Phase 1, done) | **Risk:** Low

---

### 2B-7: Effect-Type-Scoped Evidence Gating

**What:** A pattern's trust level is tracked per effect type, not globally.

**Why it matters:** Currently a `density=5.0` pattern that works for fire gets "Trusted" status and may be injected into a liquid script where it's wrong.

**Research support:**
- Autonomy Research §1.2

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/experiment_tracker_tools.py` | Add `effect_type` field to KB entries | ~30 |
| `tools/knowledge_distillation_tools.py` | Filter distilled patterns by effect type | ~15 |
| `tools/dynamic_instructions.py` | Filter KB injection by current effect type | ~15 |

**Tests:**
1. Scoped storage: store pattern for "fire" → assert `effect_type="fire"`
2. Scoped retrieval: query for "liquid" → assert fire-only patterns NOT returned
3. Cross-type: pattern used for both → assert separate trust scores per type
4. Dynamic instructions: `effect_type="fire"` → assert only fire-scoped patterns injected

**Rollback:** Set `ENABLE_EFFECT_SCOPING = False` in `config/agent_config.py`.

**Estimated effort:** ~60 lines | **Dependencies:** 2B-6 (memory decay) | **Risk:** Low

---

### 2B-8: Artifact-Based Sharing Formalization

**What:** Mandate that every agent returns a structured summary (< 500 tokens) with a file path to the full report. The orchestrator never receives full reports inline.

**Why it matters:** Research output (~3000 tokens), quality evaluation (~2000 tokens), and Blender stdout (~1000-5000 tokens) all bloat context. Artifact-based sharing reduces this to ~200 tokens per handoff.

**Research support:**
- Autonomy Research §2.2 (Anthropic), §3.2, Monitoring Architecture §2.3

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (all phase prompts) | Replace inline data with artifact paths + 1-line summaries | ~120 |
| New: `tools/artifact_tools.py` | `read_artifact` tool for agents | ~45 |
| Artifact outputs: `monitor_iter{N}.md`, `params_iter{N}.json`, `diagnosis_iter{N}.md` | Written by orchestrator/monitor | ~60 |

**Tests:**
1. Token reduction: compare prompt size with inline data vs artifact paths → assert >50% reduction
2. Artifact readability: write artifact → read with `read_artifact` → assert content matches
3. Agent access: agent calls `read_artifact` → assert full report returned (capped at 4K chars)
4. Missing artifact: call on non-existent path → assert graceful error message

**Rollback:** Set `ENABLE_ARTIFACT_SHARING = False` in `config/agent_config.py`.

**Estimated effort:** ~225 lines | **Dependencies:** 2A-4 (PipelineMonitor for monitor artifacts) | **Risk:** Low

---

### Phase 2B Summary

| Item | Lines | Impact | Tests | Sprint |
|------|-------|--------|-------|--------|
| **2B-1: Ralph stateless iterations** | **~240 actual** | **Critical — eliminates context degradation** | **23** | **DONE — branch `0.34.9/phase2b1-stateless-iteration`** |
| 2B-2: Context trimming | ~143 | High — per-agent precision | 4 | Week 3 |
| **2B-3: Multi-grader evaluation** | **~390 actual** | **High — saves $0.05/failed render** | **28** | **DONE — deterministic Tier 1 checks before LLM vision** |
| 2B-4: AdvancedSQLiteSession | ~210 | Medium — branching + token tracking | 4 | Week 3 (parallel w/ 2B-1 testing) |
| **2B-5: HITL framework** | **~830 actual** | **Medium — user-requested** | **26** | **DONE — pipeline-level checkpoints with autonomy gating** |
| 2B-6: Memory decay | ~121 | Medium — prevents KB re-poisoning | 4 | Week 3 |
| 2B-7: Effect-type evidence gating | ~60 | Medium — prevents cross-effect contamination | 4 | Week 3 (after 2B-6) |
| 2B-8: Artifact-based sharing | ~225 | High — 75% token reduction per handoff | 4 | Week 3 |
| **Total** | **~2,027** | | **35** | |

**Note:** Line estimates include a 1.5x multiplier based on Phase 1 experience.

**Phase 2B schedule (revised for Ralph isolation):**
```
Week 2: 2B-1 (Ralph iterations) ALONE
├── Day 1-2: IterationState dataclass + serialization
├── Day 2-3: Convert iteration 2 only to stateless
├── Day 3-4: E2E A/B comparison: old vs new iteration 2
├── Day 4-5: If quality holds, convert all iterations
└── E2E validation: 3-iteration fire run with full monitoring

Week 3: All remaining 2B items
├── 2B-4 (AdvancedSQLiteSession) — can start parallel with Week 2
├── 2B-6 (memory decay) — independent
├── 2B-2 (context trimming) — depends on stable 2B-1
├── 2B-3 (multi-grader eval) — independent
├── 2B-7 (effect-type evidence) — depends on 2B-6
└── 2B-8 (artifact sharing) — depends on 2A-4

Week 4: 2B-5 (HITL framework)
├── Depends on 2A-4 (PipelineMonitor)
└── E2E validation of Phase 2A+2B together
```

**Phase 2B success criteria:**
1. Known effects (fire, liquid, smoke) score >= 60 in 50%+ of runs
2. Context per agent stays within token budget (8K script writer, 4K QA, etc.)
3. 5-iteration runs show no quality cliff at iteration 3+
4. AdvancedSQLiteSession token tracking matches actual API usage within 15%
5. HITL pauses at escape level 4 with full state serialization and successful resume
6. KB entries show decay over time, no stale patterns accumulate
7. All 35 Phase 2B tests pass

---

## Orchestrator Decomposition Gate

**Position:** Between Phase 2B and Phase 2C. Mandatory before any 2C work begins.
**Type:** Refactoring gate — no new functionality.

### Why Now

**Not earlier:** Ralph (2B-1) restructures the iteration loop significantly (~600 lines changed in `orchestrator.py`). Decomposing before Ralph would mean decomposing twice.

**Not later:** Phase 2C adds UCB1 technique selection, micro-experiments, multi-physics support, and budget degradation — all of which touch `orchestrator.py`. The file is currently ~4,544 LOC. Phase 2C could push it past 6,000 LOC.

**Now is the window:** After Ralph stabilizes the iteration loop structure, before 2C adds complexity.

### Proposed Module Structure

```
orchestrator.py (~4,544 LOC current)
    ↓ decompose into:

orchestrator.py (~500 LOC)         # Pipeline coordinator — creates agents, routes between phases
phases/
├── __init__.py
├── research.py (~400 LOC)         # Phase 0: Research + technique selection
├── generation.py (~600 LOC)       # Phase 1: Script generation + truth pack validation
├── execution.py (~300 LOC)        # Phase 2: Blender execution + error handling
├── evaluation.py (~500 LOC)       # Phase 3: Quality evaluation (3-tier after 2B-3)
├── modification.py (~400 LOC)     # Phase 4: Modification strategy + script updates
├── recovery.py (~500 LOC)         # Phase 2.5-2.9: Error recovery pipeline
└── iteration.py (~300 LOC)        # Ralph iteration state management (from 2B-1)
```

### Gate Criteria (All Must Pass)

1. 2B-1 (Ralph stateless iterations) shipped and passing E2E validation
2. `create_asset_pipeline()` decomposed into phase modules
3. E2E test passes with decomposed orchestrator (identical behavior to monolith)
4. Each phase module independently testable
5. No file exceeds 700 LOC
6. `orchestrator.py` contains ONLY coordination logic

**Estimated effort:** ~200 lines (restructuring, not new logic) | **Risk:** Low (mechanical refactoring)

**Rollback:** If decomposition introduces regressions, revert to monolith and retry. Phase 2C cannot proceed until this gate passes.

---

## Phase 2C: Advanced Capabilities

**Goal:** Multi-physics support, technique diversity, experimentation mode.
**Risk:** Medium — extends architecture to new physics types.
**Principles served:** P2 (LLM creativity), P3 (Blender is truth), P6 (learning signal).
**Prerequisite:** Orchestrator Decomposition Gate (see above) must pass before starting any 2C work.

### 2C-1: UCB1 Technique Selector

**What:** Replace LLM-only technique selection with UCB1 (Upper Confidence Bound) as the default algorithm, with LLM override for novel prompts.

**Why it matters:** Technique selection is a classic multi-armed bandit problem. UCB1 provides optimal exploration/exploitation balance.

**Important dependency note:** UCB1 provides full value only after 2A-0 populates technique diversity. With only 3-4 known techniques, UCB1 reduces to round-robin.

**Research support:**
- Autonomy Research §4.2 (IBM/AAAI 2026), Mission Statement §8

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/technique_selector.py` | `select_technique_ucb1()` — UCB1 algorithm with statistics | ~120 |
| `orchestrator.py` (Phase 0.5) | UCB1 as pre-filter, then Technique Selection Coordinator | ~45 |
| `tools/experiment_tracker_tools.py` | Add `get_technique_statistics(effect_type)` | ~60 |

**Tests:**
1. UCB1 selects untried technique over tried technique with score 80 (exploration)
2. UCB1 selects highest-scoring when all tried 10+ times (exploitation)
3. UCB1 balances: 2 tries/score 50 selected over 20 tries/score 55 (exploration bonus)
4. LLM override: research agent flags specific technique → UCB1 bypassed
5. Integration: `get_technique_statistics()` correct after 5 mock runs

**Rollback:** Set `ENABLE_UCB1_SELECTOR = False` in `config/agent_config.py`.

**Estimated effort:** ~225 lines | **Dependencies:** 2B-7 (effect-type scoping); soft: 2A-0 | **Risk:** Low

---

### 2C-2: Multi-Physics Truth Pack Extension

**What:** Extend truth pack `TECHNIQUE_TYPES` mapping to cover rigid body, particle systems, cloth, soft body, and geometry nodes.

**Why it matters:** Currently `TECHNIQUE_TYPES` covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus `_common`. Missing: cloth, geometry_nodes, soft_body.

**Research support:**
- Codebase Analysis §8.1, Mission Statement §15

**Phased rollout:**
- **First:** Rigid body + particle systems (P1 priority)
- **Then:** Cloth + soft body + geometry nodes (P2 priority)

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/truth_pack.py` | Add `TECHNIQUE_TYPES` entries for `cloth`, `soft_body`, `geometry_nodes` | ~45 |
| `tools/truth_pack.py` | Add `SETTINGS_MAP` entries for new types | ~60 |
| `tools/parameter_bounds.py` | Add bounds for new physics types | ~90 |

**Tests:**
1. Truth pack generates valid introspection data for `ClothSettings`
2. Truth pack generates valid introspection data for `RigidBodyObject`
3. Parameter bounds exist and are sane for each new physics type
4. `SETTINGS_MAP` for geometry nodes includes `GeometryNodeGroup`
5. Existing mantaflow/rigid_body introspection unchanged (regression)

**Rollback:** New `TECHNIQUE_TYPES` entries are additive. Removing them restores previous behavior.

**Estimated effort:** ~195 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

---

### 2C-3: Micro-Experiment Sandbox Mode

**What:** Lightweight experiment runner that tests techniques in isolation with minimal scripts (50-100 lines, 10-30 second execution). Findings feed into the knowledge base.

**Why it matters:** When stuck (escape level 2+), the system should test a technique quickly before committing to a full 500+ line scene script.

**Research support:**
- Autonomy Research §4.1, Mission Statement §8, Interview

**Critical dependency:** Micro-experiments need technique candidates from the research agent, which depends on 2A-0 for discovering techniques beyond training data.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/micro_experiments.py` | `generate_micro_experiment()`, `run_micro_experiment()`, `ExperimentResult` | ~300 |
| `orchestrator.py` (escape velocity L2+) | Run micro-experiment before full technique switch | ~60 |
| `orchestrator.py` (novel prompt) | Run experiments on candidate techniques | ~60 |

**Triggers:**
1. Novel prompt that doesn't match known techniques
2. Escape velocity level 2+
3. New truth pack version detected (Blender update)
4. User-requested exploration
5. Between production runs (idle time)

**Tests:**
1. `generate_micro_experiment("fire", "mantaflow_gas")` produces script < 100 lines
2. `run_micro_experiment()` completes in < 30 seconds
3. Failed experiment returns `ExperimentResult(success=False)` with parsed error
4. Successful experiment feeds into KB with trust level "emerging"
5. Escape velocity L2 triggers micro-experiment before full technique switch

**Rollback:** Set `ENABLE_MICRO_EXPERIMENTS = False` in `config/agent_config.py`.

**Estimated effort:** ~420 lines | **Dependencies:** 2C-1, 2C-2; soft: 2A-0 | **Risk:** Medium

---

### 2C-4: Graceful Budget Degradation

**What:** Instead of binary budget check (full evaluation OR stop), implement tiered evaluation based on remaining budget.

**Research support:**
- Codebase Analysis §7.10

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/asset_evaluator_tools.py` | Add `evaluate_render_lightweight()` — ML metrics only | ~60 |
| `orchestrator.py` (Phase 3) | Budget-based evaluation tier selection | ~45 |
| `tools/deterministic_quality_checks.py` | Add `evaluate_render_deterministic()` | ~45 |

**Tier selection:**
```
Budget > 50% → Full evaluation (deterministic + ML + vision)
Budget 20-50% → ML-only (deterministic + ML, skip vision)
Budget < 20% → Deterministic-only (histogram + script checks)
Budget exhausted → Pipeline stops with best result
```

**Tests:**
1. Budget at 60% → full evaluation runs
2. Budget at 30% → ML-only (no vision API call)
3. Budget at 10% → deterministic-only
4. Budget at 0% → pipeline stops gracefully
5. ML-only score within 15 points of full evaluation on same render

**Rollback:** Set budget thresholds to `0%` to always use full evaluation.

**Estimated effort:** ~150 lines | **Dependencies:** 2B-3 (multi-grader evaluation) | **Risk:** Low

---

### 2C-5: Enhanced QA Diagnosis Bridge

**What:** Extend the QA bridge with automatic parameter extraction, truth-pack-enhanced ranges, and delta feedback from the monitor.

**Research support:**
- Monitoring Architecture §3.6, Autonomy Research §5.2

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/qa_diagnosis_bridge.py` | Add `extract_all_modifiable_params()` | ~90 |
| `tools/qa_diagnosis_bridge.py` | Add truth-pack-enhanced ranges | ~45 |
| `tools/qa_diagnosis_bridge.py` | Add monitor delta feedback | ~45 |

**Tests:**
1. `extract_all_modifiable_params()` finds all `domain_settings.X = Y` assignments
2. Truth-pack ranges appear: `resolution_max = 64 [range: 1-10000, default: 32]`
3. Monitor delta: parameter changed 3+ times → `OSCILLATING. Bound: [35, 65]`
4. Output format < 200 tokens for a 700-line script with 15 params
5. Regression: existing keyword matching still works alongside new extraction

**Rollback:** New methods are additive — existing keyword matching remains as fallback.

**Estimated effort:** ~180 lines | **Dependencies:** 2A-4, 2A-5 | **Risk:** Low

---

### 2C-6: Knowledge Distillation from Successful Scripts

**What:** After a run scores >= 60, extract key code sections (lighting setup, material definitions, camera placement, physics config) as named patterns with metadata.

**Research support:**
- Autonomy Research §1.4, Mission Statement §7

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/knowledge_distillation_tools.py` | Add `extract_code_patterns_from_script()` | ~150 |
| `orchestrator.py` (after quality pass) | Trigger distillation on passing scripts | ~30 |
| `tools/code_pattern_tools.py` | Enhanced `store_pattern()` with section type, effect type, quality score | ~45 |

**Pattern sections to extract:**
- Lighting setup (light types, energy values, positions, colors)
- Material definitions (shader node trees, principled BSDF settings)
- Camera placement (position, rotation, focal length, DOF)
- Physics configuration (domain settings, flow settings, bake settings)
- Scene setup (renderer, frame range, resolution, denoising)

**Tests:**
1. Identifies lighting section in test fire script (Area Light + energy setting)
2. Extracted pattern includes metadata: `{effect_type: "fire", section: "lighting", quality_score: 65}`
3. Stored pattern enters KB with trust level "Emerging"
4. Distillation does NOT trigger for scripts scoring < 60
5. Effect-type scoping: fire pattern stored with `effect_type="fire"`

**Rollback:** Distillation is post-pipeline. Disabling has zero impact on pipeline behavior.

**Estimated effort:** ~225 lines | **Dependencies:** 2B-6, 2B-7 | **Risk:** Low

---

### 2C-7: Streaming for Monitoring (`run_streamed()`) — Nice to Have

**What:** Use `Runner.run_streamed()` for the Script Writer to provide real-time progress visibility and enable mid-stream hallucination detection.

**Why deprioritized:** Streaming is a UX improvement, not a reliability or capability improvement. The truth pack validator catches hallucinations post-generation at $0. Core 2C items deliver direct capability improvements.

**Research support:**
- SDK Analysis §7

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py:_run_agent()` | Add `_run_agent_streamed()` variant | ~60 |
| New: `utils/stream_monitor.py` | Event consumer: log progress, detect hallucination patterns | ~120 |
| `orchestrator.py` (Phase 1) | Use streamed variant for Script Writer | ~15 |

**Tests:**
1. Streamed output identical to non-streamed (determinism check)
2. Stream monitor detects `resolution_divisions` mid-generation
3. Progress events logged without impacting speed
4. Fallback to non-streamed if streaming errors

**Estimated effort:** ~195 lines | **Dependencies:** None | **Risk:** Low-Medium

---

### Removed: Agent Specialization (`Agent.clone()`)

**Moved to Future Ideas.** The SDK Analysis itself notes: "dynamic instructions already solve this problem adequately. Clone is cleaner but not urgent." This was 70 lines that adds no new capability. Dynamic instructions (already implemented) achieve the same effect-type specialization. Can be revisited if dynamic instructions prove insufficient after Phase 2C ships.

---

### Phase 2C Summary

| Item | Lines | Impact | Status |
|------|-------|--------|--------|
| Prerequisite: Orchestrator decomposition | ~200 | Gate — see dedicated section | Prerequisite |
| 2C-1: UCB1 technique selector | ~225 | High — optimal exploration/exploitation | Core |
| 2C-2: Multi-physics truth pack | ~195 | High — enables rigid body, particles, cloth | Core |
| 2C-3: Micro-experiment sandbox | ~420 | High — practical technique validation | Core |
| 2C-4: Graceful budget degradation | ~150 | Medium — extends budget runway | Core |
| 2C-5: Enhanced QA bridge | ~180 | Medium — better diagnosis quality | Core |
| 2C-6: Knowledge distillation | ~225 | Medium — learns from successes | Core |
| 2C-7: run_streamed() monitoring | ~195 | Low — UX improvement only | Nice to Have |
| **Core total (incl. decomposition)** | **~1,595** | | |
| **With nice-to-have** | **~1,790** | | |

**Note:** Line estimates include a 1.5x multiplier based on Phase 1 experience.

**Phase 2C success criteria:**
1. Novel prompts (rigid body, particles) produce evaluable renders on first attempt
2. UCB1 selects different techniques for different effect types across 10 runs
3. Micro-experiments validate techniques in <30 seconds before full runs
4. KB accumulates effect-type-scoped patterns from successful runs (at least 5 patterns after 20 passing runs)
5. Budget-exhausted runs still produce deterministic quality scores
6. Orchestrator decomposition passes E2E validation before any 2C work begins

---

## Phase 2D: Full Autonomy (Directional — Design After 2C)

**Goal:** Self-improvement, autonomy progression tracking, novel prompt handling.
**Risk:** High — these are experimental capabilities.
**Principles served:** P2 (LLM creativity), P6 (learning signal), P7 (earn autonomy).

**Status:** Items 2D-1 and 2D-2 have clear scope and can be estimated. Items 2D-3, 2D-4, and 2D-5 are directional goals — their exact design depends on what Phase 2A-C reveals. They will be designed after Phase 2C ships.

### 2D-1: Autonomy Progression Tracking

**What:** Implement the 5-level autonomy system from the Mission Statement. Track pass rates per effect type. Automatically relax HITL checkpoints as reliability improves.

**Research support:**
- Mission Statement §10: "Autonomy is earned by evidence, not time."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/autonomy_tracker.py` | `AutonomyLevel` enum, per-effect-type tracking, level evaluation | ~180 |
| `orchestrator.py` | Check autonomy level at HITL checkpoints | ~45 |
| `session_manager.py` | Add autonomy metrics to session persistence | ~30 |

**Level criteria:**

| Level | Requirement | HITL Changes |
|-------|-------------|-------------|
| 0: Guided | Default | All checkpoints active |
| 1: Assisted | 10+ runs, some passes | Skip prompt approval for known types |
| 2: Semi-Autonomous | >50% pass rate, 3+ effect types | Only budget + critical issue checks |
| 3: Autonomous (Known) | >70% pass rate, 50+ runs for this type | Human sees only final result |
| 4: Autonomous (Novel) | Stable L3 + external review | Self-directed exploration |

**Tests:**
1. New effect type starts at Level 0 with all HITL checkpoints active
2. After 10 runs with 3 passes, promotes to Level 1
3. 10 runs with 0 passes stays at Level 0
4. Per-effect-type: fire at Level 2, liquid at Level 0, independent progression
5. Level demotion: pass rate drops below threshold → level decreases by 1

**Rollback:** Set `ENABLE_AUTONOMY_PROGRESSION = False` in `config/agent_config.py`.

**Estimated effort:** ~255 lines | **Dependencies:** 2B-5 (HITL framework) | **Risk:** Medium

---

### 2D-2: Cross-Session Learning Transfer

**What:** Patterns learned in one session automatically enhance future sessions. Makes dynamic instructions systematic with quality gates.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/dynamic_instructions.py` | Load top-N trusted patterns by retention score for current effect type | ~60 |
| `tools/experiment_tracker_tools.py` | Add cross-session analytics | ~90 |

**Tests:**
1. Pattern from session A (fire, score 70, lighting) appears in session B's instructions for fire
2. Pattern with retention < 0.3 NOT injected into new sessions
3. Effect-type gating: fire pattern NOT in liquid session
4. Top-N limit: 50 trusted patterns → only top 5 injected
5. Cross-session analytics correctly reports most-reused pattern

**Rollback:** Set `ENABLE_CROSS_SESSION_PATTERNS = False` in `config/agent_config.py`.

**Estimated effort:** ~150 lines | **Dependencies:** 2B-6, 2B-7, 2C-6 | **Risk:** Low

---

### 2D-3: Prompt Versioning for Script Writer (Directional)

**What:** Track which system prompt variants produce better quality scores. Auto-promote the best-performing variant.

**Research support:**
- Autonomy Research §1.1 (OpenAI cookbook)

**Design direction:** Will be informed by: which instruction components have the most variance impact (discovered during 2A-2C), whether quality variance is dominated by prompt vs technique vs scene complexity, and the `AdvancedSQLiteSession` branching capability (2B-4).

**This item will be designed after Phase 2C ships. No line estimate or timeline assigned.**

---

### 2D-4: Adaptive Replanning (Magentic-One Style) (Directional)

**What:** Replace the fixed state machine with an adaptive pipeline that can restart from research when evaluation reveals the technique is fundamentally wrong.

**Research support:**
- Autonomy Research §2.3 (Magentic-One)

**Design direction:** Key questions to resolve after 2C: Does PipelineMonitor (2A-4) already catch the cases? Does UCB1 (2C-1) + micro-experiments (2C-3) make technique-level restarts rare enough? What's the cost/benefit vs simply escalating escape velocity?

**This item will be designed after Phase 2C ships. No line estimate or timeline assigned.**

---

### 2D-5: LLM-Based Anomaly Detection (Full MASC) (Directional)

**What:** Train a lightweight anomaly detector on normal pipeline trajectories. When anomaly score exceeds threshold, a correction agent intervenes.

**Research support:**
- Autonomy Research §2.4 (MASC): "77.84% AUC-ROC on step-level error detection"

**Design direction:** The deterministic PipelineMonitor (2A-4) handles an estimated 95% of detectable failure modes. Full MASC only becomes valuable if the deterministic monitor misses failure modes, there are 50+ runs for training data, and the cost ($0.01-0.02/iteration) is justified.

**This item will be designed after Phase 2C ships. No line estimate or timeline assigned.**

---

### Phase 2D Summary

| Item | Lines | Impact | Status |
|------|-------|--------|--------|
| 2D-1: Autonomy progression | ~255 | Medium — earned independence | Concrete — ready to implement |
| 2D-2: Cross-session learning | ~150 | Medium — cumulative improvement | Concrete — ready to implement |
| 2D-3: Prompt versioning | TBD | Medium — instruction optimization | Directional — design after 2C |
| 2D-4: Adaptive replanning | TBD | High — flexible pipeline | Directional — design after 2C |
| 2D-5: Full MASC | TBD | Medium — only if needed | Directional — design after 2C |
| **Concrete total** | **~405** | | |

**Phase 2D success criteria:**
1. Autonomy level progression demonstrated: at least one effect type reaches Level 2
2. System demonstrably improves over time (same prompt scores higher after 50 runs vs 5)
3. Cross-session learning injects at least 3 trusted patterns into a new session

**Deferred success criteria (for directional items, to be refined after 2C):**
4. Prompt versioning shows measurable quality improvement (if implemented)
5. Adaptive replanning reduces technique-level failures by >30% (if implemented)
6. Anomaly detection catches failures the deterministic monitor misses (if implemented)

---

## Dependency Graph

```
Phase 1 (DONE)
│
├── 2A-0: Documentation Pipeline ──────────────────────────────────────────┐
│   └── KB Seeding (after 2A-0) ───────────────────────────────────────────┤
├── 2A-1: is_enabled ──────────────────────────────────────────────────────┤
├── 2A-2: tool_use_behavior ───────────────────────────────────────────────┤
├── 2A-3: Tool guardrails ─────────────────────────────────────────────────┤
├── 2A-4: PipelineMonitor ──────────┬──────────────────────────────────────┤
├── 2A-5: Parameter bounds ─────────┤ (depends on 2A-4)                   │
├── 2A-6: Deprecate Spec-First ─────┤                                     │
├── 2A-7: Remove Executor agent ────┤                                     │
├── 2A-8: Tool timeouts ────────────┘                                     │
│                                                                          │
│   ┌──── 2A-4 + 2A-5 ────┐                                              │
│   │                       │                                              │
│   ▼                       ▼                                              │
├── 2B-1: Ralph iterations (ISOLATED — Week 2) ─────┐                    │
├── 2B-2: Context trimming ─────────────────────────┤ (depends on 2B-1)  │
├── 2B-3: Multi-grader eval ────────────────────────┤                    │
├── 2B-4: AdvancedSQLiteSession ────────────────────┤ (parallel w/ 2B-1) │
├── 2B-5: HITL framework ─────────────────────────┤ (depends on 2A-4)  │
├── 2B-6: Memory decay ───────────────────────────┤                    │
├── 2B-7: Effect-type evidence ────────────────────┤ (depends on 2B-6)  │
├── 2B-8: Artifact-based sharing ──────────────────┘ (depends on 2A-4)  │
│                                                                          │
│   === ORCHESTRATOR DECOMPOSITION GATE ===                                │
│   Requires: 2B-1 shipped + E2E passing                                  │
│                                                                          │
├── 2C-1: UCB1 technique ──────────┐ (depends on 2B-7; soft: 2A-0)      │
├── 2C-2: Multi-physics truth pack ┤                                     │
├── 2C-3: Micro-experiments ───────┤ (depends on 2C-1 + 2C-2; soft: 2A-0)│
├── 2C-4: Budget degradation ──────┤ (depends on 2B-3)                   │
├── 2C-5: Enhanced QA bridge ──────┤ (depends on 2A-4 + 2A-5)           │
├── 2C-6: Knowledge distillation ──┤ (depends on 2B-6 + 2B-7)           │
├── 2C-7: run_streamed() ──────────┘ (nice-to-have, no hard deps)       │
│                                                                          │
├── 2D-1: Autonomy tracking ───────┐ (depends on 2B-5)                   │
├── 2D-2: Cross-session learning ──┤ (depends on 2B-6 + 2B-7 + 2C-6)   │
├── 2D-3: Prompt versioning ───────┤ (directional — design after 2C)     │
├── 2D-4: Adaptive replanning ─────┤ (directional — design after 2C)     │
└── 2D-5: Full MASC ───────────────┘ (directional — design after 2C)     │
```

**Parallelism opportunities:**
- 2A: Items 1-3 in parallel. Item 4 (monitor) parallel with 1-3. Items 5-8 parallel.
- 2B: Items 1 and 4 parallel. Items 6 and 7 sequential. Item 8 depends on 2A-4.
- 2C: Items 1 and 2 parallel. Items 4 and 5 parallel. Item 6 depends on 2B-6+7.

---

## Conflict Resolutions

Where different research agents recommended different approaches, these decisions were made:

### 1. Context Management: Ralph vs AdvancedSQLiteSession Branching

**Conflict:** Ralph (fresh context per iteration) vs AdvancedSQLiteSession (branching within a conversation).

**Resolution:** Use BOTH — they're complementary. Ralph for iteration-level freshness (2B-1): each iteration is a new `Runner.run()` with clean context. AdvancedSQLiteSession for session-level tracking (2B-4): token counting, technique branching, persistence. The session wraps the iterations; iterations don't accumulate within the session.

### 2. Monitoring: RunHooks vs PipelineMonitor vs Full MASC

**Conflict:** Three monitoring approaches at different scopes.

**Resolution:** Three layers:
- **RunHooks (existing):** Intra-agent loop detection, turn budgets, tool counting. No changes needed.
- **PipelineMonitor (2A-4):** Inter-iteration monitoring — oscillation, cascades, budget. Deterministic, $0.
- **Full MASC (2D-5):** Only if deterministic monitoring proves insufficient. Deferred.

### 3. Spec-First Pipeline: Deprecate or Keep as Safety Net?

**Conflict:** Codebase Analysis recommends deprecation. Code Writer guardrail depends on APISpec models.

**Resolution:** Deprecate the LLM-powered Spec-First pipeline (2A-6) but preserve the Pydantic models. `truth_pack_to_api_spec()` populates APISpec from truth pack data. Net effect: same validation, $0 cost.

### 4. Orchestrator Monolith: When to Decompose?

**Conflict:** Codebase Analysis §7.1 recommends extraction. Ralph (2B-1) will change the iteration loop.

**Resolution:** Decompose after 2B-1 (Ralph) ships and passes E2E, before any 2C work. See **Orchestrator Decomposition Gate** section. This is now a mandatory gate with concrete criteria.

### 5. Quality Evaluation: Who Decides the Score?

**Resolution:** The 3-tier approach (2B-3):
- Tier 1 (deterministic): Hard gates. Critical issues = auto-fail. $0.
- Tier 2 (ML metrics): 60% weight. Objective, reproducible. $0.
- Tier 3 (LLM vision): 40% weight. Subjective but valuable. $0.01-0.05.
- Short-circuit: critical issue in Tier 1 → skip Tiers 2-3.

### 6. Agent Specialization: Agent.clone() vs Dynamic Instructions

**Conflict:** SDK Analysis recommends `Agent.clone()`. Dynamic instructions already achieve the same goal.

**Resolution:** Cut Agent.clone() (originally 2C-4). Dynamic instructions are already implemented and working. If they prove insufficient after Phase 2C, Agent.clone() can be revisited.

---

## Seven Pillars Coverage

### SDK Integration
- 2A-1: `is_enabled` | 2A-2: `tool_use_behavior` | 2A-3: Tool guardrails | 2A-8: Tool timeouts
- 2B-2: `call_model_input_filter` | 2B-4: `AdvancedSQLiteSession` | 2B-5: `needs_approval`
- 2C-7: `run_streamed()` (nice-to-have)

### Self-Learning & Documentation
- **2A-0: Documentation pipeline** (enables technique diversity — the foundation of self-learning)
- **KB Seeding** (bootstraps knowledge base from rewritten manual)
- 2B-6: Ebbinghaus memory decay | 2B-7: Effect-type evidence gating
- 2C-6: Knowledge distillation | 2D-2: Cross-session learning | 2D-3: Prompt versioning (directional)

### Monitoring & Observability
- 2A-4: PipelineMonitor | 2A-5: Parameter bounds
- 2C-5: Enhanced QA bridge | 2C-7: run_streamed() (nice-to-have)
- 2D-5: Full MASC (directional)

### Context Management
- 2B-1: Ralph stateless iterations | 2B-2: Context trimming | 2B-8: Artifact-based sharing
- 2A-6: Deprecate Spec-First (reduces context)

### Multi-Physics Support
- 2C-1: UCB1 technique selector | 2C-2: Multi-physics truth pack
- 2C-3: Micro-experiments

### Autonomy Progression
- 2D-1: Autonomy tracking (L0-L4) | 2D-4: Adaptive replanning (directional)
- 2B-5: HITL framework (foundation for relaxing checkpoints)

### Experimentation Mode
- 2C-3: Micro-experiment sandbox | 2C-1: UCB1 exploration/exploitation
- 2D-2: Cross-session learning (experiments feed into future runs)

---

## Budget Impact Analysis

**Current cost per 3-iteration run:** ~$0.30-0.70 (15-46 LLM calls)

| Phase | Savings | New Costs | Net Impact |
|-------|---------|-----------|------------|
| **2A** | -$0.10-0.20/run (stop_on_first_tool, deprecate Spec-First, is_enabled) | +$0.19 one-time (manual rewrite) + ~$0.50 one-time (KB seeding) | **-$0.10-0.20/run** |
| **2B** | -$0.05-0.15/run (Ralph reduces context tokens, multi-grader skips vision) | $0 (all deterministic or SDK-native) | **-$0.05-0.15** |
| **2C** | -$0.02-0.05/run (budget degradation) | +$0.01-0.02/run (micro-experiments use Blender, not API) | **-$0.01-0.03** |
| **2D** | -$0.01-0.03/run (prompt versioning finds cheaper prompts) | +$0.01-0.02/run (autonomy tracking) | **~$0** |

**One-time costs:** ~$0.70 total (manual rewrite + KB seeding). Pays for itself in 4-7 runs.

**Projected cost per 3-iteration run after Phase 2B:** $0.15-0.35
**Monthly run capacity at $20 budget:** 57-133 runs (up from 29-67)

---

## Implementation Sequence (Recommended)

```
Week 1: Phase 2A Quick Wins
├── Day 1: 2A-0 (Documentation Pipeline) — highest ROI, start immediately
│          Run manual rewrite ($0.19), upload to vector store, validate
├── Day 1-2: 2A-1 (is_enabled) + 2A-2 (tool_use_behavior) + 2A-8 (timeouts)
│            [All trivial, independent, ship together]
├── Day 2-3: 2A-3 (tool guardrails)
│            [Moderate effort, depends on truth pack]
├── Day 3-4: 2A-6 (deprecate Spec-First) + 2A-7 (remove Executor)
│            [Independent cleanup]
├── Day 4-7: 2A-4 (PipelineMonitor) + 2A-5 (parameter bounds)
│            [Most complex 2A items]
└── Post-2A-0: KB Seeding — run research agent against 6 effect types (~$0.50)

Week 2: Phase 2B — Ralph (ISOLATED)
├── Day 1-2: IterationState dataclass + serialization
├── Day 2-3: Convert iteration 2 only to stateless
├── Day 3-4: E2E A/B comparison: old vs new iteration 2
├── Day 4-5: If quality holds, convert all iterations
└── E2E validation: 3-iteration fire run with full monitoring

Week 3: Phase 2B — Remaining Items
├── 2B-4 (AdvancedSQLiteSession) — can start parallel with Week 2
├── 2B-6 (memory decay) — independent
├── 2B-2 (context trimming) — depends on stable 2B-1
├── 2B-3 (multi-grader eval) — independent
├── 2B-7 (effect-type evidence) — depends on 2B-6
└── 2B-8 (artifact sharing) — depends on 2A-4

Week 4: Phase 2B — HITL + Validation
├── 2B-5 (HITL framework) — depends on 2A-4
└── E2E validation of Phase 2A+2B together

Week 5: Orchestrator Decomposition Gate
├── Decompose create_asset_pipeline() into phase modules
├── E2E validation: identical behavior to monolith
└── All existing tests pass

Weeks 6-9: Phase 2C Advanced Capabilities
├── 2C-1 + 2C-2 [parallel: UCB1 + multi-physics truth pack]
├── 2C-3 (micro-experiments) [depends on 2C-1 + 2C-2]
├── 2C-4 + 2C-5 [parallel: budget degradation + enhanced QA bridge]
├── 2C-6 (knowledge distillation) [depends on 2B-6 + 2B-7]
├── 2C-7 (streaming) [nice-to-have, if time permits]
└── E2E validation: novel prompt produces evaluable render

Week 10+: Phase 2D Full Autonomy
├── 2D-1 (autonomy tracking) [depends on 2B-5]
├── 2D-2 (cross-session learning) [depends on 2C-6]
└── 2D-3, 2D-4, 2D-5 [directional — design after 2C ships]
```

---

## End Goals (From Mission Statement §14)

| Goal | Phase That Delivers It | How We Measure |
|------|----------------------|----------------|
| Every run produces a render | Phase 1 (done) + 2A (monitoring catches remaining failures) | 95%+ completion rate across 20 runs |
| Known effects score >= 60 | Phase 2B (Ralph + multi-grader + context management) | 50%+ pass rate for fire, liquid, smoke |
| Novel prompts produce reasonable first attempts | Phase 2C (multi-physics + UCB1 + micro-experiments) | First attempt scores >= 30 for untried effect types |
| System demonstrably improves over time | Phase 2C-2D (distillation + cross-session learning + prompt versioning) | Same prompt scores higher after 50 runs vs 5 |
| Failed runs produce useful diagnostics | Phase 2A (PipelineMonitor) + 2B (multi-grader) | Every failed run produces structured diagnostic artifact |
| Budget respected | Phase 2A (is_enabled + stop_on_first_tool) + 2C (degradation) | Cost per 3-iteration run < $0.35 |
| Technique diversity breaks monotony | Phase 2A-0 (documentation pipeline) + KB Seeding + 2C-1 (UCB1) | Research agent discovers 3+ techniques per effect type |

---

## Appendix A: New File Inventory

| File | Phase | Lines (est) | Purpose |
|------|-------|-------------|---------|
| `scripts/upload_rewritten_manual.py` | 2A | ~120 | Upload rewritten manual to vector store |
| `tests/test_technique_discovery.py` | 2A | ~30 | Validate technique discovery from rewritten manual |
| `utils/tool_visibility.py` | 2A | ~60 | is_enabled callback functions |
| `guardrails/tool_guardrails.py` | 2A | ~150 | Truth pack + execution + script length guardrails |
| `tools/pipeline_monitor.py` | 2A | ~375 | Deterministic PipelineMonitor class |
| `tools/parameter_bounds.py` | 2A | ~180 | Per-effect-type bounds + damped convergence |
| `scripts/seed_kb.py` | 2A (post) | ~100 | KB seeding — research agent against 6 effect types |
| `config/agent_config.py` | 2A | ~45 | Feature flags for rollback |
| `utils/iteration_state.py` | 2B | ~180 | Ralph-style stateless iteration state |
| `utils/context_filters.py` | 2B | ~120 | Per-agent call_model_input_filter functions |
| `tools/deterministic_quality_checks.py` | 2B | ~270 | Tier 1 quality checks (histogram, lights, camera) |
| `utils/hitl_handler.py` | 2B | ~150 | HITL approval/resume logic |
| `tools/artifact_tools.py` | 2B | ~45 | read_artifact tool for agents |
| `utils/technique_selector.py` | 2C | ~120 | UCB1 algorithm |
| `tools/micro_experiments.py` | 2C | ~300 | Sandbox experiment runner |
| `utils/stream_monitor.py` | 2C | ~120 | Streaming event consumer (nice-to-have) |
| `utils/autonomy_tracker.py` | 2D | ~180 | L0-L4 autonomy progression |

**Total new files: 17 | Total new lines: ~2,545**

## Appendix B: Modified File Inventory

| File | Phases | Total Changes (est) |
|------|--------|-------------------|
| `orchestrator.py` | 2A, 2B, 2C, 2D | ~1,100 lines across all phases (then decomposed in Week 5) |
| `session_manager.py` | 2B, 2D | ~83 lines |
| `tools/asset_evaluator_tools.py` | 2A, 2B, 2C | ~143 lines |
| `tools/dynamic_instructions.py` | 2B, 2D | ~98 lines |
| `tools/experiment_tracker_tools.py` | 2A (seeding), 2B, 2C | ~210 lines |
| `tools/blender_executor_tools.py` | 2A | ~55 lines |
| `tools/semantic_docs_tools.py` | 2A | ~15 lines |
| `tools/qa_diagnosis_bridge.py` | 2C | ~180 lines |
| `tools/truth_pack.py` | 2C | ~105 lines |
| `tools/knowledge_distillation_tools.py` | 2B, 2C | ~165 lines |
| `tools/code_pattern_tools.py` | 2B, 2C | ~68 lines |
| `utils/code_pattern_memory.py` | 2B | ~45 lines |
| `tools/script_generator_tools.py` | 2A | ~8 lines |
| `specialized_agents/quality_analyst.py` | 2A | ~8 lines |
| `specialized_agents/docs_expert.py` | 2A | ~8 lines |
| `specialized_agents/executor.py` | 2A | ~8 lines (deprecate) |
| `specialized_agents/api_spec_agent.py` | 2A | ~8 lines (deprecate) |
| `specialized_agents/api_validator.py` | 2A | ~1 line |
| `specialized_agents/learning_agent.py` | 2A | ~1 line |
| `scripts/experiment_manual_rewrite.py` | 2A | ~75 lines |

**Total modified lines: ~2,429**

## Appendix C: Source Cross-References

Every recommendation traces back to at least one research document:

| Pattern/Feature | SDK Analysis | Codebase Analysis | Autonomy Research | Monitoring Arch. | Critique / Mission Stmt |
|----------------|:---:|:---:|:---:|:---:|:---:|
| Documentation Pipeline (2A-0) | | | | | Critique §1, MS §4, §12 |
| is_enabled | §4 | §7.9, §7.10 | | | |
| tool_use_behavior | §5 | §6.5 | §3.1 | | |
| Tool guardrails | §6 | §3.3, §7.8 | | | |
| PipelineMonitor | | §7.6 | §2.4 (MASC) | §1.2, §1.3 | MS §9 |
| Parameter bounds | | | §5.3 | §3.3 | MS §12 |
| Deprecate Spec-First | | §7.4 | | | |
| Remove Executor | | §7.3 | §3.1 | | |
| Function tool timeouts | §B1 | | | | |
| Ralph iterations | | | §2.1 | §2.1 | Critique §3 |
| call_model_input_filter | §2 | §7.5 | §3.2 | §2.2 | |
| Multi-grader eval | | | §5.1 | §3.2 | MS §4 |
| AdvancedSQLiteSession | §1 | §6.1, §7.7 | | | |
| needs_approval | §3 | §6.7 | | | |
| Memory decay | | | §1.3 (SAGE) | | MS §7 |
| Effect-type gating | | | §1.2 | | |
| Artifact sharing | | | §2.2 (Anthropic) | §2.3 | |
| UCB1 technique | | | §4.2 | | MS §8 |
| Multi-physics truth pack | | §8.1 | | | MS §15 |
| Micro-experiments | | | §4.1 | | MS §8 |
| Budget degradation | | §7.10 | | | |
| Enhanced QA bridge | | | §5.2 | §3.6 | |
| Knowledge distillation | | | §1.4 | | MS §7 |
| run_streamed() | §7 | §6.6 | | | |
| Autonomy tracking | | | | | MS §10 |
| Cross-session learning | | | §1.4 | | |
| Prompt versioning | | | §1.1 (OpenAI) | | |
| Adaptive replanning | | | §2.3 (Magentic-One) | | |
| Full MASC | | | §2.4 | | |
| KB Seeding | | | | | Critique §1, MS §8 |
| Orchestrator Decomposition | | §7.1 | | | Critique §6 |
| Agent.clone() (removed) | §8 | | | | Critique: cut |
