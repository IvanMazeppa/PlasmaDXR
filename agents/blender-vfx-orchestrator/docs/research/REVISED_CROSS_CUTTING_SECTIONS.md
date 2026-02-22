# Cross-Cutting Sections for Revised Phase 2+ Roadmap

**Date:** 2026-02-22
**Author:** Cross-cutting Writer (Claude Opus 4.6)
**Purpose:** New sections to be integrated into the revised `PHASE2_ROADMAP.md`. These address gaps identified in the roadmap critique.
**Status:** Draft for integration

---

## 1. Testing Strategy

**Applies to:** All phases (2A through 2D)

### Philosophy

Every work item ships with tests. No exceptions. Phase 1 established the standard: 16 escape velocity tests across 6 test classes in 144 lines, covering escalation paths, technique exclusion, counter resets, step-down logic, and per-level action recommendations. That's the baseline — every Phase 2 work item meets or exceeds it.

Three test categories, in order of priority:

| Category | What It Covers | When Required |
|----------|---------------|---------------|
| **Unit tests** | New classes, functions, and data structures in isolation | Every work item that adds new code |
| **Integration tests** | Pipeline behavior when new code interacts with existing components | Every work item that modifies `orchestrator.py` or agent behavior |
| **Regression tests** | Existing behavior preserved after code removal or replacement | Every work item that deprecates, removes, or replaces existing code |

### Test Location and Naming

All tests live in `tests/`. Naming convention:

```
tests/test_<module_name>.py          # Unit tests for a specific module
tests/test_<module_name>_integration.py  # Integration tests
```

Examples:
- `tests/test_pipeline_monitor.py` — Unit tests for `tools/pipeline_monitor.py`
- `tests/test_tool_guardrails.py` — Unit tests for `guardrails/tool_guardrails.py`
- `tests/test_ralph_iterations_integration.py` — Integration tests for stateless iteration loop

### Work Item Test Template

Each work item in the roadmap specifies 3-5 targeted tests inline. Format:

```
**Tests:**
1. <What is being tested> — <assertion>
2. <What is being tested> — <assertion>
3. <What is being tested> — <assertion>
[Optional: 4-5 for complex items]
```

**Example — 2A-4 (PipelineMonitor):**

```
**Tests:**
1. Oscillation detection: feed values [50, 2500, 10] → assert is_oscillating() == True
2. No false positive: feed values [50, 100, 150] → assert is_oscillating() == False
3. Cascade detection: feed score drop >5 after QA suggestion → assert warning generated
4. Budget alert: set budget to $0.50, feed $0.60 spent → assert CRITICAL alert
5. Integration: mock iteration loop, verify monitor produces status artifact file
```

**Example — 2A-6 (Deprecate Spec-First):**

```
**Tests:**
1. Script Writer receives truth pack context when Spec-First pipeline is removed
2. Code Writer guardrail still works with truth_pack_to_api_spec() data
3. Technique selection runs standalone (was previously parallel with API Spec)
```

### E2E Validation Gates Between Phases

Each phase boundary requires an E2E validation run before proceeding:

| Gate | Trigger | Validation |
|------|---------|------------|
| **Phase 1 → 2A** | Phase 1 merged (DONE) | 3 effect types produce evaluable renders. Verified via trace `63eb4ef1`. |
| **Phase 2A → 2B** | All 2A items pass unit + integration tests | Run 5 iterations on fire + liquid. Verify: no parameter oscillation (monitor catches), budget < $0.30/run, zero hallucinated attributes reach Blender. |
| **Phase 2B → Decomposition Gate** | 2B-1 (Ralph) passes E2E | Run 5 iterations with Ralph. Compare iteration 3-4 quality against old (accumulated context) baseline. Quality must not regress. |
| **Decomposition Gate → 2C** | Orchestrator decomposed into phase modules | All existing tests pass against decomposed code. No new functionality — pure refactor. |
| **Phase 2C → 2D** | 2C items pass tests + E2E | Novel prompt (rigid body) produces evaluable render. UCB1 selects different techniques for different effect types. Micro-experiment completes in <30 seconds. |

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

## 2. Rollback Strategy

**Applies to:** All phases (2A through 2D)

### Feature Flags

Every major behavioral change ships behind a feature flag. Flags live in `config/agent_config.py` with a consistent naming convention: `ENABLE_<FEATURE_NAME>`. All flags default to `True` (new behavior active). Setting a flag to `False` reverts to the previous behavior. Restart the pipeline after changing a flag.

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
ENABLE_BUDGET_DEGRADATION = True    # 2C-6: Set False → binary budget check
```

**Implementation cost:** ~3 lines per feature (flag check + conditional import/call). Total across all phases: ~45 lines.

**Usage pattern in code:**

```python
from config.agent_config import ENABLE_PIPELINE_MONITOR

# In orchestrator.py iteration loop
if ENABLE_PIPELINE_MONITOR:
    monitor_result = monitor.check_after_evaluation(session_state)
    if monitor_result.should_intervene:
        handle_monitor_alerts(monitor_result)
```

### Git Branch Strategy

Each sub-phase is developed on a branch that can be reverted independently:

| Branch | Base | Content | Revert Method |
|--------|------|---------|---------------|
| `0.33.2/phase2a-quick-wins` | `0.33.2/phase1-reliability` | All 2A items | `git revert --no-commit <merge-commit>` |
| `0.33.2/phase2b-ralph` | `phase2a-quick-wins` | 2B-1 ONLY (isolated) | Revert single merge commit |
| `0.33.2/phase2b-core` | `phase2b-ralph` | Remaining 2B items | Revert single merge commit |
| `0.33.2/decomposition` | `phase2b-core` | Orchestrator refactor | Revert single merge commit |
| `0.33.2/phase2c-advanced` | `decomposition` | All 2C items | Revert single merge commit |
| `0.33.2/phase2d-autonomy` | `phase2c-advanced` | All 2D items | Revert single merge commit |

**Critical: Ralph (2B-1) gets its own branch.** It's the highest-risk change in the entire roadmap — restructuring the core iteration loop. Isolating it allows targeted revert without touching other 2B work.

### Rollback Decision Criteria

Roll back a feature when ANY of these occur:

| Signal | Threshold | Action |
|--------|-----------|--------|
| E2E pass rate drops | Below previous phase's baseline | Disable flag, investigate |
| Cost per run increases | >50% above projected savings | Disable flag, investigate |
| New crash type introduced | Any crash not seen before the change | Disable flag, investigate |
| Quality scores regress | Average score drops >10 points across 5 runs | Disable flag, investigate |

---

## 3. Orchestrator Decomposition Gate

**Position in roadmap:** Between Phase 2B and Phase 2C
**Type:** Refactoring gate — no new functionality

### Trigger

After 2B-1 (Ralph Stateless Iterations) ships and passes E2E validation, decompose `create_asset_pipeline()` before starting ANY Phase 2C work.

### Why Now (Not Earlier, Not Later)

**Not earlier:** Ralph (2B-1) restructures the iteration loop significantly (~400 lines changed in `orchestrator.py`). Decomposing before Ralph would mean decomposing twice — once for the current structure, then again after Ralph changes it.

**Not later:** Without decomposition, Phase 2C adds UCB1 technique selection, micro-experiments, multi-physics support, streaming, and budget degradation — all of which touch `orchestrator.py`. The file is currently ~4,544 LOC. Phase 2C could push it past 6,000 LOC, making future changes increasingly fragile and error-prone.

**Now is the window:** After Ralph stabilizes the iteration loop structure, before 2C adds complexity on top.

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

### Success Criteria

1. All existing tests pass without modification
2. E2E run produces identical results (same trace structure, same agent calls)
3. No file exceeds 700 LOC
4. Each phase module has a single public entry point (e.g., `async def run_research_phase(state, agents)`)
5. `orchestrator.py` contains ONLY coordination logic — no phase implementation details

### Estimated Effort

~0 new lines (pure refactoring). Approximately 3,500 lines moved from `orchestrator.py` into phase modules with minimal interface changes. Estimated time: 2-3 days.

### Risk

Low. This is mechanical refactoring — extracting existing functions into modules with clean interfaces. The Ralph iteration structure (2B-1) provides natural decomposition boundaries since each iteration already has discrete phases.

---

## 4. KB Seeding Strategy

**Position in roadmap:** After 2A-0 (Manual Rewrite) completes
**Principle served:** P6 (every run produces learning signal) — but for the KB, we need initial signal to learn FROM.

### Context

Phase 1 wiped the KB (18 stale entries removed). The manual rewrite (2A-0) produces LLM-optimized Blender documentation. Between these two events, the KB is empty and the documentation is fresh. This is the optimal moment to bootstrap the KB with high-quality technique knowledge.

Without seeding, the system starts every run from zero knowledge — the research agent discovers techniques from scratch each time, paying API costs for information that should already be in the KB. UCB1 (2C-1) has no arms to select. Micro-experiments (2C-3) have no technique candidates. Dynamic instructions inject nothing.

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
    - trust_level: "emerging" (one research occurrence, not yet production-validated)
    - source: "manual_seed_2026-02-XX"
    - techniques: list of discovered techniques with brief descriptions
        │
        ▼
Validate seeded KB:
  - Query KB for each effect type → verify non-empty results
  - Verify technique diversity: >= 2 distinct techniques per effect type
  - Verify no hallucinated attributes in technique descriptions (truth pack cross-check)
        │
        ▼
KB ready for production runs
```

### Cost

| Item | Cost |
|------|------|
| 6 research agent runs (gpt-5-mini) | ~$0.30-0.60 |
| Manual rewrite processing (already done in 2A-0) | $0.19 |
| **Total seeding cost** | **~$0.50-0.80** |

This is a one-time cost that prevents spending $0.05-0.10 per run on redundant technique discovery.

### What Gets Seeded vs What Gets Earned

| Seeded (bootstrap) | Earned (production evidence) |
|--------------------|-----------------------------|
| Technique names and descriptions | Quality scores per technique |
| Parameter relationships from manual | Optimal parameter values |
| Known gotchas and warnings | Specific code patterns that work |
| Alternative approaches per effect type | Evidence-gated trust levels |

Seeded entries start at "emerging" trust level. They can be used by the research agent and technique selector, but they are NOT injected into the Script Writer's dynamic instructions until they earn "trusted" status through 3+ successful production uses.

### Implementation

| File | Change | Lines |
|------|--------|-------|
| New: `scripts/seed_kb.py` | Run research agent against 6 effect types, store findings in KB | ~100 |
| `tools/experiment_tracker_tools.py` | Add `seed_technique_entry()` — creates KB entries with "manual_seed" source tag | ~30 |

**Estimated effort:** ~130 lines + ~$0.50-0.80 API cost

---

## 5. Doc Search Fix Acknowledgment

**Position in roadmap:** Addition to the "Foundation: What Phase 1 Delivered" section

### What Was Found and Fixed

During the Phase 2 research session (2026-02-22), three bugs were discovered in `semantic_docs_tools.py` that explain why the research agent could not discover new techniques — the root cause of "technique monotony" (Mission Statement Section 12).

| Bug | Impact | Fix |
|-----|--------|-----|
| **API results filled shared quota** before manual queries ran | Manual/tutorial content completely blocked — agent could not read conceptual docs | Separate quotas: `manual_quota = max(max_results // 2, 3)`, `api_quota = max(max_results // 2, 3)` |
| **Manual queries not explicitly routed** to manual vector store | Queries hit API store by default, missing all conceptual content | Explicit intent routing: `intent="manual"` for conceptual queries, `intent="api"` for attribute lookups |
| **No technique discovery queries** | Agent never asked "what alternatives exist?" — only searched for specific attributes | Added technique discovery queries: "how to create X effect", "alternative methods techniques for X", "physics simulation types Blender" |

### Additional Fix: Result Interleaving

Results are now interleaved: manual results first (conceptual understanding, techniques, workflows), then API results (attribute names, types, ranges). This ensures agents see the "how/why/when" before the "what exists."

### Why This Matters for Phase 2

Without this fix, Phase 2A-0 (manual rewrite) would be useless — rewritten content would still be blocked by API results filling the quota. The fix is the prerequisite that makes the entire Research → Learning pipeline functional.

### Modified Files

| File | Changes |
|------|---------|
| `tools/semantic_docs_tools.py` | Separate quotas, intent routing, technique discovery queries, interleaved results |

### Updated Foundation Table Entry

Add to the "Foundation: What Phase 1 Delivered" table:

| Deliverable | File | Impact |
|------------|------|--------|
| Doc Search Fix (quotas + routing + technique discovery) | `tools/semantic_docs_tools.py` | Unblocks technique diversity — research agent can now discover manual content |

---

## 6. Research → Learning Pipeline Diagram

**Applies to:** All phases — this diagram shows the end-to-end flow that Phase 2 builds and optimizes.

### The Complete Pipeline

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
│                                                    │
│  Needs candidates → KB + research provide them     │
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
│  Knowledge Distillation (2C-8):                    │
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
│    - Prompt version (2D-3)                          │
└─────────────────────┬──────────────────────────────┘
                      │
                      │  feeds back to
                      │
                      └──────────────► RESEARCH AGENT (top of diagram)
```

### Why 2A-0 (Manual Rewrite) Is Critical

Every box in this diagram exists in the codebase or the roadmap. But the top-left box — "BLENDER MANUAL (LLM-optimized rewrite)" — is missing from the current manual vector store. Without it:

1. The **Research Agent** can only echo its training data — the same 3-4 techniques for every prompt
2. **Technique Selection** (UCB1) has only 3-4 arms to pull — not enough for meaningful exploration
3. **KB Seeding** discovers nothing new — the research agent can't find techniques it doesn't already know
4. **Knowledge Distillation** accumulates patterns for 3-4 techniques only — the system "learns" the same approaches
5. **Dynamic Instructions** inject the same patterns into every run — no diversity

The manual rewrite breaks this ceiling. It gives the research agent access to the full breadth of Blender's capabilities in a format it can actually parse — actionable technique knowledge with explicit TECHNIQUE sections, parameter relationships, and gotchas. Cost: $0.19 for all physics pages. Impact: enables the entire self-learning pipeline to produce value.

### The Self-Learning Feedback Loop

The pipeline is circular by design. Each production run:

1. Generates a script using knowledge from the KB (dynamic instructions)
2. Evaluates the result (multi-grader pipeline)
3. Records the outcome (learning agent)
4. Updates the KB (evidence gating promotes or demotes patterns)
5. Future runs benefit from updated KB (dynamic instructions inject new patterns)

This loop only produces value when there is **technique diversity** to learn from. A system that tries the same 3 approaches over and over "learns" nothing — it memorizes. The manual rewrite (content diversity) + KB seeding (initial population) + UCB1 (exploration incentive) + micro-experiments (practical testing) together create a system that genuinely expands its repertoire over time.

### Roadmap Items by Pipeline Stage

| Pipeline Stage | Phase 1 (Done) | Phase 2A | Phase 2B | Phase 2C | Phase 2D |
|---------------|----------------|----------|----------|----------|----------|
| Documentation | Truth pack, doc search fix | Manual rewrite (2A-0) | | | |
| Research | | KB seeding | | | |
| Technique Selection | | | | UCB1 (2C-1) | Adaptive replanning (2D-4) |
| Script Generation | Truth pack validation | Tool guardrails (2A-3) | Context trimming (2B-2), artifact sharing (2B-8) | Agent.clone (2C-4), streaming (2C-5) | Prompt versioning (2D-3) |
| Execution | | Deprecate Spec-First (2A-6), remove Executor (2A-7), timeouts (2A-8) | | | |
| Evaluation | QA bridge | | Multi-grader (2B-3) | Budget degradation (2C-6), enhanced QA bridge (2C-7) | |
| Learning | Evidence gating, KB wipe | | Memory decay (2B-6), effect-type gating (2B-7) | Knowledge distillation (2C-8) | Cross-session transfer (2D-2) |
| Monitoring | Escape velocity | PipelineMonitor (2A-4), parameter bounds (2A-5) | HITL (2B-5) | | Autonomy tracking (2D-1), MASC (2D-5) |
| Context Management | Session compaction | is_enabled (2A-1), stop_on_first_tool (2A-2) | Ralph iterations (2B-1), AdvancedSQLiteSession (2B-4) | | |

---

## Integration Notes

### Where Each Section Goes in the Revised Roadmap

| Section | Placement |
|---------|-----------|
| **Testing Strategy** | New top-level section after "Design Principles" and before "Phase 2A" |
| **Rollback Strategy** | New top-level section immediately after "Testing Strategy" |
| **Doc Search Fix Acknowledgment** | Add row to "Foundation: What Phase 1 Delivered" table |
| **KB Seeding Strategy** | New subsection under Phase 2A, after 2A-0 (Manual Rewrite) |
| **Orchestrator Decomposition Gate** | New top-level section between Phase 2B and Phase 2C |
| **Research → Learning Pipeline Diagram** | New top-level section after "Design Principles", before phase details. Provides the "big picture" that contextualizes all work items. |

### Inline Test Blocks

Each existing work item (2A-1 through 2D-5) should receive a `**Tests:**` block following the template in Section 1. These are written per-item by the phase revision writers, not in this document.
