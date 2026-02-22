# Phase 2 Roadmap Critique & Revision Guide

**Author:** Claude Opus 4.6 (reviewing the work of a 5-agent research team)
**Date:** 2026-02-22
**Purpose:** Detailed critique of `PHASE2_ROADMAP.md` for use as input to a revision pass
**Status:** Review notes — not a replacement for the roadmap

---

## Overall Assessment

The roadmap is **well-researched, well-structured, and mostly correct**. The 4 research documents (SDK Analysis, Codebase Architecture, Autonomous Systems, Monitoring Architecture) are thorough — 20+ real sources properly cited and applied. The design principles, conflict resolutions, dependency graph, and phasing are all sound.

However, there is **one critical gap** that undermines Phases 2C and 2D, and several smaller issues that should be addressed before implementation begins.

---

## CRITICAL: The Documentation & Technique Discovery Gap

### The Problem

The roadmap has zero work items addressing **how agents learn new techniques**. This is the single biggest threat to the project's stated goal of autonomous, self-learning VFX generation.

Here's the causal chain:

1. The Blender manual is written for humans — UI navigation instructions, screenshot references, verbose prose. Low information density for LLM consumption.
2. The Python API reference tells you WHAT exists (`FluidDomainSettings.burning_rate` is a float) but NOT why, when, or how to use it.
3. The research agent can only discover techniques through documentation search.
4. If documentation search returns low-quality results (human-oriented prose, missing conceptual content), the agent falls back on its training data.
5. LLM training data contains a small handful of well-known Blender techniques (basic Mantaflow fire, basic liquid, basic smoke).
6. **Result: the agent uses the same 3-4 techniques for EVERYTHING, regardless of prompt.**

This was diagnosed during the current session. Three bugs were found and fixed in `semantic_docs_tools.py`:

- **Bug 1:** API vector store results filled a shared quota before manual queries ran — manual content was completely blocked.
- **Bug 2:** Manual queries weren't explicitly routed to the manual vector store.
- **Bug 3:** No technique discovery queries existed — the system never asked "what alternative methods exist for X?"

These bugs are now fixed (separate quotas, explicit intent routing, technique discovery queries). But fixing the search routing only helps if the underlying manual content is useful for LLM consumption — and it isn't.

### Evidence: The Manual Rewrite Experiment

An experiment was run during this session that validates the solution:

**Setup:** 5 Blender manual pages (physics/fluid) processed through `gpt-5-mini` with a rewrite prompt optimizing for LLM consumption.

**Results:**

| Page | Original Words | Rewritten Words | Compression | Cost |
|------|---------------|----------------|-------------|------|
| Domain Settings | 4,041 | 898 | 78% smaller | $0.003 |
| Flow | 2,387 | 919 | 61% smaller | $0.002 |
| Gas (index) | 37 | 281 | Expanded (sparse original) | $0.001 |
| Noise | 756 | 451 | 40% smaller | $0.001 |
| Cache | 1,751 | 723 | 59% smaller | $0.002 |
| **Total** | **8,972** | **3,272** | **64% avg** | **$0.008** |

**Quality assessment:** The rewritten output is excellent. Dense, structured, with explicit TECHNIQUE and NOTES/GOTCHAS sections. Example from `flow.md`:

```
## TECHNIQUE
- For animated emission toggling, animate the Flow enable/disable property
  rather than moving domain parameters.
- Use Outflow paired with Inflow to prevent domain overfill.
- Use Sampling Substeps > 0 for fast-moving inflows to avoid gaps;
  tradeoff: more substeps → more compute.
- For planar or non-manifold meshes, enable `Is Planar` so the simulator
  treats them appropriately.
```

This is exactly what the research agent needs — actionable technique knowledge in dense format.

**Cost projections:**

| Scope | Pages | Estimated Cost | Value |
|-------|-------|---------------|-------|
| Physics only | 119 pages | ~$0.19 | Covers all physics systems the orchestrator needs |
| Full manual | 2,197 pages | ~$3.44 | Complete Blender knowledge (materials, rendering, modeling, etc.) |

Processing is offline (local gpt-5-mini calls), then upload to vector store. No ongoing cost.

### What the Roadmap Must Include

The roadmap needs a work item (ideally in Phase 2A, before anything else) that:

1. **Processes the physics section of the Blender manual** (119 pages) through the rewrite pipeline
2. **Uploads rewritten content** to the manual vector store (or a new dedicated store)
3. **Validates** that the research agent can now discover techniques it previously couldn't
4. **Establishes a workflow** for processing additional manual sections as the system expands to new physics types

Without this, the following roadmap items are building on sand:
- **2C-1 (UCB1 Technique Selector)** — UCB1 is useless if there are only 3-4 known techniques to select from
- **2C-3 (Micro-Experiment Sandbox)** — experiments need technique candidates discovered through research
- **2C-2 (Multi-Physics Truth Pack)** — the truth pack tells you what attributes exist, but agents need the manual to know what to DO with them
- **2D-2 (Cross-Session Learning)** — learning transfer requires a foundation of diverse techniques to learn from

### The Complete Research → Learning Pipeline

The roadmap treats documentation and self-learning as separate concerns. They are not. Here is the complete pipeline that must work end-to-end:

```
MANUAL (LLM-optimized)          API REFERENCE (truth pack)
        │                                │
        ▼                                ▼
   "HOW/WHY/WHEN"                  "WHAT EXISTS"
   Techniques, workflows,          Attribute names, types,
   parameter relationships,        valid ranges, defaults
   conceptual understanding
        │                                │
        └──────────┬─────────────────────┘
                   ▼
           RESEARCH AGENT
           Discovers techniques,
           understands approaches,
           knows what's possible
                   │
                   ▼
           TECHNIQUE SELECTION
           UCB1 + LLM override
           (needs candidates to select from)
                   │
                   ▼
           SCRIPT GENERATION
           Creates complete scene script
           (informed by research + truth pack)
                   │
                   ▼
           EXECUTION → EVALUATION
                   │
                   ▼
           LEARNING AGENT
           Records what worked/failed
                   │
                   ▼
           KNOWLEDGE BASE
           Stores validated patterns
           with evidence gating + decay
                   │
                   ▼
           DYNAMIC INSTRUCTIONS
           Injects trusted patterns into
           future research/generation prompts
                   │
                   └──→ feeds back to RESEARCH AGENT
```

Every box in this diagram exists in the codebase or the roadmap. But the top-left box — "MANUAL (LLM-optimized)" — is missing from the roadmap. Without it, the pipeline starts with a crippled research agent that can only echo training data.

### The Self-Learning Feedback Loop Depends on Technique Diversity

The self-learning system (evidence gating, memory decay, knowledge distillation, cross-session transfer) only produces value if the system has diverse techniques to learn FROM. If the agent only ever tries mantaflow_gas, mantaflow_liquid, and basic_smoke, then:

- The KB accumulates 3 patterns
- UCB1 has 3 arms to pull
- Micro-experiments test 3 techniques
- Cross-session learning transfers knowledge about 3 techniques

This is not self-learning. This is memorizing the same 3 approaches. The manual rewrite is what breaks this ceiling — it gives the research agent access to the full breadth of Blender's capabilities in a format it can actually use.

### Specific Workflow for Manual Processing

The experiment script (`scripts/experiment_manual_rewrite.py`) already exists and works. The workflow is:

1. **Run the script** on the physics section (119 pages): `python scripts/experiment_manual_rewrite.py --all-physics --output rewritten_manual/`
2. **Review a sample** of the output for quality
3. **Upload** to a new OpenAI vector store (or append to the existing manual store)
4. **Update** `semantic_docs_tools.py` to query the rewritten store
5. **Validate** with a test: ask the research agent "what alternative techniques exist for creating fire effects?" and verify it returns techniques beyond basic mantaflow_gas
6. **Expand** to other manual sections (materials, rendering, modeling) as the system needs them

Estimated cost: $0.19 for physics. Estimated time: 1-2 hours including review and upload. This is the highest ROI work item in the entire Phase 2 plan.

---

## CONCERN: No Testing Strategy

Phase 1 shipped 16 escape velocity tests. The Phase 2 roadmap has success criteria per phase but no test plans per work item.

### What's Needed

Every work item in 2A should specify:
- **Unit tests** for the new code (e.g., PipelineMonitor: test oscillation detection with synthetic parameter histories)
- **Integration test approach** (e.g., how to verify tool guardrails actually prevent hallucinated attributes from reaching Blender)
- **Regression tests** for removed code (e.g., after deprecating Spec-First pipeline, verify Script Writer still gets adequate context)

This doesn't need to be elaborate. For each 2A item, 3-5 targeted tests. Example:

**2A-4 (PipelineMonitor) tests:**
1. Oscillation detection: feed values [50, 2500, 10] → assert `is_oscillating() == True`
2. No false positive: feed values [50, 100, 150] → assert `is_oscillating() == False`
3. Cascade detection: feed score drop after QA suggestion → assert warning generated
4. Budget alert: set budget to $0.50, feed $0.60 spent → assert CRITICAL alert
5. Integration: mock an iteration loop, verify monitor produces status artifact

**2A-6 (Deprecate Spec-First) tests:**
1. Script Writer receives truth pack context when Spec-First pipeline is removed
2. Code Writer guardrail still works with `truth_pack_to_api_spec()` data
3. Technique selection still runs (was previously parallel with API Spec)

---

## CONCERN: Ralph Iterations (2B-1) Risk

2B-1 is correctly identified as the highest-impact change. It's also the highest-risk.

### The Risk

The iteration loop is the core of `orchestrator.py` (~2,300 lines in `create_asset_pipeline()`). Restructuring it to use fresh `Runner.run()` calls per iteration means:

- Every phase must serialize its output to the state file
- Every phase must reconstruct its input from the state file
- All failure routing (Phase 2.5-2.9 error recovery) must work across Runner.run() boundaries
- SharedContext must be rebuilt for each iteration
- The existing session/compaction system must be adapted

This is a ~400-line change to the most critical code in the system.

### Recommendation

**Do not bundle 2B-1 with other 2B work.** Give it a dedicated sprint:

1. Write the `IterationState` dataclass and serialization
2. Restructure ONE iteration to be stateless (iteration 2 only — iteration 1 stays as-is for baseline)
3. Run E2E and compare iteration 2's behavior between old (accumulated context) and new (fresh context)
4. If quality improves or stays equal, convert all iterations
5. Only then proceed to other 2B items

The roadmap currently schedules 2B-1 in Week 2 alongside AdvancedSQLiteSession migration and memory decay. That's too much high-risk change at once.

---

## CONCERN: Line Estimates Are Optimistic

Based on Phase 1 experience (truth_pack.py was estimated at ~500 lines, shipped at 743), I'd apply a 1.5x multiplier to all estimates:

| Phase | Roadmap Estimate | Realistic Estimate |
|-------|-----------------|-------------------|
| 2A | ~792 lines | ~1,200 lines |
| 2B | ~1,350 lines | ~2,000 lines |
| 2C | ~1,130 lines | ~1,700 lines |
| 2D | ~950 lines | ~1,400 lines |
| **Total** | **~4,222** | **~6,300** |

This isn't a problem — it just means timeline expectations should be adjusted. The phasing and prioritization remain correct.

---

## CONCERN: Phase 2D Is Vision, Not Plan

Items 2D-3 (Prompt Versioning), 2D-4 (Adaptive Replanning), and 2D-5 (Full MASC) are speculative. They depend on everything before them, and the design will be heavily informed by what Phase 2A-C reveals.

### Recommendation

Keep Phase 2D in the roadmap as **directional goals**, but explicitly mark them as "to be designed after Phase 2C ships." Don't assign line estimates or timelines — they'll be wrong.

---

## MINOR: Items to Cut or Deprioritize

### Cut: 2C-4 (Agent.clone())

The SDK Analysis itself says: "dynamic instructions already solve this problem adequately. Clone is cleaner but not urgent." This is 70 lines of code that adds no new capability. Cut it.

### Deprioritize: 2C-5 (run_streamed())

Streaming is a UX improvement, not a reliability or capability improvement. It should move to "nice to have" status after the core 2C items ship.

---

## MISSING: Orchestrator Decomposition Trigger

The roadmap correctly defers orchestrator decomposition until after Ralph (2B-1) ships. But it doesn't specify WHEN to do it.

### Recommendation

Add a concrete trigger: "After 2B-1 ships and passes E2E validation, decompose `create_asset_pipeline()` into phase modules before starting 2C." This prevents the 4,544-line monolith from growing further as 2C adds multi-physics support, micro-experiments, and UCB1 — all of which touch the orchestrator.

---

## MISSING: Rollback Strategy

If a 2B change makes things worse, how do we revert? The roadmap doesn't address this.

### Recommendation

Each sub-phase (2A, 2B, 2C) should be a git branch that can be reverted independently. Within sub-phases, feature flags or config options should allow disabling new behavior:

```python
# config/agent_config.py
ENABLE_RALPH_ITERATIONS = True  # 2B-1: set False to revert to accumulated context
ENABLE_TOOL_GUARDRAILS = True   # 2A-3: set False to revert to pipeline-level validation
ENABLE_PIPELINE_MONITOR = True  # 2A-4: set False to skip monitoring checkpoints
```

This is cheap insurance (3 lines per feature) that allows quick rollback without git surgery.

---

## MISSING: Doc Search Crowding Fix Acknowledgment

The doc search crowding bug (API results blocking manual results) was found and fixed during this session. It's not mentioned in the roadmap. The fix should be acknowledged as a Phase 1 deliverable that directly enables the research pipeline described above.

Key changes made to `semantic_docs_tools.py`:
- Separate quotas: `manual_quota = max(max_results // 2, 3)` and `api_quota = max(max_results // 2, 3)`
- Technique discovery queries: "how to create X effect", "alternative methods techniques for X", "physics simulation types Blender"
- Explicit intent routing: manual queries use `intent="manual"`, API queries use `intent="api"`
- Results interleaved: manual first, then API

Without this fix, the manual rewrite is useless — rewritten content would still be blocked by API results filling the quota.

---

## Suggested Revision Structure

When revising the roadmap, I recommend:

### 1. Add "2A-0: Documentation Pipeline" as the first work item

| File | Change | Lines |
|------|--------|-------|
| `scripts/experiment_manual_rewrite.py` | Already exists — extend with `--upload` flag to push to vector store | ~50 |
| `tools/semantic_docs_tools.py` | Already fixed — acknowledge in roadmap | 0 (done) |
| New: `scripts/upload_rewritten_manual.py` | Upload rewritten pages to OpenAI vector store | ~80 |
| Validation test | Research agent discovers techniques it previously couldn't | ~30 |

Estimated effort: ~160 lines new code + $0.19 processing cost.
Dependencies: None.
Impact: Enables the entire self-learning pipeline.

### 2. Restructure 2B to isolate Ralph

Week 2: 2B-1 (Ralph) alone, with dedicated E2E testing.
Week 3: Everything else in 2B, building on stable Ralph foundation.

### 3. Add test requirements to every 2A/2B item

Even 1-2 lines per item: "Test: verify X with synthetic data" or "Test: E2E run with feature enabled vs disabled."

### 4. Mark Phase 2D as directional

Change "Phase 2D: Full Autonomy" header to "Phase 2D: Full Autonomy (Directional — design after 2C)".

### 5. Cut 2C-4 (Agent.clone())

Remove entirely or move to a "Future Ideas" appendix.

### 6. Add orchestrator decomposition as a 2B/2C boundary gate

"Before starting 2C, decompose `create_asset_pipeline()` into phase modules."

### 7. Add rollback strategy section

Feature flags for each major behavioral change.

---

## Summary of Priorities

| Priority | Item | Why |
|----------|------|-----|
| **P0** | Add documentation pipeline (manual rewrite + upload) | Enables technique diversity — without this, self-learning has nothing to learn from |
| **P1** | Add test requirements to 2A/2B items | Prevents shipping untested infrastructure |
| **P1** | Isolate Ralph (2B-1) into its own sprint | Highest-risk change needs focused attention |
| **P2** | Cut Agent.clone() (2C-4) | Adds no capability, clutters plan |
| **P2** | Mark Phase 2D as directional | Prevents false precision on speculative items |
| **P2** | Add rollback strategy | Cheap insurance |
| **P3** | Add orchestrator decomposition gate | Prevents monolith from growing during 2C |
| **P3** | Acknowledge doc search fix | Completeness |
