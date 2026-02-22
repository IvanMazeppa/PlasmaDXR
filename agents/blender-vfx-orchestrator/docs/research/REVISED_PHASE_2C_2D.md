## Prerequisite Gate: Orchestrator Decomposition

**Before starting Phase 2C, the orchestrator monolith must be decomposed into phase modules.**

`create_asset_pipeline()` is currently 4,544 LOC. Phase 2C adds multi-physics support (2C-2), micro-experiments (2C-3), UCB1 technique selection (2C-1), and knowledge distillation (2C-6) — all of which touch the orchestrator. Without decomposition, the monolith grows to 5,500+ lines, making each subsequent change harder to test and more likely to introduce regressions.

**Gate criteria (all must pass before any 2C work begins):**

1. 2B-1 (Ralph stateless iterations) shipped and passing E2E validation
2. `create_asset_pipeline()` decomposed into phase modules:
   - `phases/research.py` — Research + technique selection
   - `phases/generation.py` — Script generation + validation
   - `phases/execution.py` — Blender execution + render management
   - `phases/evaluation.py` — Multi-grader evaluation + QA diagnosis
   - `phases/iteration.py` — Iteration loop + state management + learning
3. E2E test passes with decomposed orchestrator (identical behavior to monolith)
4. Each phase module independently testable

**Estimated effort:** ~200 lines (restructuring, not new logic) | **Risk:** Medium (must not change behavior)

**Rollback:** If decomposition introduces regressions, revert to monolith and retry. Phase 2C cannot proceed until this gate passes.

---

## Phase 2C: Advanced Capabilities

**Goal:** Multi-physics support, technique diversity, experimentation mode.
**Risk:** Medium — extends architecture to new physics types.
**Principles served:** P2 (LLM creativity), P3 (Blender is truth), P6 (learning signal).

### 2C-1: UCB1 Technique Selector

**What:** Replace LLM-only technique selection with UCB1 (Upper Confidence Bound) as the default algorithm, with LLM override for novel prompts.

**Why it matters:** Technique selection is a classic multi-armed bandit problem. UCB1 provides optimal exploration/exploitation balance — always trying untried techniques first, then balancing between the best-performing technique and underexplored alternatives.

**Research support:**
- Autonomy Research §4.2 (IBM/AAAI 2026): "UCB1 as default, LLM override when research agent identifies specific reason to deviate"
- Mission Statement §8: "The system should actively explore Blender's capabilities"

**Important dependency note:** UCB1 provides full value only after 2A-0 (documentation pipeline) populates technique diversity. With only 3-4 known techniques, UCB1 reduces to round-robin. Can be implemented before 2A-0 ships using the existing technique pool as bootstrap — but the real payoff comes when the research agent can discover 10-20+ technique variants per effect type from the rewritten manual.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/technique_selector.py` | `select_technique_ucb1()` — UCB1 algorithm with technique statistics | ~120 |
| `orchestrator.py` (Phase 0.5) | Use UCB1 as pre-filter, then Technique Selection Coordinator for final decision with research context | ~45 |
| `tools/experiment_tracker_tools.py` | Add `get_technique_statistics(effect_type)` — per-technique run count, avg score, last used | ~60 |

**Estimated effort:** ~225 lines | **Dependencies:** 2B-7 (effect-type scoping); soft: 2A-0 (documentation pipeline for technique diversity) | **Risk:** Low

**Rollback:** `ENABLE_UCB1_SELECTOR = True` in config. Set `False` to revert to LLM-only technique selection.

**Tests:**
1. UCB1 selects untried technique over tried technique with score 80 (exploration)
2. UCB1 selects highest-scoring technique when all have been tried 10+ times (exploitation)
3. UCB1 balances: technique with 2 tries and score 50 selected over technique with 20 tries and score 55 (exploration bonus)
4. LLM override: when research agent flags specific technique for novel prompt, UCB1 is bypassed
5. Integration: `get_technique_statistics()` returns correct counts after 5 mock runs with varying scores

---

### 2C-2: Multi-Physics Truth Pack Extension

**What:** Extend truth pack `TECHNIQUE_TYPES` mapping to cover rigid body, particle systems, cloth, soft body, and geometry nodes.

**Why it matters:** Currently `TECHNIQUE_TYPES` covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus `_common`. Missing: cloth, geometry_nodes, soft_body (Codebase Analysis §8.1). The truth pack approach is architecture-neutral — adding new physics types is a data addition, not a code change.

**Research support:**
- Codebase Analysis §8.1: "TECHNIQUE_TYPES mapping covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus _common. Missing: cloth, geometry_nodes, soft_body."
- Mission Statement §15: "P1: Rigid body, particle systems. P2: Geometry nodes, cloth/soft body."

**Phased rollout (matching Mission Statement §15 priorities):**
- **First:** Rigid body + particle systems (P1 — these are the next effect types the system needs)
- **Then:** Cloth + soft body + geometry nodes (P2 — as the system expands)

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/truth_pack.py` | Add `TECHNIQUE_TYPES` entries for `cloth`, `soft_body`, `geometry_nodes` with their respective `bpy.types` | ~45 |
| `tools/truth_pack.py` | Add `SETTINGS_MAP` entries for new types (cloth: `ClothSettings`, `ClothCollisionSettings`; geometry nodes: `GeometryNodeGroup`, etc.) | ~60 |
| `tools/parameter_bounds.py` | Add bounds for new physics types | ~90 |

**Estimated effort:** ~195 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

**Rollback:** New `TECHNIQUE_TYPES` entries are additive. Removing them restores previous behavior.

**Tests:**
1. Truth pack generates valid introspection data for `ClothSettings` (cloth type)
2. Truth pack generates valid introspection data for `RigidBodyObject` (rigid body type)
3. Parameter bounds exist and are sane for each new physics type (min < max, defaults within range)
4. `SETTINGS_MAP` for geometry nodes includes `GeometryNodeGroup` and common node types
5. Existing mantaflow/rigid_body introspection unchanged (regression test)

---

### 2C-3: Micro-Experiment Sandbox Mode

**What:** Lightweight experiment runner that tests techniques in isolation with minimal scripts (50-100 lines, 10-30 second execution). Findings feed into the knowledge base.

**Why it matters:** When the system encounters a novel prompt or is stuck (escape level 2+), it should be able to test a technique quickly before committing to a full 500+ line scene script. This builds practical Blender experience through hands-on experimentation.

**Research support:**
- Autonomy Research §4.1: "Micro-experiments: 50-100 line scripts, 10-30 seconds, $0 cost"
- Mission Statement §8: "Sandbox mode — run small experimental scripts, introspect capabilities, record findings"
- Interview: "Even a small sandbox where agents explore different ideas practically"

**Critical dependency:** Micro-experiments need technique candidates to test. These come from the research agent, which depends on the documentation pipeline (2A-0) for discovering techniques beyond the LLM's training data. Without 2A-0, experiments are limited to the same 3-4 known techniques — still useful for validation but not for discovery.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/micro_experiments.py` | `generate_micro_experiment()`, `run_micro_experiment()`, `ExperimentResult` dataclass | ~300 |
| `orchestrator.py` (escape velocity L2+) | Before full technique switch, run micro-experiment to validate technique works | ~60 |
| `orchestrator.py` (novel prompt) | If no matching technique in KB, run experiments on candidate techniques | ~60 |

**Triggers:**
1. Novel prompt that doesn't match known techniques
2. Escape velocity level 2+ (stuck — test alternative before full switch)
3. New truth pack version detected (Blender update)
4. User-requested exploration
5. Between production runs (idle time)

**Estimated effort:** ~420 lines | **Dependencies:** 2C-1 (UCB1 technique selector), 2C-2 (multi-physics truth pack); soft: 2A-0 (documentation pipeline for technique candidates) | **Risk:** Medium

**Rollback:** `ENABLE_MICRO_EXPERIMENTS = True` in config. Set `False` to skip experiments and use direct technique selection (existing behavior).

**Tests:**
1. `generate_micro_experiment("fire", "mantaflow_gas")` produces script < 100 lines with domain + flow + minimal render
2. `run_micro_experiment()` completes in < 30 seconds for a basic mantaflow_gas experiment
3. Failed experiment (syntax error in generated script) returns `ExperimentResult(success=False)` with parsed error
4. Successful experiment result feeds into KB with trust level "emerging" (not "trusted")
5. Escape velocity L2 triggers micro-experiment before full technique switch (integration test with mock Blender)

---

### 2C-4: Graceful Budget Degradation

**What:** Instead of binary budget check (full evaluation OR stop), implement tiered evaluation: full vision ($$$) → ML-only ($0) → deterministic-only ($0) based on remaining budget.

**Why it matters:** Currently the system either runs full evaluation or stops entirely when budget is low. Graceful degradation keeps the pipeline running with reduced evaluation quality, preserving learning signal even on a depleted budget.

**Research support:**
- Codebase Analysis §7.10: "No intermediate modes. Could do lightweight evaluation when budget is low."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/asset_evaluator_tools.py` | Add `evaluate_render_lightweight()` — ML metrics only, no vision | ~60 |
| `orchestrator.py` (Phase 3) | Budget-based evaluation tier selection | ~45 |
| `tools/deterministic_quality_checks.py` | Add `evaluate_render_deterministic()` — histogram + script checks only | ~45 |

**Tier selection logic:**
```
Budget > 50% remaining → Full evaluation (deterministic + ML + vision)
Budget 20-50% remaining → ML-only (deterministic + ML, skip vision)
Budget < 20% remaining → Deterministic-only (histogram + script checks)
Budget exhausted → Pipeline stops with best result
```

**Estimated effort:** ~150 lines | **Dependencies:** 2B-3 (multi-grader evaluation) | **Risk:** Low

**Rollback:** Set budget thresholds to `0%` and `0%` to always use full evaluation (original behavior).

**Tests:**
1. Budget at 60% → full evaluation runs (all 3 tiers)
2. Budget at 30% → ML-only evaluation runs (no vision API call)
3. Budget at 10% → deterministic-only evaluation runs (no ML, no vision)
4. Budget at 0% → pipeline stops gracefully with best result and quality warning
5. Score from ML-only evaluation is within 15 points of full evaluation on same render (sanity check)

---

### 2C-5: Enhanced QA Diagnosis Bridge

**What:** Extend the QA bridge with automatic parameter extraction (no keyword needed), truth-pack-enhanced ranges on every parameter, and delta feedback from the monitor.

**Why it matters:** The current QA bridge maps 12 keywords to script parameters. This misses issues that don't match any keyword. Automatic extraction gives the modification strategist a complete parameter inventory regardless of what the QA analyst says.

**Research support:**
- Monitoring Architecture §3.6: "Automatic parameter extraction, truth-pack-enhanced ranges, delta feedback"
- Autonomy Research §5.2: "Deterministic script analysis — $0 cost"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/qa_diagnosis_bridge.py` | Add `extract_all_modifiable_params()` — complete parameter inventory without issue matching | ~90 |
| `tools/qa_diagnosis_bridge.py` | Add truth-pack-enhanced ranges: `Line 245: resolution_max = 64 [range: 1-10000]` | ~45 |
| `tools/qa_diagnosis_bridge.py` | Add monitor delta: `Changed 3x in last 3 iterations. OSCILLATING. Bound: [35, 65]` | ~45 |

**Estimated effort:** ~180 lines | **Dependencies:** 2A-4 (PipelineMonitor), 2A-5 (parameter bounds) | **Risk:** Low

**Rollback:** New extraction methods are additive — existing keyword-based matching remains as fallback.

**Tests:**
1. `extract_all_modifiable_params()` finds all `domain_settings.X = Y` assignments in a test script
2. Truth-pack ranges appear in output: `resolution_max = 64 [range: 1-10000, default: 32]`
3. Monitor delta appears when parameter has been changed 3+ times: `OSCILLATING. Bound: [35, 65]`
4. Output format is dense/structured (< 200 tokens for a 700-line script with 15 modifiable params)
5. Regression: existing keyword matching ("too dark" → light params) still works alongside new extraction

---

### 2C-6: Knowledge Distillation from Successful Scripts

**What:** After a run scores >= 60, extract key code sections (lighting setup, material definitions, camera placement, physics config) as named patterns with metadata.

**Why it matters:** The system currently records what happened but doesn't systematically extract reusable code patterns from successful scripts. This is the "learn from successes" half of the learning loop — it turns passing runs into KB entries that future script generation can draw from.

**Research support:**
- Autonomy Research §1.4: "After a run scores >= 60, extract key code sections as named patterns"
- Mission Statement §7: "What to learn: script patterns/code snippets from successful runs"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/knowledge_distillation_tools.py` | Add `extract_code_patterns_from_script()` — parse script into functional sections | ~150 |
| `orchestrator.py` (after quality pass) | Trigger distillation on passing scripts | ~30 |
| `tools/code_pattern_tools.py` | Enhanced `store_pattern()` with section type, effect type, quality score | ~45 |

**Pattern sections to extract:**
- Lighting setup (light types, energy values, positions, colors)
- Material definitions (shader node trees, principled BSDF settings)
- Camera placement (position, rotation, focal length, DOF)
- Physics configuration (domain settings, flow settings, bake settings)
- Scene setup (renderer, frame range, resolution, denoising)

**Estimated effort:** ~225 lines | **Dependencies:** 2B-6 (memory decay), 2B-7 (effect-type gating) | **Risk:** Low

**Rollback:** Distillation is a post-pipeline operation. Disabling it has zero impact on pipeline behavior — patterns simply stop accumulating.

**Tests:**
1. `extract_code_patterns_from_script()` identifies lighting section in a test fire script (finds Area Light creation + energy setting)
2. Extracted pattern includes metadata: `{effect_type: "fire", section: "lighting", quality_score: 65, line_range: [340, 380]}`
3. Stored pattern enters KB with trust level "Emerging" (single evidence point, not "Trusted")
4. Distillation does NOT trigger for scripts scoring < 60 (only passing scripts produce patterns)
5. Effect-type scoping: fire pattern stored with `effect_type="fire"`, not injected into liquid scripts

---

### 2C-7: Streaming for Monitoring (`run_streamed()`) — Nice to Have

**What:** Use `Runner.run_streamed()` for the Script Writer (longest-running agent) to provide real-time progress visibility and enable mid-stream hallucination detection.

**Why it matters:** This is a UX improvement — the user can see script generation in real-time rather than waiting for it to complete. Mid-stream hallucination detection is a bonus but not critical (the truth pack validator catches hallucinations post-generation at $0).

**Research support:**
- SDK Analysis §7: "Real-time progress display, early hallucination detection, tool call monitoring"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py:_run_agent()` | Add `_run_agent_streamed()` variant | ~60 |
| New: `utils/stream_monitor.py` | Event consumer: log progress, detect hallucination patterns in generated text | ~120 |
| `orchestrator.py` (Phase 1) | Use streamed variant for Script Writer | ~15 |

**Estimated effort:** ~195 lines | **Dependencies:** None | **Risk:** Low-Medium

**Why deprioritized:** Streaming is valuable for developer experience but does not improve pipeline reliability, quality, or learning. The truth pack validator (Phase 1) already catches hallucinations post-generation. Core 2C items (UCB1, multi-physics, micro-experiments) deliver direct capability improvements.

**No rollback flag needed** — this is additive and can be enabled/disabled per agent.

**Tests:**
1. Streamed script generation produces identical output to non-streamed (determinism check)
2. Stream monitor detects known hallucination pattern (`resolution_divisions`) mid-generation
3. Stream monitor logs progress events (tool calls, section generation) without impacting generation speed
4. Fallback: if streaming errors, fall back to non-streamed `Runner.run()` transparently

---

### Removed: Agent Specialization (`Agent.clone()`)

**Moved to Future Ideas.** The SDK Analysis itself notes: "dynamic instructions already solve this problem adequately. Clone is cleaner but not urgent." This was 70 lines of code that adds no new capability — dynamic instructions (already implemented) achieve the same effect-type specialization. If dynamic instructions prove insufficient after Phase 2C ships, this can be revisited.

---

### Phase 2C Summary

| Item | Lines | Impact | Status |
|------|-------|--------|--------|
| Prerequisite: Orchestrator decomposition | ~200 | **Gate** — prevents monolith growth | Gate |
| 2C-1: UCB1 technique selector | ~225 | High — optimal exploration/exploitation | Core |
| 2C-2: Multi-physics truth pack | ~195 | High — enables rigid body, particles, cloth | Core |
| 2C-3: Micro-experiment sandbox | ~420 | High — practical technique validation | Core |
| 2C-4: Graceful budget degradation | ~150 | Medium — extends budget runway | Core |
| 2C-5: Enhanced QA bridge | ~180 | Medium — better diagnosis quality | Core |
| 2C-6: Knowledge distillation | ~225 | Medium — learns from successes | Core |
| 2C-7: run_streamed() monitoring | ~195 | Low — UX improvement only | Nice to Have |
| **Core total** | **~1,595** | | |
| **With nice-to-have** | **~1,790** | | |

**Phase 2C success criteria:**
1. Novel prompts (rigid body, particles) produce evaluable renders on first attempt
2. UCB1 selects different techniques for different effect types across 10 runs
3. Micro-experiments validate techniques in <30 seconds before full runs
4. KB accumulates effect-type-scoped patterns from successful runs (at least 5 patterns after 20 passing runs)
5. Budget-exhausted runs still produce deterministic quality scores (graceful degradation)
6. Orchestrator decomposition passes E2E validation before any 2C work begins

---

## Phase 2D: Full Autonomy (Directional — Design After 2C)

**Goal:** Self-improvement, autonomy progression tracking, novel prompt handling.
**Risk:** High — these are experimental capabilities.
**Principles served:** P2 (LLM creativity), P6 (learning signal), P7 (earn autonomy).

**Status: Items 2D-1 and 2D-2 have clear scope and can be estimated. Items 2D-3, 2D-4, and 2D-5 are directional goals — their exact design depends on what Phase 2A-C reveals about pipeline behavior, learning patterns, and remaining failure modes. They will be designed after Phase 2C ships.**

### 2D-1: Autonomy Progression Tracking

**What:** Implement the 5-level autonomy system from the Mission Statement. Track pass rates per effect type. Automatically relax HITL checkpoints as reliability improves.

**Research support:**
- Mission Statement §10: "Autonomy is earned by evidence, not time. Level 0-4 progression."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/autonomy_tracker.py` | `AutonomyLevel` enum, per-effect-type tracking, level evaluation | ~180 |
| `orchestrator.py` | Check autonomy level at HITL checkpoints; skip if level >= threshold | ~45 |
| `session_manager.py` | Add autonomy metrics to session persistence | ~30 |

**Level criteria (from Mission Statement):**

| Level | Requirement | HITL Changes |
|-------|-------------|-------------|
| 0: Guided | Default | All checkpoints active |
| 1: Assisted | 10+ runs, some passes | Skip prompt approval for known types |
| 2: Semi-Autonomous | >50% pass rate, 3+ effect types | Only budget + critical issue checks |
| 3: Autonomous (Known) | >70% pass rate, 50+ runs for this type | Human sees only final result |
| 4: Autonomous (Novel) | Stable L3 + external review | Self-directed exploration |

**Estimated effort:** ~255 lines | **Dependencies:** 2B-5 (HITL framework) | **Risk:** Medium

**Rollback:** `ENABLE_AUTONOMY_PROGRESSION = True` in config. Set `False` to keep all HITL checkpoints active (Level 0 behavior).

**Tests:**
1. New effect type starts at Level 0 with all HITL checkpoints active
2. After 10 runs with 3 passes, effect type promotes to Level 1 (prompt approval skipped)
3. Level does NOT promote on runs alone — 10 runs with 0 passes stays at Level 0
4. Per-effect-type tracking: fire at Level 2, liquid at Level 0, independent progression
5. Level demotion: if pass rate drops below threshold over last 20 runs, level decreases by 1

---

### 2D-2: Cross-Session Learning Transfer

**What:** Patterns learned in one session automatically enhance future sessions. The dynamic instructions system already supports this — this work item makes it systematic with quality gates.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/dynamic_instructions.py` | Load top-N trusted patterns (by retention score) for current effect type into agent instructions | ~60 |
| `tools/experiment_tracker_tools.py` | Add cross-session analytics: "which patterns are most effective across sessions?" | ~90 |

**Estimated effort:** ~150 lines | **Dependencies:** 2B-6 (memory decay), 2B-7 (effect-type gating), 2C-6 (distillation) | **Risk:** Low

**Rollback:** `ENABLE_CROSS_SESSION_PATTERNS = True` in config. Set `False` to use only within-session patterns.

**Tests:**
1. Pattern from session A (fire, score 70, lighting setup) appears in session B's dynamic instructions for fire
2. Pattern with retention < 0.3 (decayed) is NOT injected into new sessions
3. Effect-type gating: fire pattern does NOT appear in liquid session
4. Top-N limit: if 50 trusted patterns exist, only top 5 by retention score are injected (context budget)
5. Cross-session analytics correctly reports most-reused pattern across 3 mock sessions

---

### 2D-3: Prompt Versioning for Script Writer (Directional)

**What:** Track which system prompt variants (instructions, truth pack format, research injection style) produce better quality scores. Auto-promote the best-performing variant.

**Research support:**
- Autonomy Research §1.1 (OpenAI cookbook): "Prompt versioning system with rollback"

**Design direction:** The self-evolving agents cookbook provides a complete pattern: version tracking, aggregate scoring across all test cases, A/B comparison, auto-promotion of best variant. The VFX orchestrator's implementation will be informed by:
- Which instruction components have the most variance impact (discovered during 2A-2C)
- Whether quality score variance is dominated by prompt vs technique vs scene complexity
- The `AdvancedSQLiteSession` branching capability (2B-4) which can run variant A and B from the same checkpoint

**Dependencies:** 2B-3 (multi-grader eval for consistent scoring), 2B-4 (session branching for A/B comparison)

**This item will be designed after Phase 2C ships. No line estimate or timeline assigned.**

---

### 2D-4: Adaptive Replanning (Magentic-One Style) (Directional)

**What:** Replace the fixed state machine with an adaptive pipeline that can jump back to research/technique selection when evaluation reveals the technique is fundamentally wrong.

**Research support:**
- Autonomy Research §2.3 (Magentic-One): "When the plan is revised, all agents clear their contexts and reset states"

**Design direction:** The current state machine (PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE) with escape velocity handles 90% of cases. Adaptive replanning adds the ability to recognize when the chosen technique is fundamentally wrong (not just poorly parameterized) and restart from research. The 5-question progress ledger (Magentic-One pattern) is a lightweight version of this. Key design questions to resolve after Phase 2C:
- Does the PipelineMonitor (2A-4) already catch the cases that would trigger replanning?
- Does UCB1 (2C-1) + micro-experiments (2C-3) make technique-level restarts rare enough that adaptive replanning isn't needed?
- What's the cost/benefit vs simply escalating escape velocity?

**Dependencies:** 2B-1 (Ralph iterations), 2A-4 (PipelineMonitor), 2C-1 (UCB1), 2C-3 (micro-experiments)

**This item will be designed after Phase 2C ships. No line estimate or timeline assigned.**

---

### 2D-5: LLM-Based Anomaly Detection (Full MASC) (Directional)

**What:** Train a lightweight anomaly detector on normal pipeline trajectories. When anomaly score exceeds threshold, a correction agent intervenes.

**Research support:**
- Autonomy Research §2.4 (MASC): "77.84% AUC-ROC on step-level error detection"

**Design direction:** The deterministic PipelineMonitor (2A-4) handles an estimated 95% of detectable failure modes (oscillation, stuck loops, context bloat, cascade detection). Full MASC only becomes valuable if:
- The deterministic monitor misses failure modes that a learned model could catch
- The system has enough pipeline runs (50+) to train a meaningful anomaly detector
- The cost of running the anomaly detector ($0.01-0.02/iteration) is justified by the reduction in failed runs

The decision to implement Full MASC will be made based on PipelineMonitor coverage data collected during Phases 2A-2C.

**Dependencies:** All Phase 2A-C, plus sufficient pipeline run data for training

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
| **Directional total** | **TBD** | | |

**Phase 2D success criteria:**
1. Autonomy level progression demonstrated: at least one effect type reaches Level 2
2. System demonstrably improves over time (same prompt scores higher after 50 runs vs 5)
3. Cross-session learning injects at least 3 trusted patterns into a new session for a known effect type

**Deferred success criteria (for directional items, to be refined after 2C):**
4. Prompt versioning shows measurable quality improvement (if implemented)
5. Adaptive replanning reduces technique-level failures by >30% vs escape velocity alone (if implemented)
6. Anomaly detection catches failures the deterministic monitor misses (if implemented)
