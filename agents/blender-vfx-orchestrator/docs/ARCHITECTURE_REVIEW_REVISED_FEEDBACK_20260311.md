# Feedback on Revised Architecture Review — Claude Opus Response

**Date:** 2026-03-11
**Context:** GPT-5.4 produced a revised architecture review incorporating feedback from the initial review cycle. This document provides assessment, pushback, and answers to the new open questions so a final roadmap can be agreed.

---

## What the Revision Got Right

### Grounded Root Cause Analysis

The revised root causes now cite specific file paths and line numbers (`orchestrator.py:1325`, `phases/execution.py:274`, `hooks/enforcement_hooks.py:737`). This is real architectural analysis, not pattern-matching against failure descriptions. All 5 root causes are correctly identified and the evidence trails check out.

### Feedback Integration

The revision absorbed all 5 answers and adjusted positions:
- No separate Phase 0 metrics pass (merged with implementation)
- Incremental extraction, not big-bang refactor
- APISpec revival killed at API-call level, kept at TechniqueContract+SceneSpec level
- Teaching vs wrapping spectrum acknowledged with evidence gating for graduation
- Budget impact of changes factored into ordering

### "What to Stop Doing" Section

All 10 items are correct. Item 5 (`script.technique_used` redefining technique without adherence check) was validated by the E2E test we just ran — the contract adherence check catches exactly this class of regression. Item 9 (stop postponing extraction) aligns with the incremental approach already underway.

### SDK Features Prioritization

`call_model_input_filter` as #1 is correct. The per-phase context boundary is the highest-leverage single change for cost and quality. The rest of the ordering (AdvancedSQLiteSession, guardrails, stop_on_first_tool, parallel_tool_calls) matches implementation experience.

---

## Where We Disagree or Would Adjust

### Phase Ordering Is Too Sequential

The roadmap chains 8 phases linearly. For a solo developer, this means ~6+ months before the later phases deliver value. In practice:

- **Phases 1-3 can be parallelized.** TechniqueContract is already implemented and proven. Capability packs (Phase 2) and bounded research/context filtering (Phase 3) have no dependency on each other. Both depend on Phase 1, which is done.
- **Phase 6 (incremental extraction) should happen continuously**, not as a discrete phase. Extract each module as its contract stabilizes — this is already the stated approach but shouldn't be a numbered phase.

### Phase 5 (Deterministic Scene Assembly) Is Premature

45% of runs still fail at execution. Spatial reasoning (floating objects, contact gaps) only matters for runs that execute and render. The ROI of building a geometry solver when half the scripts crash is low.

**Recommendation:** Move Phase 5 to Wave 3 (after model upgrade and patchable scripts reduce the execution failure rate to <20%).

### Phase 7 (Model Upgrade) Should Be Pulled Forward

The revision says "delay GPT-5.4 until after contracts are in place." But contracts ARE in place — TechniqueContract is implemented, tested, and proven in E2E. GPT-5.4 is already on the `codex_upgrade` preset for script writing. The upgrade should happen now for code-critical agents (script writer, modification coordinator, research agent) because:

1. Better instruction following amplifies the binding contract we just built
2. Better coding reduces the 45% execution failure rate directly
3. 1M context window gives headroom for larger scripts + technique contracts
4. Native compaction helps with the context bloat issue (Phase 3) immediately

**Recommendation:** Full GPT-5.4 rollout for code-critical agents NOW, in parallel with Wave 1 remaining work (capability packs + bounded research).

### Missing Open Question

The most important question not asked: **What is the minimal viable capability pack set to cover 80% of requested effect types?** We have 4 packs:
- `cell_fracture_rigid_body` (destruction/shatter)
- `mantaflow_fire` (fire/smoke/explosion)
- `mantaflow_liquid` (liquid/water/pour)
- `simple_rigid_body` (basic rigid body)

What's needed for cloth, particles, geometry nodes, and combination effects? Without scoping this, Phase 2 has no definition of done.

---

## Proposed Wave Structure

Rather than 8 sequential phases, compress to 3 waves:

| Wave | Phases | Content | Status |
|------|--------|---------|--------|
| **Wave 1** (now) | 1 + 2 + 3 | TechniqueContract (DONE), capability packs, bounded research, `call_model_input_filter`, GPT-5.4 full rollout | In progress |
| **Wave 2** (next) | 4 + 6 + 7 | Patchable scripts (named sections), incremental extraction, evaluator calibration | After Wave 1 |
| **Wave 3** (later) | 5 + 8 | Deterministic scene assembly, evidence-backed learning | After Wave 2 |

This delivers value faster, respects the solo-developer constraint, and puts the model upgrade where it amplifies existing work instead of waiting for a perfect architecture.

---

## Answers to New Open Questions

### Q1: What evidence threshold should move a feature from "wrapped" to "taught"?

**Proposed metric:** A feature graduates from wrapped capability to taught behavior when the LLM demonstrates **>80% correct usage across 10+ independent runs** without the capability pack's code scaffolding or adherence check intervening.

Concretely: track `contract_adherence_violations` per technique per run. When a technique accumulates 10+ runs with 0 violations and >80% execution success, the system can relax the binding contract to advisory mode for that technique. If violations reappear, re-engage binding.

This aligns with the existing evidence-gating pattern (min_confidence, min_success_rate, decay).

### Q2: Which scene families for v1 of the spatial solver?

**Tabletop/contact scenes first** — these are the most common user request pattern ("wine glass on marble table", "vase on shelf") and the spatial constraints are simple (gravity + contact normal + containment).

After that: **container liquids** (glass contains wine — requires inside/outside reasoning) and **room interiors** (objects on floor, against walls — requires bounding box awareness).

Fracture debris and atmospheric volumes are lower priority because their spatial constraints are emergent from physics simulation, not placement.

### Q3: Manual vs automated addon installation?

**Manual for now, automated later.** Current state: Cell Fracture required `blender --command extension install -e cell_fracture` — a one-time manual step. For a solo developer, documenting this per pack is acceptable.

Automated installation should be added when: (a) the capability pack registry grows beyond 6-8 packs, or (b) the system is deployed to environments where manual setup isn't practical.

### Q4: How much structural enforcement before creativity degrades?

**Named function conventions are the sweet spot.** Requiring `setup_scene()`, `create_geometry()`, `setup_materials()`, etc. constrains structure without constraining content. The LLM has full creative freedom within each function.

The line to watch: if we start prescribing function signatures (specific parameters, return types) or internal structure (specific variable names, specific API call sequences), we've crossed into over-constraint territory. Monitor script diversity — if generated scripts become too similar across different scene descriptions, enforcement is too tight.

### Q5: Branch budget for repair experiments?

**2 branches per iteration, 4 branches per session maximum.** At current costs (~$0.15-0.30 per branch attempt), this keeps repair experiments under $1.20 per session. With a $20/month budget and ~30-40 sessions/month, repair branching would consume at most $0.50-1.20 per session, leaving headroom for the primary pipeline.

Escalation policy:
- Branch 1: patch specific function
- Branch 2: patch different function or different approach
- If both branches fail: escalate to full regeneration (not another branch)
- If regeneration fails: escalate to technique switch
- If technique switch fails: escalate to HITL (or PAUSE if non-interactive)

---

## Implementation Status Update

Since the initial review, the following has been implemented and E2E tested:

| Item | Status | Evidence |
|------|--------|----------|
| TechniqueContract model | DONE | 23/23 tests pass |
| Capability pack registry (4 packs) | DONE | Cell fracture, mantaflow fire/liquid, simple rigid body |
| Contract adherence checking | DONE | E2E: `CONTRACT ADHERENCE: PASSED` both iterations |
| Fuzzy/keyword technique matching | DONE | Maps creative names to registry packs |
| Cell Fracture Blender 5.0 extension fix | DONE | `bl_ext.blender_org.cell_fracture` — execution succeeds |
| Truth pack + API fixer updated for extension system | DONE | Both legacy names auto-fixed to correct module |
| First successful glass shatter render | DONE | Score: 34/100, execution: exit_code=0, 90s render time |

The TechniqueContract architecture is proven. The remaining work is capability pack expansion, bounded research, and the items in Wave 1.
