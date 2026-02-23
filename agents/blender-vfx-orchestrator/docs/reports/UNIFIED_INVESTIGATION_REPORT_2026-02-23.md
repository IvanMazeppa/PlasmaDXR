# Unified Investigation Report: Roadmap Critique + E2E Issues

**Date:** 2026-02-23
**Investigators:** 3 parallel Claude Opus 4.6 agents
**Scope:** Verify 12 Codex critique claims (C1-C12) + analyze E2E runtime failures + cross-reference audit
**Method:** Read-only. No code or doc changes made.

---

## Executive Summary

Three agents investigated the Phase 2 Roadmap from different angles. Here's the consolidated picture:

**The roadmap is architecturally sound. The critique overweights team-scale concerns that don't apply to a solo developer.** Of 12 Codex claims, only 5 are real issues — and none are architectural. They're all operational gaps (missing gate code, inconsistent flags, stale docs, budget enforcement, decomposition estimate).

**The E2E test failures reveal a more actionable problem:** the truth pack handles hallucinations well (59 patterns, 4 validation layers, all $0), but script LOGIC bugs have zero deterministic coverage. This is the real gap.

---

## Part 1: Codex Critique Verdicts (C1-C12)

### Real Issues (5)

| # | Claim | Actual Severity | Key Finding |
|---|-------|----------------|-------------|
| **C2** | Gate statistics too small | **High** | Gates are DOCUMENTATION-ONLY. Zero gate-checking code exists anywhere. No cross-session statistical tracking. The 5-run windows aren't even implemented. |
| **C6** | Budget worst-case missing | **High** | PipelineMonitor has `PER_RUN_BUDGET=$0.50` but it's ADVISORY (alert, not hard stop). No per-day cap. No P90/worst-case analysis. A blown run could eat 5-10% of the $20/month. |
| **C10** | Decomposition underestimated | **Medium-High** | orchestrator.py = 3,901 LOC, 19 methods, 119 `self._` references, 10+ shared instance vars. Roadmap says ~200 lines. Should be 400-600. The project's own history confirms: 2A-6 was estimated at 128 lines, actual was ~730 lines moved. |
| **C11** | Doc drift visible | **Medium** | `REVISED_CROSS_CUTTING_SECTIONS.md` still lists `Agent.clone (2C-4)` (removed), uses old numbering (2C-6/7/8 instead of 2C-4/5/6), wrong branch prefixes (`0.33.2/` vs actual `0.34.X/`). |
| **C12** | No risk register | **Medium** | 33 "risk" mentions scattered across 1,800 lines. No consolidated table with indicators, thresholds, or mitigations. |

### Partially Valid (4)

| # | Claim | Actual Severity | Why Partially |
|---|-------|----------------|---------------|
| **C1** | Scope too broad | Medium | 7/9 Phase 2A items already done. Items are NOT cross-coupled in code. BUT: 2B Week 3 packs 7 items into one week — that IS aggressive. |
| **C4** | Inconsistent DoD | Medium | Directional items lack tests BY DESIGN. Concrete items (2A-2C) all have inline test blocks. The critique's 6-field template would improve consistency. |
| **C7** | KB seeding risk | Medium-Low | Evidence gating already blocks seeded entries from Script Writer (80% success rate threshold). Only Research Agent sees seeds. The existing architecture handles this. |
| **C8** | Multi-grader instability | Medium | Not yet implemented. The planned static weights + short-circuit design mitigates most concern. Adding a divergence alert (if \|Tier2 - Tier3\| > 30, flag) when implementing 2B-3 would close the gap. |

### Not Real Issues (3)

| # | Claim | Why Not Real |
|---|-------|-------------|
| **C3** | No critical-path model | Solo developer. No competing workstreams, no reviewer bandwidth constraints. Items ship sequentially on their own branches. |
| **C5** | Directional items leaky | 2D-3/4/5 have ZERO file paths, ZERO line estimates, ZERO implementation code. The claim that they have "file-level implementation references" is factually incorrect. |
| **C9** | Flag blast radius | Only 3-4 flags exist, each shipped individually on its own branch. Phased defaults add process overhead with no benefit for local development. |

---

## Part 2: E2E Test Issues

### Issue 1: Context-Dependent Blender Enums — HIGH

**Root cause:** `bl_rna.properties` is CLASS-level introspection. It reports what a class CAN hold, not what a live INSTANCE holds. Blender uses runtime callbacks to change available enum values based on domain_type, cache_type, etc.

**Currently handled:** 3 enums hardcoded in KNOWN_HALLUCINATIONS (openvdb_data_depth, cache_particle_format, cache_mesh_format).

**Not yet handled:** `cache_data_format`, `sndparticle_*` properties (LIQUID-only), render engine properties (Cycles vs EEVEE), `ParticleSettings.physics_type` sub-properties.

**Scalability verdict:** The hardcoding approach DOES NOT scale. Each new enum requires discovery through a runtime failure.

**Recommended fix:** "Strip-if-uncertain" policy — flag ALL cache format assignments as context-dependent and let Blender use defaults. The LLM rarely has good reason to override cache formats.

### Issue 2: Script Logic Bugs — HIGH

**The truth pack's gap.** After eliminating hallucinations, the remaining E2E failures are:
- `'NoneType' object has no attribute 'type'` — object not created/found
- `NodeSocketString.default_value expected a string type, not float` — type mismatch

**None of the 4 validation layers catch these.** The truth pack validates attributes exist. The API fixer handles structural patterns. Neither can detect:
- Variables used before assignment
- `.get()` calls without null checks
- Type mismatches in shader node assignments

**Recommended fix:** Add a 5th validation layer — pre-execution AST analysis. Check for variables used before assignment, `.get()` without null checks, and `default_value` type mismatches. This is complementary to, not a replacement for, the truth pack.

### Issue 3: Truth Pack Coverage Gaps — MEDIUM

| Category | Covered | Missing |
|----------|---------|---------|
| Physics techniques | mantaflow_gas, liquid, rigid_body, particle_system | cloth, soft_body, geometry_nodes, dynamic_paint, collision |
| Shader nodes | 8 nodes in TECHNIQUE_TYPES | ShaderNodeAttribute, ShaderNodeMath, ShaderNodeEmission |
| Variable patterns | 5 patterns (dset, fset, etc.) | `ds`, `fs`, `domain`, `ps`, `rb` — common abbreviations |
| SETTINGS_MAP | 9 entries (domain/flow/effector/rigid/cycles/render/scene) | cloth, material, camera, light, modifier |

### Issue 4: Recovery Loop — MEDIUM

The recovery loop is architecturally sound (1 recovery per iteration, truth pack validated, escape velocity escalation) but has two efficiency issues:
1. Recovery prompt does NOT include failed script content (only path) — wastes 1 of 6 LLM turns on file reading
2. No deterministic recovery for known patterns (NoneType → null guard, type mismatch → coercion) before invoking expensive LLM recovery

### Issue 5: Phase 2A Status — CONFIRMED

All 7/9 claimed DONE items verified functional in code. The executive summary says "6/9" — should be "7/9" (2A-6 also done).

---

## Part 3: Cross-Reference Audit

### Document Drift
- `REVISED_CROSS_CUTTING_SECTIONS.md` uses pre-v2.0 numbering — **confirmed**
- All other research docs are consistent with canonical roadmap
- Minor: `ENABLE_CONTEXT_FILTERS` vs `ENABLE_CONTEXT_TRIMMING` naming inconsistency (planned, not implemented)

### Feature Flags
- **NOT centralized** in `config/agent_config.py` as roadmap specifies. That file is the Agent Config Manager (model presets), not feature flags.
- 4 flags exist across 4 different files with **3 inconsistent parsing patterns**:
  - Pattern A: `!= "false"` — setting `"0"` or `"no"` does NOT disable
  - Pattern B: `!= "0"` — setting `"false"` does NOT disable
  - Pattern C: `in ("1", "true", "yes")` — strictest, handles all cases
- `ENABLE_SPEC_FIRST_PIPELINE` referenced in deprecation notices but **does not exist in code**

### Dependency Graph
- Accurate. No circular dependencies. No hidden couplings.
- 2A-5's dependency on 2A-4 is wiring-level (orchestrator.py passes data between them), not module-level (parameter_bounds.py works standalone)

### Mission Statement Alignment
- **Strong.** Every success criterion maps to at least one roadmap item, sequenced in priority order.
- **Two P4 gaps:**
  1. Camera placement verification — known critical failure, no roadmap item addresses it deterministically
  2. Collision effector injection — liquid falls through containers, no planned fix

---

## Priority Actions

### Immediate (Before 2B starts)

1. **Mark `REVISED_CROSS_CUTTING_SECTIONS.md` as superseded** — 5 minutes. Fixes C11.
2. **Standardize feature flag parsing** — Use one pattern across all 4 files. Fixes the `ENABLE_PARAMETER_BOUNDS=false` doesn't-actually-disable bug.
3. **Update executive summary** — "7/9 DONE" not "6/9". Trivial.

### Short-term (During remaining 2A work)

4. **Add `ds`/`fs` to VARIABLE_PATTERNS** — 2 lines. Catches common abbreviations.
5. **Add ShaderNodeAttribute/ShaderNodeMath to mantaflow_gas TECHNIQUE_TYPES** — The API fixer's material snippets use these but the truth pack doesn't validate them.
6. **Convert PipelineMonitor budget alert to hard stop** — Change advisory alert to force-stop when per-run budget exceeded. ~30 lines. Fixes C6.
7. **Add failed script content to recovery prompt** — Saves 1 LLM turn per recovery. ~10 lines.

### When implementing 2B

8. **Add divergence alert to multi-grader design** — ~20 lines when implementing 2B-3. Fixes C8.
9. **Use staged migration for decomposition** — Extract one module at a time, not big-bang. Increase estimate to 400-600 lines. Fixes C10.
10. **Adopt standard DoD template for 2B items** — 6 fields: Objective, Primary metric, Guardrail metric, Exit test, Rollback trigger, Owner. Fixes C4.

### When implementing 2D

11. **Design proper gate statistics** — Smoke gate (5 runs) + stability gate (20 runs) + confidence bounds. Fixes C2.
12. **Create one-page risk register** — 10-row table with Risk/Indicator/Threshold/Action. Fixes C12.

### Design decisions needed

13. **Strip-if-uncertain policy for cache enums** — Stop hardcoding individual bad values. Flag the entire category as context-dependent and let Blender use defaults.
14. **Pre-execution AST analysis layer** — The gap between hallucination prevention (solved) and logic bug prevention (unsolved) is the biggest remaining reliability risk.
15. **Camera placement + collision effector verification** — Two known P4 failures not in the roadmap.

---

## Bottom Line

The Codex critique was useful but calibrated for team-scale concerns. 3 of its "Critical" findings are real (C2, C6) or partially real (C1). The other "Critical" finding (C3) is irrelevant for a solo developer.

**The more actionable finding is from E2E testing:** the system's hallucination prevention is excellent (59 patterns, 4 layers, $0 cost), but script logic bugs have zero deterministic coverage. The truth pack was the right tool for Phase 1. A pre-execution static analysis layer would be the right tool for Phase 2's remaining reliability gap.

---

## Source Reports

- `docs/reports/CRITIQUE_INVESTIGATION_REPORT.md` — Full C1-C12 analysis with code evidence
- `docs/reports/E2E_ISSUES_INVESTIGATION_REPORT.md` — Full E2E issue analysis with line references
- `docs/reports/CROSS_REFERENCE_AUDIT_REPORT.md` — Full audit with file inventory
