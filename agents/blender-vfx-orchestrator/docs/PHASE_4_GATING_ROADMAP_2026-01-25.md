# Phase 4 Gating Roadmap (2026-01-25)

## Purpose
Define the remaining **blocking issues** that must be fixed before starting
Phase 4 (session compaction + cross‑session bootstrap) or any alternative
autonomy workflow. This is a stabilization gate to prevent bad learning
signals from being amplified.

---

## Phase 4 Gate (Must Pass)
1) **Modification contract enforced**  
   - Coordinator outputs **flat, exact Config keys** only (no nested JSON, no prose).
2) **Doc grounding reliable**  
   - `doc_refs` resolve to real `DocPath`, not temp filenames.
3) **Trace correlation working**  
   - SDK trace uses `group_id=session_id` and local JSONL includes the same ID.
4) **Spec‑first doc search discipline**  
   - API Spec Agent stops searching after Turn 3 and outputs APISpec on Turn 4.  
   - Doc‑query count bounded (no semantic_search spam).

If any of the above is red, **Phase 4 must not start**.

---

## Remaining Issues (Blocking)

### 1) Modification Coordinator → `_modify_script_impl` Contract Break
**Evidence:** `traces/communication_flow.jsonl` shows `changes_made: []` despite
non‑empty coordinator parameters.  
**Impact:** Iterations do not actually change scripts → loop stalls.  
**Required fix:**  
- Enforce a strict output contract (flat key/value, exact Config names).  
- Add output guardrail to reject nested/prose output.  
- Optional: add a mapping layer (last resort).

### 2) Doc Grounding Reliability (Vector Store + DocPath)
**Evidence:** doc refs often map to temp filenames; empty results trigger guardrail failures.  
**Impact:** Research runs fail or become untrustworthy.  
**Required fix:**  
- Ensure uploaded chunks contain `DocPath` header and docs are split by store.  
- Verify coverage with a repeatable test (e.g., `bpy.types.FluidDomainSettings`).  
- Use SDK `file_search` fallback when direct search is empty/weak.  

### 3) Trace Correlation and SDK Trace Discipline
**Evidence:** local JSONL flow logs have no `group_id`; nested traces fragment runs.  
**Impact:** Hard to correlate failures to SDK trace view; debugging slows down.  
**Required fix:**  
- One outer `trace()` per pipeline run with `group_id=session_id`.  
- Inject `group_id` into JSONL flow events.  
- Remove nested `trace()` blocks.

### 4) Research Warnings Grounding
**Evidence:** Research Agent warnings reference KB but KB tools aren’t in the toolset.  
**Impact:** Warnings may be hallucinated or stale.  
**Required fix:**  
- Either add KB tools to Research Agent, or explicitly limit warnings to docs/patterns.

### 5) Manual Expert Subagent Enforcement
**Evidence:** doc queries can be skipped if search returns empty or hooks mis‑detect.  
**Impact:** System drifts into Blender 3‑4 patterns.  
**Required fix:**  
- Treat `doc_refs` as **mandatory** for new techniques or API changes.  
- Update hooks to detect **`blender_doc_search_bundle`** and file_search fallback.

### 6) Blender API Sequence Safety (Fluid modifier order)
**Evidence:** `domain_settings` is `None` unless `fluid_type='DOMAIN'` is set first.  
**Impact:** Hard crashes before execution → no learning signal.  
**Required fix:**  
- Enforce the setup order in prompts + validator (already added, needs verification).

### 7) Spec‑First Doc Search Throttling (API Spec Agent)
**Evidence:** `semantic_search_blender_docs` called ~88 times in spec‑first trace.  
**Impact:** Cost/latency blow‑up; weak doc grounding despite high volume.  
**Required fix:**  
- Reduce hook limits for API Spec Agent (consecutive/exempt tool calls).  
- Block doc searches after Turn 3; enforce “output only” on Turn 4.  
- Track doc_query_count in hook stats for regression visibility.

---

## Roadmap (Stabilization → Autonomy)

### Phase 3.5 — Stabilize (Required)
1) **Coordinator output guardrail** (flat Config keys only)  
2) **Vector store coverage validation** (manual + API queries pass)  
3) **SDK trace discipline + JSONL correlation**  
4) **Doc‑query detection updates** (bundle + file_search)  
5) **Research warning grounding decision** (KB or docs only)  
6) **Spec‑first doc search throttling** (API Spec Agent)

### Phase 4 — Long‑Run Memory (After Gate Passes)
6) **Session summarization + compaction**  
7) **Cross‑session bootstrap** (load prior patterns + warnings)

### Phase 5 — Autonomy Expansion (Optional Alternative)
8) **Planner–Executor–Verifier core**  
9) **Beam search as exploration layer (N=2–3, K=1)**  
10) **Bandit technique selection**  
11) **Small population (3–5 scripts)**

---

## Verification Checklist (Before Phase 4)
- 2‑iteration shakedown run completes without `MaxTurnsExceeded` or empty `doc_refs`.
- Coordinator modification produces **non‑empty** `changes_made`.
- SDK trace view shows **single trace row** per run and correct flow.
- `doc_refs` contain real `DocPath` entries (not temp filenames).
- API Spec Agent doc search count within target range and stops after Turn 3.

---

## Status Snapshot (Today)
- Phases 1–3: ✅ complete  
- Phase 4: ⏳ blocked by above issues  
- Alternative workflows: ✅ spec’d, ⚠️ not safe until gates pass

