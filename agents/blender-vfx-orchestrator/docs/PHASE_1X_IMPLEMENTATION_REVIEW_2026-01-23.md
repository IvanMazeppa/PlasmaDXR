# Phase 1.x Implementation Review (2026-01-23)

Goal: review Phase 1.x / Quick Wins implementation and identify issues that can block full iterations, especially Blender 5.0 API mismatches.

---

## Summary (Brutal Truth)
Phase 1.x is largely wired, but **API validation and parameter tracking are fragile**. The current fixes can **create new Blender API errors**, and parameter deltas are often lost. This directly explains why “it runs but breaks mid‑iteration.”

---

## Critical Findings

### 1) API correction can produce invalid Python
The API validator replaces **partial tokens** and even injects **comment strings** as replacements. This can produce syntactically invalid code and new runtime failures.

**Why it matters:** Your most common bug class is “API mismatch.” The current correction path can *introduce* mismatches or syntax errors.

**Where it happens:**
- `orchestrator.py`: replaces only the **last token** of an API call, then separately applies global replacements. This can produce half‑fixed calls.
- `api_validator.py`: some corrections are literal comments (e.g., `.volume_samples`), which will break code if inserted inline.

**Actionable fix:**
- Replace token‑level replacements with **full‑pattern replacements** only.
- For removals, either **delete entire lines** or replace with safe no‑ops on their own lines.
- Treat `KNOWN_API_CHANGES` entries with `# ...` as **removals**, not in‑line substitutions.

---

### 2) Lightweight API validation misses Blender 5.0.1 changes
`validate_code_api()` is **pattern‑only** and does not query Blender 5 docs. Any 5.0.1 API changes not in `KNOWN_API_CHANGES` will pass undetected and fail at runtime.

**Why it matters:** The current error class is “API mismatch,” and the lightweight validator **does not** discover new changes.

**Actionable fix:**
- Run the **full API Validator agent** (with doc lookup) at least once per iteration *before* execution.
- If cost is a concern, run full validation only on scripts that failed validation or had high severity issues.

---

## High‑Priority Findings

### 3) Parameter tracking is broken (`parameters_set` vs `key_parameters`)
`ScriptOutput` defines `parameters_set`, but the orchestrator frequently writes and reads `key_parameters`. This causes parameter deltas to drop to `{}` and can cripple learning + baseline logic.

**Why it matters:** You think you’re tracking parameter experiments, but you’re often recording **empty params**, so learning becomes noise.

**Actionable fix:**
- Standardize on `parameters_set` everywhere.
- If you need `key_parameters`, add a **model alias** or a migration layer in the orchestrator.

---

### 4) Pre‑iteration research uses truncated text, so warnings rarely trigger
`pre_iteration_research_direct()` can accept JSON history, but the orchestrator passes a **text summary** with truncated issue strings. This often fails the “same issue” detection.

**Why it matters:** Early warning detection is a key Quick Win; if it doesn’t trigger, you waste iterations.

**Actionable fix:**
- Pass a JSON list of iteration records instead of text.
- Add a `get_iteration_history_json()` helper in `SessionManager`.

---

## Medium‑Priority Findings

### 5) Learning Agent is still post‑hoc only
Docs now define Learning Agent as **pre‑generation**, but the pipeline only runs it after quality eval. This causes doc‑gated proposals and micro‑experiments to be skipped.

**Why it matters:** This allows training‑data drift and reduces Blender‑5‑only compliance.

**Actionable fix:**
- Add a **pre‑generation Learning step** that produces doc‑grounded proposals.
- Enforce doc refs in Learning output; missing refs → Docs Expert.

---

## Quick Doc Fixes
- `BEAM_SEARCH_IMPLEMENTATION_PLAN_2026-01-23.md`: Snippet 2 has a `:)` typo in the `MultiAgentOptimize(...)` example. Fix to `)`.
- `SELF_LEARNING_ARCHITECTURE_PROPOSAL.md`: Status says “ALL STRATEGIES IMPLEMENTED,” but pre‑generation learning and doc‑gating are still partial. Add a short **implementation reality check** near the top.

---

## Recommended Next Actions (Minimal)
1. Fix `parameters_set` vs `key_parameters` mismatch (1 hour).
2. Replace token‑level API fixes with full‑pattern replacements (1–2 hours).
3. Call the **full API Validator agent** on scripts that fail validation (1–2 hours).
4. Pass JSON iteration history into pre‑iteration research (30 min).

