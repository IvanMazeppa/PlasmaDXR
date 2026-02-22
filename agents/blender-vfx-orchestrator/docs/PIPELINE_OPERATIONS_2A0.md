# Phase 2A-0: Documentation Pipeline — Operations Guide

**Status:** Code complete, ready to execute
**Date:** 2026-02-22

---

## What This Pipeline Does

Processes the Blender 5.0 manual (physics section, ~119 pages) through an LLM rewrite that converts human-oriented documentation into dense, LLM-optimized reference text. The rewritten content is uploaded to an OpenAI vector store where the research agent can discover techniques, parameter relationships, and workflows it can't find in LLM training data alone.

**Why it matters:** Without this, the research agent defaults to 3-4 well-known Mantaflow techniques from its training data. Every downstream feature that depends on technique diversity (UCB1 selection, micro-experiments, multi-physics) is building on sand.

---

## Pipeline Steps

### Step 1: Rewrite Manual Pages (~5 min, ~$0.19)

```bash
cd agents/blender-vfx-orchestrator
source venv/bin/activate

# Preview what pages will be processed (no API calls, $0)
python3 scripts/experiment_manual_rewrite.py --all-physics --dry-run

# Process all physics pages ($0.15-0.25 depending on page count)
python3 scripts/experiment_manual_rewrite.py --all-physics --output rewritten_manual/
```

| Detail | Value |
|--------|-------|
| Input | ~119 HTML pages from `assets/blender_manual_html/` |
| Output | Markdown files in `rewritten_manual/` + `_rewrite_summary.json` |
| Model | gpt-5-mini (cheapest, fast) |
| Cost | ~$0.19 (measured on 5-page sample: $0.008, extrapolated) |
| Time | ~3-5 minutes (limited by API throughput, not complexity) |
| Compression | ~64% average (4,041 words → 898 words on domain settings page) |

**What the rewrite produces:**
- Dense structured markdown with `TECHNIQUE:` and `GOTCHAS:` sections
- Parameter format: `parameter_name` (type, range, default) — description
- Cause/effect relationships: "Higher resolution → more detail but slower bake"
- All UI navigation, screenshots, and verbose prose stripped

**Quality check:** Review 2-3 output files in `rewritten_manual/` before proceeding. Look for:
- TECHNIQUE sections present
- Parameter names match Blender 5.0 (not hallucinated)
- Compression is meaningful (not just truncation)

### Step 2: Upload to Vector Store (~2-3 min, $0)

```bash
# Dry run — show what would be uploaded ($0)
python3 scripts/upload_rewritten_manual.py --input rewritten_manual/ --dry-run

# Upload for real (creates new vector store, $0 — storage is free)
python3 scripts/upload_rewritten_manual.py --input rewritten_manual/
```

| Detail | Value |
|--------|-------|
| Input | `rewritten_manual/` directory from Step 1 |
| Output | New OpenAI vector store with all .md files indexed |
| Cost | $0 (vector store creation and file upload are free) |
| Time | ~2-3 minutes (upload + processing wait) |

The script will:
1. Create a new vector store named `blender-manual-rewritten-physics`
2. Upload all .md files
3. Poll until all files are processed
4. Run a test query (`"fire simulation techniques mantaflow"`)
5. Print the store ID

**Save the store ID** — you'll need it for Step 3.

### Step 3: Configure the Store ID (~30 seconds)

```bash
# Set the env var (add to your .env or shell profile)
export BLENDER_REWRITTEN_MANUAL_STORE_ID=vs_XXXXXXXX

# Verify it's working
python3 scripts/upload_rewritten_manual.py --verify-only --store-id vs_XXXXXXXX
```

This env var tells `semantic_docs_tools.py` to include the rewritten manual store when routing `intent="manual"` queries. The rewritten store is searched **first** (higher info density), then the original manual store.

### Step 4: Seed the Knowledge Base (~1-2 min, ~$0.30-0.60)

```bash
# Preview what will be seeded ($0)
python3 scripts/seed_kb.py --dry-run --verbose

# Seed all 6 effect types
python3 scripts/seed_kb.py --verbose

# Or seed one at a time
python3 scripts/seed_kb.py --effect-type fire --verbose
```

| Detail | Value |
|--------|-------|
| Effect types | fire, smoke, liquid, rigid_body, particles, cloth |
| Cost | ~$0.30-0.60 (6 vector store queries via gpt-5-mini) |
| Time | ~1-2 minutes |
| Output | KB entries with `trust_level="emerging"` |

Seeded entries are **not** injected into script writer prompts until they earn `trust_level="trusted"` through 3+ successful production uses.

### Step 5: Validate (~1 min, $0)

```bash
# Run technique discovery tests
python3 -m pytest tests/test_technique_discovery.py -v
```

Expected: all 3 tests pass — fire techniques return 2+ results, rigid body returns results, API queries still work.

---

## Total Cost Summary

| Step | Cost | Time |
|------|------|------|
| Step 1: Rewrite manual | ~$0.19 | ~5 min |
| Step 2: Upload to vector store | $0.00 | ~3 min |
| Step 3: Configure env var | $0.00 | ~30 sec |
| Step 4: Seed KB | ~$0.30-0.60 | ~2 min |
| Step 5: Validate | $0.00 | ~1 min |
| **Total** | **~$0.50-0.80** | **~12 min** |

---

## Do Other Phase 2 Upgrades Need This Pipeline First?

**Short answer: No, everything except KB seeding can be developed and tested without running this pipeline. But the system won't produce QUALITY results without it.**

### Hard Dependencies (must run pipeline first)

| Item | Why |
|------|-----|
| **KB Seeding** (scripts/seed_kb.py) | Queries the rewritten manual store — without it, there's nothing to seed from |

### Soft Dependencies (works without it, but reduced effectiveness)

| Item | Impact Without Pipeline |
|------|----------------------|
| **2C-1: UCB1 Technique Selection** | Reduces to round-robin over 3-4 known techniques instead of exploring 10+ |
| **2C-2: Escape Velocity v2** | Knowledge base empty — no fallback techniques to discover |
| **2C-3: Micro-experiments** | No technique candidates to experiment with beyond training data defaults |
| **E2E Production Runs** | Research agent finds same 3-4 Mantaflow techniques every time |

### No Dependency (fully independent)

| Item | Why Independent |
|------|----------------|
| **2A-1: Conditional Tool Enabling** | Budget/iteration gating — no docs involved |
| **2A-2: Deterministic Agent Control** | `tool_use_behavior` — pure SDK config |
| **2A-3: Tool Guardrails** | Truth pack enforcement — uses bl_rna, not docs |
| **2A-4: Pipeline Monitor** | Oscillation/loop detection — pure Python |
| **2B-1: Stateless Iterations** | Iteration loop refactor — pure architecture |
| **2B-2: Context Management** | Token budgets — pure architecture |
| **2B-3: Multi-Grader Eval** | Evaluation pipeline — uses ML metrics, not docs |
| **2B-4: HITL Framework** | Human-in-the-loop — SDK feature |
| **2B-5: Orchestrator Decomposition** | Code refactoring — pure architecture |

### Recommendation

Run the pipeline **before your first real E2E test** after Phase 2A/2B code is complete. All the code changes (2A-1 through 2B-7) can be developed and unit-tested without it. But the moment you want to run a full asset generation and see whether the system produces diverse, quality results — you need the pipeline.

---

## Rollback

If the rewritten manual produces bad results:

```bash
# Remove the env var — system reverts to original manual store
unset BLENDER_REWRITTEN_MANUAL_STORE_ID
```

The original manual vector store is untouched. The rewritten store can be deleted via the OpenAI API if desired.

---

## Files Involved

| File | Role |
|------|------|
| `scripts/experiment_manual_rewrite.py` | Step 1: LLM rewrite of manual pages |
| `scripts/upload_rewritten_manual.py` | Step 2: Upload to vector store |
| `scripts/seed_kb.py` | Step 4: KB seeding |
| `tools/semantic_docs_tools.py` | Routing: `BLENDER_REWRITTEN_MANUAL_STORE_ID` env var |
| `tools/experiment_tracker_tools.py` | KB entry creation: `seed_technique_entry()` |
| `tests/test_technique_discovery.py` | Step 5: Validation tests |
