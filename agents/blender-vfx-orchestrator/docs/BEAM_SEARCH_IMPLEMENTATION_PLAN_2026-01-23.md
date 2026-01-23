# Beam Search Implementation Plan (Quick Wins → Full Integration)

**Goal:** Add a low‑cost beam‑search workflow (N=2–3, K=1) to the existing orchestrator without new tools or sub‑agents.

This plan is laid out in **Quick Wins → Phased Steps** so you can implement incrementally.

---

## Quick Wins (Minimal changes, immediate value)

### QW‑1) Candidate data structure
Create a small `Candidate` dataclass (or dict schema) to hold:
- `script_path`
- `technique`
- `params`
- `score`
- `issues`

Why: This gives the pipeline a first‑class object to rank and select.

### QW‑2) Deterministic ranking
Sort candidates by:
1. `passed` (if available)
2. `overall_score` (desc)
3. Fewer critical issues

Why: A stable ranking function is the backbone of beam search.

### QW‑3) Keep‑top‑K selection
Add a simple `select_top_k(candidates, k)` helper.

Why: Beam search is just repeatable **generate → evaluate → keep**.

---

## Phase 1 — Minimal Beam Search (N=2, K=1)

### Step 1: Generate 2 candidates instead of 1
Replace the single Script Writer call with **two calls** (or one call that produces two variants).

Options:
- Call Script Writer twice with different random seeds/techniques.
- Call Learning Agent to propose two parameter sets.

### Step 2: Execute and evaluate both
Run Executor + Quality Analyst for each candidate (sequential is fine).

### Step 3: Keep best candidate
Select the top candidate and continue the iteration loop with that script.

---

## Phase 2 — Beam Search (N=3, K=1)

### Step 4: Expand candidate diversity
Ensure each candidate differs meaningfully:
- Different technique (if possible)
- Different parameter subset
- Different structural change (e.g., shader vs sim)

### Step 5: Add diversity guard
Reject candidate if it is too similar to the current best (e.g., identical params).

---

## Phase 3 — Beam Search with Reuse (N=3, K=2)

### Step 6: Keep top‑2 instead of top‑1
Maintain a small beam across iterations:
- Candidate A (best)
- Candidate B (runner‑up)

### Step 7: Use runner‑up as fallback
If best stalls for 2 iterations, swap in runner‑up.

---

## Phase 4 — Optional Enhancements

### Step 8: Bandit‑driven technique selection
Use a lightweight UCB/Thompson sampler to select techniques.

### Step 9: Seed candidate pool with patterns
If pattern library has high‑confidence matches, apply them as one of the beam candidates.

---

## Mapping to Existing System

| Phase | Existing Component | Beam Search Adjustment |
|------|---------------------|------------------------|
| Script Writer | `self._script_agent_standalone` | call multiple times |
| Executor | `self._executor_agent_standalone` | run per candidate |
| Quality Analyst | `self._quality_agent_standalone` | score per candidate |
| Learning Agent | `self._learning_agent_standalone` | optional candidate generator |

---

## Plugin Snippets (agent-orchestration:multi-agent-optimize)
These are **pseudo‑code templates** to speed up integration. Replace placeholders with the plugin’s actual API and config keys.

### Snippet 1 — Candidate loop (direct orchestration)
```
beam_width = 3
keep_top_k = 1
candidates = []

for i in range(beam_width):
    # Generate candidate i
    script = run_script_writer(prompt_variant=i)
    execution = run_executor(script)
    quality = run_quality_analyst(execution)

    candidates.append(
        Candidate(
            script_path=script.script_path,
            technique=script.technique_used,
            params=script.parameters_set,
            score=quality.overall_score,
            issues=quality.issues,
        )
    )

best = select_top_k(candidates, keep_top_k)[0]
```

### Snippet 2 — Plugin wrapper (beam search)
```
# PSEUDO-CODE: replace with actual plugin API
from <plugin_package> import MultiAgentOptimize

optimizer = MultiAgentOptimize(
    model="claude-opus-4.5",
    beam_width=3,
    keep_top_k=1,
    candidate_generator=generate_candidate,
    evaluator=evaluate_candidate,
)

best = optimizer.run(seed=baseline_candidate, max_rounds=5)
```

### Snippet 3 — Deterministic selector
```
def select_top_k(candidates, k):
    # Rank by pass -> score -> fewer issues
    ranked = sorted(
        candidates,
        key=lambda c: (c.passed, c.score, -len(c.issues)),
        reverse=True,
    )
    return ranked[:k]
```

---

## Estimated Effort

| Phase | Effort |
|------|--------|
| Quick Wins | 2–4 hours |
| Phase 1 | 1–2 days |
| Phase 2 | 1–2 days |
| Phase 3 | 2–3 days |
| Phase 4 | 3–5 days |

---

## Bottom Line
You already have the tools. This is primarily **orchestration logic**.  
Start with **Phase 1 (N=2, K=1)** to prove value quickly, then scale up.


