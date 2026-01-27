# Self-Improving Coding Agent (SICA)

**Source:** [arXiv:2504.15228](https://arxiv.org/html/2504.15228v1)
**Relevance:** Methodology for agents that improve through iterative self-modification

---

## Key Achievement

SICA demonstrates that "an agent system, equipped with basic coding tools, can autonomously edit itself, and thereby improve its performance" with gains of **17% to 53%** on SWE Bench Verified.

---

## Architecture

### Meta-Agent Loop

```
┌─────────────────────────────────────────────────────────┐
│                    SICA Meta-Loop                       │
├─────────────────────────────────────────────────────────┤
│  1. Start with minimal agent code                       │
│  2. Run benchmark → measure performance                 │
│  3. Best agent becomes "meta-agent"                     │
│  4. Meta-agent reviews archive of past iterations       │
│  5. Meta-agent identifies improvement opportunity       │
│  6. Meta-agent implements code change                   │
│  7. New agent enters archive                            │
│  8. GOTO 2                                              │
└─────────────────────────────────────────────────────────┘
```

### Agent Tools

| Tool | Purpose |
|------|---------|
| `open_file` | Read file contents |
| `close_file` | Release file handle |
| `overwrite_file` | Complete file replacement |
| `diff_edit` | Diff-based modifications (preserves KV cache) |
| `range_edit` | Line-range based edits |
| `run_shell` | Execute shell commands |
| `run_tests` | Execute test suite |

---

## Utility Function

**Critical:** Agent selection uses utility score, not just accuracy.

```
U = 0.5 × p_score + 0.25 × (1 − min(1, p_cost/$10)) + 0.25 × (1 − min(1, p_time/300s))
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `p_score` | 50% | Task success rate |
| `cost_penalty` | 25% | Penalize expensive agents (>$10 per task) |
| `time_penalty` | 25% | Penalize slow agents (>300s per task) |

**Timeout penalty:** If execution exceeds 300s, final utility = U × 0.5

### Application to Your System

```python
def calculate_agent_utility(result: QualityMetrics, cost: float, time_seconds: float) -> float:
    """Calculate utility score for agent/script selection."""
    score_component = 0.5 * (result.overall_score / 100)
    cost_component = 0.25 * (1 - min(1, cost / 10.0))
    time_component = 0.25 * (1 - min(1, time_seconds / 300.0))

    utility = score_component + cost_component + time_component

    if time_seconds > 300:
        utility *= 0.5  # Timeout penalty

    return utility
```

---

## Reflection Mechanism

### Async Overseer

SICA uses an asynchronous overseer LLM that monitors execution in parallel:

```
┌──────────────────┐     ┌──────────────────┐
│   Main Agent     │────▶│    Overseer      │
│   (executing)    │     │   (monitoring)   │
└──────────────────┘     └──────────────────┘
         │                        │
         │                        ▼
         │               ┌──────────────────┐
         │               │ Analyzes:        │
         │               │ - Call graphs    │
         │               │ - Event streams  │
         │               │ - Behavior       │
         │               │ - Loop detection │
         │               └──────────────────┘
         │                        │
         │◀───────────────────────┘
         │     (notify or cancel)
         ▼
```

### What Overseer Detects

| Pattern | Action |
|---------|--------|
| Repeated work | Notify agent |
| Pathological loops | Cancel execution |
| Deviation from objectives | Notify + suggest correction |
| Resource exhaustion | Hard cancel |

### Application to Your System

Your `EnforcementHooks` already does some of this. Enhance with:

```python
class OverseerHooks(EnforcementHooks):
    """Async overseer that monitors agent behavior patterns."""

    async def on_turn_end(self, context, agent, output):
        # Analyze output for deviation patterns
        if self._detect_deviation(output):
            raise DeviationDetectedError(
                "Agent deviating from objective. Intervention required."
            )

        # Detect repeated reasoning patterns
        if self._detect_reasoning_loop(output):
            await self._inject_correction(context, agent)
```

---

## Self-Modification Process

### How SICA Edits Itself

1. **Archive Review**: Meta-agent examines all previous agent versions
2. **Improvement Identification**: Finds specific weakness to address
3. **Implementation**: Uses diff-based edits to modify agent code
4. **KV Cache Preservation**: Appends file edits as diffs to maintain context

### Key Insight

> "The agent is instructed to look through the archive, identify an improvement, and implement it."

The agent doesn't randomly try things. It:
1. Reviews what worked
2. Reviews what failed
3. Identifies specific improvement
4. Implements targeted change

### Application to Your System

Your `LearningAgent` already tracks experiments. Enhance with:

```python
async def identify_improvement_opportunity(session_history: List[IterationResult]) -> str:
    """Analyze session history to identify specific improvement."""

    # Find patterns in failures
    failures = [r for r in session_history if r.success == False]
    failure_reasons = Counter(f.failure_reason for f in failures)

    # Most common failure
    top_failure = failure_reasons.most_common(1)[0]

    # Query knowledge base for solutions
    solutions = await query_knowledge_base(f"fix {top_failure}")

    return f"Improvement: Address '{top_failure}' using pattern: {solutions[0]}"
```

---

## Performance Results

| Benchmark | Improvement | Notes |
|-----------|-------------|-------|
| SWE Bench Verified (50 problems) | 17% → 53% | Core coding tasks |
| File editing accuracy | 0.82 → 0.96 | By iteration 11 |
| Symbol location | 0.43 accuracy | By iteration 11 |
| LiveCodeBench | Subtle improvement | |
| AIME/GPQA (reasoning) | Marginal | Saturates when base model already good |

### Key Takeaway

Self-improvement works best when:
1. Clear feedback signal exists (tests pass/fail)
2. Agent has tools to modify itself
3. Archive of past attempts is available
4. Utility function balances quality, cost, time

---

## Implementation Recommendations

### For blender-vfx-orchestrator

1. **Track full iteration history** with structured metrics
2. **Calculate utility scores** for each iteration
3. **Meta-analysis phase** before generating new scripts:
   - Review what worked in past sessions
   - Identify patterns in failures
   - Apply lessons learned
4. **Diff-based script modification** to preserve working patterns
5. **Async quality monitoring** during Blender execution
