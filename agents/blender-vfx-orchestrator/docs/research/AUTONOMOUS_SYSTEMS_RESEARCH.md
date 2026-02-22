# Autonomous Systems Research Report

**Date:** 2026-02-22
**Researcher:** Claude Opus 4.6 (Autonomous Systems Researcher)
**Purpose:** Actionable recommendations for self-learning, autonomous operation, and multi-agent coordination in the blender-vfx-orchestrator

---

## Executive Summary

This report synthesizes research from 20+ sources (papers, production systems, official docs) into concrete recommendations for the VFX orchestrator. Findings are ranked by **impact** (how much they improve reliability/quality) and **complexity** (implementation effort).

**Top 5 recommendations by impact-to-effort ratio:**

| # | Pattern | Impact | Complexity | Status |
|---|---------|--------|------------|--------|
| 1 | Stateless iteration + file memory (Ralph) | CRITICAL | Medium | Not implemented |
| 2 | Artifact-based context sharing (Anthropic) | HIGH | Low | Partially implemented |
| 3 | Maker-checker with iteration caps (Azure) | HIGH | Low | Partially implemented |
| 4 | Multi-grader evaluation (OpenAI cookbook) | HIGH | Medium | Partially implemented |
| 5 | Metacognitive monitoring layer (MASC) | HIGH | Medium | Not implemented |

---

## 1. Self-Learning in Multi-Agent Systems

### 1.1 Learning From Own Outputs

**Source:** [Self-Evolving Agents Cookbook (OpenAI)](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining) | [Awesome Self-Evolving Agents Survey](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents)

The OpenAI self-evolving agents cookbook implements a complete autonomous improvement cycle:

1. **Baseline agent** generates output
2. **Multi-grader evaluation** captures quality signals (4 complementary graders)
3. **Metaprompt optimizer** refines instructions based on grader failures
4. **Improved agent** replaces original
5. Cycle repeats

Key implementation details:
- **Prompt versioning system**: Each prompt version is tracked with version number, model, timestamp, eval IDs, and metadata. Enables rollback if new versions regress.
- **Aggregate scoring**: Per-version statistics accumulate across ALL test sections (not just the current one). Final prompt selection maximizes `(total_score, version)` tuple.
- **Caching**: Eval results cached by `(section, summary)` tuple to prevent redundant grader execution.
- **Retry limits**: 3 attempts per section prevent infinite loops. Manual alert triggers when max retries exceeded.

**Applicability to VFX orchestrator:** HIGH. The system already has quality evaluation and iteration. The missing piece is **systematic prompt/instruction evolution** — capturing which system prompts, dynamic instructions, and research patterns produce better scripts, then automatically promoting the best variants.

**Recommendation:** Implement prompt versioning for the script writer agent's system prompt. Track which instruction variants correlate with higher quality scores across multiple runs. Auto-promote variants that consistently score above threshold.

### 1.2 Evidence-Gating Patterns

**Source:** [Self-Evolving Agents Survey (arXiv:2508.07407)](https://arxiv.org/abs/2508.07407)

The orchestrator already implements 4-tier evidence gating (Untrusted → Emerging → Trusted → Deprecated). This aligns with production patterns in the literature:

- **Single occurrence = analytics only** (matches Untrusted)
- **Repeated success = promoted** (matches Emerging → Trusted)
- **Repeated failure = demoted** (matches Deprecated)

**What's missing:** The current implementation doesn't capture the **context** of success/failure. A pattern that works for fire but fails for liquid gets a blended score that misleads both use cases.

**Recommendation:** Add **effect-type scoping** to evidence gating. A pattern's trust level should be tracked per effect type (fire, liquid, rigid body, etc.), not globally. Implementation: add `effect_type` field to knowledge base entries, filter by it during dynamic instruction injection.

### 1.3 Memory Decay (Ebbinghaus/SAGE)

**Source:** [SAGE: Self-evolving Agents (arXiv:2409.00872)](https://arxiv.org/html/2409.00872v1) | [The Agent's Memory Dilemma (Medium, Nov 2025)](https://tao-hpu.medium.com/the-agents-memory-dilemma-is-forgetting-a-bug-or-a-feature-a7e8421793d4)

SAGE implements the Ebbinghaus forgetting curve for agent memory:

```
R(I, τ) = e^(-τ/S)
```

Where `τ` = elapsed time, `S` = information strength (importance × reinforcement count).

**Two-tier memory system:**
- **Short-term memory (Ms)**: High volatility, limited capacity. Information decays fast unless reinforced.
- **Long-term memory (Ml)**: Slow decay, larger capacity. Information transfers here only if repeatedly reinforced above threshold θ1.
- **Forgetting threshold θ2**: Below this, information is deleted entirely.

**Practical implementation for VFX orchestrator:**

```python
# Decay function for KB entries
import math
from datetime import datetime

def compute_retention(entry, now):
    days_since_last_success = (now - entry.last_reinforced).days
    strength = entry.success_count * entry.avg_quality_score / 100
    retention = math.exp(-days_since_last_success / max(strength, 0.1))
    return retention

# Apply during KB queries
def get_relevant_patterns(effect_type, threshold=0.3):
    entries = kb.query(effect_type=effect_type)
    now = datetime.now()
    return [e for e in entries if compute_retention(e, now) >= threshold]
```

**Status:** Partially implemented (evidence gating exists, but no time-based decay). The KB was recently wiped (2026-02-22), so this is a good time to implement decay before it accumulates stale data again.

**Recommendation:** Implement exponential decay with reinforcement. Each successful use of a pattern resets its decay clock and increases its strength. Entries not reinforced within 30 days begin to decay. Entries below retention threshold 0.1 are archived (not deleted — kept for analytics).

### 1.4 Knowledge Distillation From Successful Runs

**Source:** [Self-Evolving Agents Survey](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents)

The survey identifies three key mechanisms for learning from outputs:

1. **Feedback loops**: Test outputs against execution results (already implemented)
2. **Process rewards**: Learning from intermediate reasoning steps (not implemented)
3. **Preference optimization**: Training on self-generated high-quality trajectories (not applicable without fine-tuning)

**Missing in VFX orchestrator:** The system records what happened but doesn't systematically extract **reusable code patterns** from successful scripts.

**Recommendation:** After a run scores >= 60, extract key code sections (lighting setup, material definitions, camera placement, physics config) as named patterns with metadata (effect_type, quality_score, parameter ranges). These patterns become available to the script writer via dynamic instructions, evidence-gated by the standard promotion pipeline.

### 1.5 Preventing Knowledge Base Poisoning

**Source:** [OWASP Top 10 for LLM Apps — Agentic AI Threats (2025)](https://www.aigl.blog/content/files/2025/04/Agentic-AI---Threats-and-Mitigations.pdf) | [Knowledge-Based Poisoning Attacks research](https://www.emergentmind.com/topics/knowledge-based-poisoning-attacks)

Research shows that just **5 carefully crafted documents can manipulate AI responses 90% of the time** through RAG poisoning. For the VFX orchestrator, the risk isn't malicious actors but **self-poisoning** — the system learning bad patterns from its own failures.

**Prevention mechanisms (ranked by priority):**

1. **Evidence gating (already implemented)**: Nothing enters trusted knowledge without repeated success.
2. **Source attribution**: Every KB entry must trace back to a specific session, iteration, and quality score. Entries without provenance are untrusted.
3. **Outcome validation**: Before a pattern enters the KB, verify that applying it actually improves quality (A/B comparison, not just correlation).
4. **Periodic audit**: Scheduled review of KB entries against current Blender truth pack — any entry referencing deprecated attributes gets auto-archived.
5. **Separation of learning from execution**: The learning agent should analyze outcomes AFTER the pipeline completes, not during iteration (prevents mid-run contamination).

**Status:** Items 1 and 5 are partially implemented. Items 2-4 are not.

**Recommendation:** Implement source attribution (item 2) and truth-pack-based audit (item 4) as immediate priorities. Both are low-cost deterministic operations.

---

## 2. Autonomous Operation Patterns

### 2.1 Stateless Iteration + File Memory (Ralph Pattern)

**Source:** [Ralph (GitHub)](https://github.com/snarktank/ralph) | [Ralph Loop analysis (Alibaba Cloud)](https://www.alibabacloud.com/blog/from-react-to-ralph-loop-a-continuous-iteration-paradigm-for-ai-agents_602799) | [Ralph Wiggum analysis (Medium)](https://medium.com/@tentenco/what-is-ralph-loop-a-new-era-of-autonomous-coding-96a4bb3e2ac8)

**This is the single highest-impact pattern for the VFX orchestrator.**

Ralph's core insight: **statelessness is a feature, not a limitation.** Each iteration spawns a fresh AI instance with clean context. Memory persists through files:

- **prd.json** (Task Registry): Stories with completion status. Maps to our session state.
- **progress.txt** (Learning Journal): Append-only file capturing discoveries. Maps to our iteration learnings.
- **Git history** (Execution Artifacts): Previous implementations. Maps to our script versions.

**Why this is critical for the VFX orchestrator:**

The orchestrator's #1 reliability problem is **context degradation across iterations**. By iteration 3-4, the context window is polluted with verbose evaluation output, failed script fragments, and stale research. The Ralph pattern eliminates this entirely.

**Proposed implementation:**

Each iteration gets a fresh agent invocation. Between iterations, the orchestrator writes a structured state file:

```json
{
  "session_id": "session_20260222_fire_001",
  "iteration": 3,
  "effect_type": "fire",
  "current_script_path": "output/fire_001/iter3/script.py",
  "current_score": 45,
  "score_history": [0, 28, 45],
  "critical_issues": ["LIGHTING_TOO_DIM"],
  "technique": "mantaflow_gas",
  "escape_level": 1,
  "learnings": [
    "Light energy 500 too low for fire scenes — need 2000+",
    "Domain resolution_max 128 produces visible voxels — use 200+",
    "Camera at (0, -8, 3) clips domain boundary"
  ],
  "qa_diagnosis": {
    "symptoms": ["render too dark", "flame barely visible"],
    "root_causes": ["area_light energy=500 at line 342", "no emission shader on flame material"],
    "suggested_fixes": ["increase light energy to 2000-3000", "add emission shader with strength 5-10"]
  },
  "truth_pack_types": ["FluidDomainSettings", "FluidFlowSettings", "ShaderNodeEmission"]
}
```

The next iteration reads ONLY this file plus the current script. No accumulated conversation history. Fresh context, informed by structured state.

**Compatibility with OpenAI Agents SDK:** The SDK's `call_model_input_filter` can implement this by clearing conversation history between iterations, injecting only the state file contents. Alternatively, each iteration can be a new `Runner.run()` call with fresh context.

**Expected impact:** Eliminates context degradation, reduces token cost per iteration by 60-80%, and prevents the "iteration 4+ quality cliff" observed in current runs.

**Complexity:** Medium. Requires restructuring the iteration loop to be stateless, serializing state to file between iterations.

### 2.2 Artifact-Based Information Sharing (Anthropic Pattern)

**Source:** [How Anthropic Built Their Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) | [Effective Context Engineering (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

Anthropic's multi-agent research system achieves 90.2% improvement over single-agent systems. Key patterns:

1. **Agents store full results in files, pass only lightweight summaries** (~1000-2000 tokens) to the coordinator. Context grows by 1 line instead of 50 paragraphs.
2. **3-5 subagents run in parallel**, each with clean context windows.
3. **Structured note-taking**: Agents write notes to persistent files, retrieve them later. This enables progress tracking across complex multi-step tasks.
4. **Extended thinking integration**: Lead agents use extended thinking for planning; subagents use it after tool results to evaluate quality.

**Key quote from Anthropic:** "Token usage alone explains 80% of variance in performance."

**Applicability to VFX orchestrator:** DIRECT. The quality analyst's verbose output is a known context bloat problem. Research output similarly bloats context.

**Implementation:**

```python
# Instead of returning verbose evaluation inline:
evaluation_result = {
    "score": 45,
    "critical_issues": ["LIGHTING_TOO_DIM"],
    "summary": "Fire visible but underlit. Area light energy too low. No emission shader.",
    "full_report_path": "output/fire_001/iter3/evaluation.json"
}

# Instead of returning full research inline:
research_result = {
    "technique": "mantaflow_gas",
    "key_findings": ["Use burning_rate 0.3-0.8 for candle flames", "resolution_max >= 200 for detail"],
    "full_report_path": "output/fire_001/research.json"
}
```

**Status:** Partially implemented (the interview mentions artifact-based sharing was attempted). Needs to be made systematic across ALL agents.

**Recommendation:** Mandate that every agent returns a structured summary (< 500 tokens) with a file path to the full report. The orchestrator never receives full reports inline.

### 2.3 Dynamic Task Ledger (Microsoft Magentic-One)

**Source:** [Magentic-One paper (arXiv:2411.04468)](https://arxiv.org/html/2411.04468v1) | [Azure Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)

Magentic-One's architecture uses two interlocking loops:

**Outer Loop (Task Ledger):**
- Maintains the overall plan, verified facts, and educated guesses
- Gets updated when new information emerges
- When revised, **all agents clear their contexts and reset states**

**Inner Loop (Progress Ledger):**
The Orchestrator answers 5 critical questions each iteration:
1. Is the request fully satisfied (task complete)?
2. Is the team looping or repeating?
3. Is forward progress occurring?
4. Which agent should speak next?
5. What instruction should be given?

**Stall detection:** A counter tracks how long the team has been stuck. Below threshold (≤2): continue. Above threshold: re-enter outer loop with reflection — identify what went wrong, document new learnings, revise plan, start fresh.

**Applicability to VFX orchestrator:** HIGH. The orchestrator currently follows a fixed state machine (PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE). Magentic-One's approach would make this adaptive — if evaluation reveals the technique is fundamentally wrong, the system can jump back to research/technique selection rather than trying to modify a doomed script.

**Recommendation:** Implement the 5-question progress ledger as a lightweight check between iterations. This can be a deterministic function (no LLM needed) that checks:
1. Score >= threshold? → Done
2. Same error 2+ times? → Looping
3. Score improving? → Progress
4. Based on diagnosis, which agent should act? → Route
5. What specific instruction? → Construct from state

**Complexity:** Low for the 5-question check (deterministic). Medium for full adaptive replanning.

### 2.4 Metacognitive Monitoring (MASC)

**Source:** [MASC paper (arXiv:2510.14319)](https://arxiv.org/html/2510.14319v1)

MASC provides **real-time, unsupervised step-level error detection** in multi-agent systems. When an anomaly is detected, a correction agent intervenes BEFORE the error propagates downstream.

**Technical mechanism:**
1. **Next-Execution Reconstruction (NER)**: Predicts what the next step's output SHOULD look like based on history. Large reconstruction error = anomaly.
2. **Prototype-Guided Enhancement**: Maintains a learned "prototype" of normal agent behavior. Deviations from prototype = suspicious.
3. **Self-Correction Loop**: When anomaly score exceeds threshold, a dedicated correction agent revises the flagged output before downstream agents receive it.

**Key finding:** "A single faulty step can propagate across agents and disrupt the trajectory." MASC achieves 77.84% AUC-ROC on step-level error detection and up to 8.47% improvement over supervised baselines.

**Applicability to VFX orchestrator:** This is exactly the **monitoring agent** Ben described in the interview. The orchestrator's known failure modes — quality analyst giving wrong advice, parameter oscillation, stuck loops — are precisely what MASC detects.

**Practical implementation (simplified for VFX orchestrator):**

Full MASC requires training an anomaly detector on normal trajectories. A lightweight alternative for the VFX orchestrator:

```python
class PipelineMonitor:
    """Deterministic monitoring layer — no LLM needed."""

    def check_iteration(self, state: SessionState) -> MonitoringSignal:
        signals = []

        # Stuck loop detection
        if state.same_error_count >= 2:
            signals.append(("STUCK_LOOP", f"Same error {state.same_error_count}x"))

        # Score oscillation detection
        if len(state.score_history) >= 3:
            diffs = [state.score_history[i+1] - state.score_history[i]
                     for i in range(len(state.score_history)-1)]
            if all(abs(d) > 10 for d in diffs[-2:]) and diffs[-1] * diffs[-2] < 0:
                signals.append(("OSCILLATION", f"Scores oscillating: {state.score_history[-3:]}"))

        # Parameter oscillation detection
        if state.param_history:
            for param, values in state.param_history.items():
                if len(values) >= 3 and is_oscillating(values[-3:]):
                    signals.append(("PARAM_OSCILLATION", f"{param}: {values[-3:]}"))

        # Context bloat detection
        if state.context_tokens > 25000:
            signals.append(("CONTEXT_BLOAT", f"{state.context_tokens} tokens"))

        # Quality analyst contradiction detection
        if state.qa_suggests_increase and state.prev_qa_suggested_decrease:
            signals.append(("QA_CONTRADICTION", "QA flip-flopping on same parameter"))

        return MonitoringSignal(signals=signals, should_intervene=len(signals) > 0)
```

**This is entirely deterministic — $0 cost.** It detects the exact failure modes observed in production (oscillation, stuck loops, context bloat, contradictory QA advice).

**Recommendation:** Implement the deterministic monitoring layer FIRST. Evaluate whether LLM-based anomaly detection (full MASC) is needed later based on what the deterministic layer misses.

### 2.5 Maker-Checker With Iteration Caps

**Source:** [Azure Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)

The maker-checker pattern formalizes the ScriptWriter (maker) / QualityGate (checker) loop:

**Azure's definition:**
> "An iteration cap is used to prevent infinite refinement loops combined with a fallback behavior for when the cap is reached, such as escalating to a human reviewer or returning the best result with a quality warning."

**Key requirements:**
- Clear acceptance criteria for the checker (quality score >= 60, no critical issues)
- Formal turn-based sequence driven by the orchestrator
- **Fallback behavior at cap**: escalate to human, return best result with warning, or switch technique

**Status in VFX orchestrator:** Partially implemented. The system has iteration caps and quality thresholds but lacks formalized fallback behavior. When max iterations are reached, it simply stops — no structured escalation.

**Recommendation:** Implement tiered fallback at iteration cap:
1. **Cap reached, score >= 40**: Return best result with quality warning + improvement suggestions
2. **Cap reached, score < 40**: Switch technique (escape velocity L2+) and try N more iterations
3. **Cap reached after technique switch, score < 40**: Escalate to human with full diagnostic report
4. **Any iteration**: If monitoring layer detects STUCK_LOOP, trigger technique switch immediately (don't wait for cap)

---

## 3. Multi-Agent Coordination

### 3.1 Agents-as-Tools vs Handoffs vs Parallel Execution

**Source:** [OpenAI Agents SDK Multi-Agent docs](https://openai.github.io/openai-agents-python/multi_agent/) | [Azure Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns) | [AWS Strands Multi-Agent Collaboration](https://aws.amazon.com/blogs/machine-learning/multi-agent-collaboration-patterns-with-strands-agents-and-amazon-nova/)

**Decision matrix for the VFX orchestrator:**

| Pattern | Use When | VFX Orchestrator Application |
|---------|----------|------------------------------|
| **Agents-as-tools** | Orchestrator needs result back; sub-agent does focused work | Research agent, DocsExpert, API Validator — called by orchestrator, return structured results |
| **Handoffs** | Full control transfer; next agent determined dynamically | **DEPRECATED in codebase** — replaced by agents-as-tools. Correct decision. |
| **Parallel execution** | Independent subtasks that don't depend on each other | Research + truth pack generation can run in parallel. Quality eval (ML metrics + vision) can run in parallel. |

**Key insight from Azure:** "Decision-making and flow-control overhead often exceed the benefits of breaking the task into multiple agents." The VFX orchestrator has ~10 agents. Some should be **functions, not agents**.

**Agents that should REMAIN LLM-powered:**
- Research Agent (creative reasoning needed)
- Script Writer (core creative value)
- Quality Analyst vision component (semantic understanding)
- Modification Strategist (complex diagnosis)

**Agents that should become DETERMINISTIC functions:**
- Executor (subprocess management — no reasoning needed)
- API Validator (pattern matching against truth pack)
- Budget tracker (arithmetic)
- Quality threshold comparison (comparison logic)

**Recommendation:** Audit the agent roster. Convert deterministic agents to function_tools. This reduces token cost and latency without losing capability.

### 3.2 Minimizing Token Waste

**Source:** [Stop Wasting Your Tokens (arXiv:2510.26585)](https://arxiv.org/html/2510.26585v1) | [Effective Context Engineering (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

SupervisorAgent achieves **29.68% token reduction** through three techniques:

1. **Adaptive Observation Purification**: When tool outputs exceed 3,000 characters, compress them. The VFX orchestrator's Blender stdout/stderr and doc search results often exceed this.
2. **Proactive Error Correction**: Detect failures immediately and intervene before they cascade. Maps to the monitoring layer.
3. **Inefficiency Guidance**: Detect repetitive loops or unnecessarily complex paths and redirect.

**Anthropic's finding:** Using context editing + memory tools, agents completed 100-turn dialogues using **only 16% of tokens** they would have otherwise consumed.

**Context-Folding (FoldGRPO)**: Replaces intermediate steps with concise summaries. Matches or surpasses baselines with **10x smaller active context**.

**Practical token reduction strategy for VFX orchestrator:**

| Source of Bloat | Current Cost | Reduction Technique | Expected Savings |
|-----------------|-------------|---------------------|-----------------|
| Quality eval output | ~2000 tokens | Artifact-based (file + summary) | 80% |
| Research output | ~3000 tokens | Artifact-based (file + summary) | 80% |
| Blender stdout/stderr | ~1000-5000 tokens | Truncate to last 50 lines + error summary | 70% |
| Doc search results | ~2000 tokens | Return top 3 results, truncate to 200 chars each | 75% |
| Previous iteration context | ~5000+ tokens | Ralph pattern (fresh context + state file) | 90% |
| **Total per iteration** | **~13,000+ tokens** | **Combined** | **~75% reduction** |

### 3.3 Long-Running Autonomous Pipelines

**Source:** [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) | [Context Window Management (blog.jroddev.com)](https://blog.jroddev.com/context-window-management-in-agentic-systems/)

Key patterns for long-running pipelines (VFX orchestrator iterations take 5-30 min each):

1. **Checkpoint and resume**: Save full state at each phase boundary. If the pipeline fails at iteration 3, resume from iteration 3's state — don't restart from scratch.
2. **Progressive disclosure**: Don't pre-load all context. Retrieve what's needed for the current phase.
3. **Sublinear context growth**: Use AgentFold-style techniques to keep context < 7K tokens even after 100 turns.
4. **Rainbow deployments**: Keep old and new agent versions running simultaneously during updates. New iterations use new agents; in-progress iterations continue with old agents.

**Status in VFX orchestrator:** Checkpoint exists (session_state.json). Progressive disclosure is partially implemented. Sublinear context growth is NOT implemented.

**Recommendation:** Combine Ralph pattern (stateless iterations) with checkpoint/resume. Each iteration starts fresh but can resume from any checkpoint.

---

## 4. Experimentation and Exploration

### 4.1 Sandbox Mode for Capability Discovery

**Source:** [Self-Evolving AI Survey (KAD)](https://www.kad8.com/ai/why-self-evolving-ai-will-define-2026/) | Mission Statement Section 8

The mission statement describes sandbox mode as:
- Run small experimental scripts to test techniques in isolation
- Introspect Blender capabilities
- Record findings
- Feed findings into the knowledge base

**Implementation approach — Micro-experiments:**

Rather than a separate "sandbox mode," integrate micro-experiments into the pipeline:

```python
async def explore_technique(effect_type: str, technique: str) -> ExperimentResult:
    """Run a minimal Blender script to test a technique.

    This is NOT a full scene — it's a 50-100 line script that tests
    ONE specific thing (e.g., 'does cloth simulation work with this mesh?').
    Execution time: 10-30 seconds. Cost: $0 (no LLM, just Blender subprocess).
    """
    script = generate_micro_experiment(effect_type, technique)
    result = await execute_in_blender(script, timeout=30)
    return ExperimentResult(
        technique=technique,
        success=result.exit_code == 0,
        output=result.stdout[-500:],  # Last 500 chars only
        errors=parse_errors(result.stderr),
        discovered_attributes=extract_used_attributes(script)
    )
```

**Triggers for exploration:**
1. Novel prompt that doesn't match known techniques
2. Escape velocity level 2+ (stuck)
3. Between production runs (idle time)
4. User-requested ("experiment with cloth simulation")
5. New truth pack version detected (Blender update)

### 4.2 Exploitation vs Exploration Balance

**Source:** [Multi-Armed Bandits Meet LLMs (arXiv:2505.13355, IBM/AAAI 2026)](https://arxiv.org/html/2505.13355v1) | [Multi-Agent Multi-Armed Bandits research](https://www.emergentmind.com/topics/multi-agent-multi-armed-bandits-ma-mab)

The VFX orchestrator's technique selection is a classic **multi-armed bandit problem**: given an effect type, which technique (arm) should we try to maximize quality (reward)?

**UCB1 (Upper Confidence Bound) for technique selection:**

```python
import math

def select_technique(effect_type: str, available_techniques: list) -> str:
    """UCB1-based technique selection balancing exploitation and exploration."""
    total_runs = sum(t.run_count for t in available_techniques)

    best_score = -float('inf')
    best_technique = None

    for technique in available_techniques:
        if technique.run_count == 0:
            return technique.name  # Always try untried techniques first

        # Exploitation: average quality score
        exploitation = technique.avg_score

        # Exploration bonus: decreases as technique is tried more
        exploration = math.sqrt(2 * math.log(total_runs) / technique.run_count)

        ucb_score = exploitation + exploration * 10  # Scale exploration by 10 (scores are 0-100)

        if ucb_score > best_score:
            best_score = ucb_score
            best_technique = technique.name

    return best_technique
```

**Key insight from IBM/AAAI 2026 paper:** "Instead of using fixed exploration heuristics like epsilon-greedy or UCB, LLMs can analyze historical data to dynamically suggest exploration rates based on environmental changes." This means the research agent could ITSELF decide when to explore vs exploit, informed by the KB.

**Recommendation:** Implement UCB1 as the default technique selector. Override with LLM-based selection only when the research agent identifies a specific reason to deviate (novel prompt, new technique discovered, etc.).

### 4.3 Turning Experiments Into Trusted Knowledge

**Source:** [OpenAI Self-Evolving Agents Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining)

The cookbook's promotion logic:
- **Lenient passes** update `best_candidate` tracker
- **All attempts** contribute to `aggregate_prompt_stats`
- Final selection maximizes cumulative score across ALL test cases

**Applied to VFX orchestrator experiments:**

```
Experiment run → Record result → If score >= 40 (lenient pass):
  → Mark as "emerging" in KB with single evidence point
  → Next time this technique is needed, it appears in technique selection
  → If used in production and scores >= 60:
    → Increment evidence count
    → After 3 successful production uses: promote to "trusted"
  → If used in production and scores < 30:
    → Increment failure count
    → After 3 failures: demote to "deprecated"
```

**Key principle:** Experiments lower the bar for entry (score >= 40 vs production's 60), but promotion to trusted still requires production-grade evidence.

---

## 5. Quality and Evaluation

### 5.1 Multi-Grader Evaluation Architecture

**Source:** [OpenAI Self-Evolving Agents Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining)

The cookbook uses 4 complementary graders:

| Grader | Type | What It Checks | Pass Threshold |
|--------|------|----------------|----------------|
| Chemical name | **Deterministic (Python)** | Critical entities preserved | 0.8 |
| Length deviation | **Deterministic (Python)** | Output within acceptable bounds | 0.85 |
| Cosine similarity | **ML metric** | Semantic anchoring to source | 0.85 |
| LLM-as-judge | **LLM reasoning** | Nuanced quality assessment | 0.85 |

**Passage criteria:** 75% of graders must pass (binary) OR average score >= 0.85.

**Applied to VFX orchestrator — proposed 4-grader system:**

| Grader | Type | What It Checks | Implementation |
|--------|------|----------------|----------------|
| **Critical issue detector** | Deterministic | BLACK_SCREEN, WHITE_SCREEN, ZERO_LIGHTS, CLIPPING | Image histogram analysis — $0 |
| **Script quality checker** | Deterministic | Script length >= 500 lines, has lighting, has camera, has materials | AST/regex analysis — $0 |
| **ML quality metrics** | ML | CLIP score, LPIPS, TOPIQ | Existing implementation — ~$0.01 |
| **Vision analyst** | LLM | Composition, aesthetic quality, correctness | OpenAI vision — ~$0.05 |

**Passage criteria for VFX:** ALL deterministic graders must pass (hard gates). ML + LLM graders provide the quality score. Critical issues from deterministic graders auto-fail regardless of ML/LLM score.

**Key improvement:** Run deterministic graders BEFORE expensive ML/LLM graders. If script is < 500 lines or image is black, skip the $0.05 vision call entirely.

**Recommendation:** Implement the 4-grader pipeline with early termination. This saves budget on obviously failed renders while providing nuanced evaluation for borderline cases.

### 5.2 Pairing Visual Critique With Code Analysis (QA Diagnosis Bridge)

**Source:** Mission Statement Section 4 (Quality Evaluation) | [MASC paper](https://arxiv.org/html/2510.14319v1)

**The problem:** The vision analyst sees symptoms ("too dark") but not causes ("missing light at line 342"). Its suggestions target symptoms, and when the script writer follows symptom-based advice, it often makes things worse.

**The solution (already partially implemented as QA Diagnosis Bridge):**

The QA bridge pairs visual critique with script code. Extend this to be systematic:

```python
def create_diagnosis(visual_critique: str, script_path: str) -> Diagnosis:
    """Pair visual symptoms with code-level root causes."""

    symptoms = parse_symptoms(visual_critique)  # "too dark", "flame not visible"
    code_analysis = analyze_script(script_path)  # Light positions, energies, materials

    diagnosis = []
    for symptom in symptoms:
        if symptom == "TOO_DARK":
            # Check lights in script
            lights = code_analysis.find_lights()
            if not lights:
                diagnosis.append(RootCause("NO_LIGHTS", "No light objects in scene"))
            elif max(l.energy for l in lights) < 500:
                diagnosis.append(RootCause("LOW_LIGHT_ENERGY",
                    f"Max light energy {max(l.energy for l in lights)}, need 1000+",
                    line_numbers=[l.line_num for l in lights]))
        # ... more symptom → cause mappings

    return Diagnosis(symptoms=symptoms, root_causes=diagnosis)
```

**This is deterministic script analysis — $0 cost.** It replaces the current pattern where the LLM guesses at causes.

### 5.3 Preventing Evaluation Feedback Loops

**Source:** [Closing the Feedback Loop (Maxim AI)](https://www.getmaxim.ai/articles/closing-the-feedback-loop-how-evaluation-metrics-prevent-ai-agent-failures/) | [Security Degradation in Iterative AI Code (IEEE-ISTAS 2025)](https://arxiv.org/html/2506.11022) | [7 Tips for Self-Improving AI Agents (Datagrid)](https://datagrid.com/blog/7-tips-build-self-improving-ai-agents-feedback-loops)

**The danger:** Research shows a **37.6% increase in critical vulnerabilities after just 5 iterations** of LLM code improvement. The VFX orchestrator is vulnerable to the same pattern — each iteration potentially degrades the script if the feedback loop is broken.

**Prevention strategies (ranked):**

1. **Separate reflection from execution**: The learning agent should analyze outcomes AFTER the pipeline, not during. During iteration, only structured state (scores, errors, root causes) should inform the next iteration.

2. **Never promote unreviewed LLM-generated content to training data**: The quality analyst's suggestions should NEVER directly enter the knowledge base. Only outcomes (score went up/down) should be recorded as evidence.

3. **Monotonic quality requirement**: If score decreases between iterations, flag this as a regression. The monitoring layer should detect and intervene when 2 consecutive iterations show decreasing scores.

4. **Parameter clamping**: Prevent the script writer from making extreme parameter changes based on critique. Implement deterministic bounds:
   ```python
   PARAM_BOUNDS = {
       "light_energy": (100, 10000),  # Never below 100, never above 10000
       "resolution_max": (64, 512),
       "flame_vorticity": (0.0, 1.0),
   }
   ```

5. **A/B comparison**: Before applying a modification, compare the proposed change against the original. If the modification would reverse a change that improved quality in the previous iteration, flag it.

---

## 6. Specific Systems Deep Dives

### 6.1 Ralph — Key Takeaways

**Source:** [GitHub](https://github.com/snarktank/ralph) | [DeepWiki analysis](https://deepwiki.com/snarktank/ralph)

| Ralph Feature | VFX Orchestrator Equivalent | Gap |
|--------------|---------------------------|-----|
| prd.json (task registry) | session_state.json | Need to restructure for stateless iteration |
| progress.txt (learning journal) | Knowledge base | Need append-only iteration learnings file |
| AGENTS.md (knowledge base) | Dynamic instructions | Already implemented, needs refinement |
| Git history (artifacts) | Script versions | Already implemented |
| Conditional commit (only on pass) | N/A | Useful — only save script to KB if quality improved |
| Granular story sizing | N/A | Each iteration should be a single, focused task |
| Completion signal | Quality threshold pass | Already implemented |

**Most important Ralph lesson for VFX orchestrator:** "Statelessness actually becomes a strength — each fresh context prevents the model from compounding errors or getting stuck in bad patterns."

### 6.2 Magentic-One — Key Takeaways

**Source:** [Paper](https://arxiv.org/html/2411.04468v1) | [Azure docs](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)

| Magentic-One Feature | VFX Orchestrator Equivalent | Gap |
|---------------------|---------------------------|-----|
| Task ledger (outer loop) | Pipeline state machine | Need adaptive replanning |
| Progress ledger (inner loop) | Iteration state | Need 5-question monitoring check |
| Stall counter | Escape velocity levels | Already implemented, well-aligned |
| Context reset on replan | N/A | Implement via Ralph pattern |
| 4 specialized agents | 10 agents | Reduce to essential creative agents |

**Most important Magentic-One lesson:** "All agents clear their contexts and reset their states" when the plan is revised. Combine with Ralph pattern for clean iteration boundaries.

### 6.3 MASC — Key Takeaways

**Source:** [Paper](https://arxiv.org/html/2510.14319v1)

| MASC Feature | VFX Orchestrator Equivalent | Gap |
|-------------|---------------------------|-----|
| Anomaly detection | N/A | Implement deterministic monitoring first |
| Correction agent | Modification strategist | Need to ensure corrections are applied BEFORE downstream propagation |
| Cascading failure prevention | N/A | CRITICAL gap — one bad eval poisons subsequent iterations |
| Architecture-agnostic integration | Plugin-compatible | Good — can add to existing pipeline |

**Most important MASC lesson:** "A single faulty step can propagate across agents and disrupt the trajectory." The VFX orchestrator's broken QA feedback loop is EXACTLY this failure mode.

---

## 7. Implementation Priority Matrix

### Phase 2A: Quick Wins (1-2 weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|-------------|
| Artifact-based sharing for ALL agents | HIGH | Low | None |
| Deterministic monitoring layer | HIGH | Low | None |
| 4-grader evaluation with early termination | HIGH | Low | None |
| Script analysis for QA diagnosis bridge | HIGH | Low | Existing QA bridge |
| Parameter clamping bounds | MEDIUM | Low | None |

### Phase 2B: Core Architecture (2-4 weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|-------------|
| Ralph-style stateless iterations | CRITICAL | Medium | Artifact-based sharing |
| 5-question progress ledger | HIGH | Medium | Monitoring layer |
| Ebbinghaus memory decay | MEDIUM | Medium | KB wipe (done) |
| Effect-type-scoped evidence gating | MEDIUM | Medium | Memory decay |
| Agent roster audit (convert deterministic agents to functions) | MEDIUM | Medium | None |

### Phase 2C: Advanced Capabilities (4-8 weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|-------------|
| UCB1 technique selector | HIGH | Medium | Evidence gating |
| Micro-experiment sandbox mode | HIGH | Medium | Technique selector |
| Prompt versioning for script writer | MEDIUM | Medium | Multi-grader eval |
| Adaptive replanning (Magentic-One style) | MEDIUM | High | Progress ledger |
| Knowledge distillation from successful scripts | MEDIUM | Medium | Evidence gating |

### Phase 2D: Autonomy (8+ weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|-------------|
| Full self-evolving prompt optimization | HIGH | High | All Phase 2A-C |
| Autonomy progression tracking | MEDIUM | Medium | Evidence gating |
| LLM-based anomaly detection (full MASC) | MEDIUM | High | Deterministic monitoring |
| Cross-session learning transfer | MEDIUM | High | Memory decay + distillation |

---

## 8. Sources

### Primary Sources (directly fetched and analyzed)

1. [Ralph — Autonomous AI Agent Loop](https://github.com/snarktank/ralph)
2. [MASC — Metacognitive Self-Correction for Multi-Agent Systems (arXiv:2510.14319)](https://arxiv.org/html/2510.14319v1)
3. [Magentic-One — Generalist Multi-Agent System (arXiv:2411.04468)](https://arxiv.org/html/2411.04468v1)
4. [OpenAI Self-Evolving Agents Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining)
5. [SAGE — Self-evolving Agents with Reflective and Memory-augmented Abilities (arXiv:2409.00872)](https://arxiv.org/html/2409.00872v1)
6. [Anthropic — How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
7. [Anthropic — Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
8. [Azure — AI Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
9. [Stop Wasting Your Tokens — SupervisorAgent (arXiv:2510.26585)](https://arxiv.org/html/2510.26585v1)
10. [Comprehensive Survey of Self-Evolving AI Agents (arXiv:2508.07407)](https://arxiv.org/abs/2508.07407)

### Secondary Sources (search results informing analysis)

11. [Ralph Loop Analysis (Alibaba Cloud)](https://www.alibabacloud.com/blog/from-react-to-ralph-loop-a-continuous-iteration-paradigm-for-ai-agents_602799)
12. [Multi-Armed Bandits Meet LLMs (IBM/AAAI 2026)](https://arxiv.org/html/2505.13355v1)
13. [Security Degradation in Iterative AI Code (IEEE-ISTAS 2025)](https://arxiv.org/html/2506.11022)
14. [AWS — Evaluating AI Agents at Amazon](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/)
15. [7 Tips for Self-Improving AI Agents (Datagrid)](https://datagrid.com/blog/7-tips-build-self-improving-ai-agents-feedback-loops)
16. [Closing the Feedback Loop (Maxim AI)](https://www.getmaxim.ai/articles/closing-the-feedback-loop-how-evaluation-metrics-prevent-ai-agent-failures/)
17. [OWASP Top 10 for LLM Apps — Agentic AI Threats (2025)](https://www.aigl.blog/content/files/2025/04/Agentic-AI---Threats-and-Mitigations.pdf)
18. [EvoAgentX — Self-Evolving Agents Survey Repository](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents)
19. [OpenAI Agents SDK — Multi-Agent Orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
20. [AWS Strands — Multi-Agent Collaboration Patterns](https://aws.amazon.com/blogs/machine-learning/multi-agent-collaboration-patterns-with-strands-agents-and-amazon-nova/)
21. [Google ADK — Multi-Agent Patterns](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/)
22. [Supervising Ralph — Principal Skinner Pattern](https://securetrajectories.substack.com/p/ralph-wiggum-principal-skinner-agent-reliability)
