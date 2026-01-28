# AI-to-AI Communication Research for Multi-Agent Pipelines

**Date:** 2026-01-28
**Context:** VFX Orchestrator — 8-agent pipeline using OpenAI Agents SDK v0.7.0
**Status:** Research findings applied to agent configuration

---

## Core Finding

**Structured, concise output between AI agents is definitively superior to verbose natural language prose.** The research is unambiguous across multiple independent sources.

---

## Key Evidence

### 1. Structured Output Beats Prose for Agent-to-Agent Communication

The **CodeAgents framework** demonstrates that replacing verbose NL dialogue with structured pseudocode templates achieves **higher accuracy AND lower token cost**. All components of agent interaction — Task, Plan, Feedback, system roles, and external tool invocations — are codified into modular pseudocode enriched with control structures (loops, conditionals), boolean logic, and typed variables. This enhances interpretability, reduces token overhead, and improves traceability.

> Source: [CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs](https://arxiv.org/html/2507.03254v1)

### 2. Multi-Agent Systems Burn 15x More Tokens Than Chat

Agents use approximately **4x more tokens** than chat interactions. Multi-agent systems use approximately **15x more tokens** than chats. Most of this overhead is inter-agent communication — agents explaining themselves in prose to other agents. Standard chain-of-thought solutions often use significantly more tokens than necessary.

> Source: [Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](https://arxiv.org/html/2510.26585v1)

### 3. Anthropic Uses "Lightweight References, Not Full Outputs"

Anthropic's own multi-agent research system recommends that specialized agents **store structured outputs independently** and pass back only lightweight references to the main agent. This:
- Reduces token overhead from copying large outputs through conversation history
- Prevents information loss during multi-stage processing
- Works particularly well for structured outputs like code, reports, or data visualizations

> Source: [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

### 4. Natural Language Causes Semantic Redundancy and Information Loss

This paper directly argues that natural language between agents introduces:
- **Semantic redundancy** — agents re-explaining things already stated
- **Ambiguity in interpretation** — one agent misreading another's prose
- **Lossy information compression** — important details dropped in paraphrase
- **Difficulty maintaining inter-agent state consistency**

Critical design question posed: "How can we retain natural language as a human-facing debugging interface while designing structured communication protocols?"

> Source: [Why do AI agents communicate in human language?](https://arxiv.org/html/2506.02739v1)

### 5. State Delta Encoding: A Middle Ground

Rather than pure prose or pure structure, "State Delta Encoding" augments natural language tokens with the difference between hidden states of adjacent tokens. This bridges the gap between surface-level communication and latent reasoning. Natural language communication may introduce information loss due to sampling, leading to incorrect claims being transferred.

> Source: [Augmenting Multi-Agent Communication with State Delta Trajectory](https://aclanthology.org/2025.emnlp-main.518.pdf) (EMNLP 2025)

### 6. Supervision Cuts Token Waste by 70%

The **SupervisorAgent** approach adds an oversight layer using a lightweight, LLM-free adaptive filter that evaluates each step before deciding if supervision is warranted. This cuts steps by 43% and token cost by **over 70%**.

> Source: [Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](https://arxiv.org/html/2510.26585v1)

### 7. OpenAI Cookbook: Structured Outputs for Multi-Agent Chaining

OpenAI's own cookbook demonstrates using `output_type` with Pydantic models for deterministic, structured agent-to-agent communication. When chaining agents, structured outputs allow code to inspect intermediate results (e.g., classification categories) and route to the next agent programmatically.

> Source: [Structured Outputs for Multi-Agent Systems | OpenAI Cookbook](https://cookbook.openai.com/examples/structured_outputs_multi_agent)

---

## Data Format Efficiency

| Format | Relative Token Cost | Best For |
|--------|-------------------|----------|
| CSV | **40-50% less** than JSON | Tabular data |
| Flattened JSON | **69% less** than nested JSON | Structured inter-agent payloads |
| Lightweight references | **Minimal** | Large outputs (code, renders, reports) |
| Full verbose JSON | Baseline | Human debugging only |

> Source: [A Guide to Token-Efficient Data Prep for LLM Workloads](https://thenewstack.io/a-guide-to-token-efficient-data-prep-for-llm-workloads/)

---

## Communication Topology Research

| Topology | Trade-off |
|----------|-----------|
| **Sequential (pipeline)** | Simple, but information degrades along the chain |
| **Hierarchical (coordinator)** | Good control, but coordinator becomes bottleneck |
| **Peer-to-peer** | Maximum info flow, but communication overhead explodes |
| **Graph-based** | Maximum flexibility, edge-controlled communication |
| **One-shot topology** (TopoDIM) | Replaces iterative multi-round dialogues, significantly reduces token consumption |

> Source: [A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges](https://link.springer.com/article/10.1007/s44336-024-00009-2)

---

## Best Practices Summary

| Practice | Benefit |
|----------|---------|
| Use structured/Pydantic output between agents | Higher accuracy + lower token cost |
| Pass lightweight references, not full outputs | Reduces copying overhead |
| Flatten JSON, eliminate unnecessary nesting | 40-69% token reduction |
| Summarize/compact inter-iteration handoffs | Preserves key info while compressing |
| Add supervision/filter layers | Up to 70% cost reduction |
| Use separate context windows per agent role | Prevents context overflow |
| Set verbosity: medium/low for agent-to-agent | Reduces prose padding |
| Set verbosity: high ONLY for human-facing output | Detailed diagnostics where needed |

---

## Applied Configuration (VFX Orchestrator)

Based on these findings, our agent configuration uses:

- **Structured Pydantic output** (`ScriptOutput`, `QualityOutput`, `LearningOutput`, `QualityDecision`) for all inter-agent communication
- **`verbosity: medium`** for agent-to-agent handoffs (no verbose prose)
- **`verbosity: high`** ONLY for Quality Analyst (human-readable diagnostic detail)
- **`OpenAIResponsesCompactionSession`** for session history compaction between iterations
- **Code-based pipeline orchestration** (not handoff chains) to maintain programmatic control over agent routing

---

## Future Optimizations to Explore

1. **Cache-to-Cache (C2C)** — Direct semantic communication between LLMs using internal KV-cache, bypassing text generation entirely. Would be the most efficient but requires deep SDK integration.

2. **SupervisorAgent pattern** — Lightweight LLM-free filter before each agent call to skip unnecessary invocations. Could prevent wasted calls when the pipeline is clearly converging or diverging.

3. **Artifact-based communication** — Following Anthropic's pattern, have agents write to a shared filesystem and pass paths instead of embedding full output in conversation history.

4. **Per-agent session isolation** — Instead of one shared session, give each agent role its own session so they only see their own historical context.

---

## Additional Reading

- [Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems](https://arxiv.org/html/2502.14321v1)
- [Which LLM Multi-Agent Protocol to Choose?](https://arxiv.org/pdf/2510.17149)
- [MegaAgent: A Large-Scale Autonomous LLM-based Multi-Agent System](https://aclanthology.org/2025.findings-acl.259.pdf)
- [LLM-Based Multi-Agent Systems for Software Engineering](https://dl.acm.org/doi/10.1145/3712003)
- [The Five Ws of Multi-Agent Communication](https://www.techrxiv.org/users/960752/articles/1333905/master/file/data/MA_COMM_TMLR_submit_techrxiv/MA_COMM_TMLR_submit_techrxiv.pdf)
- [Multi-Agent Portfolio Collaboration with OpenAI Agents SDK](https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration)
- [Context Engineering in LLM-Based Agents](https://jtanruan.medium.com/context-engineering-in-llm-based-agents-d670d6b439bc)
