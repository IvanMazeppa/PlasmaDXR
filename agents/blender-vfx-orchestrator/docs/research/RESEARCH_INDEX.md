# Multi-Agent Autonomous Systems Research

**Compiled:** 2026-01-26
**Purpose:** Research findings to improve blender-vfx-orchestrator hallucination prevention and iterative improvement

---

## Executive Summary

This research folder contains findings from academic papers, OpenAI documentation, and industry best practices for building autonomous multi-agent systems that iteratively improve through self-evaluation.

### Key Findings

| Problem | Solution | Document |
|---------|----------|----------|
| API hallucinations | Token-level interception + grounding | `HALLUCINATION_PREVENTION.md` |
| Workflow orchestration | Code-based pipelines with LLM decision points | `MULTI_AGENT_PATTERNS.md` |
| Iterative improvement | Utility functions + self-modification | `SELF_IMPROVING_AGENTS.md` |
| Quality feedback loops | LLM-as-judge + human review | `SELF_EVOLVING_WORKFLOW.md` |
| SDK guardrails | Input/output/tool guardrails | `OPENAI_GUARDRAILS.md` |

---

## Recommended Implementation Priority

### P0: Immediate (Hallucination Prevention)

1. **Tool-level guardrails** - Intercept Blender API calls BEFORE execution
2. **Confidence scoring** - Flag low-confidence API attribute generations
3. **Self-questioning** - Agent verifies its own API calls against docs

### P1: Short-term (Workflow Improvement)

1. **Utility function** for agent selection (like SICA)
2. **Versioned prompts** with rollback capability
3. **Async overseer** to detect pathological loops

### P2: Medium-term (Self-Evolution)

1. **LLM-as-judge evaluation** for script quality
2. **Automatic prompt refinement** based on failure modes
3. **Knowledge distillation** from successful iterations

---

## Document Index

| Document | Topic | Source |
|----------|-------|--------|
| `HALLUCINATION_PREVENTION.md` | Agent hallucination taxonomy and mitigation | arXiv:2509.18970 |
| `SELF_IMPROVING_AGENTS.md` | SICA methodology for self-modification | arXiv:2504.15228 |
| `MULTI_AGENT_PATTERNS.md` | OpenAI SDK orchestration patterns | OpenAI Docs |
| `SELF_EVOLVING_WORKFLOW.md` | Continuous improvement with LLM-as-judge | OpenAI Cookbook |
| `OPENAI_GUARDRAILS.md` | Input/output/tool guardrail implementation | OpenAI SDK Docs |
| `PRACTICAL_HALLUCINATION_FIXES.md` | Developer-focused grounding techniques | Zep Guide |

---

## Sources

- [LLM-based Agents Suffer from Hallucinations: A Survey](https://arxiv.org/html/2509.18970v1)
- [Self-Improving Coding Agent (SICA)](https://arxiv.org/html/2504.15228v1)
- [OpenAI Agents SDK - Multi Agent](https://openai.github.io/openai-agents-python/multi_agent/)
- [OpenAI Agents SDK - Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
- [Self-Evolving Agents Cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)
- [Reducing LLM Hallucinations Developer Guide](https://www.getzep.com/ai-agents/reducing-llm-hallucinations/)
- [Multi-AI Agent System for Autonomous Optimization](https://arxiv.org/abs/2412.17149)
