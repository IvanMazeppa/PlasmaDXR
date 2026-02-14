# OpenAI Evals Integration Proposal

**Date:** 2026-01-25
**Status:** Research Complete - Implementation Ready
**Related:** SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md
**Freshness Note:** Historical proposal snapshot; verify OpenAI Evals/API details against current OpenAI docs before implementation.

---

## Executive Summary

OpenAI provides a comprehensive evaluation platform that can help **measure and prevent regressions** in Script Writer behavior through:
1. **Trace Grading (dashboard)** - Workflow‑level inspection of tool calls and decisions
2. **Custom Graders (Evals API)** - Validate Blender API attributes and script correctness
3. **Automated Evals** - Repeatable regression testing for multi-agent workflows

**Important:** Evals are asynchronous and **not a runtime gate**. Runtime enforcement must still use
RunHooks + guardrails (Agents SDK). Evals provide offline measurement and regression protection.

---

## Key Findings from OpenAI Evals Documentation

### 1. Agent Evals Architecture (per OpenAI docs)

OpenAI's eval tooling includes:

| Feature | Purpose | Our Use Case |
|---------|---------|--------------|
| **Trace Grading** | Workflow‑level inspection in dashboard | Verify doc query behavior and tool ordering |
| **Datasets** | Build/iterate evals | Build Blender API validation datasets |
| **Evals API** | Programmatic eval runs | Regression testing for Script Writer output |
| **Prompt Optimizer** | Dataset‑driven prompt tuning | Improve Script Writer prompts after fixes |

### 2. Grader Types Available (per graders guide)

```
┌────────────────────┬─────────────────────────────────────────────────────┐
│ Grader Type        │ Best For                                            │
├────────────────────┼─────────────────────────────────────────────────────┤
│ String Check       │ Exact attribute name verification                   │
│ Text Similarity    │ Soft matching of expected tokens                    │
│ Score Model        │ LLM‑based code quality scoring                      │
│ Python Execution   │ Custom Blender API validation logic                 │
└────────────────────┴─────────────────────────────────────────────────────┘
```

### 3. Trace-Level Visibility

From the docs:
> "Unlike black-box evaluations, trace evals provide more data to better understand WHY an agent succeeds or fails."

This is exactly what we need for monitoring whether Script Writer is:
- Querying documentation before generating attributes
- Using correct attribute names from doc results
- Ignoring doc results and hallucinating anyway

**Note:** Trace grading is configured in the OpenAI dashboard; it is not a runtime gate.

---

## Proposed Integration: Eval-Informed Script Generation

### Architecture Overview

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 0: Research Agent (existing)                         │
│  Output: ResearchOutput schema with doc_refs                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Script Writer Agent                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  TRACE GRADER (dashboard): doc_query_before_attribute │  │
│  │  - Evaluates: Did agent query docs before writing?    │  │
│  │  - Scoring: 1.0 if query found, 0.0 if missing        │  │
│  │  - Action: Record failures for triage/regression      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  PYTHON GRADER: validate_blender_attributes           │  │
│  │  - Evaluates: Every bpy.types attribute in script     │  │
│  │  - Checks against: Blender 5.0 API whitelist          │  │
│  │  - Action: Fail eval if unknown attribute detected    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1.5: API Validator (existing, enhanced)              │
│  Runtime enforcement via RunHooks + guardrails              │
│  Eval results inform prompt/validator updates               │
└─────────────────────────────────────────────────────────────┘
```

### Implementation: Python Grader for Blender API

```python
# graders/blender_api_grader.py

def grade(sample: dict, item: dict) -> float:
    """
    Evaluates Script Writer output for valid Blender 5.0 API usage.

    Args:
        sample: {"output_text": "<generated script>", "output_tools": [...]}
        item: {"valid_domain_attrs": [...], "valid_flow_attrs": [...], "valid_ops": [...]}
    """
    import re

    script = sample.get("output_text", "")

    # Extract attribute assignments using enforced variable names
    domain_pattern = r'\bdset\.(\w+)\s*='
    flow_pattern = r'\bfset\.(\w+)\s*='
    domain_attrs = set(re.findall(domain_pattern, script))
    flow_attrs = set(re.findall(flow_pattern, script))

    # Extract bpy.ops calls
    ops_pattern = r'\bbpy\.ops\.[a-zA-Z_]+\.[a-zA-Z_]+'
    used_ops = set(re.findall(ops_pattern, script))

    # Allowed sets provided by dataset or embedded API index
    valid_domain = set(item.get("valid_domain_attrs", []))
    valid_flow = set(item.get("valid_flow_attrs", []))
    valid_ops = set(item.get("valid_ops", []))

    invalid = (domain_attrs - valid_domain) | (flow_attrs - valid_flow) | (used_ops - valid_ops)
    total = len(domain_attrs) + len(flow_attrs) + len(used_ops)
    if total == 0:
        return 1.0
    return (total - len(invalid)) / total
```

**Note:** Python graders run in a sandbox (no network, 2‑minute runtime, limited disk/mem).
If you embed an API index, keep it small or load it from the dataset item.

### Trace Grading for Doc Query Verification (Dashboard)

Trace grading is configured in the **OpenAI dashboard** and evaluates agent traces directly.
Use it to verify that doc tools are called **before** `write_script` in the Script Writer trace.

**Note:** The docs describe trace grading as a dashboard workflow; there is no public API
for submitting traces programmatically in the evals guide. Treat trace grading as
monitoring/QA, not a runtime gate.

### Implementation: Eval Creation API

```python
# evals/script_writer_eval.py
from openai import OpenAI

client = OpenAI()

eval_obj = client.evals.create(
    name="ScriptWriter_BlenderAPI_Validation",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "effect_type": {"type": "string"},
                "valid_domain_attrs": {"type": "array", "items": {"type": "string"}},
                "valid_flow_attrs": {"type": "array", "items": {"type": "string"}},
                "valid_ops": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["effect_type", "valid_domain_attrs", "valid_flow_attrs", "valid_ops"],
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "python",
            "name": "blender_api_validation",
            "image_tag": "2025-05-08",
            "source": BLENDER_API_GRADER_SOURCE,
            "pass_threshold": 1.0,
        },
        {
            "type": "score_model",
            "name": "code_quality_assessment",
            # NOTE: score_model supports only specific models (no gpt-5.*)
            "model": "o4-mini-2025-04-16",
            "input": [
                {"role": "system", "content": "You are a Blender Python expert. Score 1.0 for fully correct Blender 5.0 API usage, else lower."},
                {"role": "user", "content": "Script: {{ sample.output_text }}\nEffect: {{ item.effect_type }}"}
            ],
            "range": [0, 1],
            "pass_threshold": 0.8,
            "sampling_params": {
                "max_completions_tokens": 1000,
                "reasoning_effort": "medium"
            }
        }
    ],
)

print(eval_obj)
```

### Create an eval run (from file dataset)

```python
from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("blender_eval_items.jsonl", "rb"),
    purpose="evals"
)

run = client.evals.runs.create(
    eval_obj.id,
    name="ScriptWriter regression run",
    data_source={
        "type": "responses",
        "model": "gpt-5-mini",
        "input_messages": {
            "type": "template",
            "template": [
                {"role": "developer", "content": "Generate Blender 5.0 script for {{ item.effect_type }}. Use only valid API."},
                {"role": "user", "content": "{{ item.effect_type }}"},
            ],
        },
        "source": {"type": "file_id", "id": file.id},
    },
)
print(run)
```

---

## Integration with Existing Architecture

### Option A: Offline Eval (Per-Script Regression)

Run evals **after** a run completes (async), not inline. The Evals API is
asynchronous and uses dataset files; it is not a runtime guardrail.

```python
# In orchestrator.py Phase 1

script_result = await Runner.run(script_writer_agent, prompt)
# Persist outputs for evaluation dataset
append_to_eval_dataset(script_result, effect_type=session.effect_type)
# Run evals asynchronously via /v1/evals/{eval_id}/runs
```

### Option B: Trace Grading (Dashboard Monitoring)

Use trace grading in the dashboard to inspect doc‑query behavior and tool ordering.

```python
# In hooks/enforcement_hooks.py

# Trace grading is configured in the OpenAI dashboard (no public API in docs).
# Use it to verify doc query ordering and tool behavior.
```

### Option C: Dataset-Driven Eval (Batch Testing)

Build a dataset of known-good scripts for regression testing:

```json
{ "item": { "effect_type": "explosion", "valid_domain_attrs": ["resolution_max"], "valid_flow_attrs": ["flow_type"], "valid_ops": ["bpy.ops.fluid.bake_data"] } }
{ "item": { "effect_type": "smoke_plume", "valid_domain_attrs": ["use_adaptive_timesteps"], "valid_flow_attrs": ["density"], "valid_ops": ["bpy.ops.fluid.bake_data"] } }
```

---

## Quick Integration Checklist

1. Define dataset schema in `data_source_config` and create JSONL items.
2. Build a small golden dataset (known‑good attributes + ops per effect).
3. Create eval with Python + score_model graders.
4. Upload dataset file with `purpose="evals"`.
5. Run evals via `/v1/evals/{eval_id}/runs` (async).
6. Review results + trace grading in dashboard.
7. Feed findings back into prompts/guardrails/docs.

---

## Recommended Implementation Path

### Phase 1: Python Grader (Immediate)
- Create `blender_api_grader.py` (Python grader) and dataset schema
- Validate grader via graders API
- **Timeline:** Can implement today

### Phase 2: Trace Grading (Short-term)
- Enable trace grading in the dashboard for doc‑query ordering
- Monitor Script Writer behavior over time (workflow‑level)
- **Timeline:** After Phase 1 working

### Phase 3: Full Eval Integration (Medium-term)
- Create formal eval via API
- Build golden dataset for regression testing
- Run evals in CI or on schedule (async)
- **Timeline:** When hallucination rate drops below 10%

---

## Key Insights for Our Problem

### Why This Helps (but doesn’t replace guardrails)

1. **Regression protection:** Evals quantify improvement and detect backsliding.
2. **Trace visibility:** Trace grading shows whether doc tools were used.
3. **Whitelist validation:** Python grader catches invalid attributes in batch.
4. **Continuous improvement:** Datasets grow over time with successful scripts.

**Runtime protection still comes from Spec‑First + RunHooks + guardrails.**

### What We Were Missing

Our current architecture:
- API Validator: Blacklist-based (only catches known-bad)
- validate_script: Syntax-only (doesn't check API accuracy)
- RunHooks: Can enforce tool calls but can't validate results

OpenAI Evals adds:
- **Whitelist validation:** Only allow known-good attributes
- **Trace grading:** Verify doc queries actually happened
- **LLM-as-judge:** Model-based quality assessment

---

## SDK Compatibility

Evals run on the OpenAI Platform (not inside the Agents SDK), and integrate via API calls.
Runtime enforcement still uses Agents SDK guardrails/RunHooks.

All proposed solutions are compatible with OpenAI Agents SDK v0.7.0:

| Feature | SDK Support |
|---------|-------------|
| Runtime enforcement | RunHooks + guardrails |
| Eval API calls | Standard requests or openai SDK |
| Custom graders | Via `/v1/evals` API |

---

## Next Steps

1. **Implement Python grader + dataset schema**
2. **Create eval via API** and run an initial batch
3. **Build golden dataset** from successful script runs
4. **Enable trace grading** in the dashboard
5. **Schedule eval runs** (CI or cron)

This shifts us from ad‑hoc debugging to repeatable regressions + trace‑level visibility.

---

## References

- Agent evals: https://platform.openai.com/docs/guides/agent-evals
- Trace grading: https://platform.openai.com/docs/guides/trace-grading
- Working with evals: https://platform.openai.com/docs/guides/evals
- Graders: https://platform.openai.com/docs/guides/graders
- Evals API: https://platform.openai.com/docs/api-reference/evals
