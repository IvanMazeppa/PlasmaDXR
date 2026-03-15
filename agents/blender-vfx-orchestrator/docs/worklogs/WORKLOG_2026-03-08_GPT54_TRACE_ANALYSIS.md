# Worklog: GPT-5.4 Upgrade, Trace Analysis, and Learning Feedback Loop
**Date:** 2026-03-08
**Branch:** `0.34.19/phase2c`

---

## Context

Glass shatter E2E test exposed systemic issues beyond individual API bugs. The agent generates working scripts but defaults to the same familiar techniques (grid-subdivided rectangular shards for glass fracture) rather than discovering realistic approaches. Three root problems identified:

1. **Context bloat from doc search** — drowns out useful reasoning space
2. **No learning feedback loop** — each run starts fresh, repeats same mistakes
3. **Trace data not utilized** — 100KB+ traces exist but only metrics are extracted, not reasoning

GPT-5.4 was released and Ben upgraded the `codex_upgrade` preset from gpt-5.3-codex to gpt-5.4. OpenAI published detailed prompt guidance docs with patterns that directly address our problems.

---

## What Was Done This Session

### 1. Truth Pack Fixes (3 bugs from glass shatter test)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `RigidBodyObject.linear_velocity` hallucination | Attribute doesn't exist in Blender 5.0. Agent generated code using it. | Added to KNOWN_HALLUCINATIONS. Auto-fixer comments out the line with explanation. |
| `"Color 1"` vs `"Color1"` on ShaderNodeTexBrick | Blender 5.0 removed the space. LLM uses old naming. | Added to KNOWN_HALLUCINATIONS + HARDCODED_FIXES for auto-replacement. |
| Truth pack loaded wrong types for `cell_fracture_rigid_body_mesh` | Technique name wasn't in TECHNIQUE_ALIASES, fell back to `mantaflow_gas` types | Added 4 new aliases: `cell_fracture_rigid_body_mesh`, `cell_fracture_rigid_body`, `cell_fracture`, `voronoi_fracture` → all resolve to `rigid_body` |

Also:
- Added shader node types to `rigid_body` technique type list: `ShaderNodeBsdfPrincipled`, `ShaderNodeMix`, `ShaderNodeOutputMaterial`, `ShaderNodeVolumePrincipled`
- Deleted stale truth pack cache (`5_0_1_cell_fracture_rigid_body_mesh.json`)
- New truth pack generated on next run: `5_0_1_rigid_body.json` (16 types, 819 properties)
- All 325 unit tests pass after changes

### 2. Prompt Revision (glass shatter test)

The original glass shatter prompt was too prescriptive in HARD CONSTRAINTS — it told the agent exactly how to fracture (Cell Fracture addon, manual Voronoi, 200-400 fragments, mesh collision shapes, mass formula). This violated the Prompt Enhancement Guide's principle: hard constraints should only specify simulation system, renderer, samples, frame range, cache_type, reference.

**Before (prescriptive):**
```
- Glass object: Use Cell Fracture addon (or manual Voronoi fracture via geometry) to pre-fracture the window into 200-400 fragments
- Fracture source: Impact point with radial pattern, denser fragmentation near center
- Collision shape: Mesh (not convex hull) for glass fragments
- Fragment mass: Proportional to shard volume, total glass mass ~45kg
```

**After (outcome-focused):**
```
- Physics simulation: Blender Rigid Body
- Bake frames: 1-60
- Renderer: Cycles GPU
- Samples: 256 max
- Render frame: 12
Research documentation, patterns, and APIs to find the best approach for realistic glass shattering simulation.
```

Result: Agent still chose grid-subdivision approach. This confirmed the problem is in the agent's technique discovery, not the prompt.

### 3. Manual Script Fix Validation

Manually fixed the generated script (replaced `linear_velocity` with keyframed displacement, fixed `Color1`/`Color2`) and ran through Blender CLI. Confirmed the overall approach works — rigid body bake completed 60 frames, render produced. But the rectangular shard pattern is visually unrealistic.

### 4. E2E Test with Pipeline Fixes

Re-ran glass shatter test after truth pack fixes. Pipeline ran clean:
- Zero API errors (vs 2 crashes before)
- Truth pack built with correct `rigid_body` types (16 types, 819 properties)
- API Validation: 14/14 valid
- Render produced successfully
- Artifact gate failed: cache size 0.00MB (false positive — rigid body doesn't produce fluid cache)

---

## Trace Analysis Deep Dive

### What's in the traces (96KB JSONL)

| Source | Size | % of trace data |
|--------|------|-----------------|
| `blender_doc_search_bundle` output | 25.6KB | 30% |
| `write_script` output (generated script) | 24.2KB | 28% |
| `semantic_search_blender_docs` x2 | 27.5KB | 32% |
| Everything else | 8.3KB | 10% |

### What's NOT in the traces

The `response` span events only contain `response_id` — the OpenAI SDK doesn't log what agents actually SAID locally. So:
- **CAN see:** Tool inputs/outputs, timings, guardrail results
- **CANNOT see:** Agent reasoning, technique rationale, decision logic, why it chose grid subdivision

The OpenAI dashboard has the reasoning but it's not in local traces.

### Trace analyzer metrics (glass shatter run)

```
Duration: 534.8s (8.91 min)
Script Writer: 279.0s (52.2%) — dominates the pipeline
Research Agent: 48.7s (9.1%)
Technique Selector: 25.0s (4.7%)
LLM Calls: 12, Total: 361.8s, Mean: 30.15s
Doc Searches: 6
Guardrails: 5 checks, 0 triggers
```

---

## GPT-5.4 Model Summary

**Reference docs saved at:**
- `docs/PROMPT_GUIDANCE.md` — OpenAI's official prompt patterns for GPT-5.4
- `docs/USING_GPT_5.4.md` — OpenAI's model guide for GPT-5.4

### Key features relevant to us

| Feature | What it does | Our problem it solves |
|---------|-------------|----------------------|
| `text.verbosity` parameter | Controls output token count at API level | Already configured in presets |
| `phase: "commentary"` / `"final_answer"` | Distinguishes intermediate reasoning from final output | Prevents early stopping + capturable reasoning |
| Preambles | Model states intent before each tool call | Missing reasoning in traces |
| `<empty_result_recovery>` pattern | Retry with alternate queries on empty results | Agent giving up on technique discovery |
| `<research_mode>` pattern | 3-pass: Plan → Retrieve → Synthesize | Agent does one search and calls it done |
| `<output_contract>` blocks | Enforce dense structured output | Reduce verbose agent prose |
| `<verification_loop>` | Check before finalizing | Catch wrong technique choices |
| 1M context window | Huge context capacity | Less pressure but still wasteful |
| Native compaction | API-level session compaction | Replace manual 25K token compaction |
| `allowed_tools` | Restrict tool access per phase | Phase-appropriate tool gating |
| `tool_search` | Deferred tool loading | Reduce token usage for large tool sets |

### Migration notes
- gpt-5.4 is drop-in replacement for gpt-5.2 and gpt-5.3-codex
- Reasoning effort `none` is the DEFAULT — must set explicitly (we already do)
- New `xhigh` reasoning level available (reserve for hardest tasks)
- `temperature`/`top_p`/`logprobs` only work with `reasoning_effort=none`
- `previous_response_id` preserves CoT between turns (important for multi-turn agents)

### Reasoning effort guidance from OpenAI

| Level | Use for |
|-------|---------|
| `none` | Fast execution, tool routing, simple transforms |
| `low` | Latency-sensitive tasks with complex instructions |
| `medium` | Research, synthesis, multi-document review |
| `high` | Complex code generation, strategy, evaluation |
| `xhigh` | Long agentic reasoning-heavy tasks (avoid as default) |

Our current config: script_writer=high, research_agent=high, quality_analyst=high, modification_coordinator=high, technique_coordinator=low, docs_expert=low. This aligns well with OpenAI's guidance.

---

## Root Cause Analysis: Why the Agent Repeats Techniques

### The chain

1. Research Agent searches docs once → gets 25KB of results (mostly Mantaflow fluid docs, wrong physics type)
2. Script Writer receives bloated context, defaults to the approach it's most familiar with
3. No post-run analysis captures "grid subdivision produced rectangular shards"
4. Next run starts fresh, repeats exact same sequence
5. Knowledge base is empty — nothing to learn from

### Why doc search returns wrong results

The `blender_doc_search_bundle` tool queries the vector store with the effect type ("explosion"). For glass shattering, this returns Mantaflow explosion docs — completely irrelevant for rigid body fracture. The semantic_query field helps but the doc search still prioritizes API reference over conceptual/technique content.

### Why the agent can't discover new techniques

1. No `<research_mode>` pattern — doesn't plan sub-questions or follow second-order leads
2. No `<empty_result_recovery>` — when first search returns fluid docs, doesn't retry with "rigid body fracture techniques" or "cell fracture alternatives"
3. Doc search results flood context — 25KB of irrelevant fluid docs leaves less room for actual technique reasoning
4. Knowledge base is empty — no stored patterns from previous successful/failed runs

---

## Action Plan

### Phase 1: Doc Search Output Trimming (Highest Impact)

**Problem:** `blender_doc_search_bundle` returns 25KB per call. This is the single biggest context bloat source.

**Fix:** Cap tool output to ~4KB max. Summarize or truncate results. Prioritize results matching the technique type (rigid_body for glass shatter, not mantaflow).

**Files:** `tools/semantic_docs_tools.py` (search bundle function)

### Phase 2: GPT-5.4 Prompt Patterns for Research Agent

**Problem:** Research Agent does 1 search and synthesizes. Doesn't plan, doesn't retry, doesn't follow leads.

**Fix:** Add to Research Agent system prompt:
- `<research_mode>` — 3-pass: Plan sub-questions → Retrieve each → Synthesize
- `<empty_result_recovery>` — When results are irrelevant/empty, try alternate queries
- `<output_contract>` — Dense structured output, no repetition
- "Before calling a tool, explain why" — Enables preamble capture in traces

**Files:** `specialized_agents/research_agent.py` or wherever system instructions are defined, `orchestrator.py` research phase

### Phase 3: Enable Preamble Capture in Traces

**Problem:** Traces only contain `response_id`, not agent reasoning. Can't learn from traces.

**Fix:**
1. Add preamble instruction to agent system prompts
2. Use `RunHooks.on_agent_end()` to capture final agent output text into trace events
3. Extend trace_analyzer.py to extract preamble + output text into structured "session debrief"

**Files:** `tools/trace_analyzer.py`, `orchestrator.py` (RunHooks)

### Phase 4: Post-Run Debrief → Knowledge Base

**Problem:** Each run starts fresh. No memory of "grid subdivision produced rectangular shards."

**Fix:**
1. After each pipeline run, extract structured debrief:
   - Technique tried
   - What worked / what failed
   - Qualitative assessment
   - Specific error patterns
2. Store in knowledge base as searchable entries
3. Research Agent queries KB BEFORE searching docs: "what failed before with this effect type?"

**Files:** `orchestrator.py` (post-pipeline hook), `tools/experiment_tracker_tools.py`

### Phase 5: Artifact Gate Fix for Rigid Body

**Problem:** Cache size gate fails for rigid body simulations (0.00MB cache — rigid body doesn't produce fluid cache files).

**Fix:** Make cache gate technique-aware. Skip or adjust for non-fluid physics types.

**Files:** `phases/execution.py` or wherever artifact gates are defined

### Phase 6: Update VERSION_TRUTH.md

Add gpt-5.4 to the models table, remove stale gpt-5.3-codex-only references. Update CLAUDE.md model table.

---

## Blender 5.0 Gotchas Added This Session

| Wrong (hallucinated) | Correct | Auto-fixed? |
|---------------------|---------|-------------|
| `RigidBodyObject.linear_velocity = X` | Keyframed displacement while kinematic | Yes — commented out with explanation |
| `RigidBodyObject.angular_velocity = X` | Keyframed rotation while kinematic | Yes — commented out with explanation |
| `inputs['Color 1']` on ShaderNodeTexBrick | `inputs['Color1']` (no space) | Yes — string replacement |
| `inputs['Color 2']` on ShaderNodeTexBrick | `inputs['Color2']` (no space) | Yes — string replacement |

---

## Files Modified This Session

| File | Change |
|------|--------|
| `tools/truth_pack.py` | Added 4 technique aliases, 4 KNOWN_HALLUCINATIONS, 2 HARDCODED_FIXES, shader nodes to rigid_body types, velocity auto-fix logic |
| `test_glass_shatter_codex.py` | Revised prompt (less prescriptive), updated semantic_query, gpt-5.4 references (by user) |
| `assets/blender_scripts/generated/glass_shatter_codex_manual_fix.py` | Manual fix: replaced linear_velocity with keyframed displacement, Color1/Color2 rename |
| `data/truth_packs/5_0_1_rigid_body.json` | New — generated by pipeline with correct rigid body types |

## Test Results

- **325/325 unit tests pass** after truth pack changes
- **Glass shatter E2E (pipeline):** Clean execution, zero API errors, render produced
- **Glass shatter manual fix (CLI):** Blender execution success, 60 frames baked, render produced
