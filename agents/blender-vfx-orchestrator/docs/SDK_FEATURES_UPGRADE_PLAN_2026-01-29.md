# SDK Features Upgrade Plan (2026-01-29)

This document records the **nine SDK-driven improvement opportunities** (excluding visualization) discovered in the OpenAI Agents SDK documentation, plus concrete implementation guidance tailored to the Blender VFX Orchestrator.

## 1) Parallel Sub‑Agents (Fan‑out/Fan‑in)
**What it is:** Use `asyncio.gather` to run multiple agents concurrently when tasks are independent.

**Why it matters:** Your pipeline is rigid and linear; parallel preflight reduces latency and increases robustness (docs + patterns + validation can be concurrent).

**SDK reference:** Multi‑agent orchestration and parallel runs are explicitly supported via code orchestration. ([openai.github.io](https://openai.github.io/openai-agents-python/multi_agent/))

**How to apply:**
- Run ResearchAgent and DocsExpert in parallel at PHASE 0.
- Optionally fan‑out PatternSearch / SpecValidator for later iterations.
- Merge results into a single “facts + constraints” bundle before Script Writer.

**Success criteria:**
- Both outputs appear in the research summary.
- Orchestrator proceeds only after parallel tasks finish.

---

## 2) Global RunConfig Enforcement
**What it is:** Centralized run configuration for guardrails, tracing metadata, and input filters.

**Why it matters:** Prevents per‑agent drift in settings and ensures consistent tracing/grouping.

**SDK reference:** `RunConfig` described in `running_agents.md` (global guardrails, tracing, filters). ([openai.github.io](https://openai.github.io/openai-agents-python/running_agents/))

**How to apply:**
- Build a `RunConfig` per run (`workflow_name`, `group_id`, `trace_metadata`).
- Pass to every `Runner.run(...)`.

**Success criteria:**
- All runs share the same `workflow_name` and `group_id`.
- Trace UI groups runs by session ID.

---

## 3) Local Context via RunContextWrapper / ToolContext
**What it is:** Inject local state into tools and hooks without exposing it to the LLM.

**Why it matters:** Lets tools access run‑level thresholds/paths without prompt bloat.

**SDK reference:** Context objects are passed into tools via `RunContextWrapper` / `ToolContext`. ([openai.github.io](https://openai.github.io/openai-agents-python/context/))

**How to apply:**
- Define a `RunContext` data object for run metadata (session id, cache thresholds, output dir).
- Pass it as `context=` in `Runner.run`.

**Success criteria:**
- Tools use context instead of asking the LLM for paths or thresholds.

---

## 4) Tool Guardrails for Critical Tools
**What it is:** Guardrails that validate tool inputs/outputs before execution.

**Why it matters:** Prevents invalid executor calls and halts unsafe operations early.

**SDK reference:** Guardrails can wrap tool inputs/outputs and tripwire invalid calls. ([openai.github.io](https://openai.github.io/openai-agents-python/guardrails/))

**How to apply:**
- Guard `execute_blender_script` to require output/cache dirs.
- Guard `validate_and_fix_script` to reject forbidden or unverified APIs.

**Success criteria:**
- Bad tool calls are rejected with clear diagnostics.

---

## 5) Streaming for Long‑Running Phases
**What it is:** `Runner.run_streamed()` provides a stream of tool/agent events.

**Why it matters:** Long Blender runs need progress reporting and better UX.

**SDK reference:** `RunResultStreaming` + `stream_events()` in `streaming.md`. ([openai.github.io](https://openai.github.io/openai-agents-python/streaming/))

**How to apply:**
- Use streaming for Executor phase.
- Emit progress to logs + run manifest.

**Success criteria:**
- Long bakes render progress without silence.

---

## 6) RunResult‑Based Gating
**What it is:** Use `RunResult.new_items` and guardrail outputs to gate decisions.

**Why it matters:** Enforces mechanical correctness (e.g., did docs tool run?).

**SDK reference:** `RunResult` includes tool outputs and guardrail info. ([openai.github.io](https://openai.github.io/openai-agents-python/results/))

**How to apply:**
- Verify expected tool calls in `new_items`.
- Fail early if guardrail tripwire triggered.

**Success criteria:**
- Gates rely on structured run evidence, not free‑form text.

---

## 7) Conversation Strategy (Sessions vs Server‑Managed)
**What it is:** Local Sessions or server‑managed conversation IDs.

**Why it matters:** Stabilizes long‑lived runs and reduces manual history stitching.

**SDK reference:** `running_agents.md` describes session handling and server‑managed convo IDs. ([openai.github.io](https://openai.github.io/openai-agents-python/running_agents/))

**How to apply:**
- Keep SQLiteSession for now.
- Add optional feature flag for server‑managed conversations if needed.

**Success criteria:**
- Session continuity is preserved across long runs.

---

## 8) Tracing + Metadata Discipline
**What it is:** Proper use of `trace()` and RunConfig metadata.

**Why it matters:** Enables run grouping and fast debugging.

**SDK reference:** `tracing.md` + `running_agents.md`. ([openai.github.io](https://openai.github.io/openai-agents-python/tracing/))

**How to apply:**
- One outer `trace()` per pipeline run.
- Use `group_id=session_id` and attach phase/iteration metadata.

**Success criteria:**
- Trace tool shows a single grouped run with nested spans.

---

## 9) Durable Long‑Running Workflows
**What it is:** External orchestration for durable execution (Temporal / Restate).

**Why it matters:** Blender runs are long and sometimes fail mid‑run; durability supports resumability.

**SDK reference:** `running_agents.md` mentions Temporal / Restate for durable workflows. ([openai.github.io](https://openai.github.io/openai-agents-python/running_agents/))

**How to apply:**
- Wrap execution in a durable workflow engine for production deployments.
- Maintain local mode as default for fast iteration.

**Success criteria:**
- Long runs can resume after interruption.

