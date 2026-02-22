# SDK Tracing & Actionable Logging Guide (2026-01-23)

Goal: produce **actionable** execution data from the Agents SDK (traces, spans, metadata) so workflow quality and learning outcomes can be measured and debugged.

This guide focuses on **built-in tracing** and **SDK-supported extensibility** for routing trace data to other systems. Official references: [Tracing](https://openai.github.io/openai-agents-python/tracing/), [Tracing Reference Index](https://github.com/openai/openai-agents-python/tree/main/docs/ref/tracing). 

---

## What the SDK already traces (default)
The SDK traces key events automatically during a `Runner.run()`:

- Full run wrapped in a trace
- Agent execution spans
- LLM generations
- Function tool calls
- Guardrails
- Handoffs
- Audio transcription/speech spans (if used)

This gives a full **timeline** and **causal chain** for each run without extra instrumentation.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## High‑value tracing you should add
The SDK gives you core events, but for **actionable data** you need workflow‑specific metadata and custom spans.

### 1) Wrap multi‑step workflows in a single trace
If your pipeline calls `Runner.run()` multiple times per iteration, wrap the whole iteration in `trace()` so it becomes **one trace** with nested spans.

- This prevents “fragmented” traces and makes iteration‑level analysis easy.
- You can set a **workflow name**, **trace_id**, **group_id**, and **metadata** for grouping runs.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

### 2) Use `group_id` to link multi‑iteration sessions
When you run an asset session with many iterations, use a stable `group_id` (e.g., `session_id`) so all traces can be viewed together.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

### 3) Add custom spans for key checkpoints
Use `custom_span()` to capture steps that are not inherently SDK‑tracked, such as:

- Candidate generation phase (beam/population step)
- Learning Agent proposal gate
- Micro‑experiment execution
- Doc‑gating validation
- Escape‑level switch event

Custom spans are nested under the active trace and produce high‑signal boundaries for analysis.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## Actionable metadata (suggested fields)
Add consistent metadata to traces and custom spans to make dashboards and post‑processing reliable.

**Trace metadata (iteration/session level):**
- `effect_type`
- `technique`
- `iteration`
- `escape_level`
- `candidate_count`
- `learning_doc_refs_count`
- `quality_overall_score`
- `quality_passed`

**Span metadata (step level):**
- `step_name` (e.g., `learning_proposals`, `micro_experiment`)
- `candidate_id`
- `api_used` (for new Blender 5 calls)
- `doc_refs[]` (IDs or URLs of Blender 5 docs)

Metadata is explicitly supported on traces/spans.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## Sensitive data controls
The SDK can include **inputs/outputs** of LLMs and tool calls in traces. This may be sensitive.

- Control this via `RunConfig.trace_include_sensitive_data` or env var `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA`.
- Audio data also has its own sensitive‑data toggle if voice is used.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## Extending tracing beyond OpenAI
You can route traces to additional destinations by adding or replacing trace processors:

1) **Add** processors (recommended): keep OpenAI traces + external exporter  
2) **Set** processors (replacement): remove default exporter and take full control

This is the SDK‑supported path for pushing data to tools like Weights & Biases, Langfuse, Braintrust, etc.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## Minimal implementation outline (SDK‑safe)
1) Wrap each iteration in `trace("VFX Iteration", ...)`.
2) Assign `group_id = session_id` and add trace metadata.
3) Add `custom_span()` around:
   - Learning Agent proposal gate
   - Micro‑experiment execution
   - Candidate ranking/selection
4) Configure `RunConfig.trace_include_sensitive_data` based on environment.
5) Add a **secondary trace processor** for external analytics if needed.

All steps above are supported in the official tracing docs.  
Source: [Tracing docs](https://openai.github.io/openai-agents-python/tracing/).

---

## SDK tracing reference index (for deeper setup)
The tracing reference directory includes the formal APIs for:
- setup/config
- trace creation
- spans and span data
- processors and providers
- logger utilities

See: [Tracing Reference Index](https://github.com/openai/openai-agents-python/tree/main/docs/ref/tracing).

