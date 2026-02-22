# Experimental Codex Tool (Agents SDK) - Why, How, and Where It Fits

**Status:** Experimental surface in the Agents SDK  
**Source:** OpenAI Agents SDK docs (Tools → "Experimental: Codex tool") citeturn9view0  

## Executive Summary

The Agents SDK now exposes an **experimental Codex tool** that lets an agent run **workspace-scoped tasks** (shell commands, file edits, and MCP tools) through the Codex CLI **inside a tool call**. This gives you a controlled, sandboxed way to let an agent take concrete actions in a repo without leaving the Agents SDK model loop. citeturn9view0

For the Blender VFX Orchestrator, this can be a powerful "execution arm" when you want an LLM to:
1) inspect or transform files,  
2) run local commands,  
3) apply patch-style edits,  
4) drive MCP tools in a contained workspace.

Because the API is **experimental**, expect surface changes and keep it behind feature flags. citeturn9view0

---

## How It Works (from the SDK docs)

The **codex tool** is provided by the Agents SDK extension package and **wraps the Codex CLI** so the agent can execute workspace-scoped operations during a tool call. citeturn9view0

Key operational facts:
- **Auth**: `CODEX_API_KEY` preferred, or `OPENAI_API_KEY`, or pass `codex_options={"api_key": "..."}`. citeturn9view0  
- **Inputs schema**: tool calls must include at least one `inputs` item with `{ "type": "text", "text": ... }` or `{ "type": "local_image", "path": ... }`. citeturn9view0  
- **Safety**: pair `sandbox_mode` with `working_directory`; set `skip_git_repo_check=True` when outside Git repos. citeturn9view0  
- **Session reuse**: `persist_session=True` reuses a Codex thread and returns `thread_id`. citeturn9view0  
- **Streaming**: `on_stream` receives Codex events (reasoning, command execution, MCP tool calls, file changes, web search). citeturn9view0  
- **Outputs**: tool returns `response`, `usage`, and `thread_id`, and usage is added to `RunContextWrapper.usage`. citeturn9view0  
- **Structured outputs**: `output_schema` can enforce typed Codex responses. citeturn9view0

---

## Minimal Example (Python)

```python
from agents import Agent
from agents.extensions.experimental.codex import ThreadOptions, codex_tool

agent = Agent(
    name="Codex Agent",
    instructions="Use the codex tool to inspect the workspace and answer the question.",
    tools=[
        codex_tool(
            sandbox_mode="workspace-write",
            working_directory="/path/to/repo",
            default_thread_options=ThreadOptions(
                model="gpt-5.2-codex",
                network_access_enabled=True,
                web_search_enabled=False,
            ),
            persist_session=True,
        )
    ],
)
```
citeturn9view0

---

## Why This Is Useful in Blender VFX Orchestrator

The orchestrator already runs scripts, evaluates renders, and iterates. The Codex tool can **augment** those phases when you need a **workspace-aware agent** that can:

1) **Audit and patch generated scripts**
   - Example: open the script, replace deprecated Blender API calls, write back fixes.
   - Advantage: keeps edits scoped to the repo with sandbox boundaries.

2) **Investigate execution failures**
   - Example: parse logs, grep for errors, inspect file outputs, or annotate run dirs.

3) **Generate auxiliary assets**
   - Example: create test configs, generate small helper scripts, or assemble a report.

4) **Drive MCP tooling in a single tool call**
   - Codex CLI can use MCP tools within its action loop, giving the agent a richer "toolbox" in a controlled context. citeturn9view0

5) **Unified audit trail**
   - Streaming events (`on_stream`) can be piped into your tracing or logs for visibility. citeturn9view0

---

## Integration Patterns to Consider

### Pattern A: Dedicated Codex Utility Agent (Recommended)
- Add a **CodexAgent** as a tool callable by the pipeline coordinator.
- Use it only when:
  - script fixes are needed, or
  - logs indicate repeated errors, or
  - file surgery is safer than re-running generation.

### Pattern B: Codex Tool on the Coordinator
- Mount `codex_tool(...)` directly on the Coordinator for on-demand workspace operations.
- Keep it behind a **feature flag** (`ENABLE_CODEX_TOOL`) and **rate-limit calls**.

### Pattern C: Codex Tool for Recovery Paths
- Only invoke Codex when the standard pipeline fails (e.g., executor failures 2+ times).
- Benefits: reduces cost and avoids overusing experimental surface.

---

## Safety and Governance Guardrails

Given this is experimental, treat it as a **privileged tool**:
- **Always scope** `working_directory` to a specific repo or subdir. citeturn9view0  
- **Disable network access by default** unless explicitly required.  
- **Require explicit activation** (feature flag + allowlist of tasks).  
- **Record thread_id** to make Codex sessions auditable and resumable. citeturn9view0

---

## Suggested Next Steps (No Code Changes Yet)

1) Add a design note in the orchestrator docs: when to invoke Codex vs. standard Script Writer.  
2) Decide on sandbox policy and working_directory scope.  
3) Create a small internal PoC harness (outside the main pipeline) to validate:
   - command execution,
   - file patching,
   - streaming event logging.

---

## References

- Agents SDK Tools: Experimental Codex tool (Python) citeturn9view0  
- Codex tool semantics and configuration options (auth, inputs schema, safety, streaming, outputs) citeturn9view0

