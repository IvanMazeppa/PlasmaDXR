# NVIDIA-Style Multi-Agent RAG Workflow
_Created_: 2025-11-14  
_Author_: GPT-5.1 Codex (Cursor)

This document extends `docs/CELESTIAL_AGENT_ECOSYSTEM_PLAN.md` with the concrete autonomous-agent hierarchy, division of labour, and Claude Agent SDK implementation details needed to stand up the NVIDIA-style multi-agent RAG workflow described in `docs/research/NVIDIA_Log_Analysis_Multi-Agent_RAG.pdf` and `CLAUDE.md`.

---

## 1. Guiding Principles
1. **Mission control first** – All requests flow through a single orchestration layer ("mission-control") that reads/writes the shared backlog, enforces dependencies, and resolves tool conflicts. This agent runs as an in-process SDK MCP server, exposing tools such as `dispatch_plan`, `record_decision`, `publish_status`, and `handoff_to_agent`.
2. **Specialize by responsibility, aggregate by pipeline** – Specialists remain narrow (e.g., RayQuery, RTXDI, ReSTIR, Probe Grid), but each pipeline has a mid-level coordinator (“council lead”) to prevent “too many cooks”.
3. **Persistent memory > session memory** – Mission control writes every decision, artifact path, and open question into `docs/SESSION_SUMMARY_<date>.md` + the RAG store so context survives token resets.
4. **Evidence-driven loop** – Every change produces: (a) code diff, (b) screenshot/metric bundle, (c) log/PIX ingestion entry. QA and diagnostics agents compare against baselines before sign-off.
5. **Claude Agent SDK native** – Prefer SDK MCP servers (`create_sdk_mcp_server`) for orchestration/logistics, reserving external stdio servers for GPU/ML heavy lifting (image QA, PIX parsing, pyro simulators).

---

## 2. Agent Hierarchy Overview
```
Strategic Tier
├─ mission-control (orchestrator; SDK in-process)
├─ knowledge-steward (context manager + RAG curator; SDK in-process)
└─ backlog-scribe (auto-updates NEXT_STEPS.md, docs/SESSION_SUMMARY_*.md)

Systems Tier (council leads)
├─ rendering-council
│  ├─ rt-lighting-architect (RayQuery lighting)
│  ├─ shadow-systems-engineer (PCSS → DXR inline)
│  ├─ rtxdi-director (M5/M6)
│  ├─ restir-specialist (legacy ReSTIR + research)
│  └─ probe-grid-engineer (ambient volumes)
├─ materials-council
│  ├─ gaussian-analyzer (structure/perf)
│  ├─ material-system-engineer (file ops & codegen)
│  └─ volumetric-pyro-specialist (temporal pyro specs)
├─ physics-council
│  ├─ pinn-integration-agent
│  └─ particle-physics-shader-agent
└─ diagnostics-council
   ├─ dxr-image-quality-analyst (LPIPS + grading)
   ├─ log-analysis-rag (logs + PIX embeddings)
   ├─ pix-debugger-vX (buffer validation)
   └─ metrics-runner (performance scripts, telemetry)

Operations Tier
├─ particle-pipeline-runner (build/run/screenshot automation)
├─ config-curator (JSON/scenario authoring)
├─ asset-metadata-agent (collects camera poses, runtime controls, tags)
└─ critique-agent (processes QA feedback, raises blocking issues)
```

### Division of labour guidelines
- **Granularity:** Use one agent per *debugging domain* (RayQuery shadows, RTXDI, ReSTIR, Probe Grid) because each subsystem has unique telemetry + shader stacks (per `CLAUDE.md`). Group them under a council lead to prevent deadlocks.
- **Shared utilities:** Mission control can temporarily assign two specialists to a single task (e.g., shadow engineer + DXR QA) but only one council lead reports back to mission control.
- **Runtime/UX automation:** The `asset-metadata-agent` drives the application at runtime (camera traversal, ImGui manipulation, metadata capture). Train it using mission-control prompts plus screenshot metadata JSON so it knows how to move through world-space.
- **Knowledge management:** `knowledge-steward` ingests every doc/update into the RAG DB (Chroma). If tokens expire mid-session, the next window asks the steward for `fetch_context(project_area="RTXDI")` to reload state.

---

## 3. Claude Agent SDK Implementation Plan
### 3.1 Standard options template
```python
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions

PROJECT_ROOT = Path("/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

def base_options():
    return ClaudeAgentOptions(
        cwd=PROJECT_ROOT,
        permission_mode="manual",  # tightened default
        allowed_tools=[],           # filled per agent
        mcp_servers={},             # filled per agent
        hooks={"PreToolUse": [], "PostToolUse": []}
    )
```

### 3.2 Registering agents (examples)
1. **Mission Control (SDK in-process)**
```python
from claude_agent_sdk import create_sdk_mcp_server, tool

@tool("dispatch_plan", "Route work to specialized agents", {"plan": str})
async def dispatch_plan(args):
    # write to docs/SESSION_SUMMARY_*.md, update backlog, return assignments
    ...

mission_control = create_sdk_mcp_server(
    name="mission-control",
    version="1.0.0",
    tools=[dispatch_plan, record_decision, publish_status]
)
```
```
options.mcp_servers["mission-control"] = mission_control
options.allowed_tools += ["mcp__mission-control__dispatch_plan", ...]
```

2. **Rendering council lead (external stdio)**
```
options.mcp_servers["rendering-council"] = {
    "type": "stdio",
    "command": "bash",
    "args": ["-c", "cd agents/rendering-council && ./run_server.sh"],
    "cwd": str(PROJECT_ROOT),
    "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
}
options.allowed_tools += [
    "mcp__rendering-council__assign_rt_task",
    "mcp__rendering-council__summarize_status"
]
```

3. **Hooks for safety**
```python
from claude_agent_sdk import HookMatcher

def deny_long_running_bash(input_data, tool_use_id, context):
    cmd = input_data["tool_input"].get("command", "")
    if "pix_capture" in cmd and "--background" not in cmd:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "PIX captures must run in background"
            }
        }
    return {}

options.hooks["PreToolUse"].append(
    HookMatcher(matcher="Bash", hooks=[deny_long_running_bash])
)
```

### 3.3 Agent MCP Creator checklist
When using the Claude Agent SDK plugin’s MCP creator UI:
1. **Name & description** – match the council/agent names above.
2. **Working directory** – set to the actual agent folder or project root (avoid relying on shell `cd`).
3. **Command** – for Python servers use `bash -lc "source venv/bin/activate && python server.py"` to guarantee environment activation.
4. **Environment variables** – at minimum: `PROJECT_ROOT`, `PYTORCH_ENABLE_MPS_FALLBACK=1` (for PyTorch agents), and any API keys.
5. **Allowed tools** – copy from Section 4 tables to ensure least-privilege.
6. **Hooks** – register shared safety hooks once (mission-control) so every agent inherits them.
7. **Test** – run `/mcp` in Claude Code to verify “Reconnected to <agent>”. Keep a short YAML describing each server for onboarding.

---

## 4. Agent Specialization Matrix
| Domain | When to split | Recommended agents | Notes |
| --- | --- | --- | --- |
| **Ray Tracing Lighting** | Any time a change touches RayQuery emission, multi-light math, or volumetric parameters | `rt-lighting-architect` | Owns `particle_gaussian_raytrace.hlsl` lighting sections, interacts with gaussian analyzer for performance budgets. |
| **Shadows** | DXR inline changes, PCSS → RayQuery migration, RTXDI shadow reuse | `shadow-systems-engineer` | Can spawn temporary sub-agents (e.g., “penumbra-optimizer”) but reports through rendering council. |
| **RTXDI / ReSTIR** | Weighted reservoir sampling, temporal accumulation, path-traced experiments | `rtxdi-director`, `restir-specialist` | Distinct pipelines with unique buffers/logs—keep separate to avoid context loss. |
| **Probe Grid / Ambient** | Changes to probe sampling, grid configs, or volumetric fallback | `probe-grid-engineer` | Coordinates with QA to ensure ambient light metrics remain stable. |
| **Materials & Celestial** | Struct layout, ImGui controls, procedural noise, pyro templates | `gaussian-analyzer`, `material-system-engineer`, `volumetric-pyro-specialist` | Analyzer sets spec; engineer edits files; pyro specialist designs effect metadata. |
| **Physics / PINN** | GPU physics shader vs PINN ONNX integration | `pinn-integration-agent`, `particle-physics-shader-agent` | Keeps ML and shader flows decoupled but cooperative. |
| **Operations** | Build automation, runtime navigation, metadata capture | `particle-pipeline-runner`, `asset-metadata-agent`, `config-curator` | Asset-metadata-agent manipulates runtime controls & camera to generate datasets. |
| **Diagnostics** | Image QA, log/PIX RAG, performance telemetry | `dxr-image-quality-analyst`, `log-analysis-rag`, `pix-debugger-vX`, `metrics-runner`, `critique-agent` | Critique agent reads QA+diagnostics output and files brutal, actionable feedback. |
| **Knowledge & Planning** | Backlog, context persistence, documentation | `mission-control`, `knowledge-steward`, `backlog-scribe` | `knowledge-steward` also tags artifacts for RAG search (per NVIDIA doc). |

**Rule of thumb:** If a subsystem has unique telemetry, buffers, or debugging workflows (e.g., RTXDI’s reservoir dumps vs PCSS ping-pong buffers), give it its own specialist. Otherwise, let the council lead own it.

---

## 5. Workflow Playbook
1. **Plan**: mission-control reads CLAUDE.md + SESSION docs, runs `dispatch_plan`. Knowledge-steward pulls latest notes from RAG store.
2. **Design**: council leads coordinate specialists (e.g., rtxdi-director + shadow engineer) and log decisions with backlog-scribe.
3. **Implement**: material-system-engineer or other file-op agents write code. Pipeline-runner compiles shaders, launches app, captures screenshots/logs.
4. **Evaluate**: diagnostics council runs QA (LPIPS, FPS) + log-analysis-rag diagnosis. Critique-agent summarizes issues with brutal honesty.
5. **Document**: mission-control merges findings into `docs/SESSION_SUMMARY_<date>.md`, updates roadmap checkboxes, and pushes to RAG store.
6. **Persist**: knowledge-steward updates vector DB entries (doc text, PIX logs, screenshot metadata). When a new session starts, mission-control queries `knowledge-steward.fetch_context(domain="rtxdi")` to reload state.

---

## 6. Implementation Checklist
1. **Create SDK servers** (mission-control, knowledge-steward, backlog-scribe, particle-pipeline-runner, asset-metadata-agent). Document their tool schemas.
2. **Standardize external servers**: add explicit `cwd`, `env`, and safety hooks in `.claude/mcp_settings.json` for every existing agent.
3. **Author council manifests**: each council lead needs README + AGENT_PROMPT listing subordinate agents, required telemetry, and handoff rules.
4. **Update documentation**: link this file + `CELESTIAL_AGENT_ECOSYSTEM_PLAN.md` inside `CLAUDE.md` and `MASTER_ROADMAP_V2.md`.
5. **Automate QA loop**: script `particle-pipeline-runner` to (a) run scenario, (b) wait for F2 screenshot, (c) call QA + diagnostics agents, (d) attach results to backlog ticket.
6. **Bootstrapping run**: run mission-control once per day to refresh backlog, assign tasks, and ensure knowledge-steward ingests new artifacts.
7. **Training runtime agent**: give `asset-metadata-agent` macros for camera controls + ImGui hotkeys so it can navigate world-space, capture metadata, and feed QA.

---

## 7. References
- `docs/research/NVIDIA_Log_Analysis_Multi-Agent_RAG.pdf` – Baseline architecture for multi-agent log analysis + self-correcting diagnostics (use as the canonical RAG workflow).
- `CLAUDE.md` – Repository-wide context and development philosophy.
- `docs/CELESTIAL_AGENT_ECOSYSTEM_PLAN.md` – High-level goals and checklists.
- Claude Agent SDK README & Overview – configuration, hooks, custom tools.[^1][^2]

[^1]: Claude Agent SDK for Python – README. https://github.com/anthropics/claude-agent-sdk-python
[^2]: Claude Agent SDK Overview – https://docs.claude.com/en/docs/agent-sdk/overview
