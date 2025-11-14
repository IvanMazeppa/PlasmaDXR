# Celestial Agent Ecosystem – Claude SDK Implementation Blueprint
_Created_: 2025‑11‑14  
_Author_: GPT‑5.1 Codex (Cursor)  

---

## 1. Strategic Goals
- Replace the ad-hoc “one agent does everything” loop with a **hierarchical, specialized roster** that mirrors NVIDIA’s multi-agent RAG concepts while remaining executable inside Claude Agent SDK sessions.
- Evolve the Gaussian renderer into a **celestial body toolkit** (gas clouds, stellar nurseries, rings, neutron stars, supernovae) with per-material shading controls, temporal behaviors, and procedural noise stacks.
- Embed **automated quality + diagnostic loops** (LPIPS image QA, PIX/log RAG analysis) so every run ends with verifiable metrics and searchable evidence.
- Keep development inside **Claude Agent SDK** conventions: `ClaudeAgentOptions`, SDK MCP servers, hooks, and permission modes for deterministic execution.[^1][^2]

---

## 2. Claude Agent SDK Integration Principles
| Area | Implementation Guidance |
| --- | --- |
| **Server Types** | Prefer in-process SDK MCP servers (`create_sdk_mcp_server`) for lightweight tools (metrics, config transforms) and reserve external stdio servers for heavy GPU/ML tasks (image QA, PIX analyzers). Mixed-mode configs are supported in `ClaudeAgentOptions.mcp_servers`.[^1] |
| **Agent Specialization** | Each specialized agent exposes a tight tool surface with explicit schemas. Use the SDK’s `@tool` decorator + dataclass arguments so Claude can reason about capabilities while preserving the “one agent = one specialty” rule you want. |
| **Permission & Hooks** | Set `permission_mode='acceptEdits'` only for trusted file-ops agents; use hooks (e.g., `PreToolUse`) to block destructive Bash commands or require confirmation before shader recompiles.[^1] |
| **Working Directory** | Pass `cwd` in `ClaudeAgentOptions` per task so Sonnet/Opus always operate relative to `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean`. Misaligned CWDs are the most common SDK configuration mistake. |
| **Session Forking** | For long-lived workflows (e.g., pyro design running alongside QA), fork sessions programmatically (`ClaudeSDKClient`) so each agent maintains its own state without stepping on others. |
| **Error Surfaces** | Bubble SDK errors (`ClaudeSDKError`, `ProcessError`, `CLIJSONDecodeError`) back into the RAG log so `log-analysis-rag` can correlate tool failures with project state. |

---

## 3. Current Configuration Review & Fixes
1. **Environment leakage** – Several external MCP servers rely on shell scripts that implicitly `cd` into subdirectories. Replace with explicit `cwd` in `ClaudeAgentOptions` or wrapper scripts to avoid path drift during Sonnet/Opus handoffs.  
2. **Permission defaults** – Most servers run with unrestricted Bash/Write access. Align with SDK guidance by declaring required tools per agent (`allowed_tools=["mcp__gaussian-analyzer__analyze_gaussian_parameters", ...]`) and deny everything else for safety.[^1]  
3. **Lack of hooks** – No `PreToolUse` or `PostToolUse` hooks are vetting commands. Add hook guardrails to enforce GPU capture cooldowns, limit PIX dumps, and ensure long-running scripts go to background per SDK practices.  
4. **Server taxonomy** – Some “agents” overlap (e.g., gaussian analyzer vs material engineer). Clarify capabilities in their SDK manifests and consider using SDK in-process servers for lightweight analytic helpers to reduce process sprawl, as recommended in the Agent SDK overview.[^2]  

---

## 4. Updated Agent Roster (SDK-Compliant)
| Tier | Agent | Delivery Mode | Responsibilities | Key Claude SDK Settings |
| --- | --- | --- | --- | --- |
| **Command** | `mission-control` (new) | In-process SDK server with orchestration tool | Plans sprints, assigns agents, aggregates QA. Runs `dispatch_plan`, `record_decision`, `trigger_review`. | `allowed_tools`: only orchestration tools; hooks enforce dependency completion. |
| **Design** | `gaussian-analyzer` | External MCP (already built) | Structural scans, material simulations, performance budgets. | Add SDK metadata so Sonnet can discover `analyze_gaussian_parameters`, etc. |
|  | `dxr-volumetric-pyro-specialist` | External (PyTorch heavy) | Supernova/prominence specs, temporal curves, pyro performance. | Keep as stdio server; pass `PYTORCH_ENABLE_MPS_FALLBACK=1` env for reliability. |
|  | `dxr-shadow-engineer` | External | RayQuery shadow research/codegen. | Use `ClaudeSDKClient` session forks so Sonnet can iterate without blocking others. |
| **Implementation** | `material-system-engineer` | External (file ops heavy) | Code generation, ImGui controls, presets, build instructions. | `permission_mode='manual'` so you must approve file edits. |
|  | `particle-pipeline-runner` (new) | In-process SDK server calling local scripts | Automates shader rebuilds, launch configs, screenshot captures. | Hooks enforce background execution + timeouts. |
| **Diagnostics** | `dxr-image-quality-analyst` | External (ML) | LPIPS comparisons, 7-dim grading, FPS parsing. | Provide `cwd` pointing to screenshot directory. |
|  | `log-analysis-rag` | External + Chroma DB | Log/PIX ingestion, semantic queries, hypothesis generation. | Hook ensures ingestion runs before query to keep vector store fresh. |
| **Support** | `pix-debugger` family | External | Buffer/PIX capture validation. | Keep as-is; ensure `allowed_tools` limited to PIX operations. |

---

## 5. Celestial Rendering Roadmap (Overarching)
1. **Phase 0 – Baseline & Regression Harness**  
   - Re-enable lights, verify particle buffer stride matches shader struct.  
   - Capture golden screenshots + logs; ingest via QA + RAG for future diffing.  
2. **Phase 1 – Material Schema Upgrade**  
   - Extend `MaterialTypeProperties` with procedural/noise/rim controls.  
   - Expose ImGui + JSON presets via `material-system-engineer`.  
   - Use gaussian analyzer’s `validate_particle_struct` and `estimate_performance_impact` to lock performance budgets.  
3. **Phase 2 – Celestial Presets & Pyro Integration**  
   - Gas clouds, stellar nurseries, rocky/icy bodies, neutron stars, supernovae; each with pyro specs (temporal curves, emission ramps).  
   - Add per-material noise sampling + temporal animation in `particle_gaussian_raytrace.hlsl`.  
4. **Phase 3 – Lighting & Shadows Refresh**  
   - Replace hardcoded volumetric params with per-light buffers; integrate RayQuery shadows from `dxr-shadow-engineer`.  
   - Feed RTXDI selections & multi-light data into new shading pipeline.  
5. **Phase 4 – Autonomous QA & Diagnostics Loop**  
   - Automate screenshot capture → LPIPS comparison → quality report storage.  
   - Pipe logs/PIX dumps into `log-analysis-rag` nightly for trend detection.  
6. **Phase 5 – Continuous Improvement**  
   - Add adaptive Gaussian merging, hybrid LoD, and scientific presets (NASA references).  
   - Use SDK session forking to run optimization agents (e.g., performance analyzer) in parallel without blocking mission control.  

---

## 6. Implementation Checklist (Save-Friendly)
1. **Claude SDK Config**
   - [ ] Consolidate MCP definitions in `.claude/mcp_settings.json` with explicit `cwd`, `env`, and `permission_mode` per agent.[^1][^2]
   - [ ] Register new in-process orchestration + pipeline-runner servers via `create_sdk_mcp_server`.  
2. **Agent Manifests**
   - [ ] Update each agent README/AGENT_PROMPT with tool schemas + references to mission control.  
   - [ ] Document allowed tool list + hooks.  
3. **Renderer Evolution**
   - [ ] Extend `MaterialTypeProperties` & GPU CBs.  
   - [ ] Implement per-material noise + temporal controls.  
   - [ ] Integrate pyro presets + RayQuery shadows.  
4. **Automation / QA**
   - [ ] Script screenshot capture & LPIPS runs after every material change.  
   - [ ] Nightly `log-analysis-rag ingest_logs` + summary.  
   - [ ] Mission control agent compiles decisions + metrics into `docs/SESSION_SUMMARY_<date>.md`.  

---

## 7. References
[^1]: Claude Agent SDK for Python – README, tool registration, hooks, and MCP server configuration guidance. [GitHub](https://github.com/anthropics/claude-agent-sdk-python)  
[^2]: Claude Agent SDK Overview – official documentation covering ClaudeAgentOptions, MCP integration, hooks, and Agent Skills. [Docs](https://docs.claude.com/en/docs/agent-sdk/overview)  


