# SDK Tools Review — Shell & Local Shell (2026-01-23)

This document summarizes what’s available in the Agents SDK “tools” module and focuses on `ShellTool` vs `LocalShellTool`, with recommendations for Blender VFX Orchestrator use. Sources are the official SDK tools docs and OpenAI tool guides.

References:
- Tools overview: https://openai.github.io/openai-agents-python/tools/
- Tool reference (ShellTool/LocalShellTool): https://openai.github.io/openai-agents-python/ref/tool/
- Shell tool guide: https://platform.openai.com/docs/guides/tools-shell
- Local shell tool guide: https://platform.openai.com/docs/guides/tools-local-shell

## What “tools” contains (SDK)
The SDK tools module (`src/agents/tool.py`) defines built‑in tools and types:
- `ShellTool` (next‑gen shell tool, preferred).
- `LocalShellTool` (legacy / deprecated).
- `ApplyPatchTool`, `ComputerTool`, hosted tools, and tool output types.

Key type definitions from the SDK:
- `ShellExecutor`: callable that executes shell requests and returns text or `ShellResult`.
- `LocalShellExecutor`: callable that executes a command and returns text.
- `ShellResult`, `ShellCommandOutput`, `ShellCallOutcome` for structured output.

## ShellTool vs LocalShellTool

### ShellTool (Recommended)
From the SDK reference:
- “Next‑generation shell tool. LocalShellTool will be deprecated in favor of this.”
- Accepts a `ShellExecutor` that can return structured `ShellResult`.

From the platform docs:
- Works with GPT‑5.1+ via Responses API.
- Supports multiple commands per call and returns structured outputs.
- The model proposes commands; your integration executes them.

### LocalShellTool (Legacy)
From the platform docs:
- “Local shell tool is outdated… use shell tool with GPT‑5.1 instead.”
- Tied to `codex-mini-latest`.
- Uses `local_shell_call` items in a loop and returns simple string output.

**Recommendation:** Prefer `ShellTool` for new work. Only use `LocalShellTool` if you must remain on `codex-mini-latest`.

## Why ShellTool Could Help Your Orchestrator
Potential uses (if you want agents to run commands directly):
- Run Blender CLI scripts without an extra MCP hop (single tool call).
- Collect render artifacts, list outputs, and validate file structure.
- Trigger quick diagnostics (disk space, env vars, GPU process checks).
- Run test scripts (e.g., “smoke” runs) and summarize outputs.

**Caution:** This duplicates some of your existing MCP executor functionality. Only adopt if you want the model to directly orchestrate command execution rather than using MCP tools.

## Minimal ShellTool Pattern (Python)
This mirrors the official shell guide and the SDK types. You must supply an executor.
```
from agents import ShellTool, ShellResult, ShellCommandOutput, ShellCallOutcome

class LocalShell:
    async def __call__(self, request):  # request: ShellCommandRequest
        action = request.data.action
        # Run commands safely in your environment and capture stdout/stderr.
        return ShellResult(
            output=[
                ShellCommandOutput(
                    command="...",
                    stdout="...",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                )
            ],
            max_output_length=action.max_output_length,
        )

shell_tool = ShellTool(executor=LocalShell())
```

## Safety & Operational Guidance (from official docs)
Both shell tools require **strict safety controls**:
- Sandbox or containerize execution.
- Apply allow‑/deny‑lists (block destructive commands like `rm`).
- Enforce timeouts and resource limits.
- Log every command and output for auditability.
- Avoid interactive commands (no prompts, editors).

## When to Avoid ShellTool
- If MCP toolchain already covers execution safely.
- If you need deterministic, audited operations only.
- If you cannot guarantee command sandboxing.

## LocalShellTool Notes (Legacy)
Local shell:
- Is explicitly marked **outdated** in the docs.
- Only supports `codex-mini-latest`.
- Uses a `local_shell_call` loop and string output.

Use only for backward compatibility if required.

## Other Tools Worth Noting
If you want to expand monitoring/automation:
- `ApplyPatchTool` (model proposes diffs; you apply locally).
- `ComputerTool` (GUI automation).
- Hosted tools (web search, file search, code interpreter) for richer research.

## Recommendation Summary
- Prefer `ShellTool` over `LocalShellTool`.
- Only add shell execution if you need direct model‑driven command runs.
- Keep shell access tightly sandboxed; treat as privileged capability.
