# Anthropic BYO Agent (Minimal)

This is a tiny helper to use your own Anthropic API key (e.g., Claude 3.5/4.5 Sonnet) from within the repo, aligned with the repo’s guardrails and patch policy.

## Features

- Reads recent docs (e.g., `NEXT_SESSION_START_HERE.md`, `CLAUDE.md`) to prime context
- Guardrails-aware system prompt matching MCP background agent policy
- Writes output as a versioned patch note under `Versions/` (YYYYMMDD-HHMM_slug.patch)
- Optional: print-only mode without writing a patch

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
cd tools/anthropic_agent
python agent.py "propose blit pipeline edits for HDR->SDR path" --model claude-3-5-sonnet-20241022 --slug blit_hdr_sdr
```

- To preview without writing a patch:

```bash
python agent.py "summarize current NEXT_SESSION steps" --no-write
```

## Advanced usage

- Include specific files for extra context (repeatable):

```bash
python agent.py "audit swapchain format vs blit path" \
  --include "src/core/SwapChain.cpp" \
  --include "src/core/Application.cpp"
```

- Persist simple session memory across runs:

```bash
python agent.py "next steps for HDR pipeline" --session plasma --slug hdr_followup
```

- Adjust document/context size limits:

```bash
python agent.py "short summary" --max-doc-chars 12000 --no-write
```

- Change model/provider:
  - Set `--model` to any supported Anthropic model string available to your API key.
  - To use a proxy (e.g., OpenRouter), configure their environment variables and model names, then pass `--model` accordingly.

## Notes

- This tool does not apply code edits directly; it creates patch notes under `Versions/` per project policy. Apply edits manually or with your preferred workflow.
- For background/agent-style operation, you can script calls to this CLI with different prompts.
- To route via a proxy (e.g., OpenRouter), set their compatible environment variables and adjust `--model` accordingly.

## Troubleshooting

- If you see `anthropic package not installed`, run `pip install anthropic`.
- If `ANTHROPIC_API_KEY` is missing, export it in your shell before running.
- If no docs are pulled, ensure the filenames in the repo root match those referenced in `agent.py`.

## Claude Code: add Memory MCP server

You can add the Anthropic Memory MCP server to Claude Code/Claude Desktop using their MCP config. Reference: [anthropic/memory-mcp config](https://hub.continue.dev/anthropic/memory-mcp?view=config)

Linux/macOS (jq):

```bash
CONFIG_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json" \
&& mkdir -p "$(dirname "$CONFIG_PATH")" \
&& [ -f "$CONFIG_PATH" ] || echo '{"mcpServers":{}}' > "$CONFIG_PATH" \
&& tmp="$(mktemp)" \
&& jq '.mcpServers["memory-mcp"] = {"command":"npx","args":["-y","@anthropic-ai/memory-mcp"]}' "$CONFIG_PATH" > "$tmp" \
&& mv "$tmp" "$CONFIG_PATH" \
&& echo "Added memory-mcp. Restart Claude Code/Claude Desktop."
```

Windows (PowerShell):

```powershell
$path = "$env:APPDATA\Claude\claude_desktop_config.json"
New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
if (!(Test-Path $path)) { '{"mcpServers":{}}' | Set-Content -Path $path -Encoding UTF8 }
$json = Get-Content $path -Raw | ConvertFrom-Json
if ($null -eq $json.mcpServers) { $json | Add-Member -NotePropertyName mcpServers -NotePropertyValue @{} }
$json.mcpServers.'memory-mcp' = @{ command = 'npx'; args = @('-y','@anthropic-ai/memory-mcp') }
$json | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
Write-Host "Added memory-mcp. Restart Claude Code/Claude Desktop."
```

After restarting, test with a prompt like: "memory status" or "store memory: [your note]".
