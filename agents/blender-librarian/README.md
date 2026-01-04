# Blender Librarian Agent (GPT-5.2 + Blender Manual RAG)

This agent is a **translation layer** between:

- **What the evaluator reports** (e.g., "WRONG COLOR", "TOO BRIGHT", warm_ratio mismatch)
- **What Blender knobs actually change** (Principled Volume inputs, color management, Mantaflow/VDB fields, bpy API)

It is designed to be used **only when stuck** (plateau, repeated failures, or "I don't know which Blender feature to use").

## Why this exists (vs just `blender-manual`)

`agents/blender-manual` is already an excellent local documentation MCP server:
- fast keyword search
- optional semantic search (local embeddings)
- read access to manual + Python API HTML (if present)

But it returns **raw documentation**, not **actionable decisions**.

This librarian:
- pulls the right doc snippets from `blender-manual`
- optionally uses **OpenAI GPT-5.2** (incl. vision) to propose **concrete script changes**
- returns a **structured JSON** payload that can be fed into `script-generator.modify_script()`

## Setup

1. Create a venv and install deps:

```bash
cd agents/blender-librarian
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

2. Set env vars:

- `OPENAI_API_KEY` (required for model reasoning/vision)
- `OPENAI_MODEL` (optional, default: `gpt-5.2`)

3. Run the server:

```bash
./run_server.sh
```

## Tools

- `advise_next_modifications(...)`: Return doc-grounded, (optionally) vision-assisted recommendations plus a `modifications` object compatible with `script-generator.modify_script()`.


