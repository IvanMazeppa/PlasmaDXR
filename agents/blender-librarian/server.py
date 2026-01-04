#!/usr/bin/env python3
"""
Blender Librarian MCP Server

Goal:
- Convert "what's wrong" (metrics/issues/images) into "what to change in Blender"
- Ground recommendations in Blender docs via the existing `agents/blender-manual` index
- Optionally use OpenAI (GPT-5.2, incl. vision) for targeted diagnosis/synthesis

Design constraints:
- Works WITHOUT OpenAI (doc-only mode) so it never blocks the pipeline
- Keeps token usage small by passing only top doc hits + short context
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("blender-librarian")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def _load_blender_manual_module():
    """
    Import blender-manual implementation directly so we can reuse its cache/indexing.

    This intentionally does NOT start an MCP client; it just calls the same Python
    functions the MCP server exposes.
    """
    manual_dir = PROJECT_ROOT / "agents" / "blender-manual"
    if str(manual_dir) not in sys.path:
        sys.path.insert(0, str(manual_dir))
    # Prefer wrapper `server.py` for stable import path.
    import server as blender_manual  # type: ignore
    return blender_manual


def _read_image_b64(image_path: str) -> Optional[str]:
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / image_path
    if not path.exists():
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _call_openai_responses(
    prompt: str,
    render_path: Optional[str],
    reference_path: Optional[str],
    model: str,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    inputs = [
        {
            "role": "system",
            "content": (
                "You are a Blender 5.x VFX specialist. "
                "Return ONLY valid JSON. No markdown. No prose outside JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Attach images (optional). Use data URLs to avoid hosting.
    # NOTE: This uses the "input_image" content type supported by OpenAI vision models.
    render_b64 = _read_image_b64(render_path) if render_path else None
    ref_b64 = _read_image_b64(reference_path) if reference_path else None

    if render_b64:
        inputs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "This is the current render to diagnose.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{render_b64}",
                    },
                ],
            }
        )
    if ref_b64:
        inputs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "This is the ground truth reference image to match.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{ref_b64}",
                    },
                ],
            }
        )

    payload = {
        "model": model,
        "input": inputs,
        # Keep costs bounded by default; can be overridden by env if needed.
        "max_output_tokens": int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200")),
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    # Extract text output (best-effort).
    # The Responses API returns an array of output items; we try common shapes.
    text_out = None
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                text_out = content.get("text")
                break
        if text_out:
            break

    if not text_out:
        return {"error": "No text output from OpenAI", "raw": data}

    try:
        return json.loads(text_out)
    except Exception:
        return {"error": "OpenAI output was not valid JSON", "raw_text": text_out, "raw": data}


@mcp.tool()
def advise_next_modifications(
    effect_type: str,
    problem: str,
    current_params_json: str = "{}",
    evaluator_json: str = "{}",
    render_path: str = "",
    reference_path: str = "",
    script_path: str = "",
    max_doc_results: int = 5,
) -> str:
    """
    Provide doc-grounded (optionally vision-assisted) advice for the next iteration.

    Returns JSON with:
    - doc_research: top doc hits from blender-manual
    - modifications: dict compatible with script-generator.modify_script()
    - rationale: brief explanation of why these changes should help
    - confidence: 0-1
    """
    blender_manual = _load_blender_manual_module()

    # Doc retrieval (cheap, local)
    doc_research = {
        "nodes_principled_volume": blender_manual.search_nodes("Principled Volume", "shader", limit=max_doc_results, compact=True),
        "manual_volume_rendering": blender_manual.search_manual("volume rendering emission blackbody temperature attribute", limit=max_doc_results, compact=True),
        "manual_color_management": blender_manual.search_manual("color management Filmic AgX exposure", limit=max_doc_results, compact=True),
        "python_api_volume_nodes": blender_manual.search_python_api("bpy.types.ShaderNodeVolumePrincipled", limit=max_doc_results, compact=True, api_only=True),
        "vdb_workflow": blender_manual.search_vdb_workflow("export openvdb cache bake mantaflow", limit=max_doc_results, compact=True),
    }

    # If OpenAI is not configured, return doc-only payload.
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return json.dumps(
            {
                "mode": "doc_only",
                "message": "Set OPENAI_API_KEY to enable GPT-5.2 diagnosis + vision.",
                "doc_research": doc_research,
                "modifications": {},
                "rationale": "Doc lookup only (no model synthesis).",
                "confidence": 0.0,
            },
            indent=2,
        )

    prompt = f"""
We are iterating a Blender VFX pipeline.

Effect type: {effect_type}
Problem: {problem}
Current params (JSON): {current_params_json}
Evaluator output (JSON): {evaluator_json}
Current script path: {script_path}

We have Blender docs search results (compact):
{json.dumps(doc_research)}

TASK:
Return a JSON object with:
  - "diagnosis": short string
  - "modifications": dict ONLY using keys supported by script-generator.modify_script:
        resolution, frame_end, frame_start, turbulence, vorticity, temperature, domain_scale, custom_code
    where custom_code is a dict of exact string replacements {{"search":"replace"}}
  - "rationale": short string
  - "confidence": number 0..1
  - "doc_paths_used": array of strings (paths you relied on; if unknown, use [])

Constraints:
- Keep changes small (<= 2 parameter changes + <= 3 custom_code replacements).
- Prefer changes that address the stated problem.
- If unsure, propose a SAFE diagnostic change rather than a big guess.
"""

    out = _call_openai_responses(
        prompt=prompt,
        render_path=render_path or None,
        reference_path=reference_path or None,
        model=model,
    )

    # Ensure doc research is always included in the response.
    merged = {
        "mode": "openai",
        "doc_research": doc_research,
        **out,
    }
    return json.dumps(merged, indent=2)


if __name__ == "__main__":
    mcp.run()


