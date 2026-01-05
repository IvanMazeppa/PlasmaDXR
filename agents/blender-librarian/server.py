#!/usr/bin/env python3
"""
Blender Librarian MCP Server

Purpose: Break out of local minima when the VFX iteration loop is stuck.
- Uses GPT-5.2 vision to diagnose what's ACTUALLY wrong (not just metrics)
- Grounds recommendations in Blender docs via blender-manual MCP
- Tracks budget to stay within $20/month

When to use:
- 2+ iterations without improvement
- Metrics plateau but visual quality is poor
- Same issues persist across multiple iterations

Design:
1. Check playbook first (FREE)
2. Query blender-manual docs (FREE)
3. If still stuck, use GPT-5.2 vision (PAID, budget-tracked)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from PIL import Image
import io

# Load environment from explicit path (MCP runs from different cwd)
from dotenv import load_dotenv
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

mcp = FastMCP("blender-librarian")

# Paths
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LIBRARIAN_DIR = Path(__file__).parent
PLAYBOOK_DIR = LIBRARIAN_DIR / "playbooks"
CACHE_DIR = LIBRARIAN_DIR / "cache"
BUDGET_FILE = LIBRARIAN_DIR / "budget_tracker.json"

# Ensure directories exist
PLAYBOOK_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# GPT-5.2 Pricing (as of Jan 2026)
COST_INPUT_1K = 0.00175   # $1.75 per 1M input tokens
COST_OUTPUT_1K = 0.014    # $14 per 1M output tokens
COST_IMAGE_LOW = 0.00015  # 85 tokens at low detail

# Budget limits
MONTHLY_BUDGET = 20.0
VISION_BUDGET = 10.0      # $10 for vision analysis
DOC_BUDGET = 8.0          # $8 for doc synthesis
BUFFER = 2.0              # $2 emergency buffer

# Model config
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
MAX_IMAGE_SIZE = 512  # Resize images to control token cost


# =============================================================================
# BUDGET TRACKING
# =============================================================================

@dataclass
class BudgetState:
    month: str = ""
    vision_spent: float = 0.0
    doc_spent: float = 0.0
    total_calls: int = 0
    last_updated: str = ""


def load_budget() -> BudgetState:
    """Load budget state, reset if new month."""
    current_month = datetime.now().strftime("%Y-%m")

    if BUDGET_FILE.exists():
        try:
            data = json.loads(BUDGET_FILE.read_text())
            state = BudgetState(**data)
            # Reset if new month
            if state.month != current_month:
                state = BudgetState(month=current_month)
        except Exception:
            state = BudgetState(month=current_month)
    else:
        state = BudgetState(month=current_month)

    return state


def save_budget(state: BudgetState):
    """Persist budget state."""
    state.last_updated = datetime.now().isoformat()
    BUDGET_FILE.write_text(json.dumps(state.__dict__, indent=2))


def can_afford_vision(estimated_cost: float = 0.05) -> bool:
    """Check if we have vision budget remaining."""
    state = load_budget()
    return state.vision_spent + estimated_cost <= VISION_BUDGET


def can_afford_docs(estimated_cost: float = 0.02) -> bool:
    """Check if we have doc synthesis budget remaining."""
    state = load_budget()
    return state.doc_spent + estimated_cost <= DOC_BUDGET


def record_spend(category: str, cost: float):
    """Record API spending."""
    state = load_budget()
    if category == "vision":
        state.vision_spent += cost
    else:
        state.doc_spent += cost
    state.total_calls += 1
    save_budget(state)


# =============================================================================
# PLAYBOOK SYSTEM (FREE - No API calls)
# =============================================================================

# Known fixes for common issues - these are FREE
SOLAR_PLAYBOOK = {
    "too_dark": {
        "symptoms": ["too dark", "underexposed", "insufficient brightness", "black", "dim"],
        "fix": {"blackbody_intensity": 1.5, "emission_strength": 2.0},
        "rationale": "Increase emission and blackbody intensity",
        "confidence": 0.85
    },
    "too_orange": {
        "symptoms": ["too orange", "too warm", "yellow cast", "wrong color temperature"],
        "fix": {"temperature": 5778, "blackbody_intensity": 0.8},
        "rationale": "Set to solar temperature (5778K) and reduce intensity",
        "confidence": 0.8
    },
    "blackbody_blowout": {
        "symptoms": ["overexposed", "clipped highlights", "white core", "saturated", "blowout"],
        "fix": {"blackbody_intensity": 0.3, "emission_strength": 0.5},
        "rationale": "Reduce blackbody and emission to prevent clipping",
        "confidence": 0.85
    },
    "no_limb_darkening": {
        "symptoms": ["no limb darkening", "flat edges", "uniform brightness", "no falloff"],
        "fix": {"custom_code": {"density_falloff = 0": "density_falloff = 2.5"}},
        "rationale": "Add radial density falloff for limb darkening effect",
        "confidence": 0.7
    },
    "no_structure": {
        "symptoms": ["no structure", "too smooth", "lacks detail", "blobby", "no texture"],
        "fix": {"turbulence": 0.6, "vorticity": 0.4},
        "rationale": "Increase turbulence and vorticity for surface detail",
        "confidence": 0.75
    },
    "clipping_top": {
        "symptoms": ["clipping at top", "cut off", "truncated", "domain too small"],
        "fix": {"domain_scale": 1.5},
        "rationale": "Increase domain size to prevent clipping",
        "confidence": 0.9
    },
    "prominence_missing": {
        "symptoms": ["no prominences", "no eruptions", "no loops", "flat corona"],
        "fix": {"flame_smoke": 3.0, "buoyancy_diff": 2.0},
        "rationale": "Increase flame smoke and buoyancy for prominence formation",
        "confidence": 0.65
    }
}


def check_playbook(effect_type: str, issues: List[str]) -> Optional[Dict]:
    """
    Check if any known fix applies to the current issues.
    Returns the fix if found, None otherwise.
    This is FREE - no API call needed.
    """
    playbook = SOLAR_PLAYBOOK if effect_type.lower() in ["sun", "star", "solar"] else {}

    for issue in issues:
        issue_lower = issue.lower()
        for fix_name, fix_data in playbook.items():
            if any(symptom in issue_lower for symptom in fix_data["symptoms"]):
                return {
                    "source": "playbook",
                    "cost": 0.0,
                    "fix_name": fix_name,
                    "modifications": fix_data["fix"],
                    "rationale": fix_data["rationale"],
                    "confidence": fix_data["confidence"]
                }
    return None


# =============================================================================
# TOOL CALLING AGENT (GPT-5.2 with function calling)
# =============================================================================

_tool_calling_agent = None

def get_tool_calling_agent():
    """Lazy load the tool calling agent."""
    global _tool_calling_agent
    if _tool_calling_agent is None:
        from tools.tool_calling_agent import ToolCallingAgent
        _tool_calling_agent = ToolCallingAgent()
    return _tool_calling_agent


async def search_docs_with_tool_calling(
    query: str,
    effect_type: str = "",
    current_issues: Optional[List[str]] = None,
    current_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Use GPT-5.2 with tool calling to search Blender docs intelligently.

    This is the RECOMMENDED way to search docs - GPT-5.2 decides which
    tools to call based on the query.

    Returns dict with: success, answer, modifications, doc_paths_used, cost
    """
    from tools.tool_calling_agent import search_docs_with_agent
    result = await search_docs_with_agent(
        query=query,
        effect_type=effect_type,
        current_issues=current_issues,
        current_params=current_params
    )
    return {
        "success": result.success,
        "answer": result.answer,
        "modifications": result.modifications,
        "doc_paths_used": result.doc_paths_used,
        "tool_calls_made": result.tool_calls_made,
        "cost": result.total_cost,
        "iterations": result.iterations,
        "error": result.error
    }


# =============================================================================
# BLENDER MANUAL INTEGRATION (Legacy - direct search)
# =============================================================================

_blender_manual = None

def get_blender_manual():
    """Lazy load blender-manual module."""
    global _blender_manual
    if _blender_manual is None:
        manual_dir = PROJECT_ROOT / "agents" / "blender-manual"
        if str(manual_dir) not in sys.path:
            sys.path.insert(0, str(manual_dir))
        import blender_server as blender_manual
        _blender_manual = blender_manual
    return _blender_manual


def search_docs_for_issue(issue: str, effect_type: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Query blender-manual for relevant documentation.
    This is FREE - uses local embeddings.
    """
    try:
        manual = get_blender_manual()

        # Build targeted queries based on issue
        queries = {
            "semantic": manual.search_semantic(f"{effect_type} {issue} fix solution", limit=max_results, compact=True),
            "manual": manual.search_manual(f"volume {issue}", limit=max_results, compact=True),
            "vdb": manual.search_vdb_workflow(f"{issue} mantaflow", limit=max_results, compact=True),
        }

        return {"status": "success", "results": queries}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def resize_image_for_vision(image_path: str) -> Optional[str]:
    """
    Resize image to MAX_IMAGE_SIZE and return base64.
    Smaller images = fewer tokens = lower cost.
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / image_path

    if not path.exists():
        return None

    try:
        with Image.open(path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Resize maintaining aspect ratio
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

            # Save to buffer as JPEG (smaller than PNG)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode('ascii')
    except Exception:
        return None


def get_image_hash(image_path: str) -> str:
    """Get content hash for caching."""
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / image_path

    if not path.exists():
        return ""

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# =============================================================================
# GPT-5.2 INTEGRATION
# =============================================================================

def call_gpt52_vision(
    system_prompt: str,
    user_prompt: str,
    render_b64: Optional[str] = None,
    reference_b64: Optional[str] = None,
    reasoning_effort: str = "low",
    verbosity: str = "low"
) -> tuple[Dict[str, Any], float]:
    """
    Call GPT-5.2 with optional images using Responses API.
    Returns (response_dict, cost).

    Uses correct Responses API format per OpenAI docs (Jan 2026):
    - input_text and input_image types (not text/image_url)
    - reasoning and text parameters for GPT-5.2 optimization
    """
    import requests

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}, 0.0

    # Build input with correct Responses API format
    input_data = []

    # Add text content
    content = [{"type": "input_text", "text": user_prompt}]

    # Add render image if provided
    if render_b64:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{render_b64}",
            "detail": "low"  # 85 tokens
        })

    # Add reference image if provided
    if reference_b64:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{reference_b64}",
            "detail": "low"
        })

    input_data = [{"role": "user", "content": content}]

    payload = {
        "model": OPENAI_MODEL,
        "instructions": system_prompt,
        "input": input_data,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": verbosity},
        "store": False,
        "max_output_tokens": 1200
        # Note: temperature is not supported with GPT-5.2
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        data = r.json()

        # Extract text from response
        text_out = None
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    text_out = content.get("text")
                    break
            if text_out:
                break

        if not text_out:
            return {"error": "No text output", "raw": data}, 0.0

        # Estimate cost
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 500)
        output_tokens = usage.get("output_tokens", 200)
        cost = (input_tokens / 1000 * COST_INPUT_1K) + (output_tokens / 1000 * COST_OUTPUT_1K)
        if render_b64:
            cost += COST_IMAGE_LOW
        if reference_b64:
            cost += COST_IMAGE_LOW

        # Parse JSON response
        try:
            result = json.loads(text_out)
        except json.JSONDecodeError:
            result = {"raw_text": text_out}

        return result, cost

    except requests.exceptions.RequestException as e:
        # Capture response body for debugging
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = f"{str(e)} - Response: {e.response.text[:500]}"
            except Exception:
                pass
        return {"error": f"API request failed: {error_detail}"}, 0.0


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def diagnose_render_issue(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "sun",
    current_issues: str = "[]",
    current_score: float = 0.0
) -> str:
    """
    Diagnose what's wrong with a render using GPT-5.2 vision.

    This is the PRIMARY tool for breaking out of local minima.
    It compares the render against the reference and identifies
    the actual visual problems that metrics might miss.

    Args:
        render_path: Path to the current render
        reference_path: Path to ground truth reference image
        effect_type: Type of effect (sun, star, explosion, etc.)
        current_issues: JSON array of issues from asset-evaluator
        current_score: Current quality score (0-100)

    Returns:
        JSON with diagnosis, primary_issue, and severity
    """
    # Check budget
    if not can_afford_vision():
        budget = load_budget()
        return json.dumps({
            "error": "Vision budget exhausted",
            "vision_spent": budget.vision_spent,
            "vision_budget": VISION_BUDGET,
            "suggestion": "Use get_modification_advice with manual issue description"
        }, indent=2)

    # Load images
    render_b64 = resize_image_for_vision(render_path)
    reference_b64 = resize_image_for_vision(reference_path) if reference_path else None

    if not render_b64:
        return json.dumps({"error": f"Could not load render: {render_path}"})

    # Build prompt
    system_prompt = """You are a VFX quality analyst specializing in Blender volumetric rendering.
Analyze the render image and compare it to the reference (if provided).
Be SPECIFIC and ACTIONABLE. Don't just say "needs improvement" - say exactly what's wrong.
Return ONLY valid JSON."""

    issues_list = json.loads(current_issues) if current_issues != "[]" else []

    user_prompt = f"""Analyze this {effect_type} render.

Current quality score: {current_score}/100
Known issues from metrics: {issues_list}

{"REFERENCE IMAGE PROVIDED - Compare against it." if reference_b64 else "No reference provided."}

Return JSON with:
{{
  "diagnosis": "1-2 sentence description of what's wrong",
  "primary_issue": "single most important problem to fix",
  "secondary_issues": ["list", "of", "other", "issues"],
  "severity": "critical|high|medium|low",
  "what_reference_has": "what the reference has that render lacks" (if reference provided),
  "visual_delta": "specific visual differences" (if reference provided)
}}"""

    result, cost = call_gpt52_vision(
        system_prompt,
        user_prompt,
        render_b64,
        reference_b64,
        reasoning_effort="low",  # Visual analysis doesn't need deep reasoning
        verbosity="low"
    )

    # Record spend
    record_spend("vision", cost)

    # Add metadata
    result["source"] = "gpt52_vision"
    result["cost"] = cost
    result["render_path"] = render_path

    return json.dumps(result, indent=2)


@mcp.tool()
def get_modification_advice(
    effect_type: str,
    issues: str,
    current_params: str = "{}",
    evaluator_scores: str = "{}"
) -> str:
    """
    Get parameter modification advice to fix identified issues.

    Flow:
    1. Check playbook for known fixes (FREE)
    2. Search blender-manual docs (FREE)
    3. If needed, synthesize with GPT-5.2 (PAID)

    Args:
        effect_type: Type of effect (sun, star, explosion, etc.)
        issues: JSON array of issue descriptions
        current_params: JSON of current Blender parameters
        evaluator_scores: JSON of evaluation scores/metrics

    Returns:
        JSON with modifications compatible with script-generator.modify_script()
    """
    issues_list = json.loads(issues) if issues else []
    params = json.loads(current_params) if current_params else {}
    scores = json.loads(evaluator_scores) if evaluator_scores else {}

    # Step 1: Check playbook (FREE)
    playbook_fix = check_playbook(effect_type, issues_list)
    if playbook_fix and playbook_fix["confidence"] >= 0.7:
        return json.dumps(playbook_fix, indent=2)

    # Step 2: Search docs (FREE)
    doc_results = {}
    for issue in issues_list[:3]:  # Top 3 issues
        doc_results[issue] = search_docs_for_issue(issue, effect_type)

    # Step 3: Check if we need GPT synthesis
    if not can_afford_docs():
        # Return doc-only results
        return json.dumps({
            "source": "docs_only",
            "cost": 0.0,
            "doc_research": doc_results,
            "playbook_partial": playbook_fix,
            "modifications": playbook_fix["modifications"] if playbook_fix else {},
            "rationale": "Budget exhausted - returning doc research and partial playbook match",
            "confidence": playbook_fix["confidence"] if playbook_fix else 0.3
        }, indent=2)

    # Step 4: GPT-5.2 synthesis (PAID)
    system_prompt = """You are a Blender VFX parameter optimization expert.
Given issues and documentation, suggest SPECIFIC parameter changes.
Return ONLY valid JSON. Be conservative - small changes are safer."""

    user_prompt = f"""Fix these {effect_type} render issues:

ISSUES: {json.dumps(issues_list)}
CURRENT PARAMS: {json.dumps(params)}
SCORES: {json.dumps(scores)}

DOCUMENTATION FOUND:
{json.dumps(doc_results, indent=2)}

Return JSON:
{{
  "diagnosis": "brief analysis",
  "modifications": {{
    // ONLY these keys are valid:
    // "resolution": int,
    // "frame_end": int,
    // "turbulence": float 0-1,
    // "vorticity": float 0-1,
    // "temperature": float (Kelvin),
    // "domain_scale": float,
    // "custom_code": {{"search_string": "replacement_string"}}
  }},
  "rationale": "why these changes will help",
  "confidence": 0.0-1.0,
  "doc_paths_used": ["paths", "from", "docs"]
}}

CONSTRAINTS:
- Max 2 parameter changes + 2 custom_code replacements
- Prefer small incremental changes
- If unsure, suggest safe diagnostic change"""

    result, cost = call_gpt52_vision(
        system_prompt,
        user_prompt,
        reasoning_effort="medium",  # Doc synthesis needs some reasoning
        verbosity="low"
    )

    record_spend("doc", cost)

    result["source"] = "gpt52_synthesis"
    result["cost"] = cost
    result["doc_research"] = doc_results

    return json.dumps(result, indent=2)


@mcp.tool()
def get_budget_status() -> str:
    """
    Get current budget status for the month.

    Returns:
        JSON with vision_spent, doc_spent, remaining, and can_afford flags
    """
    state = load_budget()

    return json.dumps({
        "month": state.month,
        "vision": {
            "spent": round(state.vision_spent, 4),
            "budget": VISION_BUDGET,
            "remaining": round(VISION_BUDGET - state.vision_spent, 4),
            "can_afford": can_afford_vision()
        },
        "doc": {
            "spent": round(state.doc_spent, 4),
            "budget": DOC_BUDGET,
            "remaining": round(DOC_BUDGET - state.doc_spent, 4),
            "can_afford": can_afford_docs()
        },
        "total": {
            "spent": round(state.vision_spent + state.doc_spent, 4),
            "budget": MONTHLY_BUDGET,
            "remaining": round(MONTHLY_BUDGET - state.vision_spent - state.doc_spent, 4)
        },
        "total_calls": state.total_calls,
        "last_updated": state.last_updated
    }, indent=2)


@mcp.tool()
def add_to_playbook(
    effect_type: str,
    symptom: str,
    fix: str,
    confidence: float = 0.7
) -> str:
    """
    Add a learned fix to the playbook for future FREE lookups.

    Call this when a GPT-suggested fix works well.

    Args:
        effect_type: Type of effect (sun, explosion, etc.)
        symptom: The issue this fix addresses (e.g., "too dark")
        fix: JSON of modifications that worked
        confidence: How confident we are this fix works (0-1)

    Returns:
        Confirmation JSON
    """
    # For now, store in a JSON file per effect type
    playbook_file = PLAYBOOK_DIR / f"{effect_type.lower()}_learned.json"

    if playbook_file.exists():
        learned = json.loads(playbook_file.read_text())
    else:
        learned = {"fixes": []}

    fix_dict = json.loads(fix) if isinstance(fix, str) else fix

    learned["fixes"].append({
        "symptom": symptom,
        "symptoms": [symptom.lower()],  # For matching
        "fix": fix_dict,
        "confidence": confidence,
        "added": datetime.now().isoformat()
    })

    playbook_file.write_text(json.dumps(learned, indent=2))

    return json.dumps({
        "status": "added",
        "effect_type": effect_type,
        "symptom": symptom,
        "playbook_file": str(playbook_file)
    }, indent=2)


@mcp.tool()
def search_docs_intelligent(
    query: str,
    effect_type: str = "",
    current_issues: str = "[]",
    current_params: str = "{}"
) -> str:
    """
    Intelligent documentation search using GPT-5.2 with tool calling.

    This is the RECOMMENDED tool for finding Blender documentation.
    GPT-5.2 autonomously decides which documentation tools to call
    based on your query, then synthesizes the results into actionable
    parameter modifications.

    Flow:
    1. GPT-5.2 receives your query
    2. GPT-5.2 calls search_manual, search_vdb_workflow, etc. as needed
    3. Results are fed back to GPT-5.2
    4. GPT-5.2 synthesizes final answer with parameter changes

    Args:
        query: Natural language question (e.g., "How to add limb darkening?")
        effect_type: Type of effect (sun, star, explosion, etc.)
        current_issues: JSON array of current issues
        current_params: JSON of current Blender parameters

    Returns:
        JSON with answer, modifications, doc_paths_used, and cost

    Cost: ~$0.02-0.05 per query (1-5 tool calls)
    """
    import asyncio

    # Check budget
    if not can_afford_docs(estimated_cost=0.05):
        budget = load_budget()
        return json.dumps({
            "error": "Doc budget exhausted",
            "doc_spent": budget.doc_spent,
            "doc_budget": DOC_BUDGET,
            "suggestion": "Use get_modification_advice with playbook lookup"
        }, indent=2)

    # Parse inputs
    issues_list = json.loads(current_issues) if current_issues and current_issues != "[]" else None
    params_dict = json.loads(current_params) if current_params and current_params != "{}" else None

    # Run the agent
    async def run_agent():
        return await search_docs_with_tool_calling(
            query=query,
            effect_type=effect_type,
            current_issues=issues_list,
            current_params=params_dict
        )

    # Run in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context
            import nest_asyncio
            nest_asyncio.apply()
        result = asyncio.run(run_agent())
    except RuntimeError:
        # No event loop, create one
        result = asyncio.run(run_agent())

    # Record spend
    if result.get("cost", 0) > 0:
        record_spend("doc", result["cost"])

    # Add source
    result["source"] = "gpt52_tool_calling"

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run()
