# Blender Librarian Agent - FINAL Implementation Plan

**Version:** 2.0 (GPT-5.2 Vision Librarian with Responses API)
**Date:** 2026-01-05
**Status:** Ready for Implementation

---

## Executive Summary

This plan creates a **Blender Vision Librarian** that combines:
1. **Local document retrieval** (FREE) via existing `blender-manual` MCP server
2. **GPT-5.2 vision analysis** (PAID) for render-vs-reference diagnosis
3. **Constrained patch outputs** compatible with `script-generator.modify_script()`

**Key Architecture Decisions:**
- **GPT-5.2** (released December 11, 2025) - current SOTA model
- **Responses API** - OpenAI's recommended modern API (Chat Completions is legacy)
- **Local MCP Tool Calling** - free, already built (NOT OpenAI Vector Store)
- **512×512 resized images** - token savings for vision calls

**Budget:** $20/month → $14 documentation synthesis, $6 vision analysis

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Claude Code Session                              │
│  (orchestrates via SKILL.md, calls MCP tools)                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   blender-librarian MCP Server                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Playbook Check  │→ │ Doc Retrieval   │→ │ GPT-5.2 Synthesis   │  │
│  │ (JSON, FREE)    │  │ (Local, FREE)   │  │ (Responses API)     │  │
│  └─────────────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│                                │                       │             │
│                                ▼                       ▼             │
│                   ┌─────────────────────────────────────────────┐   │
│                   │ MCP Tool: blender-manual.search_manual()    │   │
│                   │ MCP Tool: blender-manual.search_python_api()│   │
│                   │ MCP Tool: log-analysis-rag.query_logs()     │   │
│                   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   script-generator MCP Server                        │
│  modify_script(modifications={...})  ← Librarian output applied     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principle:** The Librarian doesn't hold data—it **asks specialists** via MCP tools.

---

## Pre-Implementation Checklist

### Already Complete ✅
- [x] `agents/blender-manual/` exists with semantic search (`embeddings.npy`, 11.9MB index)
- [x] `agents/log-analysis-rag/` exists with BM25 + FAISS hybrid retrieval
- [x] `agents/blender-librarian/` directory created
- [x] `script-generator` has `modify_script()` MCP tool
- [x] `experiment-tracker` has knowledge base with learnings

### Needs Setup ⚠️
- [ ] OpenAI API key configuration (USER ACTION)
- [ ] Budget tracking file creation
- [ ] Playbook JSON initialization
- [ ] Ground truth cache directory
- [ ] MCP server registration in Claude Code settings

---

## Phase 0: Environment Setup

### Task 0.1: OpenAI API Key Setup

**🔧 USER ACTION REQUIRED**

1. **Get your OpenAI API key:**
   - Go to: https://platform.openai.com/api-keys
   - Create a new secret key (name it "PlasmaDXR-Librarian")
   - Copy the key (starts with `sk-proj-...`)

2. **Create environment file:**
   ```bash
   # Create the secrets file (this should NOT be committed to git)
   touch ~/.plasmadxr_secrets
   chmod 600 ~/.plasmadxr_secrets

   # Add your key
   echo 'OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE' >> ~/.plasmadxr_secrets
   ```

3. **Add to shell profile:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   echo 'source ~/.plasmadxr_secrets' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Verify it works:**
   ```bash
   echo $OPENAI_API_KEY | head -c 20
   # Should show: sk-proj-XXXXXXXXXX
   ```

### Task 0.2: Install Python Dependencies

**🔧 USER ACTION REQUIRED**

```bash
cd /home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install openai>=1.0.0 tenacity>=8.0.0 pillow>=10.0.0 mcp>=1.0.0

# Verify installation
python -c "import openai; print(f'OpenAI SDK: {openai.__version__}')"
```

### Task 0.3: Create Directory Structure

```bash
cd /home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian

# Create required directories
mkdir -p cache/ground_truth_reports
mkdir -p playbooks
mkdir -p logs

# Initialize budget tracking
echo '{"total_budget": 20.0, "docs_budget": 14.0, "vision_budget": 6.0, "docs_spent": 0.0, "vision_spent": 0.0, "calls": []}' > budget_tracker.json

# Initialize empty playbook
echo '{"known_fixes": [], "symptom_queries": {}, "version": "1.0.0"}' > playbooks/solar_playbook.json
```

---

## Phase 1: Core Server Implementation

### Task 1.1: Create Main Server File

**File:** `agents/blender-librarian/server.py`

```python
"""
Blender Librarian MCP Server

Vision-assisted documentation synthesis for Blender 5.0.1 scripting.
Uses GPT-5.2 for vision analysis, local MCP tools for doc retrieval.
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from mcp.server import Server
from mcp.types import Tool, TextContent
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import io

# ============================================================================
# Configuration
# ============================================================================

# GPT-5.2 released December 11, 2025 - current SOTA
# Variants: gpt-5.2 (Thinking), gpt-5.2-pro (most intelligent), gpt-5.2-chat-latest (Instant)
OPENAI_MODEL = "gpt-5.2"  # GPT-5.2 Thinking - best for complex reasoning
OPENAI_MODEL_PRO = "gpt-5.2-pro"  # For most demanding vision tasks (Responses API only)
MAX_IMAGE_SIZE = 512  # Resize to 512x512 max
BUDGET_FILE = Path(__file__).parent / "budget_tracker.json"
CACHE_DIR = Path(__file__).parent / "cache" / "ground_truth_reports"
PLAYBOOK_DIR = Path(__file__).parent / "playbooks"

# Cost estimates (USD per 1M tokens, as of Jan 2026 - GPT-5.2 pricing)
# GPT-5.2: $1.75/1M input, $14/1M output, 90% discount on cached inputs
COST_INPUT_1M = 1.75     # GPT-5.2 input per 1M tokens
COST_OUTPUT_1M = 14.0    # GPT-5.2 output per 1M tokens
COST_INPUT_1K = COST_INPUT_1M / 1000   # = $0.00175 per 1K tokens
COST_OUTPUT_1K = COST_OUTPUT_1M / 1000  # = $0.014 per 1K tokens
COST_IMAGE_LOW = 0.00085  # Low detail image (85 tokens)
COST_IMAGE_HIGH_TILE = 0.00085  # Per 512x512 tile

# ============================================================================
# Budget Tracker
# ============================================================================

@dataclass
class BudgetState:
    total_budget: float = 20.0
    docs_budget: float = 14.0
    vision_budget: float = 6.0
    docs_spent: float = 0.0
    vision_spent: float = 0.0
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    @property
    def docs_remaining(self) -> float:
        return self.docs_budget - self.docs_spent

    @property
    def vision_remaining(self) -> float:
        return self.vision_budget - self.vision_spent

    def can_spend_docs(self, amount: float) -> bool:
        return self.docs_spent + amount <= self.docs_budget

    def can_spend_vision(self, amount: float) -> bool:
        return self.vision_spent + amount <= self.vision_budget

    def record_call(self, call_type: str, cost: float, description: str):
        self.calls.append({
            "timestamp": datetime.now().isoformat(),
            "type": call_type,
            "cost": cost,
            "description": description
        })
        if call_type == "vision":
            self.vision_spent += cost
        else:
            self.docs_spent += cost

def load_budget() -> BudgetState:
    if BUDGET_FILE.exists():
        data = json.loads(BUDGET_FILE.read_text())
        return BudgetState(**data)
    return BudgetState()

def save_budget(state: BudgetState):
    BUDGET_FILE.write_text(json.dumps(asdict(state), indent=2))

# ============================================================================
# Image Processing
# ============================================================================

def resize_image_for_vision(image_path: str) -> str:
    """Resize image to max 512x512 and return base64."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Resize maintaining aspect ratio
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def hash_image(image_path: str) -> str:
    """Create hash for caching."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

# ============================================================================
# Ground Truth Cache
# ============================================================================

class GroundTruthCache:
    """Cache ground truth analysis to avoid repeated vision calls."""

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_cached(self, image_path: str) -> Optional[dict]:
        cache_key = hash_image(image_path)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def save_cache(self, image_path: str, report: dict):
        cache_key = hash_image(image_path)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        cache_file.write_text(json.dumps(report, indent=2))

# ============================================================================
# Playbook System
# ============================================================================

def load_playbook(effect_type: str) -> dict:
    """Load known fixes for an effect type."""
    playbook_file = PLAYBOOK_DIR / f"{effect_type}_playbook.json"
    if playbook_file.exists():
        return json.loads(playbook_file.read_text())
    return {"known_fixes": [], "symptom_queries": {}}

def check_playbook(effect_type: str, issues: list[str]) -> Optional[dict]:
    """Check if any issues have known fixes (FREE)."""
    playbook = load_playbook(effect_type)

    for issue in issues:
        issue_lower = issue.lower()
        for fix in playbook.get("known_fixes", []):
            if fix["symptom"].lower() in issue_lower:
                return {
                    "source": "playbook",
                    "cost": 0.0,
                    "fix": fix
                }
    return None

# ============================================================================
# OpenAI Client
# ============================================================================

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=sk-proj-..."
        )
    return OpenAI(api_key=api_key)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_gpt52_vision(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    image_base64: Optional[str] = None,
    reasoning_effort: str = "low",
    verbosity: str = "low"
) -> tuple[str, float]:
    """
    Call GPT-5.2 with optional image. Returns (response, estimated_cost).
    Uses Responses API (the recommended modern API).

    Per OpenAI docs (2025): "While Chat Completions remains supported,
    Responses is recommended for all new projects."

    Key differences from Chat Completions:
    - Uses client.responses.create() instead of client.chat.completions.create()
    - Uses 'input' instead of 'messages'
    - Uses 'instructions' for system-level guidance
    - Response has 'output_text' helper instead of choices[0].message.content
    - Vision uses 'input_text' and 'input_image' types (not 'text' and 'image_url')
    - 3% improvement in SWE-bench, 40-80% better cache utilization

    GPT-5.2 specific parameters:
    - reasoning_effort: none/low/medium/high/xhigh (lower = faster + cheaper)
    - verbosity: low/medium/high (lower = fewer output tokens)
    """
    # Build input content with correct Responses API format
    if image_base64:
        # Vision requires structured input with role
        # Per IMAGES_AND_VISION.md: use 'input_text' and 'input_image' types
        input_data = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"  # 85 tokens - cost effective
                }
            ]
        }]
    else:
        # Text-only can use simple string
        input_data = user_prompt

    # Use Responses API with GPT-5.2 optimizations
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,  # System prompt goes in 'instructions'
        input=input_data,            # User content goes in 'input'
        reasoning={"effort": reasoning_effort},  # GPT-5.2: none/low/medium/high/xhigh
        text={"verbosity": verbosity},           # GPT-5.2: low/medium/high
        store=False,                 # Don't persist for privacy
        temperature=0.3              # Lower temp for consistent outputs
    )

    # Estimate cost (GPT-5.2 pricing: $1.75/1M input, $14/1M output)
    # Note: Responses API usage structure may differ - using fallbacks
    usage = getattr(response, 'usage', None)
    if usage:
        input_tokens = getattr(usage, 'input_tokens', 500)
        output_tokens = getattr(usage, 'output_tokens', 200)
    else:
        input_tokens = 500  # Conservative estimate
        output_tokens = 200

    cost = (input_tokens / 1000 * COST_INPUT_1K) + (output_tokens / 1000 * COST_OUTPUT_1K)
    if image_base64:
        cost += COST_IMAGE_LOW  # Add image cost (85 tokens for low detail)

    # Use output_text helper (Responses API convenience method)
    return response.output_text, cost

# ============================================================================
# MCP Server
# ============================================================================

server = Server("blender-librarian")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="diagnose_render_issue",
            description="Diagnose visual issues in a Blender render by comparing to reference",
            inputSchema={
                "type": "object",
                "properties": {
                    "render_path": {
                        "type": "string",
                        "description": "Path to the current render image"
                    },
                    "reference_path": {
                        "type": "string",
                        "description": "Path to reference image (optional)"
                    },
                    "effect_type": {
                        "type": "string",
                        "description": "Effect type: sun, explosion, nebula, etc."
                    },
                    "current_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Known issues from evaluator"
                    }
                },
                "required": ["render_path", "effect_type"]
            }
        ),
        Tool(
            name="get_modification_advice",
            description="Get script modification advice based on diagnosed issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Diagnosed issues"
                    },
                    "effect_type": {
                        "type": "string",
                        "description": "Effect type"
                    },
                    "current_params": {
                        "type": "object",
                        "description": "Current script parameters"
                    }
                },
                "required": ["issues", "effect_type"]
            }
        ),
        Tool(
            name="get_budget_status",
            description="Get current API budget usage",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="add_to_playbook",
            description="Add a successful fix to the playbook for future reuse",
            inputSchema={
                "type": "object",
                "properties": {
                    "effect_type": {"type": "string"},
                    "symptom": {"type": "string"},
                    "fix": {"type": "object"},
                    "confidence": {"type": "number"}
                },
                "required": ["effect_type", "symptom", "fix"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    budget = load_budget()

    if name == "get_budget_status":
        return [TextContent(
            type="text",
            text=json.dumps({
                "docs_remaining": f"${budget.docs_remaining:.2f}",
                "vision_remaining": f"${budget.vision_remaining:.2f}",
                "total_spent": f"${budget.docs_spent + budget.vision_spent:.2f}",
                "recent_calls": budget.calls[-5:] if budget.calls else []
            }, indent=2)
        )]

    elif name == "diagnose_render_issue":
        render_path = arguments["render_path"]
        reference_path = arguments.get("reference_path")
        effect_type = arguments["effect_type"]
        current_issues = arguments.get("current_issues", [])

        # Step 1: Check playbook first (FREE)
        playbook_fix = check_playbook(effect_type, current_issues)
        if playbook_fix:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "source": "playbook",
                    "cost": 0.0,
                    "diagnosis": playbook_fix["fix"]["diagnosis"],
                    "recommended_fix": playbook_fix["fix"]["modifications"]
                }, indent=2)
            )]

        # Step 2: Check budget
        estimated_cost = 0.02  # ~$0.02 per vision call with low detail
        if not budget.can_spend_vision(estimated_cost):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Vision budget exhausted",
                    "vision_remaining": budget.vision_remaining,
                    "suggestion": "Use get_modification_advice with manual issue description"
                }, indent=2)
            )]

        # Step 3: Check ground truth cache
        cache = GroundTruthCache()
        if reference_path:
            cached_report = cache.get_cached(reference_path)
            if cached_report:
                # Use cached reference analysis (FREE)
                reference_analysis = cached_report
            else:
                reference_analysis = None
        else:
            reference_analysis = None

        # Step 4: Resize and analyze render
        render_b64 = resize_image_for_vision(render_path)

        client = get_openai_client()

        system_prompt = """You are a Blender volumetric rendering expert. Analyze the provided render image and identify specific visual issues.

Focus on:
- Brightness/exposure (too dark, too bright, blown out highlights)
- Color temperature (too warm/orange, too cool/blue)
- Limb darkening (edges should be darker for stars/suns)
- Texture/granulation quality
- Prominence/corona visibility (for solar effects)
- Dynamic range

Output JSON only:
{
  "discrepancies": [
    {
      "type": "brightness|color_temperature|limb_darkening|texture|corona|prominence",
      "severity": "low|medium|high|critical",
      "evidence": "specific observation",
      "likely_blender_causes": ["cause1", "cause2"]
    }
  ],
  "overall_assessment": "brief summary"
}"""

        user_prompt = f"Analyze this {effect_type} render. Known issues from evaluator: {current_issues}"
        if reference_analysis:
            user_prompt += f"\n\nReference analysis (cached): {json.dumps(reference_analysis)}"

        response, cost = call_gpt52_vision(
            client,
            system_prompt,
            user_prompt,
            render_b64,
            reasoning_effort="low",    # Visual analysis - no deep reasoning needed
            verbosity="low"            # We want JSON output, not verbose text
        )

        # Record cost
        budget.record_call("vision", cost, f"diagnose_{effect_type}")
        save_budget(budget)

        # Cache reference analysis if we did one
        if reference_path and not reference_analysis:
            try:
                parsed = json.loads(response)
                cache.save_cache(reference_path, parsed)
            except json.JSONDecodeError:
                pass

        return [TextContent(
            type="text",
            text=json.dumps({
                "source": "gpt52_vision",
                "cost": cost,
                "diagnosis": response
            }, indent=2)
        )]

    elif name == "get_modification_advice":
        issues = arguments["issues"]
        effect_type = arguments["effect_type"]
        current_params = arguments.get("current_params", {})

        # Step 1: Check playbook (FREE)
        playbook_fix = check_playbook(effect_type, issues)
        if playbook_fix:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "source": "playbook",
                    "cost": 0.0,
                    "modifications": playbook_fix["fix"]["modifications"],
                    "rationale": playbook_fix["fix"]["rationale"]
                }, indent=2)
            )]

        # Step 2: Check budget
        estimated_cost = 0.01  # ~$0.01 for text-only synthesis
        if not budget.can_spend_docs(estimated_cost):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Documentation budget exhausted",
                    "docs_remaining": budget.docs_remaining
                }, indent=2)
            )]

        # Step 3: Build queries for blender-manual
        # (In production, these would call the actual MCP tool)
        queries = []
        for issue in issues:
            if "dark" in issue.lower() or "bright" in issue.lower():
                queries.append("Principled Volume emission blackbody intensity")
                queries.append("color management exposure view transform")
            if "color" in issue.lower() or "warm" in issue.lower():
                queries.append("blackbody temperature color mapping")
                queries.append("AgX Filmic view transform")
            if "limb" in issue.lower():
                queries.append("volumetric density falloff radial gradient")

        # Step 4: Synthesize with GPT-5.2 (text only, no image)
        client = get_openai_client()

        system_prompt = """You are a Blender scripting expert. Given issues and current parameters, suggest SAFE modifications compatible with script-generator.modify_script().

ONLY suggest these modification types:
- resolution: int
- frame_end: int
- turbulence: float (0-1)
- temperature: float
- domain_scale: float
- density: float
- custom_code: {"search_pattern": "...", "replacement": "..."}

Output JSON only:
{
  "modifications": {
    "parameter_name": new_value,
    ...
  },
  "rationale": "why these changes should help",
  "confidence": 0.0-1.0
}"""

        user_prompt = f"""Effect type: {effect_type}
Issues: {json.dumps(issues)}
Current parameters: {json.dumps(current_params)}
Relevant doc queries to run: {json.dumps(queries)}

Suggest 1-2 safe modifications maximum."""

        response, cost = call_gpt52_vision(
            client,
            system_prompt,
            user_prompt,
            image_base64=None,
            reasoning_effort="medium",  # May need reasoning for complex fixes
            verbosity="low"             # We want JSON output, not verbose text
        )

        budget.record_call("docs", cost, f"advice_{effect_type}")
        save_budget(budget)

        return [TextContent(
            type="text",
            text=json.dumps({
                "source": "gpt52_synthesis",
                "cost": cost,
                "advice": response,
                "doc_queries_used": queries
            }, indent=2)
        )]

    elif name == "add_to_playbook":
        effect_type = arguments["effect_type"]
        symptom = arguments["symptom"]
        fix = arguments["fix"]
        confidence = arguments.get("confidence", 0.8)

        playbook_file = PLAYBOOK_DIR / f"{effect_type}_playbook.json"

        if playbook_file.exists():
            playbook = json.loads(playbook_file.read_text())
        else:
            playbook = {"known_fixes": [], "symptom_queries": {}, "version": "1.0.0"}

        playbook["known_fixes"].append({
            "symptom": symptom,
            "modifications": fix,
            "confidence": confidence,
            "added": datetime.now().isoformat()
        })

        playbook_file.write_text(json.dumps(playbook, indent=2))

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "added",
                "playbook": f"{effect_type}_playbook.json",
                "total_fixes": len(playbook["known_fixes"])
            }, indent=2)
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

# ============================================================================
# Entry Point
# ============================================================================

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Phase 2: Claude Code Integration

### Task 2.1: Register MCP Server

**🔧 USER ACTION REQUIRED**

Add to your Claude Code MCP settings file (`~/.claude/settings.json` or project-level):

```json
{
  "mcpServers": {
    "blender-librarian": {
      "command": "/home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian/venv/bin/python",
      "args": ["-m", "mcp.server.stdio", "/home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian/server.py"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

**Alternative: Create run script:**

```bash
# Create run_server.sh
cat > /home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian/run_server.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
source ~/.plasmadxr_secrets
python -m mcp.server.stdio server.py
EOF

chmod +x /home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian/run_server.sh
```

### Task 2.2: Update Blender Orchestrator SKILL.md

Add librarian escalation to `.claude/skills/blender-orchestrator/SKILL.md`:

```markdown
### Stage 6: Librarian Escalation (When Stuck)

**Trigger conditions:**
- 2+ iterations without improvement
- Repeated same issue category
- Unknown failure pattern

**Escalation flow:**
1. Check budget: `mcp__blender-librarian__get_budget_status()`
2. If vision budget available AND have reference:
   ```
   mcp__blender-librarian__diagnose_render_issue(
     render_path=latest_render,
     reference_path=reference_image,
     effect_type=current_effect,
     current_issues=evaluator_issues
   )
   ```
3. Get modification advice:
   ```
   mcp__blender-librarian__get_modification_advice(
     issues=diagnosed_issues,
     effect_type=current_effect,
     current_params=current_script_params
   )
   ```
4. Apply modifications via `script-generator.modify_script()`
5. If successful, add to playbook:
   ```
   mcp__blender-librarian__add_to_playbook(
     effect_type=effect,
     symptom=original_issue,
     fix=successful_modifications
   )
   ```
```

---

## Phase 3: Initial Playbook Population

### Task 3.1: Create Solar Playbook

**File:** `agents/blender-librarian/playbooks/solar_playbook.json`

```json
{
  "version": "1.0.0",
  "known_fixes": [
    {
      "symptom": "blackbody blowout",
      "diagnosis": "Temperature attribute exceeds blackbody safe range",
      "modifications": {
        "custom_code": {
          "search_pattern": "blackbody_intensity = ",
          "replacement": "blackbody_intensity = 0.3  # Reduced from 1.0"
        }
      },
      "rationale": "Blackbody emission uses temperature^4, causing rapid blowout above ~6000K",
      "confidence": 0.9
    },
    {
      "symptom": "too orange",
      "diagnosis": "Color temperature too low for solar effect",
      "modifications": {
        "temperature": 5778.0,
        "custom_code": {
          "search_pattern": "color_temperature = \\d+",
          "replacement": "color_temperature = 5778  # Solar temperature"
        }
      },
      "rationale": "Sun's photosphere is ~5778K, lower temps appear orange/red",
      "confidence": 0.85
    },
    {
      "symptom": "no limb darkening",
      "diagnosis": "Density distribution lacks radial falloff",
      "modifications": {
        "custom_code": {
          "search_pattern": "density_scale = ([\\d.]+)",
          "replacement": "density_scale = 1.5  # Increased for limb effect"
        }
      },
      "rationale": "Limb darkening requires density gradient from center to edge",
      "confidence": 0.7
    },
    {
      "symptom": "viewport vs F12 mismatch",
      "diagnosis": "Color management or sampling difference",
      "modifications": {
        "custom_code": {
          "search_pattern": "scene.render.engine",
          "replacement": "scene.render.engine = 'CYCLES'  # Ensure CYCLES"
        }
      },
      "rationale": "Viewport uses EEVEE approximation, F12 uses full CYCLES",
      "confidence": 0.6
    }
  ],
  "symptom_queries": {
    "brightness": ["Principled Volume emission", "color management exposure"],
    "color": ["blackbody temperature", "view transform AgX Filmic"],
    "texture": ["noise texture granulation", "volumetric density variation"],
    "corona": ["emission falloff", "outer glow volumetric"]
  }
}
```

---

## Phase 4: Testing and Validation

### Task 4.1: Manual Test Script

**🔧 USER ACTION REQUIRED - Run this to verify setup:**

```bash
cd /home/maz3ppa/projects/PlasmaDXR/agents/blender-librarian
source venv/bin/activate

# Test 1: Verify OpenAI connection (using Responses API)
python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
# Using Responses API - the recommended modern API
# With GPT-5.2 optimization parameters
response = client.responses.create(
    model='gpt-5.2',
    input='Say hello',
    reasoning={'effort': 'none'},  # Simple task
    text={'verbosity': 'low'}      # Short response
)
print('OpenAI connection:', 'OK' if response.output_text else 'FAILED')
print('Response:', response.output_text)
"

# Test 2: Verify budget tracking
python -c "
from server import load_budget, save_budget
budget = load_budget()
print(f'Docs remaining: \${budget.docs_remaining:.2f}')
print(f'Vision remaining: \${budget.vision_remaining:.2f}')
"

# Test 3: Verify playbook loading
python -c "
from server import load_playbook, check_playbook
playbook = load_playbook('solar')
print(f'Solar playbook: {len(playbook.get(\"known_fixes\", []))} known fixes')
result = check_playbook('solar', ['blackbody blowout detected'])
print(f'Playbook match: {\"FOUND\" if result else \"none\"}')
"
```

### Task 4.2: Integration Test

After Claude Code is configured, test the MCP integration:

```
/librarian diagnose --render build/vdb_output/sun_test/render_0001.png --effect sun --issues "too orange, lacks structure"
```

Expected behavior:
1. Checks playbook first (free)
2. If no match, checks budget
3. If budget available, calls GPT-5.2
4. Returns structured modifications

---

## Phase 5: Monitoring and Optimization

### Task 5.1: Budget Monitoring

Check budget regularly:
```
mcp__blender-librarian__get_budget_status()
```

### Task 5.2: Playbook Growth

After each successful fix, add to playbook to avoid future API costs:
```
mcp__blender-librarian__add_to_playbook(
  effect_type="sun",
  symptom="the specific symptom",
  fix={"the": "modifications that worked"},
  confidence=0.8
)
```

---

## Summary of Manual Setup Steps

**🔧 USER ACTIONS CHECKLIST:**

1. [ ] Create OpenAI API key at https://platform.openai.com/api-keys
2. [ ] Add key to `~/.plasmadxr_secrets`
3. [ ] Source secrets in shell profile
4. [ ] Create virtual environment and install dependencies
5. [ ] Create directory structure (cache, playbooks, logs)
6. [ ] Initialize budget_tracker.json
7. [ ] Register MCP server in Claude Code settings
8. [ ] Run test script to verify setup
9. [ ] Test MCP integration in Claude Code

**Estimated setup time:** 15-20 minutes

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `agents/blender-librarian/server.py` | To Create | Main MCP server |
| `agents/blender-librarian/run_server.sh` | To Create | Launch script |
| `agents/blender-librarian/budget_tracker.json` | To Create | Cost tracking |
| `agents/blender-librarian/playbooks/solar_playbook.json` | To Create | Known fixes |
| `agents/blender-librarian/cache/` | To Create | Ground truth cache |
| `~/.plasmadxr_secrets` | USER ACTION | API key storage |
| `~/.claude/settings.json` | USER ACTION | MCP registration |

---

## Cost Projections

| Usage Pattern | Monthly Cost | Notes |
|---------------|--------------|-------|
| Light (5 vision calls) | ~$1.50 | Playbook handles most cases |
| Medium (15 vision calls) | ~$4.50 | Some novel issues |
| Heavy (30 vision calls) | ~$9.00 | Many unique problems |

**Budget safety:** Hard caps prevent overspend. Playbook learning reduces costs over time.

---

## Related Documents

- `docs/GPT52_BLENDER_LIBRARIAN_MERGED_DESIGN.md` - Full architectural design
- `docs/FEEDBACK_AND_CORRECTIONS_LIBRARIAN_AGENT.md` - Gemini corrections
- `docs/BLENDER_LIBRARIAN_AGENT_DESIGN_GPT52.md` - Original GPT analysis
- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Pipeline context
