# Multi-Agent Pipeline Improvement Plan v4

**Extends:** v3 (`docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md`)
**Focus:** GPT-5.2 Integration with Blender Documentation via Tool Calling & Agents SDK
**Priority:** OpenAI work highest priority per user request
**Created:** 2026-01-05
**Status:** IN PROGRESS

---

## Executive Summary

This plan extends v3 by implementing two approaches for connecting GPT-5.2 to the existing `blender-manual` MCP server (12 tools, ~2,500 pages):

1. **Phase 8: Tool Calling Integration** (Priority 1) - Direct OpenAI Responses API with function definitions
2. **Phase 9: OpenAI Agents SDK Integration** (Priority 2) - Multi-agent orchestration framework

Both approaches enable GPT-5.2 to intelligently search Blender documentation and provide accurate, cited guidance for VFX asset creation.

---

## Quick Status (Inherited from v3)

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Phase 0-4 | ✅ COMPLETE | Foundation, validation, technique selection, knowledge base |
| Phase 5 | 🔄 PARTIAL | Evaluation reliability (Task 5.4 done) |
| Phase 6 | ⏳ PENDING | Session resumption |
| Phase 7 | ✅ COMPLETE | Blender Librarian MCP server with GPT-5.2 vision |
| **Phase 8** | ✅ COMPLETE | Tool Calling integration with blender-manual |
| **Phase 9** | 🔄 IN PROGRESS | OpenAI Agents SDK multi-agent framework (automation + persistence) |

---

## Architecture Overview

### Current State (v3 + Phase 7)

```
blender-orchestrator SKILL.md
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   MCP Server Layer                       │
├──────────────┬──────────────┬──────────────┬────────────┤
│ script-      │ blender-     │ asset-       │ blender-   │
│ generator    │ executor     │ evaluator    │ librarian  │
│ (12 tools)   │ (5 tools)    │ (25 tools)   │ (4 tools)  │
└──────────────┴──────────────┴──────────────┴────────────┘
                                                    │
                                              GPT-5.2 Vision
                                              (PAID, $6/mo)
```

### Target State (v4)

```
blender-orchestrator SKILL.md
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   MCP Server Layer                       │
├──────────────┬──────────────┬──────────────┬────────────┤
│ script-      │ blender-     │ asset-       │ blender-   │
│ generator    │ executor     │ evaluator    │ librarian  │
└──────────────┴──────────────┴──────────────┴────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │                               │
                              ┌─────▼─────┐                   ┌─────▼─────┐
                              │ GPT-5.2   │                   │ blender-  │
                              │ Reasoning │◄──────────────────│ manual    │
                              │ + Vision  │   Tool Calling    │ MCP       │
                              └───────────┘   (Phase 8)       │ (12 tools)│
                                    │         OR              └───────────┘
                                    │         Agents SDK
                                    │         (Phase 9)
                              ┌─────▼─────┐
                              │ Playbook  │
                              │ (FREE)    │
                              └───────────┘
```

---

## Phase 8: Tool Calling Integration (Priority 1)

### Objective

Enable GPT-5.2 to directly call `blender-manual` MCP server tools via OpenAI Responses API function calling.

### Why Tool Calling First?

| Factor | Tool Calling | Agents SDK |
|--------|--------------|------------|
| **Complexity** | Low - direct API calls | Medium - framework learning curve |
| **Dependencies** | `openai` package (existing) | New `openai-agents` package |
| **Control** | Full control over tool execution | Framework-managed execution |
| **Debugging** | Simple - single API call chain | Complex - multi-agent traces |
| **Cost Control** | Manual budget checks | Requires custom implementation |

### Task 8.1: Tool Definitions for blender-manual

**File:** `agents/blender-librarian/tools/blender_manual_tools.py`

Define OpenAI function schemas for all 12 blender-manual MCP tools:

```python
BLENDER_MANUAL_TOOLS = [
    {
        "type": "function",
        "name": "search_manual",
        "description": "Search the Blender 5.0 Manual for documentation. Use for general Blender concepts, UI, rendering, physics.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords, phrases)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_python_api",
        "description": "Search Blender Python API documentation (bpy.ops, bpy.types, bpy.data). Use for scripting questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "API operation (e.g., 'bpy.ops.fluid.bake_all', 'FluidDomainSettings')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_vdb_workflow",
        "description": "Search for VDB/OpenVDB/NanoVDB workflow documentation. Use for volumetric export, caching, Mantaflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "VDB-related query (e.g., 'export vdb', 'cache smoke simulation')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_tutorials",
        "description": "Search for tutorials and getting started guides. Use for learning resources and step-by-step workflows.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Main topic (e.g., 'volumetrics', 'fluid simulation')"
                },
                "technique": {
                    "type": "string",
                    "description": "Specific technique (e.g., 'mantaflow', 'smoke')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["topic"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_nodes",
        "description": "Search for shader, compositor, or geometry nodes documentation.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "description": "Node name (e.g., 'Principled Volume', 'Volume Scatter')"
                },
                "category": {
                    "type": "string",
                    "enum": ["shader", "compositor", "geometry"],
                    "description": "Node category"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["node_type"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "read_page",
        "description": "Read the full content of a Blender manual or API page. Use after search to get detailed information.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to HTML file (from search results)"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters (default 4000)",
                    "default": 4000
                }
            },
            "required": ["path"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_modifiers",
        "description": "Search for modifier documentation (mesh, volume, simulation modifiers).",
        "parameters": {
            "type": "object",
            "properties": {
                "modifier_name": {
                    "type": "string",
                    "description": "Modifier name (e.g., 'Fluid', 'Volume Displace')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_semantic",
        "description": "AI-powered semantic search. Use for natural language queries when keyword search fails.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'how to make realistic smoke effects')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "browse_hierarchy",
        "description": "Browse the manual's hierarchical structure. Use to discover content organization.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path prefix (e.g., 'render', 'physics/fluid')"
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "list_api_modules",
        "description": "List available Python API modules.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Module category (e.g., 'bpy', 'bmesh')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum modules (default 30)",
                    "default": 30
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_bpy_operators",
        "description": "Search for bpy.ops.* operators by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Operator category (e.g., 'mesh', 'fluid', 'object')"
                },
                "operation": {
                    "type": "string",
                    "description": "Specific operation name"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["category"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_bpy_types",
        "description": "Search for bpy.types.* types by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "typename": {
                    "type": "string",
                    "description": "Type name (e.g., 'FluidModifier', 'Volume', 'Particle')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                }
            },
            "required": ["typename"],
            "additionalProperties": False
        },
        "strict": True
    }
]
```

### Task 8.2: MCP Bridge Client

**File:** `agents/blender-librarian/tools/mcp_bridge.py`

Bridge between GPT-5.2 tool calls and blender-manual MCP server:

```python
"""
MCP Bridge Client for blender-manual server.

Executes GPT-5.2 function calls against the blender-manual MCP server.
"""

import json
import asyncio
from typing import Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class MCPBridgeClient:
    """Bridge between GPT-5.2 tool calls and blender-manual MCP server."""

    MCP_TIMEOUT = 10.0  # seconds
    MAX_RETRIES = 2

    def __init__(self, mcp_base_url: str = "http://localhost:8001"):
        """Initialize with MCP server URL.

        Args:
            mcp_base_url: Base URL for blender-manual MCP server
        """
        self.mcp_base_url = mcp_base_url
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.mcp_base_url,
            timeout=self.MCP_TIMEOUT
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool and return the result as JSON string.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments

        Returns:
            JSON string result suitable for GPT-5.2 function_call_output

        Raises:
            httpx.HTTPError: On network/HTTP errors
            json.JSONDecodeError: On malformed response
        """
        if not self._client:
            raise RuntimeError("MCPBridgeClient not initialized. Use async with.")

        response = await self._client.post(
            "/tools/call",
            json={
                "name": tool_name,
                "arguments": arguments
            }
        )
        response.raise_for_status()
        result = response.json()

        # Format for GPT-5.2 consumption
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def check_health(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            response = await self._client.get("/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
```

### Task 8.3: GPT-5.2 Tool Calling Agent Loop

**File:** `agents/blender-librarian/tools/tool_calling_agent.py`

Main agent loop that orchestrates GPT-5.2 with tool calling:

```python
"""
GPT-5.2 Tool Calling Agent for Blender Documentation.

Implements the agentic loop pattern where GPT-5.2 can call
blender-manual tools to search documentation and provide
accurate, cited answers.
"""

import json
import os
from typing import Any
from openai import AsyncOpenAI

from .blender_manual_tools import BLENDER_MANUAL_TOOLS
from .mcp_bridge import MCPBridgeClient
from ..budget_tracker import BudgetTracker


class ToolCallingAgent:
    """GPT-5.2 agent with blender-manual tool access."""

    SYSTEM_INSTRUCTIONS = """You are a Blender documentation expert powered by GPT-5.2.

Your role is to help diagnose VFX rendering issues and recommend parameter fixes
by searching the official Blender 5.0 documentation.

When helping:
1. Search documentation using the available tools (search_manual, search_python_api, etc.)
2. Read relevant pages for detailed information using read_page
3. Provide accurate, cited answers with specific parameter recommendations
4. Format parameter modifications as JSON compatible with script-generator.modify_script()

Output Format for Modifications:
{
    "modifications": {
        "parameter_name": value,
        ...
    },
    "rationale": "Explanation of why these changes should help",
    "citations": ["path/to/doc1.html", "path/to/doc2.html"]
}

Be thorough but concise. Focus on actionable recommendations backed by documentation."""

    MAX_ITERATIONS = 10

    def __init__(
        self,
        openai_api_key: str | None = None,
        mcp_server_url: str = "http://localhost:8001",
        budget_tracker: BudgetTracker | None = None
    ):
        """Initialize the tool calling agent.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            mcp_server_url: URL of blender-manual MCP server
            budget_tracker: Budget tracker instance for cost control
        """
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.mcp_url = mcp_server_url
        self.budget = budget_tracker or BudgetTracker()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    async def run(
        self,
        user_query: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the agent loop until completion.

        Args:
            user_query: User's question or issue description
            context: Optional context (effect_type, current_params, etc.)

        Returns:
            Dict with 'response', 'modifications', 'citations', 'tool_calls_made'
        """
        # Check budget before starting
        if not self.budget.can_afford_doc_lookup():
            return {
                "error": "doc_budget_exhausted",
                "message": "Documentation lookup budget exhausted for this month",
                "response": None,
                "modifications": None
            }

        # Build initial input
        input_list = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTIONS}
        ]

        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            input_list.append({
                "role": "user",
                "content": f"Context:\n```json\n{context_str}\n```\n\nQuestion: {user_query}"
            })
        else:
            input_list.append({"role": "user", "content": user_query})

        tool_calls_made = []

        async with MCPBridgeClient(self.mcp_url) as mcp:
            # Check MCP server health
            if not await mcp.check_health():
                return {
                    "error": "mcp_unavailable",
                    "message": "blender-manual MCP server is unavailable",
                    "response": None,
                    "modifications": None
                }

            for iteration in range(self.MAX_ITERATIONS):
                # Call GPT-5.2 with tools
                response = await self.openai.responses.create(
                    model=self.model,
                    input=input_list,
                    tools=BLENDER_MANUAL_TOOLS,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=2000
                )

                # Track cost
                self.budget.record_doc_lookup()

                # Check for function calls
                function_calls = [
                    item for item in response.output
                    if hasattr(item, 'type') and item.type == "function_call"
                ]

                if not function_calls:
                    # No more tool calls - extract final answer
                    return self._extract_final_response(
                        response.output_text,
                        tool_calls_made
                    )

                # Add model output to conversation
                input_list.extend([
                    item.model_dump() if hasattr(item, 'model_dump') else item
                    for item in response.output
                ])

                # Execute each tool call
                for call in function_calls:
                    arguments = (
                        json.loads(call.arguments)
                        if isinstance(call.arguments, str)
                        else call.arguments
                    )

                    tool_calls_made.append({
                        "tool": call.name,
                        "arguments": arguments
                    })

                    try:
                        result = await mcp.call_tool(call.name, arguments)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result
                    })

        return {
            "error": "max_iterations",
            "message": f"Reached maximum iterations ({self.MAX_ITERATIONS})",
            "response": None,
            "modifications": None,
            "tool_calls_made": tool_calls_made
        }

    def _extract_final_response(
        self,
        output_text: str,
        tool_calls_made: list
    ) -> dict[str, Any]:
        """Extract structured response from GPT-5.2 output."""
        result = {
            "response": output_text,
            "modifications": None,
            "citations": [],
            "tool_calls_made": tool_calls_made
        }

        # Try to extract JSON modifications block
        try:
            # Look for JSON code block
            if "```json" in output_text:
                json_start = output_text.find("```json") + 7
                json_end = output_text.find("```", json_start)
                json_str = output_text[json_start:json_end].strip()
                parsed = json.loads(json_str)

                if "modifications" in parsed:
                    result["modifications"] = parsed["modifications"]
                if "citations" in parsed:
                    result["citations"] = parsed["citations"]
                if "rationale" in parsed:
                    result["rationale"] = parsed["rationale"]
        except (json.JSONDecodeError, ValueError):
            # No valid JSON found, keep raw response
            pass

        return result
```

### Task 8.4: Integration with blender-librarian MCP Server

**File:** `agents/blender-librarian/server.py` (modify existing)

Add new tool that uses the tool calling agent:

```python
@mcp.tool()
async def search_docs_with_reasoning(
    query: str,
    effect_type: str = "general",
    current_params: str = "{}",
    issues: str = "[]"
) -> str:
    """
    Search Blender documentation with GPT-5.2 reasoning.

    Uses tool calling to let GPT-5.2 intelligently search documentation
    and synthesize answers with citations.

    Args:
        query: Question or issue description
        effect_type: Effect type for context (e.g., "sun", "explosion")
        current_params: JSON string of current Blender parameters
        issues: JSON string array of current quality issues

    Returns:
        JSON with response, modifications, and citations
    """
    from tools.tool_calling_agent import ToolCallingAgent

    agent = ToolCallingAgent(budget_tracker=budget_tracker)

    context = {
        "effect_type": effect_type,
        "current_params": json.loads(current_params),
        "issues": json.loads(issues)
    }

    result = await agent.run(query, context)
    return json.dumps(result, indent=2)
```

### Task 8.5: Update SKILL.md for Tool Calling

**File:** `.claude/skills/blender-orchestrator/SKILL.md` (modify)

Add documentation for the new tool calling capability:

```markdown
### Stage 5.5: Intelligent Documentation Lookup (NEW in v4)

When stuck after 2+ iterations without improvement:

1. **Call search_docs_with_reasoning:**
   ```
   mcp__blender-librarian__search_docs_with_reasoning(
       query="How to fix blackbody emission clipping in volumetric rendering?",
       effect_type="sun",
       current_params='{"flame_max_temp": 3000, "emission_intensity": 1.0}',
       issues='["TOO DARK", "NO LIMB DARKENING"]'
   )
   ```

2. **Apply returned modifications:**
   - Parse `modifications` from response
   - Validate against script-generator.get_parameter_ranges()
   - Call script-generator.modify_script() with validated params

3. **Record successful fixes to playbook:**
   - If quality improves significantly (>5 points)
   - Call mcp__blender-librarian__add_to_playbook()
```

---

## Phase 9: OpenAI Agents SDK Integration (Priority 2)

### Objective

Implement cleaner multi-agent orchestration using OpenAI's Agents SDK for more sophisticated workflows.

### Why Agents SDK Second?

| Benefit | Description |
|---------|-------------|
| **Cleaner Handoffs** | Built-in agent-to-agent handoff primitives |
| **Automatic Tracing** | Built-in conversation tracing for debugging |
| **MCP Native Support** | Direct `mcp_servers=[]` parameter for agent definitions |
| **Session Management** | Automatic conversation state management |
| **Guardrails** | Built-in input/output validation framework |

### Task 9.1: Install and Configure Agents SDK

**File:** `agents/blender-librarian/requirements.txt` (modify)

```txt
# Existing
mcp>=1.0.0
httpx>=0.27.0
tenacity>=8.2.0
pillow>=10.0.0
openai>=1.50.0

# New for Phase 9
openai-agents>=0.1.0
```

### Task 9.2: Define Specialized Agents

**File:** `agents/blender-librarian/agents/doc_expert.py`

```python
"""
Documentation Expert Agent using OpenAI Agents SDK.

Specialized agent for searching and synthesizing Blender documentation.
"""

from agents import Agent, function_tool
from agents.mcp import MCPServerStreamableHttp


@function_tool
async def validate_parameter_range(
    parameter: str,
    value: float
) -> str:
    """Validate a parameter value against Blender 5.0 API ranges.

    Args:
        parameter: Parameter name (e.g., "turbulence", "flame_smoke")
        value: Proposed value

    Returns:
        JSON with valid (bool), clamped_value, and warning if out of range
    """
    # Import from script-generator
    from script_generator.validator import BLENDER_PARAMETER_RANGES

    if parameter not in BLENDER_PARAMETER_RANGES:
        return json.dumps({
            "valid": True,
            "warning": f"Unknown parameter '{parameter}', cannot validate"
        })

    range_info = BLENDER_PARAMETER_RANGES[parameter]
    min_val, max_val = range_info["min"], range_info["max"]

    if min_val <= value <= max_val:
        return json.dumps({"valid": True, "value": value})

    clamped = max(min_val, min(max_val, value))
    return json.dumps({
        "valid": False,
        "original_value": value,
        "clamped_value": clamped,
        "range": [min_val, max_val],
        "warning": f"Value {value} out of range [{min_val}, {max_val}], clamped to {clamped}"
    })


async def create_doc_expert(mcp_server_url: str = "http://localhost:8001") -> Agent:
    """Create the documentation expert agent with MCP server connection.

    Args:
        mcp_server_url: URL of blender-manual MCP server

    Returns:
        Configured Agent instance
    """
    # Connect to blender-manual MCP server
    mcp_server = MCPServerStreamableHttp(
        name="blender-manual",
        params={
            "url": f"{mcp_server_url}/mcp",
            "timeout": 10
        },
        cache_tools_list=True,
        max_retry_attempts=2
    )

    return Agent(
        name="Documentation Expert",
        instructions="""You are a Blender 5.0 documentation expert.

Your role:
1. Search the official Blender documentation for relevant information
2. Read specific pages when more detail is needed
3. Provide accurate, cited answers
4. Validate parameter values against API ranges

When answering VFX questions:
- Use search_vdb_workflow for volumetric/VDB topics
- Use search_python_api for scripting questions
- Use search_nodes for shader/material questions
- Always cite sources with documentation paths

Output format for parameter recommendations:
{
    "modifications": {"param": value, ...},
    "rationale": "...",
    "citations": ["path/to/doc.html"]
}""",
        model="gpt-5.2",
        mcp_servers=[mcp_server],
        tools=[validate_parameter_range]
    )
```

### Task 9.3: Vision Diagnosis Agent

**File:** `agents/blender-librarian/agents/vision_expert.py`

```python
"""
Vision Diagnosis Agent using OpenAI Agents SDK.

Specialized agent for analyzing render quality using GPT-5.2 vision.
"""

import base64
from pathlib import Path

from agents import Agent, function_tool


@function_tool
def load_image_as_base64(image_path: str, max_size: int = 512) -> str:
    """Load and resize an image for GPT-5.2 vision analysis.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height)

    Returns:
        Base64-encoded image string with data URI prefix
    """
    from PIL import Image
    import io

    with Image.open(image_path) as img:
        # Resize if needed
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(d * ratio) for d in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        b64_data = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{b64_data}"


def create_vision_expert() -> Agent:
    """Create the vision diagnosis expert agent.

    Returns:
        Configured Agent instance
    """
    return Agent(
        name="Vision Expert",
        instructions="""You are a VFX render quality analyst powered by GPT-5.2 vision.

Your role:
1. Analyze render screenshots for visual quality issues
2. Compare renders against reference images
3. Identify specific problems (color, lighting, structure, artifacts)
4. Provide actionable diagnosis

When analyzing renders, look for:
- Color temperature issues (too warm/cool)
- Brightness problems (too dark/bright, clipping)
- Structural issues (procedural artifacts, lack of detail)
- Missing features (limb darkening, prominences, granulation for solar)

Output format:
{
    "diagnosis": "Primary issue description",
    "primary_issue": "category_name",
    "severity": "critical|high|medium|low",
    "secondary_issues": ["issue1", "issue2"],
    "recommendations": ["action1", "action2"]
}""",
        model="gpt-5.2",
        tools=[load_image_as_base64]
    )
```

### Task 9.4: Orchestrator Agent with Handoffs

**File:** `agents/blender-librarian/agents/librarian_orchestrator.py`

```python
"""
Librarian Orchestrator using OpenAI Agents SDK.

Central orchestrator that routes to specialized agents via handoffs.
"""

from agents import Agent, handoff, Runner, RunContextWrapper

from .doc_expert import create_doc_expert
from .vision_expert import create_vision_expert


class LibrarianOrchestrator:
    """Orchestrates documentation and vision expert agents."""

    def __init__(self, mcp_server_url: str = "http://localhost:8001"):
        self.mcp_url = mcp_server_url
        self._doc_expert: Agent | None = None
        self._vision_expert: Agent | None = None
        self._orchestrator: Agent | None = None

    async def initialize(self):
        """Initialize all agents."""
        self._doc_expert = await create_doc_expert(self.mcp_url)
        self._vision_expert = create_vision_expert()

        self._orchestrator = Agent(
            name="Blender Librarian",
            instructions="""You are the Blender Librarian orchestrator.

Route requests to the appropriate specialist:
- For documentation questions → hand off to doc_expert
- For visual analysis of renders → hand off to vision_expert
- For combined issues (need both) → start with vision, then doc

Always start by understanding the user's need, then delegate.""",
            model="gpt-5.2",
            handoffs=[
                handoff(
                    agent=self._doc_expert,
                    tool_name_override="consult_documentation_expert",
                    tool_description_override="Hand off to documentation expert for Blender manual searches"
                ),
                handoff(
                    agent=self._vision_expert,
                    tool_name_override="consult_vision_expert",
                    tool_description_override="Hand off to vision expert for render quality analysis"
                )
            ]
        )

    async def run(self, query: str, context: dict | None = None) -> dict:
        """Run the orchestrator with a query.

        Args:
            query: User query or issue description
            context: Optional context dict

        Returns:
            Result dict with response and any extracted data
        """
        if not self._orchestrator:
            await self.initialize()

        # Build input with context
        if context:
            full_query = f"Context: {json.dumps(context)}\n\nQuery: {query}"
        else:
            full_query = query

        result = await Runner.run(self._orchestrator, full_query)

        return {
            "response": result.final_output,
            "conversation": result.to_input_list()
        }
```

### Task 9.5: Integration with Existing MCP Server

**File:** `agents/blender-librarian/server.py` (modify)

Add SDK-based tool alongside tool calling approach:

```python
@mcp.tool()
async def search_docs_with_agents_sdk(
    query: str,
    effect_type: str = "general",
    current_params: str = "{}",
    issues: str = "[]",
    include_vision: bool = False,
    render_path: str = "",
    reference_path: str = ""
) -> str:
    """
    Search Blender documentation using OpenAI Agents SDK orchestration.

    Uses multi-agent handoffs for sophisticated documentation lookup
    and optional vision analysis.

    Args:
        query: Question or issue description
        effect_type: Effect type for context
        current_params: JSON string of current parameters
        issues: JSON string array of quality issues
        include_vision: Whether to include vision analysis
        render_path: Path to current render (if include_vision)
        reference_path: Path to reference image (if include_vision)

    Returns:
        JSON with response, modifications, and agent trace
    """
    from agents.librarian_orchestrator import LibrarianOrchestrator

    orchestrator = LibrarianOrchestrator()

    context = {
        "effect_type": effect_type,
        "current_params": json.loads(current_params),
        "issues": json.loads(issues)
    }

    if include_vision and render_path:
        context["render_path"] = render_path
        context["reference_path"] = reference_path

    result = await orchestrator.run(query, context)
    return json.dumps(result, indent=2)
```

### Task 9.6: Cross-Session Learning with Sessions Primitive

**File:** `agents/blender-librarian/agents/session_manager.py`

The OpenAI Agents SDK provides a Sessions primitive for automatic conversation state management. This enables cross-session learning where knowledge accumulates across runs.

```python
"""
Session Manager for Cross-Session Learning

Uses OpenAI Agents SDK Sessions to:
1. Persist conversation history across agent runs
2. Accumulate learned parameter patterns
3. Remember successful fixes for similar issues
4. Provide context from previous sessions
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from agents import Agent, RunResult
from openai.types.beta.threads import Run


@dataclass
class SessionContext:
    """Persistent context that survives across sessions."""
    session_id: str
    created_at: str
    last_active: str
    effect_type: str = ""
    successful_fixes: List[Dict[str, Any]] = field(default_factory=list)
    failed_experiments: List[Dict[str, Any]] = field(default_factory=list)
    parameter_history: Dict[str, List[float]] = field(default_factory=dict)
    quality_trajectory: List[float] = field(default_factory=list)


class SessionManager:
    """
    Manages persistent sessions for cross-session learning.

    Key capabilities:
    - Automatic session restoration on agent start
    - Persistent storage of successful parameter patterns
    - Cross-session knowledge accumulation
    - Session context injection into agent prompts
    """

    SESSIONS_DIR = Path("agents/blender-librarian/sessions")

    def __init__(self, effect_type: str = ""):
        """Initialize session manager."""
        self.effect_type = effect_type
        self.sessions_dir = self.SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionContext] = None

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID to restore. If None, creates new session.

        Returns:
            SessionContext with accumulated learning
        """
        if session_id:
            session_path = self.sessions_dir / f"{session_id}.json"
            if session_path.exists():
                with open(session_path, 'r') as f:
                    data = json.load(f)
                self.current_session = SessionContext(**data)
                self.current_session.last_active = datetime.now().isoformat()
                return self.current_session

        # Create new session
        new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = SessionContext(
            session_id=new_id,
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            effect_type=self.effect_type
        )
        return self.current_session

    def record_successful_fix(
        self,
        issue: str,
        modifications: Dict[str, Any],
        score_improvement: float,
        doc_sources: List[str]
    ):
        """
        Record a successful fix for cross-session learning.

        Args:
            issue: The issue that was fixed
            modifications: Parameter changes that worked
            score_improvement: How much the quality score improved
            doc_sources: Documentation sources used
        """
        if not self.current_session:
            return

        self.current_session.successful_fixes.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "modifications": modifications,
            "score_improvement": score_improvement,
            "doc_sources": doc_sources
        })
        self._save_session()

    def record_failed_experiment(
        self,
        issue: str,
        modifications: Dict[str, Any],
        reason: str
    ):
        """
        Record a failed experiment to avoid repeating.

        Args:
            issue: The issue we tried to fix
            modifications: Parameter changes that didn't work
            reason: Why it failed
        """
        if not self.current_session:
            return

        self.current_session.failed_experiments.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "modifications": modifications,
            "reason": reason
        })
        self._save_session()

    def get_relevant_history(self, current_issue: str) -> Dict[str, Any]:
        """
        Get relevant history for the current issue.

        Args:
            current_issue: Description of current issue

        Returns:
            Dict with relevant past fixes and failures
        """
        if not self.current_session:
            return {"fixes": [], "failures": []}

        # Simple keyword matching (could be enhanced with embeddings)
        keywords = current_issue.lower().split()

        relevant_fixes = []
        for fix in self.current_session.successful_fixes:
            if any(kw in fix["issue"].lower() for kw in keywords):
                relevant_fixes.append(fix)

        relevant_failures = []
        for fail in self.current_session.failed_experiments:
            if any(kw in fail["issue"].lower() for kw in keywords):
                relevant_failures.append(fail)

        return {
            "fixes": relevant_fixes[-5:],  # Last 5 relevant
            "failures": relevant_failures[-3:]  # Last 3 relevant
        }

    def inject_session_context(self, base_instructions: str) -> str:
        """
        Inject session context into agent instructions.

        Args:
            base_instructions: Original agent instructions

        Returns:
            Instructions with session context appended
        """
        if not self.current_session or not self.current_session.successful_fixes:
            return base_instructions

        context_section = "\n\n## CROSS-SESSION LEARNING CONTEXT\n\n"
        context_section += "You have accumulated knowledge from previous sessions:\n\n"

        # Add successful patterns
        if self.current_session.successful_fixes:
            context_section += "### Successful Fixes (PROVEN TO WORK):\n"
            for fix in self.current_session.successful_fixes[-5:]:
                context_section += f"- Issue: {fix['issue']}\n"
                context_section += f"  Fix: {json.dumps(fix['modifications'])}\n"
                context_section += f"  Improvement: +{fix['score_improvement']:.1f} points\n\n"

        # Add failures to avoid
        if self.current_session.failed_experiments:
            context_section += "### Failed Experiments (AVOID THESE):\n"
            for fail in self.current_session.failed_experiments[-3:]:
                context_section += f"- Issue: {fail['issue']}\n"
                context_section += f"  Attempted: {json.dumps(fail['modifications'])}\n"
                context_section += f"  Reason: {fail['reason']}\n\n"

        return base_instructions + context_section

    def _save_session(self):
        """Save current session to disk."""
        if not self.current_session:
            return

        session_path = self.sessions_dir / f"{self.current_session.session_id}.json"
        with open(session_path, 'w') as f:
            json.dump({
                "session_id": self.current_session.session_id,
                "created_at": self.current_session.created_at,
                "last_active": self.current_session.last_active,
                "effect_type": self.current_session.effect_type,
                "successful_fixes": self.current_session.successful_fixes,
                "failed_experiments": self.current_session.failed_experiments,
                "parameter_history": self.current_session.parameter_history,
                "quality_trajectory": self.current_session.quality_trajectory
            }, f, indent=2)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        for session_file in self.sessions_dir.glob("*.json"):
            with open(session_file, 'r') as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "effect_type": data.get("effect_type", ""),
                    "num_fixes": len(data.get("successful_fixes", [])),
                    "num_failures": len(data.get("failed_experiments", []))
                })
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
```

**Integration with Orchestrator:**

Update `LibrarianOrchestrator` to use session management:

```python
# In librarian_orchestrator.py

from .session_manager import SessionManager

class LibrarianOrchestrator:
    def __init__(self, session_id: Optional[str] = None):
        self.session_manager = SessionManager()
        self.session = self.session_manager.get_or_create_session(session_id)

        # Inject session context into agent instructions
        self.doc_expert = Agent(
            name="doc_expert",
            instructions=self.session_manager.inject_session_context(DOC_EXPERT_INSTRUCTIONS),
            # ... rest of config
        )

    async def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Get relevant history for this issue
        history = self.session_manager.get_relevant_history(query)

        # Add history to context
        context["session_history"] = history

        # Run the agent
        result = await Runner.run(self.orchestrator, query)

        # After successful run, record the outcome
        if result.success and result.modifications:
            self.session_manager.record_successful_fix(
                issue=query,
                modifications=result.modifications,
                score_improvement=context.get("score_improvement", 0),
                doc_sources=result.doc_paths_used
            )

        return result
```

**Benefits:**
- **Accumulated Wisdom**: Each session builds on previous successes
- **Avoid Repeating Failures**: Don't try things that already failed
- **Context Injection**: Agents get relevant history automatically
- **Persistent Storage**: Knowledge survives across Claude Code windows
- **Session Resumption**: Can continue from previous session by ID

---

## Data Contracts (Critical)

### Modification Output Schema

Must be compatible with `script-generator.modify_script()`:

```json
{
    "modifications": {
        "parameter_name": "value",
        "turbulence": 0.8,
        "flame_smoke": 3.5,
        "domain_scale": [2.0, 3.0, 2.0]
    },
    "rationale": "String explaining why these changes should help",
    "citations": ["physics/fluid/type/domain/settings.html"]
}
```

### Issue Input Schema

From asset-evaluator tools:

```json
[
    {
        "category": "color_too_cool",
        "severity": "critical",
        "description": "Render appears blue/cyan when should be warm orange"
    },
    {
        "category": "no_limb_darkening",
        "severity": "high",
        "description": "Solar disk lacks realistic edge darkening"
    }
]
```

### Budget Tracker Schema

**File:** `agents/blender-librarian/budget_tracker.json`

```json
{
    "month": "2026-01",
    "vision_budget": 6.00,
    "vision_spent": 0.00,
    "doc_budget": 14.00,
    "doc_spent": 0.00,
    "last_reset": "2026-01-01T00:00:00Z",
    "call_history": [
        {
            "timestamp": "2026-01-05T10:30:00Z",
            "tool": "diagnose_render_issue",
            "cost": 0.02,
            "type": "vision"
        }
    ]
}
```

---

## Error Handling Matrix

| Error | Tool Calling Response | Agents SDK Response | Recovery |
|-------|----------------------|---------------------|----------|
| MCP timeout | `{"error": "mcp_timeout", "fallback": "playbook_only"}` | Same | Check playbook, return cached fix if available |
| Budget exhausted (vision) | `{"error": "vision_budget_exhausted"}` | Same | Switch to doc-only mode |
| Budget exhausted (doc) | `{"error": "doc_budget_exhausted"}` | Same | Return playbook-only fixes |
| No docs found | Continue with general knowledge | Same | GPT-5.2 uses training knowledge, marks as "uncited" |
| OpenAI rate limit | Retry with exponential backoff (3 attempts) | Same | Return error after retries exhausted |
| Invalid parameter range | Clamp value, return warning | Same | Include `clamped_value` in response |

---

## Testing Plan

### Unit Tests

| Test | Phase | Description |
|------|-------|-------------|
| `test_tool_definitions.py` | 8.1 | Validate all 12 tool schemas are correct |
| `test_mcp_bridge.py` | 8.2 | Mock MCP server, test call_tool, error handling |
| `test_tool_calling_agent.py` | 8.3 | Mock OpenAI, verify agent loop terminates |
| `test_agents_sdk_integration.py` | 9.4 | Test handoffs between agents |
| `test_budget_tracker.py` | Both | Budget tracking, reset, cost calculation |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_e2e_tool_calling.py` | Full flow: query → search → response with citations |
| `test_e2e_agents_sdk.py` | Full flow with agent handoffs |
| `test_orchestrator_integration.py` | Integration with blender-orchestrator SKILL |
| `test_playbook_learning.py` | Successful fix → add_to_playbook → future lookup |

### Manual Validation

1. **Documentation accuracy:** Ask 10 questions, verify citations are correct
2. **Modification compatibility:** Apply returned modifications to script-generator
3. **Budget tracking:** Run 5 queries, verify cost tracking matches OpenAI dashboard
4. **Error recovery:** Kill MCP server mid-query, verify graceful failure

---

## Files Summary

### New Files (Phase 8)

| File | Purpose |
|------|---------|
| `agents/blender-librarian/tools/__init__.py` | Package init |
| `agents/blender-librarian/tools/blender_manual_tools.py` | Tool definitions |
| `agents/blender-librarian/tools/mcp_bridge.py` | MCP client |
| `agents/blender-librarian/tools/tool_calling_agent.py` | Agent loop |

### New Files (Phase 9)

| File | Purpose |
|------|---------|
| `agents/blender-librarian/agents/__init__.py` | Package init |
| `agents/blender-librarian/agents/doc_expert.py` | Doc search agent |
| `agents/blender-librarian/agents/vision_expert.py` | Vision analysis agent |
| `agents/blender-librarian/agents/librarian_orchestrator.py` | Multi-agent orchestrator |

### Modified Files

| File | Changes |
|------|---------|
| `agents/blender-librarian/server.py` | Add new MCP tools |
| `agents/blender-librarian/requirements.txt` | Add openai-agents |
| `.claude/skills/blender-orchestrator/SKILL.md` | Document new tools |
| `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` | Reference v4 |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Doc search accuracy | N/A | >90% relevant results |
| Modification compatibility | N/A | 100% parseable by script-generator |
| Response time (tool calling) | N/A | <8s average |
| Response time (agents SDK) | N/A | <12s average |
| Budget tracking accuracy | N/A | ±$0.01 vs OpenAI dashboard |
| Playbook hit rate | 0% | >30% after 1 month |

---

## Implementation Order

### Sprint 1: Tool Calling (Priority 1)

1. [ ] Task 8.1: Tool definitions
2. [ ] Task 8.2: MCP bridge client
3. [ ] Task 8.3: Tool calling agent loop
4. [ ] Task 8.4: Integration with server.py
5. [ ] Task 8.5: Update SKILL.md
6. [ ] Unit tests for Phase 8
7. [ ] Manual validation

### Sprint 2: Agents SDK (Priority 2)

1. [ ] Task 9.1: Install SDK
2. [ ] Task 9.2: Doc expert agent
3. [ ] Task 9.3: Vision expert agent
4. [ ] Task 9.4: Orchestrator with handoffs
5. [ ] Task 9.5: Integration with server.py
6. [ ] Task 9.6: Cross-session learning (persistence)
7. [ ] Unit tests for Phase 9
8. [ ] Comparison testing (tool calling vs SDK)

---

## Related Documents

- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Previous version
- `docs/GPT52_BLENDER_LIBRARIAN_MERGED_DESIGN.md` - Phase 7 design
- `plans/feat-blender-librarian-FINAL.md` - Phase 7 implementation
- `docs/openai_api_documentation/` - OpenAI API reference
- `.claude/skills/blender-orchestrator/SKILL.md` - Orchestrator skill

---

**Last Updated:** 2026-01-05
**Plan Version:** 4.0
**Status:** Ready for implementation
