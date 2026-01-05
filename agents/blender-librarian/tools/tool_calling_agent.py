"""
GPT-5.2 Tool Calling Agent for Blender Documentation

This agent uses the OpenAI Responses API with function calling to:
1. Receive a user query about Blender (usually to fix a rendering issue)
2. Let GPT-5.2 decide which blender-manual tools to call
3. Execute those tools via MCP bridge
4. Return results to GPT-5.2 for synthesis
5. Repeat until GPT-5.2 has enough information to answer

Key features:
- Agentic loop: GPT-5.2 autonomously decides which tools to call
- Budget-aware: Tracks token usage to stay within limits
- Graceful degradation: Falls back if tools fail
- Structured output: Returns actionable parameter modifications
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .blender_manual_tools import BLENDER_MANUAL_TOOLS
from .mcp_bridge_client import MCPBridgeClient


@dataclass
class AgentResponse:
    """Response from the tool calling agent."""
    success: bool
    answer: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    doc_paths_used: List[str] = field(default_factory=list)
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    iterations: int = 0
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for the tool calling agent."""
    model: str = "gpt-5.2"
    max_iterations: int = 5
    max_output_tokens: int = 2000
    reasoning_effort: str = "medium"
    verbosity: str = "low"


# GPT-5.2 Pricing (as of Jan 2026)
COST_INPUT_1K = 0.00175   # $1.75 per 1M input tokens
COST_OUTPUT_1K = 0.014    # $14 per 1M output tokens


class ToolCallingAgent:
    """
    Agent that uses GPT-5.2 with function calling to search Blender docs.

    The agent:
    1. Takes a query about Blender/VFX issues
    2. Uses GPT-5.2 to decide which doc search tools to call
    3. Executes those tools via MCP bridge
    4. Feeds results back to GPT-5.2
    5. Continues until GPT-5.2 synthesizes an answer
    """

    SYSTEM_INSTRUCTIONS = """You are a Blender VFX documentation expert.

Your task is to search the Blender 5.0 manual and Python API documentation to find
information that will help fix VFX rendering issues.

You have access to 12 documentation search tools. Use them strategically:

SEARCH STRATEGY:
1. Start with search_semantic for natural language queries
2. Use search_vdb_workflow for volume/VDB-specific issues
3. Use search_python_api for scripting/API questions
4. Use read_page to get full content of promising results
5. Use search_bpy_types/search_bpy_operators for specific API references

WHEN TO STOP:
- You have found documentation that directly addresses the issue
- You have enough information to suggest specific parameter changes
- You've searched 3+ different paths without finding relevant info

OUTPUT FORMAT:
When you have enough information, respond with ONLY valid JSON:
{
    "answer": "1-2 sentence summary of what you found",
    "modifications": {
        // Specific Blender parameter changes (use ONLY these keys):
        // "resolution": int,
        // "frame_end": int,
        // "turbulence": float 0-1,
        // "vorticity": float 0-1,
        // "temperature": float (Kelvin),
        // "domain_scale": float,
        // "custom_code": {"search_string": "replacement_string"}
    },
    "rationale": "Why these changes should help based on docs",
    "confidence": 0.0-1.0,
    "doc_paths_used": ["paths", "from", "search", "results"]
}

CONSTRAINTS:
- Maximum 2-3 parameter changes
- Be conservative - small changes are safer
- If unsure, suggest a safe diagnostic change first"""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent."""
        self.config = config or AgentConfig()
        self._mcp_client: Optional[MCPBridgeClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._mcp_client = MCPBridgeClient()
        await self._mcp_client.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._mcp_client:
            await self._mcp_client.close()

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Run the agent to answer a Blender documentation query.

        Args:
            query: The question to answer (e.g., "How to fix limb darkening?")
            context: Optional context (effect_type, current_params, issues, etc.)

        Returns:
            AgentResponse with answer, modifications, and metadata
        """
        import requests

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return AgentResponse(
                success=False,
                answer="",
                error="OPENAI_API_KEY not set"
            )

        if not self._mcp_client:
            self._mcp_client = MCPBridgeClient()
            await self._mcp_client.initialize()

        # Build initial input
        input_list = []

        # Add context to the user message
        user_content = f"QUERY: {query}\n"
        if context:
            user_content += f"\nCONTEXT:\n"
            if context.get("effect_type"):
                user_content += f"Effect type: {context['effect_type']}\n"
            if context.get("current_issues"):
                user_content += f"Current issues: {json.dumps(context['current_issues'])}\n"
            if context.get("current_params"):
                user_content += f"Current parameters: {json.dumps(context['current_params'])}\n"
            if context.get("evaluator_scores"):
                user_content += f"Evaluation scores: {json.dumps(context['evaluator_scores'])}\n"

        input_list.append({"role": "user", "content": user_content})

        # Track state
        tool_calls_made = []
        total_cost = 0.0
        iterations = 0

        # Agentic loop
        for iteration in range(self.config.max_iterations):
            iterations += 1

            # Call GPT-5.2
            payload = {
                "model": self.config.model,
                "instructions": self.SYSTEM_INSTRUCTIONS,
                "input": input_list,
                "tools": BLENDER_MANUAL_TOOLS,
                "reasoning": {"effort": self.config.reasoning_effort},
                "text": {"verbosity": self.config.verbosity},
                "store": False,
                "max_output_tokens": self.config.max_output_tokens
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
            except requests.exceptions.RequestException as e:
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = f"{str(e)} - Response: {e.response.text[:500]}"
                    except Exception:
                        pass
                return AgentResponse(
                    success=False,
                    answer="",
                    tool_calls_made=tool_calls_made,
                    total_cost=total_cost,
                    iterations=iterations,
                    error=f"API request failed: {error_detail}"
                )

            # Calculate cost
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 500)
            output_tokens = usage.get("output_tokens", 200)
            iteration_cost = (input_tokens / 1000 * COST_INPUT_1K) + (output_tokens / 1000 * COST_OUTPUT_1K)
            total_cost += iteration_cost

            # Process response
            output = data.get("output", [])

            # Check for function calls
            function_calls = []
            text_output = None

            for item in output:
                item_type = item.get("type")

                if item_type == "function_call":
                    function_calls.append({
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments", "{}")
                    })
                elif item_type == "message":
                    for content in item.get("content", []):
                        if content.get("type") in ("output_text", "text"):
                            text_output = content.get("text")

            # If no function calls, we're done - parse the final answer
            if not function_calls:
                if text_output:
                    try:
                        result = json.loads(text_output)
                        return AgentResponse(
                            success=True,
                            answer=result.get("answer", text_output),
                            modifications=result.get("modifications", {}),
                            doc_paths_used=result.get("doc_paths_used", []),
                            tool_calls_made=tool_calls_made,
                            total_cost=total_cost,
                            iterations=iterations
                        )
                    except json.JSONDecodeError:
                        return AgentResponse(
                            success=True,
                            answer=text_output,
                            tool_calls_made=tool_calls_made,
                            total_cost=total_cost,
                            iterations=iterations
                        )
                else:
                    return AgentResponse(
                        success=False,
                        answer="",
                        tool_calls_made=tool_calls_made,
                        total_cost=total_cost,
                        iterations=iterations,
                        error="No text output from model"
                    )

            # First, add all function_call items from output to input_list
            # (Required by Responses API - must include the function_call before function_call_output)
            for item in output:
                if item.get("type") == "function_call":
                    input_list.append(item)

            # Execute function calls and add outputs
            for fc in function_calls:
                tool_name = fc["name"]
                try:
                    args = json.loads(fc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                # Record the call
                tool_calls_made.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": args
                })

                # Execute via MCP bridge
                result = await self._mcp_client.call_tool(tool_name, args)

                # Format result for GPT
                if result.success:
                    result_text = json.dumps(result.data) if isinstance(result.data, (dict, list)) else str(result.data)
                else:
                    result_text = f"Error: {result.error}"

                # Add function result to conversation
                input_list.append({
                    "type": "function_call_output",
                    "call_id": fc["id"],
                    "output": result_text[:8000]  # Truncate long results
                })

        # Max iterations reached
        return AgentResponse(
            success=False,
            answer="Max iterations reached without final answer",
            tool_calls_made=tool_calls_made,
            total_cost=total_cost,
            iterations=iterations,
            error="Agent did not converge within max iterations"
        )


# Convenience function for single queries
async def search_docs_with_agent(
    query: str,
    effect_type: str = "",
    current_issues: Optional[List[str]] = None,
    current_params: Optional[Dict[str, Any]] = None
) -> AgentResponse:
    """
    Run a single documentation search with the agent.

    Args:
        query: The question to answer
        effect_type: Type of effect (sun, explosion, etc.)
        current_issues: List of current issues
        current_params: Current Blender parameters

    Returns:
        AgentResponse with results
    """
    context = {}
    if effect_type:
        context["effect_type"] = effect_type
    if current_issues:
        context["current_issues"] = current_issues
    if current_params:
        context["current_params"] = current_params

    async with ToolCallingAgent() as agent:
        return await agent.run(query, context if context else None)


# Test the agent
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing ToolCallingAgent...")
        print("-" * 60)

        # Test a simple query
        result = await search_docs_with_agent(
            query="How do I create limb darkening effect in a sun simulation?",
            effect_type="sun",
            current_issues=["no limb darkening", "flat edges"]
        )

        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"Cost: ${result.total_cost:.4f}")
        print(f"Tool calls: {len(result.tool_calls_made)}")

        if result.success:
            print(f"\nAnswer: {result.answer}")
            print(f"Modifications: {json.dumps(result.modifications, indent=2)}")
            print(f"Doc paths: {result.doc_paths_used}")
        else:
            print(f"Error: {result.error}")

        print("\nTool call history:")
        for call in result.tool_calls_made:
            print(f"  [{call['iteration']}] {call['tool']}({json.dumps(call['arguments'])})")

    asyncio.run(test())
