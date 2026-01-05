"""
Librarian Orchestrator using OpenAI Agents SDK.

Central orchestrator that routes to specialized agents via handoffs.
Manages the flow between documentation search and vision analysis,
coordinating multi-step problem solving.

Key capabilities:
- Agent handoffs for specialized tasks (doc search, vision analysis)
- Context preservation across handoffs
- Session-aware queries with accumulated knowledge
- Structured output compatible with script-generator
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from agents import Agent, ModelSettings, handoff, Runner
from openai.types.shared import Reasoning

from .doc_expert import DocExpertAgent, create_doc_expert
from .vision_expert import VisionExpertAgent, create_vision_expert


# Orchestrator instructions for intelligent routing
ORCHESTRATOR_INSTRUCTIONS = """You are the Blender Librarian orchestrator for the PlasmaDX VFX pipeline.

Your role is to understand user requests and delegate to the appropriate specialist:

SPECIALISTS AVAILABLE:
1. **Documentation Expert** (consult_documentation_expert)
   - Search Blender 5.0 documentation
   - Validate parameter ranges
   - Find API references and tutorials
   - Use for: "how do I...", "what is the parameter for...", "documentation for..."

2. **Vision Expert** (consult_vision_expert)
   - Analyze render screenshots
   - Compare renders to references
   - Diagnose visual quality issues
   - Use for: "why does my render look...", "compare this to...", "what's wrong with..."

ROUTING LOGIC:
- Documentation questions → hand off to doc_expert
- Visual analysis requests → hand off to vision_expert
- Combined issues (need both) → start with vision diagnosis, then doc search for fixes
- Parameter optimization → doc_expert first for valid ranges, then apply

IMPORTANT:
- Always understand the user's need before delegating
- Preserve context between handoffs
- Synthesize results from multiple specialists when needed
- Return structured JSON compatible with script-generator:

{
    "answer": "Summary of findings",
    "modifications": {"param": value, ...},
    "rationale": "Why these changes help",
    "confidence": 0.0-1.0,
    "citations": ["doc/path1.html", ...],
    "diagnosis": {...}  // If vision analysis was performed
}"""


class LibrarianOrchestrator:
    """
    Orchestrates documentation and vision expert agents.

    Provides intelligent routing and synthesis of results from
    multiple specialized agents.
    """

    def __init__(
        self,
        blender_manual_path: Optional[str] = None,
        model: str = "gpt-5.2"
    ):
        """
        Initialize the orchestrator.

        Args:
            blender_manual_path: Path to blender-manual server (auto-detected if None)
            model: OpenAI model for orchestration
        """
        self.blender_manual_path = blender_manual_path
        self.model = os.getenv("OPENAI_MODEL", model)
        self._doc_expert: Optional[Agent] = None
        self._vision_expert: Optional[Agent] = None
        self._orchestrator: Optional[Agent] = None
        self._initialized = False

    async def initialize(self, session_context: str = "") -> None:
        """
        Initialize all agents.

        Args:
            session_context: Optional session context to include in instructions
        """
        if self._initialized:
            return

        # Create specialized agents (doc expert spawns blender-manual MCP server via stdio)
        self._doc_expert = await create_doc_expert(
            blender_manual_path=self.blender_manual_path,
            custom_instructions=session_context
        )

        self._vision_expert = create_vision_expert(
            custom_instructions=session_context
        )

        # Build orchestrator instructions with session context
        instructions = ORCHESTRATOR_INSTRUCTIONS
        if session_context:
            instructions = instructions + f"\n\nSESSION CONTEXT:\n{session_context}"

        # Create orchestrator with handoffs to specialists
        # Use medium reasoning for complex multi-agent coordination
        self._orchestrator = Agent(
            name="Blender Librarian",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
                verbosity="low"
            ),
            handoffs=[
                handoff(
                    agent=self._doc_expert,
                    tool_name_override="consult_documentation_expert",
                    tool_description_override=(
                        "Hand off to documentation expert for Blender manual searches, "
                        "parameter validation, and API reference lookups."
                    )
                ),
                handoff(
                    agent=self._vision_expert,
                    tool_name_override="consult_vision_expert",
                    tool_description_override=(
                        "Hand off to vision expert for render quality analysis, "
                        "screenshot comparison, and visual issue diagnosis."
                    )
                )
            ]
        )

        self._initialized = True

    @property
    def orchestrator(self) -> Agent:
        """Get the orchestrator agent."""
        if not self._orchestrator:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        return self._orchestrator

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the orchestrator with a query.

        Args:
            query: User query or issue description
            context: Optional context dict with:
                - effect_type: Type of effect (sun, explosion, etc.)
                - current_params: Current Blender parameters
                - issues: List of known issues
                - render_path: Path to current render
                - reference_path: Path to reference image

        Returns:
            Result dict with response, conversation history, and any extracted data
        """
        if not self._initialized:
            await self.initialize()

        # Build input with context
        if context:
            context_str = json.dumps(context, indent=2)
            full_query = f"Context:\n```json\n{context_str}\n```\n\nQuery: {query}"
        else:
            full_query = query

        # Run the orchestrator
        result = await Runner.run(self._orchestrator, full_query)

        # Parse response if it's JSON
        response_data = {"raw_response": result.final_output}
        try:
            parsed = json.loads(result.final_output)
            response_data.update(parsed)
        except (json.JSONDecodeError, TypeError):
            response_data["answer"] = result.final_output

        return {
            "response": response_data,
            "conversation": result.to_input_list(),
            "agents_used": self._extract_agents_used(result)
        }

    def _extract_agents_used(self, result) -> list:
        """Extract which agents were used during the run."""
        agents = ["orchestrator"]
        # The conversation history contains handoff information
        for item in result.to_input_list():
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    if "consult_documentation_expert" in content:
                        agents.append("doc_expert")
                    if "consult_vision_expert" in content:
                        agents.append("vision_expert")
        return list(set(agents))

    async def diagnose_and_fix(
        self,
        render_path: str,
        effect_type: str,
        issues: list,
        current_params: Dict[str, Any],
        reference_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combined diagnosis and fix recommendation flow.

        Args:
            render_path: Path to the render to analyze
            effect_type: Type of effect (sun, explosion, etc.)
            issues: List of known issues from evaluator
            current_params: Current Blender parameters
            reference_path: Optional reference image path

        Returns:
            Dict with diagnosis, recommended modifications, and rationale
        """
        context = {
            "effect_type": effect_type,
            "current_params": current_params,
            "known_issues": issues,
            "render_path": render_path,
        }
        if reference_path:
            context["reference_path"] = reference_path

        query = f"""Analyze the render at {render_path} and recommend parameter modifications.

Known issues from evaluator: {json.dumps(issues)}

Please:
1. Use vision analysis to diagnose the primary visual problem
2. Search documentation for the best parameter adjustments
3. Return modifications compatible with script-generator"""

        return await self.run(query, context)

    async def close(self) -> None:
        """Clean up resources."""
        # Agents SDK handles cleanup automatically
        self._doc_expert = None
        self._vision_expert = None
        self._orchestrator = None
        self._initialized = False


async def create_orchestrator(
    mcp_server_url: str = "http://localhost:8001",
    session_context: str = ""
) -> LibrarianOrchestrator:
    """
    Factory function to create and initialize an orchestrator.

    Args:
        mcp_server_url: URL of the blender-manual MCP server
        session_context: Optional session context

    Returns:
        Initialized LibrarianOrchestrator instance
    """
    orchestrator = LibrarianOrchestrator(mcp_server_url=mcp_server_url)
    await orchestrator.initialize(session_context=session_context)
    return orchestrator


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing LibrarianOrchestrator...")
        print("-" * 60)

        orchestrator = await create_orchestrator()

        # Test documentation query
        result = await orchestrator.run(
            "How do I increase the turbulence in a Mantaflow simulation?"
        )

        print(f"Response: {json.dumps(result['response'], indent=2)}")
        print(f"Agents used: {result['agents_used']}")

    asyncio.run(test())
