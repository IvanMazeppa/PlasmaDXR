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
- **STREAMING MODE**: Uses run_streamed() to keep connections alive during long operations

Architecture for timeout resilience:
- Runner.run_streamed() yields events incrementally
- Keeps MCP connection alive during multi-agent workflows
- Final result assembled from accumulated stream events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from agents import Agent, ModelSettings, handoff, Runner
from agents.stream_events import StreamEvent
from openai.types.shared import Reasoning
from pydantic import BaseModel

from .models import ModificationAdvice, RenderDiagnosis, AgentSearchResult

# Type variable for generic retry function
T = TypeVar('T')

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0  # seconds
DEFAULT_BACKOFF_MAX = 30.0  # max wait between retries


async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    retryable_exceptions: tuple = (Exception,),
    operation_name: str = "operation"
) -> T:
    """
    Retry an async operation with exponential backoff.

    Args:
        func: Async callable to retry
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        backoff_max: Maximum backoff wait time (seconds)
        retryable_exceptions: Tuple of exception types to retry on
        operation_name: Name for logging purposes

    Returns:
        Result of successful func() call

    Raises:
        Last exception if all retries exhausted
    """
    logger = logging.getLogger("librarian_orchestrator")
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                raise

            # Calculate backoff with exponential increase
            backoff = min(backoff_base * (2 ** attempt), backoff_max)
            logger.warning(
                f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            await asyncio.sleep(backoff)

    # Should never reach here, but just in case
    raise last_exception if last_exception else RuntimeError("Unexpected retry state")

from .doc_expert import DocExpertAgent, create_doc_expert, create_doc_expert_pooled, get_connection_pool
from .vision_expert import VisionExpertAgent, create_vision_expert
from .models import ModificationAdvice, RenderDiagnosis, AgentSearchResult


# =============================================================================
# DYNAMIC REASONING EFFORT
# =============================================================================

# Complexity indicators for query analysis
COMPLEXITY_INDICATORS = {
    # High complexity - need deep reasoning
    "high": [
        "why does", "why is", "debug", "diagnose", "analyze", "compare",
        "investigate", "root cause", "broken", "not working", "failing",
        "optimize", "improve", "refactor", "architecture", "design",
        "multiple", "several", "combination", "together with",
        "tradeoff", "trade-off", "best approach", "recommend",
    ],
    # Medium complexity - moderate reasoning
    "medium": [
        "how do i", "how to", "what is the", "explain", "difference between",
        "parameter for", "configure", "setup", "integrate", "implement",
        "example", "tutorial", "guide", "workflow",
    ],
    # Low complexity - quick lookup
    "low": [
        "what is", "where is", "list", "show", "get", "find",
        "default", "range", "value", "path", "location",
    ],
}


def estimate_query_complexity(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Estimate the complexity of a query to determine reasoning effort.

    Args:
        query: The user's query string
        context: Optional context dict (presence of certain keys increases complexity)

    Returns:
        Reasoning effort level: "low", "medium", or "high"
    """
    query_lower = query.lower()
    query_length = len(query.split())

    # Start with base score
    score = 0

    # Check complexity indicators
    for indicator in COMPLEXITY_INDICATORS["high"]:
        if indicator in query_lower:
            score += 3

    for indicator in COMPLEXITY_INDICATORS["medium"]:
        if indicator in query_lower:
            score += 2

    for indicator in COMPLEXITY_INDICATORS["low"]:
        if indicator in query_lower:
            score += 1

    # Query length factor
    if query_length > 50:
        score += 3
    elif query_length > 25:
        score += 2
    elif query_length > 10:
        score += 1

    # Context complexity factors
    if context:
        # Multiple issues = more complex
        issues = context.get("known_issues", context.get("issues", []))
        if isinstance(issues, list) and len(issues) > 2:
            score += 3
        elif isinstance(issues, list) and len(issues) > 0:
            score += 1

        # Vision analysis needed = more complex
        if context.get("render_path") or context.get("reference_path"):
            score += 2

        # Multiple parameters = more complex
        params = context.get("current_params", {})
        if isinstance(params, dict) and len(params) > 5:
            score += 2

    # Map score to effort level
    if score >= 8:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"


def get_reasoning_settings(effort: str) -> ModelSettings:
    """
    Get ModelSettings with appropriate reasoning effort.

    Args:
        effort: "low", "medium", or "high"

    Returns:
        ModelSettings configured for the effort level
    """
    return ModelSettings(
        reasoning=Reasoning(effort=effort),
        verbosity="low"
    )

# Configure logging for tracing
logger = logging.getLogger("librarian_orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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

        # Create specialized agents using pooled MCP connection (eliminates subprocess spawn per query)
        # The pooled version reuses a persistent connection to blender-manual MCP server
        self._doc_expert = await create_doc_expert_pooled(
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
        context: Optional[Dict[str, Any]] = None,
        use_streaming: bool = True
    ) -> Dict[str, Any]:
        """
        Run the orchestrator with a query.

        Uses streaming mode by default to keep connections alive during
        long-running multi-agent workflows. This prevents MCP timeouts.

        Args:
            query: User query or issue description
            context: Optional context dict with:
                - effect_type: Type of effect (sun, explosion, etc.)
                - current_params: Current Blender parameters
                - issues: List of known issues
                - render_path: Path to current render
                - reference_path: Path to reference image
            use_streaming: Use run_streamed() for timeout resilience (default True)

        Returns:
            Result dict with response, conversation history, and any extracted data
        """
        if not self._initialized:
            await self.initialize()

        # Estimate query complexity for dynamic reasoning effort
        complexity = estimate_query_complexity(query, context)
        logger.info(f"Query complexity estimated as '{complexity}' for: {query[:80]}...")

        # Update orchestrator's reasoning effort based on complexity
        # This allows simple queries to run faster with less reasoning overhead
        self._orchestrator.model_settings = get_reasoning_settings(complexity)

        # Build input with context
        if context:
            context_str = json.dumps(context, indent=2)
            full_query = f"Context:\n```json\n{context_str}\n```\n\nQuery: {query}"
        else:
            full_query = query

        agents_used = ["orchestrator"]
        result = None  # For non-streaming mode

        if use_streaming:
            # STREAMING MODE: Keeps connection alive during long operations
            # Events are yielded incrementally, preventing timeout
            # Wrapped with retry logic for transient failures
            #
            # IMPORTANT: MCP has ~30s timeout, so we impose 25s internal limit
            # to return partial results rather than hang
            MCP_TIMEOUT_BUFFER = 25.0  # Return before MCP times out

            # Track partial state for timeout handling
            partial_state = {
                "agents_used": ["orchestrator"],
                "event_count": 0,
                "tool_calls": [],
                "last_output": "",
                "timed_out": False,
            }

            async def _run_streamed():
                nonlocal agents_used
                _agents_used = ["orchestrator"]

                logger.info(f"Starting streamed orchestrator run for query: {query[:100]}...")

                streamed_result = Runner.run_streamed(self._orchestrator, full_query)
                _final_output = ""
                event_count = 0

                async for event in streamed_result.stream_events():
                    event_count += 1
                    partial_state["event_count"] = event_count

                    # Log event types for tracing (helps debug timeout issues)
                    if event.type == "raw_response_event":
                        # Token-by-token streaming from LLM
                        logger.debug(f"Event {event_count}: raw_response")

                    elif event.type == "agent_updated_stream_event":
                        # Agent handoff occurred
                        new_agent = getattr(event, 'new_agent', None)
                        if new_agent:
                            agent_name = new_agent.name if hasattr(new_agent, 'name') else str(new_agent)
                            logger.info(f"Event {event_count}: Handoff to {agent_name}")
                            if "doc" in agent_name.lower():
                                _agents_used.append("doc_expert")
                            elif "vision" in agent_name.lower():
                                _agents_used.append("vision_expert")

                    elif event.type == "run_item_stream_event":
                        # Tool calls, messages, etc.
                        item = getattr(event, 'item', None)
                        if item:
                            item_type = getattr(item, 'type', 'unknown')
                            logger.debug(f"Event {event_count}: run_item ({item_type})")
                            # Track tool calls for partial results
                            if item_type == "tool_call":
                                tool_name = getattr(item, 'name', 'unknown')
                                partial_state["tool_calls"].append(tool_name)

                # Update partial state
                partial_state["agents_used"] = _agents_used

                # FIXED: Access final_output directly after stream completes
                # DO NOT call await streamed_result.result() - it hangs waiting for already-complete stream
                _final_output = streamed_result.final_output
                partial_state["last_output"] = _final_output
                logger.info(f"Streamed run complete: {event_count} events, {len(_final_output) if _final_output else 0} chars output")

                agents_used = _agents_used
                return _final_output

            async def _run_with_timeout():
                try:
                    return await asyncio.wait_for(
                        retry_with_backoff(
                            _run_streamed,
                            max_retries=DEFAULT_MAX_RETRIES,
                            backoff_base=DEFAULT_BACKOFF_BASE,
                            retryable_exceptions=(TimeoutError, ConnectionError, OSError),
                            operation_name="streamed_orchestrator_run"
                        ),
                        timeout=MCP_TIMEOUT_BUFFER
                    )
                except asyncio.TimeoutError:
                    partial_state["timed_out"] = True
                    logger.warning(f"MCP timeout reached after {MCP_TIMEOUT_BUFFER}s, returning partial results")
                    # Return partial result if we have any
                    if partial_state["last_output"]:
                        return partial_state["last_output"]
                    # Otherwise return a timeout message with progress info
                    return json.dumps({
                        "timeout": True,
                        "message": f"Query took too long (>{MCP_TIMEOUT_BUFFER}s). Partial progress: {partial_state['event_count']} events, {len(partial_state['tool_calls'])} tool calls.",
                        "tool_calls_made": partial_state["tool_calls"],
                        "suggestion": "Try using search_docs_intelligent for faster responses, or simplify your query."
                    })

            final_output = await _run_with_timeout()

            # Update agents_used from partial state if we timed out
            if partial_state["timed_out"]:
                agents_used = partial_state["agents_used"]

        else:
            # NON-STREAMING MODE: Simple but may timeout on long operations
            # Wrapped with retry logic for transient failures

            async def _run_non_streamed():
                nonlocal result
                logger.info(f"Starting non-streamed orchestrator run for query: {query[:100]}...")
                result = await Runner.run(self._orchestrator, full_query)
                logger.info(f"Non-streamed run complete: {len(result.final_output)} chars output")
                return result.final_output

            # Retry transient failures with exponential backoff
            final_output = await retry_with_backoff(
                _run_non_streamed,
                max_retries=DEFAULT_MAX_RETRIES,
                backoff_base=DEFAULT_BACKOFF_BASE,
                retryable_exceptions=(TimeoutError, ConnectionError, OSError),
                operation_name="non_streamed_orchestrator_run"
            )

            agents_used = self._extract_agents_used(result)

        # Parse response - handle Pydantic models, JSON strings, or raw text
        response_data = {"raw_response": final_output}
        if isinstance(final_output, BaseModel):
            # Pydantic model output (from agents with output_type)
            response_data.update(final_output.model_dump())
        elif isinstance(final_output, str):
            try:
                parsed = json.loads(final_output)
                response_data.update(parsed)
            except (json.JSONDecodeError, TypeError):
                response_data["answer"] = final_output
        elif isinstance(final_output, dict):
            response_data.update(final_output)

        return {
            "response": response_data,
            "conversation": result.to_input_list() if not use_streaming else [],
            "agents_used": list(set(agents_used)),
            "reasoning_effort": complexity  # Dynamic reasoning effort used for this query
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
