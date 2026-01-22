"""
Enforcement Hooks for OpenAI Agents SDK.

This module implements RunHooks that mechanically enforce SDK patterns
and prevent common failure modes:

1. Loop Detection - Stops agents from calling the same tool repeatedly
2. Doc Query Requirement - Blocks code generation without prior doc lookup
3. Turn Budget - Hard limits on agent reasoning turns
4. Output Validation - Ensures structured outputs are complete

These hooks address Problems 2 and 3 from the Architecture Optimization Plan:
- Problem 2: Research loop infinity (agents calling search 15+ times)
- Problem 3: SDK docs not being used as primary source

Usage:
    from hooks import EnforcementHooks, EnforcementConfig

    config = EnforcementConfig(
        max_same_tool_calls=3,
        max_turns=10,
        require_doc_query_before=["write_script", "modify_script"]
    )
    hooks = EnforcementHooks(config)

    result = await Runner.run(agent, prompt, run_hooks=hooks)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from agents import RunHooks, Tool

if TYPE_CHECKING:
    from agents import RunContextWrapper, Agent


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class EnforcementError(Exception):
    """Base exception for enforcement hook violations."""
    pass


class LoopDetectedError(EnforcementError):
    """
    Raised when an agent is stuck in a tool-calling loop.

    This exception signals that the agent should stop researching
    and produce output with whatever information it has gathered.
    """

    def __init__(self, tool_name: str, call_count: int, max_allowed: int):
        self.tool_name = tool_name
        self.call_count = call_count
        self.max_allowed = max_allowed
        super().__init__(
            f"Loop detected: Tool '{tool_name}' called {call_count} times "
            f"(limit: {max_allowed}). Stop researching and produce output."
        )


class DocQueryRequiredError(EnforcementError):
    """
    Raised when code generation is attempted without prior documentation query.

    This exception enforces the "docs first" pattern - agents MUST query
    Blender 5.0 documentation before writing any code that uses bpy.types
    or bpy.ops.
    """

    def __init__(self, blocked_tool: str, required_tools: List[str]):
        self.blocked_tool = blocked_tool
        self.required_tools = required_tools
        super().__init__(
            f"Cannot call '{blocked_tool}' without first querying documentation. "
            f"Use one of: {', '.join(required_tools)}"
        )


class TurnBudgetExceededError(EnforcementError):
    """
    Raised when an agent exceeds its turn budget.

    This is a soft warning - the agent run continues but this exception
    can be caught to force early termination or output production.
    """

    def __init__(self, turn_count: int, max_turns: int, agent_name: str):
        self.turn_count = turn_count
        self.max_turns = max_turns
        self.agent_name = agent_name
        super().__init__(
            f"Agent '{agent_name}' exceeded turn budget: {turn_count}/{max_turns}. "
            f"Forcing output production."
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnforcementConfig:
    """
    Configuration for enforcement hooks.

    Attributes:
        max_same_tool_calls: Maximum times the same tool can be called (default: 3)
        max_turns: Maximum turns before warning (default: 10)
        hard_turn_limit: Absolute maximum turns before error (default: 15)
        require_doc_query_before: Tools that require prior doc query
        doc_query_tools: Tools that count as documentation queries
        log_to_stderr: Whether to log enforcement events to stderr
        raise_on_loop: Whether to raise LoopDetectedError or just warn
        raise_on_doc_missing: Whether to raise DocQueryRequiredError or just warn
    """

    max_same_tool_calls: int = 3
    max_turns: int = 10
    hard_turn_limit: int = 15

    require_doc_query_before: List[str] = field(default_factory=lambda: [
        "write_script",
        "generate_script",
        "modify_script",
    ])

    doc_query_tools: List[str] = field(default_factory=lambda: [
        "semantic_search_blender_docs",
        "search_blender_api_by_intent",
        "find_alternative_approaches",
        "search_code_patterns",
        "query_docs",  # context7
    ])

    # Behavior settings
    log_to_stderr: bool = True
    raise_on_loop: bool = True
    raise_on_doc_missing: bool = True

    # Tools that are exempt from loop detection (always allowed)
    exempt_from_loop_detection: List[str] = field(default_factory=lambda: [
        "validate_script",  # May need multiple validation calls
    ])


# =============================================================================
# ENFORCEMENT HOOKS
# =============================================================================

class EnforcementHooks(RunHooks):
    """
    RunHooks implementation that enforces SDK patterns and prevents failures.

    This class provides mechanical enforcement of rules that LLMs don't
    reliably follow from instructions alone:

    1. **Loop Detection**: Prevents agents from calling the same tool
       repeatedly without making progress. After N calls to the same tool,
       raises LoopDetectedError to force output production.

    2. **Doc Query Requirement**: Blocks code generation tools unless
       documentation has been queried first. This enforces the "docs first"
       pattern critical for Blender 5.0 API correctness.

    3. **Turn Budget**: Tracks turn count and warns/errors when budget
       is exceeded. Helps identify runaway agents.

    4. **Visibility**: Logs all tool calls to stderr for debugging.

    Example:
        ```python
        from hooks import EnforcementHooks, EnforcementConfig

        config = EnforcementConfig(
            max_same_tool_calls=3,
            require_doc_query_before=["write_script", "modify_script"]
        )
        hooks = EnforcementHooks(config)

        # The hooks will now enforce patterns during this run
        result = await Runner.run(
            agent,
            prompt,
            run_hooks=hooks,
            max_turns=10
        )
        ```
    """

    def __init__(self, config: Optional[EnforcementConfig] = None):
        """
        Initialize enforcement hooks.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or EnforcementConfig()

        # State tracking (reset per agent run)
        self._tool_call_counts: Dict[str, int] = {}
        self._doc_query_made: bool = False
        self._turn_count: int = 0
        self._current_agent_name: str = "Unknown"
        self._run_start_time: Optional[datetime] = None

        # Sequence tracking for pattern detection
        self._tool_call_sequence: List[str] = []
        self._consecutive_same_tool: int = 0
        self._last_tool_called: Optional[str] = None

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message to stderr if logging is enabled."""
        if self.config.log_to_stderr:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[Hooks {timestamp}] [{level}] {message}", file=sys.stderr)

    def _reset_state(self) -> None:
        """Reset all tracking state for a new agent run."""
        self._tool_call_counts.clear()
        self._doc_query_made = False
        self._turn_count = 0
        self._tool_call_sequence.clear()
        self._consecutive_same_tool = 0
        self._last_tool_called = None
        self._run_start_time = datetime.now()

    # =========================================================================
    # RUNHOOKS LIFECYCLE METHODS
    # =========================================================================

    async def on_agent_start(
        self,
        context: "RunContextWrapper",
        agent: "Agent"
    ) -> None:
        """
        Called when an agent starts execution.

        Resets all tracking state for the new run.
        """
        self._reset_state()
        self._current_agent_name = agent.name
        self._log(f"Agent '{agent.name}' starting")

    async def on_tool_start(
        self,
        context: "RunContextWrapper",
        agent: "Agent",
        tool: Tool
    ) -> None:
        """
        Called immediately before a tool is invoked.

        This is where we enforce:
        1. Loop detection - block if same tool called too many times
        2. Doc query requirement - block code gen without prior doc lookup
        """
        tool_name = tool.name

        # Track tool call
        self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1
        self._tool_call_sequence.append(tool_name)
        call_count = self._tool_call_counts[tool_name]

        # Track consecutive calls to same tool
        if tool_name == self._last_tool_called:
            self._consecutive_same_tool += 1
        else:
            self._consecutive_same_tool = 1
        self._last_tool_called = tool_name

        self._log(f"Tool '{tool_name}' called (count: {call_count}, consecutive: {self._consecutive_same_tool})")

        # Check if this is a doc query tool
        if tool_name in self.config.doc_query_tools:
            self._doc_query_made = True
            self._log(f"Doc query detected via '{tool_name}'")

        # ENFORCEMENT 1: Loop Detection
        if tool_name not in self.config.exempt_from_loop_detection:
            if call_count > self.config.max_same_tool_calls:
                self._log(
                    f"LOOP DETECTED: '{tool_name}' called {call_count} times "
                    f"(limit: {self.config.max_same_tool_calls})",
                    level="ERROR"
                )
                if self.config.raise_on_loop:
                    raise LoopDetectedError(
                        tool_name=tool_name,
                        call_count=call_count,
                        max_allowed=self.config.max_same_tool_calls
                    )

        # ENFORCEMENT 2: Doc Query Requirement
        if tool_name in self.config.require_doc_query_before:
            if not self._doc_query_made:
                self._log(
                    f"DOC QUERY REQUIRED: '{tool_name}' blocked - no prior doc query",
                    level="ERROR"
                )
                if self.config.raise_on_doc_missing:
                    raise DocQueryRequiredError(
                        blocked_tool=tool_name,
                        required_tools=self.config.doc_query_tools
                    )

    async def on_tool_end(
        self,
        context: "RunContextWrapper",
        agent: "Agent",
        tool: Tool,
        result: str
    ) -> None:
        """
        Called immediately after a tool returns.

        Used for logging and result analysis.
        """
        tool_name = tool.name
        result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        self._log(f"Tool '{tool_name}' returned: {result_preview}")

    async def on_handoff(
        self,
        context: "RunContextWrapper",
        from_agent: "Agent",
        to_agent: "Agent"
    ) -> None:
        """
        Called when control is transferred between agents.

        Logs the handoff for visibility.
        """
        self._log(f"Handoff: '{from_agent.name}' -> '{to_agent.name}'")

        # Reset tool counts for new agent (they have their own budget)
        self._tool_call_counts.clear()
        self._doc_query_made = False
        self._consecutive_same_tool = 0
        self._last_tool_called = None
        self._current_agent_name = to_agent.name

    # =========================================================================
    # ADDITIONAL ENFORCEMENT METHODS
    # =========================================================================

    def check_turn_budget(self, turn: int) -> None:
        """
        Check if turn budget is exceeded.

        Call this from the orchestrator after each agent turn.

        Args:
            turn: Current turn number

        Raises:
            TurnBudgetExceededError: If hard limit exceeded
        """
        self._turn_count = turn

        if turn > self.config.hard_turn_limit:
            self._log(
                f"HARD TURN LIMIT: {turn}/{self.config.hard_turn_limit}",
                level="ERROR"
            )
            raise TurnBudgetExceededError(
                turn_count=turn,
                max_turns=self.config.hard_turn_limit,
                agent_name=self._current_agent_name
            )
        elif turn > self.config.max_turns:
            self._log(
                f"TURN BUDGET WARNING: {turn}/{self.config.max_turns}",
                level="WARN"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current/completed run.

        Returns:
            Dict with tool call counts, turn count, etc.
        """
        return {
            "agent_name": self._current_agent_name,
            "turn_count": self._turn_count,
            "tool_call_counts": dict(self._tool_call_counts),
            "total_tool_calls": sum(self._tool_call_counts.values()),
            "doc_query_made": self._doc_query_made,
            "tool_call_sequence": list(self._tool_call_sequence),
            "run_duration_seconds": (
                (datetime.now() - self._run_start_time).total_seconds()
                if self._run_start_time else 0
            ),
        }

    def was_loop_detected(self) -> bool:
        """Check if a loop was detected during the run."""
        return any(
            count > self.config.max_same_tool_calls
            for tool, count in self._tool_call_counts.items()
            if tool not in self.config.exempt_from_loop_detection
        )

    def get_most_called_tool(self) -> Optional[tuple]:
        """Get the most frequently called tool and its count."""
        if not self._tool_call_counts:
            return None
        tool = max(self._tool_call_counts, key=self._tool_call_counts.get)
        return (tool, self._tool_call_counts[tool])


# =============================================================================
# SPECIALIZED HOOK CONFIGURATIONS
# =============================================================================

def create_research_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for research agents.

    More lenient on doc queries (they ARE the doc queries)
    but strict on preventing infinite research loops.
    """
    config = EnforcementConfig(
        max_same_tool_calls=3,  # Strict - research shouldn't repeat
        max_turns=8,
        hard_turn_limit=12,
        require_doc_query_before=[],  # Research agents ARE the doc queries
        raise_on_loop=True,
        raise_on_doc_missing=False,
    )
    return EnforcementHooks(config)


def create_script_writer_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for Script Writer agent.

    Strict doc query requirement, moderate loop tolerance
    (may need a few validation attempts).
    """
    config = EnforcementConfig(
        max_same_tool_calls=3,
        max_turns=10,
        hard_turn_limit=15,
        require_doc_query_before=[
            "write_script",
            "generate_script",
            "modify_script",
        ],
        exempt_from_loop_detection=[
            "validate_script",  # May need multiple validation calls
        ],
        raise_on_loop=True,
        raise_on_doc_missing=True,
    )
    return EnforcementHooks(config)


def create_quality_analyst_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for Quality Analyst agent.

    No doc query requirement, focus on preventing
    excessive evaluation loops.
    """
    config = EnforcementConfig(
        max_same_tool_calls=2,  # Evaluation should be decisive
        max_turns=6,
        hard_turn_limit=10,
        require_doc_query_before=[],  # QA doesn't write code
        raise_on_loop=True,
        raise_on_doc_missing=False,
    )
    return EnforcementHooks(config)


def create_learning_agent_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for Learning Agent.

    Very strict - learning should be quick and focused.
    """
    config = EnforcementConfig(
        max_same_tool_calls=2,
        max_turns=5,
        hard_turn_limit=8,
        require_doc_query_before=[],
        raise_on_loop=True,
        raise_on_doc_missing=False,
    )
    return EnforcementHooks(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test of hook creation and configuration
    print("Testing EnforcementHooks...")
    print("-" * 60)

    # Test default config
    hooks = EnforcementHooks()
    print(f"Default config: max_same_tool={hooks.config.max_same_tool_calls}")
    print(f"Doc query tools: {hooks.config.doc_query_tools}")
    print(f"Require doc before: {hooks.config.require_doc_query_before}")

    # Test specialized configs
    research_hooks = create_research_hooks()
    print(f"\nResearch hooks: max_turns={research_hooks.config.max_turns}")

    script_hooks = create_script_writer_hooks()
    print(f"Script Writer hooks: require_doc={script_hooks.config.require_doc_query_before}")

    qa_hooks = create_quality_analyst_hooks()
    print(f"Quality Analyst hooks: max_same_tool={qa_hooks.config.max_same_tool_calls}")

    print("\nAll tests passed!")
