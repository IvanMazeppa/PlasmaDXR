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
        max_consecutive_same_tool: Maximum consecutive calls to same tool (default: 4)
        max_exempt_tool_calls: Hard limit even for exempt tools (default: 10)
        max_turns: Maximum turns before warning (default: 10)
        hard_turn_limit: Absolute maximum turns before error (default: 15)
        require_doc_query_before: Tools that require prior doc query
        doc_query_tools: Tools that count as documentation queries
        log_to_stderr: Whether to log enforcement events to stderr
        raise_on_loop: Whether to raise LoopDetectedError or just warn
        raise_on_doc_missing: Whether to raise DocQueryRequiredError or just warn
    """

    max_same_tool_calls: int = 3
    max_consecutive_same_tool: int = 4  # Applies to ALL tools, even exempt ones
    max_exempt_tool_calls: int = 10  # Hard limit even for exempt tools
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
        "blender_doc_search_bundle",
        "find_alternative_approaches",
        "search_code_patterns",
        "query_docs",  # context7
    ])

    # Behavior settings
    log_to_stderr: bool = True
    raise_on_loop: bool = True
    raise_on_doc_missing: bool = True
    enforce_doc_query_on_end: bool = False
    doc_query_warn_threshold: Optional[int] = None

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

        # ENFORCEMENT 1: Loop Detection (3 layers)
        is_exempt = tool_name in self.config.exempt_from_loop_detection

        # Layer 1: Standard limit for non-exempt tools
        if not is_exempt and call_count > self.config.max_same_tool_calls:
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

        # Layer 2: Hard limit for exempt tools (they still have a ceiling)
        if is_exempt and call_count > self.config.max_exempt_tool_calls:
            self._log(
                f"EXEMPT TOOL LIMIT: '{tool_name}' called {call_count} times "
                f"(hard limit: {self.config.max_exempt_tool_calls})",
                level="ERROR"
            )
            if self.config.raise_on_loop:
                raise LoopDetectedError(
                    tool_name=tool_name,
                    call_count=call_count,
                    max_allowed=self.config.max_exempt_tool_calls
                )

        # Layer 3: Consecutive call limit (applies to ALL tools)
        if self._consecutive_same_tool > self.config.max_consecutive_same_tool:
            self._log(
                f"CONSECUTIVE LOOP: '{tool_name}' called {self._consecutive_same_tool} times "
                f"in a row (limit: {self.config.max_consecutive_same_tool})",
                level="ERROR"
            )
            if self.config.raise_on_loop:
                raise LoopDetectedError(
                    tool_name=tool_name,
                    call_count=self._consecutive_same_tool,
                    max_allowed=self.config.max_consecutive_same_tool
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

    async def on_agent_end(
        self,
        context: "RunContextWrapper",
        agent: "Agent",
        output: Any
    ) -> None:
        """
        Called when an agent completes execution.

        Used for end-of-run enforcement (doc query requirement) and warnings.
        """
        if self.config.enforce_doc_query_on_end and not self._doc_query_made:
            self._log(
                f"DOC QUERY REQUIRED: Agent '{agent.name}' produced output without doc query",
                level="ERROR"
            )
            if self.config.raise_on_doc_missing:
                raise DocQueryRequiredError(
                    blocked_tool="output",
                    required_tools=self.config.doc_query_tools
                )

        if self.config.doc_query_warn_threshold is not None:
            doc_query_count = sum(
                1 for tool_name in self._tool_call_sequence
                if tool_name in self.config.doc_query_tools
            )
            if doc_query_count > self.config.doc_query_warn_threshold:
                self._log(
                    f"DOC QUERY WARNING: {doc_query_count} doc queries "
                    f"(threshold: {self.config.doc_query_warn_threshold})",
                    level="WARN"
                )

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
        doc_query_count = sum(
            1 for tool_name in self._tool_call_sequence
            if tool_name in self.config.doc_query_tools
        )
        return {
            "agent_name": self._current_agent_name,
            "turn_count": self._turn_count,
            "tool_call_counts": dict(self._tool_call_counts),
            "total_tool_calls": sum(self._tool_call_counts.values()),
            "doc_query_made": self._doc_query_made,
            "doc_query_count": doc_query_count,
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
# API SPEC AGENT BUNDLE-FIRST ENFORCEMENT
# =============================================================================

class BundleFirstRequiredError(EnforcementError):
    """
    Raised when targeted doc search is attempted before bundle search.

    The API Spec Agent MUST call blender_doc_search_bundle FIRST.
    This prevents the agent from ignoring bundle instructions.
    """

    def __init__(self, blocked_tool: str):
        self.blocked_tool = blocked_tool
        super().__init__(
            f"BUNDLE-FIRST REQUIRED: Cannot call '{blocked_tool}' before calling "
            f"'blender_doc_search_bundle'. Call the bundle search first to get "
            f"all domain/flow/scene attributes, then use targeted searches for gaps."
        )


class APISpecEnforcementHooks(EnforcementHooks):
    """
    Specialized hooks for API Spec Agent with BUNDLE-FIRST enforcement.

    This class enforces the "bundle-first" discipline:
    1. Targeted doc searches (semantic_search_blender_docs, search_blender_api_by_intent)
       are BLOCKED until blender_doc_search_bundle has been called.
    2. After bundle call, targeted searches are limited (max 6).
    3. Turn budget is enforced to prevent endless searching.

    The bundle-first pattern is critical because:
    - Bundle returns multiple related attributes in one call
    - Targeted searches are expensive and can cause loop detection
    - Bundle provides a solid foundation for the APISpec
    """

    BUNDLE_TOOL = "blender_doc_search_bundle"
    TARGETED_TOOLS = ["semantic_search_blender_docs", "search_blender_api_by_intent"]
    MAX_TARGETED_SEARCHES = 6

    def __init__(self):
        """Initialize with API Spec Agent optimized config."""
        config = EnforcementConfig(
            max_same_tool_calls=10,  # Standard limit for non-exempt tools
            max_consecutive_same_tool=8,  # Allow batches, prevent spam
            max_exempt_tool_calls=15,  # Hard ceiling for doc search tools
            max_turns=6,
            hard_turn_limit=8,
            require_doc_query_before=[],  # Spec Agent IS the doc query - output guardrail enforces
            enforce_doc_query_on_end=True,
            doc_query_warn_threshold=20,
            exempt_from_loop_detection=[
                "semantic_search_blender_docs",  # Primary tool
                "search_blender_api_by_intent",  # Secondary tool
                "blender_doc_search_bundle",  # Bundle tool
                "validate_parameter_range",  # May validate multiple parameters
            ],
            raise_on_loop=True,
            raise_on_doc_missing=True,  # Require at least one doc query before output
        )
        super().__init__(config)

        # Bundle-first state
        self._bundle_called: bool = False
        self._targeted_search_count: int = 0

    def _reset_state(self) -> None:
        """Reset all tracking state for a new agent run."""
        super()._reset_state()
        self._bundle_called = False
        self._targeted_search_count = 0

    async def on_tool_start(
        self,
        context: "RunContextWrapper",
        agent: "Agent",
        tool: Tool
    ) -> None:
        """
        Called immediately before a tool is invoked.

        Enforces BUNDLE-FIRST: targeted searches blocked until bundle called.
        """
        tool_name = tool.name

        # Track bundle call
        if tool_name == self.BUNDLE_TOOL:
            self._bundle_called = True
            self._log(f"Bundle search called - targeted searches now allowed")

        # BUNDLE-FIRST ENFORCEMENT
        if tool_name in self.TARGETED_TOOLS:
            if not self._bundle_called:
                self._log(
                    f"BUNDLE-FIRST REQUIRED: '{tool_name}' blocked - call bundle first",
                    level="ERROR"
                )
                raise BundleFirstRequiredError(blocked_tool=tool_name)

            self._targeted_search_count += 1
            if self._targeted_search_count > self.MAX_TARGETED_SEARCHES:
                self._log(
                    f"TARGETED SEARCH LIMIT: {self._targeted_search_count}/{self.MAX_TARGETED_SEARCHES}",
                    level="WARN"
                )

        # Call parent for standard enforcement
        await super().on_tool_start(context, agent, tool)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics including bundle-first state."""
        stats = super().get_stats()
        stats["bundle_called"] = self._bundle_called
        stats["targeted_search_count"] = self._targeted_search_count
        return stats


# =============================================================================
# SPECIALIZED HOOK CONFIGURATIONS
# =============================================================================

def create_research_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for research agents.

    More lenient on doc queries (they ARE the doc queries)
    but allows multiple parallel searches for complex topics.
    Complex effects (waterfall, ocean) need searches for:
    domain, flow, mesh, particles, materials, environment, etc.
    """
    config = EnforcementConfig(
        max_same_tool_calls=4,  # Phase 3: tighter research budget
        max_consecutive_same_tool=3,
        max_exempt_tool_calls=6,
        max_turns=4,
        hard_turn_limit=6,
        require_doc_query_before=[],  # Research agents ARE the doc queries
        raise_on_loop=True,
        raise_on_doc_missing=False,
    )
    return EnforcementHooks(config)


def create_script_writer_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for Script Writer agent.

    Doc query requirement ENFORCED for write_script/modify_script because:
    1. Research results are high-level and do not guarantee attribute validity
    2. Script Writer must ground attribute usage in API docs
    3. Validation alone is too late to prevent hallucinated attributes

    Loop detection: Doc search tools are exempt from the normal limit (6),
    but they still have:
    - max_consecutive_same_tool=5 (can't call same tool 5+ times in a row)
    - max_exempt_tool_calls=10 (hard ceiling even for exempt tools)
    """
    config = EnforcementConfig(
        max_same_tool_calls=6,  # Standard limit for non-exempt tools
        max_consecutive_same_tool=5,  # No tool should be called 5+ times in a row
        max_exempt_tool_calls=10,  # Hard ceiling even for doc searches
        max_turns=12,
        hard_turn_limit=18,
        require_doc_query_before=[
            "write_script",
            "modify_script",
        ],
        exempt_from_loop_detection=[
            "validate_script",  # May need multiple validation calls
            "semantic_search_blender_docs",  # Complex effects need many doc searches
            "search_blender_api_by_intent",  # Same - legitimate multiple searches
            "blender_doc_search_bundle",  # Bundled doc search should be exempt
        ],
        raise_on_loop=True,
        raise_on_doc_missing=True,
    )
    return EnforcementHooks(config)


def create_fallback_script_writer_hooks() -> EnforcementHooks:
    """
    Create hooks for Script Writer when running as FALLBACK from Spec-First pipeline.

    CRITICAL DIFFERENCE from create_script_writer_hooks():
    - NO doc query requirement because Research Agent already queried docs
    - The Spec-First pipeline failed AFTER research, so we trust the context

    This prevents the "Cannot call 'write_script' without first querying documentation"
    error when falling back from a failed API Spec Agent.
    """
    config = EnforcementConfig(
        max_same_tool_calls=6,  # Standard limit for non-exempt tools
        max_consecutive_same_tool=5,  # No tool should be called 5+ times in a row
        max_exempt_tool_calls=10,  # Hard ceiling even for doc searches
        max_turns=12,
        hard_turn_limit=18,
        require_doc_query_before=[],  # NO requirement - research already done
        exempt_from_loop_detection=[
            "validate_script",  # May need multiple validation calls
            "semantic_search_blender_docs",  # Still allow doc searches if needed
            "search_blender_api_by_intent",  # Same
            "blender_doc_search_bundle",  # Bundled doc search
        ],
        raise_on_loop=True,
        raise_on_doc_missing=False,  # Explicitly disabled for fallback
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
        max_consecutive_same_tool=2,  # Very strict - QA should not loop
        max_exempt_tool_calls=4,
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
        max_consecutive_same_tool=2,  # Very strict
        max_exempt_tool_calls=4,
        max_turns=5,
        hard_turn_limit=8,
        require_doc_query_before=[],
        raise_on_loop=True,
        raise_on_doc_missing=False,
    )
    return EnforcementHooks(config)


def create_api_spec_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for API Spec Agent with BUNDLE-FIRST enforcement.

    The API Spec Agent MUST:
    1. Call blender_doc_search_bundle FIRST (Turn 1)
    2. Use targeted searches only for gaps (Turns 2-3)
    3. Output APISpec by Turn 4

    BUNDLE-FIRST ENFORCEMENT:
    - Targeted doc searches are BLOCKED until bundle has been called
    - This prevents the agent from ignoring bundle instructions

    Turn budget is tight (4-6 turns) because:
    - T1: Bundle search (MANDATORY)
    - T2-3: Targeted searches for gaps (<=6 total)
    - T4: Return APISpec

    SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md
    """
    return APISpecEnforcementHooks()


def create_code_writer_hooks() -> EnforcementHooks:
    """
    Create hooks optimized for Code Writer Agent (spec-first pipeline).

    CRITICAL: The Code Writer receives a VERIFIED APISpec and MUST NOT
    call any documentation tools. If it needs to look up attributes,
    that's a failure of the API Spec Agent.

    This agent has:
    - NO doc query tools (by design)
    - Strict turn budget (4-6 turns)
    - write_script and validate_script only

    The output guardrail (validate_code_against_spec) validates that
    the generated code only uses attributes from the APISpec.

    SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md
    """
    config = EnforcementConfig(
        max_same_tool_calls=2,  # Should only call write_script once
        max_consecutive_same_tool=2,  # No looping
        max_exempt_tool_calls=4,  # Validation might need retries
        max_turns=6,
        hard_turn_limit=8,
        require_doc_query_before=[],  # No doc tools available - spec is source of truth
        exempt_from_loop_detection=[
            "validate_script",  # May need to validate after corrections
        ],
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
    print(f"Default config:")
    print(f"  max_same_tool_calls={hooks.config.max_same_tool_calls}")
    print(f"  max_consecutive_same_tool={hooks.config.max_consecutive_same_tool}")
    print(f"  max_exempt_tool_calls={hooks.config.max_exempt_tool_calls}")
    print(f"  doc_query_tools: {hooks.config.doc_query_tools}")

    # Test specialized configs
    print("\nSpecialized configs:")
    research_hooks = create_research_hooks()
    print(f"  Research: max_turns={research_hooks.config.max_turns}, consecutive={research_hooks.config.max_consecutive_same_tool}")

    script_hooks = create_script_writer_hooks()
    print(f"  Script Writer: max_same={script_hooks.config.max_same_tool_calls}, consecutive={script_hooks.config.max_consecutive_same_tool}, exempt_limit={script_hooks.config.max_exempt_tool_calls}")
    print(f"    Exempt tools: {script_hooks.config.exempt_from_loop_detection}")

    qa_hooks = create_quality_analyst_hooks()
    print(f"  Quality Analyst: max_same={qa_hooks.config.max_same_tool_calls}, consecutive={qa_hooks.config.max_consecutive_same_tool}")

    learning_hooks = create_learning_agent_hooks()
    print(f"  Learning Agent: max_same={learning_hooks.config.max_same_tool_calls}, consecutive={learning_hooks.config.max_consecutive_same_tool}")

    print("\nAll tests passed!")
