"""
Blender VFX Orchestrator using OpenAI Agents SDK.

Main orchestrator that coordinates specialized agents to autonomously
generate and iterate on VFX assets until quality thresholds are met.

Key capabilities:
- Multi-agent coordination via handoffs
- Quality gate enforcement
- Stuck detection and escape strategies
- Budget enforcement ($20/month limit)
- Session persistence across context limits
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents import Agent, ModelSettings, Runner, handoff
from openai.types.shared import Reasoning

from models.shared_context import (
    AssetRequest,
    SessionState,
    SessionStatus,
    SharedContext,
    IterationResult,
)
from specialized_agents import (
    create_script_writer,
    create_executor,
    create_quality_analyst,
    create_learning_agent,
)
# NOTE: DocsExpert uses MCP client connections which CANNOT be spawned from
# within an MCP tool handler (causes anyio task group conflicts).
# We defer initialization and make it optional for now.
# from specialized_agents.docs_expert import create_docs_expert_pooled, get_docs_connection_pool
from utils import (
    BudgetTracker,
    get_budget_tracker,
    SessionPersistence,
    get_persistence,
    generate_session_id,
    create_session_from_request,
    resume_or_create_session,
)

# Flag to enable/disable docs expert (disabled by default for MCP compatibility)
ENABLE_DOCS_EXPERT = os.environ.get("ENABLE_DOCS_EXPERT", "0") == "1"


# =============================================================================
# ORCHESTRATOR INSTRUCTIONS
# =============================================================================

ORCHESTRATOR_INSTRUCTIONS = """You are an autonomous VFX asset generation orchestrator.

Your role is to coordinate specialized agents to generate high-quality Blender VFX
assets through iterative improvement. You manage the full lifecycle from script
generation through quality evaluation.

## SPECIALIZED AGENTS (use handoffs to delegate)

1. **Script Writer** - Generates and modifies Blender Python scripts
   - delegate_to_script_writer for: new scripts, script modifications
   - Uses UCB1 algorithm for technique selection

2. **Executor** - Executes Blender scripts and handles errors
   - delegate_to_executor for: running scripts, parsing errors

3. **Quality Analyst** - Evaluates render quality using ML metrics
   - delegate_to_quality_analyst for: quality evaluation, iteration comparison

4. **Learning Agent** - Maintains experiment knowledge base
   - delegate_to_learning_agent for: suggesting fixes, recording outcomes

5. **Docs Expert** - Searches Blender documentation
   - delegate_to_docs_expert for: finding new approaches when stuck

## ITERATION LOOP

For each iteration:
1. Generate/modify script → delegate_to_script_writer
2. Execute script → delegate_to_executor
3. If execution failed → parse error and modify script, retry
4. Evaluate quality → delegate_to_quality_analyst
5. If PASSED (score >= 60, no critical issues) → complete session
6. If FAILED:
   a. Get fix suggestions → delegate_to_learning_agent
   b. Apply fixes and repeat
7. If STUCK (3+ iterations with <5% improvement):
   a. Consult documentation → delegate_to_docs_expert
   b. Try fundamentally different approach

## QUALITY GATES

Pass when ALL conditions met:
- overall_score >= 60
- No critical issues (ZERO LIGHTS, BLACK SCREEN, etc.)
- If reference provided: acceptable LPIPS similarity

## STUCK DETECTION

You are STUCK if:
- 3+ consecutive iterations with <5 point improvement
- Same primary issue persists for 3+ iterations
- Quality score plateaued (< 2 point change for 3 iterations)

When STUCK:
1. First: consult Learning Agent for alternative fixes
2. Second: consult Docs Expert for new approaches
3. Third: switch technique entirely (generate new script)

## BUDGET LIMITS

Total monthly budget: $20
- Vision/evaluation: $10
- Documentation search: $8
- Emergency buffer: $2

At 80% budget: warn and reduce evaluation profile
At 100% budget: stop and return best result

## SESSION PERSISTENCE

After each iteration, the session state is automatically saved.
If context limit is reached, the session can be resumed.

## OUTPUT FORMAT

After each iteration, report:
- Iteration number and status
- Quality score and improvement
- Primary issue (if failed)
- Next action to take

When complete, report:
- Final score and iteration count
- Paths to final outputs (render, VDBs)
- Total session cost
"""


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class BlenderVFXOrchestrator:
    """
    Main orchestrator for autonomous VFX asset generation.

    Coordinates 5 specialized agents through the OpenAI Agents SDK
    handoff mechanism. Manages the full iteration loop from script
    generation through quality evaluation.
    """

    def __init__(self):
        """Initialize the orchestrator (call initialize() before use)."""
        self._orchestrator: Optional[Agent] = None
        self._script_writer: Optional[Agent] = None
        self._executor: Optional[Agent] = None
        self._quality_analyst: Optional[Agent] = None
        self._learning_agent: Optional[Agent] = None
        self._docs_expert: Optional[Agent] = None

        self._budget_tracker: BudgetTracker = get_budget_tracker()
        self._persistence: SessionPersistence = get_persistence()
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize all specialized agents and create the orchestrator.

        Must be called before create_asset().

        NOTE: DocsExpert is disabled by default when running as MCP server
        because spawning child MCP clients from within an MCP tool handler
        causes anyio task group conflicts. Set ENABLE_DOCS_EXPERT=1 to enable
        (only when running standalone, not as MCP server).
        """
        if self._initialized:
            return

        print(f"[Orchestrator] Initializing (docs_expert={ENABLE_DOCS_EXPERT})...", file=sys.stderr)

        # Create specialized agents (synchronous agents first)
        self._script_writer = create_script_writer()
        self._executor = create_executor()
        self._quality_analyst = create_quality_analyst()
        self._learning_agent = create_learning_agent()

        # Build handoffs list (core 4 agents)
        handoffs_list = [
            handoff(
                self._script_writer,
                tool_name_override="delegate_to_script_writer",
                tool_description_override="Delegate to Script Writer for generating or modifying Blender scripts"
            ),
            handoff(
                self._executor,
                tool_name_override="delegate_to_executor",
                tool_description_override="Delegate to Executor for running Blender scripts and handling errors"
            ),
            handoff(
                self._quality_analyst,
                tool_name_override="delegate_to_quality_analyst",
                tool_description_override="Delegate to Quality Analyst for evaluating render quality"
            ),
            handoff(
                self._learning_agent,
                tool_name_override="delegate_to_learning_agent",
                tool_description_override="Delegate to Learning Agent for fix suggestions and experiment recording"
            ),
        ]

        # Conditionally add docs expert (disabled by default for MCP compatibility)
        if ENABLE_DOCS_EXPERT:
            try:
                from specialized_agents.docs_expert import create_docs_expert_pooled
                self._docs_expert = await create_docs_expert_pooled()
                handoffs_list.append(
                    handoff(
                        self._docs_expert,
                        tool_name_override="delegate_to_docs_expert",
                        tool_description_override="Delegate to Docs Expert for searching Blender documentation (use when stuck)"
                    )
                )
                print("[Orchestrator] DocsExpert enabled", file=sys.stderr)
            except Exception as e:
                print(f"[Orchestrator] DocsExpert initialization failed: {e}", file=sys.stderr)
                self._docs_expert = None
        else:
            print("[Orchestrator] DocsExpert disabled (MCP compatibility mode)", file=sys.stderr)
            self._docs_expert = None

        # Create main orchestrator with handoffs to all agents
        self._orchestrator = Agent(
            name="Blender VFX Orchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2"),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
                verbosity="low"
            ),
            handoffs=handoffs_list,
        )

        self._initialized = True
        print("[Orchestrator] Initialization complete", file=sys.stderr)

    async def create_asset(self, request: AssetRequest) -> SessionState:
        """
        Main entry point for asset generation.

        Args:
            request: Asset generation request parameters

        Returns:
            SessionState with final results
        """
        if not self._initialized:
            await self.initialize()

        # Check budget before starting
        if not self._budget_tracker.can_afford_evaluation():
            raise RuntimeError(
                f"Budget exhausted. Monthly limit: ${self._budget_tracker.monthly_limit}"
            )

        # Create session
        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Build initial prompt
        prompt = self._build_generation_prompt(request, context)

        # Run orchestrator (autonomous iteration loop)
        try:
            result = await Runner.run(
                self._orchestrator,
                prompt,
                context={"shared_context": context.model_dump()}
            )

            # Parse and update session state from result
            session = self._parse_result(result, context)

        except Exception as e:
            # Handle errors gracefully
            context.session.status = SessionStatus.FAILED
            context.session.current_issues.append(f"Orchestration error: {str(e)}")
            session = context.session

        # Save session state
        self._persistence.save_session(session)

        return session

    async def resume_session(self, session_id: str) -> SessionState:
        """
        Resume a paused or incomplete session.

        Args:
            session_id: ID of session to resume

        Returns:
            Updated SessionState
        """
        if not self._initialized:
            await self.initialize()

        # Load session
        session = self._persistence.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status in [SessionStatus.PASSED, SessionStatus.CANCELLED]:
            return session  # Already complete

        # Create context from session
        context = SharedContext(session=session)

        # Build resumption prompt
        prompt = self._build_resumption_prompt(context)

        # Run orchestrator
        try:
            result = await Runner.run(
                self._orchestrator,
                prompt,
                context={"shared_context": context.model_dump()}
            )

            session = self._parse_result(result, context)

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.current_issues.append(f"Resumption error: {str(e)}")

        # Save updated state
        self._persistence.save_session(session)

        return session

    def _build_generation_prompt(
        self,
        request: AssetRequest,
        context: SharedContext
    ) -> str:
        """Build the initial prompt for asset generation."""
        return f"""Generate a VFX asset with the following specifications:

## Request
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}
- Description: {request.description}

## Parameters
- Resolution: {request.resolution}
- Frame Range: {request.frame_start} - {request.frame_end}
- Quality Threshold: {request.quality_threshold}
- Max Iterations: {request.max_iterations}

## Reference Materials
- Reference Image: {request.reference_path or "None provided"}
- Semantic Query: {request.semantic_query or request.description}

## Instructions
1. Start by generating a script using the Script Writer
2. Execute the script using the Executor
3. Evaluate quality using the Quality Analyst
4. If quality is below threshold, iterate with fixes
5. Continue until quality threshold met or max iterations reached

Session ID: {context.session.session_id}
Current Budget Used: ${self._budget_tracker.get_spent():.2f} / ${self._budget_tracker.monthly_limit}

Begin the asset generation process."""

    def _build_resumption_prompt(self, context: SharedContext) -> str:
        """Build the prompt for resuming a session."""
        session = context.session
        request = session.request

        # Get recent history
        recent_iterations = session.iterations[-3:] if session.iterations else []
        history_summary = "\n".join([
            f"  - Iteration {it.iteration}: score={it.score:.1f}, passed={it.passed}"
            for it in recent_iterations
        ])

        return f"""Resume VFX asset generation session.

## Session Info
- Session ID: {session.session_id}
- Asset Name: {request.asset_name}
- Effect Type: {request.effect_type.value}

## Progress
- Current Iteration: {session.current_iteration}
- Best Score: {session.best_score:.1f} (iteration {session.best_iteration})
- Max Iterations: {request.max_iterations}

## Recent History
{history_summary or "  No iterations completed yet"}

## Current State
- Script Path: {session.current_script_path or "None"}
- Current Issues: {', '.join(session.current_issues) or "None"}

## Instructions
Continue from where you left off. Review the current state and proceed
with the next appropriate action.

Current Budget Used: ${self._budget_tracker.get_spent():.2f} / ${self._budget_tracker.monthly_limit}

Resume the asset generation process."""

    def _parse_result(
        self,
        result: Any,
        context: SharedContext
    ) -> SessionState:
        """Parse the orchestrator result and update session state."""
        session = context.session

        # Try to extract structured output
        if hasattr(result, 'final_output'):
            output = result.final_output

            # Try to parse as JSON
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                    if 'passed' in data:
                        session.status = SessionStatus.PASSED if data['passed'] else SessionStatus.IN_PROGRESS
                    if 'final_score' in data:
                        session.best_score = max(session.best_score, data['final_score'])
                    if 'final_render_path' in data:
                        session.final_render_path = data['final_render_path']
                except json.JSONDecodeError:
                    pass

        # Check for max iterations
        if session.current_iteration >= session.request.max_iterations:
            if session.status == SessionStatus.IN_PROGRESS:
                session.status = SessionStatus.MAX_ITERATIONS

        session.update_timestamp()
        return session

    async def close(self) -> None:
        """Clean up resources."""
        # Close docs expert MCP connection if it was initialized
        if ENABLE_DOCS_EXPERT and self._docs_expert is not None:
            try:
                from specialized_agents.docs_expert import get_docs_connection_pool
                pool = get_docs_connection_pool()
                await pool.close()
            except Exception:
                pass
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return self._budget_tracker.get_status()

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List saved sessions."""
        sessions = self._persistence.list_sessions()

        if status:
            sessions = [s for s in sessions if s.get('status') == status.value]

        return sessions[:limit]


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_orchestrator: Optional[BlenderVFXOrchestrator] = None


async def get_orchestrator() -> BlenderVFXOrchestrator:
    """Get the global orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BlenderVFXOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator


def is_orchestrator_initialized() -> bool:
    """
    Check if the orchestrator singleton is initialized WITHOUT creating it.

    Use this for status checks to avoid triggering initialization from within
    MCP tool handlers (which causes asyncio task group conflicts).
    """
    return _orchestrator is not None and _orchestrator.is_initialized


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_vfx_asset(
    asset_name: str,
    description: str,
    effect_type: str = "pyro",
    reference_path: Optional[str] = None,
    resolution: int = 96,
    frame_end: int = 50,
    quality_threshold: float = 60.0,
    max_iterations: int = 5,
) -> SessionState:
    """
    Convenience function to create a VFX asset.

    Args:
        asset_name: Name for the asset
        description: Description of what to create
        effect_type: Type of effect (pyro, explosion, fire, etc.)
        reference_path: Optional reference image path
        resolution: Blender simulation resolution
        frame_end: Animation end frame
        quality_threshold: Minimum quality score to pass
        max_iterations: Maximum iteration attempts

    Returns:
        SessionState with results
    """
    from .models.shared_context import EffectType

    request = AssetRequest(
        asset_name=asset_name,
        description=description,
        effect_type=EffectType(effect_type),
        reference_path=reference_path,
        resolution=resolution,
        frame_end=frame_end,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
    )

    orchestrator = await get_orchestrator()
    return await orchestrator.create_asset(request)


async def resume_vfx_session(session_id: str) -> SessionState:
    """
    Convenience function to resume a VFX session.

    Args:
        session_id: ID of session to resume

    Returns:
        Updated SessionState
    """
    orchestrator = await get_orchestrator()
    return await orchestrator.resume_session(session_id)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing BlenderVFXOrchestrator...")
        print("-" * 60)

        # Test initialization
        orchestrator = BlenderVFXOrchestrator()
        await orchestrator.initialize()
        print(f"Initialized: {orchestrator.is_initialized}")

        # Test budget status
        status = orchestrator.get_budget_status()
        print(f"Budget: ${status['total_spent']:.2f} / ${status['total_remaining'] + status['total_spent']:.2f}")

        # Test session listing
        sessions = orchestrator.list_sessions()
        print(f"Sessions: {len(sessions)}")

        await orchestrator.close()
        print("Closed orchestrator")

    asyncio.run(test())
