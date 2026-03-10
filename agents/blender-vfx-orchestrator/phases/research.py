"""
Phase 0: Research, Technique Selection, and Truth Pack Build.

Handles the initial pipeline phases that run once before the iteration loop:
- Phase 0: Research (parallel preflight with docs expert)
- Phase 0.5: Technique selection via coordinator agent
- Phase 0.6: Truth pack build (deterministic Blender introspection)

Extracted from orchestrator.py during Phase 2C decomposition.
"""

from __future__ import annotations

import asyncio
import sys
import time as _time
from typing import Any, Optional, TYPE_CHECKING

from agents.exceptions import OutputGuardrailTripwireTriggered

from hooks import EnforcementHooks, LoopDetectedError
from hooks.enforcement_hooks import create_research_hooks
from models.pipeline_models import ResearchOutput, TechniqueDecision
from tools.truth_pack import (
    build_truth_pack,
    truth_pack_to_api_spec,
)
from tools.truth_pack_validator import set_truth_pack as set_global_truth_pack
from guardrails.tool_guardrails import set_truth_pack as set_guardrail_truth_pack

if TYPE_CHECKING:
    from agents.memory.session import SessionABC
    from models.shared_context import AssetRequest, SessionState, SharedContext
    from utils import ArtifactManager


async def run_parallel_preflight(
    orch: Any,
    request: "AssetRequest",
    context: "SharedContext",
    sdk_session: Optional["SessionABC"],
    research_hooks: EnforcementHooks,
    run_config: Optional[Any],
) -> tuple[Optional[Any], Optional[str]]:
    """Run research + docs expert in parallel (fan-out/fan-in).

    This runs the Research Agent and Docs Expert concurrently to reduce
    latency at the start of the pipeline. The Docs Expert searches for high-risk
    API usage patterns that inform the script generation.

    Args:
        orch: BlenderVFXOrchestrator instance (for _run_agent, agent refs, config)
        request: Asset request with effect type and description
        context: Shared pipeline context
        sdk_session: SDK session for conversation persistence
        research_hooks: Enforcement hooks for research agent
        run_config: Optional RunConfig for model overrides

    Returns:
        Tuple of (research_result, docs_expert_notes) where docs_expert_notes
        is a string summary or None if disabled/failed.
    """
    preflight_start = _time.perf_counter()

    research_prompt = f"""Research the best approach for creating a {request.effect_type.value} VFX effect.

Effect Type: {request.effect_type.value}
Description: {request.description}
Semantic Query: {request.semantic_query or ""}
Reference: {request.reference_path or "None"}

First, identify which Blender physics system best fits this effect (rigid body, Mantaflow gas, Mantaflow liquid, particles, cloth, geometry nodes, or a combination). Use that as the domain parameter when searching docs.

Find at least one alternative technique beyond the obvious first choice."""

    # Use standalone docs expert (no handoffs) for parallel execution
    if not orch._enable_parallel_preflight or not orch._docs_expert_standalone:
        reason = "disabled" if not orch._enable_parallel_preflight else "no standalone docs expert"
        print(f"[Parallel Preflight] SEQUENTIAL mode ({reason})", file=sys.stderr)
        research_result = await orch._run_agent(
            orch._research_agent,
            research_prompt,
            context=context,
            session=sdk_session,
            hooks=research_hooks,
            max_turns=8,
            run_config=run_config,
        )
        elapsed = _time.perf_counter() - preflight_start
        print(f"[Parallel Preflight] Sequential research completed in {elapsed:.1f}s", file=sys.stderr)
        return research_result, None

    # PARALLEL MODE: Run research and docs expert concurrently
    print(f"[Parallel Preflight] PARALLEL mode enabled for {request.effect_type.value}", file=sys.stderr)

    docs_prompt = f"""Search Blender 5.0 docs for high-risk API usage for a {request.effect_type.value} effect.

Description: {request.description[:200]}

Identify which physics system this effect uses (rigid body, Mantaflow gas, Mantaflow liquid, particles, cloth, geometry nodes) and search for that specific domain.
Focus on: known Blender 5.0 API renames, bake operations, cache settings, and common pitfalls.
Return a concise bullet list with doc_refs."""

    print(f"[Parallel Preflight] Launching Research Agent + DocsExpert in parallel...", file=sys.stderr)
    research_task = orch._run_agent(
        orch._research_agent,
        research_prompt,
        context=context,
        session=sdk_session,
        hooks=research_hooks,
        max_turns=8,
        run_config=run_config,
    )
    docs_task = orch._run_agent(
        orch._docs_expert_standalone,  # Use standalone (no handoffs)
        docs_prompt,
        context=context,
        session=sdk_session,
        max_turns=3,
        run_config=run_config,
    )

    results = await asyncio.gather(research_task, docs_task, return_exceptions=True)
    elapsed = _time.perf_counter() - preflight_start

    # S0-4 FIX: Re-raise guardrail tripwire exceptions instead of swallowing them.
    from agents import OutputGuardrailTripwireTriggered as _OGTT
    for i, result in enumerate(results):
        if isinstance(result, _OGTT):
            phase_name = "Research" if i == 0 else "DocsExpert"
            print(f"[Parallel Preflight] {phase_name} guardrail tripwire — re-raising", file=sys.stderr)
            raise result

    # Extract results, handling non-tripwire exceptions gracefully
    research_result = results[0] if not isinstance(results[0], Exception) else None
    docs_result = results[1] if not isinstance(results[1], Exception) else None

    # Log results and any exceptions for debugging
    research_ok = not isinstance(results[0], Exception)
    docs_ok = not isinstance(results[1], Exception)

    if isinstance(results[0], Exception):
        print(f"[Parallel Preflight] Research FAILED: {results[0]}", file=sys.stderr)
    if isinstance(results[1], Exception):
        print(f"[Parallel Preflight] DocsExpert FAILED: {results[1]}", file=sys.stderr)

    docs_text = getattr(docs_result, "final_output", None) if docs_result else None
    docs_preview = docs_text[:80] + "..." if docs_text and len(docs_text) > 80 else docs_text

    print(f"[Parallel Preflight] Completed in {elapsed:.1f}s", file=sys.stderr)
    print(f"[Parallel Preflight]   Research: {'OK' if research_ok else 'FAILED'}", file=sys.stderr)
    print(f"[Parallel Preflight]   DocsExpert: {'OK' if docs_ok else 'FAILED'} | Notes: {docs_preview or 'None'}", file=sys.stderr)

    return research_result, docs_text


async def run_research_phase(
    orch: Any,
    request: "AssetRequest",
    context: "SharedContext",
    session: "SessionState",
    sdk_session: Optional["SessionABC"],
    artifact_mgr: "ArtifactManager",
    is_resuming: bool,
) -> tuple[str, Optional[TechniqueDecision]]:
    """Execute Phase 0 + 0.5 + 0.6: Research, technique selection, truth pack.

    Args:
        orch: BlenderVFXOrchestrator instance
        request: Asset request
        context: Shared pipeline context
        session: Current session state
        sdk_session: SDK session for conversation persistence
        artifact_mgr: Artifact manager for writing research artifacts
        is_resuming: Whether this is a resumed session

    Returns:
        Tuple of (research_text, selected_technique) where selected_technique
        may be None if resuming or if technique coordinator failed.
    """
    selected_technique: Optional[TechniqueDecision] = None
    research_text = ""

    if not is_resuming:
        # ====== PHASE 0: RESEARCH (once at start) ======
        print("[Pipeline] PHASE 0: Research", file=sys.stderr)
        phase0_research_hooks = create_research_hooks()
        research_output: Optional[ResearchOutput] = None
        try:
            run_config = orch._build_run_config(
                session=session,
                request=request,
                iteration=0,
                phase="research",
            )
            research_result, docs_notes = await run_parallel_preflight(
                orch=orch,
                request=request,
                context=context,
                sdk_session=sdk_session,
                research_hooks=phase0_research_hooks,
                run_config=run_config,
            )
            # Phase 3: Structured output - ResearchOutput schema
            research_output = research_result.final_output if research_result else None
            if research_output:
                # Build text summary from structured output for downstream use
                research_text = f"""## Research Summary
Recommended Approach: {research_output.recommended_approach}
Key Parameters: {research_output.key_parameters}
API Modules: {', '.join(research_output.api_modules) if research_output.api_modules else 'None'}
Warnings: {'; '.join(research_output.warnings) if research_output.warnings else 'None'}
Doc Refs: {', '.join(research_output.doc_refs) if research_output.doc_refs else 'None'}"""
                if docs_notes:
                    research_text += f"\n\n## Docs Expert Notes\n{docs_notes}"
            else:
                research_text = "No research findings"
        except LoopDetectedError as e:
            # Research got stuck - use partial results
            print(f"[Pipeline] WARN: Research loop detected: {e}", file=sys.stderr)
            research_text = "Research incomplete due to loop - proceeding with default approach"
        except OutputGuardrailTripwireTriggered as e:
            # Strict grounding mode: do not proceed when research is ungrounded.
            raise RuntimeError(
                f"Research grounding guardrail tripped; aborting run: {e}"
            ) from e
        except Exception as e:
            # SDK wraps LoopDetectedError in UserError - check for it
            if "Loop detected" in str(e) or "LoopDetectedError" in str(e):
                print(f"[Pipeline] WARN: Research loop detected (wrapped): {e}", file=sys.stderr)
                research_text = "Research incomplete due to loop - proceeding with default approach for this effect type"
            else:
                raise  # Re-raise other exceptions
        print(f"[Pipeline] Research complete: {research_text[:80]}...", file=sys.stderr)
        print(f"[Pipeline] Research hooks stats: {phase0_research_hooks.get_stats()}", file=sys.stderr)

        # Store research in session for persistence and handoff to Script Writer
        session.research_text = research_text

        # Phase 3: Use structured output directly instead of regex parsing
        if research_output and research_output.alternative_approaches:
            for approach in research_output.alternative_approaches:
                if approach and approach not in session.alternative_approaches:
                    session.alternative_approaches.append(approach)
            print(f"[Pipeline] Found {len(session.alternative_approaches)} alternative approaches (structured)", file=sys.stderr)

        # Artifact-First: Write research to file for downstream agents
        research_artifact_path = None
        if research_output:
            research_artifact_path = artifact_mgr.write_research_from_output(
                recommended_approach=research_output.recommended_approach,
                key_parameters=research_output.key_parameters,
                api_modules=research_output.api_modules,
                warnings=research_output.warnings,
                alternative_approaches=research_output.alternative_approaches,
                doc_refs=research_output.doc_refs,
            )
            print(f"[Pipeline] Research artifact: {research_artifact_path}", file=sys.stderr)

        # ====== PHASE 0.5: TECHNIQUE SELECTION ======
        print("[Pipeline] PHASE 0.5: Technique Selection", file=sys.stderr)

        research_ref = f"Research artifact: {research_artifact_path}" if research_artifact_path else "No research artifact"
        technique_prompt = f"""Select the best technique for creating a {request.effect_type.value} VFX effect.

## Research Summary
{research_ref}
Recommended: {research_output.recommended_approach if research_output else 'Default approach'}
Key params: {list(research_output.key_parameters.keys()) if research_output and research_output.key_parameters else 'None'}

## Effect Parameters
- Effect Type: {request.effect_type.value}
- Description: {request.description}
- Reference: {request.reference_path or "None"}
- Quality Threshold: {request.quality_threshold}

## Available Alternatives
{chr(10).join(f'- {a}' for a in session.alternative_approaches[:5]) if session.alternative_approaches else 'None identified'}

Select the optimal technique and provide starting parameters."""

        try:
            technique_result = await orch._run_agent(
                orch._technique_coordinator,
                technique_prompt,
                context=context,
                session=sdk_session,
                max_turns=6,
                run_config=orch._build_run_config(
                    session=session,
                    request=request,
                    iteration=0,
                    phase="technique_select",
                ),
            )
            selected_technique = technique_result.final_output
            print(f"[Pipeline] Coordinator selected: {selected_technique.selected_technique}", file=sys.stderr)
            print(f"[Pipeline] Reasoning: {selected_technique.reasoning[:60]}...", file=sys.stderr)

            session.current_technique = selected_technique.selected_technique
            if selected_technique.alternative_techniques:
                for alt in selected_technique.alternative_techniques:
                    if alt not in session.alternative_approaches:
                        session.alternative_approaches.append(alt)

        except Exception as e:
            print(f"[Pipeline] WARN: Technique Coordinator failed: {e}", file=sys.stderr)
            selected_technique = None

    else:
        # ====== RESUME: Skip Phase 0 and 0.5 ======
        print("[Pipeline] RESUME: Skipping Phase 0/0.5 (already stored in session)", file=sys.stderr)
        research_text = session.research_text or "Resumed session"
        # selected_technique stays None — not needed for iteration 2+

    # ====== PHASE 0.6: BUILD TRUTH PACK ======
    technique_for_tp = (
        session.current_technique
        or (selected_technique.selected_technique if selected_technique else None)
        or "mantaflow_gas"
    )
    try:
        print(f"[Pipeline] PHASE 0.6: Building truth pack for '{technique_for_tp}'...", file=sys.stderr)
        truth_pack = await build_truth_pack(technique_for_tp)
        context.truth_pack = truth_pack
        set_global_truth_pack(truth_pack)
        set_guardrail_truth_pack(truth_pack)  # Phase 2A-3: tool guardrails

        # Build APISpec from truth pack for backward compatibility
        context.api_spec = truth_pack_to_api_spec(
            truth_pack, request.effect_type.value, technique_for_tp
        )
        print(f"[Pipeline] Truth pack built: {len(truth_pack)} types, "
              f"APISpec populated with {len(context.api_spec.domain_attributes)} "
              f"domain attrs", file=sys.stderr)
    except Exception as e:
        print(f"[Pipeline] WARNING: Truth pack build failed: {e}. "
              f"Continuing without truth pack.", file=sys.stderr)
        # Don't block pipeline — Script Writer can still generate scripts

    return research_text, selected_technique
