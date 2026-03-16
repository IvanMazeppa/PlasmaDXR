"""
Phase 2: Execution, Error Recovery, and Artifact Gates.

Handles Blender script execution and all recovery/validation paths:
- Phase 2: Deterministic execution via _execute_blender_script_impl
- Phase 2.5-2.6: Diagnosis and fix plan on execution failure
- Phase 2.7: AI-driven error recovery (Script Writer re-generates)
- Phase 2.7b: Artifact gates (deterministic bake/render validation)

Extracted from orchestrator.py during Phase 2C decomposition.
"""

from __future__ import annotations

import glob as _glob
import json as _json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hooks.enforcement_hooks import create_error_recovery_hooks
from models.pipeline_models import ExecutionOutput, QualityOutput, ScriptOutput
from tools.blender_executor_tools import _execute_blender_script_impl
from tools.truth_pack_validator import validate_and_fix_script
from guardrails.artifact_gates import (
    validate_execution_artifacts,
    format_gate_failure_for_diagnosis,
)

if TYPE_CHECKING:
    from agents.memory.session import SessionABC
    from models.shared_context import AssetRequest, SessionState, SharedContext
    from session_manager import SessionManager
    from utils import ArtifactManager


@dataclass
class ExecutionPhaseResult:
    """Result of Phase 2 execution. Caller checks should_continue."""
    execution: ExecutionOutput
    script: ScriptOutput  # May be replaced by recovery script
    quality: Optional[QualityOutput] = None  # Set if execution/gate failed
    should_continue: bool = False  # True = skip to next iteration
    previous_params: Dict[str, Any] = field(default_factory=dict)
    previous_score: float = 0.0


def _discover_render_current_run(run_dir: Optional[str]) -> Optional[str]:
    """Find render files produced by the current execution only.

    Searches ONLY the current execution's run_dir — never the shared
    per-asset output directory.  This prevents stale renders from
    previous iterations being mistaken for current output.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    for ext in ("*.png", "*.exr"):
        found = sorted(_glob.glob(os.path.join(run_dir, "**", ext), recursive=True))
        if found:
            return found[-1]
    return None


def _parse_exec_result(exec_json_str: str) -> ExecutionOutput:
    """Parse executor JSON into ExecutionOutput."""
    exec_data = _json.loads(exec_json_str)

    render_path = None
    if exec_data.get("render_files"):
        render_path = exec_data["render_files"][0]
    vdb_path = None
    if exec_data.get("vdb_files"):
        vdb_path = os.path.dirname(exec_data["vdb_files"][0])
    error_msg = None
    if not exec_data.get("success", False):
        errors = exec_data.get("errors", [])
        if errors:
            error_msg = "; ".join(
                e.get("message", str(e))[:200] for e in errors[:3]
            )
        else:
            error_msg = exec_data.get("error", "Unknown execution error")

    return ExecutionOutput(
        success=exec_data.get("success", False),
        render_path=render_path,
        vdb_path=vdb_path,
        run_dir=exec_data.get("run_dir"),
        error_message=error_msg,
        execution_time_seconds=exec_data.get("duration_seconds", 0.0),
    ), exec_data


def _apply_render_discovery(execution: ExecutionOutput, exec_output_dir: str, label: str = "") -> None:
    """Apply render discovery scoped to the current run only.

    Never promotes a failed execution to success — stale renders from
    previous iterations must not corrupt the feedback loop.
    """
    current_render = _discover_render_current_run(execution.run_dir)
    prefix = f"[Pipeline] {label} " if label else "[Pipeline] "

    if execution.success:
        # Successful run missing render_path — check run_dir
        if not execution.render_path and current_render:
            execution.render_path = current_render
            print(f"{prefix}Render discovered in run_dir: {current_render}", file=sys.stderr)
        return

    # Failed execution — store as diagnostic only, never promote to success
    if current_render:
        execution.partial_render_path = current_render
        print(
            f"{prefix}DIAGNOSTIC: Failed execution produced partial render at "
            f"{current_render}. Keeping success=False — will not evaluate stale render.",
            file=sys.stderr,
        )


async def run_execution_phase(
    orch: Any,
    request: "AssetRequest",
    context: "SharedContext",
    session: "SessionState",
    sdk_session: Optional["SessionABC"],
    iter_session: Optional[Any],
    artifact_mgr: "ArtifactManager",
    session_mgr: "SessionManager",
    script: ScriptOutput,
    iteration: int,
) -> ExecutionPhaseResult:
    """Execute Phase 2 + 2.5-2.7 + artifact gates.

    Args:
        orch: BlenderVFXOrchestrator instance
        request: Asset request
        context: Shared pipeline context
        session: Current session state
        sdk_session: SDK session for conversation persistence
        iter_session: Per-iteration SDK session (Phase 2B-1)
        artifact_mgr: Artifact manager
        session_mgr: Session manager for recording results
        script: Script to execute
        iteration: Current iteration number

    Returns:
        ExecutionPhaseResult with execution output, possibly updated script,
        and should_continue flag indicating whether to skip to next iteration.
    """
    # ====== PHASE 2: EXECUTION ======
    print(f"[Pipeline] PHASE 2: Deterministic Executor", file=sys.stderr)

    # Pre-execution truth pack validation (catches hallucinations the API
    # validator and API fixer miss, e.g. Color 1, Action.fcurves)
    from tracing import pipeline_event
    if context.truth_pack and script.script_path:
        try:
            _, fixes = validate_and_fix_script(
                script.script_path, context.truth_pack
            )
            if fixes:
                print(f"[Pipeline] Pre-exec truth pack fixed {len(fixes)} issues:",
                      file=sys.stderr)
                for fix in fixes[:5]:
                    print(f"  - {fix}", file=sys.stderr)
                pipeline_event("truth_pack_fix", {
                    "script_path": script.script_path,
                    "fixes_count": len(fixes),
                    "fixes": fixes[:10],
                    "iteration": iteration,
                })
        except Exception as e:
            print(f"[Pipeline] WARNING: Pre-exec truth pack validation failed: {e}",
                  file=sys.stderr)

    # phases/ → blender-vfx-orchestrator/ → agents/ → PlasmaDXR/
    _orchestrator_root = Path(__file__).resolve().parent.parent
    project_root = str(_orchestrator_root.parent.parent)
    exec_output_dir = os.path.join(project_root, "build", "vdb_output", request.asset_name)
    os.makedirs(exec_output_dir, exist_ok=True)

    try:
        exec_json_str = await _execute_blender_script_impl(
            script_path=script.script_path,
            script_args={"bake": "1"},
            output_dir=exec_output_dir,
            timeout_seconds=600,
        )
        execution, exec_data = _parse_exec_result(exec_json_str)
        print(
            f"[Pipeline] Deterministic exec: exit_code={exec_data.get('exit_code')} "
            f"success={execution.success} render={execution.render_path} "
            f"vdbs={len(exec_data.get('vdb_files', []))} "
            f"time={execution.execution_time_seconds:.1f}s",
            file=sys.stderr,
        )
        pipeline_event("execution_result", {
            "exit_code": exec_data.get("exit_code"),
            "success": execution.success,
            "has_render": execution.render_path is not None,
            "vdb_count": len(exec_data.get("vdb_files", [])),
            "execution_time": execution.execution_time_seconds,
            "iteration": iteration,
        })
    except Exception as e:
        print(f"[Pipeline] Executor error: {e}", file=sys.stderr)
        execution = ExecutionOutput(
            success=False,
            render_path=None,
            vdb_path=None,
            run_dir=None,
            error_message=f"Execution failed: {e}",
            execution_time_seconds=0.0,
        )

    # Fallback render discovery
    _apply_render_discovery(execution, exec_output_dir)

    if not execution.success or not execution.render_path:
        error_msg = execution.error_message or "Unknown execution error"
        print(f"[Pipeline] ERROR: Execution failed: {error_msg}", file=sys.stderr)

        # ====== PHASE 2.5: DIAGNOSE ======
        print(f"[Pipeline] PHASE 2.5: DIAGNOSE (execute failure)", file=sys.stderr)
        execute_diagnosis = (
            f"Execution failed for script {script.script_path or 'unknown'}.\n"
            f"Error: {error_msg}\n"
            "Failure class: execute_failure\n"
            "Use spec-first code fix path with verified APIs only."
        )
        execute_diag_artifact = artifact_mgr.write_diagnosis_from_values(
            iteration=iteration,
            phase="execute",
            issue_type="execute_failure",
            summary=error_msg,
            details=execute_diagnosis,
            recommended_fixes=[
                "Ensure domain object is active and selected before bake operators.",
                "Ensure Fluid modifier types are correct for domain/flow/effector objects.",
                "Verify operator context and depsgraph updates before bpy.ops calls.",
            ],
            script_path=script.script_path if script else None,
            render_path=execution.render_path,
        )
        context.last_execution_diagnosis = execute_diagnosis
        print(f"[Pipeline] Diagnosis artifact: {execute_diag_artifact}", file=sys.stderr)

        # ====== PHASE 2.6: FIX PLAN ======
        print(f"[Pipeline] PHASE 2.6: FIX (execute failure)", file=sys.stderr)
        execute_fix_steps = [
            "Regenerate/fix script via Script Writer with truth pack validation.",
            "Preserve verified API usage; do not introduce new unverified attributes.",
            "Re-run executor once with recovered script.",
        ]
        execute_fix_artifact = artifact_mgr.write_fix_from_values(
            iteration=iteration,
            phase="execute",
            strategy="script_recovery",
            instructions=execute_fix_steps,
            script_input_path=script.script_path if script else None,
            script_output_path=None,
            notes=[f"diagnosis_artifact={execute_diag_artifact}"],
        )
        print(f"[Pipeline] Fix artifact: {execute_fix_artifact}", file=sys.stderr)

        # ====== PHASE 2.7: ERROR RECOVERY ======
        recovery_succeeded = False

        if not getattr(context, '_recovery_attempted_this_iter', False):
            context._recovery_attempted_this_iter = True
            print(f"[Pipeline] PHASE 2.5: Error Recovery (AI-driven script fix)", file=sys.stderr)

            recovery_prompt = f"""CRITICAL: The Blender script FAILED to execute. You must fix the script.

## Error Message
{error_msg}

## Failed Script
Path: {script.script_path}

## What Went Wrong
This is a STRUCTURAL code issue, not a parameter issue. Do NOT use modify_script
(which only changes parameter values). Instead:
1. Read the failing script to understand its structure
2. Identify the root cause of the error
3. Use write_script to generate a FIXED version of the complete script

## Common Structural Fixes
- Bake context: bpy.ops.fluid.bake_data() requires the domain object to be
  bpy.context.view_layer.objects.active AND selected
- Effector setup: Collision objects need a Fluid modifier with fluid_type='EFFECTOR'
- Object context: bpy.ops calls require correct context (active object, selection)
- Missing depsgraph update: call bpy.context.view_layer.depsgraph.update() after setup

## Effect Type: {request.effect_type.value}
## Technique: {script.technique_used}

Fix the script and output it with write_script. Keep the same technique name
but append '_errfix' to the output name."""

            try:
                recovery_hooks = create_error_recovery_hooks()
                recovery_result = await orch._run_agent(
                    orch._script_agent_standalone,
                    recovery_prompt,
                    context=context,
                    session=iter_session,
                    hooks=recovery_hooks,
                    max_turns=6,
                    run_config=orch._build_run_config(
                        session=session,
                        request=request,
                        iteration=iteration,
                        phase="recover_script",
                    ),
                )
                recovery_script = recovery_result.final_output

                # Truth pack validation on recovered script
                if context.truth_pack and recovery_script and recovery_script.script_path:
                    try:
                        _, fixes = validate_and_fix_script(
                            recovery_script.script_path, context.truth_pack
                        )
                        if fixes:
                            print(f"[Recovery] Truth pack auto-fixed {len(fixes)} issues:",
                                  file=sys.stderr)
                            for fix in fixes[:5]:
                                print(f"  - {fix}", file=sys.stderr)
                    except Exception as e:
                        print(f"[Recovery] WARNING: Truth pack validation failed: {e}",
                              file=sys.stderr)

                if recovery_script and recovery_script.script_path:
                    print(f"[Pipeline] Recovery produced: {recovery_script.script_path}", file=sys.stderr)
                    artifact_mgr.write_fix_from_values(
                        iteration=iteration,
                        phase="execute",
                        strategy="script_recovery_applied",
                        instructions=execute_fix_steps,
                        script_input_path=script.script_path if script else None,
                        script_output_path=recovery_script.script_path,
                        notes=[f"recovery_source={execute_fix_artifact}"],
                    )

                    # Deterministic re-execution of recovered script
                    print(f"[Pipeline] PHASE 2.5b: Deterministic re-execution of recovered script", file=sys.stderr)
                    try:
                        reexec_json = await _execute_blender_script_impl(
                            script_path=recovery_script.script_path,
                            script_args={"bake": "1"},
                            output_dir=exec_output_dir,
                            timeout_seconds=600,
                        )
                        reexec_output, _ = _parse_exec_result(reexec_json)

                        # Fallback render discovery for recovery
                        _apply_render_discovery(reexec_output, exec_output_dir, label="Recovery")

                        if reexec_output.success and reexec_output.render_path:
                            print(f"[Pipeline] Recovery SUCCEEDED - render: {reexec_output.render_path}", file=sys.stderr)
                            execution = reexec_output
                            script = recovery_script
                            recovery_succeeded = True
                        else:
                            re_err = reexec_output.error_message or "No output"
                            print(f"[Pipeline] Recovery re-execution also failed: {re_err}", file=sys.stderr)
                    except Exception as e:
                        print(f"[Pipeline] Recovery re-execution error: {e}", file=sys.stderr)
                else:
                    print(f"[Pipeline] Recovery produced no script_path", file=sys.stderr)
            except Exception as e:
                print(f"[Pipeline] Error recovery failed: {e}", file=sys.stderr)

        # Reset recovery flag for next iteration
        context._recovery_attempted_this_iter = False

        if not recovery_succeeded:
            # Record failure and signal caller to continue loop
            quality = QualityOutput(
                overall_score=0,
                passed=False,
                primary_issue=f"Execution failed: {error_msg}",
                issues=[error_msg],
                suggestions=["Fix script errors and retry"]
            )
            current_params = script.parameters_set if hasattr(script, 'parameters_set') and script.parameters_set else {}
            session_mgr.record_result(
                iteration=iteration,
                params=current_params,
                score=0,
                issues=quality.issues,
                primary_issue=quality.primary_issue,
                technique=script.technique_used if script else None
            )
            print(f"[Pipeline] Execution failure recorded: consecutive_same_issue={session_mgr.issue_tracker.consecutive_same_issue}", file=sys.stderr)
            return ExecutionPhaseResult(
                execution=execution,
                script=script,
                quality=quality,
                should_continue=True,
                previous_params=current_params,
                previous_score=0,
            )

        # Recovery succeeded — fall through to artifact gates
        print(f"[Pipeline] Continuing to quality evaluation with recovered script", file=sys.stderr)

    print(f"[Pipeline] Render: {execution.render_path} ({execution.execution_time_seconds:.1f}s)", file=sys.stderr)

    # ====== PHASE 2.7: ARTIFACT GATES ======
    print(f"[Pipeline] PHASE 2.7: Artifact Gates", file=sys.stderr)

    artifact_output_dir = None
    if execution.render_path:
        render_parent = Path(execution.render_path).parent
        if render_parent.name == "renders":
            artifact_output_dir = str(render_parent.parent)
        else:
            artifact_output_dir = str(render_parent)

    executor_run_dir = execution.run_dir
    current_script_path = script.script_path if script and hasattr(script, 'script_path') else None
    gates_passed, gate_results, artifact_summary = validate_execution_artifacts(
        output_dir=artifact_output_dir,
        run_dir=executor_run_dir,
        effect_type=request.effect_type.value,
        verbose=True,
        script_path=current_script_path,
        technique=session.current_technique,
    )

    if not gates_passed:
        failure_diagnosis = format_gate_failure_for_diagnosis(gate_results, artifact_summary)
        print(f"[Pipeline] ARTIFACT GATE FAILED - skipping quality evaluation", file=sys.stderr)

        gate_issues = [g.reason for g in gate_results if not g.passed]
        print(f"[Pipeline] PHASE 2.8: DIAGNOSE (artifact gate failure)", file=sys.stderr)
        gate_diag_artifact = artifact_mgr.write_diagnosis_from_values(
            iteration=iteration,
            phase="artifact_gate",
            issue_type="artifact_gate_failure",
            summary=gate_issues[0] if gate_issues else "Artifact gate failure",
            details=failure_diagnosis,
            recommended_fixes=[
                "Verify cache outputs exist and exceed minimum size.",
                "Ensure at least one render file is produced in expected run/output directory.",
                "Check cache path wiring (BLENDER_CACHE_DIR / script output directories).",
            ],
            script_path=script.script_path if script else None,
            render_path=execution.render_path,
        )
        print(f"[Pipeline] Diagnosis artifact: {gate_diag_artifact}", file=sys.stderr)

        print(f"[Pipeline] PHASE 2.9: FIX (artifact gate failure)", file=sys.stderr)
        gate_fix_steps = [
            "Apply structural script updates based on gate diagnostics.",
            "Regenerate/modify script before next execute attempt.",
            "Re-validate cache and render artifacts after re-run.",
        ]
        gate_fix_artifact = artifact_mgr.write_fix_from_values(
            iteration=iteration,
            phase="artifact_gate",
            strategy="next_iteration_structural_fix",
            instructions=gate_fix_steps,
            script_input_path=script.script_path if script else None,
            script_output_path=None,
            notes=[f"diagnosis_artifact={gate_diag_artifact}"],
        )
        print(f"[Pipeline] Fix artifact: {gate_fix_artifact}", file=sys.stderr)

        quality = QualityOutput(
            overall_score=0,
            passed=False,
            primary_issue=f"Artifact gate failed: {gate_issues[0] if gate_issues else 'Unknown'}",
            issues=gate_issues,
            suggestions=[
                "Check simulation settings (e.g., use_plane_init for liquid emitters)",
                "Verify bake completed with non-empty cache",
                "Ensure render output path is valid",
            ],
            vision_assessment=f"No vision evaluation - artifact gates failed. Cache: {artifact_summary.cache_size_mb:.2f}MB, Renders: {artifact_summary.render_count}",
        )

        current_params = script.parameters_set if hasattr(script, 'parameters_set') and script.parameters_set else {}
        session_mgr.record_result(
            iteration=iteration,
            params=current_params,
            score=0,
            issues=quality.issues,
            primary_issue=quality.primary_issue,
            technique=script.technique_used if script else None
        )
        print(f"[Pipeline] Artifact gate failure recorded: {gate_issues}", file=sys.stderr)

        # Store gate diagnostics in context
        context.last_gate_failure = failure_diagnosis
        context.last_artifact_summary = artifact_summary
        context.pending_gate_fix_instructions = gate_fix_steps

        return ExecutionPhaseResult(
            execution=execution,
            script=script,
            quality=quality,
            should_continue=True,
            previous_params=current_params,
            previous_score=0,
        )

    print(f"[Pipeline] Artifact gates PASSED: cache={artifact_summary.cache_size_mb:.2f}MB, renders={artifact_summary.render_count}", file=sys.stderr)

    # Success — execution and gates passed, proceed to quality evaluation
    return ExecutionPhaseResult(
        execution=execution,
        script=script,
        quality=None,
        should_continue=False,
    )
