"""
Blender VFX Orchestrator Agent

Autonomous Claude Agent SDK agent that orchestrates the Blender VFX asset
generation pipeline with adaptive autonomy and token guardrails.
"""

import asyncio
import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

try:
    from .autonomy import (
        AutonomyConfig,
        AutonomyController,
        AutonomyLevel,
        DecisionType,
        TrustEvent,
    )
    from .guardrails import GuardrailAction, GuardrailConfig, TokenGuardrails
    from .state import AssetRequest, SessionManager, SessionState
    from .workflow import (
        QualityConfig,
        StageResult,
        WorkflowConfig,
        WorkflowOutcome,
        WorkflowStage,
        WorkflowStateMachine,
    )
except ImportError:
    from autonomy import (
        AutonomyConfig,
        AutonomyController,
        AutonomyLevel,
        DecisionType,
        TrustEvent,
    )
    from guardrails import GuardrailAction, GuardrailConfig, TokenGuardrails
    from state import AssetRequest, SessionManager, SessionState
    from workflow import (
        QualityConfig,
        StageResult,
        WorkflowConfig,
        WorkflowOutcome,
        WorkflowStage,
        WorkflowStateMachine,
    )

logger = logging.getLogger("blender-orchestrator")


class BlenderOrchestratorAgent:
    """
    Autonomous orchestrator for Blender VFX asset generation.

    Features:
    - Adaptive autonomy (trust score based)
    - Token usage guardrails
    - Session persistence for resumability
    - Integration with Blender pipeline MCP servers
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config_path: Path to config.yaml (defaults to module directory)
            project_root: Project root directory
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_path = config_path or Path(__file__).parent / "config.yaml"

        # Load configuration
        self.config = self._load_config()

        # Initialize configs
        self.workflow_config = WorkflowConfig.from_yaml(self.config)
        self.quality_config = QualityConfig.from_yaml(self.config)
        self.autonomy_config = AutonomyConfig.from_yaml(self.config)
        self.guardrail_config = GuardrailConfig.from_yaml(self.config)

        # State directory
        persistence = self.config.get("persistence", {})
        self.state_dir = self.project_root / persistence.get(
            "state_directory", "build/orchestrator_state/"
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Session manager
        self.session_manager = SessionManager(
            state_directory=self.state_dir,
            workflow_config=self.workflow_config,
            quality_config=self.quality_config,
            autonomy_config=self.autonomy_config,
            guardrail_config=self.guardrail_config,
            retention_days=persistence.get("session_retention_days", 30)
        )

        # Autonomy controller (shared across sessions)
        self.autonomy = self.session_manager.create_autonomy_controller()

        # Current session state
        self.session: Optional[SessionState] = None
        self.workflow: Optional[WorkflowStateMachine] = None
        self.guardrails: Optional[TokenGuardrails] = None

        # Claude SDK client
        self.client: Optional[ClaudeSDKClient] = None
        self.options: Optional[ClaudeAgentOptions] = None

        # Approval callback (for external integration)
        self.approval_callback: Optional[Callable[[str], str]] = None

        logger.info(f"BlenderOrchestratorAgent initialized")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  State: {self.state_dir}")
        logger.info(f"  Trust: {self.autonomy.get_trust_score():.2f}")
        logger.info(f"  Level: {self.autonomy.get_level().value}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _create_system_prompt(self) -> str:
        """Create system prompt for the orchestrator agent."""
        level = self.autonomy.get_level()
        trust = self.autonomy.get_trust_score()

        return f"""You are Blender VFX Orchestrator, an autonomous agent for generating
volumetric VFX assets using Blender Mantaflow simulation and the NanoVDB pipeline.

**Current Autonomy Status:**
- Trust Score: {trust:.2f}
- Autonomy Level: {level.value.upper()}
- Permissions: {self._format_permissions(level)}

**Your Workflow:**
1. GENERATE_SCRIPT - Create Blender Python scripts using script-generator
2. EXECUTE_BLENDER - Run simulations using blender-executor
3. EVALUATE_QUALITY - Assess VFX quality using asset-evaluator (VFX metrics, NOT LPIPS)
4. DECIDE_NEXT_ACTION - Based on scores and autonomy level:
   - Iterate with adjusted parameters
   - Change technique if plateaued
   - End session if quality passed or max iterations

**Quality Metrics (VFX-specific, NOT LPIPS/CLIP):**
- VFX Quality Score: 0-100 (pass threshold: {self.quality_config.vfx_pass_threshold})
- Plateau Detection: <{self.quality_config.plateau_threshold} change over {self.quality_config.plateau_iterations} iterations
- Temporal Consistency: ≥{self.quality_config.temporal_threshold}

**Token Guardrails:**
- Per Session: {self.guardrail_config.token_limits.per_session:,} tokens (${self.guardrail_config.cost_limits.per_session:.2f})
- Per Day: ${self.guardrail_config.cost_limits.per_day:.2f}
- At 80%: Warning | At 95%: Pause for approval | At 100%: Hard stop

**Available MCP Tools:**
- script-generator: generate_script, modify_script, list_techniques, validate_parameters
- blender-executor: execute_blender_script, parse_blender_errors, list_run_outputs
- asset-evaluator: evaluate_vfx_quality, compare_vfx_iterations, analyze_temporal_quality
- experiment-tracker: record_experiment_result, suggest_experiments, query_knowledge_base
- iteration-controller: diagnose_vfx_issues, get_next_iteration_params, save_iteration_state
- blender-manual: search_python_api, search_nodes, search_vdb_workflow

**Decision Framework:**
1. After each evaluation, check autonomy permissions for next action
2. If ALLOW: proceed autonomously with brief notification
3. If ASK: format approval request and await response
4. If STOP (guardrails): save state and end session gracefully

**Communication Style (Per CLAUDE.md):**
- Brutal honesty: "Score 42/100 - FAILING, needs major structure improvement"
- Specific: "Adding turbulence 0.3→0.6 to increase texture variation"
- Evidence-based: Reference actual scores, parameter values, technique names

**On Errors:**
- Blender execution failure: Query blender-manual, fix script, retry (max 3)
- Evaluation failure: Skip evaluation, ask for human guidance
- Critical failure: Save state, report error, end session

Remember: Your autonomy is earned. Good performance increases trust, failures decrease it.
Work efficiently but within your current permission level.
"""

    def _format_permissions(self, level: AutonomyLevel) -> str:
        """Format permission summary for current level."""
        perms = self.autonomy.PERMISSION_MATRIX.get(level, {})
        allowed = [d.value for d, p in perms.items() if p == "allow"]
        ask = [d.value for d, p in perms.items() if p == "ask"]

        result = []
        if allowed:
            result.append(f"Auto: {', '.join(allowed)}")
        if ask:
            result.append(f"Ask: {', '.join(ask)}")

        return " | ".join(result) if result else "All decisions require approval"

    def _create_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Create MCP server configuration for Blender pipeline."""
        servers = self.config.get("mcp_servers", {})

        mcp_config = {}

        for name, server_config in servers.items():
            if not server_config.get("enabled", True):
                continue

            cwd = self.project_root / server_config.get("cwd", f"agents/{name}")

            mcp_config[name] = {
                "type": "stdio",
                "command": "bash",
                "args": ["-c", f"cd {cwd} && {server_config.get('command', './run_server.sh')}"],
                "cwd": str(self.project_root),
                "env": {"PROJECT_ROOT": str(self.project_root)}
            }

        return mcp_config

    def _create_allowed_tools(self) -> List[str]:
        """Create list of allowed MCP tools."""
        return [
            # Script Generator
            "mcp__script-generator__list_templates",
            "mcp__script-generator__get_template",
            "mcp__script-generator__analyze_script",
            "mcp__script-generator__generate_script",
            "mcp__script-generator__modify_script",
            "mcp__script-generator__list_techniques",
            "mcp__script-generator__validate_parameters",
            "mcp__script-generator__get_parameter_ranges",

            # Blender Executor
            "mcp__blender-executor__execute_blender_script",
            "mcp__blender-executor__parse_blender_errors",
            "mcp__blender-executor__list_run_outputs",
            "mcp__blender-executor__get_latest_run",
            "mcp__blender-executor__list_available_scripts",

            # Asset Evaluator
            "mcp__asset-evaluator__evaluate_vfx_quality",
            "mcp__asset-evaluator__compare_vfx_iterations",
            "mcp__asset-evaluator__extract_vfx_diagnostics",
            "mcp__asset-evaluator__analyze_temporal_quality",
            "mcp__asset-evaluator__compare_lpips",
            "mcp__asset-evaluator__enhanced_evaluate",
            "mcp__asset-evaluator__evaluate_ground_truth",
            "mcp__asset-evaluator__get_alternative_approaches",

            # Experiment Tracker
            "mcp__experiment-tracker__start_experiment_session",
            "mcp__experiment-tracker__record_baseline",
            "mcp__experiment-tracker__record_experiment_result",
            "mcp__experiment-tracker__get_warnings_before_change",
            "mcp__experiment-tracker__suggest_experiments",
            "mcp__experiment-tracker__query_knowledge_base",
            "mcp__experiment-tracker__get_parameter_knowledge",
            "mcp__experiment-tracker__add_manual_learning",
            "mcp__experiment-tracker__get_experiment_statistics",

            # Iteration Controller
            "mcp__iteration-controller__diagnose_vfx_issues",
            "mcp__iteration-controller__get_next_iteration_params",
            "mcp__iteration-controller__save_iteration_state",
            "mcp__iteration-controller__load_iteration_state",
            "mcp__iteration-controller__list_orchestration_sessions",
            "mcp__iteration-controller__run_iteration",

            # Blender Manual
            "mcp__blender-manual__search_manual",
            "mcp__blender-manual__search_python_api",
            "mcp__blender-manual__search_nodes",
            "mcp__blender-manual__search_modifiers",
            "mcp__blender-manual__search_vdb_workflow",
            "mcp__blender-manual__search_tutorials",
            "mcp__blender-manual__read_page",
        ]

    def create_options(self) -> ClaudeAgentOptions:
        """Create ClaudeAgentOptions for the orchestrator."""
        logger.info("Creating orchestrator agent configuration...")

        options = ClaudeAgentOptions(
            cwd=str(self.project_root),
            mcp_servers=self._create_mcp_config(),
            allowed_tools=self._create_allowed_tools(),
            system_prompt=self._create_system_prompt(),
        )

        logger.info(f"MCP servers configured: {len(options.mcp_servers)}")
        logger.info(f"Allowed tools: {len(options.allowed_tools)}")

        return options

    async def start(self) -> None:
        """Initialize the orchestrator agent."""
        logger.info("Starting Blender VFX Orchestrator...")

        self.options = self.create_options()
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.__aenter__()

        logger.info("✅ Blender VFX Orchestrator ready")
        logger.info(f"   Trust: {self.autonomy.get_trust_score():.2f}")
        logger.info(f"   Level: {self.autonomy.get_level().value}")

    async def stop(self) -> None:
        """Shutdown the orchestrator agent."""
        # Save session state if active
        if self.session:
            self._save_current_state()

        if self.client:
            await self.client.__aexit__(None, None, None)

        logger.info("Blender VFX Orchestrator stopped")

    def _save_current_state(self) -> None:
        """Save current session state."""
        if self.session and self.workflow and self.guardrails:
            self.session_manager.update_session_from_workflow(self.session, self.workflow)
            self.session_manager.update_session_from_guardrails(self.session, self.guardrails)
            self.session_manager.save_session(self.session)

    async def create_asset(
        self,
        asset_name: str,
        effect_type: str,
        description: str,
        reference_path: Optional[str] = None,
        semantic_query: Optional[str] = None,
        resolution: int = 96,
        frame_end: int = 50,
        technique_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new asset generation session.

        Args:
            asset_name: Name for the asset
            effect_type: Effect type (pyro, explosion, fire, etc.)
            description: Description of what to create
            reference_path: Optional reference image path
            semantic_query: Optional text description for evaluation
            resolution: Blender simulation resolution
            frame_end: Animation end frame
            technique_name: Optional specific technique to use

        Returns:
            Session result summary
        """
        if not self.client:
            raise RuntimeError("Orchestrator not started. Call start() first.")

        # Create asset request
        request = AssetRequest(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            semantic_query=semantic_query,
            resolution=resolution,
            frame_end=frame_end,
            technique_name=technique_name
        )

        # Create session
        session_id = self.session_manager.create_session(request)
        self.session = self.session_manager.load_session(session_id)

        # Create workflow and guardrails
        self.workflow = self.session_manager.create_workflow(self.session)
        self.guardrails = self.session_manager.create_guardrails(session_id)

        logger.info(f"Starting asset generation: {asset_name}")
        logger.info(f"  Session: {session_id}")
        logger.info(f"  Effect: {effect_type}")
        logger.info(f"  Description: {description[:50]}...")

        # Run the workflow
        result = await self._run_workflow()

        return result

    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume an interrupted session.

        Args:
            session_id: Session ID to resume

        Returns:
            Session result summary
        """
        if not self.client:
            raise RuntimeError("Orchestrator not started. Call start() first.")

        self.session = self.session_manager.load_session(session_id)
        if not self.session:
            raise ValueError(f"Session not found: {session_id}")

        if self.session.status != "in_progress":
            raise ValueError(f"Session {session_id} is not resumable (status: {self.session.status})")

        # Restore workflow and create new guardrails
        self.workflow = self.session_manager.create_workflow(self.session)
        self.guardrails = self.session_manager.create_guardrails(session_id)

        logger.info(f"Resuming session: {session_id}")
        logger.info(f"  Iteration: {self.workflow.iteration}")
        logger.info(f"  Best Score: {self.workflow.best_score:.1f}")
        logger.info(f"  Stage: {self.workflow.current_stage.value}")

        # Continue the workflow
        result = await self._run_workflow()

        return result

    async def _run_workflow(self) -> Dict[str, Any]:
        """
        Run the main workflow loop.

        Returns:
            Final session result
        """
        try:
            while self.workflow.current_stage != WorkflowStage.SESSION_END:
                # Check guardrails before each stage
                can_proceed, message = self.guardrails.can_proceed()
                if not can_proceed:
                    logger.warning(f"Guardrail stop: {message}")
                    self.autonomy.record_event(TrustEvent.TOKEN_LIMIT_EXCEEDED)
                    break

                # Execute current stage
                stage = self.workflow.current_stage
                result = await self._execute_stage(stage)

                # Record token usage (estimate based on stage)
                self._record_stage_tokens(stage, result)

                # Advance workflow
                self.workflow.advance(result)

                # Save state after each stage
                self._save_current_state()

                # Log progress
                logger.info(self.workflow.format_progress())
                logger.info(self.guardrails.format_status_line())

            # Finalize session
            final_status = "completed" if self.workflow.check_quality_passed() else "failed"
            self.session_manager.finalize_session(self.session, final_status)

            # Record trust event
            if final_status == "completed":
                self.autonomy.record_event(TrustEvent.ASSET_COMPLETED)
            else:
                self.autonomy.record_event(TrustEvent.QUALITY_DEGRADED)

            return self._create_result_summary()

        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            self.autonomy.record_event(TrustEvent.CRITICAL_FAILURE)
            self.session_manager.finalize_session(self.session, "failed")
            raise

    async def _execute_stage(self, stage: WorkflowStage) -> StageResult:
        """
        Execute a workflow stage.

        Args:
            stage: Stage to execute

        Returns:
            Stage execution result
        """
        start_time = datetime.now()

        try:
            if stage == WorkflowStage.SESSION_START:
                result = await self._stage_session_start()

            elif stage == WorkflowStage.GENERATE_SCRIPT:
                result = await self._stage_generate_script()

            elif stage == WorkflowStage.EXECUTE_BLENDER:
                result = await self._stage_execute_blender()

            elif stage == WorkflowStage.EVALUATE_QUALITY:
                result = await self._stage_evaluate_quality()

            elif stage == WorkflowStage.DECIDE_NEXT_ACTION:
                result = await self._stage_decide_next_action()

            elif stage == WorkflowStage.RECORD_LEARNING:
                result = await self._stage_record_learning()

            elif stage == WorkflowStage.ERROR_RECOVERY:
                result = await self._stage_error_recovery()

            elif stage == WorkflowStage.AWAITING_APPROVAL:
                result = await self._stage_awaiting_approval()

            else:
                # SESSION_END handled by workflow loop
                result = StageResult(
                    stage=stage,
                    outcome=WorkflowOutcome.SUCCESS
                )

            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            return StageResult(
                stage=stage,
                outcome=WorkflowOutcome.FAILURE,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def _stage_session_start(self) -> StageResult:
        """Initialize session with experiment tracker."""
        logger.info("Stage: SESSION_START")

        # Build initialization prompt
        prompt = f"""Initialize asset generation session:

Asset: {self.session.request.asset_name}
Effect Type: {self.session.request.effect_type}
Description: {self.session.request.description}
Resolution: {self.session.request.resolution}
Frames: 1-{self.session.request.frame_end}

Actions:
1. Call experiment-tracker.start_experiment_session() to initialize tracking
2. Query knowledge base for known issues with this effect type
3. List available techniques for {self.session.request.effect_type}
4. Report ready status

Do NOT generate the script yet - just initialize and report what techniques are available.
"""

        response = await self._query_agent(prompt)

        return StageResult(
            stage=WorkflowStage.SESSION_START,
            outcome=WorkflowOutcome.SUCCESS,
            data={"initialization_response": response}
        )

    async def _stage_generate_script(self) -> StageResult:
        """Generate Blender Python script."""
        logger.info(f"Stage: GENERATE_SCRIPT (iteration {self.workflow.iteration + 1})")

        # Build generation prompt
        if self.workflow.iteration == 0:
            # First iteration
            prompt = f"""Generate initial Blender script for:

Asset: {self.session.request.asset_name}
Effect Type: {self.session.request.effect_type}
Description: {self.session.request.description}
Resolution: {self.session.request.resolution}
Frames: 1-{self.session.request.frame_end}
Technique: {self.session.request.technique_name or "auto-select based on description"}

Actions:
1. Use script-generator.generate_script() to create the Blender Python script
2. Validate parameters against documented ranges using validate_parameters()
3. Report the script path and key parameters
"""
        else:
            # Subsequent iterations
            prev_issues = self.session.iterations[-1].issues if self.session.iterations else []
            prompt = f"""Generate improved Blender script for iteration {self.workflow.iteration + 1}:

Previous Score: {self.workflow.scores[-1] if self.workflow.scores else 'N/A'}
Previous Issues: {', '.join(prev_issues) if prev_issues else 'None identified'}
Current Technique: {self.session.current_technique}
Current Parameters: {json.dumps(self.session.current_parameters, indent=2)}

Actions:
1. Query experiment-tracker for warnings before changing parameters
2. Use iteration-controller.get_next_iteration_params() to get adjusted parameters
3. Use script-generator.modify_script() to update the script
4. Validate parameters using validate_parameters()
5. Report changes made and rationale
"""

        response = await self._query_agent(prompt)

        # Try to extract script path from response
        script_path = self._extract_script_path(response)

        return StageResult(
            stage=WorkflowStage.GENERATE_SCRIPT,
            outcome=WorkflowOutcome.SUCCESS if script_path else WorkflowOutcome.FAILURE,
            data={
                "script_path": script_path,
                "response": response
            },
            error="Failed to extract script path" if not script_path else None
        )

    async def _stage_execute_blender(self) -> StageResult:
        """Execute Blender simulation."""
        logger.info("Stage: EXECUTE_BLENDER")

        script_path = self.session.current_script_path

        prompt = f"""Execute Blender simulation:

Script: {script_path}
Timeout: {self.workflow_config.blender_timeout} seconds

Actions:
1. Use blender-executor.execute_blender_script() to run the simulation
2. If errors occur, use parse_blender_errors() to diagnose
3. If execution fails, query blender-manual for fixes
4. Report success/failure with VDB output path or error details
"""

        response = await self._query_agent(prompt)

        # Check for success indicators
        success = "success" in response.lower() or "vdb" in response.lower()

        return StageResult(
            stage=WorkflowStage.EXECUTE_BLENDER,
            outcome=WorkflowOutcome.SUCCESS if success else WorkflowOutcome.FAILURE,
            data={"response": response}
        )

    async def _stage_evaluate_quality(self) -> StageResult:
        """Evaluate VFX quality."""
        logger.info("Stage: EVALUATE_QUALITY")

        prompt = f"""Evaluate VFX quality for iteration {self.workflow.iteration}:

Effect Type: {self.session.request.effect_type}
Target Score: {self.quality_config.get_pass_threshold(self.session.request.effect_type)}

Actions:
1. Find the latest render output using blender-executor.get_latest_run()
2. Use asset-evaluator.evaluate_vfx_quality() to assess quality (NOT LPIPS/CLIP!)
3. If animation, use analyze_temporal_quality() for temporal consistency
4. Use iteration-controller.diagnose_vfx_issues() to identify problems
5. Report:
   - VFX Quality Score (0-100)
   - Specific issues found
   - Recommended next steps
"""

        response = await self._query_agent(prompt)

        # Try to extract score from response
        score = self._extract_score(response)

        if score is not None:
            # Update trust based on quality change
            if len(self.workflow.scores) > 0:
                if score > self.workflow.scores[-1]:
                    self.autonomy.record_event(TrustEvent.QUALITY_IMPROVED)
                elif score < self.workflow.scores[-1]:
                    self.autonomy.record_event(TrustEvent.QUALITY_DEGRADED)

            if score >= self.quality_config.get_pass_threshold(self.session.request.effect_type):
                self.autonomy.record_event(TrustEvent.QUALITY_THRESHOLD_MET)

        return StageResult(
            stage=WorkflowStage.EVALUATE_QUALITY,
            outcome=WorkflowOutcome.SUCCESS if score is not None else WorkflowOutcome.SKIP,
            data={
                "score": score,
                "response": response
            }
        )

    async def _stage_decide_next_action(self) -> StageResult:
        """Decide next action based on quality and autonomy level."""
        logger.info("Stage: DECIDE_NEXT_ACTION")

        # Determine decision type
        if self.workflow.check_quality_passed():
            if self.workflow.check_quality_excellent():
                decision_type = DecisionType.END_SESSION_PASS
                action = "end_excellent"
            else:
                decision_type = DecisionType.END_SESSION_PASS
                action = "end_passed"
        elif self.workflow.check_iteration_limit():
            decision_type = DecisionType.END_SESSION_FAIL
            action = "end_max_iterations"
        elif self.workflow.check_plateau():
            decision_type = DecisionType.CHANGE_TECHNIQUE
            action = "change_technique"
            self.workflow.record_technique_change()
            self.autonomy.record_event(TrustEvent.TECHNIQUE_CHANGE)
        else:
            decision_type = DecisionType.ITERATE
            action = "iterate"

        # Check autonomy permission
        permission = self.autonomy.can_proceed(decision_type)

        if permission == "allow":
            # Proceed autonomously
            logger.info(f"Decision: {action} (autonomous)")
            return self._handle_decision(action)

        elif permission == "notify":
            # Proceed but notify
            logger.info(f"Decision: {action} (notify)")
            # Log notification
            self._notify_decision(decision_type, action)
            return self._handle_decision(action)

        elif permission == "ask":
            # Need approval
            logger.info(f"Decision: {action} (requesting approval)")

            # Set pending approval
            self.workflow.set_pending_approval(
                decision={
                    "type": decision_type.value,
                    "action": action,
                    "current_score": self.workflow.scores[-1] if self.workflow.scores else None,
                    "best_score": self.workflow.best_score,
                    "iteration": self.workflow.iteration,
                },
                resume_stage=self._get_resume_stage(action)
            )

            return StageResult(
                stage=WorkflowStage.DECIDE_NEXT_ACTION,
                outcome=WorkflowOutcome.AWAITING_INPUT,
                data={"decision": action, "permission": "ask"}
            )

        else:
            # Denied
            return StageResult(
                stage=WorkflowStage.DECIDE_NEXT_ACTION,
                outcome=WorkflowOutcome.FAILURE,
                error=f"Decision denied: {decision_type.value}"
            )

    def _handle_decision(self, action: str) -> StageResult:
        """Handle an approved decision."""
        if action in ["end_excellent", "end_passed"]:
            self.workflow.current_stage = WorkflowStage.SESSION_END
            return StageResult(
                stage=WorkflowStage.DECIDE_NEXT_ACTION,
                outcome=WorkflowOutcome.SUCCESS,
                data={"action": action, "final": True}
            )

        elif action == "end_max_iterations":
            self.workflow.current_stage = WorkflowStage.SESSION_END
            return StageResult(
                stage=WorkflowStage.DECIDE_NEXT_ACTION,
                outcome=WorkflowOutcome.SUCCESS,
                data={"action": action, "final": True}
            )

        else:
            # iterate or change_technique
            return StageResult(
                stage=WorkflowStage.DECIDE_NEXT_ACTION,
                outcome=WorkflowOutcome.SUCCESS,
                data={"action": action}
            )

    def _get_resume_stage(self, action: str) -> WorkflowStage:
        """Get the stage to resume after approval."""
        if action in ["end_excellent", "end_passed", "end_max_iterations"]:
            return WorkflowStage.SESSION_END
        else:
            return WorkflowStage.RECORD_LEARNING

    async def _stage_record_learning(self) -> StageResult:
        """Record learning to experiment tracker."""
        logger.info("Stage: RECORD_LEARNING")

        prompt = f"""Record experiment results for iteration {self.workflow.iteration}:

Score: {self.workflow.scores[-1] if self.workflow.scores else 'N/A'}
Technique: {self.session.current_technique}
Parameters: {json.dumps(self.session.current_parameters, indent=2)}

Actions:
1. Use experiment-tracker.record_experiment_result() to save this iteration
2. If quality improved, record the successful parameter changes
3. If quality degraded, record what NOT to do
4. Update knowledge base with any new learnings
"""

        response = await self._query_agent(prompt)

        return StageResult(
            stage=WorkflowStage.RECORD_LEARNING,
            outcome=WorkflowOutcome.SUCCESS,
            data={"response": response}
        )

    async def _stage_error_recovery(self) -> StageResult:
        """Handle errors and attempt recovery."""
        logger.info("Stage: ERROR_RECOVERY")

        # Get last error from history
        last_result = self.workflow.history[-1] if self.workflow.history else None
        error = last_result.error if last_result else "Unknown error"

        if self.workflow.check_retry_limit():
            # Too many retries
            return StageResult(
                stage=WorkflowStage.ERROR_RECOVERY,
                outcome=WorkflowOutcome.FAILURE,
                error=f"Retry limit exceeded: {error}"
            )

        prompt = f"""Error recovery needed:

Last Error: {error}
Retry Count: {self.workflow.retry_count}/{self.workflow_config.max_retries}
Failed Stage: {last_result.stage.value if last_result else 'unknown'}

Actions:
1. Analyze the error message
2. Query blender-manual for relevant documentation
3. Suggest fix and prepare for retry
4. If unrecoverable, recommend aborting

Permission to retry: {self.autonomy.can_proceed(DecisionType.RETRY_EXECUTION)}
"""

        response = await self._query_agent(prompt)

        # Record error event
        self.autonomy.record_event(TrustEvent.EXECUTION_ERROR)

        return StageResult(
            stage=WorkflowStage.ERROR_RECOVERY,
            outcome=WorkflowOutcome.SUCCESS,  # Will retry
            data={"recovery_response": response}
        )

    async def _stage_awaiting_approval(self) -> StageResult:
        """Wait for human approval."""
        logger.info("Stage: AWAITING_APPROVAL")

        decision = self.workflow.pending_decision

        # Format approval request
        request = self.autonomy.format_approval_request(
            decision=DecisionType(decision["type"]),
            session_id=self.session.session_id,
            iteration=self.workflow.iteration,
            context={
                **decision,
                "tokens_used": self.guardrails.session.total_tokens,
                "tokens_limit": self.guardrail_config.token_limits.per_session,
                "max_iterations": self.workflow_config.max_iterations,
            }
        )

        # Get approval (either via callback or interactive)
        if self.approval_callback:
            response = self.approval_callback(request)
        else:
            print(request)
            response = input().strip()

        # Parse response
        if response in ["1", "approve", "yes", "y"]:
            self.autonomy.record_event(TrustEvent.HUMAN_APPROVAL)
            return StageResult(
                stage=WorkflowStage.AWAITING_APPROVAL,
                outcome=WorkflowOutcome.SUCCESS,
                data={"approved": True, "response": response}
            )

        elif response in ["4", "abort", "no", "n"]:
            self.workflow.current_stage = WorkflowStage.SESSION_END
            return StageResult(
                stage=WorkflowStage.AWAITING_APPROVAL,
                outcome=WorkflowOutcome.SUCCESS,
                data={"approved": False, "aborted": True}
            )

        else:
            # Modify or override - record as override
            self.autonomy.record_event(TrustEvent.HUMAN_OVERRIDE)
            return StageResult(
                stage=WorkflowStage.AWAITING_APPROVAL,
                outcome=WorkflowOutcome.SUCCESS,
                data={"approved": True, "modified": True, "response": response}
            )

    async def _query_agent(self, prompt: str) -> str:
        """Send query to Claude agent and get response."""
        if not self.client:
            raise RuntimeError("Agent not started")

        logger.debug(f"Query: {prompt[:100]}...")

        await self.client.query(prompt)

        full_response = ""
        async for message in self.client.receive_response():
            message_text = str(message) if not isinstance(message, str) else message
            full_response += message_text

        logger.debug(f"Response: {full_response[:200]}...")

        return full_response

    def _notify_decision(self, decision_type: DecisionType, action: str) -> None:
        """Log notification for decision made autonomously."""
        status = self.workflow.format_progress()
        logger.info(f"[AUTONOMOUS] {decision_type.value}: {action}")
        logger.info(f"  {status}")

    def _extract_script_path(self, response: str) -> Optional[str]:
        """Extract script path from agent response."""
        # Look for common patterns
        import re

        patterns = [
            r"Script path:\s*([^\s]+\.py)",
            r"Generated:\s*([^\s]+\.py)",
            r"saved to\s*([^\s]+\.py)",
            r"([^\s]+blender_scripts[^\s]+\.py)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                path = match.group(1)
                self.session.current_script_path = path
                return path

        return None

    def _extract_score(self, response: str) -> Optional[float]:
        """Extract VFX quality score from agent response."""
        import re

        patterns = [
            r"VFX Quality Score:\s*(\d+(?:\.\d+)?)",
            r"Quality Score:\s*(\d+(?:\.\d+)?)",
            r"Score:\s*(\d+(?:\.\d+)?)/100",
            r"overall_score[\"']?\s*:\s*(\d+(?:\.\d+)?)",
            r"composite_score[\"']?\s*:\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def _record_stage_tokens(self, stage: WorkflowStage, result: StageResult) -> None:
        """Record estimated token usage for a stage."""
        # Estimate tokens based on stage
        estimates = {
            WorkflowStage.SESSION_START: (500, 1000),
            WorkflowStage.GENERATE_SCRIPT: (800, 2000),
            WorkflowStage.EXECUTE_BLENDER: (400, 1500),
            WorkflowStage.EVALUATE_QUALITY: (600, 1800),
            WorkflowStage.DECIDE_NEXT_ACTION: (300, 800),
            WorkflowStage.RECORD_LEARNING: (400, 1000),
            WorkflowStage.ERROR_RECOVERY: (600, 1500),
            WorkflowStage.AWAITING_APPROVAL: (100, 200),
            WorkflowStage.SESSION_END: (100, 200),
        }

        input_est, output_est = estimates.get(stage, (300, 500))
        result.tokens_used = input_est + output_est

        self.guardrails.record_usage(
            input_tokens=input_est,
            output_tokens=output_est,
            operation=stage.value
        )

    def _create_result_summary(self) -> Dict[str, Any]:
        """Create final result summary."""
        return {
            "session_id": self.session.session_id,
            "asset_name": self.session.request.asset_name,
            "status": self.session.status,
            "best_score": self.workflow.best_score,
            "best_iteration": self.workflow.best_iteration,
            "total_iterations": self.workflow.iteration,
            "technique_changes": self.workflow.technique_changes,
            "quality_passed": self.workflow.check_quality_passed(),
            "tokens_used": self.guardrails.session.total_tokens,
            "cost_usd": self.guardrails.session.cost_usd,
            "trust_score": self.autonomy.get_trust_score(),
            "autonomy_level": self.autonomy.get_level().value,
            "best_render_path": self.session.best_render_path,
            "best_vdb_path": self.session.best_vdb_path,
        }

    # Public API methods

    def set_trust_score(self, score: float) -> None:
        """Manually set trust score."""
        self.autonomy.set_trust_score(score)

    def set_autonomy_override(self, level: Optional[str]) -> None:
        """Set or clear autonomy level override."""
        self.autonomy.set_override(level)

    def get_status(self) -> Dict[str, Any]:
        """Get full orchestrator status."""
        status = {
            "autonomy": self.autonomy.get_status(),
            "session": None,
            "workflow": None,
            "guardrails": None,
        }

        if self.session:
            status["session"] = self.session.to_dict()

        if self.workflow:
            status["workflow"] = self.workflow.get_status()

        if self.guardrails:
            status["guardrails"] = self.guardrails.get_status()

        return status

    def list_sessions(
        self,
        status_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List available sessions."""
        return self.session_manager.list_sessions(status_filter, limit)

    def get_resumable_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions that can be resumed."""
        return self.session_manager.get_resumable_sessions()

    def set_approval_callback(self, callback: Callable[[str], str]) -> None:
        """Set callback for approval requests."""
        self.approval_callback = callback
