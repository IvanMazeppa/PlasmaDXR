# Blender VFX Orchestrator Reboot Proposal

**Date:** 2026-02-18  
**Author:** Codex (GPT-5)  
**Scope:** Clean-slate architecture and implementation philosophy for an autonomous multi-agent Blender 5 asset generator using the OpenAI Agents SDK + OpenAI API.

## 1. Why Reboot (From Your Mission Draft)

Your mission statement implies a system that is:

- General across physics domains, not Mantaflow-only
- Adaptive and research-driven, not template-bound
- Honest about failures and cost
- Learned over time, with evidence-gated autonomy
- Human-collaborative with explicit intervention points

My interpretation: this is not a "better prompt" problem. It is a **systems architecture** problem where the LLM is one component inside a rigorously instrumented control loop.

## 2. Core Philosophy I Would Use

1. **Control plane vs execution plane split**
   - Control plane = reasoning, planning, policy, evaluation, memory updates.
   - Execution plane = deterministic tooling (Blender headless runs, validators, file ops, metrics).
   - Rationale: keep creativity where needed, determinism where required.

2. **Central orchestration first, decentralized only where earned**
   - Default to the SDK Manager pattern (agents-as-tools), not free-form peer handoffs.
   - Use handoffs only when explicit conversational ownership transfer is valuable.
   - Rationale: easier observability, cheaper debugging, tighter safety control.

3. **Typed artifacts between stages**
   - Every stage emits structured outputs (`output_type`/schema), never untyped prose.
   - Rationale: less ambiguity, lower regression risk, easier eval automation.

4. **Evidence over intuition for learning**
   - No knowledge promotion without repeated wins and defined confidence thresholds.
   - Rationale: avoids "self-reinforced wrongness".

5. **Fail loudly, route intentionally**
   - Error classes map to explicit next actions (retry/fix/escalate/abort), not generic retries.
   - Rationale: prevents infinite loops and token burn.

## 3. High-Level Architecture

## 3.1 Orchestration Pattern

Use a **Supervisor Agent** as the only component allowed to advance workflow state.

Specialists run as tools under supervisor control:

- `IntentPlannerAgent`
- `TechniqueResearchAgent`
- `SceneSpecAgent`
- `ScriptWriterAgent`
- `PreflightValidatorAgent`
- `ExecutionAgent` (tooling wrapper, deterministic first)
- `RenderCriticAgent`
- `ExperimentPolicyAgent`
- `KnowledgeCuratorAgent`
- `UserApprovalAgent` (HITL interaction policy)

Manager pattern keeps a stable audit trail and deterministic stage transitions.

## 3.2 State Machine (Mandatory)

Keep your required flow, but make each transition contract-based:

`PLAN -> GENERATE -> VALIDATE -> EXECUTE -> EVALUATE -> DECIDE`

With explicit branches:

- `VALIDATE -> IMPROVE_GENERATION`
- `EXECUTE -> DIAGNOSE_EXECUTION`
- `EVALUATE -> IMPROVE_TECHNIQUE`
- `DECIDE -> HITL_APPROVAL`
- `DECIDE -> COMPLETE`

Each state has:

- Entry criteria
- Max attempts / budget slice
- Output schema
- Allowed next states

No ad-hoc loops.

## 3.3 Data Contracts (Non-Optional)

Define versioned schemas for:

- `IntentSpec`
- `TechniquePlan`
- `SceneBlueprint`
- `BlenderScriptBundle`
- `ExecutionReport`
- `RenderScorecard`
- `ExperimentRecord`
- `KnowledgeCandidate`
- `DecisionRecord`

These become the only message payloads across agents/tools.

## 4. OpenAI Agents SDK Choices I Would Make

## 4.1 Core SDK Usage

- `Runner.run`/`run_streamed` for stage execution
- `Agent.as_tool` for specialist delegation (manager pattern)
- `handoff()` only for explicit ownership transfer cases
- `input_guardrails`/`output_guardrails`/tool guardrails where policy-critical
- `needs_approval` + `RunState` pause/resume for HITL on risky operations
- tracing enabled with `workflow_name`, `group_id`, metadata
- `RunConfig.call_model_input_filter` for context trimming/redaction

## 4.2 Context + Memory Strategy

Three-tier memory:

1. **Run memory (ephemeral):** per-iteration context
2. **Session memory (short-term):** via SDK `Sessions` for conversation continuity
3. **Knowledge memory (durable):** external store of validated learnings

Important discipline:

- Do not rely on long chat history as truth.
- Always reconstruct current decision context from durable artifacts + selected memory.
- If using SDK `session`, do not mix with `conversation_id`/`previous_response_id` on the same run.

## 4.3 Model Allocation Policy

- Supervisor / critical planning: `gpt-5.2` with explicit `model_settings`
- Cheap classification/triage steps: smaller model (e.g. `gpt-4.1` class)
- Judge/eval agent: strongest practical judge model available to you
- Latency-sensitive paths: lower reasoning effort where safe

Model choice is per-stage, not global.

## 5. Blender-Specific Execution Design

## 5.1 Deterministic Tool Layer

Implement deterministic tools for:

- Blender script execution (headless)
- Scene introspection (object graph, modifiers, caches)
- Simulation cache checks
- Render export checks
- Numeric sanity checks (NaN/extents/clipping)

LLM should call tools; tools produce machine-readable reports.

## 5.2 API Truth & Deprecation Containment

Given Blender API churn and deprecated code risk:

- Introduce a `BlenderCapabilityRegistry` with versioned feature flags
- Resolve API choices through adapters (not direct inline usage everywhere)
- Add a `TruthProbe` step that can query docs/MCP before uncertain API use
- Mark every generated script with target Blender version + capability set

This isolates deprecations to adapter boundaries instead of infecting orchestration logic.

## 5.3 Script Generation Policy

The writer agent must emit:

- One primary script
- One minimal diagnostic script
- One rollback/fallback script (reduced complexity)

Execution agent selects script variant by policy (normal vs recovery path).

## 6. Evaluation and Learning Flywheel

## 6.1 Multi-Layer Evaluation

Score each iteration with:

1. **Execution validity:** did it run, bake, render, produce artifacts?
2. **Physical plausibility proxies:** domain-specific checks (bounds, continuity, temporal behavior)
3. **Visual quality metrics:** image similarity / artifact checks / judge rubric
4. **Prompt-intent fit:** does output match requested effect semantics?

Use weighted score + hard fail gates (e.g. execution failure always blocks promotion).

## 6.2 Evals Program (Continuous)

Build a persistent eval suite across:

- Known effects
- Novel prompts
- Edge/adversarial prompts
- Multi-agent handoff/tool-selection correctness

Track regressions per architecture change and per policy change.

## 6.3 Knowledge Promotion Rules

Promote a `KnowledgeCandidate` only if:

- Repeated success on held-out prompts
- Improvement vs baseline is statistically meaningful
- No safety/cost regressions
- Human review passed for high-impact rule updates

Otherwise keep it as tentative memory.

## 7. Cost, Safety, and Honesty Controls

- Per-run budget envelope and per-stage spend caps
- Stop conditions for non-improving iterations
- Mandatory transparency record (`DecisionRecord`) per loop
- Approval gates for destructive/expensive actions
- No synthetic success claims; scorecards tied to artifact evidence only

## 8. Recommended Build Plan (From Scratch)

## Phase 0: Kernel (1-2 weeks)

- Define schemas + state machine + tracing conventions
- Build deterministic execution/report tools
- Build supervisor + one specialist path end-to-end

## Phase 1: P0 capability (2-4 weeks)

- Gas + liquid workflows
- Eval harness + baseline datasets
- HITL approvals for risky operations

## Phase 2: Generalization (4-8 weeks)

- Add rigid body + particles
- Technique research lane and fallback policies
- Knowledge promotion pipeline

## Phase 3: Autonomy progression

- Gradually reduce HITL based on measured reliability tiers
- Expand to multi-physics scenes only after per-domain stability targets are met

## 9. Decisions I Would Explicitly Make Up Front

1. **Use Manager pattern as default orchestration.**
2. **Require typed outputs for every stage transition.**
3. **Disallow ad-hoc retries; only policy-routed retries.**
4. **Separate durable knowledge from chat/session memory.**
5. **Treat Blender API compatibility as a first-class subsystem.**
6. **Run continuous evals before adding new physics domains.**

## 10. Practical Notes for Your Situation

- If deprecated code is widespread, do not patch it incrementally in place.
- Build a thin "new kernel" beside the old system and migrate lanes one by one.
- Keep legacy components behind adapters until replaced.
- Your mission statement is strong; the risk is implementation drift without contracts and hard gates.

## 11. References Used

- Mission draft you provided: `agents/blender-vfx-orchestrator/docs/MISSION_STATEMENT_DRAFT.md`
- OpenAI Agents SDK docs (official):
  - https://openai.github.io/openai-agents-python/
  - https://openai.github.io/openai-agents-python/agents/
  - https://openai.github.io/openai-agents-python/running_agents/
  - https://openai.github.io/openai-agents-python/tools/
  - https://openai.github.io/openai-agents-python/handoffs/
  - https://openai.github.io/openai-agents-python/guardrails/
  - https://openai.github.io/openai-agents-python/human_in_the_loop/
  - https://openai.github.io/openai-agents-python/sessions/
  - https://openai.github.io/openai-agents-python/tracing/
  - https://openai.github.io/openai-agents-python/models/
- OpenAI API docs (official):
  - https://platform.openai.com/docs/guides/conversation-state
  - https://platform.openai.com/docs/guides/background
  - https://platform.openai.com/docs/webhooks
  - https://platform.openai.com/docs/guides/evaluation-best-practices
  - https://platform.openai.com/docs/guides/reasoning-best-practices

