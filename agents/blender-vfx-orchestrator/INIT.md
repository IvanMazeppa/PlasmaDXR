# Blender VFX Orchestrator /INIT

## Purpose
- Autonomous, self-improving VFX asset generation via OpenAI Agents SDK (agents-as-tools).
- Blender 5.0 API is the only supported runtime target.

## Ground Truth
- Use OpenAI Agents SDK docs as the source of truth (context7 + OpenAI Developer Docs).
- Follow `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md` for model names and Blender API correctness.
- Do not rely on model memory for SDK or Blender API details.

## Non-Negotiables
- Agents-as-tools only; handoffs are deprecated.
- One outer `trace()` per pipeline run with `group_id=session_id`; do not nest traces.
- Tools run in-process only; no MCP subprocess calls inside tools.
- Two-layer tool pattern: `_impl()` + `@function_tool` wrapper.
- Script Writer must not write without a doc query in the same run (RunHooks enforced).
- Learning Agent proposals require Blender 5 doc_refs; new API use requires a micro-experiment first.
- API validation is strict: unknown attribute == invalid.

## Pipeline (Code-Based)
1. Research (structured `ResearchOutput` with `doc_refs`).
2. TechniqueSelector (Coordinator).
3. Iteration loop:
   - ModificationStrategist (Coordinator) for iteration 2+.
   - Learning Agent (doc-grounded proposals + micro-experiments).
   - Script Writer (doc query first).
   - API Validator (strict).
   - Executor.
   - Quality Analyst.
   - Learning Agent (record outcomes).
   - QualityGateJudge (Coordinator).
4. Shared SDK Session across all agents in the same VFX session.

## Known Bugs / Failure Modes (2026-01-24 to 2026-01-26 docs)
- Attribute hallucination: invalid Blender 5.0 properties used in scripts (e.g., `resolution_divisions`, `use_adaptive_time_steps`, `absolute_density`, `timesteps_per_frame`, `time_scale`).
- Validation defaulting to valid for unknown attributes (must be strict invalid).
- Script Writer sometimes skips doc tools; pipeline stops on DocQueryRequiredError.
- Doc search returning manual-only pages (API refs missing) can still happen if routing breaks.
- Learning Agent tries to `record_experiment_result` without `record_baseline`.
- Modification Coordinator proposes code fixes, but `_modify_script_impl` only edits `class Config` params, so code-pattern fixes are ignored.
- Quality model mismatch under quick_test if VISION_MODEL override is not honored.
- Schema mismatch risks (e.g., ScriptOutput `parameters_set` vs pipeline using `key_parameters`).

## TODOs (Priority-Ordered)
1. Enforce doc query before `write_script`/`modify_script` via RunHooks (must block non-doc tool proxies).
2. Add doc-coverage guardrail: every `bpy.*` attribute used must map to an API doc_ref.
3. Make API Validator strict (unknown attribute == invalid).
4. Insert APISpec stage: structured list of allowed attributes + doc refs; Script Writer can only use APISpec.
5. Fix modification pipeline: support code-pattern replacements or regenerate script when fix_type is API/code.
6. Ensure Learning Agent calls `record_baseline` before `record_experiment_result`.
7. Verify VISION_MODEL override used by Quality Analyst in quick_test preset.
8. Resolve schema mismatches in pipeline models (ScriptOutput, ExecutionOutput).
9. Add regression tests for known attribute errors and doc coverage enforcement.

## Model Truth (2026-01)
- Allowed: `gpt-5.2`, `gpt-5-mini`, `o3`, `o4-mini`.
- Disallowed: any `gpt-4*` or `gpt-3.5*`.

## SDK Compliance Checklist
- context7 query completed for relevant SDK feature.
- OpenAI Developer Docs search completed.
- Tracing enabled (single outer trace + group_id).
- Guardrails + RunHooks wired for all applicable agents.

## Resources
- **MCP Integration:** See `MCP_INTEGRATION_GUIDE.md` for setup instructions.
