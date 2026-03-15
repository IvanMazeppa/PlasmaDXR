---
description: 
alwaysApply: true
---

# Blender VFX Orchestrator Guidelines

## Scope
- This file applies to `agents/blender-vfx-orchestrator/` only.
- Focus exclusively on the Blender 5.0 multi-agent VFX asset creation system.
- Do not modify unrelated DX12/C++ engine code outside this folder unless explicitly requested.

## Mission
- Build, stabilize, and operate the Blender 5.0 autonomous asset pipeline.
- Prioritize deterministic execution, strict API correctness, and reproducible outputs.

## Read First (Mandatory)
1. `agents/blender-vfx-orchestrator/docs/README.md`
2. `agents/blender-vfx-orchestrator/docs/CURRENT_STATE.md`
3. `agents/blender-vfx-orchestrator/docs/CURRENT_ROADMAP.md`
4. `agents/blender-vfx-orchestrator/docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
5. `agents/blender-vfx-orchestrator/docs/DOCS_AUTHORITY_AND_FRESHNESS_2026-02-13.md`
6. `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md`
7. `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md`
8. `agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md`

## Ground Truth Rules
- Blender runtime target is Blender 5.0 only.
- Treat Blender API memory as untrusted; verify `bpy.*` usage against Blender docs/tools.
- Unknown attributes/operators are invalid until verified.
- Keep OpenAI Agents SDK patterns aligned with project docs and current SDK references.
- Treat dated planning docs as historical unless explicitly referenced by runtime-truth docs.

## Required Orchestration Pattern
- Use code-based pipeline orchestration (`create_asset_pipeline`).
- Do not reintroduce deprecated handoff-based `create_asset()` flow.
- Required state machine:
  - `PLAN -> GENERATE -> VALIDATE -> EXECUTE -> EVALUATE -> DECIDE`
- Required failure routing:
  - `VALIDATE fail -> GENERATE`
  - `EXECUTE fail -> DIAGNOSE -> FIX -> EXECUTE`
  - `EVALUATE fail -> IMPROVE -> GENERATE`

## Non-Negotiables
- Agents-as-tools architecture only.
- One outer `trace()` per run with `group_id=session_id`; no nested traces.
- Keep RunHooks + guardrails enforced (do not bypass by direct internal calls).
- Script writing/modification requires doc-query evidence in the same run.
- Learning proposals must include Blender doc refs.
- Use artifact-first handoffs (script/log/render/cache paths), not inline dumps.
- Preserve deterministic bake/render gates (cache sanity, render count, fatal log checks).

## Blender 5.0 Headless Gotchas
- `bpy.ops.fluid.bake_all()` needs active domain object in `OBJECT` mode.
- Planar liquid inflows require `flow_settings.use_plane_init = True`.
- `cache_type='REPLAY'` is not acceptable for full bake outputs; use `ALL` when full cache is required.
- Headless still output should render representative frames directly; do not rely on `animation=True` for stills.
- Honor `BLENDER_OUTPUT_DIR` and `BLENDER_CACHE_DIR` for path consistency.

## Code Map
- `orchestrator.py`: pipeline and state-machine control.
- `specialized_agents/`: Script Writer, Executor, Quality Analyst, Learning Agent, Docs Expert, API validators/spec.
- `tools/`: execution, API fixer, docs search, pattern and experiment tooling.
- `guardrails/`: output/input/API/artifact gates.
- `hooks/`: RunHooks enforcement and diagnostics.
- `models/`: structured schemas and shared context.
- `config/`: presets and runtime policy.

## Development Commands
```bash
cd agents/blender-vfx-orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start MCP server
./run_server.sh

# Quick 2-iteration shakedown
./run_shakedown.sh fire

# Direct quick test
python test_quick_e2e.py --preset quick_test --effect smoke --iterations 2

# Focused tests
python -m pytest tests/test_artifact_gates.py -q
```

## Change Discipline
- Keep fixes minimal and local to the failing phase.
- When changing pipeline behavior, update guardrails/hooks/tests in the same change.
- Keep outputs reproducible and path-stable; prefer env-driven paths over hardcoded local paths.
- Do not commit generated runtime artifacts (`build/`, `renders/`, `traces/`, `sessions/`, `venv/`, cache outputs).
