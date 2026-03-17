# Specialized Section Builders Implementation Plan

Status: active plan
Last verified: 2026-03-17
Companion to:
- `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_DESIGN_NOTE.md`
- `docs/reviews/2026-03/20260316_SCENE_AESTHETIC_REALISM_ACTION_PLAN.md`
- `docs/CURRENT_ROADMAP.md`

## Purpose

This document turns the section-builder design note into an implementation sequence that can be executed against the current codebase.

The goal is to raise scene quality, hero-object quality, and reusability without destabilizing the existing pipeline. The plan explicitly reuses the current:

- seven canonical script sections in `utils/script_sections.py`
- section patching in `tools/script_generator_tools.py`
- `TechniqueContract` and `StyleSpec` in `models/pipeline_models.py`
- current orchestrator repair flow in `orchestrator.py`

This is not a rewrite plan. It is an incremental extension plan.

## Decision Summary

The recommended architecture is:

1. keep one top-level Director / planner for global coherence
2. formalize section ownership and section manifests
3. keep the current single Script Writer as the default baseline
4. introduce specialist builders one at a time, starting with `HeroAssetBuilder`
5. keep assembly deterministic
6. keep repair routing section-local
7. defer true parallel section generation until manifests and dependencies are stable

## What Success Looks Like

This work is successful when all of the following become true:

- hero objects stop defaulting to low-detail primitives
- scene environment and mood improve without breaking execution reliability
- section patching becomes more useful because builders own distinct sections
- the system can reuse strong assets or section modules across scenes
- at least one specialist builder improves production outputs without regressing execution stability

Concrete measurable outcomes:

- percentage of scripts where `create_geometry()` contains explicit refinement operators for hero objects
- percentage of scripts where `setup_scene()` / `setup_lighting()` contain meaningful world strategy and mood lighting
- percentage of failed evaluations that can be routed to a specific owned section
- quality delta between baseline single-writer runs and specialist-assisted runs on a fixed benchmark set
- percentage of repairs that patch one section instead of rewriting the whole script

## Non-Goals

This plan does not aim to:

- replace the current state machine
- make every section independently authored from day one
- allow uncontrolled multi-agent editing of shared Python
- force a rigid template workflow that removes LLM creativity
- optimize for shorter scripts

## Prerequisites and Gating

This work should be layered on top of the current roadmap state, not started in isolation.

Current assumptions from `docs/CURRENT_ROADMAP.md`:

- `TechniqueContract` is implemented
- `StyleSpec` is implemented
- section patching exists
- `modify_code` has already been exercised end to end
- aesthetic improvement work is already in progress

Practical gate before introducing the first specialist:

- section patching must be stable enough that a single bad specialist section does not force a full rewrite every time
- `structured_issues` must keep routing structural/aesthetic issues to `modify_code`
- the single-writer scaffold must be canonical and testable

## Recommended Rollout Order

1. Canonical scaffold and section ownership
2. `SceneBuildPlan` and `BuildRegistry`
3. `HeroAssetBuilder`
4. `EnvironmentAtmosphereBuilder`
5. `MaterialsLookDevBuilder`
6. Asset-lab mode
7. Optional DAG-based parallelism for independent builder stages

Do not start with 5 builders in parallel.

## Phase 0: Baseline, Flags, and Benchmarking

Goal:
- make specialist rollout measurable and reversible

### Tasks

1. Add feature flags for specialist rollout.
2. Add a benchmark set for specialist-vs-baseline comparisons.
3. Add deterministic script metrics for section richness and ownership compliance.

### File Targets

- `config/agent_config.py`
- `config/presets.yaml`
- `tools/script_analysis_tools.py`
- `tests/test_codex_rollout_config.py`
- `tests/test_script_sections.py`
- `docs/current/AESTHETIC_BENCHMARK_SET.md` or `tests/fixtures/aesthetic_benchmarks.json`

### Implementation Notes

Add rollout controls to the preset config:

```python
@dataclass
class PresetConfig:
    ...
    specialist_builders_enabled: bool = False
    specialist_builders_mode: str = "off"   # off | hero_only | hero_env | full
    asset_lab_enabled: bool = False
```

Add first-pass coverage metrics:

```python
@dataclass
class SectionRichnessMetrics:
    section_name: str
    line_count: int
    helper_count: int
    material_count: int
    modifier_count: int
    light_count: int
    created_object_count: int
    uses_dof: bool
    uses_world_strategy: bool
```

### Exit Criteria

- rollout can be turned on/off by preset
- a benchmark pack exists
- section richness can be measured before any specialist ships

## Phase 1: Formalize the Canonical Script Scaffold

Goal:
- make the current single-writer output structurally ready for section specialization

### Why First

If the scaffold is unstable, specialists will just multiply the repair problem.

### Tasks

1. Standardize the current seven canonical sections as mandatory for all production scripts.
2. Add lightweight section ownership markers or manifests.
3. Add a deterministic scaffold generator so all first-draft scripts start from the same frame.
4. Ensure the single Script Writer can still fill all sections as the fallback path.

### File Targets

- `utils/script_sections.py`
- `tools/script_generator_tools.py`
- `orchestrator.py`
- `guardrails/script_guardrails.py`
- `tests/test_script_sections.py`
- `tests/test_patch_tool.py`

### Implementation Notes

Add a scaffold helper:

```python
CANONICAL_SECTION_ORDER = [
    "setup_scene",
    "create_geometry",
    "setup_materials",
    "setup_physics",
    "setup_lighting",
    "setup_camera",
    "bake_and_render",
]


def render_base_scaffold() -> str:
    return """
import bpy
import math

def setup_scene(ctx):
    raise NotImplementedError

def create_geometry(ctx):
    raise NotImplementedError

def setup_materials(ctx):
    raise NotImplementedError

def setup_physics(ctx):
    raise NotImplementedError

def setup_lighting(ctx):
    raise NotImplementedError

def setup_camera(ctx):
    raise NotImplementedError

def bake_and_render(ctx):
    raise NotImplementedError

ctx = {}
setup_scene(ctx)
create_geometry(ctx)
setup_materials(ctx)
setup_physics(ctx)
setup_lighting(ctx)
setup_camera(ctx)
bake_and_render(ctx)
"""
```

Add section markers that can later be parsed cheaply:

```python
def create_geometry(ctx):
    # SECTION_OWNER: hero_asset_builder
    # CREATES: hero_glass, table_surface, support_props
    ...
```

### Tests

- verify scaffold always contains all seven sections
- verify `patch_script_section()` still works against scaffold-first scripts
- verify missing section output is rejected before execution

### Exit Criteria

- all newly generated scripts use the canonical scaffold
- all scripts can be statically checked for required sections
- single-writer fallback remains intact

## Phase 2: Add Shared Planning Artifacts

Goal:
- introduce typed section planning before code generation

### Tasks

1. Add `SceneBuildPlan`, `SectionAssignment`, `SectionBuildOutput`, and `BuildRegistry` models.
2. Add corresponding fields to session/context state.
3. Add a planner step that decides which sections are owned by which builder role.
4. Keep the planner advisory at first, while still using the single writer.

### File Targets

- `models/pipeline_models.py`
- `models/shared_context.py`
- `orchestrator.py`
- `tests/test_state_authority_wiring.py`
- new: `tests/test_section_build_plan.py`

### Implementation Notes

Add models:

```python
class SectionAssignment(BaseModel):
    builder_kind: Literal[
        "single_writer",
        "hero_asset",
        "environment_atmosphere",
        "materials_lookdev",
        "camera_composition",
        "scene_dressing",
    ]
    owned_section: Literal[
        "setup_scene",
        "create_geometry",
        "setup_materials",
        "setup_physics",
        "setup_lighting",
        "setup_camera",
        "bake_and_render",
    ]
    depends_on: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    must_create: list[str] = Field(default_factory=list)
    must_not_change: list[str] = Field(default_factory=list)


class SceneBuildPlan(BaseModel):
    scene_goal: str
    realism_obligations: list[str] = Field(default_factory=list)
    assignments: list[SectionAssignment] = Field(default_factory=list)


class SectionBuildOutput(BaseModel):
    section_name: str
    code: str
    created_objects: list[str] = Field(default_factory=list)
    created_materials: list[str] = Field(default_factory=list)
    registry_updates: dict[str, str] = Field(default_factory=dict)
```

Extend session/context:

```python
class SessionState(BaseModel):
    ...
    scene_build_plan: Optional[Dict[str, Any]] = None
    build_registry: Dict[str, Any] = Field(default_factory=dict)


class SharedContext(BaseModel):
    ...
    scene_build_plan: Optional[Dict[str, Any]] = None
    build_registry: Dict[str, Any] = Field(default_factory=dict)
```

### Orchestrator Integration

First rollout should plan, but still use the single writer:

```python
scene_build_plan = plan_scene_build(
    description=request.description,
    technique_contract=session.technique_contract,
    style_spec=session.style_spec,
)
session.scene_build_plan = scene_build_plan.model_dump()
context.scene_build_plan = scene_build_plan.model_dump()
```

### Tests

- planner produces valid assignments for baseline prompts
- no two builders own the same canonical section in the same plan
- planner output is serializable into session persistence

### Exit Criteria

- a `SceneBuildPlan` exists on every iteration 1 run
- single-writer scripts can be compared against planned ownership
- the plan can later drive specialist dispatch without schema churn

## Phase 3: Add Specialist Guardrails and Base Infrastructure

Goal:
- prepare specialist builders without changing production routing yet

### Tasks

1. Add a reusable base factory for section builders.
2. Add input/output guardrails that reject edits outside owned sections.
3. Add a deterministic section assembler.
4. Add builder result validation.

### File Targets

- new: `specialized_agents/section_builder_base.py`
- new: `guardrails/section_builder_guardrails.py`
- new: `utils/section_assembler.py`
- `specialized_agents/__init__.py`
- `tests/test_tool_guardrails.py`
- new: `tests/test_section_builder_guardrails.py`
- new: `tests/test_section_assembler.py`

### Implementation Notes

Base factory:

```python
def create_section_builder(
    name: str,
    instructions: str,
    owned_sections: list[str],
) -> Agent:
    return Agent(
        name=name,
        instructions=instructions,
        model=os.getenv("SCRIPT_WRITER_MODEL", "gpt-5.4"),
        tools=[
            semantic_search_blender_docs,
            search_blender_api_by_intent,
            search_code_patterns,
            get_pattern_code,
            patch_script_section,
        ],
        output_type=AgentOutputSchema(SectionBuildOutput, strict_json_schema=False),
        input_guardrails=[...],
        output_guardrails=[...],
    )
```

Output guardrail sketch:

```python
def validate_section_build_output(output: SectionBuildOutput, owned_sections: set[str]) -> None:
    if output.section_name not in owned_sections:
        raise ValueError(f"Builder returned unauthorized section: {output.section_name}")
```

Assembler:

```python
def assemble_sections(base_script: str, outputs: list[SectionBuildOutput]) -> str:
    source = base_script
    for out in outputs:
        source = replace_section(source, out.section_name, out.code)
    return source
```

### Tests

- builder cannot return code for an unowned section
- assembler preserves untouched sections
- duplicate section outputs are rejected

### Exit Criteria

- specialist outputs are type-checked and bounded
- assembly is deterministic
- specialist misbehavior fails early, before Blender execution

## Phase 4: HeroAssetBuilder MVP

Goal:
- improve hero object quality without touching the rest of the pipeline

### Why First

`create_geometry()` is currently one of the biggest visible quality bottlenecks. It is also easier to evaluate than whole-scene mood.

### Tasks

1. Add `HeroAssetBuilder`.
2. Restrict it to `create_geometry()`.
3. Feed it `StyleSpec`, `TechniqueContract`, and hero-object obligations.
4. Keep the rest of the script generated by the single writer.
5. Compare baseline vs specialist-assisted outputs on a fixed prompt pack.

### File Targets

- new: `specialized_agents/hero_asset_builder.py`
- `orchestrator.py`
- `tools/dynamic_instructions.py`
- `tools/code_pattern_tools.py`
- `tests/test_effect_type_scoping.py`
- new: `tests/test_hero_asset_builder.py`

### Implementation Notes

Specialist prompt shape:

```python
HERO_ASSET_BUILDER_INSTRUCTIONS = """
You own ONLY the create_geometry() section.

Your job:
- create the hero object(s) and support geometry required for physical credibility
- use refinement operators and modifiers where appropriate
- avoid low-detail primitive-only outputs for hero objects
- do not modify materials, lighting, camera, or render settings

Return a SectionBuildOutput for create_geometry only.
"""
```

Orchestrator integration:

```python
if config.preset.specialist_builders_enabled and config.preset.specialist_builders_mode in {"hero_only", "hero_env", "full"}:
    hero_output = await self._run_agent(
        self._hero_asset_builder,
        hero_prompt,
        context=context,
        session=iter_session,
        max_turns=8,
        run_config=self._build_run_config(..., phase="hero_asset_builder"),
    )
    assembled_source = assemble_sections(base_scaffold, [hero_output.final_output, single_writer_sections...])
```

Safer first version:

- keep the single writer generating the full script
- then replace only `create_geometry()` with specialist output

That way the specialist remains additive rather than foundational.

### Tests

- hero builder never touches non-geometry sections
- hero builder output compiles after insertion
- section patch fallback still works if hero builder fails
- benchmark prompts show better geometry richness signals than baseline

### Exit Criteria

- at least one benchmark family shows better hero geometry without worse execution reliability
- `create_geometry()` ownership is clean enough to patch independently

## Phase 5: EnvironmentAtmosphereBuilder

Goal:
- make scenes feel designed, not isolated

### Tasks

1. Add `EnvironmentAtmosphereBuilder`.
2. Let it own `setup_scene()` and optionally `setup_lighting()` in a controlled mode.
3. Add world strategy, background treatment, practicals, and atmospheric cues as explicit obligations.
4. Allow it to run in a second mode later for reusable environment recipes.

### File Targets

- new: `specialized_agents/environment_atmosphere_builder.py`
- `orchestrator.py`
- `utils/prompt_enhancer.py`
- `tools/dynamic_instructions.py`
- `tests/test_deterministic_quality_checks.py`
- new: `tests/test_environment_builder.py`

### Implementation Notes

Environment builder prompt shape:

```python
ENVIRONMENT_ATMOSPHERE_BUILDER_INSTRUCTIONS = """
You own setup_scene() and setup_lighting() only.

Your job:
- establish the world/background strategy
- place practical or cinematic lights that fit the StyleSpec
- add atmosphere, volume, and environmental context where appropriate
- support the hero object rather than compete with it

Do not modify geometry topology, materials, or camera.
"""
```

Recommended first ownership mode:

- `setup_scene()` only

Later expansion:

- `setup_scene()` + `setup_lighting()` together

This is safer than splitting scene/world and lighting too early, because mood and environment are tightly coupled.

### Tests

- environment builder produces explicit world strategy
- lighting mood is present without clobbering camera/render settings
- environment builder failure falls back cleanly to baseline writer

### Exit Criteria

- scenes stop reading as empty blockouts on at least part of the benchmark pack
- environment setup improves without a large execution regression

## Phase 6: MaterialsLookDevBuilder

Goal:
- improve material credibility once geometry and environment are stronger

### Why Third

High-quality materials depend on better geometry and better light. Running this before the earlier phases will create polished shaders on weak assets.

### Tasks

1. Add `MaterialsLookDevBuilder`.
2. Restrict it to `setup_materials()`.
3. Feed it the shared registry so it knows exact object names.
4. Expand QA issue kinds to route material/look-dev failures to the specialist.

### File Targets

- new: `specialized_agents/materials_lookdev_builder.py`
- `models/pipeline_models.py`
- `tools/qa_diagnosis_bridge.py`
- `orchestrator.py`
- `tests/test_repair_routing.py`
- new: `tests/test_materials_lookdev_builder.py`

### Implementation Notes

Recommended routing expansion:

```python
IssueKind = Literal[
    "parameter",
    "structural",
    "technique",
    "camera",
    "lighting",
    "materials",
    "environment",
    "composition",
    "hero_object",
    "lookdev",
]
```

Specialist routing:

```python
SPECIALIST_SECTION_BY_ISSUE = {
    "materials": "setup_materials",
    "lookdev": "setup_materials",
    "environment": "setup_scene",
    "lighting": "setup_lighting",
    "hero_object": "create_geometry",
}
```

### Tests

- material/look-dev issues route to `setup_materials()`
- material specialist reads registry entries rather than guessing object names
- section-local material patches preserve geometry and environment

### Exit Criteria

- hero materials get visibly better on benchmark prompts
- material fixes no longer require rewriting geometry or world setup

## Phase 7: Section-Local Repair Routing

Goal:
- make specialist ownership useful during failure recovery, not just first draft

### Tasks

1. Map `QualityIssue.kind` and `target` to the owning specialist and canonical section.
2. Patch only the relevant section on `modify_code`.
3. Fall back to the single writer only when section-local repair fails or the issue spans multiple sections.
4. Track which builder produced the faulty section for later evaluation.

### File Targets

- `orchestrator.py`
- `models/shared_context.py`
- `tests/test_repair_routing.py`
- `tests/test_structural_repair_regression.py`
- new: `tests/test_specialist_repair_routing.py`

### Implementation Notes

Add section ownership mapping to session state:

```python
class SessionState(BaseModel):
    ...
    section_owners: Dict[str, str] = Field(default_factory=dict)
```

Routing sketch:

```python
def resolve_repair_section(issue: QualityIssue, session: SessionState) -> Optional[str]:
    if issue.target and issue.target in session.section_owners:
        return issue.target
    return SPECIALIST_SECTION_BY_ISSUE.get(issue.kind)
```

Then inside the modify-code path:

```python
if repair_intent.mode == "modify_code":
    target_sections = {
        resolve_repair_section(issue, session)
        for issue in quality.structured_issues
    } - {None}
```

### Tests

- material defect patches only `setup_materials()`
- hero-object defect patches only `create_geometry()`
- environment defect patches only `setup_scene()` or `setup_lighting()`
- multi-section defects fall back honestly when needed

### Exit Criteria

- specialist-owned sections are genuinely easier to repair than the current baseline
- section-local repair succeeds on at least one real benchmark family

## Phase 8: Asset Lab Mode

Goal:
- convert strong specialist builders into reusable asset and environment generators

### Tasks

1. Add `asset_lab` mode to builder requests.
2. Add asset-specific quality obligations.
3. Save promoted assets and metadata into a reusable registry.
4. Keep asset generation separate from live scene iteration.

### File Targets

- `models/pipeline_models.py`
- `models/shared_context.py`
- new: `tools/asset_library_tools.py`
- new: `tests/test_asset_lab_mode.py`
- docs: `docs/current/ASSET_LIBRARY_POLICY.md`

### Implementation Notes

Add asset-lab request model:

```python
class AssetLabRequest(BaseModel):
    asset_type: str
    style_family: str = ""
    variant_count: int = 1
    quality_threshold: float = 70.0
    reusable_in_effect_types: list[str] = Field(default_factory=list)
```

Add saved asset metadata:

```python
class SavedAssetMetadata(BaseModel):
    asset_id: str
    asset_type: str
    builder_name: str
    script_fragment_path: str
    preview_render_path: Optional[str] = None
    quality_score: float = 0.0
    tags: list[str] = Field(default_factory=list)
```

Important policy:

- asset-lab mode should not pollute the production scene loop
- promotion should require stronger evidence than a one-off scene success

### Tests

- asset-lab outputs are saved with metadata
- failed assets are not promoted
- scene mode can consume a saved asset without changing the section ownership model

### Exit Criteria

- at least one reusable asset family exists
- scenes can consume a promoted asset instead of rebuilding it from scratch

## Phase 9: Optional Safe Parallelism

Goal:
- use parallelism only where dependencies are truly bounded

### Important Constraint

Do not start here. This is later work.

### Safe First Parallel Candidates

- design/planning passes that do not emit Python yet
- asset-variant generation in asset-lab mode
- independent environment recipe exploration

### Risky Early Parallel Candidates

- hero geometry and materials generated at the same time without a registry
- camera and environment generated before geometry anchors exist
- multiple builders all writing overlapping canonical sections

### Implementation Notes

Once manifests are mature, add a DAG scheduler:

```python
class BuildNode(BaseModel):
    node_id: str
    builder_kind: str
    section_name: str
    depends_on: list[str] = Field(default_factory=list)
```

Only nodes with satisfied dependencies should run in parallel.

### Exit Criteria

- no parallel branch can produce overlapping section ownership
- deterministic assembly remains the final authority

## Cross-Cutting Guardrails

These changes should land alongside the rollout, not afterward.

### Guardrail 1: Specialists must not escape their owned section

Implement in:

- `guardrails/section_builder_guardrails.py`
- `tests/test_section_builder_guardrails.py`

### Guardrail 2: Specialists must respect `TechniqueContract` and `StyleSpec`

Implement as:

- explicit prompt contract
- output validation against section purpose
- benchmark checks for style/technique adherence

### Guardrail 3: Single-writer fallback remains available

If any specialist fails:

- log the failure
- fall back to the baseline Script Writer
- do not corrupt the rest of the iteration state

### Guardrail 4: Specialist rollout must be feature-flagged

The user needs to be able to test:

- baseline
- hero-only
- hero+environment
- full specialist stack

## Testing Strategy

### Unit Tests

Add:

- `tests/test_section_build_plan.py`
- `tests/test_section_builder_guardrails.py`
- `tests/test_section_assembler.py`
- `tests/test_hero_asset_builder.py`
- `tests/test_environment_builder.py`
- `tests/test_materials_lookdev_builder.py`
- `tests/test_specialist_repair_routing.py`
- `tests/test_asset_lab_mode.py`

### Regression Tests

Extend:

- `tests/test_patch_tool.py`
- `tests/test_script_sections.py`
- `tests/test_repair_routing.py`
- `tests/test_structural_repair_regression.py`
- `tests/test_deterministic_quality_checks.py`

### Live Benchmarking

Run on a fixed set of prompts:

- baseline single writer
- hero-only specialist
- hero+environment specialists
- hero+environment+materials specialists

Record:

- execution success rate
- average quality score
- aesthetic issue distribution
- section-local repair rate
- script richness metrics

## Recommended First Milestone

The highest-value first milestone is:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4 only for `HeroAssetBuilder`

Do not implement environment or materials specialists before that milestone is proven.

Deliverable:

- the pipeline can produce a scaffolded script
- a `SceneBuildPlan` exists
- `HeroAssetBuilder` can replace `create_geometry()`
- the assembled script still executes
- hero objects improve on a small benchmark pack

## Recommended Job Order

If this work starts now, the job sequence should be:

1. add feature flags and metrics
2. add scaffold generator and section ownership markers
3. add planning models and session/context fields
4. add `SceneBuildPlan` planner
5. add section-builder base factory
6. add section-builder guardrails
7. add assembler helper
8. implement `HeroAssetBuilder`
9. wire hero-only rollout into orchestrator behind a flag
10. run benchmark comparison
11. only after that, start environment builder work

## What To Defer

Defer these until after the hero-only milestone:

- true parallel code generation
- more than two specialists
- asset-lab promotion logic
- environment and materials specialists in the same rollout
- major changes to the current state machine

## Recommendation

Proceed with a hero-first specialist rollout, not a full multi-builder launch. The implementation plan above gives you a way to:

- improve quality where the system is currently weakest
- preserve the existing patching investment
- keep a strong fallback path
- build toward reusable assets without overcommitting early

If the first specialist builder proves out, this architecture can become one of the most important quality-leverage changes in the whole system. If it does not, the rollout is still safe because the work up to that point improves scaffold quality, section ownership, and repairability even without many specialists.
