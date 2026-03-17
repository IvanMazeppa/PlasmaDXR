# Specialized Section Builders Implementation Plan

Status: candidate final plan
Last verified: 2026-03-17
Companion to:
- `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_DESIGN_NOTE.md`
- `docs/reviews/2026-03/20260317_SECTION_BUILDERS_REVIEW_AND_REFINEMENTS.md`
- `docs/reviews/2026-03/20260316_SCENE_AESTHETIC_REALISM_ACTION_PLAN.md`
- `docs/CURRENT_ROADMAP.md`

## Purpose

This document converts the section-builder design into an implementation sequence that fits the current codebase and incorporates the valid refinements raised during review.

The goal is to improve:

- hero-object quality
- environment and mood
- material credibility
- section-local patchability
- reusable asset accumulation

without destabilizing the current pipeline or reintroducing prompt-level coordination chaos.

This is an extension plan, not a rewrite plan.

## Finalized Planning Decisions

The following decisions are treated as resolved before implementation starts:

1. `BuildRegistry` is typed from day one.
   No loose `ctx` dict.

2. The Director is deterministic for the initial rollout.
   `plan_scene_build()` should be a pure function, not a new LLM coordinator.

3. Specialist-owned sections use stubs.
   The single writer should not waste tokens generating code that will be replaced immediately.

4. Specialists get bundled doc grounding plus builder-level truth-pack validation.
   The builder boundary should be a validation boundary.

5. Scaffold generation derives from the existing canonical section list.
   `utils/script_sections.py` remains the source of truth.

6. Asset-lab promotion hooks are designed into the models now.
   Full asset-lab rollout is later, but the model shape should not need retrofitting.

7. Repair escalation is section-local.
   Same specialist first, then single-writer section fallback, then full-script fallback.

8. Parallelism is deferred and kept simple.
   Start with bounded `asyncio.gather()` groups only if and when the dependency structure proves stable.

## Why This Is the Right Shape

The current Script Writer is overloaded. It has to solve technique correctness, geometry, materials, environment, composition, and execution safety in one pass. The result is often conservative code: primitives, limited refinement, thin environment design, and low scene density.

Specialization can help, but only if it operates over a deterministic scaffold. The system should not move to "many agents all writing Python." It should move to:

- one deterministic planner
- one typed registry
- explicit section ownership
- section-local generation and repair
- deterministic assembly and validation

That is the shape that preserves both creativity and controllability.

## Success Criteria

This work is successful when:

- hero objects stop defaulting to low-detail primitives
- scenes feel intentionally lit and situated rather than isolated
- material/look-dev failures can be routed to `setup_materials()` reliably
- section patching becomes more useful because sections actually have owners
- at least one specialist improves outputs without worsening execution reliability
- at least one builder can later promote outputs toward reusable assets

Track:

- percentage of scripts where `create_geometry()` contains explicit refinement operators
- percentage of scripts where `setup_scene()` or `setup_lighting()` contain explicit world strategy and mood lighting
- percentage of failing QA outputs that resolve to a canonical section
- percentage of repairs that patch one section instead of rewriting the whole script
- benchmark delta between baseline and specialist rollouts
- percentage of assembled scripts that pass cross-section registry validation before Blender execution

## Non-Goals

This plan does not aim to:

- replace the current state machine
- introduce multiple free-form parallel coders
- force template-only generation
- optimize for short scripts
- introduce asset-library-first composition before specialist builders are proven

## Fit With the Current Roadmap

This plan assumes the current roadmap state is broadly true:

- `TechniqueContract` exists
- `StyleSpec` exists
- section patching exists
- `modify_code` exists and has been exercised
- aesthetic realism work is already underway

This specialist-builder plan should begin only after:

- section patching is stable enough that a bad specialist section does not force a whole-script rewrite every time
- `structured_issues` continue routing structural/aesthetic failures to `modify_code`
- the canonical scaffold is made reliable

## Rollout Order

1. Phase 0: baseline metrics and rollout flags
2. Phase 1: canonical scaffold + typed registry + section stubs
3. Phase 2: deterministic `SceneBuildPlan` + section models
4. Phase 3: specialist base infrastructure + builder-level validation
5. Phase 4: `HeroAssetBuilder` MVP
6. Phase 5: `EnvironmentAtmosphereBuilder`
7. Phase 6: `MaterialsLookDevBuilder`
8. Phase 7: section-local repair escalation
9. Phase 8: asset-lab mode
10. Phase 9: optional bounded parallelism

Do not start with more than one specialist builder.

## Phase 0: Baseline, Flags, and Benchmarking

Goal:
- make the rollout measurable, comparable, and reversible

### Tasks

1. Add feature flags for specialist rollout.
2. Add a benchmark set for baseline vs specialist comparisons.
3. Add deterministic section richness metrics.
4. Add builder identity/version tracking so future A/B comparisons are possible.

### File Targets

- `config/agent_config.py`
- `config/presets.yaml`
- `tools/script_analysis_tools.py`
- `tests/test_codex_rollout_config.py`
- `tests/test_script_sections.py`
- `docs/current/AESTHETIC_BENCHMARK_SET.md` or `tests/fixtures/aesthetic_benchmarks.json`

### Implementation Notes

Add rollout controls:

```python
@dataclass
class PresetConfig:
    ...
    specialist_builders_enabled: bool = False
    specialist_builders_mode: str = "off"   # off | hero_only | hero_env | full
    asset_lab_enabled: bool = False
```

Add deterministic section metrics:

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


@dataclass
class SectionBuildMetricRecord:
    section_name: str
    builder_name: str = ""
    builder_version: str = ""
    quality_proxy_score: float = 0.0
```

### Exit Criteria

- rollout is feature-flagged
- a stable benchmark set exists
- builder identity/version can be tracked even before specialists ship

## Phase 1: Canonical Scaffold, Typed Registry, and Section Stubs

Goal:
- make the current script structure ready for section specialization

### Tasks

1. Make all seven canonical sections mandatory for new production scripts.
2. Introduce a typed `BuildRegistry`.
3. Add section ownership markers.
4. Derive scaffold generation from the existing canonical section list.
5. Make specialist-owned sections emit stubs rather than wasted full implementations.
6. Keep the single Script Writer as the fallback that can still fill every section.

### File Targets

- `utils/script_sections.py`
- `models/pipeline_models.py`
- `tools/script_generator_tools.py`
- `guardrails/script_guardrails.py`
- `orchestrator.py`
- `tests/test_script_sections.py`
- `tests/test_patch_tool.py`

### Implementation Notes

Add a typed registry:

```python
class BuildRegistry(BaseModel):
    hero_objects: dict[str, str] = Field(default_factory=dict)
    support_objects: dict[str, str] = Field(default_factory=dict)
    materials: dict[str, str] = Field(default_factory=dict)
    collections: dict[str, str] = Field(default_factory=dict)
    prebuilt_assets: dict[str, str] = Field(default_factory=dict)
    camera_name: Optional[str] = None
    world_name: Optional[str] = None
```

Keep `CANONICAL_SECTIONS` as the source of truth and derive the scaffold from it:

```python
def render_base_scaffold(plan: Optional["SceneBuildPlan"] = None) -> str:
    lines = ["import bpy", "import math", ""]

    for section in CANONICAL_SECTIONS:
        owner = _get_owner(plan, section) if plan else "single_writer"
        lines.append(f"def {section}(registry: BuildRegistry):")
        lines.append(f"    # SECTION_OWNER: {owner}")
        lines.append("    raise NotImplementedError")
        lines.append("")

    lines.append("registry = BuildRegistry()")
    for section in CANONICAL_SECTIONS:
        lines.append(f"{section}(registry)")
    return "\n".join(lines)
```

Specialist-owned sections should use stubs:

```python
def create_geometry(registry: BuildRegistry):
    # SECTION_OWNER: hero_asset_builder
    # SPECIALIST_PLACEHOLDER: HeroAssetBuilder will replace this section
    # MUST_CREATE: wine_glass, candle_holder, table_surface
    raise NotImplementedError("specialist section placeholder")
```

### Tests

- scaffold always derives from `CANONICAL_SECTIONS`
- scaffold uses typed registry signatures consistently
- missing section output is rejected before Blender execution
- patching still works on scaffold-first scripts

### Exit Criteria

- all new scripts use the canonical scaffold
- registry naming is typed from the start
- specialist placeholders can be inserted without breaking the scaffold contract

## Phase 2: Deterministic SceneBuildPlan and Shared Models

Goal:
- introduce typed section planning without adding another ambiguous agent

### Tasks

1. Add `SceneBuildPlan`, `SectionAssignment`, and `SectionBuildOutput`.
2. Extend session/context state with plan, registry, section ownership, and builder metadata.
3. Add deterministic `plan_scene_build()` with complete default section ownership.
4. Formalize section-level issue mapping early.
5. Add asset-lab-ready fields now so later promotion does not require schema churn.

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
    builder_name: str = ""
    builder_version: str = ""
    created_objects: list[str] = Field(default_factory=list)
    created_materials: list[str] = Field(default_factory=list)
    registry_updates: dict[str, str] = Field(default_factory=dict)
    asset_reuse_candidates: list[str] = Field(default_factory=list)
    standalone_test_passed: bool = False
```

Extend state:

```python
class SessionState(BaseModel):
    ...
    scene_build_plan: Optional[Dict[str, Any]] = None
    build_registry: Dict[str, Any] = Field(default_factory=dict)
    section_owners: Dict[str, str] = Field(default_factory=dict)
    section_repair_counts: Dict[str, int] = Field(default_factory=dict)
```

Use a deterministic planner:

```python
def plan_scene_build(
    style_spec: StyleSpec,
    technique_contract: TechniqueContract,
    specialist_mode: str,
) -> SceneBuildPlan:
    assignments = [
        SectionAssignment(builder_kind="single_writer", owned_section="setup_scene"),
        SectionAssignment(builder_kind="single_writer", owned_section="create_geometry"),
        SectionAssignment(builder_kind="single_writer", owned_section="setup_materials"),
        SectionAssignment(builder_kind="single_writer", owned_section="setup_physics"),
        SectionAssignment(builder_kind="single_writer", owned_section="setup_lighting"),
        SectionAssignment(builder_kind="single_writer", owned_section="setup_camera"),
        SectionAssignment(builder_kind="single_writer", owned_section="bake_and_render"),
    ]

    if specialist_mode in {"hero_only", "hero_env", "full"} and style_spec.hero_objects:
        assignments = [
            a for a in assignments
            if a.owned_section != "create_geometry"
        ]
        assignments.append(SectionAssignment(
            builder_kind="hero_asset",
            owned_section="create_geometry",
            objectives=[f"Build hero: {h}" for h in style_spec.hero_objects],
            must_create=style_spec.hero_objects,
        ))

    return SceneBuildPlan(
        scene_goal=style_spec.setting or "scene",
        realism_obligations=[],
        assignments=assignments,
    )
```

Important rule:

- `section_owners` must be complete for all canonical sections on every run, even when specialists are disabled

Formalize early issue mapping:

```python
ISSUE_KIND_TO_SECTION = {
    "materials": "setup_materials",
    "environment": "setup_scene",
    "composition": "setup_camera",
    "lighting": "setup_lighting",
    "hero_object": "create_geometry",
    "lookdev": "setup_materials",
}
```

### Tests

- planner produces deterministic assignments for representative prompts
- no two builders own the same canonical section
- plan is serializable and round-trips through session persistence
- section ownership can be derived from plan without LLM involvement

### Exit Criteria

- every first iteration can produce a `SceneBuildPlan`
- section ownership is explicit and durable
- models are ready for both scene mode and later asset-lab mode

## Phase 3: Specialist Base Infrastructure and Validation Boundaries

Goal:
- build the infrastructure so specialists fail safely and predictably

### Tasks

1. Add a reusable base factory for section builders.
2. Add section-builder guardrails that reject output outside owned sections.
3. Add deterministic assembly.
4. Add builder-level truth-pack validation before assembly.
5. Add post-assembly cross-reference validation.

### File Targets

- new: `specialized_agents/section_builder_base.py`
- new: `guardrails/section_builder_guardrails.py`
- new: `utils/section_assembler.py`
- `specialized_agents/__init__.py`
- `tools/truth_pack_validator.py`
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
            blender_doc_search_bundle,
            search_code_patterns,
            get_pattern_code,
            patch_script_section,
            validate_script_truth_pack,
        ],
        output_type=AgentOutputSchema(SectionBuildOutput, strict_json_schema=False),
        input_guardrails=[...],
        output_guardrails=[...],
    )
```

Output guardrail sketch:

```python
def validate_section_build_output(
    output: SectionBuildOutput,
    owned_sections: set[str],
) -> None:
    if output.section_name not in owned_sections:
        raise ValueError(f"Unauthorized section: {output.section_name}")
    expected_def = f"def {output.section_name}("
    if expected_def not in output.code:
        raise ValueError(
            f"Section code/function mismatch: expected {expected_def}"
        )
```

Deterministic assembler:

```python
def assemble_sections(base_script: str, outputs: list[SectionBuildOutput]) -> str:
    source = base_script
    for out in outputs:
        source = replace_section(source, out.section_name, out.code)
    return source
```

Validate builder output on a temporary scaffold before full assembly:

```python
def validate_specialist_section(
    base_scaffold: str,
    output: SectionBuildOutput,
    truth_pack: dict,
) -> tuple[str, list[str]]:
    source = replace_section(base_scaffold, output.section_name, output.code)
    temp_path = write_temp_script(source)
    return validate_and_fix_script(temp_path, truth_pack)
```

Add cross-reference validation:

```python
def validate_cross_references(
    assembled_script: str,
    registry: BuildRegistry,
) -> list[str]:
    warnings = []
    for logical_name, blender_name in registry.hero_objects.items():
        if blender_name not in assembled_script:
            warnings.append(f"{logical_name} -> {blender_name} never referenced")
    return warnings
```

### Tests

- specialist cannot return an unowned section
- specialist cannot return code whose function name does not match `section_name`
- specialist output is doc-grounded and truth-pack-validated before assembly
- assembler preserves untouched sections
- duplicate section outputs are rejected
- cross-reference validation catches missing object/material references

### Exit Criteria

- the specialist boundary is a real validation boundary
- assembly is deterministic
- integration bugs fail early, before Blender execution

## Phase 4: HeroAssetBuilder MVP

Goal:
- improve hero-object quality first, with minimal orchestration risk

### Why First

`create_geometry()` is one of the most visible quality bottlenecks and one of the easiest specialist wins.

### Tasks

1. Add `HeroAssetBuilder`.
2. Restrict it to `create_geometry()`.
3. Ground it directly in `StyleSpec` hero objects, material goals, and camera distance.
4. Keep the rest of the script owned by the single writer.
5. Have the single writer emit a stub for `create_geometry()` in specialist mode.
6. Benchmark hero-only rollout against baseline.

### File Targets

- new: `specialized_agents/hero_asset_builder.py`
- `orchestrator.py`
- `tools/dynamic_instructions.py`
- `tools/code_pattern_tools.py`
- new: `tests/test_hero_asset_builder.py`

### Implementation Notes

Prompt shape:

```python
HERO_ASSET_BUILDER_INSTRUCTIONS = """
You own ONLY create_geometry().

Your job:
- create the hero object(s) and support geometry needed for physical credibility
- use refinement operators and modifiers where appropriate
- avoid low-detail primitive-only outputs for hero objects
- do not modify materials, lighting, camera, or render settings

Return SectionBuildOutput only.
"""
```

Ground the prompt in `StyleSpec` fields directly:

```python
hero_prompt = f"""
You own ONLY create_geometry().

## Hero Objects
{", ".join(style_spec.hero_objects)}

## Material Goals
{", ".join(style_spec.hero_material_goals)}

## Camera Context
distance={style_spec.camera_distance_class}
dof={style_spec.dof_intent}

Close-up shots need more geometric detail.
Return SectionBuildOutput only.
"""
```

Safer rollout shape:

- single writer generates the script with a stubbed `create_geometry()`
- hero builder fills only `create_geometry()`
- fallback is single-writer section repair, not whole-script replacement

### Tests

- hero builder never touches non-geometry sections
- hero builder output compiles after insertion
- hero builder produces better geometry richness metrics than baseline on target prompts
- specialist failure falls back cleanly without corrupting the iteration

### Exit Criteria

- at least one benchmark family shows better hero geometry without worse execution reliability
- `create_geometry()` ownership is clean enough to patch independently

## Phase 5: EnvironmentAtmosphereBuilder

Goal:
- make scenes feel authored and situated

### Tasks

1. Add `EnvironmentAtmosphereBuilder`.
2. Start with `setup_scene()` ownership only.
3. Later allow combined `setup_scene()` + `setup_lighting()` ownership if evidence shows this improves mood coherence.
4. Ground prompts in `StyleSpec` mood, palette, and world lighting mode.
5. Keep environment builder failure paths reversible.

### File Targets

- new: `specialized_agents/environment_atmosphere_builder.py`
- `orchestrator.py`
- `utils/prompt_enhancer.py`
- `tools/dynamic_instructions.py`
- new: `tests/test_environment_builder.py`

### Implementation Notes

Prompt shape:

```python
ENVIRONMENT_ATMOSPHERE_BUILDER_INSTRUCTIONS = """
You own ONLY setup_scene().

Your job:
- establish the world/background strategy
- create environmental context and mood
- support the hero object rather than compete with it
- honor the StyleSpec palette and lighting mode

Do not modify geometry topology, materials, or camera.
"""
```

Later, if needed, combine `setup_scene()` and `setup_lighting()` under one builder because mood and world strategy are tightly coupled.

### Tests

- environment builder produces explicit world strategy
- mood cues align with `StyleSpec`
- environment builder failure falls back cleanly

### Exit Criteria

- scenes stop reading as empty blockouts on at least part of the benchmark pack
- environment setup improves without a large execution regression

## Phase 6: MaterialsLookDevBuilder

Goal:
- improve material credibility once geometry and environment are stronger

### Why Third

Materials depend on both geometry and lighting context. Running this before Phases 4-5 would polish weak assets.

### Tasks

1. Add `MaterialsLookDevBuilder`.
2. Restrict it to `setup_materials()`.
3. Feed it the `BuildRegistry` so it knows exact object/material targets.
4. Expand `IssueKind` and section routing for material/look-dev repair.
5. Keep material fixes section-local.

### File Targets

- new: `specialized_agents/materials_lookdev_builder.py`
- `models/pipeline_models.py`
- `tools/qa_diagnosis_bridge.py`
- `orchestrator.py`
- new: `tests/test_materials_lookdev_builder.py`
- `tests/test_repair_routing.py`

### Implementation Notes

Recommended routing categories:

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

Section routing:

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
- material builder reads registry entries rather than guessing names
- section-local material patches preserve geometry and environment

### Exit Criteria

- hero materials improve on benchmark prompts
- material fixes no longer require rewriting geometry or world setup

## Phase 7: Section-Local Repair Escalation

Goal:
- make section ownership useful during failure recovery

### Tasks

1. Map `QualityIssue.kind` and `target` to a canonical section.
2. Attempt repair with the owning specialist first.
3. Demote to single-writer repair for that section after repeated same-section failure.
4. Fall back to full-script repair only when section-local repair fails or multiple sections are implicated.
5. Preserve the existing escape-velocity behavior above that.

### File Targets

- `orchestrator.py`
- `models/shared_context.py`
- `tests/test_repair_routing.py`
- `tests/test_structural_repair_regression.py`
- new: `tests/test_specialist_repair_routing.py`

### Implementation Notes

Section routing helper:

```python
def resolve_repair_section(issue: QualityIssue, session: SessionState) -> Optional[str]:
    if issue.target and issue.target in session.section_owners:
        return issue.target
    return ISSUE_KIND_TO_SECTION.get(issue.kind)
```

Repair escalation:

```python
def choose_section_repair_owner(section_name: str, session: SessionState) -> str:
    attempts = session.section_repair_counts.get(section_name, 0)
    if attempts < 2:
        return session.section_owners.get(section_name, "single_writer")
    return "single_writer"
```

Escalation path:

- same specialist repair
- single-writer repair for that section
- full-script rewrite / technique switch if still failing

### Tests

- hero-object defect repairs `create_geometry()` first
- material defect repairs `setup_materials()` first
- repeated section failure demotes to single-writer section repair
- multi-section defects fall back honestly

### Exit Criteria

- section-owned failures are easier to repair than in the current baseline
- repeated specialist failure does not create repair loops

## Phase 8: Asset Lab Mode

Goal:
- turn successful specialist work into compounding reusable value

### Why It Matters

Everything else in this plan improves a single run. Asset Lab is what makes the gains compound across future runs.

### Tasks

1. Add `asset_lab` mode to builder requests.
2. Add asset-specific quality obligations.
3. Save promoted outputs and metadata into an asset registry.
4. Keep asset-lab execution logically separate from the live scene loop.
5. Let scene mode optionally consume `prebuilt_assets` from `BuildRegistry`.

### File Targets

- `models/pipeline_models.py`
- `models/shared_context.py`
- new: `tools/asset_library_tools.py`
- new: `tests/test_asset_lab_mode.py`
- docs: `docs/current/ASSET_LIBRARY_POLICY.md`

### Implementation Notes

Asset-lab request:

```python
class AssetLabRequest(BaseModel):
    asset_type: str
    style_family: str = ""
    variant_count: int = 1
    quality_threshold: float = 70.0
    reusable_in_effect_types: list[str] = Field(default_factory=list)
```

Saved asset metadata:

```python
class SavedAssetMetadata(BaseModel):
    asset_id: str
    asset_type: str
    builder_name: str
    builder_version: str
    script_fragment_path: str
    preview_render_path: Optional[str] = None
    quality_score: float = 0.0
    tags: list[str] = Field(default_factory=list)
```

Important policy:

- asset-lab mode must not pollute the live scene loop
- promotion must require stronger evidence than one successful scene use

### Tests

- asset-lab outputs are saved with metadata
- failed assets are not promoted
- scene mode can consume a promoted asset reference from `BuildRegistry`

### Exit Criteria

- at least one reusable asset family exists
- scenes can consume a promoted asset instead of rebuilding it from scratch

## Phase 9: Optional Bounded Parallelism

Goal:
- use parallelism only where dependencies are actually bounded

### Important Constraint

Do not start here.

### Safe First Candidates

- planning passes that do not emit Python yet
- asset-variant generation in asset-lab mode
- environment exploration that does not depend on finalized geometry names

### Unsafe Early Candidates

- hero geometry and materials generated in parallel without a registry
- camera and environment generated before geometry anchors exist
- overlapping ownership of the same canonical section

### Implementation Notes

If parallelism is later introduced, start with bounded groups:

```python
geometry_out, scene_out = await asyncio.gather(
    run_hero_asset_builder(registry, plan),
    run_environment_builder(registry, plan),
)

materials_out, lighting_out = await asyncio.gather(
    run_materials_builder(registry, plan),
    run_lighting_builder(registry, plan),
)
```

Only move to a real DAG scheduler if the dependency graph becomes materially more complex than the canonical-section graph.

### Exit Criteria

- no parallel branch can create overlapping section ownership
- deterministic assembly remains the final authority

## Cross-Cutting Guardrails

### Guardrail 1: Specialists stay inside owned sections

Implement in:

- `guardrails/section_builder_guardrails.py`
- `tests/test_section_builder_guardrails.py`

### Guardrail 2: Specialists respect `TechniqueContract` and `StyleSpec`

Implement via:

- explicit prompt contract
- output validation against section purpose
- benchmark checks for style/technique adherence

### Guardrail 3: Builder-level validation happens before assembly

Implement via:

- bundled doc grounding
- truth-pack validation of temporary scaffolded section output
- cross-reference validation after assembly

### Guardrail 4: Single-writer fallback always exists

If a specialist fails:

- log it
- preserve iteration state
- fall back to single-writer section repair or baseline generation

### Guardrail 5: Specialist rollout remains feature-flagged

The system must support at least:

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

Run on a fixed prompt pack:

- baseline single writer
- hero-only specialist
- hero+environment specialists
- hero+environment+materials specialists

Record:

- execution success rate
- average quality score
- issue distribution
- section-local repair rate
- richness metrics
- builder version identity

## Recommended First Milestone

The best first milestone is:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4 only for `HeroAssetBuilder`

Deliverable:

- scaffolded scripts exist
- typed registry exists
- deterministic `SceneBuildPlan` exists
- `HeroAssetBuilder` can fill `create_geometry()`
- builder-level validation exists
- hero-only rollout can be benchmarked against baseline

Do not start environment or materials specialists before this milestone is proven.

## Recommended Job Order

1. add rollout flags and benchmark fixtures
2. add section richness and builder-version metrics
3. add typed `BuildRegistry`
4. derive scaffold generation from `CANONICAL_SECTIONS`
5. add section ownership markers and stubs
6. add `SceneBuildPlan` / `SectionAssignment` / `SectionBuildOutput`
7. add deterministic `plan_scene_build()`
8. add section-builder base factory
9. add section-builder guardrails
10. add assembler + temporary scaffold truth-pack validation
11. add cross-reference validation
12. implement `HeroAssetBuilder`
13. wire hero-only mode behind a flag
14. add section-local repair escalation counters
15. run hero-only benchmark comparison
16. only after that, start environment builder work

## What To Defer

Defer until after the hero-only milestone:

- more than one specialist in production by default
- generalized parallel section generation
- full asset-lab promotion workflows
- deeper orchestrator decomposition specifically for specialists

## Recommendation

Proceed with the specialist-builder direction, but only in the revised shape defined here:

- deterministic planner first
- typed registry first
- stubbed specialist-owned sections
- builder-level validation boundaries
- hero-first rollout
- section-local repair escalation

That is the highest-leverage, lowest-regret path. It preserves the existing patching investment, improves the scaffold even if specialists disappoint, and creates a realistic path toward reusable assets once the first specialist proves itself.
