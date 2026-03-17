# Specialized Section Builders Design Note

Status: active
Last verified: 2026-03-17
Purpose: design note for introducing specialist script-section builders without destabilizing the existing pipeline

## Executive Summary

The right next step is not "several agents all writing Python in parallel." The right next step is a section-builder architecture: one director defines the scene and quality obligations, then specialized builders own bounded script sections or section fragments under strict contracts, with a deterministic assembler preserving naming, order, and execution safety.

This can increase aesthetic quality, asset richness, and reusability while also making patching much easier. It is especially attractive for hero-object modeling, materials/look-dev, environment mood, and scene dressing, because those are currently underdeveloped by the single Script Writer. The least disruptive rollout is to keep the current seven canonical sections first, introduce section ownership and manifests, and only then allow specialist builders to replace the single writer on selected sections.

## Why This Direction Makes Sense

The current Script Writer is trying to solve too many jobs in one pass:

- technique correctness
- geometry creation
- modifier/refinement decisions
- material look-dev
- environment/world design
- lighting mood
- camera/composition
- execution safety
- repairability

That encourages conservative scripts. The model optimizes for "likely to run" rather than "beautiful and convincing," which is why outputs often collapse into basic primitives, simple shaders, thin scene dressing, and low environment presence.

Specialization helps because the reasoning surface becomes narrower:

- a hero-object builder can spend its budget on silhouette, bevels, subdivision, thickness, and asset credibility
- a materials/look-dev builder can focus on shader graphs, roughness variation, transmission, color separation, and surface detail
- an environment builder can focus on world setup, practical lights, atmosphere, background treatment, and mood
- a scene-dressing builder can focus on secondary objects and lived-in context

The second major benefit is reuse. The same specialist builder that creates a good chair or table lamp for one scene can later be run in an asset-lab mode to create reusable library assets.

## What This Is Not

This note does **not** propose:

- unrestricted parallel free-writing of a shared script
- removing the current pipeline state machine
- replacing LLM creativity with templates
- switching to asset-library-only workflows
- forcing scenes to become short or minimal

The goal is richer, more beautiful, more modular scripts, not simpler ones.

## Core Design Principles

1. One agent should own global coherence.
   A single `Director` or planning phase must own scene intent, composition priorities, realism target, and overall aesthetic direction.

2. Specialists should own bounded responsibilities.
   A builder should own either a canonical section or a tightly scoped subsection with clear inputs and outputs.

3. Integration must be deterministic.
   Builders should not negotiate state implicitly in prompt prose. They should exchange typed manifests.

4. Patchability must improve, not regress.
   Any specialization step that makes section repair harder is the wrong shape.

5. Reuse is a first-class outcome.
   Strong builders should eventually support both `scene mode` and `asset lab mode`.

## Proposed Architecture

### 1. Director

The Director remains the top-level authorial brain. It owns:

- `TechniqueContract`
- `StyleSpec`
- scene-level composition intent
- required hero objects
- minimum realism obligations
- section build order and dependencies

It should output a `SceneBuildPlan`, not Python code.

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field


BuilderKind = Literal[
    "hero_asset",
    "support_props",
    "materials_lookdev",
    "environment_atmosphere",
    "camera_composition",
    "physics_integration",
]


class SectionAssignment(BaseModel):
    builder_kind: BuilderKind
    owned_section: str = Field(description="Canonical section name")
    owned_subsections: list[str] = Field(default_factory=list)
    priority: int = 0
    depends_on: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    must_create: list[str] = Field(default_factory=list)
    must_not_change: list[str] = Field(default_factory=list)


class SceneBuildPlan(BaseModel):
    scene_goal: str
    technique_contract: dict
    style_spec: dict
    realism_obligations: list[str]
    assignments: list[SectionAssignment]
```

### 2. Specialist Builders

Specialist builders should not emit a full script. They should emit one of:

- a replacement canonical section
- a subsection function plus registration manifest
- an asset module that can be composed into a canonical section

The first rollout should keep them bound to current canonical sections:

- `setup_scene`
- `create_geometry`
- `setup_materials`
- `setup_physics`
- `setup_lighting`
- `setup_camera`
- `bake_and_render`

That preserves the existing patching machinery.

Recommended first specialists:

- `HeroAssetBuilder`
  Owns: `create_geometry`
  Focus: modifiers, silhouette, bevels, thickness, subdivision, non-blocky hero objects

- `EnvironmentAtmosphereBuilder`
  Owns: `setup_scene` and/or `setup_lighting`
  Focus: world/background, practical lights, volume, atmosphere, mood, environmental storytelling

- `MaterialsLookDevBuilder`
  Owns: `setup_materials`
  Focus: hero shaders, variation, transmission, roughness breakup, palette coherence

Recommended later specialists:

- `SceneDressingBuilder`
- `CameraCompositionBuilder`
- `AssetReuseComposer`

### 3. Deterministic Assembler

The assembler merges builder outputs into the final script. It owns:

- canonical function order
- shared registry injection
- import hygiene
- name collision prevention
- object/material dependency ordering
- section manifest validation

```python
class SectionBuildOutput(BaseModel):
    section_name: str
    code: str = Field(description="Complete canonical function definition")
    created_objects: list[str] = Field(default_factory=list)
    created_materials: list[str] = Field(default_factory=list)
    exports: dict[str, str] = Field(default_factory=dict)
    depends_on_objects: list[str] = Field(default_factory=list)
    touched_globals: list[str] = Field(default_factory=list)
    quality_obligations_met: list[str] = Field(default_factory=list)


def assemble_script(
    base_scaffold: str,
    section_outputs: list[SectionBuildOutput],
) -> str:
    source = base_scaffold
    for out in section_outputs:
        source = replace_section(source, out.section_name, out.code)
    return source
```

### 4. Shared Registry

To avoid prompt-level coupling, builders should work against a shared registry contract.

```python
class BuildRegistry(BaseModel):
    hero_objects: dict[str, str] = Field(default_factory=dict)       # logical -> blender object name
    support_objects: dict[str, str] = Field(default_factory=dict)
    materials: dict[str, str] = Field(default_factory=dict)          # logical -> material name
    collections: dict[str, str] = Field(default_factory=dict)
    camera_name: Optional[str] = None
    world_name: Optional[str] = None
```

Example:

- `HeroAssetBuilder` creates `hero_glass` -> `WineGlass_Main`
- `MaterialsLookDevBuilder` reads that registry entry and knows exactly which object to shade
- `CameraCompositionBuilder` can frame `hero_glass` without guessing names

### 5. Scene Mode vs Asset Lab Mode

A major upside of specialization is that the same builder can operate in two modes.

`Scene mode`:
- build assets for one scene only
- optimize for immediate coherence

`Asset lab mode`:
- create a reusable asset with stricter geometry/material obligations
- save outputs as asset modules or reusable script fragments
- promote only after evaluation passes a reusable-asset bar

```python
BuildMode = Literal["scene", "asset_lab"]


class BuilderRequest(BaseModel):
    mode: BuildMode = "scene"
    asset_type: Optional[str] = None
    style_spec: dict
    section_assignment: dict
    registry_snapshot: dict = Field(default_factory=dict)
```

This is how you get from "build a wooden chair for this tavern scene" to "generate three high-quality wooden chair variants and save them as reusable assets."

## Where This Fits the Current Codebase

The current code already has several pieces that make this feasible:

- `TechniqueContract` in `models/pipeline_models.py`
- `StyleSpec` in `models/pipeline_models.py`
- canonical section parsing/replacement in `utils/script_sections.py`
- `patch_script_section()` in `tools/script_generator_tools.py`
- section patch routing already exercised in `orchestrator.py`

That means the least painful design is **not** to replace the whole script-generation stack. It is to add a section-planning and section-ownership layer on top of what already exists.

## Least Disruptive Rollout

### Phase A: Modularize Without More Agents

First, keep the existing single Script Writer but force it into an explicit scaffold:

- every script must contain all canonical sections
- sections must declare owned objects/materials in comments or manifest blocks
- section outputs should align to `StyleSpec` and hero-object obligations

This produces two immediate benefits:

- the script becomes easier to patch
- the future specialist interface becomes concrete

Example scaffold:

```python
def setup_scene(ctx):
    # owns: collections, world baseline, support surfaces
    ...


def create_geometry(ctx):
    # owns: hero objects, support props, layout anchors
    ...


def setup_materials(ctx):
    # owns: hero shaders, support materials
    ...


def setup_lighting(ctx):
    # owns: key/fill/rim or practicals, atmospheric volume hooks
    ...
```

### Phase B: Introduce One Specialist

Replace only `create_geometry` first with `HeroAssetBuilder`.

Reason:

- it is one of the biggest visible quality gaps
- modifier-driven refinement is highly specialist work
- it is easier to evaluate than full-scene aesthetic coherence

Success criteria:

- fewer blocky hero objects
- stronger silhouette and material-readiness
- section patching still works

### Phase C: Add Environment/Atmosphere Specialist

Next replace `setup_scene` plus part of `setup_lighting`, or keep them separate but coordinated.

This builder should own:

- world/background strategy
- environmental context
- atmospheric volume
- practical light placement
- mood coherence

This is the highest-leverage specialist for making scenes feel immersive rather than isolated objects in empty space.

### Phase D: Add Materials/Look-Dev Specialist

This builder becomes worthwhile once hero objects and environment are better, because high-quality materials depend on believable geometry and lighting context.

It should own:

- hero material graphs
- support material variation
- palette coherence
- look-dev polish passes

### Phase E: Asset Lab

Once one or two builders are working in scene mode, allow them to run in `asset_lab` mode and save reusable outputs.

Asset lab candidates:

- chairs
- tables
- bottles and glasses
- candles and holders
- lanterns
- masonry fragments

## Script Length: What to Optimize For

The concern about script length is directionally correct: for the image quality you want, very short scripts are usually not enough. Realistic immersive scenes often require more code because they need:

- richer geometry
- more material work
- more world setup
- more props
- more compositional logic
- more lighting control

But raw line count is still the wrong primary target. It is too easy to game and too indirect.

The better policy is:

- **do not optimize for short scripts**
- **do require scene richness obligations**
- **accept that richer scenes will often produce longer scripts**

So instead of "minimum 1200 lines," enforce things like:

- at least one hero-object refinement path
- explicit world/background strategy
- explicit composition/camera choice
- secondary scene dressing for non-minimal prompts
- material variation beyond defaults
- lighting separation between subject and background

If those are met honestly, scripts will often grow naturally.

## Advantages of This Architecture

1. Better visual quality.
   Specialists can spend their budget on craft rather than generic execution safety.

2. Easier patching.
   Section-local ownership maps cleanly to `patch_script_section()`.

3. Better reuse.
   The same builder can be promoted into asset-library generation.

4. More honest complexity.
   You can allow richer scripts without making every repair a whole-script rewrite.

5. Better evaluation.
   Quality issues can target the responsible builder/section more directly.

6. Better experimentation.
   You can compare two environment builders or two chair-builder variants without rewriting the full scene pipeline.

## Main Risks and Mitigations

### Risk: Style incoherence across builders

Mitigation:

- strong `StyleSpec`
- one Director
- section obligations expressed as typed fields, not prose only
- optional final look-dev polish pass

### Risk: Dependency mismatches

Mitigation:

- shared registry
- deterministic assembler
- section manifests
- ownership rules about who can create or modify what

### Risk: Too much orchestration complexity too early

Mitigation:

- keep current seven canonical sections first
- add one specialist at a time
- do not introduce many new agents before patching is stable

### Risk: Specialists becoming repetitive

Mitigation:

- scene mode vs asset-lab mode
- preserved LLM creative latitude within `StyleSpec`
- retrieval of varied high-quality code patterns

## Concrete Implementation Sketch

### 1. Add builder-role metadata

```python
from typing import Literal

OwnedSection = Literal[
    "setup_scene",
    "create_geometry",
    "setup_materials",
    "setup_physics",
    "setup_lighting",
    "setup_camera",
    "bake_and_render",
]


class BuilderRole(BaseModel):
    name: str
    owned_sections: list[OwnedSection]
    allowed_tools: list[str]
    forbidden_sections: list[OwnedSection] = Field(default_factory=list)
```

### 2. Create specialist agent factories

```python
def create_hero_asset_builder() -> Agent:
    return Agent(
        name="Hero Asset Builder",
        model=os.getenv("SCRIPT_WRITER_MODEL", "gpt-5.4"),
        instructions=HERO_ASSET_BUILDER_INSTRUCTIONS,
        tools=[
            semantic_search_blender_docs,
            search_blender_api_by_intent,
            search_code_patterns,
            get_pattern_code,
            write_script,
            patch_script_section,
        ],
    )
```

### 3. Add a planning step before code generation

```python
class SectionPlanOutput(BaseModel):
    assignments: list[SectionAssignment]
    rationale: str = ""


async def plan_section_builds(ctx: SharedContext) -> SectionPlanOutput:
    # Director or deterministic planner creates the build DAG
    ...
```

### 4. Let the orchestrator call specialist builders selectively

```python
if use_specialists and "create_geometry" in target_sections:
    hero_out = await run_hero_asset_builder(...)
    assembled_script = assemble_script(base_scaffold, [hero_out, ...])
else:
    script_out = await run_single_script_writer(...)
```

### 5. Keep repair routing section-local

```python
if repair_intent.mode == "modify_code":
    for issue in quality.structured_issues:
        if issue.target in {"create_geometry", "setup_materials", "setup_lighting"}:
            # Patch only the owned section using the responsible builder
            ...
```

## Recommended First Slice

If this direction is pursued, the best first slice is:

1. keep the existing single Script Writer as the default
2. formalize section ownership and manifests
3. add `HeroAssetBuilder` for `create_geometry`
4. use the current patching infrastructure to patch only that section
5. evaluate whether hero objects improve without destabilizing execution

If that works, the next slice should be:

6. add `EnvironmentAtmosphereBuilder` for `setup_scene` / `setup_lighting`
7. only after that, add `MaterialsLookDevBuilder`

That order attacks the most visually obvious problems first.

## Recommendation

This is a strong direction, but it should be implemented as **section specialization over a deterministic scaffold**, not as a free-form multi-agent rewrite. If done carefully, it can solve several problems at once:

- richer scenes
- better hero objects
- stronger environment mood
- cleaner patchability
- reusable assets

The painless path is incremental:

1. modularize the current script format more explicitly
2. introduce one specialist builder
3. prove that section-level generation and patching improve quality
4. expand to environment and material specialists
5. later split off asset-lab workflows for reusable scene assets

If this works, it becomes a natural bridge between the current script-centric pipeline and a more mature system that can both build complete scenes and accumulate reusable artistic assets over time.
