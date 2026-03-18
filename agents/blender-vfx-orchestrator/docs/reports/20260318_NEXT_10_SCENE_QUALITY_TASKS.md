# Next 10 Scene Quality Tasks

Status: active
Last verified: 2026-03-18
Purpose: short execution list for improving scene and image quality without jumping prematurely into the full specialist-builder architecture

Companion docs:
- `docs/CURRENT_ROADMAP.md`
- `docs/reviews/2026-03/20260316_SCENE_AESTHETIC_REALISM_ACTION_PLAN.md`
- `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_IMPLEMENTATION_PLAN.md`

## Sequencing Principle

Do not go straight into the full section-builder plan.

The best order is:

1. take the highest-leverage visual quick wins inside the current single-writer pipeline
2. add the minimum measurement needed to tell if the changes actually helped
3. only then start the hero-only specialist-builder milestone

This gives faster visible improvement while keeping the architecture change bounded.

## The Next 10 Tasks

### 1. Create the aesthetic benchmark pack and capture a baseline

Purpose:
- create a fixed comparison set before making visual changes

Why first:
- without this, “looks better” will drift with memory and recent examples

File targets:
- `docs/current/AESTHETIC_BENCHMARK_SET.md` or `tests/fixtures/aesthetic_benchmarks.json`

Definition of done:
- 8-12 prompts covering tabletop glass, candlelit macro, cloth, window/glass, atmospheric interior, product close-up
- baseline renders and current scores saved for comparison

### 2. Add deterministic look-dev coverage metrics

Purpose:
- make scene richness measurable at the script level

Why second:
- this is the cheapest way to see whether new scripts are actually getting richer before the render stage

File targets:
- `tools/script_analysis_tools.py` or `tools/lookdev_metrics.py`

Definition of done:
- coverage metrics for color management, DOF, world strategy, material count, hero-object refinement count, atmosphere presence

### 3. Add `hero_object` and `lookdev` issue kinds

Purpose:
- let the system describe the most important visual failures explicitly

Why third:
- until these exist, the runtime still under-describes weak hero geometry and sterile look-dev

File targets:
- `models/pipeline_models.py`
- `docs/CURRENT_ROADMAP.md` if categories change materially

Definition of done:
- `IssueKind` includes `hero_object` and `lookdev`
- those values are accepted anywhere `QualityIssue.kind` is validated

### 4. Update Quality Analyst prompting to emit those kinds with section targets

Purpose:
- make visual failures actionable instead of generic prose

Why fourth:
- new issue kinds are only useful if the evaluator actually emits them

File targets:
- `tools/dynamic_instructions.py`
- `guardrails/quality_guardrails.py`
- `orchestrator.py`

Definition of done:
- close-up primitive/faceted hero objects emit `hero_object`
- flat/sterile/tone-mapping failures emit `lookdev`
- issues point at canonical targets like `create_geometry`, `setup_materials`, `setup_scene`, `setup_camera`

### 5. Route aesthetic issue kinds to structural repair, not parameter tweaking

Purpose:
- stop visual quality problems from being sent down parameter-only paths

Why fifth:
- once the evaluator emits better issue kinds, the repair router needs to respect them immediately

File targets:
- `orchestrator.py`
- `tests/test_repair_routing.py`
- `tests/test_structural_repair_regression.py`

Definition of done:
- `hero_object` routes to `create_geometry`
- `lookdev` routes to `setup_materials`
- environment/composition issues continue routing to code change, not scalar tweaks

### 6. Implement the hero-object refinement helper on the current live path

Purpose:
- get an immediate visible upgrade in hero geometry without waiting for specialist builders

Why sixth:
- this is probably the highest-value quick win for image quality

File targets:
- `tools/hero_object_refinement.py`
- `orchestrator.py`
- `tools/dynamic_instructions.py`

Definition of done:
- hero objects like glass, bottles, windows, vessels, candles, cloth, masonry get an explicit refinement pass
- the helper can be called from the existing single-writer workflow

### 7. Enforce camera/readability decisions

Purpose:
- improve image composition and subject clarity

Why seventh:
- many weak renders are not just under-modeled; they are badly framed or visually flat

File targets:
- `tools/dynamic_instructions.py`
- `orchestrator.py`
- `utils/prompt_enhancer.py`

Definition of done:
- prompts produce explicit camera/readability intent
- close-up prompts reliably include subject-emphasizing camera choices and DOF decisions

### 8. Add the minimum viable iteration manifest for visual work

Purpose:
- capture enough evidence to compare visual changes honestly

Why eighth:
- before going into section builders, you want script hash, render path, score, issue kinds, and change tags recorded consistently

File targets:
- `models/shared_context.py`
- `tools/artifact_tools.py`
- `tools/script_analysis_tools.py`
- `tests/test_artifact_gates.py`

Definition of done:
- each benchmark run records script hash, render path, score, issue kinds, and change label or builder label

### 9. Run a focused benchmark pass and decide what actually moved quality

Purpose:
- lock in the quick wins that produce real visual gains

Why ninth:
- this is the checkpoint before starting the first specialist-builder milestone

File targets:
- benchmark fixture
- trace/manifest outputs
- comparison note under `docs/reports/`

Definition of done:
- baseline vs post-quick-win comparison exists
- at least 1-2 changes show clear positive visual impact without major execution regression

### 10. Start only the hero-only specialist-builder milestone

Purpose:
- begin the architecture change at the narrowest high-upside point

Why tenth:
- after quick wins and measurement, this becomes the right next step instead of a speculative jump

File targets:
- `specialized_agents/section_builder_base.py`
- `specialized_agents/hero_asset_builder.py`
- `guardrails/section_builder_guardrails.py`
- `utils/section_assembler.py`
- `orchestrator.py`

Definition of done:
- typed `BuildRegistry`
- scaffold-first generation with specialist stubs
- deterministic `SceneBuildPlan`
- builder-level truth-pack validation
- `HeroAssetBuilder` only
- hero-only benchmark comparison against baseline

## Do Not Start Yet

These should wait until Task 10 is complete and benchmarked:

- `EnvironmentAtmosphereBuilder`
- `MaterialsLookDevBuilder`
- asset-lab promotion workflows
- generalized parallel section generation
- broader autonomy/HITL expansion tied to this workstream

## Recommendation

If the goal is faster visual improvement, the immediate highest-value slice is:

1. Tasks 1-5
2. Tasks 6-7
3. Task 8
4. Task 9
5. then Task 10

That is the best balance between quick wins and not fooling yourself about whether quality is actually improving.
