# Scene Aesthetic Realism Deep Dive

Status: active analysis
Date: 2026-03-16
Scope: why recent scenes feel blockout-like or aesthetically weaker, and what needs to change to restore richer materials, mood, composition, and hero-object quality without turning the system into a template filler

## Executive Summary

The current regression is not primarily a "scripts are too short" problem. Recent weak scripts are often as long as, or longer than, older stronger scripts; the difference is that more of the code budget is now spent on technique plumbing, guardrails, repair scaffolding, and broad scene assembly, while less of it is spent on deliberate look-dev, hero-object refinement, tone mapping, and composition.

The system still contains evidence of strong aesthetic capability. Older scripts, seeded code patterns, and historical worklogs show that it can produce convincing mood, better glass, better materials, DOF, bokeh, and more intentional camera/light setups. The main problem is that this knowledge is no longer on the hot path of generation and repair.

The two biggest architectural gaps are:
- there is still no integrated prompt enhancement / creative brief stage in the live pipeline, even though the mission statement assumes one
- proven visual craft is stored mostly as historical scripts and code patterns, but the active writer and repair loop are not set up to retrieve and apply that craft structurally

My conclusion is:
- do not focus on making scripts longer
- do make scene look-dev a first-class artifact, alongside `TechniqueContract`
- do move visual craft knowledge onto the hot path through structural pattern application, better prompt enrichment, richer issue typing, and a deterministic "hero object + look-dev floor" pass

## Evidence Reviewed

Code and docs:
- `specialized_agents/script_writer.py`
- `tools/dynamic_instructions.py`
- `tools/script_generator_tools.py`
- `orchestrator.py`
- `utils/code_pattern_memory.py`
- `tools/code_pattern_tools.py`
- `tools/qa_diagnosis_bridge.py`
- `utils/quality_parameter_map.py`
- `tools/blender_api_fixer.py`
- `docs/MISSION_STATEMENT_2026-02-22.md`
- `docs/PROMPT_ENHANCEMENT_GUIDE.md`
- `docs/worklogs/WORKLOG_2026-02-25_GREY_ROOM_FIX.md`
- `docs/S1_UPGRADE_STATUS_AND_NEXT_STEPS_2026-02-14.md`

Representative scripts and artifacts:
- `assets/blender_scripts/generated/wine_pour_v1.py`
- `assets/blender_scripts/generated/wine_pour_codex.py`
- `assets/blender_scripts/generated/wine_pour_v5_water_vfx_v1.py`
- `assets/blender_scripts/generated/library_candle_v1.py`
- `assets/blender_scripts/generated/glass_shatter_ctxfilter_v2_iter3_paramfix.py`
- `sessions/artifacts/wine_pour_codex_20260311_033209/quality_iter1.json`
- `sessions/artifacts/glass_shatter_ctxfilter_v2_20260312_001350/quality_iter3.json`
- `sessions/artifacts/glass_shatter_codex_20260311_202642/quality_iter2.json`
- `sessions/artifacts/wine_pour_v5_20260225_221532/quality_iter1.json`

Official Blender 5.0 manual research:
- `render/color_management/displays_views.html`
- `render/lights/world.html`
- `render/shader_nodes/textures/environment.html`
- `render/shader_nodes/shader/principled.html`
- `render/shader_nodes/shader/glass.html`
- `render/cameras.html`
- `modeling/modifiers/generate/subdivision_surface.html`
- `modeling/modifiers/generate/solidify.html`
- `render/materials/components/volume.html`

## Direct Answer: Are the Scripts Too Short?

Mostly no.

The clearest counterexample is:

| Script | Lines | Notable visual features | Outcome |
|---|---:|---|---|
| `wine_pour_v1.py` | 702 | Filmic, DOF, bokeh, Solidify, Subsurf, more intentional hero-glass modeling | historically much stronger aesthetic direction |
| `wine_pour_codex.py` | 838 | no explicit view transform, no DOF, broader scene setup, still decent material work | scored 34 |
| `glass_shatter_ctxfilter_v2_iter3_paramfix.py` | 840 | no view transform, no DOF, no hero-object refinement, mostly primitives/material tweaks | scored 18 |

This is the core point: a 700-line script can look better than an 840-line script if the extra 140 lines in the latter are not buying visual craft.

The current problem is better described as:
- insufficient look-dev density per line
- weak reuse of proven visual craft
- too much of the repair loop spent on scalar edits and scene validity rather than aesthetic redesign

## What Regressed

### 1. The hot path is no longer biased toward visual craft

The active Script Writer tool surface is narrow:
- `write_script`
- `validate_script`
- `modify_script`
- doc search

It does not expose code-pattern retrieval or structural pattern application to the writer in the initial generation path. See `specialized_agents/script_writer.py`.

That means the writer can verify APIs, but it cannot directly ask for "show me the proven glass mesh stack" or "apply the photoreal wine-glass pattern."

The repository still remembers those patterns:
- `scripts/seed_code_patterns.py` seeds `glass_mesh_solidify_subdivision`
- it seeds `glass_material_principled_transmission`
- it seeds `filmic_color_management`
- it seeds `subdivision_surface_smooth_objects`

But those patterns are mostly not available where they matter most: the first draft.

### 2. Pattern memory is being used as parameter memory

This is one of the biggest findings.

In `orchestrator.py`, Phase 0.95 searches for high-confidence patterns. But when a pattern is chosen, the pipeline tries to parse assignment lines out of the code snippet and turn them into scalar modifications before sending them through `_apply_script_modifications()`.

That works for:
- `bg.inputs['Strength'].default_value = 0.2`
- `noise.inputs['Scale'].default_value = 12.0`

It does not work for the patterns that actually improve realism:
- add Solidify modifier
- add Subdivision Surface modifier
- build a lathed glass profile
- add DOF focus object
- add tone mapping
- add subtle atmosphere volume
- add practical-light meshes plus colocated light objects

So the system stores the right kind of knowledge, then routes it through the wrong application path.

### 3. Prompt enhancement is still outside the live runtime

This is a mission/runtime gap, and it matters directly for aesthetics.

The mission statement says the product experience includes a prompt enhancement step that expands a short prompt into a richer creative brief. The current live pipeline still starts from `AssetRequest.description`.

The repo already has strong evidence that richer prompts materially improve scene quality. `docs/S1_UPGRADE_STATUS_AND_NEXT_STEPS_2026-02-14.md` records that the same kitchen leak scenario scored about `40` with an enhanced prompt vs `22` with a bare prompt.

This makes sense. Aesthetic quality depends on:
- mood
- palette
- surface choices
- shot framing
- environment cues
- light motivation

If the runtime never guarantees that those decisions are explicit, the writer has to invent too much from sparse input.

### 4. The current instructions over-allow primitive-based blockouts

`tools/dynamic_instructions.py` correctly says that supporting props can be simple primitives with good materials. That rule is useful for set dressing, but in practice the system appears to be over-generalizing it to hero objects and key scene reads.

This is especially visible in glass shatter scenes:
- the brick reads as a small cube
- the pane/frame reads as a blockout
- the warehouse reads as a few primitives plus procedural textures

The missing distinction is:
- background props may be simple
- hero objects and key read surfaces must receive a refinement pass

At the moment, that distinction is not enforced strongly enough.

### 5. The repair loop is better at tweaking than at re-art-directing

The repair system now has:
- `RepairIntent`
- section patching
- deterministic routing

That is correct for reliability. But aesthetically weak scenes often require coordinated changes across:
- `setup_materials`
- `setup_lighting`
- `setup_camera`
- `setup_scene` / world background
- sometimes `create_geometry`

The current loop still spends a lot of its effort in:
- `utils/quality_parameter_map.py`
- `_modify_script_impl()` scalar replacements in `tools/script_generator_tools.py`
- parameter-oriented QA grounding in `tools/qa_diagnosis_bridge.py`

That means many aesthetic failures get reduced to:
- world strength
- light energy
- a few shader values

when the real fix is:
- different shot
- different world/readability strategy
- better hero-object silhouette
- more believable material layering
- more motivated environment

### 6. There is a real color-management / look-dev regression

Older stronger scripts frequently included deliberate color management:
- `wine_pour_v1.py` uses `Filmic` + `Medium High Contrast`
- `library_candle_v1.py` uses `Filmic` + `Medium High Contrast`

Recent weaker scripts often omit view transform setup entirely:
- `wine_pour_codex.py`
- `glass_shatter_ctxfilter_v2_iter3_paramfix.py`
- `campfire_mantaflow_baseline.py`
- `clothesline_sheet_wind_v1.py`

This matters because Blender 5.0’s manual explicitly says:
- `Filmic` is deprecated
- `AgX` supersedes it and gives more photorealistic results

The exact fix is not "go back to Filmic." The fix is:
- make color management explicit again
- move to `AgX` as the baseline on Blender 5.0
- stop letting many scripts render with whatever default color-management state happens to be active

### 7. Environment lighting is too often a flat background color plus area lights

Official Blender docs say world lighting can be a fixed color, sky model, or HDRI/environment texture, and that environment textures are the correct way to do image-based lighting.

In the current scripts, the dominant pattern is:
- near-black world color
- a handful of area/point lights

That can work for stylized or highly controlled scenes, but it often produces:
- hard rectangular speculars
- weak environmental reflections
- poor material integration
- "studio test" lighting instead of believable setting

This matches the quality feedback in `wine_pour_v5_20260225_221532/quality_iter1.json`, which explicitly calls out:
- hard rectangular speculars
- reflections that do not integrate like a real environment

Given the mission statement, I would not make HDRI dependency mandatory. But the system needs a richer world-lighting strategy than flat background plus a few lamps.

### 8. The QA bridge is still weak for aesthetic diagnosis

`tools/qa_diagnosis_bridge.py` is useful, but it is still mostly keyword-to-parameter matching:
- dark -> energy
- smoke -> density
- camera -> lens/location

That is not enough for issues like:
- "the warehouse does not read as derelict"
- "glass lacks edge sparkle"
- "brick still reads as a cube"
- "the shot feels like previs, not a cinematic hero frame"

Those are not parameter-only issues. They are multi-section look-dev issues.

### 9. Visual learnings are siloed too narrowly

There are visual learnings in:
- old scripts
- worklogs
- code patterns
- the experiment tracker

But they are fragmented:
- dynamic instructions inject experiment-tracker learnings
- code patterns live in a separate store
- pattern retrieval is mostly keyed by issue strings
- effect-type filtering can hide generally useful visual patterns

For example, `filmic_color_management` exists as a `water` pattern, even though tone mapping is a cross-effect visual concern.

This makes the system worse at transferring scene craft across effect families.

## Blender 5 Research Notes That Matter Here

### AgX should replace implicit/default color management

Blender 5.0 manual, `render/color_management/displays_views.html`:
- `AgX` improves on `Filmic`
- `Filmic` is deprecated and superseded by `AgX`

Actionable implication:
- every generated scene should deliberately set a view transform
- Blender 5 default look-dev should target `AgX`, not implicit defaults

### Principled + thickness + smoothing still matters for hero objects

Blender 5.0 manual:
- `render/shader_nodes/shader/principled.html`
- `modeling/modifiers/generate/subdivision_surface.html`
- `modeling/modifiers/generate/solidify.html`

The docs align with what the old better scripts already discovered:
- Principled BSDF is the right general-purpose physically based surface shader
- Subdivision Surface is for smooth appearance from simple meshes
- Solidify adds actual thickness

This matches the historical glass improvements in `scripts/seed_code_patterns.py`.

### DOF and focus objects are still first-class realism tools

Blender 5.0 manual, `render/cameras.html`:
- DOF depends on focal point and aperture / f-stop
- focus objects are the clean way to drive focal distance

Older stronger scripts use this. Many recent weaker scripts do not.

### Better world lighting needs a world-lighting strategy, not just stronger backgrounds

Blender 5.0 manual:
- `render/lights/world.html`
- `render/shader_nodes/textures/environment.html`

Environment lighting can come from:
- fixed background
- sky model
- environment texture

The current system mostly uses the first option. That is the weakest option for realistic reflections and believable ambient integration.

## Root Causes

1. Visual craft knowledge is present but not first-class in generation.
2. Code pattern memory is structurally rich but applied through a scalar modification path.
3. The runtime still lacks automatic prompt enhancement / creative brief generation.
4. The system has no explicit `StyleSpec` or equivalent artifact to carry mood, palette, camera intent, hero-object expectations, and environment-readability constraints through the loop.
5. QA and repair routing are still too parameter-centric for materials, composition, environment, and hero-object realism.
6. There is no deterministic aesthetic floor for color management, camera/readability, and hero-object finishing.

## What Should Change

## 1. Add a `StyleSpec` or `LookDevContract`

Do for aesthetics what `TechniqueContract` started doing for physics/approach.

It should be generated before script writing and carried through the loop. Suggested fields:

```python
class StyleSpec(BaseModel):
    setting: str
    mood: str
    palette: list[str]
    camera_framing: str
    camera_distance_class: str
    dof_intent: str
    world_lighting_mode: str   # flat_bg | procedural_sky | env_texture | dark_world_plus_practicals
    hero_objects: list[str]
    hero_material_goals: list[str]
    support_prop_budget: str
    realism_target: str
```

This is not a template. It is a binding creative brief so the writer does not have to reconstruct aesthetic intent from scattered prose.

## 2. Integrate prompt enhancement into the runtime

This should happen before research and script generation.

The purpose is not verbosity for its own sake. It is to reliably surface:
- mood
- palette
- shot type
- hero object requirements
- material cues
- lighting intent
- environmental context

Without this, the system is still trying to make cinematic decisions from underspecified prompts.

## 3. Put visual code patterns on the hot path

The writer should be able to use structural visual patterns during generation, not only after failure.

Concretely:
- add `search_code_patterns`
- add `get_pattern_code`
- add structural pattern application to the writer or a small look-dev helper

Do not parse these patterns into scalar param edits. Apply them as:
- section patches
- helper-function insertion
- explicit object/material modifier blocks

This is especially important for:
- glass
- cloth
- wood/stone/metal material graphs
- DOF camera setups
- practical light rigs
- atmosphere volumes
- color-management setup

## 4. Split hero-object rules from support-prop rules

The current instruction "props can be simple primitives with good materials" is fine for support props. It is not fine for hero objects.

Add an explicit distinction:
- support props: primitives allowed
- hero objects: must receive a refinement pass

That refinement pass should be deterministic where possible:
- transparent vessel -> thickness + smoothing + highlight-catching edges
- soft fabric hero -> subdivision + cloth-friendly silhouette cleanup
- masonry projectile -> real proportions + edge wear + believable scale
- window pane -> thickness/readability pass, not just a flat slab

## 5. Add a deterministic "look-dev floor"

Not a template. A floor.

Every renderable script should pass a minimum look-dev checklist:
- explicit color management
- explicit world/background strategy
- explicit camera framing
- DOF decision made, even if disabled
- at least one motivated key light
- at least one hero-object finish pass if the prompt implies a close-up or product-like read

For Blender 5.0, the baseline should become:
- `AgX` if available
- fallback to a verified supported transform if not

## 6. Expand QA issue typing for aesthetics

Current issue kinds are too coarse. Add categories like:
- `materials`
- `environment`
- `composition`
- `hero_object`
- `lookdev`

Then map them to section targets:
- `setup_materials`
- `setup_lighting`
- `setup_camera`
- `setup_scene`
- `create_geometry`

That gives repair routing a better chance of triggering structural aesthetic repair rather than parameter nudging.

## 7. Stop using line count as the main proxy for richness

Keep the existing minimum warning if useful, but replace it with richer coverage metrics.

Examples:
- `has_explicit_color_management`
- `has_dof_decision`
- `hero_object_modifier_count`
- `material_node_complexity_score`
- `distinct_material_count`
- `practical_light_count`
- `world_lighting_mode`
- `hero_object_readability_features`

This will catch the real regressions more directly than `500+ lines`.

## 8. Upgrade world-lighting strategy

In order of compatibility with the mission:

1. procedural world modes by scene type
2. motivated practical-light rigs
3. procedural atmosphere volumes for depth
4. optional environment-texture path if using a tiny curated local library is acceptable

If external HDRIs conflict with the "from scratch" vision, then prefer:
- procedural sky for outdoor/daylight
- dark world + practicals + large soft cards for interiors
- subtle local atmosphere volumes for depth separation

## 9. Make visual knowledge cross-effect, not effect-siloed

Some visual craft is general:
- color management
- DOF
- smooth shading
- subdivision
- glass/refraction principles
- practical-light composition

Do not store these as `water`-only patterns or retrieve them only via effect-specific issue strings.

Introduce tags such as:
- `hero_object/glass`
- `lookdev/color_management`
- `camera/dof`
- `lighting/practicals`
- `materials/wood`
- `materials/aged_metal`

## 10. Allow aesthetic repair to patch multiple sections intentionally

Some scene failures are not one-section failures.

Examples:
- flat glass read -> materials + lighting + camera
- warehouse not believable -> scene + materials + lighting
- wine not premium -> materials + camera + world + lighting

The repair system should support a small multi-section aesthetic patch budget when QA issues are clearly look-dev driven.

## Ordered Plan

### Phase A: Immediate, highest leverage

1. Integrate prompt enhancement into the runtime.
2. Add `StyleSpec` / `LookDevContract`.
3. Expose code-pattern retrieval to the Script Writer on first draft.
4. Stop auto-applying structural code patterns as scalar modifications.

### Phase B: Make aesthetic knowledge executable

5. Create a small set of reusable visual pattern groups:
   - hero glass close-up
   - derelict industrial night
   - intimate candlelit tabletop
   - outdoor daylight soft-sky
   - atmospheric interior depth
6. Add structural application paths for these patterns via section patching or helper insertion.
7. Make general visual patterns cross-effect.

### Phase C: Improve repair semantics

8. Add new QA issue kinds for materials/environment/composition/hero_object/lookdev.
9. Route those kinds to multi-section patching.
10. Improve `qa_diagnosis_bridge` so it can say "missing DOF" or "no explicit tone mapping" or "hero object has no refinement modifiers," not only "light energy low."

### Phase D: Replace blunt richness proxies

11. Add script richness / look-dev coverage metrics.
12. Keep script length only as a weak heuristic, not a policy anchor.

## A Very Practical Near-Term Experiment Set

If the goal is to validate this without a large refactor, run these in order:

### Experiment 1: Prompt enhancement only

Take the same weak bare prompt and run:
- current raw prompt path
- enhanced prompt path

Measure:
- aesthetic score deltas
- issue type mix
- script feature coverage

### Experiment 2: AgX + DOF + hero-object finish floor

Add a deterministic look-dev floor only:
- explicit view transform
- explicit DOF decision
- hero glass/window/brick finish pass based on prompt entity detection

Do not change technique logic.

If this alone materially improves scores, it confirms that the current regression is partly look-dev omission, not only physics.

### Experiment 3: Structural pattern application on first draft

Enable the writer to retrieve and structurally apply:
- glass modifier/material pattern
- tone-mapping pattern
- practical-light / atmosphere pattern

Compare against current writer with the same prompt.

### Experiment 4: Aesthetic issue typing

Take recent failed renders and manually relabel issues as:
- materials
- composition
- environment
- hero_object
- lookdev

Then compare repair outcomes against the current parameter-heavy route.

## Bottom Line

The system has not lost the ability to create richer scenes. It has lost the architecture that makes that ability easy to express reliably.

The strongest evidence for that is:
- old good scripts and seeded patterns still exist
- scene-rich instructions still exist
- recent weaker scripts are not shorter
- the live loop still underuses prompt enrichment, structural pattern reuse, and aesthetic repair semantics

If you want the agent to recover its older mood and object quality and then surpass it, the right move is not "demand 1000-line scripts." The right move is to make scene look-dev an explicit, reusable, enforceable artifact in the pipeline, and to stop forcing visual craft through parameter-only repair channels.
