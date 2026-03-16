# Scene Aesthetic Realism Action Plan

Status: active plan
Date: 2026-03-16
Companion to: `20260316_SCENE_AESTHETIC_REALISM_DEEP_DIVE.md`

## Purpose

This document converts the deep-dive analysis into an ordered implementation plan.

The goal is not "make scripts longer." The goal is to make scene look-dev, hero-object quality, mood, and material realism first-class in the live pipeline without collapsing into templates.

The recommended order is:
1. front-load aesthetic intent
2. make visual craft reusable on the hot path
3. add a deterministic look-dev floor
4. make QA and repair routing understand aesthetic failures
5. replace crude richness proxies with better metrics

## Success Criteria

The work below is successful when the system can reliably improve these dimensions:
- better material readability in close-ups
- stronger mood coherence between prompt and scene
- better hero-object refinement for glass, cloth, bricks, windows, vessels
- less "grey room / blockout / previs" feedback
- better aesthetic repair after failure, not just parameter nudging

Concrete metrics to track:
- percent of scripts with explicit color management
- percent of scripts with explicit DOF decision
- percent of close-up prompts whose hero object gets a refinement pass
- percent of failing QA outputs that emit non-parameter aesthetic issue kinds
- aesthetic score deltas on a fixed benchmark pack

## Recommended Execution Order

## Phase 0: Baseline and Guardrails for the Work

Do this first so the rest of the changes are measurable.

### Task 0.1: Create a small aesthetic benchmark pack

Create 8-12 prompts that stress scene realism:
- intimate tabletop glass / bottle / ceramic
- derelict industrial window shatter
- candlelit macro scene
- outdoor cloth / flag
- polished product-like close-up
- soft atmospheric interior

Store them under a stable doc or fixture file:
- `docs/current/AESTHETIC_BENCHMARK_SET.md`
- or `tests/fixtures/aesthetic_benchmarks.json`

Include for each:
- bare prompt
- enhanced prompt version
- expected scene read
- expected hero objects
- expected look-dev cues

### Task 0.2: Add coverage metrics for visual craft

File targets:
- `tools/script_analysis_tools.py`
- or create `tools/lookdev_metrics.py`

Add a deterministic feature extractor like:

```python
from dataclasses import dataclass

@dataclass
class LookDevCoverage:
    has_explicit_color_management: bool
    has_world_strategy: bool
    has_dof_decision: bool
    practical_light_count: int
    distinct_material_count: int
    hero_object_refinement_count: int
    atmosphere_volume_present: bool
    environment_texture_present: bool
```

and a collector:

```python
def analyze_lookdev_coverage(script_text: str) -> LookDevCoverage:
    return LookDevCoverage(
        has_explicit_color_management=("view_transform" in script_text),
        has_world_strategy=("Background" in script_text or "ShaderNodeTexEnvironment" in script_text),
        has_dof_decision=("focus_object" in script_text or "aperture_fstop" in script_text or "use_dof" in script_text),
        practical_light_count=script_text.lower().count("candle") + script_text.lower().count("practical"),
        distinct_material_count=script_text.count("bpy.data.materials.new("),
        hero_object_refinement_count=script_text.count("SOLIDIFY") + script_text.count("SUBSURF") + script_text.count("BEVEL"),
        atmosphere_volume_present=("VolumeAbsorption" in script_text or "ShaderNodeVolumePrincipled" in script_text),
        environment_texture_present=("ShaderNodeTexEnvironment" in script_text),
    )
```

This should not be the final metric system. It is a first pass so you can quickly see whether new changes are actually affecting script content.

### Exit criteria

- benchmark prompts are committed
- at least one metric extractor exists and can be run over a directory of generated scripts

## Phase 1: Put Aesthetic Intent on the Hot Path

This is the highest-leverage phase.

### Task 1.1: Add `StyleSpec` to the schema layer

File targets:
- `models/pipeline_models.py`
- `models/shared_context.py`

Add a new artifact next to `TechniqueContract`, not instead of it:

```python
class StyleSpec(BaseModel):
    setting: str = ""
    mood: str = ""
    palette: List[str] = Field(default_factory=list)
    camera_framing: str = ""
    camera_distance_class: str = ""
    dof_intent: str = ""
    world_lighting_mode: str = ""
    hero_objects: List[str] = Field(default_factory=list)
    hero_material_goals: List[str] = Field(default_factory=list)
    support_prop_budget: str = "minimal"
    realism_target: str = "cinematic"

    def to_script_constraints(self) -> str:
        return "\n".join([
            "## STYLE SPEC (BINDING CREATIVE BRIEF)",
            f"Setting: {self.setting}",
            f"Mood: {self.mood}",
            f"Palette: {', '.join(self.palette)}",
            f"Camera: {self.camera_framing} | distance={self.camera_distance_class} | dof={self.dof_intent}",
            f"World Lighting Mode: {self.world_lighting_mode}",
            f"Hero Objects: {', '.join(self.hero_objects)}",
            f"Hero Material Goals: {', '.join(self.hero_material_goals)}",
            f"Support Prop Budget: {self.support_prop_budget}",
            f"Realism Target: {self.realism_target}",
        ])
```

Then carry it through session state:

```python
class AssetRequest(BaseModel):
    ...
    enhanced_description: Optional[str] = None
    style_spec: Optional[Dict[str, Any]] = None

class SessionState(BaseModel):
    ...
    style_spec: Optional[Dict[str, Any]] = Field(default=None)
    enhanced_prompt: str = Field(default="")

class SharedContext(BaseModel):
    ...
    style_spec: Optional[Dict[str, Any]] = Field(default=None)
```

### Task 1.2: Add prompt enhancement into the runtime

File targets:
- create `utils/prompt_enhancer.py`
- update `orchestrator.py`

Do not wait for a full HITL integration to get value. Add a first-pass internal enrichment stage before research.

Suggested implementation:

```python
from pydantic import BaseModel, Field
from typing import List

class PromptEnhancementOutput(BaseModel):
    enhanced_description: str
    style_spec: StyleSpec
    hero_objects: List[str] = Field(default_factory=list)
    material_cues: List[str] = Field(default_factory=list)
```

Then a helper:

```python
async def enhance_prompt(
    orch: "BlenderVFXOrchestrator",
    request: AssetRequest,
    context: SharedContext,
    sdk_session,
) -> PromptEnhancementOutput:
    prompt = f"""
Expand this VFX request into a dense, scene-building creative brief.

Original description:
{request.description}

You must:
1. preserve user intent
2. infer mood, palette, shot framing, lighting intent, material cues
3. identify hero objects that need refinement
4. produce both enhanced_description and structured StyleSpec
"""
    result = await orch._run_agent(
        orch._research_agent,
        prompt,
        context=context,
        session=sdk_session,
        max_turns=4,
    )
    return result.final_output
```

In `orchestrator.py`, run this before research, then persist:

```python
enhanced = await enhance_prompt(self, request, context, sdk_session)
request.description = enhanced.enhanced_description
request.enhanced_description = enhanced.enhanced_description
request.style_spec = enhanced.style_spec.model_dump()
session.enhanced_prompt = enhanced.enhanced_description
session.style_spec = enhanced.style_spec.model_dump()
context.style_spec = enhanced.style_spec.model_dump()
```

### Task 1.3: Inject `StyleSpec` into script generation

File targets:
- `orchestrator.py`
- `tools/dynamic_instructions.py`

Where `TechniqueContract` is already inserted, add `StyleSpec`:

```python
style_guidance = ""
if session.style_spec:
    style_guidance = StyleSpec(**session.style_spec).to_script_constraints()

script_prompt = f"""Generate a Blender Python script for {request.effect_type.value} VFX.
{technique_guidance}
{style_guidance}

## Research Findings
{research_text[:1500]}
...
"""
```

Also add dynamic-instruction support so the writer sees this as a binding artifact rather than loose prose.

### Exit criteria

- live pipeline stores enhanced prompt
- `StyleSpec` exists in session/context
- script prompts include both `TechniqueContract` and `StyleSpec`

## Phase 2: Make Visual Craft Reusable During Generation

This phase moves strong aesthetic knowledge onto the first-draft path.

### Task 2.1: Expose visual pattern retrieval to the Script Writer

File target:
- `specialized_agents/script_writer.py`

Add these tools to the writer:

```python
from tools.code_pattern_tools import (
    search_code_patterns,
    get_pattern_code,
)
```

and include them:

```python
tools=[
    write_script,
    validate_script,
    modify_script,
    search_code_patterns,
    get_pattern_code,
    semantic_search_blender_docs,
    search_blender_api_by_intent,
]
```

Then strengthen instructions:

```python
## VISUAL CRAFT REUSE
Before inventing a hero object or signature look from scratch, check for relevant code patterns.

Use search_code_patterns() for:
- glass realism
- candlelit scenes
- industrial materials
- DOF setups
- practical lights
- atmosphere depth

If a pattern is structural, retrieve it with get_pattern_code() and adapt it in your generated code.
Do NOT reduce structural patterns to scalar parameter edits.
```

### Task 2.2: Add visual pattern tagging

File targets:
- `utils/code_pattern_memory.py`
- `scripts/seed_code_patterns.py`

Add optional tags:

```python
@dataclass
class CodePattern:
    ...
    tags: List[str] = field(default_factory=list)
```

Seed general visual patterns with tags like:
- `lookdev/color_management`
- `hero_object/glass`
- `camera/dof`
- `lighting/practicals`
- `materials/wood`
- `materials/aged_metal`

Example:

```python
pattern = CodePattern(
    ...,
    tags=["hero_object/glass", "lookdev/refraction", "cross_effect"],
)
```

Update retrieval to score tag matches:

```python
if requested_tags:
    overlap = len(set(pattern.tags) & set(requested_tags))
    score += overlap * 15
```

### Task 2.3: Stop routing structural code patterns through scalar patch logic

File target:
- `orchestrator.py`

Current bad path:
- search pattern
- parse assignment lines
- feed to `_apply_script_modifications()`

Replace with a structural split:

```python
def is_structural_pattern(pattern: CodePattern) -> bool:
    structural_markers = [
        "modifiers.new(",
        "bpy.ops.mesh",
        "bpy.ops.object",
        "ShaderNode",
        "focus_object",
        "view_transform",
    ]
    return any(marker in pattern.code_snippet for marker in structural_markers)
```

Then:

```python
if is_structural_pattern(best_pattern):
    # route to writer with retrieved code context
    pattern_prompt = f"""
Use this proven pattern as a starting point when writing the script:

Pattern name: {best_pattern.name}
Pattern tags: {best_pattern.tags}
Pattern code:
{best_pattern.code_snippet}
"""
else:
    # scalar tweaks can still go through modify path
```

### Task 2.4: Add a `HeroObjectRefinement` helper layer

File target:
- create `tools/hero_object_refinement.py`

This should be deterministic and small, not a template system.

Example API:

```python
class HeroObjectRefinement(BaseModel):
    object_type: str
    required_modifiers: List[str]
    required_material_traits: List[str]
    quality_checks: List[str]
```

Starter rules:
- glass vessel -> `SOLIDIFY`, `SUBSURF`, low roughness, transmission, believable IOR
- masonry brick -> proper proportions, bevel or edge breakup, non-uniform color/roughness
- pane/window -> thickness, edge readability, specular response

This can feed either:
- initial generation prompt
- deterministic post-generation checks

### Exit criteria

- writer can retrieve visual code patterns on first draft
- structural patterns are no longer flattened into scalar param edits
- at least 5 visual pattern groups are tagged and retrievable

## Phase 3: Add a Deterministic Look-Dev Floor

This is the phase that prevents many blockout-like scripts from surviving first draft.

### Task 3.1: Add explicit color-management baseline

File target:
- create `tools/lookdev_floor.py`

Use Blender 5 guidance: prefer `AgX`, fallback safely if unavailable.

```python
COLOR_MANAGEMENT_SNIPPET = '''
try:
    transforms = scene.view_settings.bl_rna.properties['view_transform'].enum_items.keys()
    if 'AgX' in transforms:
        scene.view_settings.view_transform = 'AgX'
    elif 'Filmic' in transforms:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium High Contrast'
except Exception:
    pass
'''
```

Apply this only if the script does not already make an explicit color-management decision.

### Task 3.2: Require a camera/readability decision

Add a deterministic check:
- if close-up or tabletop cue is present in `StyleSpec`, the script must either:
  - enable DOF intentionally
  - or explicitly disable it with a reason in code comments / config block

Starter checker:

```python
def has_dof_decision(script_text: str) -> bool:
    return any(token in script_text for token in [
        "use_dof",
        "focus_object",
        "aperture_fstop",
        "dof.aperture",
        "DOF disabled intentionally",
    ])
```

### Task 3.3: Add hero-object finish requirements

If `StyleSpec.hero_objects` includes glass or similar close-up read objects, enforce a minimum finish pass.

Example:

```python
def hero_object_needs_refinement(style_spec: StyleSpec) -> bool:
    return any(obj in {"glass", "wine glass", "tumbler", "bottle", "window pane"} for obj in style_spec.hero_objects)

def script_has_glass_refinement(script_text: str) -> bool:
    return "SOLIDIFY" in script_text and "SUBSURF" in script_text and "Transmission" in script_text
```

### Task 3.4: Add world-lighting mode enforcement

Instead of only checking that a world exists, require a lighting strategy compatible with the style spec:
- `dark_world_plus_practicals`
- `procedural_sky`
- `env_texture`
- `studio_cards`

Starter enum:

```python
class WorldLightingMode(str, Enum):
    DARK_WORLD_PRACTICALS = "dark_world_practicals"
    PROCEDURAL_SKY = "procedural_sky"
    ENV_TEXTURE = "env_texture"
    STUDIO_CARDS = "studio_cards"
```

Then check the script for compatibility markers and inject guidance or fail softly if absent.

### Integration point

Hook this in right after `write_script` and before execution, similar to the API fixer path:
- `orchestrator.py`
- possibly next to Phase 1.5 / 1.9

### Exit criteria

- scripts consistently make explicit color-management decisions
- close-up shots consistently make explicit DOF decisions
- hero-object prompts consistently trigger hero-object finish checks

## Phase 4: Teach QA and Repair Routing About Aesthetic Failures

This is where you stop treating aesthetic failures as mostly parameter failures.

### Task 4.1: Expand `IssueKind`

File target:
- `models/pipeline_models.py`

Change:

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

### Task 4.2: Update Quality Analyst instructions

File targets:
- `orchestrator.py`
- `tools/dynamic_instructions.py`

When prompting the Quality Analyst, explicitly tell it to classify:
- poor material read -> `materials`
- weak setting / placeholder environment -> `environment`
- framing / empty frame / blocked focal event -> `composition`
- hero object looks primitive / faceted / unreadable -> `hero_object`
- missing tone mapping / flat specular / weak atmosphere / sterile preview look -> `lookdev`

Example prompt fragment:

```python
CRITICAL: Use these issue kinds when applicable:
- materials: surfaces, shading, refraction, texture realism
- environment: setting readability, contextual set dressing, world lighting integration
- composition: framing, empty space, occlusion of focal action
- hero_object: key object silhouette, finish, proportions, refinement
- lookdev: color management, tone mapping, specular shaping, atmosphere depth
```

### Task 4.3: Route these issue kinds to structural repair

File target:
- `phases/repair_routing.py`

Change the classifier:

```python
if kind in ("structural", "camera", "lighting", "materials", "environment", "composition", "hero_object", "lookdev"):
    structural += 1
```

This immediately prevents many aesthetic failures from getting trapped in parameter-only repair loops.

### Task 4.4: Allow bounded multi-section aesthetic patching

File target:
- `orchestrator.py`

If structured issues are mostly aesthetic, permit patching up to 3 sections in one repair pass:
- `setup_materials`
- `setup_lighting`
- `setup_camera`
- optionally `setup_scene`

Suggested logic:

```python
AESTHETIC_KINDS = {"materials", "environment", "composition", "hero_object", "lookdev"}

def is_aesthetic_repair(quality: QualityOutput) -> bool:
    return sum(1 for i in quality.structured_issues if i.kind in AESTHETIC_KINDS) >= 2
```

Then:

```python
max_sections_to_patch = 3 if is_aesthetic_repair(quality) else 2
```

### Task 4.5: Upgrade `qa_diagnosis_bridge`

File target:
- `tools/qa_diagnosis_bridge.py`

Stop treating it as a parameter mapper only. Add script-level feature checks:

```python
LOOKDEV_FEATURES = {
    "explicit_color_management": lambda txt: "view_transform" in txt,
    "dof_decision": lambda txt: "focus_object" in txt or "aperture_fstop" in txt,
    "hero_refinement": lambda txt: "SOLIDIFY" in txt or "SUBSURF" in txt or "BEVEL" in txt,
    "world_lighting_strategy": lambda txt: "Background" in txt or "ShaderNodeTexEnvironment" in txt,
}
```

Then include missing-feature diagnostics in the feedback output.

### Exit criteria

- failing QA outputs commonly emit non-parameter aesthetic issue kinds
- aesthetic failures route to `modify_code`
- section patching can intentionally update multiple look-dev sections

## Phase 5: Replace Line Count with Better Richness Proxies

### Task 5.1: Deprecate line count as primary richness heuristic

File target:
- `guardrails/tool_guardrails.py`

Keep line count as a soft warning only. Add a richer failure/warning path using look-dev coverage:

```python
coverage = analyze_lookdev_coverage(text)
if not coverage.has_explicit_color_management:
    warnings.append("No explicit color management")
if coverage.distinct_material_count < 4:
    warnings.append("Low material diversity")
if coverage.hero_object_refinement_count == 0 and is_closeup_prompt:
    warnings.append("No hero-object refinement")
```

### Task 5.2: Add a simple aesthetic score prior

Not a replacement for QA, just an early signal:

```python
def compute_script_aesthetic_prior(cov: LookDevCoverage) -> float:
    score = 0.0
    score += 10 if cov.has_explicit_color_management else 0
    score += 10 if cov.has_dof_decision else 0
    score += min(cov.distinct_material_count, 6) * 3
    score += min(cov.hero_object_refinement_count, 4) * 5
    score += 8 if cov.atmosphere_volume_present else 0
    return score
```

Use this only for monitoring and regression detection at first.

### Exit criteria

- the repo has richer script-quality heuristics than line count
- regressions in DOF/color-management/hero refinement become visible automatically

## Minimum Viable Route

If you want the shortest path to visible improvement, do these in order:

1. Add runtime prompt enhancement.
2. Add `StyleSpec`.
3. Expose `search_code_patterns` and `get_pattern_code` to the Script Writer.
4. Stop flattening structural patterns into scalar parameter edits.
5. Add explicit color-management baseline with Blender 5 `AgX` preference.
6. Expand `IssueKind` and route aesthetic issues to `modify_code`.

That subset should already improve:
- mood fidelity
- glass/object finish
- composition/readability
- quality of repair suggestions

## Validation Plan

Run the same benchmark scenes before and after each phase.

For each run, capture:
- render score
- structured issue kinds
- look-dev coverage metrics
- whether hero-object refinement is present
- whether prompt enhancement was used

Suggested table:

| Benchmark | Baseline Score | After P1 | After P2 | After P3 | After P4 | Notes |
|---|---:|---:|---:|---:|---:|---|
| tabletop wine glass |  |  |  |  |  |  |
| brick through window |  |  |  |  |  |  |
| candlelit desk |  |  |  |  |  |  |
| cloth in outdoor light |  |  |  |  |  |  |

## What Not to Do

- Do not respond by only raising the minimum script length.
- Do not solve this by adding giant prompt blobs without structured artifacts.
- Do not route hero-object refinement through scalar parameter edits.
- Do not make external HDRIs mandatory if that conflicts with the product vision.
- Do not treat `Filmic` as the long-term baseline on Blender 5 just because older scripts used it successfully.
- Do not make the writer memorize more examples while keeping the same narrow tool surface.

## Final Recommendation

Treat this as a missing-artifact and missing-hot-path problem, not a missing-creativity problem.

The system already demonstrated aesthetic capability. The job now is to:
- front-load aesthetic intent
- make proven scene craft structurally reusable
- add a small deterministic look-dev floor
- make repair routing understand aesthetic failure modes

That is the shortest path to recovering the older mood/object quality and then extending it beyond what the historical scripts managed.
