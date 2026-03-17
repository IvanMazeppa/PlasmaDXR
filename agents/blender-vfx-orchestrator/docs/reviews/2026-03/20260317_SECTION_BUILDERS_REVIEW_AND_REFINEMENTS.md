# Section Builders: Review, Concerns, and Refinements

Status: review
Date: 2026-03-17
Reviewer: Claude (Opus 4.6)
Reviews:
- `20260317_SPECIALIZED_SECTION_BUILDERS_DESIGN_NOTE.md`
- `20260317_SPECIALIZED_SECTION_BUILDERS_IMPLEMENTATION_PLAN.md`

## Overall Assessment

These are strong documents. The architecture is sound, the rollout strategy is disciplined, and the core insight — that the single Script Writer is trying to solve too many jobs in one pass — is correct. GPT-5.4-xhigh made unusually good sequencing decisions: scaffold first, one specialist, prove it, then expand.

This review identifies specific concerns, refinements, and additions that should be resolved before implementation begins.

---

## Concern 1: The `ctx` Dict Will Cause Assembly Bugs

### Problem

The scaffold uses `def setup_scene(ctx):` with a plain Python dict. This will cause name collisions, missing key errors, and implicit coupling between builders. A dict has no schema — builders will write keys the assembler doesn't expect, or read keys that another builder hasn't created yet because of ordering.

Example failure: `MaterialsLookDevBuilder` tries `ctx["hero_glass"]` but `HeroAssetBuilder` wrote it as `ctx["wine_glass_main"]`. Nothing catches this until Blender execution fails.

### Recommendation

Replace the `ctx` dict with the typed `BuildRegistry` from day one. The overhead is a single Pydantic model. The payoff is compile-time detection of registry mismatches.

```python
def create_geometry(registry: BuildRegistry) -> None:
    obj = create_wine_glass()
    registry.hero_objects["hero_glass"] = obj.name
    # Type-safe, discoverable, serializable
```

The assembler should validate registry completeness after each builder runs:
- hero objects declared in `SceneBuildPlan.must_create` all exist in `registry.hero_objects`
- materials referencing objects in `registry.hero_objects` can find them
- camera referencing focus objects can resolve them

### Impact

Trivial implementation cost. Prevents an entire class of integration bugs. Should be a Phase 1 requirement, not deferred.

---

## Concern 2: The Director Should Be Deterministic Initially

### Problem

The design note says the Director "remains the top-level authorial brain" and owns scene intent, composition priorities, and build order. The implementation plan adds a `plan_scene_build()` step but doesn't specify whether it's an LLM agent or a deterministic function.

Adding an LLM Director agent costs ~$0.03-0.05/run, adds 15-20 seconds of latency, and introduces a new point of ambiguity. Given our experience with coordinators (Modification Coordinator, Quality Gate) sometimes making worse decisions than deterministic routing, this is a real risk.

### Recommendation

Make the Director a **deterministic function** for Phases 0-4. The `SceneBuildPlan` can be computed from existing data without an LLM call:

```python
def plan_scene_build(
    style_spec: StyleSpec,
    technique_contract: TechniqueContract,
    specialist_mode: str,  # "off" | "hero_only" | "hero_env" | "full"
) -> SceneBuildPlan:
    assignments = []

    if specialist_mode in ("hero_only", "hero_env", "full"):
        assignments.append(SectionAssignment(
            builder_kind="hero_asset",
            owned_section="create_geometry",
            objectives=[f"Build hero: {h}" for h in style_spec.hero_objects],
            must_create=style_spec.hero_objects,
        ))
    # All unassigned sections default to single_writer
    ...
```

Why this works:
- The hero object list comes from `StyleSpec` (already deterministic, computed by `extract_style_spec()`)
- The section → builder mapping is static by design (hero → geometry, environment → scene/lighting, materials → materials)
- The dependencies are also static (materials depends on geometry, camera depends on geometry)

Only promote the Director to an LLM agent when there's evidence that deterministic assignment is making wrong choices (e.g., assigning hero builder to a scene that doesn't need hero objects). That evidence should be collected during Phase 4 benchmarking.

### Impact

Saves $0.03-0.05 per run, 15-20 seconds latency, and removes a coordination failure mode. Can always be upgraded to an LLM agent later.

---

## Concern 3: Generate-Then-Replace Wastes Tokens

### Problem

Phase 4 says: "keep the single writer generating the full script, then replace only `create_geometry()` with specialist output." This means you pay for the single writer to generate a full `create_geometry()` implementation that gets immediately thrown away. At gpt-5.4 pricing, that's wasted output tokens and wasted reasoning budget.

### Recommendation

Give the single writer explicit instructions to emit a **stub** for specialist-owned sections:

```python
def create_geometry(ctx):
    # SPECIALIST_PLACEHOLDER: HeroAssetBuilder will replace this section
    # Hero objects needed: wine_glass, candle, table
    pass
```

Then the specialist fills in the real code. Benefits:
- Cheaper (fewer output tokens from single writer)
- Faster (single writer focuses on its owned sections)
- Cleaner (no merge conflicts between writer and specialist geometry code)
- Testable (scaffold with stubs compiles and runs — just produces no geometry)

The stub should include a comment listing what the specialist is expected to create, derived from the `SceneBuildPlan`. This gives the assembler a verification point.

### Impact

Moderate token savings per run. More importantly, eliminates the risk of the single writer and specialist producing conflicting geometry that the assembler has to reconcile.

---

## Concern 4: Specialists Need Full Doc Bundle + Truth Pack

### Problem

The base factory gives specialists `semantic_search_blender_docs` and `search_blender_api_by_intent`, but NOT `blender_doc_search_bundle` (the bundled search that includes API + manual + patterns in one call). Specialists also don't get explicit truth pack validation on their output.

The candle wine glass E2E test just produced 11 instances of `Fac` → `Factor` hallucinations. With specialists, this would occur in the `MaterialsLookDevBuilder` output — and if truth pack validation doesn't run on specialist output before assembly, the errors propagate into the assembled script.

### Recommendation

1. Give all specialists `blender_doc_search_bundle` as their primary doc search tool (instead of the lower-level individual tools).
2. Run truth pack validation on each `SectionBuildOutput.code` BEFORE assembly — not just on the final assembled script.
3. Add `blender_doc_search_bundle` and `validate_blender_script` to the base factory tool list.

```python
def create_section_builder(name, instructions, owned_sections) -> Agent:
    return Agent(
        name=name,
        instructions=instructions,
        tools=[
            blender_doc_search_bundle,     # NOT individual search tools
            search_code_patterns,
            get_pattern_code,
            patch_script_section,
            validate_blender_script,       # Truth pack validation
        ],
        ...
    )
```

### Impact

Prevents hallucinated API calls from surviving into assembled scripts. Catches errors at the builder level where they're cheaper to fix.

---

## Concern 5: Section Dependencies Are Simpler Than a DAG

### Problem

Phase 9 proposes a DAG scheduler with `BuildNode` and dependency tracking. This adds significant complexity for what is actually a simple dependency graph.

### Recommendation

The real dependency graph between canonical sections is almost entirely linear:

```
create_geometry (hero objects exist)
    ↓
setup_materials (needs object names from registry)
    ↓
setup_physics (needs objects + materials)

setup_scene (independent — world, background, collections)
setup_lighting (needs objects for framing, but can draft without)
setup_camera (needs objects for framing/focus)
```

The practical parallelism opportunities are:
1. `create_geometry` and `setup_scene` can run in parallel (truly independent)
2. `setup_materials` and `setup_lighting` can run in parallel (both depend on geometry, but not each other)
3. `setup_camera` depends on geometry being finalized

This doesn't need a DAG scheduler. It needs `asyncio.gather()` with two parallel groups:

```python
# Group 1: independent
geometry_out, scene_out = await asyncio.gather(
    run_hero_asset_builder(registry, plan),
    run_environment_builder(registry, plan),
)
registry.merge(geometry_out.registry_updates)
registry.merge(scene_out.registry_updates)

# Group 2: depends on geometry
materials_out, lighting_out = await asyncio.gather(
    run_materials_builder(registry, plan),
    run_lighting_builder(registry, plan),
)
```

### Impact

Much simpler implementation. Same performance benefit. The DAG scheduler can be added later if the dependency graph actually becomes complex (e.g., sub-section dependencies, conditional builders).

---

## Concern 6: Missing — How Specialists Interact with Repair Routing

### Problem

The implementation plan covers repair routing in Phase 7 but doesn't address a critical question: when a specialist-owned section fails, does the repair loop call the same specialist or the single writer?

This matters because:
- If the specialist produced bad geometry, sending the same specialist the same inputs may produce the same bad geometry
- The single writer doesn't know about section ownership and may overwrite adjacent sections
- Escape velocity (technique switching) currently resets the entire script — does it also reset specialist assignments?

### Recommendation

Define a clear repair escalation path:

```
specialist repair (same builder, modified inputs)
    ↓ (if same issue 2x)
single writer fallback for that section
    ↓ (if still failing)
full script rewrite (existing escape velocity path)
```

Add a `builder_repair_count` to session state per section:

```python
class SessionState(BaseModel):
    ...
    section_repair_counts: Dict[str, int] = Field(default_factory=dict)
```

When `section_repair_counts["create_geometry"] >= 2`, demote to single writer for that section. This preserves escape velocity at the section level without losing the whole specialist investment.

### Impact

Without this, specialist failures will either loop forever or fall back too aggressively. This is the repair-routing analog of the escape velocity system we already have.

---

## Concern 7: Asset Lab Is the Compounding Value — Keep It Visible

### Problem

Asset Lab mode is deferred to Phase 8, which is correct for rollout safety. But the documents treat it as a nice-to-have extension rather than the eventual primary value driver.

### Why This Matters

Every other improvement in this plan is per-run. A better hero builder makes THIS wine glass scene better. Asset Lab means a good wine glass, once built and promoted, makes EVERY future wine glass scene better. That's where the system gets compounding returns on LLM spend.

The current pipeline already has the seeds of this:
- `code_pattern_tools.py` stores and retrieves code patterns
- `search_code_patterns` and `get_pattern_code` are exposed to the Script Writer
- The knowledge base stores technique outcomes

Asset Lab would turn these from incidental byproducts into intentional outputs.

### Recommendation

Don't defer Asset Lab thinking to Phase 8. Instead, design the `SectionBuildOutput` and `BuildRegistry` models NOW (Phases 1-2) with Asset Lab promotion in mind:

```python
class SectionBuildOutput(BaseModel):
    section_name: str
    code: str
    created_objects: list[str] = Field(default_factory=list)
    created_materials: list[str] = Field(default_factory=list)
    registry_updates: dict[str, str] = Field(default_factory=dict)
    # Asset Lab fields — populated even in scene mode, used for promotion
    asset_reuse_candidates: list[str] = Field(default_factory=list)
    standalone_test_passed: bool = False
```

The `asset_reuse_candidates` field costs nothing to add now but makes Phase 8 trivial: any builder output with `standalone_test_passed=True` is a promotion candidate.

Similarly, design the `BuildRegistry` to support asset library lookup from the start:

```python
class BuildRegistry(BaseModel):
    hero_objects: dict[str, str] = Field(default_factory=dict)
    materials: dict[str, str] = Field(default_factory=dict)
    # Asset library references — resolved before builder runs
    prebuilt_assets: dict[str, str] = Field(default_factory=dict)
```

A builder seeing `registry.prebuilt_assets["wine_glass"] = "assets/wine_glass_v3.py"` can import it instead of building from scratch. This is the reuse path.

### Impact

No extra implementation cost in Phases 0-4. Makes Phase 8 a natural extension rather than a retrofit.

---

## Concern 8: The Scaffold Generator Should Use the Existing Section Parser

### Problem

Phase 1 proposes a `render_base_scaffold()` function that generates a hardcoded stub. But `utils/script_sections.py` already has section detection and replacement logic. Creating a second source of truth for "what are the canonical sections" will cause drift.

### Recommendation

Define `CANONICAL_SECTION_ORDER` in `utils/script_sections.py` (if it's not already there) and have the scaffold generator derive from it:

```python
# In utils/script_sections.py
CANONICAL_SECTIONS = [
    "setup_scene",
    "create_geometry",
    "setup_materials",
    "setup_physics",
    "setup_lighting",
    "setup_camera",
    "bake_and_render",
]

def render_base_scaffold(
    plan: Optional[SceneBuildPlan] = None,
) -> str:
    """Generate a scaffold with stubs for all canonical sections."""
    lines = ["import bpy", "import math", ""]
    for section in CANONICAL_SECTIONS:
        owner = _get_owner(plan, section) if plan else "single_writer"
        lines.append(f"def {section}(registry):")
        lines.append(f"    # SECTION_OWNER: {owner}")
        lines.append(f"    raise NotImplementedError")
        lines.append("")
    # Main execution block
    lines.append("registry = BuildRegistry()")
    for section in CANONICAL_SECTIONS:
        lines.append(f"{section}(registry)")
    return "\n".join(lines)
```

### Impact

Single source of truth for canonical sections. Scaffold generator, section parser, and section patcher all derive from the same list.

---

## Idea 1: Section-Level Quality Evaluation

### Opportunity

The design note mentions "better evaluation" as an advantage but doesn't specify how. Currently, the Quality Analyst evaluates the entire render. With section ownership, you could also evaluate per-section:

- `create_geometry` → geometry quality (silhouette, detail level, modifier count)
- `setup_materials` → material quality (shader complexity, roughness variation, transmission)
- `setup_lighting` → lighting quality (key/fill/rim separation, practical count, mood coherence)
- `setup_scene` → environment quality (world strategy, atmosphere, context)

This would let you route repair feedback to the exact builder that needs it, with section-specific quality criteria.

### Recommendation

Defer implementation but design the `QualityIssue.target` field to accept canonical section names. The aesthetic realism work already started this — `IssueKind` now includes `materials`, `environment`, `composition`. The next step is mapping these to section names:

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

This mapping already exists informally in the implementation plan (Phase 6, `SPECIALIST_SECTION_BY_ISSUE`). Formalize it early.

---

## Idea 2: Builder Versioning for A/B Comparison

### Opportunity

Once specialists exist, you'll want to compare them against each other and against the single writer. The implementation plan mentions benchmarking but doesn't specify how to track which builder version produced which output.

### Recommendation

Add a `builder_version` field to `SectionBuildOutput`:

```python
class SectionBuildOutput(BaseModel):
    section_name: str
    code: str
    builder_name: str = ""
    builder_version: str = ""  # e.g., "hero_asset_v1", "single_writer_baseline"
    ...
```

Store this in the iteration manifest (Phase 4 of the CURRENT_ROADMAP). Then benchmark analysis can compare:
- `hero_asset_v1` vs `single_writer` on `create_geometry` quality
- `environment_atmosphere_v1` vs `single_writer` on scene richness

### Impact

Minimal implementation cost. Enables the evidence-based promotion that CURRENT_ROADMAP Phase 5 requires.

---

## Idea 3: Specialist Instructions Should Reference StyleSpec Fields Directly

### Opportunity

The specialist prompt shapes in the implementation plan are generic ("create the hero object(s) and support geometry"). They should reference the specific `StyleSpec` fields that are relevant to their section.

### Recommendation

Inject StyleSpec fields directly into specialist prompts:

```python
def build_hero_prompt(style_spec: StyleSpec, plan: SceneBuildPlan) -> str:
    return f"""You own ONLY the create_geometry() section.

## Hero Objects (from StyleSpec)
{', '.join(style_spec.hero_objects)}

## Material Goals (your geometry must support these)
{', '.join(style_spec.hero_material_goals)}

## Camera Context (affects detail level)
Distance: {style_spec.camera_distance_class}
DOF: {style_spec.dof_intent}

Close-up shots need more geometric detail (subdivision, bevels, thickness).
Wide shots can use simpler geometry."""
```

This grounds the specialist in the specific creative brief rather than asking it to infer intent from a general description.

---

## Idea 4: The Assembler Should Validate Cross-Section References

### Opportunity

After assembly, the script may contain references that cross section boundaries — e.g., `setup_materials` references an object created in `create_geometry`. If the names don't match, the script fails at runtime.

### Recommendation

Add a lightweight post-assembly validation pass:

```python
def validate_cross_references(
    assembled_script: str,
    registry: BuildRegistry,
) -> list[str]:
    """Check that all registry entries are referenced in the script."""
    warnings = []
    for logical_name, blender_name in registry.hero_objects.items():
        if blender_name not in assembled_script:
            warnings.append(
                f"Hero object '{logical_name}' ({blender_name}) created in registry "
                f"but never referenced in assembled script"
            )
    return warnings
```

This catches the most common assembly bug (name mismatches) before Blender execution.

---

## Summary of Recommendations

### Must-Do Before Implementation

| # | Recommendation | Phase | Effort |
|---|---------------|-------|--------|
| 1 | Replace `ctx` dict with typed `BuildRegistry` | Phase 1 | Low |
| 2 | Make Director deterministic initially | Phase 2 | Low |
| 3 | Use stub sections instead of generate-then-replace | Phase 4 | Low |
| 4 | Give specialists `blender_doc_search_bundle` + truth pack | Phase 3 | Low |

### Should-Do (High Value, Low Risk)

| # | Recommendation | Phase | Effort |
|---|---------------|-------|--------|
| 5 | Use `asyncio.gather()` instead of DAG scheduler | Phase 9→5 | Low |
| 6 | Define specialist repair escalation path | Phase 7 | Medium |
| 7 | Design models with Asset Lab fields from day one | Phase 1-2 | Trivial |
| 8 | Derive scaffold from existing section parser | Phase 1 | Low |

### Good Ideas to Include

| # | Idea | Phase | Effort |
|---|------|-------|--------|
| 9 | Section-level quality evaluation mapping | Phase 2 | Low |
| 10 | Builder versioning for A/B comparison | Phase 0 | Trivial |
| 11 | StyleSpec-grounded specialist prompts | Phase 4 | Low |
| 12 | Post-assembly cross-reference validation | Phase 3 | Low |

---

## Conclusion

The section-builder architecture is the right direction. The documents are unusually well-structured for a first proposal. The refinements above are about de-risking the implementation, not changing the architecture.

The single most important insight in these documents is: **specialization should happen over a deterministic scaffold, not as free-form multi-agent writing.** That constraint makes everything else tractable — patching, repair routing, benchmarking, and eventually Asset Lab.

If the hero-first milestone (Phases 0-4) succeeds, this becomes one of the highest-leverage changes in the project. If it doesn't, the scaffold and registry work still improves script quality and repairability even without specialists.
