# Worklog: Grey Room Template Fix (2026-02-25)

## Problem

Every E2E test render — regardless of input prompt — produced the **same grey room**: concrete walls, concrete floor, floating ceiling panel, generic workbench/shelf props, overhead fluorescent-style lighting. This happened across 4 consecutive renders spanning two different prompts (fire AND wine pour).

### Evidence

| Test | Prompt | Expected | Got |
|------|--------|----------|-----|
| phase2a_e2e_fire | "Small orange fire plume" | Fire in some setting | Grey room, no visible fire |
| fire_decay_test_v2 (iter 1) | Same | Fire | Grey room, workshop furniture, no fire |
| fire_decay_test_v2 (iter 2) | Same | Fire | Grey room, brick cylinder, gaps |
| wine_pour_decay_v1 | 25-line detailed prompt: "warm amber lighting, dark wood, cozy evening" | Wine glass on dark wood, warm candlelight | Same grey room, concrete walls, overhead strip light |

Score: 0.0 (fire), 55.0 (wine pour — only because liquid sim existed).

### Root Cause Analysis

**Two compounding bugs:**

#### Bug 1: Dynamic instructions acted as a rigid template

`tools/dynamic_instructions.py` lines 349-443 contained a "SCENE DESIGN REQUIREMENTS" section that described **one specific room layout** in MANDATORY language:

```
MINIMUM for indoor scenes:
- Floor with appropriate material (tile, wood, concrete)    ← LLM always picked concrete
- 2-3 wall surfaces (back wall + at least one side wall)   ← same BackWall + SideWall every time
- Ceiling or overhead element (even partial)                ← same floating panel
- 3-5 CONTEXT PROPS that belong in the described location  ← workbench/shelf/pipe
```

The section included grey-toned example materials (`tile_color=(0.82, 0.78, 0.72)`, `grout_color=(0.3, 0.28, 0.25)`) and room-type examples (Workshop, Basement, Lab) that the LLM followed literally, ignoring the user prompt's creative direction entirely.

The instructions said "MANDATORY" and "REQUIRED" — the LLM obeyed those markers over the prompt's "warm amber lighting on dark wood."

#### Bug 2: Camera distance validator pushed cameras outside rooms

`tools/blender_api_fixer.py` computed the bounding box of ALL mesh objects (including 6m walls and floors), then checked if the camera was "inside" that box. A camera placed inside a room (normal for interior shots) triggered `_inside = True` and got pushed to 7.47m — far outside the room.

The wine_pour prompt asked for "close-up, slightly low angle" at 1.55m from subject. The validator overrode this to 7.47m, turning a close-up into a distant establishing shot where the wine glass was barely visible.

## Fixes Applied

### Fix 1: Prompt-responsive scene design instructions

Rewrote lines 349-510 of `tools/dynamic_instructions.py`:

**Key changes:**
- Added **"RULE #1: THE PROMPT OVERRIDES EVERYTHING BELOW"** at the top of the section
- Instructions now say to EXTRACT mood, colors, materials, setting from the prompt
- Removed all grey example material code (tile, wood material function templates)
- Added distinct structural minimums for room-scale vs close-up scenes: *"A close-up of a wine glass does NOT need 4 walls and a ceiling"*
- Lighting section now references prompt mood: warm/cozy → amber tones, clinical → cool tones
- World background matched to prompt mood instead of hardcoded near-black
- Kept structural minimums (8+ objects, 5+ materials, 3+ lights) as floors, not templates

**Validation:** wine_pour_v2 generated: candles, dark wood table, warm backdrop, napkin, coaster, bottle silhouette — zero concrete, zero grey, zero BackWall/SideWall. Scene matched the prompt's creative direction.

### Fix 2: Camera distance validator excludes room geometry

Rewrote `CAMERA_DISTANCE_FIX_SNIPPET` in `tools/blender_api_fixer.py`:

**Key changes:**
- Filters out objects with any world-space dimension >2.5m (walls, floors, ceilings)
- Filters out objects with environment names (floor, wall, ceiling, ground, room, backdrop)
- Computes bounding box from SUBJECT objects only
- Falls back to all objects if no subject objects qualify
- Logs "Camera OK" when no repositioning needed (aids debugging)

**Result:** Camera at 1.55m for a close-up stays at 1.55m instead of being pushed to 7.47m.

### Fix 3: Code pattern memory seeded with proven techniques

Created `scripts/seed_code_patterns.py` — extracts 7 techniques from the successful wine_pour_v1 script (the one that produced near-photorealistic renders):

| Pattern | What It Does | Why It Matters |
|---------|-------------|----------------|
| glass_mesh_solidify_subdivision | Solidify + SubdivisionSurface modifiers on glass | Transforms faceted chalice into smooth, realistic wine glass |
| glass_material_principled_transmission | Principled BSDF, Transmission=1.0, IOR=1.47 | Clear glass with proper Fresnel reflections |
| glass_bottom_collision_disc | Hidden disc at bowl-stem junction | Prevents liquid leaking through hollow stem |
| liquid_mesh_quality_smoothing | mesh_smoothen_pos/neg=4, mesh_concave_upper=3.0 | Smooth liquid surface instead of blobby chunks |
| smooth_shading_all_meshes | use_smooth=True on all visible geometry | Eliminates faceting on ALL objects |
| filmic_color_management | Filmic + Medium High Contrast + exposure 0.25 | Cinematic tone mapping, better dynamic range |
| subdivision_surface_smooth_objects | SubdivisionSurface level 2 (ALL effect types) | Smooth curves on any object, not just glass |

These are the first patterns in the code pattern store. The Learning Agent can now retrieve them during iteration when quality feedback mentions faceted surfaces, blobby liquid, washed-out colors, etc.

## Before/After Comparison

### Scene design (same wine pour prompt)

| Aspect | v1 (old instructions) | v2 (new instructions) |
|--------|----------------------|----------------------|
| Floor | `mat_concrete` grey | `mat_wood` dark wood |
| Walls | BackWall + SideWall concrete | Soft dark backdrop |
| Ceiling | Floating grey panel | None (close-up scene) |
| Props | Shelf, metal pipe | Coaster, napkin, plate, bottle, candle holders |
| Lights | Generic overhead AREA | Warm key (fireplace), dim fill, candle practicals |
| Materials | `make_concrete()` | `material_dark_wood()`, wax, warm emissive |
| Theme | Industrial workshop | "Warm, intimate darkness" |

### Mesh quality gap (still to address)

| Aspect | v1 earlier (successful) | v2 current |
|--------|------------------------|------------|
| Glass mesh | bmesh spin + Solidify + SubdivisionSurface = smooth curves | Raw bmesh shell = faceted chalice |
| Glass material | Transmission=1.0, IOR=1.47 = realistic glass | Similar but less refined |
| Liquid mesh | mesh_smoothen=4, concave_upper=3.0 = smooth surface | Default settings = blobby |

The code pattern seed addresses this gap — the Learning Agent can now retrieve these techniques. The real test is whether the agent actually uses them during iteration.

## Remaining Issues

1. **`use_nodes` deprecation warnings treated as execution failures** — Blender 5.0 prints a warning but the line still works. The pipeline error classifier needs to distinguish warnings from errors.
2. **`keyframe_insert("flow_settings.use_inflow")` not found** — script tried to keyframe a property that doesn't support it. Needs truth pack pattern or doc search coverage.
3. **Code pattern retrieval ranking** — keyword overlap produces noisy results (glass_bottom_collision_disc matches "washed out colors"). Semantic search would improve precision.
4. **Prompt enhancement pipeline step (2B-9)** — still not implemented. Even with better instructions, vague prompts will produce mediocre results. The wine_pour prompt is 25 lines; most user prompts will be 1-2 lines.

## Files Changed

| File | Change |
|------|--------|
| `tools/dynamic_instructions.py` | Rewrote scene design section (lines 349-510) |
| `tools/blender_api_fixer.py` | Rewrote camera distance validator snippet |
| `scripts/seed_code_patterns.py` | New: seeds 7 code patterns from wine_pour_v1 |
| `data/code_patterns/*.json` | New: 7 pattern files + index |
| `docs/PHASE2_ROADMAP.md` | Added 2B-9 (Prompt Enhancement Pipeline Step) |
