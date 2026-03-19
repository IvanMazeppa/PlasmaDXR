"""
Dynamic Instructions Module for Self-Learning VFX Orchestrator.

This module implements the SDK pattern of dynamic instructions - functions that
generate agent instructions at runtime based on accumulated knowledge from the
knowledge base.

KEY PRINCIPLE: No hardcoded rules. Rules emerge from experimentation and are
injected only when validated (success_rate > threshold).

Usage:
    from tools.dynamic_instructions import dynamic_script_writer_instructions

    agent = Agent[SharedContext](
        name="Script Writer",
        instructions=dynamic_script_writer_instructions,  # Function, not string!
        ...
    )
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Logger for dynamic instructions - makes KB failures visible
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper
    from models.shared_context import SharedContext

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
if str(ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_ROOT))

# Lazy import to avoid circular dependencies
_knowledge_base = None


def _get_knowledge_base():
    """Lazy import of knowledge base query function."""
    global _knowledge_base
    if _knowledge_base is None:
        try:
            from tools.experiment_tracker_tools import _query_knowledge_base_impl
            _knowledge_base = _query_knowledge_base_impl
        except ImportError:
            # Fallback: return empty results
            _knowledge_base = lambda q: json.dumps({"results": []})
    return _knowledge_base


def query_validated_learnings(
    effect_type: str,
    category: str = "physics",
    min_success_rate: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Query knowledge base for validated learnings.

    Only returns learnings that have been tested and have a success rate
    above the threshold. This prevents injecting untested rules.

    Args:
        effect_type: Type of effect (sun, explosion, fire, etc.)
        category: Category of learning (physics, visual, parameter)
        min_success_rate: Minimum success rate to include (0.0-1.0)

    Returns:
        List of validated learnings with their success rates
    """
    kb_query = _get_knowledge_base()

    try:
        # Phase 2B-7: Scope KB query by effect_type when enabled
        scope_effect_type = ""
        try:
            from config.agent_config import get_config
            if get_config().use_effect_type_scoping():
                scope_effect_type = effect_type
        except Exception:
            pass  # Config unavailable — no scoping

        result = kb_query(
            f"{category} rules for {effect_type}",
            effect_type=scope_effect_type,
        )
        data = json.loads(result) if isinstance(result, str) else result
        results = data.get("results", [])

        # Filter by success rate
        validated = []
        for learning in results:
            success_rate = learning.get("success_rate", 0)
            if success_rate >= min_success_rate:
                validated.append(learning)

        # Phase 2B-6: Filter by Ebbinghaus retention
        if validated:
            try:
                from config.agent_config import get_config
                if get_config().use_memory_decay():
                    from datetime import datetime
                    from utils.code_pattern_memory import compute_retention, RETENTION_ACTIVE
                    now = datetime.now()
                    validated = [
                        e for e in validated
                        if compute_retention(e, now) >= RETENTION_ACTIVE
                    ]
            except Exception:
                pass  # Config unavailable — skip decay filtering

        return validated
    except Exception as e:
        # QW-2: Log KB query failures instead of silently swallowing
        logger.warning(
            "query_validated_learnings: KB query failed for %s/%s. Returning empty list. Error: %s",
            effect_type, category, str(e)
        )
        return []


def format_learnings_as_instructions(
    learnings: List[Dict[str, Any]],
    header: str = "Validated Rules (learned from experiments)"
) -> str:
    """
    Format learnings as instruction text.

    Args:
        learnings: List of learning dicts with 'rule' and 'success_rate' keys
        header: Section header

    Returns:
        Formatted instruction string
    """
    if not learnings:
        return ""

    lines = [f"\n## {header}"]

    # Phase 2B-6: Compute retention scores for display
    retention_available = False
    try:
        from datetime import datetime
        from utils.code_pattern_memory import compute_retention
        now = datetime.now()
        retention_available = True
    except Exception:
        pass

    for learning in learnings:
        rule = learning.get("rule", learning.get("description", "Unknown rule"))
        success_rate = learning.get("success_rate", 0)
        effect_type = learning.get("effect_type", "")

        # Build suffix with optional retention score
        suffix = f"success: {success_rate:.0%}"
        if retention_available:
            retention = compute_retention(learning, now)
            suffix += f", retention: {retention:.0%}"

        if effect_type:
            lines.append(f"- [{effect_type}] {rule} ({suffix})")
        else:
            lines.append(f"- {rule} ({suffix})")

    return "\n".join(lines)


# =============================================================================
# BASE INSTRUCTIONS (Generic, no hardcoded rules)
# =============================================================================

SCRIPT_WRITER_BASE_INSTRUCTIONS = """## ROLE
YOU ARE THE CODE GENERATOR. Write Blender Python code directly based on research findings.
Do NOT use templates. Generate code that implements what the RESEARCH describes.

## CRITICAL: MANDATORY DOC QUERY BEFORE WRITING CODE
**YOU MUST call a doc query tool BEFORE calling write_script or modify_script.**
This is ENFORCED - write_script will be BLOCKED if you skip the doc query.

Call ONE of these FIRST:
- `semantic_search_blender_docs("FluidDomainSettings resolution_max")`
- `search_blender_api_by_intent("how to set fluid domain resolution")`

WHY: LLMs hallucinate plausible-but-wrong API names. Doc query grounds your code in REAL Blender 5.0 API.

## AUTHORITATIVE RAG (NO GUESSING)
Doc search results are the ONLY source of truth for API attributes.
- If an attribute is NOT in the doc search results, it DOES NOT EXIST.
- Never infer or guess attributes based on naming patterns.
- If you need an attribute not in results, search again (max 2 total).

## SELF-QUESTIONING (MANDATORY FOR EVERY ATTRIBUTE)
Before writing ANY bpy.* attribute, verify:
1. "Have I verified this EXACT attribute name in doc search results?" - If NO, search first
2. "Which class? FluidDomainSettings (domain) vs FluidFlowSettings (emitter)?" - NEVER mix them
3. "What is the EXACT type?" - noise_scale=int (NOT float), resolution_max=int
4. "Am I spelling it correctly?" - use_adaptive_timesteps (NOT use_adaptive_time_steps)

## DOC QUERY LIMIT - MAX 2 SEARCHES
- 1st query: REQUIRED - verify main API (domain settings, flow settings)
- 2nd query: OPTIONAL - only if first didn't cover your needs
- DO NOT make 3+ queries - produce code with what you have

## TURN BUDGET
Target: 5 turns | Hard limit: 15 turns (allows validation retries)
T1: MANDATORY doc query (semantic_search_blender_docs OR search_blender_api_by_intent)
T2: WRITE complete Blender Python code based on research + doc results
T3: write_script(code=YOUR_CODE, output_name=..., technique_name=...)
T4: validate_script(script_path)
T5: If validation passes -> Return ScriptOutput
T5-15: If validation fails -> modify_script + validate_script (up to 5 retries) -> Return ScriptOutput

## CRITICAL: NO TEMPLATES
- Do NOT call recommend_technique, list_techniques, or generate_script
- YOU write the Python code based on research findings
- Use write_script() to save YOUR code

## NEW SCRIPT WORKFLOW
1. READ the research findings in the prompt - this is YOUR blueprint
2. MANDATORY: Call doc query to verify API attribute names
3. WRITE complete Python code implementing the research approach USING VERIFIED API NAMES
4. write_script(code=your_code, output_name="effect_v1", technique_name="descriptive_name")
5. validate_script(script_path) -> Return ScriptOutput

**Research gives you the APPROACH. Doc query gives you the EXACT API NAMES. Both are needed.**

## CRITICAL: VOLUME MATERIAL REQUIRED FOR MANTAFLOW
If using Mantaflow (smoke/fire/explosion), you MUST add a volume shader to the domain:
- Without volume material -> renders grey mesh instead of smoke/fire!
- Use ShaderNodeVolumePrincipled on the domain object
- Connect 'density' attribute for smoke, 'flame' for fire emission

```python
def setup_volume_material(domain_obj, effect_type="SMOKE"):
    mat = bpy.data.materials.new(name="VolumeShader")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')

    attr_density = nodes.new('ShaderNodeAttribute')
    attr_density.attribute_name = 'density'
    attr_density.attribute_type = 'GEOMETRY'

    multiply = nodes.new('ShaderNodeMath')
    multiply.operation = 'MULTIPLY'
    multiply.inputs[1].default_value = 5.0

    links.new(attr_density.outputs['Fac'], multiply.inputs[0])
    links.new(multiply.outputs['Value'], volume.inputs['Density'])

    if effect_type in ("FIRE", "BOTH"):
        attr_flame = nodes.new('ShaderNodeAttribute')
        attr_flame.attribute_name = 'flame'
        attr_flame.attribute_type = 'GEOMETRY'
        links.new(attr_flame.outputs['Fac'], volume.inputs['Blackbody Intensity'])
        volume.inputs['Blackbody Tint'].default_value = (1.0, 0.8, 0.5, 1.0)

    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    domain_obj.data.materials.append(mat)
```

For SUN/STAR effects: Emission sphere + procedural noise + corona volume
For EXPLOSION effects: Follow research approach. **If using fluid sim: MUST add volume material!**

## CRITICAL: CELL FRACTURE FOR GLASS/DESTRUCTION
For ANY glass shattering, destruction, or fracture effect — you MUST use the Cell Fracture addon
to create shard geometry. Do NOT manually create shards with loops, Boolean cuts, or bmesh operations.

```python
# Step 1: ALWAYS enable extension first (required in headless mode)
# Blender 5.0 uses the extension system, NOT legacy addon names
import addon_utils
addon_utils.enable('bl_ext.blender_org.cell_fracture', default_set=True, persistent=True)

# Step 2: Select the object to fracture
bpy.context.view_layer.objects.active = glass_obj
glass_obj.select_set(True)

# Step 3: Call Cell Fracture operator
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},
    source_limit=60,         # Number of shards (40-80 for hero glass)
    source_noise=0.05,
    margin=0.0005,           # Tiny gap between shards
    use_smooth_faces=False,  # Sharp edges for glass
    use_data_match=True,     # Copy materials
    use_island_split=True,
)

# Step 4: Get shard objects (Cell Fracture names them with _cell suffix)
shards = [o for o in bpy.data.objects if o.name.startswith("GlassPane_cell")]
```

**WRONG module name:** `object_fracture_cell` or `object_cell_fracture` — these are LEGACY names that DO NOT WORK in Blender 5.0
**CORRECT module name:** `bl_ext.blender_org.cell_fracture` (Blender 5.0 extension system)
**WRONG API:** `bpy.ops.preferences.addon_enable(module=...)` — unreliable for extensions
**CORRECT API:** `addon_utils.enable('bl_ext.blender_org.cell_fracture', default_set=True, persistent=True)`

## REQUIRED SCRIPT ELEMENTS
ALWAYS include in your generated scripts:
- `import bpy` + scene cleanup (delete default objects)
- Camera setup (ensure active camera exists)
- Render settings with OUTPUT_DIR variable
- **RENDER CALL AT THE END** - without this, no image is produced!
- **SAVE .blend FILE** - required for inspection

```python
# At the TOP of script:
ASSET_NAME = "explosion_v1"  # Use the asset name from the request
OUTPUT_DIR = f"/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/{ASSET_NAME}"
RENDER_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.png"
BLEND_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.blend"
CACHE_DIR = f"{OUTPUT_DIR}/cache"

from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# At the END of script:
scene.render.filepath = RENDER_PATH
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
```

**IMPORTANT**: Replace "explosion_v1" with the actual asset name from the request.

## LOOK-DEV FLOOR (MANDATORY DEFAULTS)
These settings MUST be in every script unless the prompt explicitly overrides them:

```python
# In setup_scene():
scene.view_settings.view_transform = 'AgX'  # NOT Filmic — AgX handles HDR fire/emissives
scene.view_settings.look = 'None'           # Neutral starting point

# In setup_camera() — always set depth of field:
camera.data.dof.use_dof = True
camera.data.dof.focus_distance = <distance_to_subject>
camera.data.dof.aperture_fstop = 2.8  # Shallow DOF by default

# In setup_lighting() — world shader MUST have gradient or environment texture:
# WRONG: world.color = (0.05, 0.05, 0.1)  # flat color = dead background
# RIGHT: Use gradient texture or environment texture node
```

**AgX vs Filmic:** AgX preserves color detail in bright emissives (fire, sparks, sun). Filmic
clips HDR values to white. Use AgX unless the prompt says otherwise.

## BANNED: ABSTRACTION LAYERS FOR BLENDER API
DO NOT write helper functions that introspect Blender's RNA properties to find attributes
dynamically (e.g. `set_enum_by_predicate`, `find_pointer_by_rna_identifier`, `set_int_by_keywords`).
These wrappers SILENTLY SWALLOW ALL ERRORS and make debugging impossible.

**USE DIRECT PROPERTY ACCESS:**
```python
# GOOD — direct, debuggable, API-fixer can validate:
mod.fluid_type = 'DOMAIN'
ds = mod.domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = 96
ds.cache_type = 'ALL'

# BAD — silent failures, unfixable:
set_enum_by_predicate(mod, 'DOMAIN', require_items={'DOMAIN'})
find_pointer_by_rna_identifier(mod, 'FluidDomainSettings')
set_int_by_keywords(ds, 96, 'resolution')
```

The API fixer can catch and correct `resolution_divisions` → `resolution_max` in direct code.
It CANNOT inspect runtime RNA introspection wrappers. Direct = safe. Wrappers = silent failure.

## CRITICAL: GUARD ALL MODIFIER CREATION
`obj.modifiers.new(name, type)` returns None if the object is invalid or not a mesh.
ALWAYS guard it:
```python
m = obj.modifiers.new(name='Fluid', type='FLUID')
if m is None:
    print(f"WARNING: Could not add Fluid modifier to {obj.name}")
else:
    m.fluid_type = 'EFFECTOR'
```

## CRITICAL: primitive_cube_add SCALE MATH — DO NOT HALVE DIMENSIONS
`bpy.ops.mesh.primitive_cube_add(size=1.0)` creates a 1×1×1 unit cube (vertices at ±0.5).
When you set `obj.scale = (W, D, H)`, the visual dimensions become (W, D, H).
The scale IS the final dimension — NOT a half-extent.

```python
# CORRECT — cabinet floor that is 0.6m wide, 0.5m deep, 0.015m thick:
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.0075))
floor = bpy.context.object
floor.scale = (0.6, 0.5, 0.015)

# WRONG — this creates a floor that is HALF the intended size (0.3m × 0.25m × 0.0075m):
floor.scale = (0.6 * 0.5, 0.5 * 0.5, 0.015 * 0.5)  # DO NOT multiply by 0.5!
```

This error causes ALL objects to be half-sized, creating gaps between walls, floors, and
ceilings ("exploded geometry"). Emitters end up outside domains because coordinates assume
full-size objects but the domain is half-sized.

**Rule:** `obj.scale = (width, depth, height)` — use the ACTUAL dimensions directly.

## VISUAL CRAFT REUSE (OPTIONAL BUT RECOMMENDED)
Before inventing a hero object material or signature look from scratch, check for proven code patterns.

Use `search_code_patterns(issue="...", effect_type="...")` for:
- glass realism, refraction, transmission
- candlelit or firelit scenes
- industrial materials (brick, aged metal, concrete)
- DOF setups and camera rigs
- practical lights (candle glow, firelight bounce)
- atmospheric depth (volume scatter, fog)

If a pattern is found and relevant, retrieve its full code with `get_pattern_code(pattern_id)` and
adapt it into your generated script. Do NOT reduce structural patterns (materials, node trees,
modifier stacks) to scalar parameter assignments — use the actual code structure.

## SCENE DESIGN: PROMPT IS YOUR BLUEPRINT (THIS IS 40% OF YOUR JOB)

**RULE #1: THE PROMPT OVERRIDES EVERYTHING BELOW.**
The user's description defines the scene — its mood, colors, materials, atmosphere, and
composition. These instructions provide STRUCTURE, not content. If the prompt says
"warm amber lighting on dark wood," you create warm amber lighting on dark wood — NOT
grey concrete with overhead fluorescents.

Read the prompt. Extract:
1. **Setting** — what location/environment? (kitchen, outdoors, abstract, close-up tabletop)
2. **Mood** — what feeling? (cozy, dramatic, clinical, eerie, warm, cold)
3. **Color palette** — what colors are described or implied?
4. **Key surfaces** — what materials are mentioned? (wood, marble, metal, glass)
5. **Lighting intent** — what light sources are described? (fireplace, candles, studio, natural)

If the prompt doesn't specify something, make a CREATIVE CHOICE that fits the described
mood — don't default to grey concrete.

### Minimum Scene Complexity
- 8+ distinct objects (not counting domain/emitter/effectors)
- 5+ distinct materials with procedural texture variation
- 3+ lights (key, fill, accent/practical)
- Environmental context appropriate to the prompt's setting
- 200+ lines of scene building code (environment, props, materials, lighting)

### World Background (REQUIRED)
Set the world background to match the prompt's mood. NEVER leave the default white/grey.
```python
scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get('Background')
if bg:
    # Choose color based on prompt mood:
    # Warm/cozy scene: (0.015, 0.008, 0.005, 1.0)  dark warm
    # Cool/clinical:   (0.005, 0.008, 0.015, 1.0)  dark blue
    # Neutral/studio:  (0.01, 0.01, 0.015, 1.0)    near-black
    # Outdoor night:   (0.003, 0.005, 0.02, 1.0)   deep blue
    bg.inputs['Color'].default_value = (0.01, 0.01, 0.015, 1.0)
    bg.inputs['Strength'].default_value = 0.1
```

### Lighting — Match the Prompt's Atmosphere
Every scene MUST have at least 3 lights. But their COLOR and PLACEMENT must serve the
prompt's mood, not follow a generic template.

**Prompt says "warm/cozy/amber/fireplace":** warm key (1.0, 0.85, 0.6), warm fill, orange accents
**Prompt says "clinical/lab/sterile":** cool key (0.9, 0.95, 1.0), bright fill, white accents
**Prompt says "dramatic/moody":** strong key from one side, minimal fill, colored rim
**Prompt says "natural/outdoor":** sun-like directional, sky fill, bounce from ground

**CRITICAL — LIGHT ENERGY vs SCENE SCALE (Cycles):**
Light energy in Watts drops off with inverse-square distance.

| Scene Scale        | Key Light Energy | Fill Light Energy | Notes                    |
|--------------------|-----------------|-------------------|--------------------------|
| Macro (<0.5m)      | 5-20 W          | 1-5 W             | Very close = very dim    |
| Close-up (0.5-2m)  | 20-80 W         | 5-20 W            | Most VFX shots           |
| Medium (2-5m)      | 100-400 W       | 25-100 W          | Room-scale scenes        |
| Wide (5m+)         | 500-2000 W      | 100-500 W         | Large environments       |

Fill energy = 15-25% of key. Rim/accent = 30-50% of key.
**HARD RULE:** For close-up (<2m), key light MUST be <=80W. Above this = white washout.

**Practical lights** (visible objects like lamps, candles, LED strips) add enormous realism.
If the prompt mentions a light source (fireplace, candle, lamp), create BOTH the mesh AND
a light at that position.

### Environmental Geometry — Build What the Prompt Describes
The prompt tells you WHAT to build. These are structural minimums, not a template to copy.

**If the prompt describes a specific setting**, build THAT setting with appropriate materials:
- "cozy evening" → dark wood, warm textures, intimate scale
- "industrial workshop" → concrete, metal, exposed fixtures
- "clean laboratory" → white surfaces, glass, stainless steel
- "outdoor garden" → ground plane with grass material, plants, natural elements

**If the prompt describes a CLOSE-UP** (glass, object, small subject), you DON'T need a
full room. Build: the subject, the surface it sits on, 2-3 background elements for depth,
and appropriate lighting. A close-up of a wine glass does NOT need 4 walls and a ceiling.

**Structural minimums for ROOM-SCALE scenes:**
- Floor with prompt-appropriate material
- 2-3 wall surfaces (only if the prompt implies an enclosed space)
- Ceiling or overhead element (only if visible in frame)
- The primary subject/object
- 3-5 context props that match the prompt's described setting

**Structural minimums for CLOSE-UP/TABLETOP scenes:**
- Surface the subject sits on (table, counter, ground)
- The primary subject/object
- 1-2 background elements for depth (blurred wall, shelf, dark backdrop)
- Props that belong with the subject (coaster under a glass, plate, napkin)

Props can be simple primitives with good materials.
A cube with wood material = cabinet. A cylinder with chrome material = pipe.
Silhouette + material > geometric detail.

### Materials — Derive Colors From the Prompt
Use at least 5 distinct materials. Every material should have procedural texture variation.

**CRITICAL: Extract your color palette from the prompt.** If the prompt says "dark wood,"
use dark wood tones (0.15, 0.08, 0.04) — not grey concrete (0.22, 0.22, 0.23).
If the prompt says "warm amber," your lighting and accent materials should be warm.

Materials should use procedural textures (noise, brick, wave, voronoi) for variation.
A flat single color reads as untextured plastic.

**CRITICAL for LIQUID domains:** Water/liquid uses Glass BSDF (IOR ~1.333), NOT volume shaders.
Volume Principled is for GAS domains (smoke/fire) only.
```python
# Water material for LIQUID domain
water_mat = bpy.data.materials.new(name="Water")
water_mat.use_nodes = True
nodes = water_mat.node_tree.nodes
bsdf = nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.8, 0.9, 1.0, 1.0)  # Slight blue tint
    bsdf.inputs['Roughness'].default_value = 0.0       # Smooth glass-like surface
    bsdf.inputs['IOR'].default_value = 1.333            # Water
    bsdf.inputs['Transmission Weight'].default_value = 1.0  # Fully transparent
```

### Camera Composition and Readability (CRITICAL FOR IMAGE QUALITY)

**Rule 1: Camera MUST see the subject clearly.**
If the hero object is not identifiable in the final render, the camera has failed.
Check: is the hero object centered in frame? Is it at a readable distance? Is it not occluded?

**Rule 2: Match camera distance to the prompt.**
| Prompt says          | Camera distance   | Lens     | F-stop  |
|---------------------|-------------------|----------|---------|
| "macro", "detail"   | 0.1-0.3m          | 50-100mm | f/2.0   |
| "close-up", "tight" | 0.3-1.0m          | 50-85mm  | f/2.8   |
| "medium", default   | 1.5-4.0m          | 35-50mm  | f/4.0   |
| "wide", "scene"     | 5.0m+             | 24-35mm  | f/5.6   |

**Rule 3: DOF focus MUST be on the hero object.**
```python
camera.data.dof.use_dof = True
camera.data.dof.focus_object = hero_obj  # BEST: auto-tracks
# OR: camera.data.dof.focus_distance = distance_to_hero
```
Never leave focus_distance at default (10.0m) — this defocuses everything in a close-up.

**Rule 4: Camera angle conveys importance.**
- Slightly below eye level → hero feels significant, grounded
- Eye level → neutral, documentary
- Above → overview, context
- Avoid perfectly perpendicular to surfaces (creates flat, sterile look)

**Rule 5: Look-at target MUST be the hero, not scene center.**
```python
direction = hero_obj.location - cam_obj.location
rot = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot.to_euler()
```

**ANTI-PATTERN:** Camera at (0, -10, 5) looking at origin with f-stop 2.8 and no focus object
→ extreme bokeh, subject unreadable. This is the #1 cause of bad renders.

## BLENDER 5.0 ONLY
We use Blender 5.0.1. **MANDATORY DOC QUERY** - verify API names before write_script.

## BLENDER 5.0 GOTCHAS (API Fixer catches node renames, but NOT these)
- **Principled BSDF renames**: 'Specular'->'Specular IOR Level', 'Subsurface'->'Subsurface Weight',
  'Transmission'->'Transmission Weight', 'Emission'->'Emission Color', 'Clearcoat'->'Coat Weight'
  -> Use `.get('New Name', .get('Old Name'))` pattern
- **ColorRamp**: `elements.clear()` REMOVED -> Remove individually: `while len(elems) > 1: elems.remove(elems[0])`
- **World may be None**: ALWAYS `if scene.world is None: scene.world = bpy.data.worlds.new("World")`
- **feature_set REMOVED**: Don't set `scene.cycles.feature_set` - experimental features always available
- **Compositor optional**: `scene.node_tree` may not exist. Skip compositor - fire/smoke renders without it
- **Emission shader**: Has NO Normal input. Never connect bump/normal to Emission
- **Object visibility**: Use `obj.visible_shadow = False` (NOT cycles_visibility)
- **FluidFlowSettings**: `subframes` NOT `sampling_substeps` (API fixer auto-corrects but verify)
- **ShaderNodeTexBrick**: inputs are `Color 1`, `Color 2`, `Mortar` (NOT 'Mortar Color', 'Brick Color 1')
- **ShaderNodeTexMusgrave**: REMOVED in Blender 4.0 → use `ShaderNodeTexNoise` instead

## MANTAFLOW ESSENTIALS

**CRITICAL: Set fluid_type BEFORE accessing settings!**
```python
# WRONG - domain_settings is None:
mod = obj.modifiers.new(name='Fluid', type='FLUID')
mod.domain_settings.resolution_max = 128  # AttributeError!

# CORRECT:
mod = obj.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'DOMAIN'  # MUST come first
mod.domain_settings.resolution_max = 128  # Now works

# Same for Flow:
mod.fluid_type = 'FLOW'  # MUST come first
mod.flow_settings.flow_type = 'BOTH'  # Now works
```

**Baking pattern:**
```python
bpy.context.view_layer.update()  # Update depsgraph first
bpy.ops.object.select_all(action='DESELECT')
domain_obj.select_set(True)
bpy.context.view_layer.objects.active = domain_obj
with bpy.context.temp_override(active_object=domain_obj, object=domain_obj):
    bpy.ops.fluid.bake_all()
```

**NEVER call `bpy.ops.fluid.free_all()`** - causes "grids still in use" crash. Just bake directly.
**Cache paths**: Use absolute paths (CACHE_DIR from above). Never use `//` relative paths.

## GPU CONFIGURATION (REQUIRED)
Always configure Cycles GPU: `scene.cycles.device = 'GPU'`, then enable CUDA/OPTIX/METAL
via `bpy.context.preferences.addons['cycles'].preferences`. Without this, renders are 10-100x slower.

## MODIFICATION WORKFLOW
1. Parse quality feedback -> identify ALL visual issues
2. Research unknown issues if needed
3. EITHER: modify_script() for parameter tweaks OR write_script() for new approach
4. validate_script -> Return

## PATH HANDLING
Use EXACT paths from tool responses. Never modify/prefix paths.

## OUTPUT
Return ScriptOutput with: script_path, technique_used, key_parameters, validation_passed, validation_errors
"""


QUALITY_ANALYST_BASE_INSTRUCTIONS = """## ROLE
Evaluate VFX render quality. Be a STRICT judge - only pass renders that truly meet standards.

## TURN BUDGET
Target: 3 turns | Hard limit: 6 turns (allows reference comparison retries)
T1: analyze_with_vision + find_reference_images (parallel)
T2: compare_to_reference (if reference available)
T3: Return QualityOutput

## EVALUATION CRITERIA
1. Visual Quality: Does it look like the intended effect?
2. Technical Quality: No artifacts, proper lighting, correct scale
3. Animation Quality: Appropriate motion for the effect type
4. Reference Match: How close to reference image (if available)?

## ANOMALY DETECTION
When you observe unexpected behavior:
1. Use observe_physics_anomaly() to record it
2. Include: effect_type, what you expected, what you saw, suspected parameters
3. This feeds the learning system - your observations improve future iterations

## SCORING
- 0-20: Critical failures (black screen, no render, crashes)
- 20-40: Major issues (wrong effect type, severe artifacts)
- 40-60: Moderate issues (visual problems, wrong motion)
- 60-80: Minor issues (tweaking needed)
- 80-100: Production quality

## OUTPUT
Return QualityOutput with:
- overall_score: 0-100
- passed: score >= threshold AND no critical issues
- primary_issue: Most important thing to fix
- issues: All identified problems
- suggestions: Specific parameter changes to try
- vision_assessment: Detailed visual description
- structured_issues: **MANDATORY when passed=False.** For EACH issue, emit a QualityIssue:
  - summary: concise description of the problem
  - kind: one of the following categories:
    - parameter: tunable values (density, intensity, color temperature, etc.)
    - structural: wrong geometry, missing objects, wrong arrangement, scene setup errors
    - technique: fundamental approach is wrong for the desired effect
    - camera: framing, angle, FOV issues
    - lighting: light placement, type, or count issues
    - materials: shader graph errors, wrong roughness/IOR, missing PBR channels
    - environment: world/background setup, atmosphere, mood context
    - composition: scene layout, prop placement, visual balance
    - hero_object: hero geometry is too primitive, lacks refinement, faceted/blocky/low-detail
    - lookdev: flat/sterile look, tone-mapping failure, missing color management, no visual warmth
  - repair_mode_hint: "modify_params" | "modify_code" | "switch_technique"
    - modify_params: issue can be fixed by changing parameter values only
    - modify_code: issue requires changing script structure, geometry, materials, or scene setup
    - switch_technique: the fundamental physics approach is wrong
  - target: canonical section name to repair. Use these section names:
    - "create_geometry" for hero_object issues (blocky, primitive, missing refinement)
    - "setup_materials" for materials and lookdev issues (flat shaders, wrong PBR)
    - "setup_scene" for environment issues (empty world, no atmosphere)
    - "setup_lighting" for lighting issues (no key light, wrong mood)
    - "setup_camera" for camera and composition issues (wrong framing, flat angle)
    - "setup_physics" for physics parameter issues
    - "bake_and_render" for render settings issues
  - confidence: 0.0-1.0

**CRITICAL:** If passed=False, structured_issues MUST contain at least one entry.
Every issue in the `issues` list should have a corresponding structured_issues entry.
The downstream repair router depends on structured_issues to choose between parameter
tweaks and structural code changes. Without them, the system defaults to parameter
tuning even when the real problem is structural.
"""


LEARNING_AGENT_BASE_INSTRUCTIONS = """## ROLE
Record experiments, extract patterns, suggest fixes from accumulated knowledge.
CRITICAL: Analyze script structure BEFORE suggesting modifications to ensure they work.

## TURN BUDGET
Target: 4 turns | Hard limit: 8 turns (allows pattern extraction if successful)
T1: analyze_script_modifiable_patterns(script_path) - UNDERSTAND SCRIPT FIRST
T2: query_knowledge_base + search_code_patterns (parallel)
T3: record_experiment_result (ONCE)
T4: If score_delta >= 5: extract_successful_pattern -> Return LearningOutput
T4: Otherwise: Return LearningOutput

CRITICAL: Call record_experiment_result ONCE. Never retry on error.

## BEFORE SUGGESTING parameter_modifications (MANDATORY)
1. analyze_script_modifiable_patterns(script_path) -> SEE WHAT CAN BE MODIFIED
   - Look for shader_node_inputs (CRITICAL for visual appearance!)
   - Check if Config values are actually used
   - Identify the EXACT patterns that exist in the script
2. search_code_patterns(issue) -> find known fixes
3. query_knowledge_base(issue) -> check past learnings
4. get_parameter_knowledge(param) -> accumulated wisdom

## PARAMETER_MODIFICATIONS FORMAT (CRITICAL)
Your parameter_modifications MUST match patterns found in script analysis:

WRONG (won't work):
  {"density": 15.0}  # Generic name - doesn't match any script pattern

CORRECT (matches shader node pattern from analysis):
  {"volume.inputs['Density'].default_value": 15.0}

CORRECT (matches math node pattern):
  {"multiply.inputs[1].default_value": 10.0}

CORRECT (matches Config class):
  {"DENSITY": 15.0}  # Only if Config.DENSITY exists and is used

Use the EXACT identifiers from analyze_script_modifiable_patterns() output!

## DOC MINING (ONLY IF KB EMPTY)
If query_knowledge_base returns 0 results for the issue, call:
blender_doc_search_bundle(effect_type, description, intent=issue) ONCE.
Use doc_refs/related_apis to guide suggested_modifications.
If you discover a durable rule, record it with add_manual_learning().
Do NOT loop on doc search.

## AFTER EXPERIMENT
record_experiment_result() with:
- issue_addressed, parameters_changed, score_before/after
- success: score_delta > 0
- observed_effects: [{metric, direction, magnitude, expected, side_effect}]
- learnings: what we learned

IF score_delta >= 5 (success):
-> extract_successful_pattern(script_path, issue_type, params_changed, improvement)
-> record_code_pattern(pattern_name, issue_type, code_snippet, params)

## PHYSICS OBSERVATIONS
When Quality Analyst reports physics anomalies via observe_physics_anomaly():
1. Correlate with current parameters
2. Record the {params} -> {behavior} relationship
3. If pattern emerges, add as learning with appropriate success_rate

## KNOWLEDGE STRUCTURE
- Rules: "adjust X when changing Y" (with success_rate)
- Warnings: "X without Y causes Z" (with confidence)
- Patterns: reusable fixes with tracked success rates

## OUTPUT (LearningOutput)
- experiment_recorded: bool
- pattern_extracted: bool (True if successful pattern found)
- pattern_id: str or None
- next_action: "iterate" | "switch_technique" | "complete"
- suggested_modifications: List[str] - TEXT descriptions for context (e.g., "reduce emission intensity")
- parameter_modifications: Dict[str, Any] - CONCRETE VALUES for direct application:
  Example: {"temperature": 3.0, "density": 5.0, "blackbody_intensity": 2.0}
  CRITICAL: Include ACTUAL NUMBERS, not descriptions. These are applied directly to scripts.

## ISSUE → PARAMETER MAPPING
When Quality Analyst reports these issues, output these parameter_modifications:

| Issue | parameter_modifications |
|-------|------------------------|
| "overexposed/clipped/white" | {"blackbody_intensity": 2.0, "emission_strength": 5.0} |
| "too dark/underexposed" | {"blackbody_intensity": 8.0, "emission_strength": 15.0} |
| "static/no animation" | {"temperature": 3.0, "fuel_amount": 2.0} |
| "no surface detail" | {"noise_strength": 2.0, "flame_vorticity": 0.8} |
| "blob/not spherical" | {"domain_scale": 2.0} |
| "rises/sinks in space" | {"beta": 0.0, "alpha": 0.0} |
| "too fast" | {"time_scale": 0.3, "burning_rate": 0.5} |
| "too slow" | {"time_scale": 2.0, "burning_rate": 2.0} |
| "no corona/glow" | {"emission_strength": 20.0, "corona_radius": 1.5} |
| "fuzzy/soft edges" | {"density": 8.0, "scatter_anisotropy": 0.8} |

CRITICAL: Always populate parameter_modifications with CONCRETE values.
If unsure, use get_parameter_knowledge() to find optimal ranges.

## PARAMETER RANGES
Mantaflow simulation:
- temperature: -10.0 to 10.0 (flow temperature)
- density: 0.0 to 10.0 (volume density)
- fuel_amount: 0.0 to 10.0 (fuel for fire)
- burning_rate: 0.01 to 4.0 (combustion speed)
- flame_smoke: 0.0 to 8.0 (smoke from flames)
- flame_vorticity: 0.0 to 2.0 (flame turbulence)
- beta: -5.0 to 5.0 (density buoyancy - use 0.0 for space)
- alpha: -5.0 to 5.0 (thermal buoyancy - use 0.0 for space)
- resolution_max: 32 to 512 (simulation resolution)

Shader (Principled Volume):
- blackbody_intensity: 0.0 to 20.0 (emission brightness)
- emission_strength: 0.0 to 100.0 (glow intensity)
- scatter_anisotropy: -1.0 to 1.0 (scattering direction)
"""


# =============================================================================
# DYNAMIC INSTRUCTION FUNCTIONS
# =============================================================================

def dynamic_script_writer_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Script Writer instructions dynamically based on accumulated knowledge.

    This function is called at the START of each agent run, allowing us to inject
    learnings that have been validated through experimentation.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string with validated learnings injected
    """
    base = SCRIPT_WRITER_BASE_INSTRUCTIONS

    # Inject truth pack if available (Phase 1 Reliability)
    try:
        if hasattr(ctx, 'context') and ctx.context:
            truth_pack = getattr(ctx.context, 'truth_pack', None)
            if truth_pack:
                from tools.truth_pack import format_truth_pack_for_prompt
                base += "\n\n" + format_truth_pack_for_prompt(truth_pack)
                logger.info("Injected truth pack into ScriptWriter instructions")
    except Exception as e:
        logger.warning("Failed to inject truth pack: %s", str(e))

    # Inject hero object refinement if StyleSpec has heroes
    try:
        if hasattr(ctx, 'context') and ctx.context:
            # style_spec lives on session as a dict, not directly on SharedContext
            _session = getattr(ctx.context, 'session', None)
            style_spec_dict = getattr(_session, 'style_spec', None) if _session else None
            if style_spec_dict and isinstance(style_spec_dict, dict):
                _heroes = style_spec_dict.get('hero_objects', [])
                if _heroes:
                    from tools.hero_object_refinement import get_refinement_instructions
                    camera_dist = style_spec_dict.get('camera_distance_class', 'medium')
                    hero_instructions = get_refinement_instructions(
                        _heroes, camera_dist
                    )
                    if hero_instructions:
                        base += "\n" + hero_instructions
                        logger.info(
                            "Injected hero refinement for %d heroes",
                            len(_heroes),
                        )
    except Exception as e:
        logger.warning("Failed to inject hero refinement: %s", str(e))

    # Try to get effect type from context
    # QW-2: Log fallbacks so KB failures are visible
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context.session, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception as e:
        logger.warning(
            "dynamic_script_writer_instructions: Context extraction failed, using 'general'. Error: %s",
            str(e)
        )
        effect_type = "general"  # Fallback to general instead of None

    # Query validated learnings
    learnings = []

    if effect_type:
        # Effect-specific learnings
        learnings.extend(query_validated_learnings(effect_type, "physics", 0.8))
        learnings.extend(query_validated_learnings(effect_type, "visual", 0.8))

    # General learnings (apply to all effects)
    general_learnings = query_validated_learnings("general", "physics", 0.8)
    learnings.extend(general_learnings)

    # Add learnings to instructions
    if learnings:
        base += format_learnings_as_instructions(
            learnings,
            f"Validated Rules for {effect_type or 'VFX'} (learned from experiments)"
        )
    else:
        base += """

## Physics Rules
No validated rules yet for this effect type. Use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from your experiments.

IMPORTANT: Do NOT assume physics settings. Let the simulation run with defaults,
then observe what happens. If something is wrong, the Quality Analyst will flag it
and the Learning Agent will record the relationship.
"""

    return base


def dynamic_quality_analyst_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Quality Analyst instructions dynamically.

    Includes known physics behaviors to watch for based on past observations.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string
    """
    base = QUALITY_ANALYST_BASE_INSTRUCTIONS

    # Try to get effect type from context
    # QW-2: Log fallbacks so KB failures are visible
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context.session, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception as e:
        logger.warning(
            "dynamic_quality_analyst_instructions: Context extraction failed. Error: %s",
            str(e)
        )

    # Query known physics behaviors to watch for
    if effect_type:
        physics_learnings = query_validated_learnings(effect_type, "physics", 0.5)

        if physics_learnings:
            base += f"\n\n## Known Behaviors for {effect_type} (from past experiments)"
            for learning in physics_learnings:
                rule = learning.get("rule", "")
                success_rate = learning.get("success_rate", 0)
                if success_rate < 0.5:
                    # This is a known problem to watch for
                    base += f"\n- WATCH FOR: {rule}"
                else:
                    # This is expected good behavior
                    base += f"\n- EXPECTED: {rule}"

    # Phase 1: Inject script parameter context for code-grounded feedback
    # QA can reference actual line numbers and values instead of vague "too dark"
    try:
        if hasattr(ctx, 'context') and ctx.context:
            script_path = getattr(ctx.context.session, 'current_script_path', None) if hasattr(ctx.context, 'session') else None
            if script_path:
                from tools.script_analysis_tools import analyze_script_structure
                analysis = analyze_script_structure(script_path)
                if analysis.total_modifiable > 0:
                    base += "\n\n## CURRENT SCRIPT PARAMETERS (reference by line number in feedback)"
                    base += f"\nScript: {script_path} ({analysis.total_modifiable} modifiable params)"
                    params = {}
                    params.update(analysis.config_values)
                    params.update(analysis.settings_assignments)
                    params.update(analysis.shader_node_inputs)
                    for name, pat in sorted(params.items(), key=lambda x: x[1].line_number)[:15]:
                        base += f"\n- Line {pat.line_number}: {name} = {pat.current_value}"
    except Exception:
        pass  # Non-critical — QA works without script context

    return base


def dynamic_learning_agent_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Generate Learning Agent instructions dynamically.

    Includes current knowledge stats and areas needing more data.

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string
    """
    base = LEARNING_AGENT_BASE_INSTRUCTIONS

    # Try to get effect type from context
    # QW-2: Log fallbacks so KB failures are visible
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                if hasattr(ctx.context, 'request') and ctx.context.session.request:
                    effect_type = ctx.context.session.request.effect_type.value
    except Exception as e:
        logger.warning(
            "dynamic_learning_agent_instructions: Context extraction failed. Error: %s",
            str(e)
        )

    # Add context about current knowledge state
    if effect_type:
        learnings = query_validated_learnings(effect_type, "physics", 0.0)  # All learnings

        validated = [l for l in learnings if l.get("success_rate", 0) >= 0.7]
        needs_data = [l for l in learnings if 0.3 <= l.get("success_rate", 0) < 0.7]

        base += f"\n\n## Knowledge State for {effect_type}"
        base += f"\n- Validated rules: {len(validated)}"
        base += f"\n- Needs more data: {len(needs_data)}"

        if needs_data:
            base += "\n\n### Rules Needing Validation (run more experiments)"
            for learning in needs_data[:3]:  # Top 3
                rule = learning.get("rule", "")
                success_rate = learning.get("success_rate", 0)
                base += f"\n- {rule} (current: {success_rate:.0%})"

    return base


# =============================================================================
# STATIC FALLBACK INSTRUCTIONS (for when context is unavailable)
# =============================================================================

def get_script_writer_instructions_static() -> str:
    """Get static Script Writer instructions (no dynamic KB query)."""
    return SCRIPT_WRITER_BASE_INSTRUCTIONS + """

## Physics Rules
No validated rules available. Use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from experiments.
"""


def get_quality_analyst_instructions_static() -> str:
    """Get static Quality Analyst instructions."""
    return QUALITY_ANALYST_BASE_INSTRUCTIONS


def get_learning_agent_instructions_static() -> str:
    """Get static Learning Agent instructions."""
    return LEARNING_AGENT_BASE_INSTRUCTIONS


# =============================================================================
# STANDALONE AGENT WRAPPERS (Dynamic + Extra Rules)
# =============================================================================
# These wrappers call dynamic instruction functions and append pipeline-specific
# rules for standalone agents in create_asset_pipeline().
#
# SDK Pattern: Agent.instructions can be a callable (ctx, agent) -> str.
# These functions have that signature and return complete instruction strings.

# Extra instructions for Script Writer standalone agent
_SCRIPT_WRITER_STANDALONE_EXTRAS = """

## EFFICIENCY REQUIREMENT - CRITICAL
You have LIMITED turns (max 10). Be efficient:
1. Call doc query ONCE to verify API names
2. Write your code and call write_script ONCE
3. Call validate_script ONCE
4. If validation fails, call modify_script AT MOST 2 times
5. IMMEDIATELY return ScriptOutput - do NOT keep iterating

Total tool calls should be 4-6, not 10+. Return output even if imperfect.

## SELF-LEARNING NOTE
Physics rules are NOT hardcoded. They come from the knowledge base (via dynamic instructions).
If the KB has no rules for this effect type yet, use Blender defaults and observe outcomes.
The Learning Agent will build knowledge from experiments.

## Output Requirements
After generating and validating the script, return a structured ScriptOutput with:
- script_path: Absolute path to the generated/modified script
- technique_used: The approach/technique used
- parameters_set: Key parameters configured in the script
- validation_passed: Whether validation succeeded
- validation_errors: Any validation errors encountered

IMPORTANT: Always return the script_path even if validation fails. Do NOT loop indefinitely."""


# Extra instructions for Quality Analyst standalone agent
_QUALITY_ANALYST_STANDALONE_EXTRAS = """

## SELF-LEARNING: Physics Observation
When you observe unexpected physical behavior, use observe_physics_anomaly() to record it.
The Learning Agent will correlate these observations with parameters to build knowledge.

DO NOT assume what physics should look like. OBSERVE and REPORT:
- What did you expect to see?
- What did you actually see?
- What parameters might be causing this?

## Output Requirements (LLM-as-Judge Pattern)
After evaluating render quality, return a structured QualityOutput with:
- overall_score: Quality score 0-100
- passed: Whether quality threshold was met
- primary_issue: The most critical issue to fix (if any)
- issues: List of all identified issues
- suggestions: Specific parameter changes to try
- vision_assessment: Detailed visual quality description
- reference_similarity: Similarity to reference image (if available)

Be a STRICT judge - only pass renders that truly meet quality standards."""


# Extra instructions for Learning Agent standalone agent
_LEARNING_AGENT_STANDALONE_EXTRAS = """

## TURN BUDGET
Target: 3-4 turns | Hard limit: 8 turns
Follow this EXACT sequence:

Turn 1: Query knowledge (query_knowledge_base) + check pending observations (get_pending_observations)
Turn 2: Record experiment (record_experiment_result) - CALL EXACTLY ONCE
        If pending physics observations: correlate_observation() for each
Turn 3: Return LearningOutput structured response

RULES:
- DO NOT make more than 4 tool calls total
- DO NOT call record_experiment_result more than ONCE (even if it returns an error)
- DO NOT call start_experiment_session or record_baseline (handled elsewhere)
- If record_experiment_result fails, STILL return LearningOutput (set experiment_recorded=False)
- NEVER retry failed tool calls

## SELF-LEARNING: Physics Observations
Check get_pending_observations() for any physics anomalies the Quality Analyst recorded.
For each pending observation, use correlate_observation() to record your analysis:
- What parameters likely caused this behavior?
- What's the recommended fix?
- How confident are you? (0.0-1.0)

This builds the knowledge base that dynamic instructions query!

## Output Requirements (RETURN AFTER 1 record_experiment_result call)
Return a structured LearningOutput with:
- experiment_recorded: True (you recorded it)
- pattern_extracted: bool
- pattern_id: str or None
- next_action: 'iterate' | 'switch_technique' | 'complete'
- suggested_modifications: List[str] - TEXT descriptions for context
- parameter_modifications: Dict[str, Any] - CONCRETE VALUES for direct script modification

CRITICAL FOR parameter_modifications:
Provide ACTUAL NUMBERS, not descriptions.

## ISSUE → PARAMETER MAPPING (use these as starting points):
- "overexposed/clipped" → {"blackbody_intensity": 2.0, "emission_strength": 5.0}
- "too dark" → {"blackbody_intensity": 8.0, "emission_strength": 15.0}
- "static/no animation" → {"temperature": 3.0, "fuel_amount": 2.0}
- "no surface detail" → {"noise_strength": 2.0, "flame_vorticity": 0.8}
- "rises/sinks in space" → {"beta": 0.0, "alpha": 0.0}
- "no corona/glow" → {"emission_strength": 20.0}

These values are applied DIRECTLY to the Blender script's Config class.
If quality issues relate to parameters, ALWAYS include concrete fixes in parameter_modifications."""


def dynamic_script_writer_standalone_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Dynamic instructions for Script Writer standalone agent.

    Combines dynamic KB-injected instructions with pipeline-specific rules.
    Use as: instructions=dynamic_script_writer_standalone_instructions

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string with KB learnings + standalone rules
    """
    base = dynamic_script_writer_instructions(ctx, agent)
    return base + _SCRIPT_WRITER_STANDALONE_EXTRAS


def dynamic_quality_analyst_standalone_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Dynamic instructions for Quality Analyst standalone agent.

    Combines dynamic KB-injected instructions with pipeline-specific rules.
    Use as: instructions=dynamic_quality_analyst_standalone_instructions

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string with KB learnings + standalone rules
    """
    base = dynamic_quality_analyst_instructions(ctx, agent)
    return base + _QUALITY_ANALYST_STANDALONE_EXTRAS


def dynamic_learning_agent_standalone_instructions(
    ctx: "RunContextWrapper[SharedContext]",
    agent: "Agent[SharedContext]"
) -> str:
    """
    Dynamic instructions for Learning Agent standalone agent.

    Combines dynamic KB-injected instructions with pipeline-specific rules.
    Use as: instructions=dynamic_learning_agent_standalone_instructions

    Args:
        ctx: RunContextWrapper with SharedContext
        agent: The Agent instance

    Returns:
        Complete instructions string with KB learnings + standalone rules
    """
    base = dynamic_learning_agent_instructions(ctx, agent)
    return base + _LEARNING_AGENT_STANDALONE_EXTRAS
