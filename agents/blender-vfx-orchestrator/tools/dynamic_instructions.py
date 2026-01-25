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
    min_success_rate: float = 0.7
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
        result = kb_query(f"{category} rules for {effect_type}")
        data = json.loads(result) if isinstance(result, str) else result
        results = data.get("results", [])

        # Filter by success rate
        validated = []
        for learning in results:
            success_rate = learning.get("success_rate", 0)
            if success_rate >= min_success_rate:
                validated.append(learning)

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
    for learning in learnings:
        rule = learning.get("rule", learning.get("description", "Unknown rule"))
        success_rate = learning.get("success_rate", 0)
        effect_type = learning.get("effect_type", "")

        if effect_type:
            lines.append(f"- [{effect_type}] {rule} (success: {success_rate:.0%})")
        else:
            lines.append(f"- {rule} (success: {success_rate:.0%})")

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

WHY: LLMs hallucinate plausible-but-wrong API names (e.g., "resolution_divisions" doesn't exist).
The doc query grounds your code in REAL Blender 5.0 API attributes.

## DOC QUERY LIMIT - MAX 2 SEARCHES
After your MANDATORY first query, you may do ONE more if needed. Then STOP.
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
- Do NOT call recommend_technique or list_techniques
- Do NOT use generate_script (it uses hardcoded templates)
- YOU write the Python code based on research findings
- Use write_script() to save YOUR code

## NEW SCRIPT WORKFLOW
1. READ the research findings in the prompt - this is YOUR blueprint
2. MANDATORY: Call doc query to verify API attribute names (semantic_search_blender_docs or search_blender_api_by_intent)
3. WRITE complete Python code that implements the research approach USING VERIFIED API NAMES
4. write_script(code=your_code, output_name="effect_v1", technique_name="descriptive_name")
5. validate_script(script_path) -> Return ScriptOutput

**REMEMBER: Research gives you the APPROACH. Doc query gives you the EXACT API NAMES. Both are needed.**

## CODE GENERATION GUIDELINES
For SUN/STAR effects (from typical research):
- Create UV sphere for photosphere with Emission material
- Use procedural noise for granulation (ShaderNodeTexNoise)
- Create larger sphere for corona with Volume Scatter/Emission
- Set up compositor glare/bloom for outer glow
- Animate noise coordinates for evolution

For EXPLOSION effects:
- Research will specify: fluid sim vs shader-based
- Follow the research approach, don't default to fluid sim

ALWAYS include:
- import bpy
- Scene cleanup
- Camera setup
- Render settings with OUTPUT_DIR variable (populated from asset name)
- **RENDER CALL AT THE END** - without this, no image is produced!

```python
# At the TOP of script - define output paths from asset name:
ASSET_NAME = "explosion_v1"  # Use the asset name from the request
OUTPUT_DIR = f"/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/{ASSET_NAME}"
RENDER_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.png"
BLEND_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.blend"
CACHE_DIR = f"{OUTPUT_DIR}/cache"

# Create directories
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# At the end of main():
scene.render.filepath = RENDER_PATH
bpy.ops.render.render(write_still=True)  # CRITICAL: actually renders the image
print(f"Rendered to: {scene.render.filepath}")

# ALWAYS save .blend file for inspection/rebaking
bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
print(f"Saved .blend to: {BLEND_PATH}")
```

**IMPORTANT**: Replace "explosion_v1" with the actual asset name from the request.
The asset name is provided in the prompt (e.g., "Asset Name: shakedown_123456").

## CRITICAL: BLENDER 5.0 ONLY
We use Blender 5.0.1. Generate code for THIS VERSION ONLY.

**MANDATORY DOC QUERY** - You MUST query docs to verify API names before write_script.
Example queries:
- `semantic_search_blender_docs("FluidDomainSettings attributes resolution")`
- `search_blender_api_by_intent("how to configure mantaflow domain settings")`

## KNOWN BLENDER 5.0 API CHANGES
Handle these SPECIFIC changes (verified for Blender 5.0):

### Principled BSDF Socket Renames:
- 'Specular' -> 'Specular IOR Level'
- 'Subsurface' -> 'Subsurface Weight'
- 'Transmission' -> 'Transmission Weight'
- 'Emission' -> 'Emission Color' (also add 'Emission Strength')
- 'Clearcoat' -> 'Coat Weight'

Pattern for Principled BSDF sockets:
```python
# Use .get() for these SPECIFIC renamed sockets:
bsdf.inputs.get('Emission Color', bsdf.inputs.get('Emission')).default_value = (1,1,1,1)
bsdf.inputs.get('Specular IOR Level', bsdf.inputs.get('Specular')).default_value = 0.5
```

### Compositor Setup (BLENDER 5.0 - CRITICAL):
In Blender 5.0, `scene.node_tree` may not exist. The compositor is OPTIONAL for VFX.
**RECOMMENDED: Skip compositor setup entirely** - fire/smoke renders fine without glare.
```python
def setup_compositor_glare(scene):
    # OPTIONAL compositor glare - skip if API not available
    try:
        scene.use_nodes = True
        # Check if node_tree exists (may not in Blender 5.0)
        if not hasattr(scene, 'node_tree') or scene.node_tree is None:
            print("WARN: Compositor node_tree not available, skipping glare")
            return
        nt = scene.node_tree
        # ... setup nodes
    except AttributeError:
        print("WARN: Compositor setup failed, skipping")
        return
```
**SIMPLER: Just don't call setup_compositor at all** - the effect renders without it.

### Object Visibility (cycles_visibility REMOVED):
```python
# Use direct property (Blender 5.0):
obj.visible_shadow = False
obj.visible_diffuse = False
```

### Emission Shader:
Emission shaders have NO Normal input. Never connect bump/normal to Emission.

### ShaderNodeSeparateRGB/ShaderNodeCombineRGB REMOVED (Blender 5.0):
The 'Separate RGB' and 'Combine RGB' nodes were removed. Use unified color nodes:
```python
# OLD (FAILS in Blender 5.0):
sep = nodes.new('ShaderNodeSeparateRGB')  # RuntimeError: Node type undefined
links.new(color_output, sep.inputs['Image'])
r_value = sep.outputs['R']

# NEW (Blender 5.0):
sep = nodes.new('ShaderNodeSeparateColor')  # Unified node
sep.mode = 'RGB'  # Can also be 'HSV', 'HSL'
links.new(color_output, sep.inputs['Color'])
r_value = sep.outputs['Red']  # Also: 'Green', 'Blue', 'Alpha'

# SIMPLER ALTERNATIVE (for grayscale):
bw = nodes.new('ShaderNodeRGBToBW')
links.new(color_output, bw.inputs['Color'])
gray_value = bw.outputs['Val']
```

### Material shadow_method REMOVED:
```python
# Use object-level shadow control:
obj.visible_shadow = False
```

### use_nodes Deprecated:
`material.use_nodes = True` works but shows deprecation warning. Skip these lines entirely.

### ColorRamp.elements.clear() REMOVED (Blender 5.0):
The `color_ramp.elements.clear()` method does NOT exist. Remove elements individually:
```python
# OLD (FAILS in Blender 5.0):
ramp.color_ramp.elements.clear()  # AttributeError: 'bpy_prop_collection' has no attribute 'clear'

# NEW (Blender 5.0) - Remove elements by index:
while len(ramp.color_ramp.elements) > 1:  # Keep at least 1 element
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])

# OR: Just set positions/colors on existing elements:
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (1.0, 0.0, 0.0, 1.0)
# Add new elements with .new(position)
ramp.color_ramp.elements.new(0.5)
ramp.color_ramp.elements[1].color = (1.0, 1.0, 0.0, 1.0)
```

### World May Not Exist (CRITICAL - Blender 5.0):
In Blender 5.0, `scene.world` may be None when starting from a new/bare scene.
**ALWAYS create world if missing** before setting world properties:
```python
def setup_world(scene):
    # CRITICAL: Create world if it doesn't exist
    if scene.world is None:
        scene.world = bpy.data.worlds.new(name="World")

    world = scene.world
    world.use_nodes = True
    nt = world.node_tree
    # ... setup nodes
```

### CyclesRenderSettings.feature_set REMOVED (Blender 5.0):
The `scene.cycles.feature_set = 'EXPERIMENTAL'` line is REMOVED in Blender 5.0.
Experimental features like adaptive subdivision are always available now.
```python
# OLD (FAILS in Blender 5.0):
scene.cycles.feature_set = 'EXPERIMENTAL'  # AttributeError

# NEW (Blender 5.0):
# Just remove the line - experimental features are always enabled
# For adaptive subdivision, just set:
mod.use_adaptive_subdivision = True
mat.cycles.displacement_method = 'DISPLACEMENT'
```

## MANTAFLOW CLI USAGE (CRITICAL - READ FIRST)

**Mantaflow works perfectly with Cycles and GPU rendering.** Use it for fire, smoke, explosions.

### CRITICAL: Fluid Modifier Setup Order (Blender 5.0)
**You MUST set `fluid_type='DOMAIN'` BEFORE accessing `domain_settings`!**

```python
# WRONG - domain_settings is None, causes AttributeError:
mod = domain.modifiers.new(name='Fluid', type='FLUID')
mod.domain_settings.resolution_max = 128  # ERROR: 'NoneType' has no attribute 'resolution_max'

# CORRECT - Set fluid_type first, then access domain_settings:
mod = domain.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'DOMAIN'  # REQUIRED - makes domain_settings available
mod.domain_settings.domain_type = 'GAS'  # Now this works
mod.domain_settings.resolution_max = 128  # And this works too
```

The same applies to Flow emitters:
```python
mod = emitter.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'FLOW'  # REQUIRED - makes flow_settings available
mod.flow_settings.flow_type = 'BOTH'  # Now this works
```

### Complete Mantaflow + Cycles Setup Pattern:
```python
import bpy
from pathlib import Path

# ASSET PATHS - use asset name from request, NOT /tmp/
ASSET_NAME = "fire_explosion_v1"  # Replace with actual asset name from request
OUTPUT_DIR = f"/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/{ASSET_NAME}"
CACHE_DIR = f"{OUTPUT_DIR}/cache"
RENDER_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.png"
BLEND_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.blend"

def setup_mantaflow_scene():
    # Complete pattern for Mantaflow VFX with CLI rendering
    scene = bpy.context.scene

    # 0. CREATE OUTPUT DIRECTORIES
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # 1. WORLD: Create if missing (CLI starts with no world)
    if scene.world is None:
        scene.world = bpy.data.worlds.new(name="World")

    # 2. CACHE: Use absolute paths (// paths fail in CLI)
    # Use CACHE_DIR from asset folder, NOT /tmp/

    # 3. DOMAIN: Configure with absolute cache path
    # CRITICAL: You MUST set fluid_type='DOMAIN' BEFORE accessing domain_settings!
    # Without this, domain_settings is None and you'll get AttributeError
    domain = bpy.context.active_object  # Your domain object
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'  # SET THIS FIRST!
    settings = domain.modifiers["Fluid"].domain_settings  # Now this exists
    settings.domain_type = 'GAS'  # For smoke/fire simulations
    settings.cache_directory = CACHE_DIR  # Asset folder, not /tmp/
    settings.cache_data_format = 'OPENVDB'  # Best for volumetrics

    # 4. GPU: Configure Cycles with GPU (10-100x faster than CPU)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
    except Exception:
        pass  # Falls back to CPU if no GPU

def bake_mantaflow(domain_obj):
    # Safe Mantaflow baking for CLI
    scene = bpy.context.scene

    # CRITICAL: Update depsgraph BEFORE baking
    bpy.context.view_layer.update()
    scene.frame_set(scene.frame_start)

    # Select and activate domain
    bpy.ops.object.select_all(action='DESELECT')
    domain_obj.select_set(True)
    bpy.context.view_layer.objects.active = domain_obj

    # Bake with context override
    with bpy.context.temp_override(active_object=domain_obj, object=domain_obj):
        bpy.ops.fluid.bake_all()  # Bakes data + noise + mesh
```

### CLI Mantaflow Checklist:
1. ✅ **World exists** - Create with `bpy.data.worlds.new()` if None
2. ✅ **Absolute cache paths** - Never use `//` relative paths
3. ✅ **Depsgraph update** - Call `view_layer.update()` before bake
4. ✅ **GPU configured** - Set `cycles.device = 'GPU'` + preferences
5. ✅ **No free_all()** - NEVER call `bpy.ops.fluid.free_all()` - causes "grids still in use" crash

### CRITICAL: DO NOT USE free_all()
```python
# WRONG - causes "can't clean grid cache, grids still in use" crash
bpy.ops.fluid.free_all()
bpy.ops.fluid.bake_data()

# CORRECT - just bake directly, skip free_all entirely
bpy.context.view_layer.update()  # Update depsgraph first
bpy.ops.fluid.bake_data()        # Bake directly without free_all
```

### Noise/Upres (Blender 5.0)
- **DO NOT** use `dsettings.noise_res_factor` (removed in 5.0)
- Use:
  - `dsettings.use_noise = True`
  - `dsettings.noise_strength = 0.7`
  - `dsettings.noise_scale = 1.0`
  - `bpy.ops.fluid.bake_data()`
  - `bpy.ops.fluid.bake_noise()`

### Common CLI Errors and Fixes:
| Error | Cause | Fix |
|-------|-------|-----|
| `'NoneType' has no attribute 'resolution_max'` | domain_settings is None | Set `fluid_type='DOMAIN'` BEFORE accessing domain_settings |
| `'NoneType' has no attribute 'domain_type'` | domain_settings is None | Set `fluid_type='DOMAIN'` BEFORE accessing domain_settings |
| `'NoneType' has no attribute 'use_nodes'` | scene.world is None | Create world first |
| `Permission denied: '//vdb_cache'` | Relative path | Use absolute path |
| `can't clean grid cache, grids still in use` | Using `free_all()` | **REMOVE** `bpy.ops.fluid.free_all()` - just bake directly |
| `'FluidDomainSettings' has no attribute 'use_adaptive_time_steps'` | **TYPO** | Correct spelling: `use_adaptive_timesteps` (no underscore between time/steps) |
| `'FluidDomainSettings' has no attribute 'resolution_divisions'` | Wrong attribute | Use `resolution_max` instead |
| Render very slow | CPU rendering | Configure GPU preferences |

### ⚠️ COMMON LLM TYPOS - VERIFY SPELLING!
These attributes are frequently misspelled. **ALWAYS double-check**:
| WRONG (typo) | CORRECT |
|--------------|---------|
| `use_adaptive_time_steps` | `use_adaptive_timesteps` |
| `resolution_divisions` | `resolution_max` |
| `use_dissolve` | `use_dissolve_smoke` (also: `use_dissolve_smoke_log`) |
| `cache_format` | `cache_data_format` |
| `flow.velocity` | `velocity_factor`, `velocity_normal`, `velocity_random` |
| `flame_smoke_color` | `flame_smoke` |
| `noise_res_factor` | REMOVED in Blender 5.0 |

### GPU CONFIGURATION FOR CYCLES (REQUIRED FOR ALL EFFECTS):
**ALWAYS use Cycles with GPU** for best quality. Configure GPU properly:
```python
def setup_cycles_gpu():
    # Configure Cycles to use GPU rendering
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # Enable GPU compute device (try CUDA, then OPTIX, then METAL)
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'CUDA'
    except Exception:
        try:
            prefs.compute_device_type = 'OPTIX'
        except Exception:
            try:
                prefs.compute_device_type = 'METAL'
            except Exception:
                pass  # Fallback to CPU

    # Refresh and enable all GPU devices
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True  # Enable all available devices
```
**CRITICAL**: Without this setup, Cycles runs on CPU which is 10-100x slower.

## MODIFICATION WORKFLOW
1. Parse quality feedback -> identify ALL visual issues
2. Research unknown issues if needed
3. EITHER: modify_script() for parameter tweaks
   OR: write_script() with entirely new code if approach needs change
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
"""


LEARNING_AGENT_BASE_INSTRUCTIONS = """## ROLE
Record experiments, extract patterns, suggest fixes from accumulated knowledge.

## TURN BUDGET
Target: 3 turns | Hard limit: 8 turns (allows pattern extraction if successful)
T1: query_knowledge_base + search_code_patterns (parallel)
T2: record_experiment_result (ONCE)
T3: If score_delta >= 5: extract_successful_pattern -> Return LearningOutput
T3: Otherwise: Return LearningOutput

CRITICAL: Call record_experiment_result ONCE. Never retry on error.

## BEFORE CHANGES
1. search_code_patterns(issue) -> find known fixes
2. query_knowledge_base(issue) -> check past learnings
3. get_parameter_knowledge(param) -> accumulated wisdom

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
        learnings.extend(query_validated_learnings(effect_type, "physics", 0.7))
        learnings.extend(query_validated_learnings(effect_type, "visual", 0.7))

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
1. Call recommend_technique ONCE to pick approach
2. Call generate_script ONCE to create initial script
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
