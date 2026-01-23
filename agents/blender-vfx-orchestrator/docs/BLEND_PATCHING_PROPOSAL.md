# .blend File Patching & Analysis Proposal

**Date:** 2026-01-22
**Status:** Ready for Implementation (Phase 2 Complete ✅)
**Priority:** High Impact, Medium Effort
**Last Updated:** 2026-01-23

---

## Executive Summary

Instead of recreating Blender scenes from scratch on each iteration, the orchestrator could **load and patch existing .blend files**. This enables:

1. **Faster iterations** - Modify only what's broken, not rebuild everything
2. **Learning from examples** - Analyze .blend files from online sources
3. **Knowledge extraction** - Reverse-engineer techniques and save to knowledge base
4. **Scene manipulation** - Add objects, modify materials, adjust parameters on existing scenes

---

## Core Concept: Iteration Patching

### Current Approach (Inefficient)

```
Iteration 1: Generate full script → Create scene from scratch → Render
Iteration 2: Generate full script → Create scene from scratch → Render
Iteration 3: Generate full script → Create scene from scratch → Render
...
```

**Problems:**
- Rebuilds entire scene even if only one parameter needs changing
- 20-30 seconds per iteration just for scene setup
- Risk of introducing new bugs while recreating working parts
- No continuity between iterations

### Proposed Approach (Patching)

```
Iteration 1: Generate full script → Create scene → Save .blend → Render
Iteration 2: Load .blend → Apply targeted patch → Save .blend → Render
Iteration 3: Load .blend → Apply targeted patch → Save .blend → Render
...
```

**Benefits:**
- Only modify what Quality Analyst flagged
- 2-3 seconds to load and patch vs 20-30s to recreate
- Preserves working parts of the scene
- Can compare iterations by loading different .blend files

---

## Blender CLI for Patching

### Basic Commands

```bash
# Create new scene
blender --background --python create_sun.py -- --output sun_v1.blend

# Patch existing scene
blender sun_v1.blend --background --python patch_emission.py -- --strength 0.7

# Analyze existing scene
blender downloaded_sun.blend --background --python analyze_scene.py
```

### Patch Script Structure

```python
"""patch_emission.py - Reduce emission strength on sun core"""
import bpy
import sys

def get_arg(name, default=None):
    """Get command line argument."""
    argv = sys.argv
    if f"--{name}" in argv:
        idx = argv.index(f"--{name}")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default

def main():
    # Scene is already loaded (from .blend file)
    strength = float(get_arg("strength", "0.8"))

    # Find the sun object
    sun = bpy.data.objects.get("Sun")
    if sun and sun.active_material:
        # Find emission node
        for node in sun.active_material.node_tree.nodes:
            if node.type == 'EMISSION':
                node.inputs['Strength'].default_value = strength
                print(f"[patch] Emission strength set to {strength}")
                break

    # Save the patched scene
    output = get_arg("output", bpy.data.filepath.replace(".blend", "_patched.blend"))
    bpy.ops.wm.save_as_mainfile(filepath=output)
    print(f"[patch] Saved: {output}")

if __name__ == "__main__":
    main()
```

---

## Extended Applications

### 1. Learning from Online .blend Files

Download high-quality .blend files from:
- Blender Market
- Blender Artists forum
- CGTrader / TurboSquid (free models)
- Blender demo files

**Workflow:**
```
Download .blend → Analyze with agent → Extract techniques → Save to knowledge base
```

**Analysis Script Example:**
```python
"""analyze_scene.py - Extract techniques from a .blend file"""
import bpy
import json

def analyze_materials():
    """Extract material node setups."""
    materials = []
    for mat in bpy.data.materials:
        if mat.node_tree:
            nodes = []
            for node in mat.node_tree.nodes:
                nodes.append({
                    "type": node.type,
                    "name": node.name,
                    "inputs": {inp.name: getattr(inp, 'default_value', None)
                              for inp in node.inputs if hasattr(inp, 'default_value')},
                })
            materials.append({
                "name": mat.name,
                "nodes": nodes,
                "links": len(mat.node_tree.links)
            })
    return materials

def analyze_modifiers():
    """Extract modifier stacks."""
    modifiers = {}
    for obj in bpy.data.objects:
        if obj.modifiers:
            modifiers[obj.name] = [
                {"type": mod.type, "name": mod.name}
                for mod in obj.modifiers
            ]
    return modifiers

def analyze_compositor():
    """Extract compositor node setup."""
    if bpy.context.scene.use_nodes:
        tree = bpy.context.scene.node_tree
        return {
            "nodes": [{"type": n.type, "name": n.name} for n in tree.nodes],
            "links": len(tree.links)
        }
    return None

def main():
    analysis = {
        "file": bpy.data.filepath,
        "objects": len(bpy.data.objects),
        "materials": analyze_materials(),
        "modifiers": analyze_modifiers(),
        "compositor": analyze_compositor(),
        "render_engine": bpy.context.scene.render.engine,
        "resolution": [
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y
        ],
    }

    print(json.dumps(analysis, indent=2, default=str))

if __name__ == "__main__":
    main()
```

### 2. Technique Extraction for Knowledge Base

When analyzing a high-quality sun .blend file:

```json
{
  "technique_id": "sun_layered_emission_compositor_glare",
  "source": "downloaded_sun_example.blend",
  "effect_type": "sun",
  "quality_score": 85,
  "components": {
    "geometry": "UV sphere with subdivision",
    "materials": [
      {
        "name": "Sun_Core",
        "technique": "Emission + Noise texture for surface detail",
        "key_params": {
          "emission_strength": 50,
          "noise_scale": 4.0,
          "color_ramp": "orange_to_white"
        }
      },
      {
        "name": "Sun_Corona",
        "technique": "Volume scatter with density falloff",
        "key_params": {
          "density": 0.1,
          "anisotropy": 0.3
        }
      }
    ],
    "compositor": {
      "technique": "Glare node (Fog Glow) + Color correction",
      "key_params": {
        "glare_type": "FOG_GLOW",
        "threshold": 0.5,
        "size": 8
      }
    }
  },
  "learnings": [
    "Surface detail comes from noise texture driving emission color",
    "Corona uses separate mesh with volume shader",
    "Glare threshold should be below peak emission to catch falloff"
  ]
}
```

### 3. A/B Testing with Patching

```
Original .blend → Patch A (increase emission) → Render A
Original .blend → Patch B (add noise texture) → Render B
Original .blend → Patch C (adjust compositor) → Render C

Compare renders → Pick best → Record winning parameters
```

### 4. Scene Composition

Combine elements from multiple .blend files:

```python
# Load base scene
bpy.ops.wm.open_mainfile(filepath="base_space.blend")

# Append sun from another file
bpy.ops.wm.append(
    filepath="sun_v3.blend/Object/Sun",
    directory="sun_v3.blend/Object/",
    filename="Sun"
)

# Append nebula from another file
bpy.ops.wm.append(
    filepath="nebula_v2.blend/Object/Nebula",
    directory="nebula_v2.blend/Object/",
    filename="Nebula"
)

# Save composed scene
bpy.ops.wm.save_as_mainfile(filepath="composed_scene.blend")
```

---

## Integration with Orchestrator

### Phase 1: Basic .blend Saving (Foundation)

Add to all generated scripts:
```python
# At end of script
blend_path = f"{Config.OUTPUT_DIR}/{Config.ASSET_NAME}.blend"
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"[script] Saved: {blend_path}")
```

### Phase 2: Patch Script Generation

Script Writer agent generates targeted patches:

```python
# Instead of full script for iteration 2+
PATCH_TEMPLATES = {
    "reduce_emission": "patch_emission_strength.py",
    "add_surface_noise": "patch_add_noise.py",
    "adjust_compositor_glare": "patch_glare_settings.py",
    "modify_color_ramp": "patch_color_ramp.py",
}
```

### Phase 3: Blend Analyzer Agent

New specialized agent:
```python
blend_analyzer = Agent(
    name="Blend Analyzer",
    instructions="""You analyze .blend files to extract techniques.

    For each .blend file:
    1. List all objects and their purposes
    2. Analyze material node setups
    3. Document modifier stacks
    4. Extract compositor configuration
    5. Identify key parameters that affect quality

    Output a structured analysis suitable for the knowledge base.""",
    tools=[
        analyze_blend_file,
        extract_material_setup,
        extract_compositor_setup,
        save_technique_to_kb,
    ]
)
```

### Phase 4: Learning from Examples

Workflow for processing downloaded .blend files:

```
1. User provides .blend file URL or path
2. Blend Analyzer examines the file
3. Techniques extracted and rated
4. High-quality techniques saved to knowledge base
5. Future generations can reference these techniques
```

---

## Implementation Priority

| Phase | Task | Effort | Impact | Dependency |
|-------|------|--------|--------|------------|
| 1 | Add .blend saving to scripts | 1h | Foundation | None |
| 2 | Create patch script templates | 3h | Faster iterations | Phase 1 |
| 3 | Blend Analyzer agent | 4h | Learning capability | Phase 2 |
| 4 | Knowledge base integration | 2h | Self-improvement | Phase 3 |
| 5 | Online .blend ingestion | 3h | External learning | Phase 3, 4 |

**Total: ~13 hours**

---

## File Structure Proposal

```
assets/
├── blender_scripts/
│   ├── generated/           # Full scripts (iteration 1)
│   │   └── sun_v1.py
│   ├── patches/             # Patch scripts (iteration 2+)
│   │   ├── patch_emission_strength.py
│   │   ├── patch_add_noise.py
│   │   └── patch_glare_settings.py
│   └── analysis/            # Analysis scripts
│       └── analyze_scene.py
├── blend_files/
│   ├── generated/           # .blend files from our scripts
│   │   └── sun_v1.blend
│   ├── downloaded/          # .blend files from online
│   │   └── pro_sun_example.blend
│   └── analyzed/            # Analysis results
│       └── pro_sun_example_analysis.json
```

---

## Potential Use Cases

### 1. Rapid Iteration
"The sun core is too bright" → Load .blend → Patch emission → Re-render (5 seconds vs 30 seconds)

### 2. Learn from Pros
Download professional VFX artist's .blend → Analyze → Extract techniques → Apply to future generations

### 3. Scene Libraries
Build library of reusable .blend components (sun, nebula, explosion) → Compose complex scenes

### 4. Version Control
Save each iteration as separate .blend → Compare versions → Revert if needed

### 5. User Customization
User provides their own .blend → Agent enhances it with VFX → Returns improved version

### 6. Technique Archaeology
"How did they achieve this effect?" → Analyze .blend → Document technique → Teach to other agents

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Large .blend files (100MB+) | Compress, store only essential data |
| Incompatible Blender versions | Validate version on load, convert if needed |
| Corrupted .blend files | Validate before processing, keep backups |
| Licensing issues with downloaded files | Only use CC0/public domain for KB |
| Complexity of patch scripts | Start simple, expand based on common issues |

---

## Success Metrics

| Metric | Current | With Patching |
|--------|---------|---------------|
| Iteration time | 30-60s | 5-10s |
| Techniques in KB | ~10 | 100+ (from analysis) |
| Scene rebuild bugs | Common | Rare (preserves working parts) |
| Learning from external sources | None | Active |

---

## Next Steps (Phase 2 Complete ✅)

Phase 2 (agents-as-tools with Coordinators) is now complete. The patching system can integrate with the existing architecture:

1. **Immediate:** Add `bpy.ops.wm.save_as_mainfile()` to script generation
   - Modify Script Writer to save .blend after each iteration
   - Store path in SessionState for reference

2. **Short-term:** Create basic patch script templates for common fixes
   - `patch_emission_strength.py` - Adjust emission levels
   - `patch_color_ramp.py` - Modify color gradients
   - `patch_volume_density.py` - Adjust smoke/fire density

3. **Medium-term:** Build Blend Analyzer agent using `as_tool()` pattern
   - Create `blend_analyzer.py` in `specialized_agents/`
   - Wrap as tool for Coordinator: `analyzer.as_tool("analyze_blend", ...)`
   - Integrate with TechniqueSelector for learning from examples

4. **Long-term:** Implement external .blend ingestion and learning pipeline
   - Download .blend files from approved sources
   - Analyze with Blend Analyzer agent
   - Extract techniques to knowledge base

### Integration with Coordinator Architecture

The Coordinators can leverage .blend patching:

```python
# TechniqueSelector can reference analyzed .blend files
technique_prompt = f"""Select technique based on:
- Research findings: {research_text}
- Analyzed .blend techniques: {analyzed_techniques}
"""

# ModificationStrategist can suggest patch vs regenerate
mod_decision = ModificationDecision(
    action='patch_blend',  # New action type
    patch_script='patch_emission_strength.py',
    parameter_changes={'strength': 0.7},
)
```

---

*Phase 2 (agents-as-tools) is complete. This proposal is ready for implementation as a future enhancement.*
