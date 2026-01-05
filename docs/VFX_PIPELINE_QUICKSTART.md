# VFX Pipeline Quick Start Guide

**For:** Claude Code operators and agents
**Version:** 2.0 (GPT-5.2 Integration)

---

## The #1 Rule

**ALWAYS use `blender-librarian` before modifying Blender parameters.**

This ensures you're using current Blender 5.0 documentation, not stale training data.

---

## Quick Start: Create an Explosion

```python
# Step 1: Research (NEVER SKIP)
mcp__blender-librarian__get_modification_advice(
    effect_type="explosion",
    issues='["need bright flames", "want mushroom cloud shape"]',
    current_params='{}'
)

# Step 2: Generate script
mcp__script-generator__generate_script(
    effect_type="pyro",
    description="bright orange mushroom cloud explosion",
    output_name="explosion_v1",
    resolution=96,
    frame_end=50
)

# Step 3: Validate
mcp__script-generator__validate_script(
    script_path="assets/blender_scripts/generated/explosion_v1.py"
)

# Step 4: Execute
mcp__blender-executor__execute_blender_script(
    script_path="assets/blender_scripts/generated/explosion_v1.py",
    script_args={"--bake": "1"},
    output_dir="build/vdb_output/explosion_v1"
)

# Step 5: Evaluate
mcp__asset-evaluator__evaluate_vfx_quality(
    image_path="build/vdb_output/explosion_v1/render_0050.png",
    effect_type="explosion"
)
```

---

## Quick Start: Create a Sun

```python
# Step 1: Research limb darkening, prominences, etc.
mcp__blender-librarian__get_modification_advice(
    effect_type="sun",
    issues='["need limb darkening", "want solar prominences"]',
    current_params='{}'
)

# Step 2: Generate
mcp__script-generator__generate_script(
    effect_type="pyro",
    description="realistic sun with limb darkening and prominences",
    output_name="sun_v1",
    resolution=128,
    frame_end=120
)

# Step 3-4: Validate and Execute (same as above)

# Step 5: Evaluate against NASA reference
mcp__asset-evaluator__evaluate_ground_truth(
    image_path="build/vdb_output/sun_v1/render_0060.png",
    effect_type="sun",
    pass_threshold=0.65
)
```

---

## When Stuck: Checklist

1. **Score not improving?**
   ```python
   # Try different technique
   mcp__script-generator__recommend_technique(
       effect_type="pyro",
       description="...",
       prefer_untried=True
   )
   ```

2. **Unknown parameter effect?**
   ```python
   # Look it up!
   mcp__blender-manual__search_bpy_types(typename="FluidDomainSettings")
   ```

3. **Texture looks synthetic?**
   ```python
   mcp__asset-evaluator__analyze_texture_procedural(render_path="...")
   ```

4. **Need to resume later?**
   ```python
   # Save state
   mcp__iteration-controller__save_iteration_state(
       session_id="my_asset",
       ...
   )

   # Resume
   mcp__iteration-controller__load_iteration_state(session_id="my_asset")
   ```

---

## Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Skip documentation lookup | Wrong parameters, wasted iterations | Always use blender-librarian first |
| Guess parameter ranges | Blender errors or silent failures | Use `validate_parameters()` |
| Don't record experiments | Repeat same failures | Use experiment-tracker |
| Same technique repeatedly | Stuck in local optima | Use `recommend_technique()` |

---

## Key Thresholds

| Metric | Pass | Good | Excellent |
|--------|------|------|-----------|
| VFX Quality Score | ≥60 | ≥75 | ≥85 |
| Ground Truth (sun) | ≥0.65 | ≥0.75 | ≥0.85 |
| Procedural Score | <50 | <30 | <15 |

---

## Documentation Search Cheat Sheet

| Need | Tool | Example |
|------|------|---------|
| Natural language question | `search_semantic` | "how to make smoke rise faster" |
| Specific API | `search_bpy_types` | "FluidDomainSettings" |
| Operator reference | `search_bpy_operators` | category="fluid", operation="bake" |
| VDB/volume stuff | `search_vdb_workflow` | "export openvdb" |
| Shader nodes | `search_nodes` | "Principled Volume" |
| Full page content | `read_page` | "physics/fluid/type/domain/cache.html" |

---

## See Also

- Full manual: `docs/VFX_PIPELINE_OPERATOR_MANUAL_V2.md`
- Architecture: `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md`
- GPT-5.2 design: `docs/BLENDER_LIBRARIAN_AGENT_DESIGN_GPT52.md`
