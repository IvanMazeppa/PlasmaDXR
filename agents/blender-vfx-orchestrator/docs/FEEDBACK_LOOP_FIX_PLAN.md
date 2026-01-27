# Feedback Loop Fix Plan: Learning-Based Approach

**Problem**: Learning Agent suggestions (`parameter_modifications`) don't reach shader node values because `modify_script` only handles Config class and direct settings patterns.

**Goal**: Make agents LEARN what works, not hardcode fixes.

---

## Current State (Broken)

```
Quality Analyst → "density too low"
    ↓
Learning Agent → {"density": 15.0}
    ↓
modify_script → changes Config.DENSITY = 15.0
    ↓
BUT: volume.inputs['Density'].default_value = 10.0 (UNCHANGED)
    ↓
Render → Still too dark (no improvement)
```

---

## Solution Overview (SICA + Self-Evolving Pattern)

### Phase 1: Script Structure Analysis Tool

Add a tool that the Learning Agent can call to understand script structure BEFORE suggesting modifications.

```python
@function_tool
async def analyze_script_modifiable_patterns(script_path: str) -> str:
    """
    Analyze a Blender script to identify all modifiable value patterns.

    Returns categorized list of patterns the script uses:
    - config_values: {"DENSITY": {"line": 42, "current": 10.0}}
    - settings_assignments: {"settings.density": {"line": 103, "current": 5.0}}
    - shader_node_inputs: {"volume.inputs['Density']": {"line": 215, "current": 10.0}}

    Use this BEFORE suggesting parameter_modifications to ensure your
    suggestions match the script's actual structure.
    """
    # Parse script and extract all value assignment patterns
    # Return structured data about what CAN be modified
```

### Phase 2: Modification Outcome Tracking

Track whether modifications actually changed output values (not just wrote to file).

```python
@function_tool
async def report_modification_outcome(
    modification: str,  # "density: 15.0"
    pattern_used: str,  # "config_class" | "settings_attr" | "shader_node"
    visual_change_observed: bool,  # Did render look different?
    score_before: float,
    score_after: float
) -> str:
    """
    Report whether a modification attempt actually worked.

    This builds the knowledge base of:
    - Which modification patterns actually change visual output
    - Which parameters have real visual impact
    - Which suggestions are "no-ops" that don't reach the render
    """
```

### Phase 3: Learning Agent Script Analysis Prompt

Update Learning Agent to analyze script before suggesting:

```
## BEFORE SUGGESTING parameter_modifications

1. Call analyze_script_modifiable_patterns(current_script_path)
2. Review the patterns available in the script
3. ONLY suggest modifications that match existing patterns:
   - If script uses Config.DENSITY → suggest {"DENSITY": 15.0}
   - If script uses settings.density → suggest {"settings.density": 15.0}
   - If shader nodes use hardcoded values → suggest the EXACT pattern:
     {"volume.inputs['Density'].default_value": 15.0}

4. If the parameter you want to change doesn't exist in script:
   - Recommend 'switch_technique' or 'regenerate_script'
   - Or provide custom_code injection pattern
```

### Phase 4: Self-Evolving Modification Strategy

Based on SICA meta-loop:

```python
class ModificationStrategyEvolver:
    """
    Tracks which modification strategies work for which issues.

    After each iteration:
    1. Record: issue + modification_type + pattern_used + score_delta
    2. Build success rates: {(issue, pattern_type): success_rate}
    3. Inject into Learning Agent instructions: "For 'too dark' issues,
       modifying shader_node patterns has 85% success rate vs 12% for config_class"
    """

    async def get_best_strategy(self, issue: str, script_patterns: List[str]) -> str:
        """Return the modification strategy with highest success rate for this issue."""
```

---

## Implementation Sequence

### Step 1: Add Script Analysis Tool
- Location: `tools/script_analysis_tools.py`
- Parse script AST to find all modifiable patterns
- Categorize: Config class, settings assignments, shader nodes, custom patterns
- Return structured JSON that Learning Agent can reason about

### Step 2: Enhance modify_script for Shader Nodes
- Add pattern recognition for `node.inputs['X'].default_value = Y`
- NOT hardcoded fixes - just extend the modification capability
- The Learning Agent decides WHAT to modify; modify_script just handles HOW

### Step 3: Add Modification Outcome Tracking
- After each iteration, compare what was suggested vs what actually changed
- Build knowledge base: {modification_type: success_rate}
- Identify "no-op" patterns that never improve scores

### Step 4: Dynamic Learning Agent Instructions
- Inject best modification strategies from knowledge base
- Example: "For fire effects with 'too dark' issues, shader node modifications
  have 78% success rate. Config-only modifications have 15% success rate."

### Step 5: SICA Meta-Loop (Archive Review)
- Before iteration N, review iterations 1 to N-1
- Identify: What modifications worked? What were no-ops?
- Avoid repeating ineffective patterns

---

## Success Criteria

The system is working when:
1. Learning Agent suggestions match actual script patterns (no more no-ops)
2. Score improvements correlate with modifications actually being applied
3. Knowledge base accumulates: {issue_type + script_pattern → effective_modification}
4. Subsequent sessions benefit from past learning

---

## Key Principle: No Hardcoded Rules

Every improvement comes from:
1. Observing what works (modification outcome tracking)
2. Building knowledge base (success rates per pattern type)
3. Injecting learnings into agent instructions (dynamic instructions)
4. Agents making better decisions based on accumulated knowledge

NOT from:
- Hardcoded regex patterns for specific fixes
- Pre-defined parameter boost values
- Static rules that bypass agent reasoning
