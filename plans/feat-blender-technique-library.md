# Blender Technique Library - Expanding AI's Creative Range

## Problem Statement

The system produces repetitive Blender assets because:
1. Script Generator relies on a narrow set of templates and patterns
2. Blender 5 is too new for training data
3. The AI doesn't know about techniques it hasn't seen in training

**Validation is the wrong frame.** The system isn't producing invalid scripts - it's producing boring, repetitive ones.

## Solution: Technique Discovery, Not Validation

Use the Blender Manual MCP not to validate, but to **teach** the Script Generator about techniques it doesn't know.

---

## Phase 1: Build Technique Catalog (1 session)

### Extract Technique Variations from Manual

Query the Blender Manual to build a structured catalog of distinct approaches:

```python
# agents/script-generator/technique_catalog.py

PYRO_TECHNIQUES = {
    "rising_mushroom": {
        "description": "Upward mushroom cloud with hot core",
        "key_params": {
            "flame_vorticity": (0.8, 1.2),  # High turbulence
            "flame_smoke": (1.5, 2.0),       # Heavy smoke production
            "burning_rate": (1.0, 1.5),      # Fast combustion
            "emitter_z": "bottom",           # Emit from ground
            "flow_behavior": "INFLOW"
        },
        "visual_signature": "Rising column with curling cap"
    },

    "ground_hugger": {
        "description": "Low, spreading explosion that hugs terrain",
        "key_params": {
            "flame_vorticity": (0.2, 0.4),   # Low turbulence
            "flame_smoke": (2.5, 3.0),       # Very heavy smoke
            "burning_rate": (0.3, 0.6),      # Slow burn
            "emitter_z": "ground",
            "buoyancy": -0.3                 # Negative buoyancy
        },
        "visual_signature": "Spreading ground fog with embedded fire"
    },

    "aerial_burst": {
        "description": "Mid-air explosion expanding spherically",
        "key_params": {
            "flame_vorticity": (0.5, 0.7),
            "flame_smoke": (0.8, 1.2),       # Light smoke
            "burning_rate": (2.0, 3.0),      # Very fast
            "emitter_z": "elevated",
            "flow_behavior": "GEOMETRY"      # Shape-based emission
        },
        "visual_signature": "Spherical fireball with minimal smoke"
    },

    "slow_smolder": {
        "description": "Slow-burning with heavy smoke trails",
        "key_params": {
            "flame_vorticity": (0.1, 0.2),
            "flame_smoke": (3.0, 4.0),       # Maximum smoke
            "burning_rate": (0.1, 0.3),      # Very slow
            "flame_max_temp": (1.5, 2.0),    # Cool flames
        },
        "visual_signature": "Mostly smoke with occasional flame wisps"
    },

    "plasma_jet": {
        "description": "Directed, high-temperature plasma stream",
        "key_params": {
            "flame_vorticity": (1.5, 2.0),   # Maximum turbulence
            "flame_smoke": (0.1, 0.3),       # Minimal smoke
            "burning_rate": (3.0, 4.0),      # Maximum combustion
            "flame_max_temp": (8.0, 10.0),   # Extremely hot
            "flow_behavior": "INFLOW"
        },
        "visual_signature": "Intense directed flame with no smoke"
    }
}

# Discovered from Blender Manual - techniques most AI won't know
ADVANCED_TECHNIQUES = {
    "adaptive_domain": {
        "source": "physics/fluid/type/domain/settings.html",
        "description": "Dynamic domain that expands with simulation",
        "benefit": "Allows unbounded effects without wasted voxels"
    },

    "noise_injection": {
        "source": "physics/fluid/type/domain/gas.html",
        "description": "Add procedural noise to existing simulation",
        "benefit": "Adds detail without increasing resolution"
    },

    "colored_smoke": {
        "source": "physics/fluid/type/flow.html",
        "description": "Per-emitter smoke color via flow settings",
        "benefit": "Multi-colored effects from single simulation"
    },

    "effector_guides": {
        "source": "physics/fluid/type/effector.html",
        "description": "Guide smoke/fire along surfaces",
        "benefit": "Direct simulation flow without visible geometry"
    }
}
```

### How to Populate This

Run these queries against Blender Manual MCP once and cache results:

```python
# One-time extraction script
from blender_manual import search_python_api, read_page

# Get all gas domain parameters
gas_params = search_python_api("FluidDomainSettings gas")
# Get all flow types
flow_types = search_python_api("FluidFlowSettings flow_type")
# Get effector documentation
effectors = read_page("physics/fluid/type/effector.html")
```

---

## Phase 2: Technique-Aware Generation (1 session)

### Modify Script Generator to Select Technique First

Current behavior:
```
User: "Create an explosion"
Generator: [uses same template every time]
```

New behavior:
```
User: "Create an explosion"
Generator: [selects from 5+ distinct techniques]
          [applies technique-specific parameters]
          [varies within technique's parameter ranges]
```

### Implementation

```python
# agents/script-generator/server.py - modify generate_script()

import random
from technique_catalog import PYRO_TECHNIQUES

def generate_script(effect_type: str, description: str, ...):
    # BEFORE: Always used similar parameters
    # AFTER: Select technique based on description or randomly

    technique = select_technique(effect_type, description)

    # Apply technique's parameter ranges with randomization
    params = {}
    for param, range_or_value in technique["key_params"].items():
        if isinstance(range_or_value, tuple):
            # Randomize within range
            params[param] = random.uniform(range_or_value[0], range_or_value[1])
        else:
            params[param] = range_or_value

    # Generate script with technique-specific parameters
    return generate_from_template(effect_type, params, technique)

def select_technique(effect_type: str, description: str) -> dict:
    """Select technique based on description keywords or randomly."""

    if effect_type == "pyro":
        # Keyword matching
        if "mushroom" in description.lower() or "rising" in description.lower():
            return PYRO_TECHNIQUES["rising_mushroom"]
        elif "ground" in description.lower() or "spreading" in description.lower():
            return PYRO_TECHNIQUES["ground_hugger"]
        elif "plasma" in description.lower() or "jet" in description.lower():
            return PYRO_TECHNIQUES["plasma_jet"]
        else:
            # Random selection encourages variety
            return random.choice(list(PYRO_TECHNIQUES.values()))

    return default_technique(effect_type)
```

---

## Phase 3: Experiment-Driven Technique Discovery (ongoing)

### Learn Which Techniques Work Best

The Experiment Tracker already records parameter→outcome. Extend it to track techniques:

```python
# In experiment tracker
def record_experiment_result(..., technique_name: str):
    # Record which technique was used
    # Over time, build statistics on which techniques produce best scores
```

### Automatic Technique Expansion

When scores plateau (detected via `get_alternative_approaches()`):

```python
def suggest_technique_change(current_technique: str, scores_history: list):
    if is_plateaued(scores_history):
        # Suggest a DIFFERENT technique, not just parameter tweaks
        alternatives = [t for t in PYRO_TECHNIQUES if t != current_technique]
        return random.choice(alternatives)
```

---

## Phase 4: Manual-Guided Exploration (optional enhancement)

### Query Manual for Unknown Techniques

When all catalog techniques have been tried:

```python
async def discover_new_technique(effect_type: str):
    """Query Blender Manual for techniques not in catalog."""

    # Search for tutorials and advanced features
    results = await mcp__blender-manual__search_tutorials(
        topic=effect_type,
        technique="advanced"
    )

    # Parse results into new technique entry
    new_technique = parse_tutorial_to_technique(results)

    # Add to catalog
    PYRO_TECHNIQUES[new_technique["name"]] = new_technique
```

This creates a self-expanding knowledge base.

---

## Why This Works Better Than Validation

| Approach | Addresses Repetition? | Addresses Blender 5? | Complexity |
|----------|----------------------|---------------------|------------|
| Parameter Validation | ❌ No | ✅ Yes | High |
| Technique Library | ✅ Yes | ✅ Yes | Low |

**Validation** ensures correctness but not variety. You can validate 100 identical scripts.

**Technique Library** ensures variety by design. Each technique is categorically different.

---

## Implementation Effort

| Phase | Effort | Impact |
|-------|--------|--------|
| Phase 1: Build catalog | 2-3 hours | Immediate variety |
| Phase 2: Technique selection | 1-2 hours | Automatic diversity |
| Phase 3: Learning integration | 1 hour | Continuous improvement |
| Phase 4: Auto-discovery | Optional | Self-expanding |

**Total: 4-6 hours for significant improvement**

---

## Immediate Actions

1. Query Blender Manual for all gas domain parameters and their ranges
2. Identify 5-8 categorically different pyro techniques
3. Build initial `technique_catalog.py`
4. Modify `generate_script()` to use technique selection
5. Test variety by generating 5 explosions and verifying they're distinct

---

## Success Criteria

- [ ] 5 explosions generated with same prompt produce 5 visually distinct results
- [ ] Each technique uses different parameter ranges (not just values)
- [ ] System can explain why it chose a technique
- [ ] Experiment Tracker records technique names for learning
