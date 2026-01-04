# Sun Rendering Issues Analysis & Blender Librarian Agent Proposal

**Document Version:** 1.0
**Date:** 2026-01-04
**Purpose:** Detail current sun rendering pipeline issues and propose GPT-5.2 "Blender Librarian" agent
**For:** Compilation with Gemini feedback into Multi-Agent Improvement Plan V4

---

## Executive Summary

Our VFX asset generation pipeline has hit a fundamental knowledge gap when working with Blender 5.0.1's volumetric rendering system. Despite having sophisticated ML-based evaluation tools (LPIPS, CLIP, ground truth comparison), we lack the domain expertise to translate evaluation feedback into actionable Blender parameter changes. This document details the specific issues encountered and proposes a "Blender Librarian" agent trained on the complete Blender 5.0.1 documentation and bpy API.

---

## Part 1: Current Issues in Detail

### 1.1 The Core Problem: Viewport vs F12 Render Discrepancy

**Symptom:** User observes beautiful orange sun in Blender's viewport "Rendered" mode, but F12 final render produces completely blown-out white image.

**Investigation Results:**
- Viewport preview uses different sampling/resolution than final render
- Temperature attribute from VDB cache contains extremely high values
- When multiplied by blackbody intensity, emission values exceed displayable range
- Color management (Standard vs Filmic) handles HDR differently

**Why This Matters:** This discrepancy makes iterative tuning nearly impossible. Users cannot trust viewport previews to predict final output.

### 1.2 The Warm Ratio Problem

**Target Metric:** NASA SDO solar footage has warm_ratio ≈ 19.16
**Current Best Result:** warm_ratio ≈ 68.99 (3.6× too warm/orange)

**What is warm_ratio?** Ratio of warm pixels (red/orange dominated) to cool pixels (blue/white dominated). Lower = more white/yellow, Higher = more orange/red.

**Attempted Solutions and Results:**

| Attempt | Scattering Color | Blackbody | warm_ratio | Issue |
|---------|------------------|-----------|------------|-------|
| User's original | (1.0, 0.11, 0.0) | 0.77 | N/A | Completely white blowout |
| Orange + tiny BB | (1.0, 0.3, 0.05) | 0.0001 | 96.17 | Too orange |
| Yellow shift | (1.0, 0.7, 0.4) | 0.01 | 70.02 | Still too orange |
| White-hot | (1.0, 0.95, 0.85) | 0.01 | 68.99 | Blackbody overrides color |
| White emission | (1.0, 0.95, 0.9) | 0.0 | 1.00 | Too white, no warmth |

**Key Insight:** We're oscillating between extremes because we don't understand the interaction between:
- Scattering color (affects scattered light)
- Blackbody intensity (temperature-driven emission)
- Temperature attribute (from VDB simulation)
- Emission color/strength (direct emission)
- Color management pipeline (exposure, gamma, view transform)

### 1.3 Color Temperature Mismatch

**Target:** Real sun is ~5778K (appears white-yellow)
**Current Results:** 3100-3200K (appears orange-red) when warm, or exactly 5778K when completely desaturated

**The Paradox:**
- When we achieve correct color temperature (5778K), the image is completely desaturated white
- When we have visible structure and color, the temperature reads 3100K (too orange)
- We cannot find the middle ground

### 1.4 Missing Limb Darkening

**What it is:** Real sun's edges appear ~40% darker than center due to viewing angle through atmosphere.

**Current State:** Our renders show uniform brightness across the disk, or inverse limb darkening (edges brighter than center).

**Why we can't fix it:** We don't know how to implement radial falloff in Blender's Principled Volume shader. Options might include:
- Gradient texture mapped to object coordinates?
- Shader nodes with distance-from-center calculation?
- Post-processing in compositor?
- Different density field in simulation?

We lack the Blender-specific knowledge to implement this.

### 1.5 Structure vs Brightness Trade-off

**Observation:**
- High blackbody/emission = visible structure but blown out colors
- Low blackbody/emission = correct colors but invisible/black volume
- Zero emission = nearly black image (scattering alone insufficient)

**Root Cause Unknown:** Is this:
- A limitation of Principled Volume shader?
- Incorrect density field values in VDB?
- Wrong approach entirely (should use different shader setup)?
- Color management misconfiguration?

---

## Part 2: Knowledge Gaps Identified

### 2.1 Principled Volume Shader Internals

We don't fully understand:
- How "Color" input interacts with "Emission Color"
- How "Blackbody Intensity" multiplies with "Temperature Attribute"
- The physical units expected (Kelvin? Normalized? Arbitrary?)
- How density affects both scattering AND emission
- The role of "Anisotropy" in volumetric appearance

### 2.2 VDB/OpenVDB Data Interpretation

Questions unanswered:
- What range should temperature values be in the VDB?
- Should we normalize or clamp temperature during bake?
- How does Mantaflow's FIRE simulation populate temperature grid?
- What's the relationship between flame_max_temp and exported temperature values?

### 2.3 Color Management Pipeline

Unclear interactions:
- Standard vs Filmic vs AgX view transforms for volumetrics
- How exposure affects volumetric emission differently than surfaces
- Gamma's role in volumetric rendering
- Whether "film_transparent" affects color calculations

### 2.4 Blender 5.0 API Changes

Known issues:
- `Material.use_nodes` deprecated (warning appears)
- Potential other API changes from 4.x we're unaware of
- New features in 5.0 we could leverage but don't know about

---

## Part 3: Current Tooling Limitations

### 3.1 What We Have

| Tool | Capability | Limitation |
|------|------------|------------|
| asset-evaluator | ML metrics (LPIPS, CLIP, warm_ratio) | Cannot suggest Blender fixes |
| script-generator | Creates Blender Python scripts | Uses templates, lacks deep understanding |
| blender-executor | Runs scripts, captures output | No interpretation of results |
| experiment-tracker | Records parameter→result mappings | Doesn't know WHY things work |
| blender-manual MCP | Searches documentation | Keyword-based, no semantic understanding |

### 3.2 The Gap

**Current Flow:**
1. Evaluate render → "warm_ratio too high (68 vs 19)"
2. ??? (How to fix?)
3. Guess at parameter changes
4. Re-render and evaluate
5. Often makes things worse

**Missing Link:** Translation layer from "what's wrong" to "how to fix it in Blender"

### 3.3 blender-manual MCP Limitations

The existing MCP server provides:
- Keyword search across manual HTML
- Python API documentation lookup
- Tutorial/guide discovery

But it lacks:
- **Semantic understanding** of volumetric rendering concepts
- **Cross-referencing** between manual sections
- **Vision capability** to analyze render results
- **Contextual advice** based on specific symptoms
- **Pre-loaded knowledge** (must search each time)
- **Blender version awareness** (5.0.1 specific behaviors)

---

## Part 4: Proposed Solution - GPT-5.2 "Blender Librarian" Agent

### 4.1 Concept Overview

A specialized agent that:
1. **Has pre-trained knowledge** of entire Blender 5.0.1 manual and bpy API
2. **Uses vision** to analyze render outputs and identify issues
3. **Provides actionable advice** with specific parameter values and code
4. **Understands causality** - knows which parameters affect which visual properties
5. **Maintains context** of our specific pipeline (Mantaflow → VDB → Cycles)

### 4.2 Training Data Sources

| Source | Content | Purpose |
|--------|---------|---------|
| Blender 5.0.1 Manual | All HTML pages (~2000+) | Conceptual understanding |
| bpy API Reference | All modules, classes, properties | Code generation |
| Blender Stack Exchange | Q&A pairs | Problem→solution mapping |
| Blender Artists Forum | Real-world workflows | Practical techniques |
| Our experiment history | Parameter→result logs | Project-specific tuning |

### 4.3 Key Capabilities Required

#### 4.3.1 Vision-Based Analysis
```
Input: Render image + target reference image
Output:
- Identified discrepancies (color, brightness, structure)
- Root cause hypothesis (shader settings, simulation params, color management)
- Specific fix recommendations with code
```

#### 4.3.2 Semantic Query Understanding
```
Query: "My volumetric sun is too orange, how do I make it more white-yellow?"
Response:
1. Diagnosis: Blackbody emission at low temperatures produces orange
2. Fix options:
   a) Increase temperature values (requires rebake)
   b) Reduce blackbody intensity, increase emission color brightness
   c) Add blue channel to scattering color
3. Code example for option (b):
   volume.inputs['Blackbody Intensity'].default_value = 0.001
   volume.inputs['Emission Color'].default_value = (1.0, 0.98, 0.95, 1.0)
   volume.inputs['Emission Strength'].default_value = 2.0
```

#### 4.3.3 Causal Knowledge Graph
```
Parameter: blackbody_intensity
Affects:
  - Emission brightness (proportional)
  - Color temperature appearance (indirect)
  - HDR range (can cause blowout)
Interactions:
  - Multiplied by temperature_attribute values
  - Competes with emission_color
  - Affected by view_transform
Common Issues:
  - Values > 0.1 often cause blowout with high temperature VDBs
  - Very low values (< 0.001) may produce no visible emission
Recommended Range: 0.001 - 0.05 for solar simulations
```

#### 4.3.4 Version-Specific Awareness
```
Blender 5.0.1 Changes:
- Material.use_nodes deprecated (use mat.node_tree directly)
- New Principled Volume inputs: [list]
- Mantaflow improvements: [list]
- Known bugs: [list]
```

### 4.4 Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Iteration Controller                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Script     │    │   Blender    │    │    Asset     │  │
│  │  Generator   │    │   Executor   │    │  Evaluator   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           GPT-5.2 BLENDER LIBRARIAN                 │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Pre-trained Knowledge Base                  │   │   │
│  │  │  - Blender 5.0.1 Manual (embedded)          │   │   │
│  │  │  - bpy API Reference (embedded)             │   │   │
│  │  │  - Stack Exchange Q&A (fine-tuned)          │   │   │
│  │  │  - Project experiments (RAG)                │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Vision Module                               │   │   │
│  │  │  - Render analysis                          │   │   │
│  │  │  - Reference comparison                     │   │   │
│  │  │  - Issue localization                       │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Advice Engine                               │   │   │
│  │  │  - Symptom → cause mapping                  │   │   │
│  │  │  - Parameter recommendation                 │   │   │
│  │  │  - Code generation                          │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 Example Interaction Flow

```
1. Asset Evaluator reports:
   - warm_ratio: 68.99 (target: 19.16)
   - color_temperature: 3130K (target: 5778K)
   - structure_score: 0.48 (acceptable)

2. Blender Librarian receives:
   - Evaluation metrics
   - Current render image
   - Reference image (NASA SDO frame)
   - Current script parameters

3. Blender Librarian analyzes:
   [Vision] Render shows orange-dominated emission, reference shows white-yellow
   [Knowledge] Orange appearance indicates blackbody at ~3000K effective temp
   [Knowledge] Real sun (5778K) appears white-yellow due to peak emission
   [Knowledge] Principled Volume's blackbody uses temperature attribute × intensity

4. Blender Librarian responds:

   DIAGNOSIS: Blackbody emission is dominating with orange color because
   temperature values in VDB are being interpreted at face value. The
   simulation's temperature field likely contains values in the 1000-3000
   range, which produces orange blackbody emission.

   ROOT CAUSE: Mismatch between simulation temperature scale and
   physical Kelvin scale for blackbody radiation.

   RECOMMENDED FIX (Option A - No Rebake):
   Disable blackbody entirely and use emission color directly:
   ```python
   volume.inputs['Blackbody Intensity'].default_value = 0.0
   volume.inputs['Emission Strength'].default_value = 3.0
   volume.inputs['Emission Color'].default_value = (1.0, 0.95, 0.88, 1.0)
   ```

   RECOMMENDED FIX (Option B - Rebake Required):
   Modify simulation to output higher temperature values:
   - flame_max_temp: 5778.0 (physical sun temperature)
   - Or add temperature multiplier in shader using Math node

   RECOMMENDED FIX (Option C - Shader Modification):
   Add Math node to multiply temperature attribute before blackbody:
   ```python
   temp_mult = nodes.new('ShaderNodeMath')
   temp_mult.operation = 'MULTIPLY'
   temp_mult.inputs[1].default_value = 2.0  # Scale factor
   # Connect temperature attribute through multiplier
   ```

   CONFIDENCE: 85% (based on similar issues in Blender Stack Exchange)
   REFERENCES:
   - Manual: Rendering > Shader Nodes > Volume > Principled Volume
   - API: bpy.types.ShaderNodeVolumePrincipled
```

### 4.6 Training Approach Options

#### Option A: Fine-tuned Foundation Model
- Base: GPT-4V, Claude 3.5, or Gemini Pro Vision
- Fine-tune on Blender-specific corpus
- Pros: Powerful reasoning, vision built-in
- Cons: Expensive, may drift from base capabilities

#### Option B: RAG + Vision Pipeline
- Embed entire manual into vector database
- Use vision model for image analysis
- Query relevant docs based on symptoms
- Pros: Always up-to-date, cheaper
- Cons: May miss cross-references, slower

#### Option C: Hybrid Agent
- Pre-loaded knowledge graph of Blender concepts
- RAG for detailed documentation lookup
- Vision module for render analysis
- Reasoning module for fix generation
- Pros: Best of both worlds
- Cons: Complex to build and maintain

### 4.7 Implementation Phases

**Phase 1: Knowledge Base Construction**
- Scrape and embed Blender 5.0.1 manual
- Index bpy API with code examples
- Build concept graph (parameter → effect mappings)
- Collect Stack Exchange Q&A pairs

**Phase 2: Vision Module**
- Train/configure for Blender render analysis
- Implement reference comparison
- Build symptom detection (blowout, banding, noise, etc.)

**Phase 3: Advice Engine**
- Map symptoms to root causes
- Generate parameter recommendations
- Produce code snippets with confidence scores

**Phase 4: Integration**
- MCP server interface
- Connect to iteration controller
- Feedback loop for learning from results

---

## Part 5: Immediate Action Items

### 5.1 Short-term (Current Sprint)
1. Document all parameter combinations tested with results
2. Manually research Principled Volume shader behavior
3. Test shader node approach for temperature scaling

### 5.2 Medium-term (Next Sprint)
1. Prototype Blender Librarian knowledge base
2. Test vision analysis on render comparison
3. Build symptom → fix mapping database

### 5.3 Long-term (V4 Plan)
1. Full Blender Librarian agent implementation
2. Integration with asset generation pipeline
3. Continuous learning from experiment results

---

## Part 6: Questions for Gemini Feedback Integration

1. **Architecture:** Is RAG+Vision or fine-tuning better for domain-specific agents?
2. **Training Data:** What's the best way to structure Blender manual for embedding?
3. **Vision Analysis:** How to train/prompt for render issue detection?
4. **Confidence Calibration:** How to provide reliable confidence scores on advice?
5. **Feedback Loop:** Best practices for learning from experiment outcomes?

---

## Appendix A: Test Results Log

| Test | Parameters | warm_ratio | color_temp | Notes |
|------|------------|------------|------------|-------|
| Scattering Only | BB=0, Emit=0 | N/A | N/A | Nearly black |
| Tiny BB 0.0001 | BB=0.0001, T=500 | 96.17 | ~2900K | Too orange |
| Tiny BB 0.001 | BB=0.001, T=500 | 329.52 | ~2800K | Very orange |
| Yellow Shift | Color=(1,0.7,0.4), BB=0.01 | 70.02 | 3094K | Better |
| White-hot | Color=(1,0.95,0.85), BB=0.01 | 68.99 | 3130K | BB dominates |
| White Emission | Emit=0.5, BB=0 | 1.00 | 5778K | Too white |

**Target:** warm_ratio ≈ 19.16, color_temp ≈ 5778K

---

## Appendix B: Current Agent Capabilities Matrix

| Agent | Can Diagnose | Can Fix | Has Blender Knowledge | Has Vision |
|-------|--------------|---------|----------------------|------------|
| asset-evaluator | ✅ Metrics | ❌ | ❌ | ❌ |
| script-generator | ❌ | ✅ Templates | ⚠️ Limited | ❌ |
| blender-manual MCP | ❌ | ❌ | ⚠️ Search only | ❌ |
| experiment-tracker | ⚠️ Patterns | ⚠️ Suggestions | ❌ | ❌ |
| **Blender Librarian** | ✅ | ✅ | ✅ Pre-trained | ✅ |

---

## Appendix C: Blender Manual Sections to Prioritize

1. **Rendering > Cycles > Volume Rendering** - Core concepts
2. **Shader Nodes > Volume > Principled Volume** - Parameter reference
3. **Physics > Fluid > Mantaflow** - Simulation understanding
4. **Compositing > Color Management** - Display pipeline
5. **Python API > bpy.types.ShaderNode*** - Code generation
6. **Python API > bpy.types.FluidDomainSettings** - Simulation control

---

**Document End**

*This document should be compiled with Gemini feedback to create Multi-Agent Improvement Plan V4, which will include detailed specifications for the GPT-5.2 Blender Librarian agent.*
