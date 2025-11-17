# Multi-Agent System Roadmap – PlasmaDX

**Created:** 2025-11-15
**Purpose:** Prioritize agent infrastructure fixes before tackling core Gaussian rendering improvements

---

## Current Status Summary

### ✅ Working Systems
- **log-analysis-rag** (Tier 4 Diagnostics)
  - All 6 tools operational after Path fix
  - Successfully ingests PIX/buffer dumps
  - RAG database with 374+ documents
  - LangGraph diagnostic workflow functional

- **dxr-image-quality-analyst** (Tier 4 Diagnostics)
  - LPIPS ML comparison working
  - Performance analysis operational
  - Screenshot capture (F2) functional

- **Supporting agents** (all operational)
  - gaussian-analyzer
  - material-system-engineer
  - dxr-shadow-engineer
  - dxr-volumetric-pyro-specialist

---

## 🚧 Critical Infrastructure Issues (Blocking)

### 1. pix-debug Agent – OUTDATED ⚠️
**Problem:** Still diagnosing for ReSTIR/RTXDI (shelved system), not probe-grid (active system)

**Impact:** Provides misleading diagnostics (e.g., "ReSTIR under-sampling" when not using ReSTIR)

**Evidence:** Collaboration test (2025-11-15)
- pix-debug diagnosed: "ReSTIR M values too low"
- Reality: Project uses probe-grid, not ReSTIR
- log-analysis-rag correctly diagnosed AABB sizing issue

**Fix Required:**
- [ ] Update pix-debug to detect current rendering mode
- [ ] Add probe-grid-specific diagnostics
- [ ] Remove/flag ReSTIR diagnostics as "system shelved"
- [ ] OR deprecate and replace with new `path-and-probe` specialist

**Files:** `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/`

---

### 2. Missing RT Specialists (Per AGENT_HIERARCHY_AND_ROLES.md)

**Planned Tier 3 Specialists (not yet created):**

#### A. rt-lighting-engineer
- **Scope:** RayQuery particle-to-particle lighting, emission models, intensity coupling
- **Files:** `particle_gaussian_raytrace.hlsl`, lighting CBs, emission coupling
- **Status:** ❌ Doesn't exist (old agents predate new SDK)

#### B. sampling-and-distribution (RTXDI/ReSTIR)
- **Scope:** Light sampling strategies (RTXDI M4/M5/M6), spatial reuse, reservoir diagnostics
- **Files:** `rtxdi_raygen.hlsl`, temporal passes, configs
- **Status:** ❌ Doesn't exist
- **Note:** ReSTIR currently shelved, but specialist needed for future revival attempt

#### C. volumetric-materials
- **Scope:** Material schema, per-material noise/rim/temporal curves, celestial presets
- **Files:** `MaterialTypeProperties`, ImGui, JSON presets, shader material blocks
- **Status:** ⚠️ Partially exists (gaussian-analyzer + material-system-engineer, but not unified)

#### D. path-and-probe ⭐ **PRIORITY**
- **Scope:** Probe-grid lighting (current active system), path/volume interpolation
- **Files:** Probe structs/buffers, sampling & trilinear, grid updates
- **Status:** ❌ Doesn't exist
- **Impact:** **No specialist for active lighting system!**

---

### 3. MCP Server Reconnection Issue
**Problem:** log-analysis-rag fails to reconnect during Claude Code runtime

**Workaround:** Restart Claude Code to reconnect

**Root Cause:** Global `_retriever` singleton holds ChromaDB/FAISS connections, not cleaned up on disconnect

**Fix Required:**
- [ ] Add cleanup function to reset global state
- [ ] OR document workaround in README

**Priority:** Low (workaround is acceptable)

---

## 🎯 Core Rendering Goals (Why We Built This System)

### Phase 1: Gaussian Volumetric Rendering Improvements

**Current Issues:**

#### 1. Lighting System Quality vs Performance Trade-off ⭐ PRIMARY CHALLENGE
- **Multi-light scattering quality recreation**
  - Multi-light (16 lights) produces beautiful volumetric scattering but expensive (72 FPS)
  - Goal: Recreate scattering quality with probe-grid or ReSTIR at 90+ FPS
  - Challenge: Probe-grid lacks subtle rim lighting and grazing-angle scattering
  - ReSTIR M5 has temporal instability issues (patchwork pattern)

- **Probe-grid quality improvements**
  - Rim lighting at grazing angles inferior to multi-light
  - Henyey-Greenstein phase function averaging vs per-particle variation
  - Far-distance particle coverage gaps (>1200 units)
  - Trilinear interpolation too coarse for particle edges

- **Volumetric depth/scattering enhancement**
  - Beer-Lambert law tuning for material variation
  - Temperature-based blackbody emission accuracy
  - Scattering subtlety at particle boundaries

#### 2. Material Diversity (Phase 8 – Long-term)
- **Heterogeneous particles:** Stars, gas clouds, dust, rocky, icy
- **Material-aware RT lighting**
- **LOD and culling systems**

#### 3. Performance Optimization
- **Target:** 280+ FPS @ 100K particles with PINN + DLSS
- **Current bottleneck:** RayQuery traversal (BLAS rebuild: 2.1ms/frame)
- **Planned:**
  - PINN ML physics integration (C++ side pending)
  - BLAS update mode (+25% FPS)
  - Particle LOD culling (+50% FPS)

---

## 📋 Execution Plan

### **PHASE 0: Agent Infrastructure** (1-2 sessions)
**Goal:** Fix agent system so it can actually help with rendering work

- [x] **0.1** Create `path-and-probe` specialist (probe-grid expert)
  - 6 diagnostic tools operational
  - MCP server connected
  - See: `agents/path-and-probe/README.md`
  - Current active lighting system needs a specialist
  - Deprecates/replaces pix-debug probe-grid diagnostics

- [ ] **0.2** Update pix-debug or mark as deprecated
  - Add detection for current rendering mode
  - Flag ReSTIR diagnostics as "shelved system"

- [ ] **0.3** Create `rt-lighting-engineer` specialist
  - Particle-to-particle lighting expert
  - Emission model tuning

- [ ] **0.4** Test full agent collaboration workflow
  - path-and-probe → diagnoses probe-grid issue
  - log-analysis-rag → confirms with RAG data
  - rt-lighting-engineer → implements fix
  - dxr-image-quality-analyst → validates fix

---

### **PHASE 1: Lighting Quality vs Performance** (Primary Goal)
**Goal:** Recreate multi-light scattering quality with probe-grid/ReSTIR at 90+ FPS

- [ ] **1.1** Baseline quality comparison
  - Agent: lighting-quality-comparison skill (NEW!)
  - LPIPS comparison: multi-light vs probe-grid vs ReSTIR
  - Identify specific scattering quality losses
  - Quantify performance gains available

- [ ] **1.2** Probe-grid rim lighting improvement
  - Agent: path-and-probe + rt-lighting-engineer
  - Fix grazing-angle scattering quality loss
  - Increase grid resolution OR implement hybrid approach
  - Add per-probe phase function coefficients

- [ ] **1.3** Hybrid lighting system exploration
  - Agent: rt-lighting-engineer + path-and-probe
  - Test: Probe-grid (ambient) + 4-8 multi-lights (rim/scattering)
  - Target: 85-95 FPS with 90%+ multi-light quality
  - Implementation: ProbeGrid.cpp + MultiLightSystem.cpp integration

- [ ] **1.4** Quality validation & iteration
  - Agent: dxr-image-quality-analyst
  - LPIPS before/after comparison
  - 7-dimension quality rubric (focus: rim lighting, scattering)
  - Performance profiling (FPS vs quality trade-off matrix)

---

### **PHASE 2: Material Diversity** (Future)
**Goal:** Heterogeneous celestial bodies (stars, gas, dust, rocky, icy)

- [ ] **2.1** Material type system design
  - Agent: volumetric-materials (gaussian-analyzer + material-system-engineer)
  - Particle struct extensions
  - Material property presets

- [ ] **2.2** Material-aware RT lighting
  - Agent: rt-lighting-engineer
  - Per-material scattering/absorption
  - Temperature-dependent opacity

---

### **PHASE 3: Performance & ML Integration** (Future)
**Goal:** Hit 280+ FPS target with PINN physics

- [ ] **3.1** PINN ML physics C++ integration
  - ONNX Runtime model loading
  - Hybrid mode (PINN + traditional)

- [ ] **3.2** BLAS optimization
  - Update mode instead of full rebuild
  - Instance culling

---

## 🚀 Immediate Next Steps

**START HERE:**

1. **✅ DONE: path-and-probe specialist created**
   - 6 diagnostic tools operational
   - MCP server connected

2. **✅ DONE: lighting-quality-comparison skill created**
   - Autonomous multi-agent workflow
   - LPIPS + quality rubric + performance analysis
   - Historical context from RAG

3. **Test lighting quality comparison** (30 min)
   - Say: "Compare probe-grid to multi-light quality"
   - Skill auto-orchestrates: path-and-probe + dxr-image-quality-analyst + log-analysis-rag
   - Get baseline LPIPS scores and specific quality loss areas

4. **Begin lighting system improvements** (PRIMARY GOAL)
   - Implement top recommendations from skill analysis
   - Test hybrid probe-grid + selective multi-light approach
   - Target: 90+ FPS with 90%+ multi-light scattering quality

---

## Success Metrics

**Agent Infrastructure:**
- ✅ All Tier 3 specialists operational
- ✅ Collaboration workflow functional
- ✅ Diagnostics match active systems (not shelved ones)

**Rendering Quality:**
- ✅ Multi-light scattering quality recreated at 90+ FPS (LPIPS <0.15 vs multi-light baseline)
- ✅ Rim lighting at grazing angles improved (quality rubric score >8/10)
- ✅ Scattering realism maintained (quality rubric score >8/10)

**Performance:**
- ✅ 280+ FPS @ 100K particles (PINN + DLSS)
- ✅ <2ms BLAS rebuild time

---

## Notes

- **ReSTIR/RTXDI Status:** Shelved (couldn't get working). May revisit with multi-agent workflow later.
- **Current Lighting:** Probe-grid (active system, no specialist yet!)
- **Agent SDK:** New specialists must follow AGENT_HIERARCHY_AND_ROLES.md pattern
- **Old Agents:** rt-lighting-engineer, rt-shadow-engineer predate new SDK - need recreation

---

**Last Updated:** 2025-11-15
**Next Review:** After Phase 0 completion
