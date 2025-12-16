# Blender 5 to PlasmaDX VDB Workflow Plan

**Status:** Planning Phase
**Created:** 2025-12-07
**Purpose:** Bridge creative volumetric content creation in Blender with real-time rendering in PlasmaDX

---

## 1. Vision

Create a **bi-directional workflow** between Blender 5 and PlasmaDX-Clean that allows:
- Art-directed volumetric content (nebulae, gas clouds, explosions) created in Blender
- Exported as OpenVDB → loaded as NanoVDB in PlasmaDX at runtime
- Optional: Python automation to streamline repetitive export tasks

### Why This Matters
- **Procedural generation** (current approach) is great for infinite variation but hard to art-direct
- **Pre-authored VDB volumes** allow precise artistic control for hero assets
- **Hybrid approach** is ideal: procedural for background, VDB for focal points

---

## 2. The Complexity Concern

> "This idea is a bit nebulous and I worry this might be too overly complex"

**Valid concern.** Here's how we address it:

### Guiding Principles
1. **Try before you build** - Manually do the workflow in Blender first
2. **Automate only pain points** - Don't script what's easy to click
3. **One agent per clear responsibility** - No Swiss Army knife agents
4. **Validate at each phase** - Don't stack assumptions

### Complexity Budget
We have limited complexity tokens. Spend them wisely:

| Investment | Complexity Cost | Value |
|------------|-----------------|-------|
| Learning Blender basics | Medium | Required |
| Manual VDB export workflow | Low | Validation |
| MCP docs server (already done) | 0 | High |
| Simple export automation script | Low | Medium |
| Full multi-agent orchestration | High | Only if proven needed |

**Strategy:** Start at the top, stop when value plateaus.

---

## 3. Phased Approach

### Phase 0: Hands-On Validation (Do This First!)
**Goal:** Understand what Blender actually requires before building anything

**Tasks:**
1. Install Blender 5.0 (if not installed)
2. Create a simple smoke/volume simulation manually
3. Export as OpenVDB manually
4. Load the VDB in a test viewer (or directly in PlasmaDX NanoVDB loader)
5. Document pain points, surprises, and "aha" moments

**Deliverable:** `BLENDER_HANDS_ON_NOTES.md` - raw notes from first session

**Why First:** Every assumption we make now could be wrong. 30 minutes in Blender will teach us more than 3 hours of planning.

---

### Phase 1: Document the Manual Workflow
**Goal:** Create a reproducible manual workflow before any automation

**Workflow to Document:**
```
[Blender 5]                    [File System]              [PlasmaDX]
    |                               |                         |
    v                               v                         v
Create Volume Object  -->  Export .vdb file  -->  NanoVDB loader reads
(Mantaflow/GeoNodes)       (OpenVDB format)       and renders in RT
```

**Key Questions to Answer:**
- What Blender volume creation method works best? (Mantaflow vs Geometry Nodes vs Volume Modifier)
- What export settings matter? (Compression, precision, frame range)
- What coordinate system transformations are needed?
- What scale/units alignment is required?
- What data channels do we need? (density, temperature, velocity?)

---

### Phase 2: Minimal Python Automation
**Goal:** Script only the repetitive/error-prone parts

**Candidate Automation Tasks:**
- Batch export multiple frames
- Standardize export settings
- Auto-name files with metadata
- Coordinate system conversion (if needed)

**NOT Worth Automating:**
- Volume creation (creative work)
- One-off exports
- Anything that takes <30 seconds manually

**Deliverable:** `blender_vdb_export.py` - Blender addon or script

---

### Phase 3: Agent Architecture ✅ DESIGNED

**Status:** Agent architecture designed and documented (2025-12-07)

#### Agent Ecosystem

```
                         ┌─────────────────────────────────┐
                         │      User Intent (Ben)          │
                         │  "Create a nebula for my scene" │
                         └───────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    v                    v                    v
         ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
         │ blender-manual  │  │ blender-scripting│  │celestial-body-  │
         │   (MCP docs)    │  │    (new agent)   │  │    curator      │
         │   ✅ EXISTS     │  │   ✅ CREATED     │  │  ✅ CREATED     │
         │   12 tools      │  │                  │  │                 │
         └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
                  │                    │                     │
                  v                    v                     v
           Blender API           Python Scripts        Recipe Library
           Documentation          for Blender         (curated docs)
                  │                    │                     │
                  └────────────────────┼─────────────────────┘
                                       │
                                       v
                         ┌─────────────────────────────────┐
                         │         VDB Export              │
                         │   (OpenVDB files on disk)       │
                         └───────────────┬─────────────────┘
                                         │
                         ┌───────────────┼───────────────┐
                         │               │               │
                         v               v               v
              ┌─────────────────┐ ┌────────────┐ ┌──────────────────┐
              │gaussian-analyzer│ │ materials- │ │dxr-volumetric-   │
              │  (existing)     │ │ council    │ │pyro-specialist   │
              │                 │ │ (existing) │ │    (existing)    │
              └─────────────────┘ └────────────┘ └──────────────────┘
                         │               │               │
                         v               v               v
                         └───────────────┼───────────────┘
                                         │
                         ┌───────────────────────────────┐
                         │   PlasmaDX-Clean Renderer     │
                         │  (Real-time volumetric RT)    │
                         └───────────────────────────────┘
```

#### Agent Responsibilities (Finalized)

| Agent | Status | Responsibility | Location |
|-------|--------|---------------|----------|
| `blender-manual` | ✅ Exists | Blender 5.0 documentation (12 search tools) | `agents/blender-manual/` |
| `blender-scripting` | ✅ Created | Write/debug bpy Python scripts, teach Blender patterns | `agents/blender-scripting/` |
| `celestial-body-curator` | ✅ Created | Author/maintain recipe library for celestial bodies | `agents/celestial-body-curator/` |
| `gaussian-analyzer` | ✅ Exists | Validate material properties, performance estimates | `agents/gaussian-analyzer/` |
| `materials-council` | ✅ Exists | Material system design, particle structure | `agents/materials-council/` |
| `dxr-volumetric-pyro-specialist` | ✅ Exists | Pyro/explosion effect design | `agents/dxr-volumetric-pyro-specialist/` |

#### Recipe Library Structure

```
docs/blender_recipes/
├── README.md                     # Library index ✅ Created
├── emission_nebulae/
│   ├── hydrogen_cloud.md         # Basic wispy cloud
│   ├── emission_pillar.md        # Pillars of Creation style
│   └── orion_style.md            # Complex star-forming region
├── explosions/
│   ├── supernova_remnant.md      # Expanding shell
│   ├── stellar_flare.md          # Arc plasma
│   └── coronal_ejection.md       # CME event
├── stellar_phenomena/
│   ├── protoplanetary_disk.md    # Young star disk
│   ├── accretion_corona.md       # Hot plasma corona
│   └── planetary_nebula.md       # Dying star shell
├── dark_structures/
│   ├── dark_nebula.md            # Absorption cloud
│   └── dust_lane.md              # Galaxy dust lane
└── scripts/
    ├── quick_smoke_setup.py      # Basic smoke domain
    ├── vdb_export_batch.py       # Batch export
    └── celestial_presets.py      # Material presets
```

#### How Agents Work Together

**Example: "Create a supernova for my accretion disk scene"**

1. **celestial-body-curator** → Provides supernova recipe from library
2. **blender-scripting** → Generates/debugs Python script from recipe
3. **blender-manual** → Provides API documentation when script fails
4. **dxr-volumetric-pyro-specialist** → Designs explosion dynamics
5. **gaussian-analyzer** → Validates material properties for PlasmaDX
6. **materials-council** → Maps Blender properties to particle materials

**Example: "Why isn't my VDB loading?"**

1. **blender-scripting** → Checks export script for issues
2. **blender-manual** → Looks up cache settings documentation
3. **celestial-body-curator** → Provides troubleshooting from recipes

---

## 4. What We Already Have

### Existing Infrastructure
- **blender-manual MCP server** - 9 tools for Blender docs (tested, working)
- **NanoVDB volumetric system** - Core implementation in PlasmaDX (working)
- **Claude Agent SDK plugins** - Framework for building new agents

### Existing Knowledge
- VDB format basics
- NanoVDB vs OpenVDB differences
- PlasmaDX rendering pipeline

---

## 5. Open Questions (To Answer in Phase 0)

### Blender Side
1. Which Blender volume workflow is most intuitive?
   - Mantaflow (fluid simulation) - for realistic smoke/fire
   - Geometry Nodes (procedural) - for abstract volumes
   - Volume objects (static) - for simple density fields

2. What export settings produce clean VDB files?
3. Does Blender 5 have any VDB improvements over 4?

### Integration Side
1. Does our NanoVDB loader handle Blender's VDB output correctly?
2. Any coordinate system mismatches? (Y-up vs Z-up)
3. What grid resolutions are practical for real-time?

### Workflow Side
1. What's the iteration loop? (Edit in Blender -> Export -> View in PlasmaDX)
2. Can we hot-reload VDB files without restarting PlasmaDX?
3. What metadata should travel with the VDB file?

---

## 6. Recommended Next Steps

### Immediate (Today/This Week)
1. [ ] **Hands-on Blender session** - Create smoke, export VDB, load in viewer
2. [ ] **Document findings** in `BLENDER_HANDS_ON_NOTES.md`
3. [ ] **Test VDB in PlasmaDX** - Does our NanoVDB loader work with Blender exports?

### Near-Term (After Validation)
4. [ ] **Write manual workflow guide** - Step-by-step for reproducibility
5. [ ] **Identify automation candidates** - What's tedious?
6. [ ] **Create minimal Python script** - If needed

### Future (Only If Proven Needed)
7. [ ] **Build blender-scripting agent** - For complex scripting tasks
8. [ ] **Build nanovdb-engineer agent** - For C++ integration work
9. [ ] **Integrate workflow orchestration** - If multi-step workflows are common

---

## 7. Success Criteria

### Phase 0 Success
- Can create a volume in Blender and see it in PlasmaDX
- Understand the manual workflow end-to-end
- Have a list of actual (not assumed) pain points

### Phase 1 Success
- Documented workflow that someone else could follow
- Know which Blender features we actually use

### Phase 2 Success
- Automation reduces repetitive work by >50%
- Scripts are reliable and don't need constant fixing

### Phase 3 Success
- Agents handle their responsibilities without hand-holding
- Workflow orchestration catches errors before they cascade

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Over-engineering before validation | Phase 0 forces hands-on first |
| Building agents nobody uses | Only build after proving manual workflow |
| Blender learning curve too steep | Focus on minimum viable skills |
| VDB format incompatibilities | Test early with real exports |
| Complexity spirals out of control | Strict phase gates, complexity budget |

---

## Appendix A: Resources

### Blender Documentation (via MCP)
- Use `mcp__blender-manual__search_vdb_workflow` for VDB-specific docs
- Use `mcp__blender-manual__browse_hierarchy("physics/fluid")` for fluid sim docs
- Use `mcp__blender-manual__search_nodes("Principled Volume")` for shader docs

### PlasmaDX NanoVDB
- Current implementation: `src/rendering/NanoVDBSystem.h/cpp` (if exists)
- Shader integration: `shaders/nanovdb/` directory

### External
- [OpenVDB Documentation](https://www.openvdb.org/documentation/)
- [NanoVDB GitHub](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb)
- [Blender Python API](https://docs.blender.org/api/current/)

---

**Document maintained by:** Claude Code sessions
**Last Updated:** 2025-12-07
