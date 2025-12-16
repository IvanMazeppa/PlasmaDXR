# Findings and Patterns: Wolf-Rayet Bubble Session

**Date:** 2025-12-15
**Primary Agent:** Claude Code (Opus 4.5)
**Session Type:** Multi-agent celestial body creation

---

## Key Findings

### 1. MCP Blender Manual Server Capabilities

**Source:** Documentation review by Claude Code (Opus 4.5)

The blender-manual MCP server provides 12 specialized tools:

| Tool | Best Use Case | Priority |
|------|---------------|----------|
| `search_semantic` | Natural language queries ("how to create smoke") | High |
| `search_vdb_workflow` | VDB export, Mantaflow, caching | High |
| `search_python_api` | bpy.ops, bpy.types lookups | High |
| `search_bpy_types` | Type definition verification | Critical |
| `search_bpy_operators` | Finding operators by category | Medium |
| `search_nodes` | Shader/geometry node lookup | Medium |
| `read_page` | Full documentation pages | Medium |
| `browse_hierarchy` | Manual navigation | Low |

**Critical Finding:** `search_semantic` requires embeddings setup but enables conceptual queries that keyword search misses.

### 2. Blender 5.0 API Changes (GPT-5.2 Research)

**Source:** `MCP_SEARCH_TOOL_FINDINGS.md` (GPT-5.2)

| Change | Impact | Mitigation |
|--------|--------|------------|
| BLOSC compression removed | `openvdb_cache_compress_type = 'BLOSC'` fails | Use `_safe_enum_set()` with ZIP fallback |
| `read_page` truncation | Long API pages cut off at ~8000 chars | Use property-specific searches |
| FluidDomainSettings indexing | Some properties not searchable | Query type first, then read full page |

**Pattern Established:**
```python
def _safe_enum_set(obj, prop_name: str, desired: str) -> bool:
    try:
        prop = obj.bl_rna.properties[prop_name]
        valid = {it.identifier for it in prop.enum_items}
        if desired in valid:
            setattr(obj, prop_name, desired)
            return True
        return False
    except Exception:
        return False
```

### 3. TDR-Safe Workflow Patterns (GPT-5.2 Research)

**Source:** `TDR_SAFE_WORKFLOW.md` (GPT-5.2)

Windows GPU TDR (Timeout Detection and Recovery) crashes are common with volumetric rendering. Mitigation:

| Setting | Safe Value | Why |
|---------|------------|-----|
| Render Device | CPU | GPU volume rendering triggers TDR |
| Cycles Samples | 16 | Lower samples = faster GPU kernels |
| Volume Step Rate | 2.0-4.0 | Fewer ray march steps |
| Volume Max Steps | 128 | Hard cap on ray marching |
| Volume Bounces | 0-2 | Minimize path complexity |

**Script Pattern:**
```python
def apply_tdr_safe_render_preset(*, enable: bool) -> None:
    if not enable:
        return
    # Force EEVEE or CPU Cycles
    # Clamp volume settings
```

### 4. Multi-Emitter Patterns

**Source:** Analysis of existing scripts by Explore Agent (a9a124b)

Effective celestial body simulations use multiple emitters:

| Recipe | Emitters | Structure |
|--------|----------|-----------|
| Supergiant Star | 2 | Core + hotspot |
| Bipolar Nebula | 3 | Ring + 2 jets |
| Wolf-Rayet Bubble | 3 | Inner wind + outer shell + breakout |

**Key Insight:** Single emitters produce uniform results. Multiple emitters with different properties create more organic structures.

### 5. Space-Like Physics Configuration

**Source:** All existing scripts (pattern extraction)

Celestial simulations must disable Earth-like physics:

```python
settings.gravity = (0.0, 0.0, 0.0)  # No gravity
settings.alpha = 0.0                 # No density buoyancy
settings.beta = 0.0                  # No heat buoyancy
```

**Why:** Default Mantaflow settings create "rising smoke" which looks wrong for space objects.

### 6. Three Wind Model (Astrophysics)

**Source:** Web research by Claude Code (Opus 4.5)

Wolf-Rayet nebulae form through three distinct mass-loss phases:

```
Phase 1: Main Sequence → Slow, sparse wind (outermost)
Phase 2: Red Supergiant → Dense, slow wind (middle shell)
Phase 3: Wolf-Rayet → Fast, hot wind (inner cavity)
```

**Implementation:**
- Inner emitter: High velocity (1.8), low density (0.6), high temp (5.0)
- Outer emitter: Low velocity (0.4), high density (1.8), low temp (1.5)
- Breakout emitter: Very high velocity (2.0), tilted geometry

### 7. Agent Timeout Patterns

**Source:** Session observation by Claude Code (Opus 4.5)

Two general-purpose agents timed out during research:

| Agent | Task | Timeout | Cause |
|-------|------|---------|-------|
| a3788d4 | Volumetric techniques | >60s | Broad research scope |
| a444d1c | Celestial body comparison | >60s | Multiple web searches |

**Lesson:** Research tasks with broad scope should either:
- Have longer timeouts (90-120s)
- Be split into focused sub-tasks
- Use `run_in_background=true` with periodic checks

---

## Patterns Discovered

### Pattern 1: Documentation-First Development

```
1. Read ALL relevant docs first
2. Extract patterns from existing scripts
3. Design new script using patterns
4. Create documentation alongside code
```

**Benefit:** Ensures consistency with existing codebase.

### Pattern 2: MCP-First API Verification

```
Before writing bpy code:
1. search_bpy_types("ClassName")
2. Verify property exists and type
3. Check enum values if applicable
4. Document verification in comments
```

**Benefit:** Catches Blender version differences before runtime.

### Pattern 3: Safe Fallback Chains

```python
# Try best option first, fall back gracefully
if not _safe_enum_set(settings, "compress", "BLOSC"):
    if not _safe_enum_set(settings, "compress", "ZIP"):
        _safe_enum_set(settings, "compress", "NONE")
```

**Benefit:** Scripts work across Blender versions.

### Pattern 4: Parallel Agent Deployment

```
Launch independent research tasks simultaneously:
- Agent 1: Codebase exploration
- Agent 2: Technical research
- Agent 3: Alternative comparison
```

**Benefit:** Faster overall research phase.

### Pattern 5: Emitter Geometry Variety

| Celestial Structure | Emitter Geometry | Why |
|--------------------|------------------|-----|
| Spherical atmosphere | UV Sphere | Uniform radial emission |
| Bipolar jets | Cones | Directed outflow |
| Equatorial disk | Torus | Ring-like density |
| Break-out | Tilted cone | Asymmetric penetration |

---

## Recommendations

### For Future Sessions

1. **Always verify API via MCP before coding**
   - Don't trust training data for Blender 5.0
   - Document MCP query results

2. **Use focused agent tasks**
   - "Find FluidDomainSettings properties" > "Research volumetric techniques"
   - Shorter, more specific prompts complete faster

3. **Test incrementally**
   - Create minimal script first
   - Add features one at a time
   - Validate each addition

4. **Document findings immediately**
   - Create session reports during work
   - Capture patterns as they emerge

### For Script Development

1. **Always include `_safe_enum_set()`**
2. **Always include `_safe_rna_set()`**
3. **Always set space-like physics (gravity=0)**
4. **Always save .blend before baking**
5. **Always provide TDR-safe defaults**

### For Documentation

1. **Include astrophysical background**
2. **Provide parameter tables with defaults**
3. **Show comparison to related recipes**
4. **Document troubleshooting steps**
5. **Credit contributing agents**

---

## Appendix: Astronomical Color References

| Emission Line | Wavelength | Color | Celestial Object |
|--------------|------------|-------|------------------|
| H-alpha | 656.3 nm | Red/Pink | Emission nebulae |
| H-beta | 486.1 nm | Cyan | Hot gas |
| OIII | 500.7 nm | Blue-green | WR nebulae, planetary nebulae |
| NII | 658.4 nm | Red | Shock regions |
| SII | 671.6 nm | Deep red | Supernova remnants |

**RGB Approximations for Blender:**
- H-alpha: (0.9, 0.3, 0.4)
- OIII: (0.3, 0.75, 0.85)
- Mixed emission: (0.6, 0.5, 0.7)

---

*Documented by: Claude Code (Opus 4.5)*
*Contributing: Explore Agent (a9a124b), GPT-5.2 (prior research)*
