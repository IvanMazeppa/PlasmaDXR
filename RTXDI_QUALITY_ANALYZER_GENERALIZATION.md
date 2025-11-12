# RTXDI Quality Analyzer → General Visual Quality Analyzer

**Document Purpose:** Roadmap for generalizing rtxdi-quality-analyzer into a reusable visual quality analysis tool for any rendering system
**Current Name:** `rtxdi-quality-analyzer` (RTXDI-specific)
**Proposed Name:** `visual-quality-analyzer` or `rendering-quality-analyzer`
**Priority:** MEDIUM (post-Sprint 1)

---

## Motivation

The current `rtxdi-quality-analyzer` MCP server provides excellent ML-powered visual validation tools, but:
- Name implies RTXDI-only usage
- Some tools reference RTXDI-specific concepts
- Could be useful for:
  - Material system validation (current need)
  - Shadow quality analysis
  - DLSS quality assessment
  - God rays validation
  - Any before/after rendering comparison

**Goal:** Make the tool **rendering-agnostic** while maintaining RTXDI-specific features as optional presets.

---

## Current Architecture Analysis

### What's Already General-Purpose ✅

**These tools work for any rendering system:**
1. ✅ `compare_screenshots_ml` - Pure image comparison (no RTXDI dependency)
2. ✅ `list_recent_screenshots` - File system utility (no RTXDI dependency)
3. ⚠️ `assess_visual_quality` - 7-dimension rubric (volumetric-focused, could be generalized)

**These tools are RTXDI-specific:**
4. ❌ `compare_performance` - Parses RTXDI-specific log formats
5. ❌ `analyze_pix_capture` - Looks for RTXDI buffers

### What Needs Generalization

| **Tool** | **Current State** | **Generalization Needed** |
|----------|-------------------|---------------------------|
| `compare_screenshots_ml` | ✅ General-purpose | None |
| `list_recent_screenshots` | ✅ General-purpose | None |
| `assess_visual_quality` | ⚠️ Volumetric-focused | Add rendering mode parameter |
| `compare_performance` | ❌ RTXDI log parser | Add pluggable log parsers |
| `analyze_pix_capture` | ❌ RTXDI buffer names | Add buffer search patterns |

---

## Generalization Strategy

### Phase 1: Rename & Rebrand (2-3 hours)

**Rename MCP Server:**
```bash
# Old structure
agents/rtxdi-quality-analyzer/
├── rtxdi_server.py
├── src/tools/
│   ├── ml_visual_comparison.py     ✅ General
│   ├── performance_comparison.py   ❌ RTXDI-specific
│   ├── pix_analysis.py             ❌ RTXDI-specific
│   └── visual_quality_assessment.py ⚠️ Volumetric-focused

# New structure
agents/visual-quality-analyzer/
├── visual_quality_server.py
├── src/tools/
│   ├── ml_visual_comparison.py     ✅ General (no changes)
│   ├── performance_comparison.py   ⚠️ Refactor (pluggable parsers)
│   ├── pix_analysis.py             ⚠️ Refactor (search patterns)
│   ├── visual_quality_assessment.py ⚠️ Refactor (rendering modes)
│   └── presets/
│       ├── rtxdi_preset.py         (RTXDI-specific logic)
│       ├── material_preset.py      (Material system logic)
│       └── shadow_preset.py        (Shadow system logic)
```

**Update MCP Server Configuration:**
```json
// .claude.json
{
  "mcpServers": {
    "visual-quality-analyzer": {  // Renamed from rtxdi-quality-analyzer
      "command": "bash",
      "args": [".../visual-quality-analyzer/run_server.sh"]
    }
  }
}
```

**Backward Compatibility:**
- Keep `rtxdi-quality-analyzer` as symlink to new server
- Tools remain accessible via both namespaces:
  - `mcp__visual-quality-analyzer__compare_screenshots_ml`
  - `mcp__rtxdi-quality-analyzer__compare_screenshots_ml` (deprecated alias)

---

### Phase 2: Generalize Visual Quality Assessment (3-4 hours)

**Current 7-Dimension Rubric (Volumetric-Focused):**
1. Volumetric depth
2. Rim lighting
3. Temperature gradient
4. RTXDI stability
5. Shadows
6. Scattering
7. Temporal stability

**Problem:** Assumes volumetric rendering with RTXDI

**Proposed: Rendering Mode Parameter**
```python
@tool("assess_visual_quality",
      "Assess visual quality using rendering-mode-specific rubrics")
async def assess_visual_quality(
    screenshot_path: str,
    rendering_mode: str = "volumetric",  # NEW PARAMETER
    comparison_before: str = None
) -> str:
    # Select rubric based on rendering mode
    if rendering_mode == "volumetric":
        rubric = VolumetricQualityRubric()
    elif rendering_mode == "surface":
        rubric = SurfaceQualityRubric()
    elif rendering_mode == "hybrid":
        rubric = HybridQualityRubric()
    elif rendering_mode == "material_system":
        rubric = MaterialSystemRubric()
    else:
        rubric = GeneralQualityRubric()

    return rubric.assess(screenshot_path, comparison_before)
```

**Material System Rubric (New):**
1. Material distinctiveness (are types visually different?)
2. Albedo accuracy (correct colors?)
3. Emission consistency (physically plausible?)
4. Scattering quality (phase function working?)
5. Shadow quality (material-aware shadows?)
6. Opacity accuracy (transparent vs opaque?)
7. Temporal stability (no flickering?)

**Surface Rendering Rubric (New):**
1. Surface detail (normal maps visible?)
2. Reflection quality (PBR accuracy?)
3. Specular highlights (correct Fresnel?)
4. Shadow quality (contact shadows, AO?)
5. Lighting accuracy (physically correct?)
6. Material variation (roughness/metallic?)
7. Temporal stability

**General Rubric (Fallback):**
1. Image clarity
2. Color accuracy
3. Contrast
4. Brightness
5. Sharpness
6. Noise level
7. Artifacts

---

### Phase 3: Pluggable Performance Parsers (4-6 hours)

**Current Problem:**
`compare_performance` tool hardcodes RTXDI log parsing:
```python
# Looks for:
# "RTXDI M4 FPS: 120.5"
# "RTXDI M5 FPS: 115.3"
```

**Proposed: Parser Registry**
```python
# performance_comparison.py

class PerformanceLogParser(ABC):
    @abstractmethod
    def parse(self, log_path: str) -> dict:
        """Parse log file and return metrics dict"""
        pass

# RTXDI parser (existing logic)
class RTXDILogParser(PerformanceLogParser):
    def parse(self, log_path: str) -> dict:
        # Existing RTXDI parsing logic
        return {
            "rtxdi_mode": "M5",
            "fps": 115.3,
            "shadow_rays": 1
        }

# Material system parser (new)
class MaterialSystemLogParser(PerformanceLogParser):
    def parse(self, log_path: str) -> dict:
        # Parse material system logs
        return {
            "particle_struct_size": 48,
            "material_types_active": 5,
            "fps": 115.0
        }

# General parser (fallback)
class GeneralLogParser(PerformanceLogParser):
    def parse(self, log_path: str) -> dict:
        # Parse generic FPS logs
        return {
            "fps": self._extract_fps(log_path)
        }

# Parser registry
PARSERS = {
    "rtxdi": RTXDILogParser(),
    "material_system": MaterialSystemLogParser(),
    "shadow_system": ShadowSystemLogParser(),
    "general": GeneralLogParser()
}

# Tool with parser parameter
@tool("compare_performance",
      "Compare performance metrics between configurations")
async def compare_performance(
    before_log: str,
    after_log: str,
    parser_type: str = "auto"  # NEW: auto-detect or specify
) -> str:
    # Auto-detect parser if not specified
    if parser_type == "auto":
        parser_type = detect_parser_type(before_log)

    parser = PARSERS.get(parser_type, PARSERS["general"])

    before_metrics = parser.parse(before_log)
    after_metrics = parser.parse(after_log)

    return format_comparison(before_metrics, after_metrics)
```

**Usage Examples:**
```python
# RTXDI comparison (existing)
compare_performance(
    "logs/rtxdi_m4.log",
    "logs/rtxdi_m5.log",
    parser_type="rtxdi"  # or "auto"
)

# Material system comparison (new)
compare_performance(
    "logs/legacy_32byte.log",
    "logs/material_48byte.log",
    parser_type="material_system"
)

# Shadow system comparison (new)
compare_performance(
    "logs/pcss_4rays.log",
    "logs/rtxdi_shadows.log",
    parser_type="shadow_system"
)
```

---

### Phase 4: Flexible PIX Analysis (4-6 hours)

**Current Problem:**
`analyze_pix_capture` hardcodes RTXDI buffer names:
```python
# Searches for:
# "g_currentReservoirs"
# "g_prevReservoirs"
# "g_rtLighting"
```

**Proposed: Search Pattern Configuration**
```python
@tool("analyze_pix_capture",
      "Analyze PIX GPU capture for performance and correctness")
async def analyze_pix_capture(
    capture_path: str = None,  # Auto-detect latest if None
    search_patterns: dict = None  # NEW: Customizable search
) -> str:
    # Default patterns (RTXDI)
    if search_patterns is None:
        search_patterns = RTXDI_PATTERNS

    # User can provide custom patterns
    # search_patterns = {
    #     "buffers": ["g_particles", "g_materialProperties"],
    #     "shaders": ["particle_gaussian_raytrace"],
    #     "metrics": ["DrawCall", "Dispatch", "BLAS Rebuild"]
    # }

    # Analyze capture with custom patterns
    return analyze_with_patterns(capture_path, search_patterns)

# Preset patterns for common use cases
RTXDI_PATTERNS = {
    "buffers": ["g_currentReservoirs", "g_prevReservoirs", "g_rtLighting"],
    "shaders": ["rtxdi_raygen", "rtxdi_temporal_accumulate"],
    "metrics": ["ReSTIR Sample", "Temporal Reuse"]
}

MATERIAL_SYSTEM_PATTERNS = {
    "buffers": ["g_particles", "g_materialProperties"],
    "shaders": ["particle_gaussian_raytrace", "ComputeMaterialEmission"],
    "metrics": ["Material Switch", "BLAS Rebuild", "Phase Function"]
}

SHADOW_SYSTEM_PATTERNS = {
    "buffers": ["g_shadowBuffer", "g_prevShadow"],
    "shaders": ["pcss_shadows", "shadow_rays"],
    "metrics": ["Shadow Ray", "Poisson Sampling", "Temporal Filter"]
}
```

**Usage Examples:**
```python
# RTXDI analysis (existing)
analyze_pix_capture(
    capture_path="PIX/Captures/rtxdi_test.wpix"
)  # Uses RTXDI_PATTERNS by default

# Material system analysis (new)
analyze_pix_capture(
    capture_path="PIX/Captures/material_test.wpix",
    search_patterns=MATERIAL_SYSTEM_PATTERNS
)

# Custom analysis (new)
analyze_pix_capture(
    capture_path="PIX/Captures/custom.wpix",
    search_patterns={
        "buffers": ["g_myBuffer"],
        "shaders": ["my_shader"],
        "metrics": ["My Metric"]
    }
)
```

---

### Phase 5: Preset System (2-3 hours)

**Create preset modules for common use cases:**

```python
# src/tools/presets/rtxdi_preset.py
from ..performance_comparison import PerformanceLogParser
from ..pix_analysis import PIXSearchPatterns

class RTXDIPreset:
    @staticmethod
    def get_parser():
        return RTXDILogParser()

    @staticmethod
    def get_search_patterns():
        return RTXDI_PATTERNS

    @staticmethod
    def get_quality_rubric():
        return RTXDIQualityRubric()

# src/tools/presets/material_preset.py
class MaterialSystemPreset:
    @staticmethod
    def get_parser():
        return MaterialSystemLogParser()

    @staticmethod
    def get_search_patterns():
        return MATERIAL_SYSTEM_PATTERNS

    @staticmethod
    def get_quality_rubric():
        return MaterialSystemRubric()

# src/tools/presets/__init__.py
PRESETS = {
    "rtxdi": RTXDIPreset,
    "material_system": MaterialSystemPreset,
    "shadow_system": ShadowSystemPreset,
    "dlss": DLSSPreset,
    "god_rays": GodRaysPreset
}

# Usage
preset = PRESETS["material_system"]
parser = preset.get_parser()
patterns = preset.get_search_patterns()
rubric = preset.get_quality_rubric()
```

---

## Migration Plan

### Step 1: Duplicate & Test (Low Risk)
1. Copy `rtxdi-quality-analyzer` → `visual-quality-analyzer`
2. Rename server file: `rtxdi_server.py` → `visual_quality_server.py`
3. Update imports, keep functionality identical
4. Test both servers side-by-side

**Risk:** LOW (no changes to existing functionality)
**Time:** 1-2 hours

---

### Step 2: Add Generalization (Medium Risk)
1. Add rendering mode parameter to `assess_visual_quality`
2. Add parser type parameter to `compare_performance`
3. Add search patterns parameter to `analyze_pix_capture`
4. Keep defaults that match current behavior

**Risk:** MEDIUM (backward compatible but new parameters)
**Time:** 8-12 hours

---

### Step 3: Create Presets (Low Risk)
1. Extract RTXDI logic into `rtxdi_preset.py`
2. Create `material_preset.py` for material system
3. Create `shadow_preset.py` for shadow system
4. Register presets in tool

**Risk:** LOW (additive feature)
**Time:** 4-6 hours

---

### Step 4: Update Documentation (Low Risk)
1. Rename documentation files
2. Add preset usage examples
3. Document custom patterns
4. Update integration guides

**Risk:** LOW (documentation only)
**Time:** 2-3 hours

---

### Step 5: Deprecate Old Name (Low Risk)
1. Create symlink: `rtxdi-quality-analyzer` → `visual-quality-analyzer`
2. Add deprecation warning in old server
3. Update all references in CLAUDE.md
4. Keep both namespaces working for 2+ months

**Risk:** LOW (gradual transition)
**Time:** 1 hour

---

## Implementation Timeline

### Immediate (Sprint 1)
- **Nothing** - Use existing rtxdi-quality-analyzer as-is
- Works perfectly for material system validation

### Short-Term (Sprint 2)
- Step 1: Duplicate & test (1-2 hours)
- Step 2: Add rendering mode parameter (2-3 hours)
- Step 3: Create material system preset (1-2 hours)

**Total:** 4-7 hours

### Medium-Term (Sprint 3)
- Step 2: Add parser registry (4-6 hours)
- Step 3: Create shadow/DLSS presets (3-4 hours)
- Step 4: Update documentation (2-3 hours)

**Total:** 9-13 hours

### Long-Term (Post-Sprint 3)
- Step 2: Add PIX search patterns (4-6 hours)
- Step 5: Deprecate old name (1 hour)
- Community presets (ongoing)

**Total:** 5-7 hours

**Grand Total:** 18-27 hours (spread across multiple sprints)

---

## Use Cases After Generalization

### Use Case 1: Material System Validation
```python
# Compare legacy vs material system
visual_quality_analyzer.compare_screenshots_ml(
    before="baseline_plasma.bmp",
    after="material_system.bmp"
)

visual_quality_analyzer.assess_visual_quality(
    screenshot="material_system.bmp",
    rendering_mode="material_system"
)

visual_quality_analyzer.compare_performance(
    before_log="legacy.log",
    after_log="material.log",
    parser_type="material_system"
)
```

### Use Case 2: Shadow Quality Analysis
```python
visual_quality_analyzer.assess_visual_quality(
    screenshot="pcss_shadows.bmp",
    rendering_mode="shadows"
)

visual_quality_analyzer.analyze_pix_capture(
    capture_path="shadow_test.wpix",
    search_patterns=SHADOW_SYSTEM_PATTERNS
)
```

### Use Case 3: DLSS Quality Validation
```python
visual_quality_analyzer.compare_screenshots_ml(
    before="native_1440p.bmp",
    after="dlss_quality_1440p.bmp"
)

visual_quality_analyzer.assess_visual_quality(
    screenshot="dlss_quality_1440p.bmp",
    rendering_mode="dlss"
)
```

### Use Case 4: Custom Rendering Feature
```python
visual_quality_analyzer.assess_visual_quality(
    screenshot="my_feature.bmp",
    rendering_mode="general"  # Uses general rubric
)

visual_quality_analyzer.analyze_pix_capture(
    capture_path="my_feature.wpix",
    search_patterns={
        "buffers": ["g_myBuffer"],
        "shaders": ["my_shader"]
    }
)
```

---

## Benefits of Generalization

### For Material System (Current Need)
- ✅ Reuse ML comparison tools (already working)
- ✅ Add material-specific quality rubric
- ✅ Add material-specific performance parser
- ✅ Validate 5 material types systematically

### For Future Features
- ✅ Shadow system validation (raytraced shadows)
- ✅ DLSS quality assessment
- ✅ God rays debugging
- ✅ Any before/after comparison

### For Community
- ✅ Reusable tool for other DirectX 12 projects
- ✅ Preset system allows custom extensions
- ✅ No vendor lock-in (works with any renderer)

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation:** Backward compatibility via default parameters and symlinks

### Risk 2: Increased Complexity
**Mitigation:** Preset system hides complexity, defaults work like current

### Risk 3: Maintenance Burden
**Mitigation:** Presets are optional, core tools remain simple

### Risk 4: Time Investment
**Mitigation:** Phased rollout, immediate benefit in Sprint 2

---

## Decision: Proceed or Defer?

### Arguments for Immediate Generalization (Sprint 2)
- Material system needs custom quality rubric anyway
- Refactoring now avoids tech debt
- Small time investment (4-7 hours Sprint 2)

### Arguments for Deferred Generalization (Post-Sprint 3)
- Current tools work fine for Sprint 1
- Focus on material system implementation
- Can generalize later when more use cases emerge

### Recommendation: **Hybrid Approach**
1. **Sprint 1:** Use existing tools as-is (0 hours)
2. **Sprint 2:** Add rendering mode parameter + material preset (4-7 hours)
3. **Sprint 3+:** Full generalization with parser registry (18-27 hours total)

**Rationale:** Get immediate value in Sprint 2 without full commitment. Validate approach before full refactor.

---

## Related Documentation

- **Current Server:** `agents/rtxdi-quality-analyzer/README.md`
- **Integration Guide:** `agents/gaussian-analyzer/INTEGRATION_GUIDE.md`
- **Fixes Needed:** `RTXDI_QUALITY_ANALYZER_FIXES.md`
- **Sprint 1 Plan:** `SPRINT_1_MATERIAL_SYSTEM_IMPLEMENTATION.md`

---

**Last Updated:** 2025-11-11
**Status:** PROPOSAL (not yet implemented)
**Next Action:** Discuss with team, decide on Sprint 2 scope
