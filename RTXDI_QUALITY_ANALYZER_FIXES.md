# RTXDI Quality Analyzer - Bugs & Repairs Needed

**Document Purpose:** Track known issues and required fixes for the rtxdi-quality-analyzer MCP server
**Current Status:** Operational but needs improvements
**Priority:** MEDIUM (working but could be better)

---

## Known Issues

### Issue 1: Dimension Mismatch Handling ⚠️ MEDIUM PRIORITY

**Problem:**
When comparing screenshots of different dimensions, the tool fails with a dimension mismatch error instead of providing a helpful message or auto-resize option.

**Current Behavior:**
```python
# Comparing 581×2246 vs 1440×2560 fails
ValueError: Image dimensions don't match
```

**Expected Behavior:**
- **Option A:** Auto-resize both images to common dimensions (smallest)
- **Option B:** Provide clear error with resize suggestion
- **Option C:** Offer both options via tool parameter

**Files Affected:**
- `agents/rtxdi-quality-analyzer/src/tools/ml_visual_comparison.py`

**Proposed Fix:**
```python
# In compare_screenshots_ml():
if before_img.shape != after_img.shape:
    # Option A: Auto-resize (default)
    target_size = (min(before_img.shape[0], after_img.shape[0]),
                   min(before_img.shape[1], after_img.shape[1]))

    before_img = cv2.resize(before_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    after_img = cv2.resize(after_img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Warn user
    print(f"⚠️  Images resized to {target_size} for comparison")
```

**Estimated Effort:** 30 minutes
**Risk:** LOW (improves usability, no breaking changes)

---

### Issue 2: Missing Metadata Support ⚠️ MEDIUM PRIORITY

**Problem:**
The tool lists screenshots and notes "Metadata: ❌ Not available" but doesn't provide guidance on:
1. How to enable metadata capture
2. What metadata fields should be included
3. How to use metadata in analysis

**Current State:**
- F2 screenshot capture creates `.bmp` file only
- No `.bmp.json` sidecar file generated
- Metadata reading code exists but unused

**Required Documentation:**
1. Document metadata JSON schema
2. Update F2 capture to write metadata
3. Add metadata-aware comparison features

**Proposed Metadata Schema:**
```json
{
  "timestamp": "2025-11-11T08:30:00Z",
  "resolution": [1440, 2560],
  "rendering": {
    "rtxdi_enabled": true,
    "rtxdi_mode": "M5",
    "shadow_rays_per_light": 1,
    "particle_count": 10000,
    "light_count": 13,
    "material_system_enabled": true,  // NEW for material system
    "material_types_count": 5         // NEW for material system
  },
  "performance": {
    "fps": 115.3,
    "frame_time_ms": 8.7,
    "gpu_memory_mb": 1024
  },
  "camera": {
    "position": [500, 250, 500],
    "rotation": [45, 0, 0],
    "fov": 60
  }
}
```

**Files to Modify:**
- `src/main.cpp` (F2 capture code) - Add JSON write
- `agents/rtxdi-quality-analyzer/src/tools/ml_visual_comparison.py` - Use metadata in reports

**Estimated Effort:** 2-3 hours
**Risk:** LOW (additive feature, no breaking changes)

---

### Issue 3: Lazy Loading Timeout Risk ⚠️ LOW PRIORITY

**Problem:**
PyTorch + LPIPS models (528 MB) are lazy-loaded on first tool call. If loading takes > 30 seconds, MCP connection times out.

**Current Status:**
- **WORKING** on current system (loads in ~8 seconds)
- **RISK:** Could fail on slower systems or network-mounted drives

**Proposed Fix:**
1. Add loading progress messages
2. Pre-warm models during server init (optional)
3. Increase timeout to 60 seconds for first call

**Files Affected:**
- `agents/rtxdi-quality-analyzer/rtxdi_server.py`
- `agents/rtxdi-quality-analyzer/src/tools/ml_visual_comparison.py`

**Proposed Enhancement:**
```python
# In rtxdi_server.py
@server.list_tools()
async def list_tools():
    # Pre-warm ML models during server initialization (optional)
    if os.getenv("PREWARM_ML_MODELS", "false").lower() == "true":
        print("Prewarming ML models...")
        import lpips
        _ = lpips.LPIPS(net='vgg')
        print("✅ ML models ready")

    return [...]
```

**Estimated Effort:** 1 hour
**Risk:** LOW (optional optimization)

---

### Issue 4: Heatmap File Naming Collision ⚠️ LOW PRIORITY

**Problem:**
Heatmap filenames use screenshot names which can collide:
- `diff_image copy 42_vs_image copy 42.png` (actual output from test)
- Spaces in filenames (not ideal for command-line tools)

**Proposed Fix:**
- Use timestamp-based naming: `diff_2025-11-11_08-30-00.png`
- Or hash-based: `diff_a1b2c3d4_vs_e5f6g7h8.png`
- Sanitize filenames (remove spaces)

**Files Affected:**
- `agents/rtxdi-quality-analyzer/src/tools/ml_visual_comparison.py`

**Proposed Enhancement:**
```python
# Generate clean heatmap filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
before_hash = hashlib.md5(str(before_path).encode()).hexdigest()[:8]
after_hash = hashlib.md5(str(after_path).encode()).hexdigest()[:8]

heatmap_filename = f"diff_{timestamp}_{before_hash}_vs_{after_hash}.png"
```

**Estimated Effort:** 30 minutes
**Risk:** LOW (cosmetic improvement)

---

### Issue 5: No Batch Comparison Support ⚠️ LOW PRIORITY

**Problem:**
To compare 5 material types pairwise, user must call `compare_screenshots_ml` 10 times manually (5 choose 2 = 10 comparisons).

**Proposed Feature:**
Add `batch_compare_screenshots` tool:
```python
@tool("batch_compare_screenshots",
      "Compare multiple screenshots pairwise and generate comparison matrix")
async def batch_compare_screenshots(
    screenshot_paths: list[str],
    save_heatmaps: bool = False
) -> str:
    # Compare all pairs
    # Return matrix of LPIPS scores
    # Optionally generate heatmaps
```

**Use Case (Material System Validation):**
```python
batch_compare_screenshots([
    "screenshots/plasma.bmp",
    "screenshots/star.bmp",
    "screenshots/gas.bmp",
    "screenshots/rocky.bmp",
    "screenshots/icy.bmp"
])

# Returns:
#         PLASMA  STAR  GAS   ROCKY  ICY
# PLASMA  0.000   0.45  0.38  0.52   0.41
# STAR    0.45    0.000 0.56  0.61   0.48
# GAS     0.38    0.56  0.000 0.44   0.39
# ROCKY   0.52    0.61  0.44  0.000  0.29
# ICY     0.41    0.48  0.39  0.29   0.000
```

**Estimated Effort:** 2-3 hours
**Risk:** LOW (new feature, no breaking changes)
**Priority:** LOW (nice-to-have for Sprint 1, not critical)

---

## Enhancement Requests

### Enhancement 1: Add Screenshot Annotation Tool ✨ LOW PRIORITY

**Feature:**
Add tool to annotate screenshots with text/arrows before comparison:
- Mark regions of interest
- Add labels ("PLASMA region", "STAR region")
- Draw attention to specific features

**Use Case:**
When creating comparison reports, annotate what changed between screenshots.

**Estimated Effort:** 4-6 hours
**Priority:** LOW (Sprint 3+ feature)

---

### Enhancement 2: Video Frame Comparison ✨ LOW PRIORITY

**Feature:**
Compare frames from video captures to detect:
- Temporal instability (flickering)
- Performance drops (frame stuttering)
- Animation quality

**Use Case:**
Validate that material system doesn't introduce flickering.

**Estimated Effort:** 8-10 hours (new tool)
**Priority:** LOW (post-Sprint 1)

---

### Enhancement 3: Statistical Analysis Tool ✨ MEDIUM PRIORITY

**Feature:**
Add tool to analyze screenshot datasets:
- Mean LPIPS across multiple comparisons
- Standard deviation (consistency)
- Outlier detection
- Regression detection (new vs old build)

**Use Case:**
After Sprint 1, compare 10 test scenarios (old build vs new build) and report:
- Average regression: LPIPS = 0.012 ✅ (< 0.02 threshold)
- Max regression: LPIPS = 0.018 ✅ (< 0.02 threshold)
- Outliers: None

**Estimated Effort:** 3-4 hours
**Priority:** MEDIUM (useful for Sprint 2+)

---

## Repair Priority & Roadmap

### Critical Path (Block Sprint 1)
- **None** - All critical functionality working

### High Priority (Sprint 1 Nice-to-Have)
- Issue 2: Metadata support (2-3 hours)
  - Enables richer comparison reports
  - Documents rendering configuration

### Medium Priority (Sprint 2)
- Issue 1: Dimension mismatch handling (30 min)
- Enhancement 3: Statistical analysis (3-4 hours)

### Low Priority (Sprint 3+)
- Issue 3: Lazy loading optimization (1 hour)
- Issue 4: Heatmap naming (30 min)
- Issue 5: Batch comparison (2-3 hours)
- Enhancement 1: Screenshot annotation (4-6 hours)
- Enhancement 2: Video frame comparison (8-10 hours)

---

## Testing Requirements

### Test Case 1: Dimension Mismatch (Issue 1)
```bash
# Create two screenshots of different sizes
# Expected: Auto-resize or helpful error message
# Actual: ValueError (needs fix)
```

### Test Case 2: Metadata Capture (Issue 2)
```bash
# Press F2 in PlasmaDX
# Expected: .bmp + .bmp.json files created
# Actual: .bmp only (needs implementation)
```

### Test Case 3: Lazy Loading (Issue 3)
```bash
# First call to compare_screenshots_ml after server start
# Expected: < 30 second load time
# Actual: ~8 seconds (working, but monitor)
```

---

## Maintenance Schedule

### Before Sprint 1
- [ ] Test all tools with Sprint 1 screenshots
- [ ] Verify dimension mismatch handling (Issue 1 - optional)
- [ ] Document metadata schema (Issue 2 - defer implementation)

### During Sprint 1
- [ ] Use tools at each checkpoint
- [ ] Log any new issues discovered
- [ ] Document workarounds if needed

### After Sprint 1
- [ ] Implement high-priority fixes (Issue 2)
- [ ] Add batch comparison for Sprint 2 validation (Issue 5)
- [ ] Create statistical analysis tool (Enhancement 3)

---

## Related Documentation

- **MCP Server Setup:** `agents/rtxdi-quality-analyzer/README.md`
- **Tool Usage Guide:** `agents/rtxdi-quality-analyzer/ML_COMPARISON_USAGE.md`
- **Integration Guide:** `agents/gaussian-analyzer/INTEGRATION_GUIDE.md`
- **Sprint 1 Tasks:** `SPRINT_1_MATERIAL_SYSTEM_IMPLEMENTATION.md`

---

**Last Updated:** 2025-11-11
**Maintainer:** PlasmaDX Development Team
**Status:** Living document - update as issues discovered
