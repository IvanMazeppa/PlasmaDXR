# DXR Image Quality Analyst
**Version:** 1.21.0
**Agent Type:** Visual Quality Assessment & ML-Powered Image Analysis
**Purpose:** Autonomous visual quality tracking for volumetric particle rendering development

---

## OVERVIEW

The **DXR Image Quality Analyst** is an MCP (Model Context Protocol) agent designed to provide **brutally honest, actionable visual quality feedback** during PlasmaDX development. It combines ML-powered perceptual similarity analysis (LPIPS) with traditional computer vision metrics to track visual improvements/regressions.

### Core Capabilities

✅ **ML-Powered Visual Comparison** - Perceptual similarity using LPIPS (92% correlation with human judgment)
✅ **Automated Visual Quality Assessment** - 7-dimension rubric for volumetric rendering
✅ **Performance Analysis** - FPS/frame time tracking with bottleneck identification
✅ **PIX Capture Analysis** - GPU profiling integration
✅ **Screenshot Management** - Organized capture and metadata tracking

### Key Features

- **Honest Feedback:** No sugar-coating - identifies critical issues directly
- **Actionable Recommendations:** Specific file:line references for fixes
- **Context-Aware:** Understands PlasmaDX goals (volumetric plasma, not geometric primitives)
- **ML Vision:** Can "see" visual differences humans would notice
- **Metadata Integration:** Reads screenshot JSON sidecars for config-specific advice

---

## QUICK START

### Installation Check

Agent is already installed. Verify connection:

```bash
# Check MCP connection
/mcp

# Should show: "Reconnected to dxr-image-quality-analyst"
```

### First Use: Capture & Analyze

```bash
# 1. Run PlasmaDX and capture screenshot (F2 key)
./build/Debug/PlasmaDX-Clean.exe

# 2. Analyze most recent screenshot
/mcp dxr-image-quality-analyst assess_visual_quality \
  --screenshot_path "build/bin/Debug/screenshots/screenshot_latest.bmp"
```

### First Comparison: Before/After Fix

```bash
# 1. Capture baseline BEFORE fix (F2)
# 2. Apply fix, rebuild, capture AFTER (F2)

# 3. ML comparison
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "screenshots/before.bmp" \
  --after_path "screenshots/after.bmp" \
  --save_heatmap true

# 4. Check heatmap in PIX/heatmaps/
```

---

## TOOL SUITE (5 Tools)

### 1. `compare_screenshots_ml` ⭐ PRIMARY TOOL

**ML-powered before/after comparison using LPIPS perceptual similarity**

```bash
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "screenshots/before_fix.bmp" \
  --after_path "screenshots/after_fix.bmp" \
  --save_heatmap true
```

**Output:**
- LPIPS Perceptual Score (0.0 = identical, 1.0 = completely different)
- Traditional Metrics (SSIM, MSE, PSNR, histogram correlation)
- Difference Statistics (mean/max difference, changed pixel %)
- Difference Heatmap (saved to PIX/heatmaps/)
- Auto-resize Warning (if dimensions differ)

**Interpretation:**

| LPIPS Score | Similarity | Human Perception |
|-------------|------------|------------------|
| 0.00 - 0.10 | 90-100% | Imperceptible difference |
| 0.10 - 0.30 | 70-90% | Noticeable but minor |
| 0.30 - 0.50 | 50-70% | Obvious differences |
| 0.50 - 0.70 | 30-50% | Very different |
| 0.70 - 1.00 | 0-30% | Completely different |

### 2. `assess_visual_quality`

**Comprehensive visual quality assessment using 7-dimension rubric**

```bash
/mcp dxr-image-quality-analyst assess_visual_quality \
  --screenshot_path "build/bin/Debug/screenshots/screenshot_latest.bmp"
```

**Output:**
- Overall Grade (A-F, 0-100 score)
- 7 Quality Dimensions (Volumetric Depth, Lighting, Temperature, RTXDI, Shadows, Scattering, Temporal)
- Key Observations (what's visible)
- Comparison to Goals (vs target aesthetic)
- Actionable Recommendations (specific file:line fixes)
- Technical Notes (root cause analysis)

### 3. `list_recent_screenshots`

**Quick overview of available screenshots sorted by time**

```bash
/mcp dxr-image-quality-analyst list_recent_screenshots --limit 10
```

**Output:** Filename, path, timestamp, file size, metadata status

### 4. `compare_performance`

**Compare RTXDI performance metrics between renderers**

```bash
/mcp dxr-image-quality-analyst compare_performance \
  --legacy_log "logs/legacy.log" \
  --rtxdi_m4_log "logs/rtxdi_m4.log" \
  --rtxdi_m5_log "logs/rtxdi_m5.log"
```

### 5. `analyze_pix_capture`

**Analyze PIX GPU captures for bottlenecks**

```bash
/mcp dxr-image-quality-analyst analyze_pix_capture \
  --capture_path "PIX/Captures/capture_latest.wpix"
```

---

## EFFECTIVE WORKFLOWS

### Workflow 1: Material System Development (CURRENT USE CASE)

**Goal:** Track visual quality during Gaussian particle material system development

```bash
# 1. Baseline capture before changes (F2)
# 2. Make material system changes
# 3. Rebuild + capture after changes (F2)

# 4. ML comparison
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "baseline.bmp" \
  --after_path "after_material_system.bmp" \
  --save_heatmap true

# 5. If LPIPS > 0.30 (significant difference), get detailed assessment
/mcp dxr-image-quality-analyst assess_visual_quality \
  --screenshot_path "after_material_system.bmp"
```

**Thresholds:**
- **LPIPS < 0.10:** No visual regression, changes safe
- **LPIPS 0.10-0.30:** Minor differences, verify intentional
- **LPIPS > 0.30:** Significant differences, needs investigation
- **LPIPS > 0.50:** Critical visual regression OR breakthrough improvement

### Workflow 2: Performance Optimization

**Goal:** Ensure performance optimizations don't degrade visual quality

```bash
# 1. Capture baseline (quality preset)
# 2. Apply optimization (performance preset)
# 3. Capture after optimization

# 4. ML comparison
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "quality_8rays.bmp" \
  --after_path "performance_1ray.bmp" \
  --save_heatmap true

# Acceptance: FPS >= +50%, LPIPS < 0.30
```

### Workflow 3: Weekly Visual Quality Audit

**Goal:** Catch visual regressions early

```bash
# 1. Standard test capture (same config every week)
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/standard_test.json  # F2

# 2. Visual assessment
/mcp dxr-image-quality-analyst assess_visual_quality \
  --screenshot_path "weekly_test_2025-11-12.bmp"

# 3. Compare to last week
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "weekly_test_2025-11-05.bmp" \
  --after_path "weekly_test_2025-11-12.bmp" \
  --save_heatmap true
```

**Red Flags:**
- LPIPS > 0.50 (major visual change)
- FPS regression >20%
- Grade drop (B+ → D-)
- Lights disabled (metadata shows 0 lights)

---

## BEST PRACTICES

### 1. Capture Consistency

**Always use same camera distance/angle for comparisons**
- Use standard test position (press '1' in PlasmaDX)
- Different angles create false positives in ML comparison

### 2. Metadata is Critical

**Always capture with F2 (in-app), not external screenshot tools**
- External tools (PrintScreen, OBS) don't generate metadata JSON
- Agent needs metadata for config-specific recommendations

### 3. ML Comparison Thresholds

| Scenario | Threshold | Action |
|----------|-----------|--------|
| Bug fix (expect no visual change) | < 0.10 | Accept if LPIPS < 0.10 |
| Performance optimization | < 0.30 | Accept if quality trade-off acceptable |
| New feature (expect visual change) | > 0.30 | Expected, validate improvement |
| Regression detection | > 0.50 | RED FLAG - investigate immediately |

### 4. Heatmap Analysis

**Read heatmaps correctly:**
- **Blue (cool):** No difference
- **Yellow:** Minor differences
- **Red (hot):** Major differences

**Expected patterns:**
- Shadow changes: Red edges around particles (expected)
- Material changes: Red in particle centers (expected)
- Uniform red everywhere: Critical regression (NOT expected)

---

## TROUBLESHOOTING

### MCP Server Not Connected

```bash
# 1. Check MCP log
tail -20 ~/.cache/claude-cli-nodejs/.../mcp-logs-dxr-image-quality-analyst/latest.txt

# 2. Restart server
cd agents/dxr-image-quality-analyst
./run_server.sh

# 3. Reconnect
/mcp
```

### ModuleNotFoundError (numpy, torch)

```bash
cd agents/dxr-image-quality-analyst
source venv/bin/activate
pip install numpy pandas Pillow opencv-python-headless scikit-image torch torchvision lpips tqdm
```

### Heatmap Not Saving

```bash
# Create directory
mkdir -p PIX/heatmaps
chmod 755 PIX/heatmaps

# Verify parameter
--save_heatmap true  # Must be 'true', not 'True' or '1'
```

---

## IMPROVEMENT IDEAS

### Short-Term (1-2 weeks)

1. **Batch Comparison Mode** - Compare 5+ screenshots simultaneously
2. **Temporal Stability Analysis** - Analyze 60 consecutive frames for flickering
3. **Material-Specific Quality Rubric** - Different criteria for PLASMA vs STAR vs GAS
4. **Regression Test Suite** - Automated visual regression testing
5. **Reference Image Library** - "Golden standard" images for comparison

### Medium-Term (2-4 weeks)

6. **Perceptual Quality Predictor** - ML model predicts quality score instantly
7. **Artifact Detection** - Auto-detect fireflies, banding, aliasing
8. **Interactive Heatmap Viewer** - Web-based zoom/pan/click analysis
9. **Performance-Quality Trade-off Analyzer** - Plot FPS vs LPIPS Pareto frontier

### Long-Term (1-2 months)

10. **Real-Time Quality Monitoring** - Live LPIPS overlaid in PlasmaDX viewport
11. **Volumetric Rendering Simulator** - Predict visual outcome before rendering
12. **Aesthetic Style Transfer** - "Make it look like [NASA reference image]"

---

## ADDITIONAL TOOL SUGGESTIONS

### Tool 7: `extract_best_frame` (Video Analysis)

Analyze 60-frame video capture and extract best single frame

### Tool 8: `suggest_config_changes` (Config Optimizer)

Given screenshot + target aesthetic, suggest config changes

### Tool 9: `generate_visual_report` (Batch Reporting)

Generate PDF report with 10+ screenshots, ML comparisons, assessments

### Tool 10: `detect_visual_bugs` (Bug Detection)

Automatically detect black screens, NaN particles, uniform colors

---

## AGENT PHILOSOPHY

### Core Principles

**1. Brutal Honesty Over Sugar-Coating**

❌ **Bad:** "Lighting could use some refinement to improve visual quality"
✅ **Good:** "ZERO LIGHTS ACTIVE - this is catastrophic, cannot assess visual quality without lights"

**2. Actionable Recommendations**

❌ **Bad:** "Gaussian rendering needs work"
✅ **Good:** "Fix Gaussian rendering in `particle_gaussian_raytrace.hlsl:234` - ray-ellipsoid intersection returning NaN"

**3. Context-Aware Analysis**

Understands:
- PlasmaDX goals: Smooth volumetric plasma, not geometric primitives
- Material system status: In development, expect early-stage issues
- Performance targets: 120 FPS @ 1440p with 13 lights
- Visual aesthetic: Glowing accretion disk with rim lighting

**4. Performance Anomalies are Red Flags**

❌ **Ignore:** "19 FPS is a bit slow, consider optimization"
✅ **Flag:** "19 FPS with ZERO lights is nonsensical - something burning GPU cycles for no benefit. Investigate immediately."

**5. Quantitative + Qualitative**

Combine numbers (LPIPS scores, FPS metrics) with descriptions (root causes, visual observations)

---

## TECHNICAL DETAILS

**Language:** Python 3.12
**Framework:** Claude Agent SDK 0.1.4
**ML Framework:** PyTorch 2.9.1 + LPIPS 0.1.4
**Computer Vision:** OpenCV 4.12, scikit-image 0.25.2
**Total size:** ~1.2GB (including PyTorch + LPIPS weights)

**Performance:**
- 1920×1080 images: ~3-5 seconds (first run), ~1-2 seconds (cached)
- 4K images: ~8-10 seconds (auto-resize)
- Lazy loading: PyTorch loaded only when `compare_screenshots_ml` called

**Dependencies:**
- Core: `claude-agent-sdk` 0.1.4, `mcp` 0.1.4, `python-dotenv`, `rich`
- ML/CV: `torch` 2.9.1, `lpips` 0.1.4, `numpy`, `pandas`, `opencv-python-headless`, `scikit-image`, `Pillow`

---

## DIRECTORY STRUCTURE

```
agents/dxr-image-quality-analyst/
├── rtxdi_server.py              # MCP server entry point
├── run_server.sh                # Launch script
├── requirements.txt             # Python dependencies
├── .env                         # Environment config
├── README.md                    # This file
├── VISUAL_QUALITY_RUBRIC.md     # 7-dimension quality rubric
├── src/
│   └── tools/
│       ├── ml_visual_comparison.py       # ML comparison logic
│       ├── visual_quality_assessment.py  # Quality assessment
│       ├── performance_comparison.py     # Performance analysis
│       ├── screenshot_management.py      # Screenshot utils
│       └── pix_analysis.py              # PIX capture analysis
└── venv/                        # Python virtual environment
```

---

## LPIPS TECHNICAL DETAILS

**What is LPIPS?**

Learned Perceptual Image Patch Similarity - a deep learning metric that measures perceptual similarity with ~92% correlation to human judgment.

**How It Works:**
1. Extract features using pre-trained AlexNet
2. Compute distance in feature space (not pixel space)
3. Return similarity score (0.0 = identical, 1.0 = completely different)

**Why Better Than Traditional Metrics:**
- Understands semantic content (edges, textures, objects)
- Matches human perception (92% correlation)
- Detects perceptual differences humans care about

**Example:**

Modification A: Shift image 1 pixel left
- MSE: High (every pixel different)
- LPIPS: Low (humans barely notice)

Modification B: Change plasma color blue → red
- MSE: Low (only hue changed)
- LPIPS: High (humans immediately notice)

LPIPS correctly identifies Modification B as more significant.

---

## CHANGELOG

### v1.21.0 (2025-11-12)
- ✅ Added auto-resize for mismatched image dimensions
- ✅ Added ML visual comparison (PyTorch + LPIPS)
- ✅ Extended metadata schema (5 new sections)
- ✅ Renamed from rtxdi-quality-analyzer to dxr-image-quality-analyst
- ✅ Added comprehensive README and cleaned up redundant docs

### v1.20.0 (2025-10-24)
- ✅ Added visual quality assessment tool
- ✅ Created 7-dimension quality rubric
- ✅ Added metadata Phase 1 integration

### v1.10.0 (2025-10-23)
- ✅ Initial MCP server implementation
- ✅ Performance comparison, PIX analysis, screenshot listing tools

---

## SUPPORT

**Questions/Issues:** Open issue in PlasmaDX-Clean repository
**MCP Logs:** `~/.cache/claude-cli-nodejs/.../mcp-logs-dxr-image-quality-analyst/`
**Maintainer:** Ben (PlasmaDX project lead)
**Agent Philosophy:** Brutal honesty - no sugar-coating!

---

**Last Updated:** 2025-11-12
**Agent Version:** 1.21.0
**Status:** ✅ Fully Operational - ML Visual Analysis Ready
