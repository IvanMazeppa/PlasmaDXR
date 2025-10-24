# Visual Quality Assessment Tool - Complete Guide

## Overview

**Version:** 1.0 (using Claude Agent SDK 0.1.4 with base64 image support)
**Created:** 2025-10-24
**Author:** Ben + Claude Code

This tool provides AI-powered visual quality assessment for PlasmaDX's volumetric black hole accretion disk renderer. It leverages Claude's vision capabilities to analyze screenshots against a comprehensive quality rubric.

---

## Key Innovation: Image Content Blocks

**Claude Agent SDK 0.1.4** (released October 2025) added support for **image content blocks** in MCP tool responses. This means the tool can now:

1. Load a screenshot from disk
2. Encode it as base64
3. Return it alongside text context
4. **Claude Code sees both** and performs visual analysis

**No manual screenshot sharing required!** The workflow is fully automated.

---

## Architecture

### Components

1. **Visual Quality Rubric** (`VISUAL_QUALITY_RUBRIC.md`)
   - 500+ line document describing what "good" looks like
   - 7 quality dimensions (volumetric depth, rim lighting, temperature, RTXDI, shadows, scattering, temporal stability)
   - Reference to "golden standard" renders
   - Structured assessment format

2. **Quality Assessment Tool** (`src/tools/visual_quality_assessment.py`)
   - Loads the rubric
   - Prepares context for AI analysis
   - Manages reference images
   - Saves/loads quality annotations (JSON)

3. **MCP Server Integration** (`rtxdi_server.py`)
   - New `assess_visual_quality` tool
   - Returns image content blocks using MCP standard format
   - Uses lazy loading to avoid timeout (PyTorch/LPIPS loaded on-demand)

---

## MCP Tool: `assess_visual_quality`

### Description

Analyzes a screenshot for volumetric rendering quality using Claude's vision model. Returns both the quality rubric context and the actual screenshot as an image.

### Parameters

```json
{
  "screenshot_path": "string (required)",
  "comparison_before": "string (optional)"
}
```

### Example Usage

```
User: "Assess the visual quality of screenshot_2025-10-24_03-31-46.bmp"

MCP Tool Returns:
[
  {
    "type": "text",
    "text": "<Quality rubric + analysis prompt>"
  },
  {
    "type": "image",
    "mimeType": "image/bmp",
    "data": "<base64_encoded_image>"
  }
]

Claude Code (me) sees both and responds with:
"
VOLUMETRIC RENDERING QUALITY ASSESSMENT

Screenshot: screenshot_2025-10-24_03-31-46.bmp
Overall Grade: A- (85/100)

=== CRITICAL DIMENSIONS ===

1. Volumetric Depth & Atmosphere: EXCELLENT (92/100)
   ✅ Clear 3D depth perception visible
   ✅ Smooth volumetric falloff between particles
   ✅ Atmospheric scattering creates soft glow
   ⚠️  Minor: Some edge particles slightly too sharp

...
"
```

### Return Format (MCP Standard)

The tool returns a list containing:

1. **Text Content**: Quality rubric + analysis prompt
   ```python
   TextContent(type="text", text=context_string)
   ```

2. **Image Content**: Base64-encoded screenshot
   ```python
   ImageContent(
       type="image",
       data=base64_encoded_string,
       mimeType="image/png" or "image/bmp"
   )
   ```

**IMPORTANT:** Uses MCP standard format (not Anthropic message API format):
- ✅ Correct: `{"type": "image", "data": "...", "mimeType": "..."}`
- ❌ Incorrect: `{"type": "image", "source": {"type": "base64", ...}}`

Reference: [GitHub PR #175](https://github.com/anthropics/claude-agent-sdk-python/pull/175)

---

## Quality Rubric Summary

### The 7 Dimensions

1. **Volumetric Depth & Atmosphere** (CRITICAL)
   - Can you perceive 3D depth?
   - Smooth volumetric density falloff?
   - Atmospheric scattering visible?

2. **Lighting Quality & Rim Lighting** (CRITICAL)
   - Rim lighting halos on backlit particles?
   - Multi-directional shadowing?
   - Soft shadow penumbra?

3. **Temperature Gradient** (HIGH)
   - Smooth hot→cool color transition?
   - Physically accurate blackbody colors?
   - Hottest near black hole?

4. **RTXDI Sampling Quality** (HIGH)
   - Smooth, temporally stable lighting?
   - No patchwork pattern (M4 artifact)?
   - Even light distribution?

5. **Shadow Quality** (MEDIUM)
   - Soft shadow edges (PCSS)?
   - Particle-to-particle occlusion?

6. **Anisotropic Scattering** (MEDIUM)
   - Forward scattering dominates?
   - Directional vs uniform?

7. **Temporal Stability** (MEDIUM)
   - Smooth frame timing?
   - Quick convergence?

### Grading Scale

- **A+ (95-100):** Golden standard - "gorgeous"
- **A (90-94):** Excellent quality
- **B (80-89):** Good quality, minor issues
- **C (70-79):** Acceptable, noticeable issues
- **D (60-69):** Poor quality, major problems
- **F (<60):** Broken render

---

## Workflow Examples

### Example 1: Analyze Single Screenshot

**User request:**
> "Analyze screenshot_2025-10-24_03-31-46.bmp for rendering quality"

**What happens:**
1. Tool loads visual quality rubric
2. Tool loads and encodes screenshot as base64
3. Tool returns both text and image to Claude Code
4. Claude Code sees the rubric and image
5. Claude Code provides detailed assessment across 7 dimensions
6. (Optional) Claude Code saves assessment as JSON annotation

**Expected output:**
- Overall grade (A-F)
- Score for each dimension (0-100)
- Specific observations (what's working, what's not)
- Actionable recommendations
- Comparison to "golden standard" aesthetic

### Example 2: Compare Before/After

**User request:**
> "Compare RTXDI M4 vs M5 quality using screenshot_before.bmp and screenshot_after.bmp"

**What happens:**
1. Tool loads both screenshots
2. Tool prepares comparison context (rubric + before/after prompt)
3. Tool returns: rubric text + before image + after image
4. Claude Code performs side-by-side quality assessment
5. Claude Code identifies improvements/regressions

**Expected output:**
- What changed between the two?
- Is it an improvement or regression?
- Which quality dimensions were affected?
- Did RTXDI M5 eliminate the patchwork pattern (expected improvement)?

### Example 3: Build Reference Library

**User request:**
> "Help me identify my best screenshots to use as golden standard references"

**What happens:**
1. Tool lists all screenshots
2. Claude Code analyzes each one
3. Claude Code ranks by quality score
4. User copies top screenshots to `screenshots/reference/golden_standard/`
5. Tool saves annotations for each

**Result:** Curated library of reference images for future comparisons

---

## Technical Implementation Details

### Base64 Encoding

```python
# Load screenshot
with open(screenshot_path, 'rb') as f:
    image_bytes = f.read()

# Encode to base64
encoded_image = base64.b64encode(image_bytes).decode('utf-8')

# Determine MIME type
mime_type = "image/png" if screenshot_path.endswith(".png") else "image/bmp"

# Return as ImageContent
ImageContent(
    type="image",
    data=encoded_image,
    mimeType=mime_type
)
```

### Image Size Limits

- **BMP screenshots:** ~6 MB each @ 1440p (1920×1080 upscaled)
- **Base64 encoded:** ~8 MB (4/3 overhead)
- **Claude's limit:** ~100 MB per request (plenty of headroom)

### Supported Formats

- ✅ PNG (compressed, smaller)
- ✅ BMP (uncompressed, larger but lossless)
- ✅ JPEG (if needed, though not recommended for technical analysis)

---

## File Structure

```
agents/rtxdi-quality-analyzer/
├── VISUAL_QUALITY_RUBRIC.md          # 500+ line quality guide
├── VISUAL_QUALITY_TOOL_GUIDE.md      # This file
├── rtxdi_server.py                   # MCP server with 5 tools
├── run_server.sh                     # Launcher script
├── src/
│   └── tools/
│       ├── visual_quality_assessment.py    # Quality tool implementation
│       ├── ml_visual_comparison.py         # LPIPS comparison (existing)
│       ├── performance_comparison.py       # Performance metrics (existing)
│       └── pix_analysis.py                 # PIX capture analysis (existing)
└── requirements.txt                  # Python dependencies

screenshots/
├── screenshot_*.bmp                  # F2 captures (main working directory)
├── reference/                        # Curated reference library
│   ├── golden_standard/              # Best renders (A+ grade)
│   ├── good/                         # Acceptable (B grade)
│   ├── issues/                       # Known problems (C-D grade)
│   └── failures/                     # Broken renders (F grade)
└── annotations/                      # JSON quality assessments
    └── screenshot_*.json             # Saved analysis results
```

---

## Setup Instructions

### 1. Ensure SDK 0.1.4 Installed

```bash
pip install claude-agent-sdk==0.1.4
```

### 2. Verify All Dependencies

```bash
source venv/bin/activate
pip list | grep -E "claude-agent-sdk|Pillow|torch|lpips"

# Should see:
# claude-agent-sdk    0.1.4
# Pillow              10.1.0
# torch               2.9.0+cpu
# lpips               0.1.4
```

### 3. Test Server Startup

```bash
cd agents/rtxdi-quality-analyzer
./run_server.sh &
sleep 5
kill %1

# Should start without errors (~8 seconds with lazy loading)
```

### 4. Reconnect MCP Server in Claude Code

After updating the server, reconnect to get the new `assess_visual_quality` tool:

1. Run: `/mcp disconnect rtxdi-quality-analyzer`
2. Run: `/mcp connect rtxdi-quality-analyzer`
3. Verify: Tool list should include `assess_visual_quality`

---

## Usage Tips

### Pause Physics for Comparison

When comparing rendering settings (not physics), pause particle simulation:

**Why:** Particle movement creates noise in visual comparisons. Pausing isolates rendering quality changes.

**How:** Add a keyboard shortcut (e.g., `P` key) to toggle `m_physicsPaused`

### Use Consistent Camera Angles

When building reference library, capture from standard angles:
- Front view (default)
- Side view (90° rotation)
- Top-down (looking down at disk)
- Close-up (near black hole)

This enables better before/after comparisons.

### Annotate Your Best Renders

After getting a great screenshot, immediately analyze it and save the assessment:

```
User: "Analyze screenshot_X.bmp and save the assessment"
```

This builds your reference library organically over time.

---

## Troubleshooting

### Issue: "Screenshot not found"

**Cause:** Path is relative instead of absolute

**Fix:** Use absolute paths or paths relative to project root:
```
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/screenshots/screenshot_X.bmp
```

### Issue: "Module not found: skimage"

**Cause:** MCP server using wrong Python environment

**Fix:** Verify `run_server.sh` sources the correct venv:
```bash
source "$PROJECT_ROOT/venv/bin/activate"
```

### Issue: Image not appearing in Claude's analysis

**Cause:** Using wrong image content format (Anthropic API instead of MCP standard)

**Fix:** Use MCP format:
```python
ImageContent(type="image", data="...", mimeType="...")
```

NOT:
```python
{"type": "image", "source": {"type": "base64", ...}}
```

### Issue: Server timeout on startup

**Cause:** PyTorch/LPIPS loading at import time (528MB)

**Fix:** Already implemented! Lazy loading pattern defers ML imports until first use.

---

## Future Enhancements

### Phase 1: Automated Quality Tracking ✅ COMPLETE

- ✅ Visual quality rubric documented
- ✅ MCP tool returns images
- ✅ Claude analyzes against 7 dimensions
- ✅ Annotations saved as JSON

### Phase 2: Reference Library (IN PROGRESS 🔄)

- 🔄 Curate golden standard renders
- 🔄 Build comparison database
- ⏳ Automated similarity scoring vs references

### Phase 3: Regression Detection (PLANNED ⏳)

- ⏳ Track quality scores over time
- ⏳ Alert on quality regressions
- ⏳ Git integration (link commits to quality changes)

### Phase 4: Automated A/B Testing (PLANNED ⏳)

- ⏳ Capture before/after for every setting change
- ⏳ Automated comparison + report generation
- ⏳ "Did this change improve quality?" answer

---

## Credits

**Built with:**
- Claude Agent SDK 0.1.4 (Anthropic)
- Claude Sonnet 4.5 (vision model)
- LPIPS (Learned Perceptual Image Patch Similarity)
- PyTorch (ML framework)

**Developed by:** Ben + Claude Code
**Date:** 2025-10-24
**Project:** PlasmaDX-Clean (Volumetric Black Hole Renderer)

---

**Ready to use!** Try it now:

```
"Assess the visual quality of my most recent screenshot"
```

The tool will load the rubric, encode the image, and I'll provide a comprehensive quality assessment across all 7 dimensions! 🎉
