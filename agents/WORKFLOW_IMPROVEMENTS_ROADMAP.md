# Workflow Improvements Roadmap

**Goal:** Create an intelligent, context-aware workflow that proactively uses the right tools for the right tasks, reducing cognitive load and preventing tool underutilization.

**Problem Statement:** Despite having powerful diagnostic tools (rtxdi-quality-analyzer, pix-debug), workflow tends toward using only the base model (Sonnet 4.5), leading to tunnel vision and overwhelm when tracking multiple complex tasks.

---

## Phase 1: Smart Context Detection (Immediate - No Code Changes)

**Status:** Ready to implement today
**Effort:** Configuration only
**Impact:** High - Immediate workflow improvement

### Implementation

Claude (Sonnet 4.5) already has the capability to:
1. **Recognize context keywords** ("performance", "FPS", "bottleneck", "slower", "RTXDI")
2. **Auto-invoke tools** based on conversation context
3. **Chain tool outputs** intelligently
4. **Suggest next steps** based on findings

### Pattern Recognition Examples

**Pattern 1: Performance Investigation**
```
User: "RTXDI M4 seems slower than legacy"

Claude detects: performance comparison context
→ Auto-invokes: rtxdi-quality-analyzer compare_performance
→ Shows results inline
→ If bottleneck found → Suggests: "Should I analyze with pix-debug?"
```

**Pattern 2: Visual Issues**
```
User: "The lighting looks wrong"

Claude detects: visual debugging context
→ Suggests: capture_screenshot (when implemented)
→ Invokes: pix-debug diagnose_visual_artifact
→ Provides: Specific diagnosis with file:line fixes
```

**Pattern 3: Buffer Analysis**
```
User: "Reservoir buffers seem corrupted"

Claude detects: buffer validation context
→ Invokes: pix-debug analyze_restir_reservoirs
→ Provides: Statistical analysis
→ Suggests: Next debugging steps
```

### Activation

**No code changes needed!** Just natural conversation:
- Describe problems naturally
- Claude recognizes patterns
- Tools are invoked automatically
- Results integrated into analysis

---

## Phase 2: Visual Feedback System (Next Session - 1 Hour)

**Status:** Not started
**Effort:** 1-2 hours implementation
**Impact:** Game-changing - Direct visual debugging

### Core Components

#### 2.1 Screenshot Capture Tool

**File:** `agents/rtxdi-quality-analyzer/src/tools/visual_capture.py`

**Features:**
- Capture PlasmaDX window screenshot
- Metadata embedding (renderer mode, FPS, particle count)
- Base64 encoding for Claude analysis
- Configurable save location

**Implementation:**
```python
import subprocess
from PIL import ImageGrab
import base64

async def capture_screenshot(
    save_path: str = None,
    include_metadata: bool = True
) -> dict:
    """
    Capture PlasmaDX window screenshot

    Returns:
        {
            "image_base64": "...",
            "metadata": {
                "renderer": "RTXDI_M4",
                "fps": 115.0,
                "particle_count": 10000,
                "timestamp": "2025-10-23T..."
            }
        }
    """
    # Use Windows API to capture specific window title
    # Or ImageGrab for full screen
    # Downscale to 800x600 for token efficiency
    pass
```

**Token Efficiency:**
- Full resolution (1920x1080): ~1000 tokens
- Downscaled (800x600): ~500 tokens
- ROI only (UI region): ~300 tokens

**Recommended:** Downscaled for balance of quality vs. tokens

#### 2.2 Hotkey Trigger (Optional Enhancement)

**File:** `agents/rtxdi-quality-analyzer/src/tools/screenshot_listener.py`

**Hotkey:** Ctrl+Shift+C (configurable)

**Behavior:**
1. User presses hotkey while PlasmaDX is running
2. Screenshot captured automatically
3. Saved to `PIX/screenshots/screenshot_<timestamp>.png`
4. Notification sent to Claude Code session
5. Claude analyzes immediately

**Implementation:**
```python
from pynput import keyboard
import asyncio

class ScreenshotListener:
    def __init__(self):
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse('<ctrl>+<shift>+c'),
            self.on_activate
        )

    def on_activate(self):
        # Trigger screenshot capture
        # Send notification to Claude session
        pass
```

#### 2.3 Smart Triggered Captures

**Trigger Types:**

**A. Renderer Switch Detection**
```python
# Monitor for F7/F8 key presses (renderer toggles)
# Capture before/after screenshots
# Auto-invoke compare_visual_changes tool
```

**B. Performance Drop Detection**
```python
# Monitor FPS in real-time via log file
# If FPS drops > 20% → Auto-capture
# Send to Claude: "FPS dropped from 120 to 95, analyzing..."
```

**C. User-Initiated Manual Capture**
```python
# Hotkey: Ctrl+Shift+C
# Captures current frame
# Claude analyzes immediately
```

### Claude's Visual Analysis Capabilities

Once images are provided, Claude can detect:
- ✅ Light distribution patterns (uneven coverage, saturation)
- ✅ Temporal accumulation artifacts (patchwork patterns)
- ✅ Color/intensity issues
- ✅ UI state (active renderer mode from UI text)
- ✅ FPS counter values
- ✅ Visual glitches, missing effects
- ✅ Particle density and distribution
- ✅ Shadow quality issues

**Example Analysis:**
```
Claude receives screenshot of RTXDI M4:

"I can see several issues:
1. Light grid coverage is uneven - bright spots at disk edges suggest
   cell saturation. Check light_grid_coverage metric.
2. FPS counter shows 115 - matches our performance data
3. Temporal patchwork pattern visible in lower-right quadrant
   (expected in M4, should smooth out in M5)
4. Particle density looks correct (~10K particles visible)

Recommendation: The uneven lighting is your primary issue.
Let me analyze the latest PIX capture for light grid metrics..."

→ Auto-invokes: pix-debug analyze_pix_capture
```

---

## Phase 3: Before/After Visual Comparison (2-3 Hours)

**Status:** Not started
**Effort:** 2-3 hours
**Impact:** High - Quantify visual improvements

### Features

**Visual Diff Analysis**
```python
Tool(
    name="compare_visual_changes",
    description="Compare screenshots to detect visual differences",
    inputSchema={
        "before_path": "screenshot_before.png",
        "after_path": "screenshot_after.png",
        "diff_mode": "pixel|structural|perceptual"
    }
)
```

**Diff Modes:**
1. **Pixel diff** - Exact pixel-by-pixel comparison
2. **Structural diff** - SSIM (Structural Similarity Index)
3. **Perceptual diff** - Human-visible differences only

**Output:**
- Side-by-side comparison
- Difference heatmap
- Quantitative metrics (% changed, affected regions)
- Claude's qualitative analysis

**Use Cases:**
- Compare legacy vs RTXDI M4 rendering
- Validate optimization impact
- Detect visual regressions
- A/B test shadow quality presets

---

## Phase 4: Automated Visual Regression Detection (4-6 Hours)

**Status:** Future enhancement
**Effort:** 4-6 hours
**Impact:** Medium - Continuous quality monitoring

### Features

**Baseline Capture System**
- Capture "golden" screenshots for each renderer mode
- Store in `PIX/baselines/`
- Version control baselines in git

**Regression Detection**
- After code changes, auto-capture new screenshots
- Compare against baselines
- Flag differences > threshold
- Claude analyzes: "Regression detected in RTXDI M4 - light intensity reduced by 15%"

**Implementation:**
```python
async def detect_visual_regression(
    baseline_dir: str,
    current_screenshot: str,
    threshold: float = 0.95  # SSIM similarity threshold
) -> dict:
    """
    Compare current screenshot against baseline
    Returns regression report if similarity < threshold
    """
    pass
```

---

## Phase 5: Real-Time Performance Monitoring (3-4 Hours)

**Status:** Future enhancement
**Effort:** 3-4 hours
**Impact:** High - Proactive issue detection

### Features

**Live FPS Monitoring**
```python
# Watch log file for FPS updates
# Real-time streaming to Claude session
# Alert on performance drops
```

**Auto-Analysis on Drops**
```
FPS drops from 120 to 95:
→ Auto-capture screenshot
→ Invoke pix-debug for diagnosis
→ Claude provides immediate recommendations
```

**Integration with TodoWrite**
```
Claude detects FPS drop:
→ Creates todo: "Investigate FPS drop from 120 to 95"
→ Automatically adds sub-tasks:
  - Capture PIX trace
  - Analyze BLAS rebuild times
  - Check RTXDI overhead
```

---

## Implementation Priority

### **Immediate (This Session):**
1. ✅ Smart context detection - Use tools proactively based on conversation
2. ✅ TodoWrite tracking - Maintain session context visibility
3. ✅ Proactive suggestions - "Should I analyze that with X?"

### **Next Session (1-2 hours):**
1. Add `capture_screenshot` tool to rtxdi-quality-analyzer
2. Test manual screenshot capture
3. Verify Claude can analyze images effectively

### **Short Term (Next Few Sessions - 2-4 hours):**
1. Implement hotkey trigger (Ctrl+Shift+C)
2. Add before/after comparison tool
3. Integrate screenshot metadata (renderer mode, FPS)

### **Medium Term (Future - 4-8 hours):**
1. Smart triggered captures (renderer switch, FPS drop)
2. Visual regression detection
3. Real-time performance monitoring
4. Automated baseline management

### **Long Term (Future - 8+ hours):**
1. ML-based visual anomaly detection
2. Predictive performance alerts
3. Automated optimization suggestions
4. Integration with CI/CD for regression tests

---

## Token Budget Planning

**Conservative Estimate:**
- 10 screenshots/session × 500 tokens (downscaled) = **5,000 tokens**
- Typical debugging session without images = **50,000 tokens**
- **Net ROI:** Hours saved in debugging >>> token cost

**Token Efficiency Strategies:**
1. ✅ Downscale to 800x600 (saves 50% tokens)
2. ✅ ROI capture only (saves 70% tokens)
3. ✅ User-controlled triggers (no auto-spam)
4. ✅ Smart intervals (not every frame)
5. ✅ Diff-based (only capture when screen changes significantly)

**Recommended Budget:**
- **Development mode:** 10-15 screenshots/session (5,000-7,500 tokens)
- **Production mode:** 5-10 screenshots/session (2,500-5,000 tokens)
- **Emergency debug:** Unlimited (diagnose critical issues)

---

## Success Metrics

**Workflow Efficiency:**
- ❌ Current: Manually invoke tools ~10% of the time
- ✅ Target: Tools invoked automatically ~80% of the time

**Debugging Speed:**
- ❌ Current: 30-60 minutes to diagnose visual issues
- ✅ Target: 5-10 minutes with screenshot analysis

**Context Retention:**
- ❌ Current: Lose track after 3-4 tasks in parallel
- ✅ Target: TodoWrite maintains context for 10+ tasks

**Cognitive Load:**
- ❌ Current: Manually remember to use tools
- ✅ Target: Claude proactively suggests correct tools

---

## Next Steps

1. **Test smart context detection** - Start describing problems naturally, watch Claude auto-invoke tools
2. **Implement screenshot capture** - Add to rtxdi-quality-analyzer (30 min)
3. **Test visual analysis** - Send Claude a screenshot, verify analysis quality
4. **Iterate based on feedback** - Adjust token budget, capture frequency, analysis depth

---

**Last Updated:** 2025-10-23
**Status:** Phase 1 ready, Phase 2-5 planned
**Owner:** Claude Code + Ben
