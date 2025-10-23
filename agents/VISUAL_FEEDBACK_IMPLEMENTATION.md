# Visual Feedback System - Implementation Guide

**Goal:** Enable Claude to see what you see - screenshots of PlasmaDX running, analyzed in real-time for visual debugging, performance analysis, and optimization suggestions.

**Status:** Not started
**Effort:** 1-2 hours for basic implementation
**Impact:** Game-changing for visual debugging

---

## Architecture Overview

```
PlasmaDX-Clean.exe
       ↓
   (Hotkey: Ctrl+Shift+C)
       ↓
Screenshot Capture Tool
       ↓
   Metadata Embedding
       ↓
Base64 Encode + Send to Claude
       ↓
Claude Visual Analysis
       ↓
Diagnostic Report + Recommendations
```

---

## Phase 1: Basic Screenshot Capture (30 minutes)

### Implementation

**File:** `agents/rtxdi-quality-analyzer/src/tools/visual_capture.py`

```python
"""
Visual capture tool for PlasmaDX screenshot analysis
"""

import os
import base64
from pathlib import Path
from PIL import ImageGrab, Image
from datetime import datetime
from typing import Dict, Any, Optional
import json


def capture_window_screenshot(
    window_title: str = "PlasmaDX-Clean",
    downscale_width: int = 800
) -> Optional[Image.Image]:
    """
    Capture screenshot of specific window

    Args:
        window_title: Window title to capture
        downscale_width: Target width for token efficiency (default: 800px)

    Returns:
        PIL Image object or None if capture failed
    """
    try:
        import win32gui
        import win32ui
        import win32con
        from ctypes import windll

        # Find window
        hwnd = win32gui.FindWindow(None, window_title)
        if not hwnd:
            # Try partial match
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if window_title.lower() in title.lower():
                        windows.append(hwnd)
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)
            hwnd = windows[0] if windows else None

        if not hwnd:
            print(f"Window '{window_title}' not found")
            return None

        # Get window rect
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        # Capture window
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Copy window to bitmap
        windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

        # Convert to PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)

        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # Clean up
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # Downscale for token efficiency
        if downscale_width and img.width > downscale_width:
            ratio = downscale_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((downscale_width, new_height), Image.Resampling.LANCZOS)

        return img

    except Exception as e:
        print(f"Error capturing window: {e}")
        # Fallback to full screen capture
        img = ImageGrab.grab()
        if downscale_width and img.width > downscale_width:
            ratio = downscale_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((downscale_width, new_height), Image.Resampling.LANCZOS)
        return img


def extract_metadata_from_screenshot(img: Image.Image) -> Dict[str, Any]:
    """
    Extract metadata from screenshot (OCR FPS counter, UI state, etc.)

    Args:
        img: PIL Image

    Returns:
        Metadata dictionary
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "resolution": f"{img.width}x{img.height}",
        "captured": True
    }

    # TODO: OCR to extract:
    # - FPS counter value
    # - Renderer mode (Legacy/RTXDI M4/M5)
    # - Particle count
    # - Light count

    # For now, return basic metadata
    return metadata


def encode_image_base64(img: Image.Image) -> str:
    """
    Encode PIL Image to base64 for Claude

    Args:
        img: PIL Image

    Returns:
        Base64 encoded string
    """
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


async def capture_screenshot(
    save_path: Optional[str] = None,
    include_metadata: bool = True,
    downscale_width: int = 800
) -> Dict[str, Any]:
    """
    Main screenshot capture function

    Args:
        save_path: Optional path to save screenshot (default: auto-generate)
        include_metadata: Extract metadata from screenshot
        downscale_width: Target width for token efficiency

    Returns:
        {
            "success": True/False,
            "image_base64": "...",
            "save_path": "path/to/screenshot.png",
            "metadata": {...}
        }
    """
    try:
        # Capture screenshot
        img = capture_window_screenshot(downscale_width=downscale_width)

        if not img:
            return {
                "success": False,
                "error": "Failed to capture screenshot"
            }

        # Generate save path if not provided
        if not save_path:
            project_root = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
            screenshots_dir = Path(project_root) / "PIX" / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = screenshots_dir / f"screenshot_{timestamp}.png"

        # Save screenshot
        img.save(save_path)

        # Extract metadata
        metadata = {}
        if include_metadata:
            metadata = extract_metadata_from_screenshot(img)

        # Encode for Claude
        img_base64 = encode_image_base64(img)

        return {
            "success": True,
            "image_base64": img_base64,
            "save_path": str(save_path),
            "metadata": metadata,
            "size_kb": len(img_base64) * 3 / 4 / 1024  # Approximate size
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Integration with rtxdi-quality-analyzer

**File:** `agents/rtxdi-quality-analyzer/src/agent.py`

Add new tool to `list_tools()`:

```python
Tool(
    name="capture_screenshot",
    description="Capture screenshot of PlasmaDX-Clean window for visual analysis",
    inputSchema={
        "type": "object",
        "properties": {
            "save_path": {
                "type": "string",
                "description": "Optional path to save screenshot (auto-generated if not provided)"
            },
            "downscale_width": {
                "type": "integer",
                "description": "Target width for token efficiency (default: 800)",
                "default": 800
            }
        },
        "required": []
    }
)
```

Add tool handler in `call_tool()`:

```python
elif name == "capture_screenshot":
    from .tools.visual_capture import capture_screenshot

    result = await capture_screenshot(
        save_path=arguments.get("save_path"),
        downscale_width=arguments.get("downscale_width", 800)
    )

    if result["success"]:
        response = f"Screenshot captured successfully!\n\n"
        response += f"Saved to: {result['save_path']}\n"
        response += f"Size: {result['size_kb']:.1f} KB\n"
        response += f"Metadata: {json.dumps(result['metadata'], indent=2)}\n\n"
        response += f"Image (base64): {result['image_base64'][:100]}... (truncated)"

        return [TextContent(type="text", text=response)]
    else:
        return [TextContent(
            type="text",
            text=f"Screenshot capture failed: {result.get('error', 'Unknown error')}"
        )]
```

### Testing

```bash
# In Claude Code session:
"Capture a screenshot of PlasmaDX"

# Claude invokes: capture_screenshot tool
# Returns: Screenshot saved + base64 encoded for analysis
```

---

## Phase 2: Hotkey Trigger (Optional - 30 minutes)

**File:** `agents/rtxdi-quality-analyzer/src/tools/screenshot_listener.py`

```python
"""
Hotkey listener for triggering screenshots
"""

from pynput import keyboard
import asyncio
from pathlib import Path


class ScreenshotListener:
    """Listen for Ctrl+Shift+C hotkey and trigger screenshot"""

    def __init__(self, capture_callback):
        self.capture_callback = capture_callback
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse('<ctrl>+<shift>+c'),
            self.on_activate
        )

    def on_activate(self):
        """Hotkey activated - trigger screenshot"""
        print("Hotkey detected - capturing screenshot...")
        asyncio.create_task(self.capture_callback())

    def start(self):
        """Start listening for hotkey"""
        with keyboard.Listener(
            on_press=self.hotkey.press,
            on_release=self.hotkey.release
        ) as listener:
            listener.join()


async def screenshot_callback():
    """Callback when hotkey pressed"""
    from .visual_capture import capture_screenshot

    result = await capture_screenshot()
    if result["success"]:
        print(f"Screenshot saved: {result['save_path']}")
        # TODO: Send notification to Claude Code session
    else:
        print(f"Screenshot failed: {result.get('error')}")


def run_listener():
    """Run hotkey listener"""
    listener = ScreenshotListener(screenshot_callback)
    print("Listening for Ctrl+Shift+C...")
    listener.start()


if __name__ == "__main__":
    run_listener()
```

**Usage:**

```bash
# Run in background
python -m src.tools.screenshot_listener &

# Or as systemd service (Linux)
# Or Windows Task Scheduler task
```

---

## Phase 3: Claude Visual Analysis

### What Claude Can See

When you send a screenshot, Claude can analyze:

**✅ Technical Details:**
- Light distribution patterns (uneven, saturated, sparse)
- Temporal accumulation artifacts (patchwork, streaking)
- Shadow quality (soft shadows, hard edges, PCSS effectiveness)
- Particle density and distribution
- Color/intensity issues
- Missing visual effects

**✅ UI State:**
- FPS counter values (via OCR or manual inspection)
- Active renderer mode (from UI text)
- ImGui window contents
- Debug visualizations

**✅ Visual Issues:**
- Black frames
- Color artifacts
- Missing lighting
- Incorrect shadows
- Performance degradation (visible stutter)

### Example Analysis

```
User: [Sends screenshot]

Claude:
"I can analyze this screenshot of PlasmaDX running RTXDI M4:

Visual Analysis:
1. Light Grid Coverage: Uneven - bright concentrated spots at disk edges
   suggest cell saturation. The upper-left quadrant is noticeably darker.

2. Temporal Accumulation: Visible patchwork pattern in the lower-right,
   characteristic of M4's weighted sampling before M5 temporal smoothing.

3. FPS: I can see the counter shows ~115 FPS, matching our performance data.

4. Particle Distribution: Density looks correct (~10K particles visible),
   proper accretion disk structure.

5. Shadow Quality: Soft shadows visible but some hard edges near the
   innermost stable circular orbit (ISCO). PCSS may need tuning.

Recommendations:
1. Investigate light grid saturation → Check light_grid_coverage metric
   File: src/lighting/RTXDILightingSystem.cpp:234

2. The temporal patchwork is expected in M4, should smooth out in M5 once
   ping-pong buffers accumulate samples over 60ms.

3. Shadow hardness near ISCO suggests distance-based PCSS scaling isn't
   working properly. Check blocker search radius.

Would you like me to analyze the latest PIX capture for detailed metrics?"
```

---

## Token Efficiency Guidelines

### Image Sizes vs Token Cost

| Resolution | File Size | Approx Tokens | Use Case |
|-----------|-----------|---------------|----------|
| 1920x1080 | ~300KB | 1000-1200 | Full detail analysis |
| 800x450 | ~80KB | 500-600 | Standard analysis |
| 640x360 | ~50KB | 300-400 | Quick check |
| 400x225 | ~25KB | 150-200 | Minimal (ROI only) |

**Recommended:** 800x450 for balance

### Optimization Strategies

**1. Downscaling (Implemented)**
```python
capture_screenshot(downscale_width=800)  # Saves 50% tokens
```

**2. ROI Capture**
```python
# Crop to region of interest (viewport only, no UI)
img = img.crop((100, 50, 1820, 1030))  # Saves 30% tokens
```

**3. Compression**
```python
# Use JPEG instead of PNG for photos
img.save(buffer, format="JPEG", quality=85)  # Saves 40% tokens
```

**4. Diff-Based**
```python
# Only send if screen changed significantly
if image_diff(previous, current) > threshold:
    send_to_claude(current)
```

### Budget Planning

**Conservative:**
- 5-10 screenshots/session
- 800x450 resolution
- ~3,000-6,000 tokens total
- Cost: Minimal vs debugging time saved

**Aggressive:**
- 15-20 screenshots/session
- Full resolution when needed
- ~15,000-20,000 tokens total
- Still worth it for complex visual debugging

---

## Integration with Existing Tools

### Workflow: Performance Issue → Screenshot → Analysis

```
1. User: "RTXDI M4 is slower than legacy"
   → Claude: Invokes compare_performance tool

2. Claude: "Ray dispatch is 2.8ms, primary bottleneck. Let me see the visual output"
   → Claude: Invokes capture_screenshot tool

3. Claude analyzes screenshot: "I can see light grid is saturated in certain cells..."
   → Claude: Suggests pix-debug analyze_pix_capture

4. User: "Yes, analyze the PIX capture"
   → Claude: Invokes pix-debug tool

5. Claude provides comprehensive diagnosis with file:line fixes
```

---

## Future Enhancements

### Phase 4: Before/After Comparison

```python
async def compare_screenshots(
    before_path: str,
    after_path: str,
    diff_mode: str = "structural"  # pixel|structural|perceptual
) -> Dict[str, Any]:
    """
    Compare two screenshots and show differences

    Returns:
        {
            "similarity": 0.92,  # SSIM score
            "differences": ["Light intensity reduced by 15%", ...],
            "diff_heatmap_base64": "..."
        }
    """
    pass
```

### Phase 5: Video Capture

```python
async def capture_video(
    duration_seconds: int = 10,
    fps: int = 30
) -> str:
    """
    Capture video of PlasmaDX running
    Returns path to MP4 file
    """
    pass
```

### Phase 6: Real-Time Streaming

```python
async def stream_to_claude(
    interval_seconds: int = 5
):
    """
    Stream screenshots to Claude at regular intervals
    Claude provides running commentary
    """
    pass
```

---

## Dependencies

**Python Packages:**

```bash
# requirements.txt additions
Pillow==10.1.0           # Image processing
pywin32==306             # Windows window capture
pynput==1.7.6            # Hotkey listener (optional)
pytesseract==0.3.10      # OCR for metadata extraction (optional)
```

**System Dependencies:**
- Tesseract OCR (optional, for text extraction)
- Windows API (for window capture)

---

## Next Steps

1. **Add visual_capture.py** to rtxdi-quality-analyzer
2. **Update agent.py** with capture_screenshot tool
3. **Test manual capture** - "Capture a screenshot"
4. **Verify Claude analysis** - Send screenshot, check quality
5. **Iterate** - Adjust downscaling, add OCR, etc.

---

**Last Updated:** 2025-10-23
**Status:** Design complete, ready for implementation
**Effort:** 1-2 hours
**Owner:** Ben + Claude Code
