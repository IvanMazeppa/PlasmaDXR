# Screenshot Sharing Solution for Windows/WSL

**Date:** 2025-10-19
**Context:** Need Peekaboo-like screenshot sharing for Windows/WSL environment
**Status:** Native support already exists! + Enhancement script provided

---

## TL;DR - You Already Have It!

**Good News:** Claude Code's Read tool natively supports viewing images, including screenshots!

**Current Workflow (Already Working):**
1. Take screenshot with Windows Snipping Tool (**Win + Shift + S**)
2. Save to `C:\Users\dilli\Pictures\Screenshots\`
3. Provide path to Claude Code: `/mnt/c/Users/dilli/Pictures/Screenshots/filename.png`
4. Claude Code uses Read tool to view it directly

**Proof:** I successfully viewed your screenshot in this session:
- Path: `/mnt/c/Users/dilli/Pictures/Screenshots/Screenshot 2025-10-19 191007.png`
- Result: ✅ Displayed RTXDI debug visualization with Disk preset

---

## What Is Peekaboo?

**Peekaboo** (by Peter Steinberger) is a macOS-specific tool that:
- Captures screenshots of applications or entire system
- Uses AI vision models (GPT-4V, Claude, Ollama) to analyze images
- Works as both CLI tool and MCP server
- Built in Swift (macOS-only) with TypeScript MCP wrapper

**GitHub:** https://github.com/steipete/Peekaboo
**Stars:** ~640
**Platform:** macOS only (uses macOS Core Graphics APIs)

**Key Features:**
- Lightning-fast window/screen capture
- AI-powered image analysis
- GUI automation (v3)
- Menu bar extraction
- Natural language automation

---

## Windows/WSL Equivalent - Three Options

### Option 1: Current Workflow (Recommended - Already Works!)

**Pros:**
- ✅ Zero setup required
- ✅ Works right now
- ✅ Claude Code natively supports image viewing via Read tool
- ✅ No additional dependencies

**Cons:**
- ⚠️ Requires manual save step
- ⚠️ Need to provide full path each time

**Workflow:**
```
1. Win + Shift + S (Windows Snipping Tool)
2. Capture region/window/screen
3. Clipboard → Save to known location
4. Provide path to Claude Code
5. Claude Code reads and displays image
```

**Example:**
```
User: "Here's a screenshot showing the issue"
User: /mnt/c/Users/dilli/Pictures/Screenshots/issue_2025-10-19.png
Claude: <uses Read tool to view image>
Claude: "I can see the RTXDI debug visualization. The patchwork pattern is..."
```

### Option 2: PowerShell Screenshot Automation Script (Enhanced)

**Create a helper script to streamline the process:**

**Script:** `tools/take-screenshot.ps1`

```powershell
# PlasmaDX Screenshot Helper
# Usage: ./take-screenshot.ps1 [description]

param(
    [string]$Description = "screenshot"
)

# Screenshot directory
$BaseDir = "C:\Users\dilli\Pictures\PlasmaDX-Screenshots"
if (-not (Test-Path $BaseDir)) {
    New-Item -ItemType Directory -Path $BaseDir | Out-Null
}

# Generate filename with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Filename = "${Description}_${Timestamp}.png"
$FullPath = Join-Path $BaseDir $Filename

# Take screenshot using Windows API
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Capture entire screen
$Screen = [System.Windows.Forms.Screen]::PrimaryScreen
$Bounds = $Screen.Bounds
$Bitmap = New-Object System.Drawing.Bitmap $Bounds.Width, $Bounds.Height
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
$Graphics.CopyFromScreen($Bounds.Location, [System.Drawing.Point]::Empty, $Bounds.Size)

# Save
$Bitmap.Save($FullPath, [System.Drawing.Imaging.ImageFormat]::Png)
$Graphics.Dispose()
$Bitmap.Dispose()

# Output WSL path (for easy copy-paste to Claude Code)
$WSLPath = "/mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/$Filename"
Write-Host "Screenshot saved!"
Write-Host "WSL Path: $WSLPath"
Write-Host ""
Write-Host "Paste this path into Claude Code to share the screenshot."
```

**Usage from WSL:**
```bash
# Take screenshot
powershell.exe -File tools/take-screenshot.ps1 "rtxdi-sphere-preset"

# Output:
# Screenshot saved!
# WSL Path: /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi-sphere-preset_20251019_192345.png
#
# Paste this path into Claude Code to share the screenshot.

# Then paste the path in Claude Code chat
```

**Pros:**
- ✅ Automated screenshot capture
- ✅ Auto-generates WSL path for easy copy-paste
- ✅ Timestamped filenames
- ✅ Organized in dedicated folder
- ✅ Can add description to filename

**Cons:**
- ⚠️ Captures entire screen (not selective region)
- ⚠️ Requires PowerShell script creation
- ⚠️ Still requires manual path pasting

### Option 3: Windows Screenshot MCP Server (Custom Development)

**Create a custom MCP server for Windows screenshot capture:**

**Approach:** TypeScript MCP server using Windows APIs via node packages

**Architecture:**
```
Windows Node.js App (screenshot-win package)
    ↓
TypeScript MCP Server (exposes screenshot tool)
    ↓
Claude Code (uses tool to capture and view screenshots)
```

**Implementation Packages:**
- `screenshot-desktop` - Cross-platform screenshot capture (Node.js)
- `active-win` - Get active window info
- `@modelcontextprotocol/sdk` - MCP server SDK

**Estimated Time:** 2-3 hours to implement basic version

**Pros:**
- ✅ Fully integrated with Claude Code
- ✅ One-command screenshot capture
- ✅ Can capture specific windows (like Peekaboo)
- ✅ Automatic path handling

**Cons:**
- ⚠️ Requires development time
- ⚠️ Additional dependency (Node.js MCP server)
- ⚠️ May require Windows permissions setup

---

## Recommended Approach

### For Immediate Use (Right Now):

**Use Option 1** - Current workflow already works perfectly!

1. Take screenshot: **Win + Shift + S**
2. Save to: `C:\Users\dilli\Pictures\Screenshots\`
3. Provide WSL path to Claude Code: `/mnt/c/Users/dilli/Pictures/Screenshots/filename.png`
4. Claude Code displays it instantly

**Pro Tip:** Keep Screenshots folder open in File Explorer for quick access to filenames.

### For Streamlined Workflow (15 minutes):

**Implement Option 2** - PowerShell automation script

This gives you:
- One-command screenshot capture from WSL
- Auto-generated WSL paths for easy copy-paste
- Organized screenshot folder

**Quick Setup:**
```bash
# 1. Create tools directory
mkdir -p tools

# 2. Create PowerShell script (see script above)
# 3. Make it easily accessible via bash alias

# Add to ~/.bashrc:
alias screenshot='powershell.exe -File /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/tools/take-screenshot.ps1'

# Usage:
screenshot "rtxdi-preset-comparison"
# Outputs WSL path → Copy-paste to Claude Code
```

### For Full Integration (Future):

**Develop Option 3** - Custom MCP server (2-3 hours)

Only pursue this if the streamlined workflow (Option 2) isn't sufficient.

---

## Comparison to Peekaboo

| Feature | Peekaboo (macOS) | Option 1 (Current) | Option 2 (Script) | Option 3 (MCP) |
|---------|------------------|-------------------|-------------------|----------------|
| Screenshot capture | ✅ Automated | ⚠️ Manual (Win+Shift+S) | ✅ Automated | ✅ Automated |
| Window selection | ✅ Specific apps | ❌ Manual region | ❌ Full screen | ✅ Possible |
| AI image viewing | ✅ Built-in | ✅ Claude Code Read | ✅ Claude Code Read | ✅ Claude Code Read |
| WSL compatibility | ❌ macOS only | ✅ Works now | ✅ Works | ✅ Works |
| Setup time | 5 min (brew) | 0 min (ready!) | 15 min (script) | 2-3 hours (dev) |
| Integration level | Tight (MCP) | Loose (manual) | Medium (scripted) | Tight (MCP) |

---

## Implementation - Option 2 Script (Recommended)

Let me create the PowerShell script for you:

**File:** `tools/take-screenshot.ps1`

```powershell
<#
.SYNOPSIS
    PlasmaDX Screenshot Helper for Windows/WSL

.DESCRIPTION
    Captures full-screen screenshot and outputs WSL-compatible path for Claude Code.
    Screenshots saved to: C:\Users\dilli\Pictures\PlasmaDX-Screenshots\

.PARAMETER Description
    Optional description to include in filename (default: "screenshot")

.EXAMPLE
    powershell.exe -File tools/take-screenshot.ps1 "rtxdi-sphere-preset"
    # Outputs: /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi-sphere-preset_20251019_192345.png

.EXAMPLE
    powershell.exe -File tools/take-screenshot.ps1
    # Outputs: /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/screenshot_20251019_192345.png
#>

param(
    [Parameter(Position=0)]
    [string]$Description = "screenshot"
)

# Configuration
$BaseDir = "C:\Users\dilli\Pictures\PlasmaDX-Screenshots"
$Username = "dilli"

# Ensure directory exists
if (-not (Test-Path $BaseDir)) {
    New-Item -ItemType Directory -Path $BaseDir | Out-Null
    Write-Host "Created directory: $BaseDir"
}

# Generate filename with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDescription = $Description -replace '[\\/:*?"<>|]', '_'  # Remove invalid chars
$Filename = "${SafeDescription}_${Timestamp}.png"
$FullPath = Join-Path $BaseDir $Filename

# Load required assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

try {
    # Capture entire primary screen
    $Screen = [System.Windows.Forms.Screen]::PrimaryScreen
    $Bounds = $Screen.Bounds
    $Bitmap = New-Object System.Drawing.Bitmap $Bounds.Width, $Bounds.Height
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)

    # Copy screen to bitmap
    $Graphics.CopyFromScreen($Bounds.Location, [System.Drawing.Point]::Empty, $Bounds.Size)

    # Save to file
    $Bitmap.Save($FullPath, [System.Drawing.Imaging.ImageFormat]::Png)

    # Cleanup
    $Graphics.Dispose()
    $Bitmap.Dispose()

    # Output results
    $WSLPath = "/mnt/c/Users/$Username/Pictures/PlasmaDX-Screenshots/$Filename"

    Write-Host ""
    Write-Host "✅ Screenshot saved successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Windows Path:" -ForegroundColor Cyan
    Write-Host "  $FullPath"
    Write-Host ""
    Write-Host "WSL Path (for Claude Code):" -ForegroundColor Yellow
    Write-Host "  $WSLPath"
    Write-Host ""
    Write-Host "📋 Copy the WSL path above and paste into Claude Code chat."
    Write-Host ""

} catch {
    Write-Host "❌ Error capturing screenshot:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}
```

**Bash Wrapper:** `tools/screenshot.sh`

```bash
#!/bin/bash
# Bash wrapper for PowerShell screenshot tool

DESCRIPTION="${1:-screenshot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERSHELL_SCRIPT="$SCRIPT_DIR/take-screenshot.ps1"

# Convert WSL path to Windows path for PowerShell
WINDOWS_SCRIPT_PATH=$(wslpath -w "$POWERSHELL_SCRIPT")

# Execute PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "$WINDOWS_SCRIPT_PATH" "$DESCRIPTION"
```

**Installation:**

```bash
# 1. Create tools directory
mkdir -p tools

# 2. Create the scripts (use Write tool - I'll do this below)

# 3. Make bash wrapper executable
chmod +x tools/screenshot.sh

# 4. Add alias to ~/.bashrc (optional)
echo 'alias screenshot="$HOME/AndroidStudioProjects/PlasmaDX-Clean/tools/screenshot.sh"' >> ~/.bashrc
source ~/.bashrc

# 5. Test it!
screenshot "test-shot"
# Should output WSL path ready to copy-paste
```

---

## Usage Example

### Before (Manual):

```
1. Win + Shift + S
2. Capture region
3. Save As... → C:\Users\dilli\Pictures\Screenshots\issue.png
4. Open File Explorer → Get filename
5. Type path manually: /mnt/c/Users/dilli/Pictures/Screenshots/issue.png
6. Paste into Claude Code
```

### After (Automated):

```bash
# From WSL terminal:
screenshot "rtxdi-sphere-preset"

# Output:
# ✅ Screenshot saved successfully!
#
# WSL Path (for Claude Code):
#   /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi-sphere-preset_20251019_192345.png
#
# 📋 Copy the WSL path above and paste into Claude Code chat.

# Copy path, paste into Claude Code → Done!
```

**Time Saved:** 30 seconds → 5 seconds per screenshot

---

## Alternative: Windows Snipping Tool with Auto-Save

**Configure Windows Snipping Tool for auto-save:**

1. Open Snipping Tool settings
2. Set default save location: `C:\Users\dilli\Pictures\Screenshots\`
3. Enable auto-save (if available in your Windows version)
4. Screenshots now auto-save with timestamp filenames

**Pro:** No script needed, native Windows tool
**Con:** Still need to manually get filename for WSL path

---

## Recommendation

**For You Specifically:**

1. **Immediate (0 setup):** Keep using current workflow (Win + Shift + S → provide path)
   - Already works perfectly
   - Zero friction once you're used to it

2. **Next 15 minutes:** Implement Option 2 (PowerShell script)
   - Streamlines the process significantly
   - Auto-generates WSL paths
   - Organized screenshot folder

3. **Future (if needed):** Consider custom MCP server (Option 3)
   - Only if Option 2 doesn't meet needs
   - Full Peekaboo-like integration
   - 2-3 hour investment

**My Suggestion:** Start with what you have (it works!), implement Option 2 script when you have 15 minutes, evaluate if you need more automation after using it for a while.

---

## Next Steps

Would you like me to:

1. ✅ **Create the PowerShell script and bash wrapper** (15 minutes)
2. ⏳ **Design a custom Windows screenshot MCP server** (2-3 hours, future work)
3. ⏳ **Document the current manual workflow with best practices** (5 minutes)

**Choose option 1** for immediate improvement with minimal setup.

---

**Status:** Ready to implement! Current workflow already functional, enhancement scripts ready to create.
