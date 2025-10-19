# PlasmaDX Tools

Utility scripts for development workflow automation.

---

## Screenshot Helper

**Purpose:** Streamline screenshot sharing with Claude Code for visual debugging

### Quick Start

```bash
# From project root:
./tools/screenshot.sh "description-of-what-im-showing"

# Example:
./tools/screenshot.sh "rtxdi-sphere-preset-applied"

# Output:
# ✅ Screenshot saved successfully!
#
# WSL Path (for Claude Code):
#   /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi-sphere-preset-applied_20251019_193045.png
#
# 📋 Copy the WSL path above and paste into Claude Code chat.
```

### Installation (Optional Alias)

Add to `~/.bashrc` for convenient access from anywhere:

```bash
# Add this line to ~/.bashrc:
alias screenshot="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/tools/screenshot.sh"

# Reload config:
source ~/.bashrc

# Now use from anywhere:
screenshot "my-awesome-feature"
```

### Files

- **take-screenshot.ps1** - PowerShell script (does actual screen capture via Windows API)
- **screenshot.sh** - Bash wrapper (calls PowerShell script with proper paths)
- **README.md** - This file

### How It Works

1. Bash script converts WSL path to Windows path
2. Calls PowerShell script via `powershell.exe`
3. PowerShell uses Windows Forms API to capture screen
4. Saves to: `C:\Users\dilli\Pictures\PlasmaDX-Screenshots\`
5. Outputs WSL-compatible path for easy copy-paste

### Requirements

- Windows 10/11 (for PowerShell and System.Windows.Forms)
- WSL (for bash wrapper)
- No additional dependencies

### Limitations

- Captures **entire primary screen** (not selective region)
- For selective capture, use **Win + Shift + S** (Windows Snipping Tool) instead
- PowerShell execution policy must allow script execution (handled via `-ExecutionPolicy Bypass`)

### Troubleshooting

**Error: "Screenshot cannot be created"**
- Check if `C:\Users\dilli\Pictures\` exists
- Ensure PowerShell has permissions to create files

**Error: "wslpath: command not found"**
- You're not in WSL environment
- Run directly: `powershell.exe -File tools/take-screenshot.ps1 "description"`

**Screenshots folder not found**
- Script auto-creates `C:\Users\dilli\Pictures\PlasmaDX-Screenshots\` on first run
- Check Windows username matches "dilli" (hardcoded in script)

---

## Future Tools

Potential additions to this directory:

- **build-all-configs.sh** - Build Debug, DebugPIX, Release in sequence
- **run-stress-tests.sh** - Automated particle count/light count scaling tests
- **pix-capture-analyze.sh** - Automated PIX capture + analysis pipeline
- **git-branch-snapshot.sh** - Save current build state before experimental changes

---

**Last Updated:** 2025-10-19
