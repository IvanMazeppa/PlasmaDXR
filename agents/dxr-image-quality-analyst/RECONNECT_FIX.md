# MCP Server Reconnection Fix

**Issue:** rtxdi-quality-analyzer server won't reconnect after adding ML comparison tool

**Root cause:** MCP might have cached the old server definition or the connection is stale

---

## Solution Steps

### Step 1: Remove existing MCP server registration

```bash
claude mcp remove rtxdi-quality-analyzer
```

### Step 2: Re-add with correct absolute path

```bash
claude mcp add --transport stdio rtxdi-quality-analyzer \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  --env PIX_PATH="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64" \
  -- /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/run_server.sh
```

**Important:** Use the full absolute path to `run_server.sh` (not relative)

### Step 3: Verify server is listed

```bash
claude mcp list
```

You should see:
```
rtxdi-quality-analyzer (connected) - 3 tools
```

### Step 4: Test in Claude Code

In Claude Code session, type:
```
What tools do you have available from rtxdi-quality-analyzer?
```

Expected response should list **3 tools**:
1. `compare_performance`
2. `analyze_pix_capture`
3. `compare_screenshots_ml` ✨ NEW

---

## Alternative: Manual Server Test

If the above doesn't work, test the server directly:

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer

# Activate venv
source venv/bin/activate

# Run server (should wait for stdio input)
python -m src.agent
```

**Expected:** Server starts and waits (no errors)
**Press Ctrl+C to stop**

If you see errors here, that's the root cause.

---

## Troubleshooting

### Error: "command not found: claude"

Claude CLI might not be in your PATH. Try:
```bash
# Check if claude command exists
which claude

# If not found, use full path (adjust to your installation)
~/.local/bin/claude mcp add ...
```

### Error: "Server not responding"

1. Check if Python venv is properly activated:
   ```bash
   cd agents/rtxdi-quality-analyzer
   source venv/bin/activate
   which python  # Should show venv path
   ```

2. Test imports manually:
   ```bash
   python -c "from src.tools.ml_visual_comparison import compare_screenshots_ml; print('OK')"
   ```

3. Check for port conflicts:
   ```bash
   # If using stdio transport, there should be no port conflicts
   # But check if another instance is running
   ps aux | grep "src.agent"
   ```

### Error: "Import failed"

Reinstall dependencies:
```bash
cd agents/rtxdi-quality-analyzer
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## Verification Checklist

After following the steps above, verify:

- [ ] `claude mcp list` shows rtxdi-quality-analyzer as "connected"
- [ ] Claude Code shows 3 tools when queried
- [ ] `compare_screenshots_ml` appears in tool list
- [ ] No errors in Claude Code console

If all checks pass, the server is working correctly!

---

## Next Steps (After Reconnection)

Once reconnected, you can test the ML comparison:

```
"Compare these two screenshots using ML:
  before: /mnt/c/Users/dilli/Pictures/Screenshots/Screenshot 2025-10-20 194805.png
  after: /mnt/c/Users/dilli/Pictures/Screenshots/Screenshot 2025-10-21 185625.png"
```

Expected output:
- Overall similarity score (0-100%)
- LPIPS perceptual similarity (~92% human-aligned)
- Traditional metrics (SSIM, MSE, PSNR)
- Difference heatmap path
- Human-readable interpretation

---

**Last Updated:** 2025-10-23
**Status:** Server tested and working, awaiting reconnection
