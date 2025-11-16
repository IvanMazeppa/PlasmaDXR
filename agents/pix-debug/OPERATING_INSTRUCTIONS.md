# Autonomous Debugging Agent - Operating Instructions

**Status:** ✅ PRODUCTION READY
**Last Tested:** 2025-11-01 00:15
**All Components:** Verified Working

---

## Quick Reference

### Single Test (Fast Check)
```
In Claude Code:
  "Use diagnose_gpu_hang to test at 2045 particles"
```

### Threshold Testing (Find Crash Point)
```
In Claude Code:
  "Use diagnose_gpu_hang to find the crash threshold at 2045 particles"
```

### Manual Verification
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug
./PlasmaDX-Clean.exe --particles 100 --restir
# Press Ctrl+C or Alt+F4 to exit
# Check logs/PlasmaDX-Clean_*.log for "Lighting system: Volumetric ReSTIR"
```

---

## Verified Components ✅

### 1. Command-Line Flag Implementation
**Status:** ✅ WORKING

**What it does:**
- `--restir` flag added to Application.cpp (lines 111-113)
- Enables Volumetric ReSTIR from startup
- Logged as "Lighting system: Volumetric ReSTIR (autonomous testing mode)"

**Verification:**
```bash
./PlasmaDX-Clean.exe --particles 100 --restir
# Log shows: [INFO] Lighting system: Volumetric ReSTIR (autonomous testing mode)
```

### 2. Working Directory Fix
**Status:** ✅ WORKING

**What it does:**
- Launches PlasmaDX from `build/bin/Debug/` directory
- Ensures shaders load from correct location
- Logs captured from `build/bin/Debug/logs/`

**Verification:**
- No shader loading errors (`Failed to load *.dxil`)
- Logs show successful initialization
- Latest log timestamp matches test time

### 3. Log Capture
**Status:** ✅ WORKING

**What it does:**
- Captures first 50 lines (initialization, --restir verification)
- Captures last 100 lines (hang analysis, frame data)
- Avoids duplicates for short logs

**Verification:**
- Init logs show "Lighting system: Volumetric ReSTIR"
- Final logs show frame numbers and physics updates
- Both sections captured in single test

### 4. Process Termination
**Status:** ✅ WORKING

**What it does:**
- Kills PlasmaDX after timeout using `taskkill.exe /F`
- Handles Windows process tree properly
- No orphaned processes after test

**Verification:**
```bash
tasklist.exe | grep -i plasma
# Should return empty after test completes
```

### 5. Status Detection
**Status:** ✅ WORKING

**What it does:**
- `timeout` - Process exceeded time limit (normal operation)
- `hung` - Process actually hung (infinite loop, TDR)
- `completed` - Process exited cleanly
- `crashed` - Process returned non-zero exit code

**Verification:**
- With PopulateVolumeMip2 disabled: Status = "timeout" (expected)
- With PopulateVolumeMip2 enabled at 2045: Status = "hung" (crash)

---

## Tool Parameters Reference

### `particle_count` (integer, default: 2045)
Number of particles to spawn.

**Example:**
```
diagnose_gpu_hang at 100 particles
diagnose_gpu_hang at 2045 particles
diagnose_gpu_hang at 10000 particles
```

### `timeout_seconds` (integer, default: 10)
How long to wait before considering test complete/hung.

**Guidelines:**
- Normal runs: 5-10 seconds sufficient
- Large particle counts: 15-20 seconds
- Slow systems: Increase as needed

**Example:**
```
diagnose_gpu_hang at 2045 particles with 15 second timeout
```

### `test_threshold` (boolean, default: false)
If true, tests 5 particle counts to find exact crash boundary.

**Tests performed:**
- `particle_count - 5`
- `particle_count - 1`
- `particle_count` (exact)
- `particle_count + 3`
- `particle_count + 5`

**Example:**
```
diagnose_gpu_hang at 2045 particles with threshold testing
```

**Output:**
```json
{
  "analysis": {
    "crash_threshold": "2044 → 2045",
    "success_particle_counts": [2040, 2044],
    "hung_particle_counts": [2045, 2048, 2050]
  }
}
```

### `capture_logs` (boolean, default: true)
If true, captures logs from latest run.

**Captured:**
- First 50 lines (initialization, --restir flag)
- Last 100 lines (hang point, frame data)

**Example:**
```
diagnose_gpu_hang at 2045 particles without log capture
```

---

## Usage Scenarios

### Scenario 1: Verify Tool is Working
**Before debugging PopulateVolumeMip2:**

```
In Claude Code:
  "Use diagnose_gpu_hang to test at 100 particles with 5 second timeout"
```

**Expected output:**
- Status: "timeout" (normal - program ran successfully)
- Logs contain: "Lighting system: Volumetric ReSTIR (autonomous testing mode)"
- Logs contain: "Volumetric ReSTIR System initialized successfully"
- Frame count: 300-600+ frames (30-60 FPS)

### Scenario 2: Find PopulateVolumeMip2 Crash Threshold
**After re-enabling PopulateVolumeMip2:**

```
In Claude Code:
  "Use diagnose_gpu_hang to find the crash threshold at 2045 particles"
```

**Expected output:**
- 5 tests run (2040, 2044, 2045, 2048, 2050)
- Status: Some "timeout", some "hung"
- Exact boundary identified (e.g., "2044 works, 2045 crashes")
- Logs show PopulateVolumeMip2 dispatch before hang
- Recommendations include PIX capture suggestions

### Scenario 3: Single Targeted Test
**Testing specific particle count:**

```
In Claude Code:
  "Use diagnose_gpu_hang to test at 2045 particles"
```

**Expected output:**
- Single test at exactly 2045 particles
- Status: "hung" if PopulateVolumeMip2 crashes
- Logs captured from crash point
- Recommendations for next debugging steps

### Scenario 4: Extended Timeout for Slow Systems
**If default 10s timeout is too short:**

```
In Claude Code:
  "Use diagnose_gpu_hang to test at 10000 particles with 20 second timeout"
```

### Scenario 5: Quick Multi-Particle Sweep
**Testing multiple counts manually:**

```
In Claude Code:
  "Use diagnose_gpu_hang to test at 1000 particles"
  (wait for result)
  "Use diagnose_gpu_hang to test at 2000 particles"
  (wait for result)
  "Use diagnose_gpu_hang to test at 3000 particles"
```

---

## Interpreting Results

### Status: "timeout"
**Meaning:** Program ran successfully but exceeded timeout.

**Interpretation:**
- ✅ No crash detected
- ✅ Program responding normally
- ℹ️ Check frame count in logs to confirm progress

**Next steps:**
- Increase particle count or decrease timeout
- This is **expected behavior** with PopulateVolumeMip2 disabled

### Status: "hung"
**Meaning:** Program actually hung (GPU TDR, infinite loop).

**Interpretation:**
- ❌ Crash detected
- ⚠️ Check logs for last operation before hang
- 🔍 Look for PopulateVolumeMip2 dispatch

**Next steps:**
1. Review captured logs for patterns
2. Create PIX capture at crash particle count
3. Use tool recommendations for file:line investigation

### Status: "completed"
**Meaning:** Program exited cleanly (rare in testing).

**Interpretation:**
- ✅ No crash
- ℹ️ Program may have self-terminated
- ℹ️ Check exit code in results

### Status: "crashed"
**Meaning:** Program returned non-zero exit code.

**Interpretation:**
- ❌ Hard crash detected
- ⚠️ Check logs for error messages
- 🔍 May indicate resource exhaustion

---

## Log Analysis Patterns

### Pattern 1: Successful --restir Activation
**Look for:**
```
[INFO] Lighting system: Volumetric ReSTIR (autonomous testing mode)
[INFO] Initializing Volumetric ReSTIR System (Phase 1)...
[INFO] Volumetric ReSTIR System initialized successfully
```

**Meaning:** Tool working correctly, ReSTIR enabled.

### Pattern 2: PopulateVolumeMip2 Dispatch
**Look for:**
```
[INFO] PopulateVolumeMip2: Dispatching X thread groups
[INFO] VolumetricReSTIR: About to ExecuteCommandList
```

**Meaning:** Shader dispatched just before hang - likely crash point.

### Pattern 3: Normal Frame Progress
**Look for:**
```
[INFO] Physics update 2340 (totalTime=19.499788, ...)
[INFO] RT Lighting computed with dynamic emission (frame 2340)
[INFO] Physics update 2400 (totalTime=19.999780, ...)
```

**Meaning:** Program running normally, frames advancing.

### Pattern 4: Shader Loading
**Look for:**
```
[INFO] Found particle_physics.dxil at: shaders/particles/...
[INFO] Loaded Gaussian shader: 24600 bytes
[INFO] Loaded motion vector shader: 5392 bytes
```

**Meaning:** Shaders loaded successfully (working directory correct).

### Pattern 5: Shader Errors (If Any)
**Look for:**
```
[ERROR] Failed to load compute_motion_vectors.dxil
[ERROR] Failed to initialize Gaussian renderer
```

**Meaning:** Working directory issue - report to maintainer.

---

## Troubleshooting

### Issue: --restir flag not found in logs

**Symptoms:**
- Tool reports "❌ FAILED - --restir flag not detected"
- Logs don't show "Volumetric ReSTIR (autonomous testing mode)"

**Diagnosis:**
```bash
# Check if rebuild applied
ls -lt /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/PlasmaDX-Clean.exe
# Should show recent timestamp (after 23:02 on Oct 31)
```

**Solution:**
1. Rebuild PlasmaDX: `MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64`
2. Verify flag in code: Check Application.cpp line 111-113 for `--restir` case
3. Test manually: `./PlasmaDX-Clean.exe --particles 100 --restir`

### Issue: Shader loading errors

**Symptoms:**
```
[ERROR] Failed to load *.dxil
[ERROR] Failed to initialize Gaussian renderer
```

**Diagnosis:**
- Working directory issue
- Shaders not in expected location

**Solution:**
```bash
# Verify shaders exist
ls /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/shaders/particles/*.dxil

# Should show:
# - particle_physics.dxil
# - particle_gaussian_raytrace.dxil
# - compute_motion_vectors.dxil
```

If missing, rebuild with shader compilation enabled.

### Issue: Process not terminating

**Symptoms:**
- Multiple PlasmaDX processes accumulate
- `tasklist.exe` shows orphaned processes

**Diagnosis:**
```bash
tasklist.exe | grep -i plasma
# Should return empty after test
```

**Solution:**
```bash
# Manual cleanup
taskkill.exe /F /IM PlasmaDX-Clean.exe

# Then reconnect MCP
/mcp reconnect pix-debug
```

If problem persists, report to maintainer.

### Issue: Empty/wrong logs captured

**Symptoms:**
- Logs don't match test time
- No initialization logs
- Very old timestamps

**Diagnosis:**
```bash
# Check latest log
ls -lt /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/logs/*.log | head -1

# Compare timestamp to test time
```

**Solution:**
- Verify log_dir calculation in mcp_server.py (should use exe_dir/logs)
- Delete old logs if confused: `rm build/bin/Debug/logs/*.log`
- Reconnect MCP: `/mcp reconnect pix-debug`

### Issue: Tool not found in Claude Code

**Symptoms:**
```
Error: name 'diagnose_gpu_hang' is not defined
```

**Solution:**
1. Check MCP connection: `/mcp list` (should show pix-debug)
2. Reconnect: `/mcp reconnect pix-debug`
3. Verify tool registration:
   ```bash
   cd /mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4
   grep -A 5 '"diagnose_gpu_hang"' mcp_server.py
   ```

---

## Integration with Other Tools

### Workflow 1: Threshold → PIX → Buffer Analysis

**Step 1: Find threshold**
```
diagnose_gpu_hang with threshold testing at 2045 particles
→ Output: "Threshold 2044 → 2045"
```

**Step 2: Create PIX capture**
```
pix_capture with 2 frames
→ Captures frame 1 (works) and frame 2 (hangs)
```

**Step 3: Analyze buffers**
```
analyze_particle_buffers at PIX/buffer_dumps/g_particles.bin
→ Validates particle data at crash point
```

### Workflow 2: Visual Artifact Diagnosis

**Step 1: Describe symptom**
```
diagnose_visual_artifact with symptom "black dots at far distances"
→ Provides hypothesis and checks
```

**Step 2: Verify with threshold test**
```
diagnose_gpu_hang at suspected particle count
→ Confirms if particle count related
```

### Workflow 3: Performance Regression Testing

**Test before changes:**
```
diagnose_gpu_hang at 10000 particles
→ Note: Status, runtime, frame count
```

**Make code changes, rebuild**

**Test after changes:**
```
diagnose_gpu_hang at 10000 particles
→ Compare: Status, runtime, frame count
```

**Analyze difference:**
- Faster: ✅ Optimization worked
- Slower: ⚠️ Performance regression
- Crashes: ❌ Introduced bug

---

## Expected Behavior Reference

### With PopulateVolumeMip2 Disabled (Current)

**Single test at any particle count:**
- Status: "timeout"
- Runtime: ~10 seconds (timeout value)
- Frame count: 600+ frames (60 FPS × 10s)
- Hang detected: False
- Logs: Normal frame progression

**Interpretation:** ✅ Working as expected (no crash)

### With PopulateVolumeMip2 Enabled (Target)

**Single test at 2044 particles:**
- Status: "timeout" or "completed"
- Runtime: <10 seconds
- Frame count: Variable
- Hang detected: False
- Logs: Normal operation

**Single test at 2045 particles:**
- Status: "hung"
- Runtime: ~10 seconds (timeout before kill)
- Frame count: Low (1-2 frames)
- Hang detected: True
- Logs: PopulateVolumeMip2 dispatch, then freeze

**Threshold test at 2045:**
- 2040: "timeout" ✅
- 2044: "timeout" ✅
- 2045: "hung" ❌
- 2048: "hung" ❌
- 2050: "hung" ❌
- Analysis: "Threshold 2044 → 2045"

---

## Performance Metrics

**Single Test:**
- Launch time: ~2 seconds (D3D12 init)
- Run time: 5-10 seconds (configurable timeout)
- Log analysis: <1 second
- Total: ~12-15 seconds per test

**Threshold Test (5 counts):**
- 2 successes × 12s = 24 seconds
- 3 hangs × 12s = 36 seconds
- Total: ~60 seconds for full threshold sweep

**Manual Testing (for comparison):**
- Launch, test, observe, kill, repeat × 5
- Total: 5-10 minutes
- **Autonomous speedup: 5-10×**

---

## Advanced Usage

### Custom Timeout per Test
```
diagnose_gpu_hang at 2045 particles with 15 second timeout
```

### Disable Log Capture (Faster)
```
diagnose_gpu_hang at 2045 particles without log capture
```

### Multiple Sequential Tests
```
In Claude Code:
  "Run diagnose_gpu_hang at these particle counts: 1000, 2000, 3000, 4000, 5000"
```

The agent will execute 5 sequential tests and summarize results.

### Combined with Buffer Dumps
After identifying crash point, manually run:
```bash
./PlasmaDX-Clean.exe --particles 2045 --restir --dump-buffers 2
```

Then analyze with:
```
analyze_particle_buffers at PIX/buffer_dumps/g_particles.bin
```

---

## Maintenance

### When to Rebuild PlasmaDX
- After modifying Application.cpp (--restir flag)
- After changing PopulateVolumeMip2 shader
- After any core rendering system changes

### When to Reconnect MCP
- After modifying mcp_server.py
- After updating tool logic
- After changing tool parameters
- If tool stops responding

### Verifying Tool Health
```
In Claude Code:
  "Use diagnose_gpu_hang to test at 100 particles with 5 second timeout"
```

**Healthy output:**
- Status: "timeout"
- --restir flag detected in logs
- Process terminated cleanly
- No shader errors

---

## Contact & Support

**Tool Status:** Production Ready
**Last Verified:** 2025-11-01 00:15
**Documentation:** See DEPLOYMENT_SUMMARY_FINAL.md for technical details

**Quick Checks:**
- Is PlasmaDX rebuilt? Check exe timestamp
- Is MCP connected? Run `/mcp list`
- Are shaders present? Check `build/bin/Debug/shaders/`
- Are logs recent? Check `build/bin/Debug/logs/`

**Common Questions:**
- Q: Why "timeout" instead of "hung"?
  - A: PopulateVolumeMip2 disabled - program runs normally

- Q: How do I know if --restir worked?
  - A: Check logs for "Lighting system: Volumetric ReSTIR (autonomous testing mode)"

- Q: Can I test multiple particle counts at once?
  - A: Yes, use threshold testing or request sequential tests

- Q: What if I need longer timeout?
  - A: Use "with X second timeout" in your request

---

## Success Criteria Checklist

Before using for critical debugging, verify:

- [ ] Manual test: `./PlasmaDX-Clean.exe --particles 100 --restir` works
- [ ] Autonomous test: Status = "timeout", --restir flag detected
- [ ] Process cleanup: `tasklist.exe | grep plasma` returns empty after test
- [ ] Log capture: Both init and final logs present
- [ ] Shader loading: No `Failed to load *.dxil` errors
- [ ] MCP connection: `diagnose_gpu_hang` available in Claude Code

If all checked, tool is ready for production debugging! ✅

---

**End of Operating Instructions**
**Ready to debug Volumetric ReSTIR! 🚀**
