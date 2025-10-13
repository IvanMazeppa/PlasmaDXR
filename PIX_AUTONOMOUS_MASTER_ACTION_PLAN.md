# PIX Autonomous Capture - Master Action Plan
**Date:** 2025-10-13
**Status:** Ready for Implementation
**Goal:** Achieve <60s fully autonomous capture → analyze → report workflow

---

## Executive Summary

**Current State:**
- ✅ Infrastructure 95% complete (dual builds, config system, PIX integration)
- ✅ Python agent exists with analysis capabilities
- ✅ Manual PIX GUI workflow functional
- ❌ Programmatic `PIXBeginCapture` API returns E_FAIL (0x80004005)

**Solution:**
Abandon problematic programmatic API, adopt proven `pixtool launch + take-capture` workflow used in successful manual captures.

**Timeline:** 4-6 hours implementation, 1-2 hours testing

---

## Priority Implementation Plan

### PRIORITY 1: Stable Capture Path (2-3 hours)
**Source:** Remediation Plan §13-33, Architecture §17-21

**Goal:** Replace programmatic API with reliable pixtool workflow

**Implementation:**
1. Update `autonomous_capture_and_analyze()` in pix_autonomous_agent.py
2. Split into two commands:
   - Launch: `pixtool.exe launch app.exe --working-directory=<dir> --command-line=<args>`
   - After delay: `pixtool.exe take-capture --frames=1 --open save-capture <out.wpix>`

**Code Changes:**
```python
# In PIXAutonomousAgent.autonomous_capture_and_analyze_stable()
def autonomous_capture_and_analyze_stable(self, config_preset: str = "far", warmup_seconds: float = 2.0):
    """Stable capture using pixtool launch + take-capture (no programmatic API)"""

    # Step 1: Build launch command
    exe_win = self.pixtool.wsl_to_windows_path(str(self.app_exe))
    working_dir = self.pixtool.wsl_to_windows_path(str(self.project_root))
    config_file = f"config_pix_{config_preset}.json"

    launch_args = [
        "launch",
        exe_win,
        f"--working-directory={working_dir}",
        f"--command-line=--config {config_file}"
    ]

    # Step 2: Launch app (non-blocking)
    stdout, stderr, code = self.pixtool.run_command(launch_args, timeout=5)

    # Step 3: Wait for warmup
    time.sleep(warmup_seconds)

    # Step 4: Trigger capture
    capture_path = self.captures_dir / f"auto_{config_preset}_{timestamp}.wpix"
    capture_win = self.pixtool.wsl_to_windows_path(str(capture_path))

    capture_args = [
        "take-capture",
        "--frames=1",
        "--open",
        "save-capture",
        capture_win
    ]

    stdout, stderr, code = self.pixtool.run_command(capture_args, timeout=30)

    # Step 5: Verify
    if not capture_path.exists() or capture_path.stat().st_size < 100_000:
        # Retry with longer delay
        time.sleep(warmup_seconds * 2)
        stdout, stderr, code = self.pixtool.run_command(capture_args, timeout=30)

    # Step 6: Analyze...
```

**Acceptance Criteria:**
- [ ] Single capture completes in <30s
- [ ] .wpix file >100KB created
- [ ] Works from WSL and Windows
- [ ] No E_FAIL errors

---

### PRIORITY 2: CLI Presets & Batch Mode (1-2 hours)
**Source:** Robustness §13-14, Remediation §39-42

**Goal:** Enable multi-scenario testing with one command

**Implementation:**
```python
# Add to main()
parser.add_argument('--preset', type=str, choices=['far', 'close', 'veryclose', 'inside'],
                   help='Use preset config file')
parser.add_argument('--scenarios', type=str,
                   help='Comma-separated list: far,close,inside')

# Usage
if args.scenarios:
    scenarios = args.scenarios.split(',')
    analyses = []
    for scenario in scenarios:
        print(f"\n[Agent] ===== SCENARIO: {scenario.upper()} =====")
        analysis = agent.autonomous_capture_and_analyze_stable(config_preset=scenario)
        if analysis:
            analyses.append(analysis)

    # Generate aggregate report
    aggregate_path = agent.reports_dir / f"BATCH_{timestamp}_summary.md"
    generate_batch_summary(analyses, aggregate_path)
```

**Acceptance Criteria:**
- [ ] `--preset far` uses config_pix_far.json
- [ ] `--scenarios far,close,inside` runs 3 captures
- [ ] Aggregate summary generated
- [ ] Each capture named with scenario prefix

---

### PRIORITY 3: Robust Waiting & Verification (30 min)
**Source:** Remediation §34-37, Robustness §8

**Goal:** Eliminate race conditions and failed captures

**Implementation:**
```python
def verify_capture(self, capture_path: Path, min_size: int = 100_000, retry_delay: float = 2.0) -> bool:
    """Verify capture with retry logic"""

    # Initial check
    if capture_path.exists() and capture_path.stat().st_size >= min_size:
        return True

    # Retry once with longer delay
    print(f"[Agent] Capture not ready, retrying with {retry_delay}s delay...")
    time.sleep(retry_delay)

    if capture_path.exists():
        size = capture_path.stat().st_size
        if size >= min_size:
            return True
        else:
            print(f"[Agent] ⚠ Capture too small: {size:,} bytes (min: {min_size:,})")
            return False

    print(f"[Agent] ✗ Capture file not created: {capture_path}")
    return False
```

**Acceptance Criteria:**
- [ ] Retries once if capture missing
- [ ] Checks file size >100KB
- [ ] Clear error messages with diagnostics

---

### PRIORITY 4: Defensive Options (30 min)
**Source:** Remediation §66-69

**Goal:** Make agent robust to different environments

**Implementation:**
```python
parser.add_argument('--shell', type=str, choices=['windows', 'wsl'], default='auto',
                   help='Force Windows Python or WSL execution')
parser.add_argument('--verify-dll', action='store_true',
                   help='Check WinPixGpuCapturer.dll exists before running')
parser.add_argument('--timeout', type=int, default=300,
                   help='Maximum seconds for capture (default: 300)')
parser.add_argument('--min-size', type=int, default=100_000,
                   help='Minimum capture size in bytes')

# Verification
if args.verify_dll:
    dll_path = project_root / "build/DebugPIX/WinPixGpuCapturer.dll"
    if not dll_path.exists():
        print(f"[Agent] ERROR: PIX DLL not found at {dll_path}")
        return 1
```

**Acceptance Criteria:**
- [ ] --verify-dll checks for DLL before running
- [ ] --timeout allows longer waits for complex scenes
- [ ] --min-size configurable for different capture types

---

## Secondary Implementation (Optional - Later)

### OPTIONAL 1: Readiness Polling (1 hour)
**Source:** Robustness §9, §38

**Goal:** Eliminate fixed sleep delays

**App Changes:**
```cpp
// In Application.cpp after warmup
if (m_frameCount == 60) {  // Or whenever ready
    LOG_INFO("[APP] READY_FOR_CAPTURE");
}
```

**Agent Changes:**
```python
def wait_for_readiness(self, log_file: Path, marker: str = "READY_FOR_CAPTURE", timeout: float = 10.0) -> bool:
    """Poll log file for readiness marker"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if marker in content:
                    return True
        time.sleep(0.1)
    return False
```

**Benefits:**
- Faster captures (no conservative sleep)
- Adapts to different hardware speeds
- More reliable timing

---

### OPTIONAL 2: COM Initialization Test (30 min)
**Source:** Remediation §48-56

**Goal:** Test if COM init fixes programmatic capture

**Implementation:**
```cpp
// In PIXCaptureHelper::InitializeWithConfig()
#include <combaseapi.h>

HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
if (FAILED(hr)) {
    LOG_WARN("[PIX] COM initialization failed: 0x{:08X}", hr);
}

// Then try PIXGpuCaptureNextFrames alternative API
PIXGpuCaptureNextFrames(L"D:\\capture.wpix", 1);
```

**Status:** Low priority, only if stable path has issues

---

## Modified Files Checklist

### Python Files
- [ ] **pix/pix_autonomous_agent.py**
  - [ ] Add `autonomous_capture_and_analyze_stable()` method
  - [ ] Add `verify_capture()` helper
  - [ ] Add `--preset`, `--scenarios`, `--shell`, `--verify-dll`, `--timeout` args
  - [ ] Add batch summary generation
  - [ ] Update `PIXToolWrapperWSL.run_command()` to support `take-capture`

### C++ Files (Optional - for readiness polling)
- [ ] **src/core/Application.cpp**
  - [ ] Add `LOG_INFO("[APP] READY_FOR_CAPTURE")` after warmup

### Config Files (Already Created)
- [x] config_pix_far.json (distance 800)
- [x] config_pix_close.json (distance 300)
- [x] config_pix_veryclose.json (distance 150)
- [x] config_pix_inside.json (distance 50)

---

## Testing Plan

### Phase 1: Single Scenario (15 min)
```bash
# Test far preset
python pix/pix_autonomous_agent.py --preset far

# Expected:
# - Launches app with config_pix_far.json
# - Waits 2 seconds
# - Captures frame
# - .wpix file created (>100KB)
# - Analysis report generated
# - Frame image extracted
```

**Success Criteria:**
- [ ] Capture file exists
- [ ] Size >100KB
- [ ] Analysis report generated
- [ ] Frame image extracted
- [ ] Total time <60s

### Phase 2: Multi-Scenario (30 min)
```bash
# Test batch mode
python pix/pix_autonomous_agent.py --scenarios far,close,veryclose,inside

# Expected:
# - 4 captures created
# - 4 analysis reports
# - 1 aggregate summary
# - Total time <4 minutes
```

**Success Criteria:**
- [ ] All 4 captures successful
- [ ] All 4 reports generated
- [ ] Aggregate summary shows comparison
- [ ] Each capture shows different camera distance

### Phase 3: Edge Cases (15 min)
```bash
# Test with longer timeout
python pix/pix_autonomous_agent.py --preset inside --timeout 600

# Test DLL verification
python pix/pix_autonomous_agent.py --verify-dll --preset far

# Test min size threshold
python pix/pix_autonomous_agent.py --preset far --min-size 500000
```

**Success Criteria:**
- [ ] Timeout allows slower machines
- [ ] DLL check catches missing prerequisites
- [ ] Min size validation works

---

## Troubleshooting Guide

### Issue: No capture file created
**Symptoms:** Agent completes but .wpix missing

**Diagnosis:**
1. Check app logs for "READY_FOR_CAPTURE" or rendering messages
2. Verify working directory is correct
3. Check if app exits prematurely

**Fix:**
- Increase warmup delay: `--timeout 10`
- Verify config file loads correctly
- Check app renders (run manually first)

### Issue: Capture file too small (<100KB)
**Symptoms:** .wpix created but <100KB

**Diagnosis:**
- Capture timing is off (captured during load/init, not rendering)
- App didn't complete frame render

**Fix:**
- Increase warmup: 3-5 seconds
- Use `take-capture --frames=2` instead of `--frames=1`
- Add readiness polling

### Issue: pixtool returns error
**Symptoms:** "take-capture failed" in stderr

**Diagnosis:**
- App not running when capture triggered
- PIX version mismatch
- Path issues (WSL/Windows conversion)

**Fix:**
- Verify app is running before capture
- Check pixtool version matches PIX install
- Test path conversion with `wslpath -w`

---

## Success Metrics

### Must Have
- [x] Single capture completes in <30s (currently ~60s due to programmatic API delay)
- [ ] Multi-scenario batch completes in <5 minutes for 4 captures
- [ ] Success rate >95% (19/20 captures successful)
- [ ] Works from both WSL and Windows natively

### Nice to Have
- [ ] Readiness polling reduces capture time to <15s
- [ ] Automatic retry on failure
- [ ] Detailed error diagnostics
- [ ] CI integration working

---

## Implementation Order

**Day 1 (4-6 hours):**
1. Implement stable capture path (Priority 1)
2. Add CLI presets (Priority 2)
3. Add verification logic (Priority 3)
4. Test single scenario

**Day 2 (2-3 hours):**
1. Add batch mode (Priority 2 cont.)
2. Add defensive options (Priority 4)
3. Test multi-scenario
4. Document edge cases

**Day 3 (Optional - 2 hours):**
1. Add readiness polling (Optional 1)
2. Test COM initialization (Optional 2)
3. Performance tuning
4. CI integration setup

---

## Document Cross-Reference

This plan consolidates recommendations from:

1. **PIX_AUTONOMOUS_CAPTURE_INFRASTRUCTURE.md** (Section 6.1-6.7)
   - Detailed technical analysis of PIXBeginCapture E_FAIL issue
   - 7 attempted solutions documented
   - Questions for advanced LLM analysis

2. **20251013-0416_pix_autonomous_capture_remediation_plan.md** (§13-33)
   - Priority plan: adopt pixtool launch + take-capture
   - Detailed working-directory and command-line usage
   - WSL interop checklist

3. **20251011-1502_pix_autonomous_agent_runbook.md**
   - Operational procedures
   - Common failures and fixes
   - Useful pixtool commands

4. **20251013-0416_pix_agent_robustness_improvements.md** (§13-14)
   - CLI presets and batch mode design
   - Analysis depth improvements
   - Roadmap timeline

5. **20251011-1502_pix_autonomous_agent_architecture.md** (§17-21)
   - Launch contract specification
   - Health check requirements
   - Acceptance criteria

---

## Next Actions

**Immediate (Today):**
1. ✅ Read all technical documents
2. ✅ Create master action plan (this document)
3. ⏳ Implement Priority 1 (stable capture path)
4. ⏳ Test single scenario

**This Week:**
1. Implement Priorities 2-4
2. Test multi-scenario batch
3. Document findings
4. Create PR with changes

**Future:**
1. Optional: Readiness polling
2. Optional: COM initialization test
3. CI integration
4. Performance benchmarking

---

**Status:** Ready for implementation
**Estimated Completion:** 2-3 days
**Risk Level:** Low (using proven pixtool workflow)
**Dependencies:** None (all infrastructure in place)
