# PIX Agent Overhaul - Technical Design Document

## Executive Summary

The current PIX autonomous agent implementation is failing to capture GPU frames using `pixtool.exe` from WSL. Multiple approaches have been attempted (stable launch, attach, programmatic hybrid) but all encounter fundamental issues with path conversion, process attachment, and WSL/Windows interop. This document details the specific technical problems and outlines requirements for a complete overhaul to achieve fully autonomous PIX capture via `pixtool.exe` without hybrid approaches.

---

## Current State: What's Broken

### 1. **PIX Attach Mode Failures**

**Problem**: `pixtool.exe attach <PID>` consistently fails with exit code 255

**Symptoms**:
```bash
[Agent] ⚠ take-capture returned code 1
[Agent] Error: '"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"' is not recognized
```

**Root Causes**:
- PIX attach requires the target process to have PIX instrumentation loaded (WinPixGpuCapturer.dll)
- Standard Debug builds don't load PIX DLLs by default
- DebugPIX builds load PIX but may not expose the necessary hooks for external attachment
- WSL process IDs don't directly map to Windows PIX process attachment mechanisms

**What We Tried**:
```python
# multi_capture.py - FAILED
cmd = [
    str(PIX_TOOL),
    "attach", str(pid),  # PID from tasklist.exe
    f"--working-directory={win_project}",
    "take-capture", "--frames=1",
    "save-capture", win_capture
]
# Result: Exit code 255, no capture created
```

---

### 2. **PIX Stable Launch Mode Failures**

**Problem**: `pixtool.exe launch` command fails to find pixtool.exe itself when invoked from WSL

**Symptoms**:
```bash
[Agent] Step 3: Triggering PIX capture...
[Agent] ⚠ take-capture returned code 1
[Agent] Error: '"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"' is not recognized
[Agent] ✗ FAILED: Capture file not created
```

**Root Causes**:
- PIX tool path is being passed to Windows subprocess but shell context is lost
- When `subprocess.run()` executes pixtool from WSL, the command interpreter doesn't recognize the Windows path format
- The `pixtool launch` command spawns a child process that can't find its own executable

**What We Tried**:
```python
# pix_autonomous_agent.py stable mode - FAILED
pix_cmd = [
    str(PIX_TOOL),  # /mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe
    "launch", str(APP_PATH),
    f"--working-directory={win_project_root}",
    "take-capture", "--frames=1",
    "save-capture", win_output
]
# Result: "pixtool.exe is not recognized"
```

---

### 3. **Path Conversion Issues**

**Problem**: Inconsistent WSL-to-Windows path conversion causing subprocess failures

**WSL Paths Used**:
```
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe
```

**Windows Paths Needed**:
```
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
C:\Program Files\Microsoft PIX\2509.25\pixtool.exe
```

**Current Conversion Logic**:
```python
def convert_to_windows_path(linux_path):
    path_str = str(linux_path)
    if path_str.startswith('/mnt/'):
        drive = path_str[5].upper()
        rest = path_str[7:]
        return f"{drive}:\\{rest.replace('/', chr(92))}"
    return path_str
```

**Issues**:
- Conversion works for file paths but fails for executable paths in subprocess context
- `cwd=win_project_root` parameter causes `[Errno 2] No such file or directory` in subprocess
- PIX tool can't find its own DLLs when launched from WSL with Windows paths
- Spaces in paths ("Program Files") require special quoting that varies by shell context

---

### 4. **Process Lifecycle Confusion**

**Problem**: Unclear whether PIX needs to launch the app or attach to running app

**Current Approach**:
- App is launched manually by user
- Agent tries to attach to running process
- Attach fails, so we fall back to programmatic capture

**What Actually Works**:
- Programmatic capture (app controls PIX via WinPixGpuCapturer.dll)
- Manual PIX GUI workflow (user launches PIX, PIX launches app)

**What Doesn't Work**:
- `pixtool.exe attach` to running non-PIX-instrumented process
- `pixtool.exe launch` from WSL with proper path resolution

---

### 5. **Hybrid Approach Limitations**

**Current Workaround**: Using programmatic capture (app-controlled) instead of pixtool

**Problems with This**:
- Requires app restart for each capture (can't capture from running session)
- Can only capture at predetermined frame numbers (set via config)
- No real-time triggering capability
- Defeats the purpose of autonomous agent (agent should control PIX, not app)
- Can't capture arbitrary frames on demand

**Code Example**:
```python
# capture_5_frames.py - HYBRID APPROACH
config['debug']['pixAutoCapture'] = True
config['debug']['pixCaptureFrame'] = frame_num
# App controls capture, not agent - NOT IDEAL
```

---

## Goals: What We Need to Achieve

### Primary Goal
**Fully autonomous PIX capture via `pixtool.exe` from WSL-based Python agent**

The agent should:
1. Launch the app via `pixtool.exe launch`
2. Trigger captures at arbitrary times via `pixtool.exe take-capture`
3. Save captures with controlled naming
4. Extract analysis data via `pixtool.exe` commands
5. All without requiring app restarts or programmatic capture mode

### Secondary Goals

1. **Multi-Capture Capability**
   - Take 5+ captures from a single running app session
   - No app restarts between captures
   - Configurable delay between captures
   - Support for different camera presets/configurations

2. **Robust Path Handling**
   - Reliable WSL-to-Windows path conversion
   - Handle spaces in paths ("Program Files", "AndroidStudioProjects")
   - Work with both `/mnt/` paths and native Windows paths

3. **Process Control**
   - Launch app through PIX for proper instrumentation
   - Verify PIX instrumentation is active before capture attempts
   - Clean shutdown and process cleanup

4. **Error Recovery**
   - Detect failed captures immediately
   - Retry with exponential backoff
   - Clear error reporting for debugging

---

## Technical Requirements

### 1. PIX Tool Invocation from WSL

**Requirement**: Successfully invoke `pixtool.exe` from Python subprocess in WSL

**Considerations**:
- Must use correct shell wrapper (cmd.exe or PowerShell)
- Must properly quote paths with spaces
- Must set working directory correctly
- Must handle PATH environment for PIX DLL dependencies

**Proposed Solution**:
```python
# Use cmd.exe as wrapper to ensure proper Windows context
import subprocess

def run_pixtool(args, working_dir=None):
    """
    Run pixtool.exe via cmd.exe wrapper for proper Windows context

    Args:
        args: List of pixtool arguments (already in Windows path format)
        working_dir: Windows-format working directory

    Returns:
        subprocess.CompletedProcess
    """
    pix_exe = r"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"

    # Build command for cmd.exe
    cmd_args = [pix_exe] + args
    cmd_str = ' '.join(f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in cmd_args)

    full_cmd = ["/mnt/c/Windows/System32/cmd.exe", "/c", cmd_str]

    # Set environment
    env = os.environ.copy()
    if working_dir:
        env['CD'] = working_dir

    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=30
    )
```

**Testing Needed**:
- [ ] Verify pixtool.exe can be found and executed
- [ ] Verify pixtool.exe can find its own DLLs
- [ ] Verify working directory is respected
- [ ] Verify output capture works correctly

---

### 2. PIX Launch Workflow

**Requirement**: Launch application via PIX with proper instrumentation

**Command Structure**:
```bash
pixtool.exe launch <app_path> \
    --working-directory=<project_root> \
    take-capture --frames=<N> \
    save-capture <output_path>
```

**Critical Parameters**:
- `<app_path>`: Full Windows path to .exe (Debug or DebugPIX build)
- `<project_root>`: Windows path for DLL/shader/config loading
- `--frames=<N>`: Number of frames to capture (use 1 for single-frame capture)
- `<output_path>`: Full Windows path for .wpix file

**Implementation Requirements**:
```python
def launch_and_capture(app_path, output_path, frames=1, warmup_frames=120):
    """
    Launch app via PIX and capture frames

    Args:
        app_path: Windows path to exe
        output_path: Windows path for .wpix output
        frames: Number of frames to capture
        warmup_frames: Frames to render before capture

    Returns:
        (success: bool, capture_path: Path)
    """
    # Convert paths
    app_win = wsl_to_windows_path(app_path)
    output_win = wsl_to_windows_path(output_path)
    working_win = wsl_to_windows_path(project_root)

    # Build command
    args = [
        "launch", app_win,
        f"--working-directory={working_win}",
        "take-capture", f"--frames={frames}",
        "save-capture", output_win
    ]

    # Execute via cmd wrapper
    result = run_pixtool(args, working_dir=working_win)

    # Verify capture
    if result.returncode == 0 and Path(output_path).exists():
        return True, Path(output_path)

    return False, None
```

**Testing Needed**:
- [ ] App launches successfully
- [ ] App loads shaders/configs from correct working directory
- [ ] Capture triggers after warmup period
- [ ] .wpix file is created with valid data
- [ ] App exits cleanly after capture

---

### 3. PIX Attach Workflow (If Possible)

**Requirement**: Attach to running app and trigger capture on demand

**Challenge**: This may not be possible without PIX pre-instrumentation

**Investigation Needed**:
```python
# Can we make attach work?
def investigate_attach_requirements():
    """
    Determine what's needed for pixtool attach to work

    Questions:
    1. Does app need to load WinPixGpuCapturer.dll at startup?
    2. Does app need to call PIXLoadLatestWinPixGpuCapturerLibrary()?
    3. Can we inject PIX DLL into running process?
    4. Does DebugPIX build enable attach hooks?
    """
    pass
```

**If Attach is Viable**:
```python
def attach_and_capture(pid, output_path, frames=1):
    """
    Attach to running process and capture

    Args:
        pid: Windows process ID (from tasklist)
        output_path: Windows path for .wpix output
        frames: Number of frames to capture

    Returns:
        (success: bool, capture_path: Path)
    """
    output_win = wsl_to_windows_path(output_path)
    working_win = wsl_to_windows_path(project_root)

    args = [
        "attach", str(pid),
        f"--working-directory={working_win}",
        "take-capture", f"--frames={frames}",
        "save-capture", output_win
    ]

    result = run_pixtool(args, working_dir=working_win)

    return result.returncode == 0 and Path(output_path).exists(), Path(output_path)
```

**Testing Needed**:
- [ ] Identify exact requirements for attach to work
- [ ] Test with Debug vs DebugPIX builds
- [ ] Test with/without PIXLoadLatestWinPixGpuCapturerLibrary() call
- [ ] Document attach limitations

**Fallback Position**: If attach truly doesn't work, focus 100% on launch workflow

---

### 4. Multi-Capture Agent Architecture

**Requirement**: Agent that can take N captures in sequence

**Architecture**:
```python
class PIXCaptureAgent:
    """
    Autonomous PIX capture agent using pixtool.exe

    Capabilities:
    - Launch app via PIX with instrumentation
    - Capture multiple frames from single session (if attach works)
    - Capture multiple sessions with different configs (if attach doesn't work)
    - Automatic file naming and organization
    - Verification and retry logic
    """

    def __init__(self, project_root, pix_tool_path=None):
        self.project_root = Path(project_root)
        self.pix_tool = pix_tool_path or self._find_pix_tool()
        self.captures_dir = self.project_root / "pix" / "Captures"

    def capture_sequence(self, count=5, method="launch", config=None):
        """
        Capture N frames using specified method

        Args:
            count: Number of captures to take
            method: "launch" or "attach"
            config: Optional config override (for launch method)

        Returns:
            List of capture file paths
        """
        if method == "launch":
            return self._capture_sequence_launch(count, config)
        elif method == "attach":
            return self._capture_sequence_attach(count)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _capture_sequence_launch(self, count, config):
        """
        Capture by launching app N times
        Each launch captures at a different frame number for variety
        """
        captures = []
        frame_numbers = [60, 120, 180, 240, 300][:count]

        for i, frame_num in enumerate(frame_numbers, 1):
            print(f"[{i}/{count}] Launching for frame {frame_num} capture...")

            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"sequence_{i}_frame{frame_num}_{timestamp}.wpix"
            output_path = self.captures_dir / output_name

            # Launch and capture
            success, capture_path = self._launch_and_capture(
                output_path=output_path,
                warmup_frames=frame_num,
                config=config
            )

            if success:
                captures.append(capture_path)
                print(f"        ✓ {output_name}")
            else:
                print(f"        ✗ Failed")

            # Wait between launches
            if i < count:
                time.sleep(3)

        return captures

    def _capture_sequence_attach(self, count):
        """
        Capture from running app N times
        Requires app to be running with PIX instrumentation
        """
        # Find running process
        pid = self._find_running_app()
        if not pid:
            raise RuntimeError("App not running")

        captures = []
        for i in range(1, count + 1):
            print(f"[{i}/{count}] Capturing from running app (PID {pid})...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"attach_{i}_{timestamp}.wpix"
            output_path = self.captures_dir / output_name

            success, capture_path = self._attach_and_capture(
                pid=pid,
                output_path=output_path,
                frames=1
            )

            if success:
                captures.append(capture_path)
                print(f"        ✓ {output_name}")
            else:
                print(f"        ✗ Failed")

            # Wait between captures
            if i < count:
                time.sleep(5)

        return captures

    def _launch_and_capture(self, output_path, warmup_frames, config=None):
        """Launch app and capture (implementation from section 2)"""
        pass

    def _attach_and_capture(self, pid, output_path, frames):
        """Attach to app and capture (implementation from section 3)"""
        pass

    def _find_running_app(self):
        """Find PID of running PlasmaDX-Clean.exe"""
        result = subprocess.run(
            ["/mnt/c/Windows/System32/tasklist.exe", "/FI", "IMAGENAME eq PlasmaDX-Clean.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if "PlasmaDX-Clean.exe" in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    return parts[1].strip('"')
        return None
```

---

### 5. Path Conversion Utilities

**Requirement**: Bulletproof WSL ↔ Windows path conversion

**Implementation**:
```python
def wsl_to_windows_path(wsl_path):
    """
    Convert WSL path to Windows path

    Examples:
        /mnt/d/Users/... → D:\Users\...
        /mnt/c/Program Files/... → C:\Program Files\...

    Args:
        wsl_path: Path-like object or string in WSL format

    Returns:
        String in Windows format with backslashes
    """
    path_str = str(wsl_path)

    # Handle /mnt/X/ prefix
    if path_str.startswith('/mnt/'):
        drive_letter = path_str[5].upper()
        rest_of_path = path_str[7:]  # Skip /mnt/X/
        windows_path = f"{drive_letter}:\\{rest_of_path.replace('/', chr(92))}"
        return windows_path

    # Already a Windows path?
    if len(path_str) >= 2 and path_str[1] == ':':
        return path_str.replace('/', chr(92))

    # Unknown format
    raise ValueError(f"Cannot convert path: {path_str}")


def windows_to_wsl_path(windows_path):
    """
    Convert Windows path to WSL path

    Examples:
        D:\Users\... → /mnt/d/Users/...
        C:\Program Files\... → /mnt/c/Program Files/...

    Args:
        windows_path: Path-like object or string in Windows format

    Returns:
        String in WSL format with forward slashes
    """
    path_str = str(windows_path)

    # Handle drive letter
    if len(path_str) >= 2 and path_str[1] == ':':
        drive_letter = path_str[0].lower()
        rest_of_path = path_str[3:] if len(path_str) > 3 else ""  # Skip C:\
        wsl_path = f"/mnt/{drive_letter}/{rest_of_path.replace(chr(92), '/')}"
        return wsl_path

    # Already WSL format?
    if path_str.startswith('/mnt/'):
        return path_str

    # Unknown format
    raise ValueError(f"Cannot convert path: {path_str}")


def quote_for_cmd(path_str):
    """
    Quote path for cmd.exe if it contains spaces

    Args:
        path_str: Path string

    Returns:
        Quoted string if spaces present, otherwise original
    """
    if ' ' in path_str:
        return f'"{path_str}"'
    return path_str
```

**Testing Suite**:
```python
def test_path_conversion():
    """Test path conversion edge cases"""

    test_cases = [
        # (input, expected_windows, expected_wsl)
        ("/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean",
         "D:\\Users\\dilli\\AndroidStudioProjects\\PlasmaDX-Clean",
         "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"),

        ("/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe",
         "C:\\Program Files\\Microsoft PIX\\2509.25\\pixtool.exe",
         "/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe"),

        ("D:\\temp\\test.txt",
         "D:\\temp\\test.txt",
         "/mnt/d/temp/test.txt"),
    ]

    for wsl_input, expected_win, expected_wsl in test_cases:
        # Test WSL → Windows
        win_result = wsl_to_windows_path(wsl_input)
        assert win_result == expected_win, f"WSL→Win failed: {win_result} != {expected_win}"

        # Test Windows → WSL
        wsl_result = windows_to_wsl_path(expected_win)
        assert wsl_result == expected_wsl, f"Win→WSL failed: {wsl_result} != {expected_wsl}"

        print(f"✓ {wsl_input}")

    print("All path conversion tests passed!")
```

---

## Implementation Plan

### Phase 1: Core PIX Tool Integration (Priority: CRITICAL)

**Goal**: Get basic `pixtool.exe launch` working from WSL

**Tasks**:
1. [ ] Implement `run_pixtool()` wrapper function with cmd.exe
2. [ ] Implement robust path conversion utilities
3. [ ] Create minimal test script to verify pixtool execution
4. [ ] Test with simple launch command (no capture)
5. [ ] Add capture command once launch works
6. [ ] Verify .wpix file creation

**Success Criteria**:
- Single capture works reliably from Python script in WSL
- .wpix file is created and valid (can open in PIX GUI)
- No path-related errors
- Clean process shutdown

**Test Script**:
```python
# test_pixtool_basic.py
def test_basic_launch():
    """Most basic possible test of pixtool launch"""
    agent = PIXCaptureAgent(project_root)

    output_path = agent.captures_dir / "test_basic.wpix"
    success, path = agent._launch_and_capture(
        output_path=output_path,
        warmup_frames=60
    )

    assert success, "Launch failed"
    assert path.exists(), "Capture file not created"
    assert path.stat().st_size > 1000, "Capture file too small"

    print("✓ Basic pixtool launch working!")
```

---

### Phase 2: Multi-Capture Support (Priority: HIGH)

**Goal**: Capture 5 frames using launch method

**Tasks**:
1. [ ] Implement `PIXCaptureAgent` class structure
2. [ ] Implement `capture_sequence()` with launch method
3. [ ] Add proper file naming with timestamps
4. [ ] Add progress reporting
5. [ ] Add error handling and retry logic
6. [ ] Test with 5 sequential captures

**Success Criteria**:
- Can capture 5 frames in sequence
- Each capture has unique filename
- Failed captures don't break the sequence
- Total runtime under 5 minutes for 5 captures

**Test Script**:
```python
# test_multi_capture.py
def test_five_captures():
    """Test capturing 5 frames in sequence"""
    agent = PIXCaptureAgent(project_root)

    captures = agent.capture_sequence(
        count=5,
        method="launch"
    )

    assert len(captures) >= 3, f"Expected at least 3 captures, got {len(captures)}"

    for capture in captures:
        assert capture.exists(), f"Capture missing: {capture}"
        print(f"✓ {capture.name} ({capture.stat().st_size:,} bytes)")

    print(f"✓ Captured {len(captures)}/5 frames successfully!")
```

---

### Phase 3: PIX Attach Investigation (Priority: MEDIUM)

**Goal**: Determine if attach mode is viable and implement if possible

**Tasks**:
1. [ ] Research PIX attach requirements in Microsoft docs
2. [ ] Test attach with DebugPIX build
3. [ ] Test attach with PIXLoadLatestWinPixGpuCapturerLibrary() call
4. [ ] Test attach with different timing/startup conditions
5. [ ] Document findings
6. [ ] Implement attach method if viable
7. [ ] Add attach method to PIXCaptureAgent if working

**Success Criteria** (if viable):
- Can attach to running process and trigger capture
- Capture doesn't interfere with app rendering
- Can take multiple captures from single running session

**Fallback**:
- If attach doesn't work, document why and focus on launch method
- Ensure launch method is optimized for speed

---

### Phase 4: Polish and Features (Priority: LOW)

**Goal**: Add nice-to-have features for production use

**Tasks**:
1. [ ] Add config preset support (far, close, veryclose, inside)
2. [ ] Add capture analysis extraction (pixtool export commands)
3. [ ] Add automatic report generation
4. [ ] Add capture comparison tools
5. [ ] Add scheduling/automation hooks
6. [ ] Add web dashboard (optional)

---

## Testing Strategy

### Unit Tests
```python
# test_pix_agent.py

def test_path_conversion():
    """Test all path conversion edge cases"""
    # Implementation in section 5

def test_pixtool_wrapper():
    """Test run_pixtool() wrapper"""
    result = run_pixtool(["--help"])
    assert result.returncode == 0
    assert "pixtool" in result.stdout.lower()

def test_find_running_app():
    """Test process detection"""
    # Start test app
    # Verify agent can find it
    # Clean up

def test_capture_verification():
    """Test .wpix file validation"""
    # Create test capture
    # Verify it meets minimum size
    # Verify it can be opened by pixtool
```

### Integration Tests
```python
def test_end_to_end_single_capture():
    """Test complete workflow for single capture"""
    # Clean captures directory
    # Run agent with count=1
    # Verify capture created
    # Verify capture is valid

def test_end_to_end_five_captures():
    """Test complete workflow for 5 captures"""
    # Clean captures directory
    # Run agent with count=5
    # Verify all captures created
    # Verify timing is reasonable
```

### Manual Tests
- [ ] Launch via pixtool from PowerShell (baseline test)
- [ ] Launch via pixtool from WSL bash (directly)
- [ ] Launch via pixtool from WSL Python
- [ ] Verify captures in PIX GUI
- [ ] Test with different config presets
- [ ] Test error recovery (kill app mid-capture, etc.)

---

## Success Metrics

### Must-Have (MVP)
- [ ] Single capture works 100% of the time via pixtool launch
- [ ] 5-capture sequence works 90%+ of the time
- [ ] No manual intervention required
- [ ] Captures are valid and openable in PIX GUI
- [ ] Total runtime for 5 captures: < 5 minutes

### Nice-to-Have
- [ ] Attach mode working for on-demand captures
- [ ] Automatic analysis extraction
- [ ] Config preset integration
- [ ] < 30 second per-capture overhead

### Documentation
- [ ] Complete API documentation for PIXCaptureAgent
- [ ] Usage examples for common scenarios
- [ ] Troubleshooting guide
- [ ] Known limitations documented

---

## Known Constraints

1. **PIX Tool Location**: Assumes PIX is installed at standard location
   - Current: `C:\Program Files\Microsoft PIX\2509.25\pixtool.exe`
   - Need to support version detection or config override

2. **Build Requirements**: App must be built with D3D12 debug info
   - Release builds may not capture correctly
   - Need to document build requirements

3. **Performance**: Each launch-based capture requires full app startup
   - ~20-30 seconds per capture with warmup
   - Attach mode would reduce this significantly if viable

4. **WSL Limitations**: Some Windows interop features may be restricted
   - Process attachment across WSL boundary is complex
   - Path conversion must be robust

---

## Request for Implementation

**Please implement a complete overhaul of the PIX autonomous agent** following this design document. The implementation should:

1. **Start with Phase 1** (Core PIX Tool Integration)
   - Focus on getting ONE capture working reliably
   - Don't proceed to Phase 2 until Phase 1 is 100% solid

2. **Use the proposed architecture** from Section 4
   - Implement `PIXCaptureAgent` class
   - Use helper functions from Sections 1 and 5
   - Follow the testing strategy from Section 8

3. **Prioritize the launch method**
   - Attach is nice-to-have but not required for MVP
   - If launch works well, that's sufficient for autonomous capture

4. **Include comprehensive error handling**
   - Every pixtool call should have try/except
   - Every capture should be verified before returning success
   - Failed captures should log detailed diagnostics

5. **Test thoroughly before declaring success**
   - Run the test scripts from each phase
   - Verify captures in PIX GUI manually
   - Test failure scenarios (app crashes, disk full, etc.)

**Expected Deliverables**:
- `pix/pix_agent_v4.py` - New implementation of PIXCaptureAgent
- `pix/test_pix_agent.py` - Unit and integration tests
- `pix/run_5_captures.py` - Simple script to capture 5 frames (demonstrates usage)
- Updated documentation in this file with results/findings

**Timeline**: This is critical infrastructure - take the time needed to get it right rather than rushing to a broken solution.

---

## Questions to Resolve During Implementation

1. Does `pixtool.exe launch` require any special environment variables?
2. What is the exact format for the `--working-directory` parameter?
3. Can we use `wslpath` utility to aid path conversion?
4. Does PIX output any logs we can check for diagnostics?
5. What is the minimum size for a valid .wpix file?
6. Does the app need to be built with specific PIX compile flags?
7. Can we verify PIX instrumentation is active before attempting capture?

These should be answered through testing and documented in the implementation.

---

## Conclusion

The current PIX agent is fundamentally broken due to WSL/Windows interop issues with `pixtool.exe` invocation and path handling. A complete overhaul is needed that:

1. Uses proper Windows command execution context (cmd.exe wrapper)
2. Implements bulletproof path conversion
3. Focuses on the `launch` workflow (not hybrid/attach)
4. Includes comprehensive error handling and testing
5. Provides a clean API for multi-capture workflows

With this overhaul, we can achieve fully autonomous PIX capture without requiring app-side programmatic capture, enabling real-time debugging and analysis workflows.

**Please proceed with implementation starting at Phase 1.**
