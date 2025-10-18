# Legacy ReSTIR Cleanup - Remaining Tasks

**Status**: 90% complete
**Branch**: 0.7.5 (in progress, save as 0.7.7 when done)
**Context**: Session ran low on context during cleanup

---

## Completed ✅

### ParticleRenderer_Gaussian.h
- ✅ Removed reservoir buffer members (lines 138-145)
- ✅ Removed GetCurrentReservoirs() and GetPrevReservoirs() methods
- ✅ Removed RenderConstants::useReSTIR, restirInitialCandidates, frameIndex, restirTemporalWeight

### ParticleRenderer_Gaussian.cpp
- ✅ Removed reservoir buffer creation in Initialize() (63 MB × 2)
- ✅ Removed reservoir binding in Render()
- ✅ Removed reservoir swap logic (ping-pong)
- ✅ Removed reservoir cleanup in Resize()
- ✅ Updated root signature from 10 to 8 parameters
- ✅ Updated UAV barrier count from 3 to 2

### Application.cpp (Partial)
- ✅ Removed RenderConstants assignment (lines 552-556)
- ✅ Removed ReSTIR logging (lines 568-591)

---

## Remaining Tasks (10% - ~15 minutes)

### 1. Application.cpp - Remove Reservoir Getter Calls (~5 min)

**File**: `src/core/Application.cpp`

**Search for**: `GetCurrentReservoirs` and `GetPrevReservoirs` (around line 1332)

**Action**: Comment out or remove the entire buffer dump block that calls these methods

**Example**:
```cpp
// OLD (will fail to compile):
ID3D12Resource* reservoirs = m_gaussianRenderer->GetCurrentReservoirs();
ID3D12Resource* prevReservoirs = m_gaussianRenderer->GetPrevReservoirs();

// FIX: Remove or comment out
// (These methods no longer exist)
```

### 2. Application.cpp - Remove ReSTIR ImGui Controls (~5 min)

**File**: `src/core/Application.cpp`

**Search for**: Line 1871 (ImGui::Checkbox("ReSTIR"))

**Action**: Remove this entire section:
```cpp
// REMOVE:
ImGui::Checkbox("ReSTIR (F7)", &m_useReSTIR);
if (m_useReSTIR) {
    ImGui::SliderFloat("Temporal Weight (Ctrl/Shift+F7)", &m_restirTemporalWeight, 0.0f, 1.0f);
    ImGui::SliderInt("Initial Candidates", reinterpret_cast<int*>(&m_restirInitialCandidates), 8, 32);
}
```

### 3. Application.cpp - Remove F7 Keyboard Handler (~2 min)

**File**: `src/core/Application.cpp`

**Search for**: Lines 1107-1116 (F7, Ctrl+F7, Shift+F7 handling)

**Action**: Remove ReSTIR toggle logic:
```cpp
// REMOVE:
} else if (key == VK_F7) {
    if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
        m_restirTemporalWeight = (std::min)(1.0f, m_restirTemporalWeight + 0.1f);
        LOG_INFO("ReSTIR Temporal Weight: {:.1f}", m_restirTemporalWeight);
    } else if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
        m_restirTemporalWeight = (std::max)(0.0f, m_restirTemporalWeight - 0.1f);
        LOG_INFO("ReSTIR Temporal Weight: {:.1f}", m_restirTemporalWeight);
    } else {
        m_useReSTIR = !m_useReSTIR;
        LOG_INFO("ReSTIR: {} (speedup: {})",
                 m_useReSTIR ? "ON" : "OFF",
                 m_useReSTIR ? "10-60x" : "");
    }
}
```

### 4. Application.cpp - Remove Config Saving (~2 min)

**File**: `src/core/Application.cpp`

**Search for**: Lines 1496-1502 (SaveFrameStats function)

**Action**: Remove ReSTIR config save lines:
```cpp
// REMOVE:
fprintf(file, "  \"restir_enabled\": %s,\n", m_useReSTIR ? "true" : "false");
fprintf(file, "  \"restir_temporal_weight\": %.2f,\n", m_restirTemporalWeight);
fprintf(file, "  \"restir_initial_candidates\": %u,\n", m_restirInitialCandidates);
```

### 5. Application.cpp - Remove Title Bar Display (~1 min)

**File**: `src/core/Application.cpp`

**Search for**: Line 1551-1553 (window title bar)

**Action**: Remove ReSTIR display:
```cpp
// REMOVE:
if (m_useReSTIR) {
    swprintf_s(buf, L"[F7:ReSTIR:%.1f] ", m_restirTemporalWeight);
    wcscat_s(titleBar, buf);
}
```

### 6. Application.cpp - Remove Command-Line Args (~1 min)

**File**: `src/core/Application.cpp`

**Search for**: Lines 103-106 (argument parsing)

**Action**: Remove `--restir` and `--no-restir` handling:
```cpp
// REMOVE:
} else if (arg == "--no-restir") {
    m_useReSTIR = false;
} else if (arg == "--restir") {
    m_useReSTIR = true;
}
```

**Also remove from help text** (lines 135-136):
```cpp
// REMOVE:
LOG_INFO("  --restir             : Enable ReSTIR");
LOG_INFO("  --no-restir          : Disable ReSTIR");
```

### 7. Application.h - Remove Member Variables (~1 min)

**File**: `src/core/Application.h`

**Search for**: Member variables (likely around line 200-250)

**Action**: Remove:
```cpp
// REMOVE:
bool m_useReSTIR = false;
uint32_t m_restirInitialCandidates = 16;
float m_restirTemporalWeight = 0.1f;
```

### 8. Config Files - Remove ReSTIR Settings (Optional)

**Files**: `configs/user/*.json`, `configs/builds/*.json`

**Search for**: `"enableReSTIR"`, `"restirCandidates"`, `"restirTemporalWeight"`

**Action**: Remove these JSON entries (or leave them - config system will ignore unknown keys)

---

## Build & Test

After completing remaining tasks:

```bash
# Rebuild
MSBuild.exe PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

# Test multi-light still works
build\Debug\PlasmaDX-Clean.exe

# Expected:
#   ✅ 13 lights visible
#   ✅ 120 FPS @ 10K particles
#   ✅ No "Creating ReSTIR reservoir buffers..." log
#   ✅ No F7 key toggle
#   ✅ No ReSTIR checkbox in ImGui

# Save successful build
git add .
git commit -m "chore: Remove legacy ReSTIR implementation"
git tag 0.7.7
```

---

## Verification Checklist

After cleanup complete:

- [ ] Build succeeds with zero errors
- [ ] Application runs without crashes
- [ ] Multi-light system works (13 lights, 120 FPS)
- [ ] No reservoir buffer creation logs
- [ ] F7 key does nothing (previously toggled ReSTIR)
- [ ] ImGui has no ReSTIR checkbox
- [ ] Grep shows no remaining references:
  ```bash
  grep -r "ReSTIR\|m_useReSTIR\|reservoir" src/ --exclude-dir=.git
  # Should return ZERO matches (except comments about RTXDI vs legacy)
  ```

---

## Memory Savings

**Before cleanup**:
- Legacy ReSTIR buffers: 126 MB (2× 63 MB @ 1080p)
- PCSS shadow buffers: 7 MB (2× 3.5 MB)
- Gaussian output: 16 MB
- **Total**: 149 MB

**After cleanup**:
- PCSS shadow buffers: 7 MB
- Gaussian output: 16 MB
- **Total**: 23 MB
- **Savings**: 126 MB (84% reduction!)

---

## Next Steps After Cleanup

1. ✅ Verify multi-light works (should be immediate)
2. ✅ Save branch 0.7.7
3. 🎯 **Start RTXDI Milestone 4**: Reservoir sampling
   - Deploy rtxdi-integration-specialist-v4
   - Implement RTXDI reservoir buffers (different structure than legacy)
   - First visual test: RTXDI vs multi-light comparison

**Estimated time for M4**: 6-8 hours
**Expected FPS impact**: <5% (115-120 FPS)

---

**Last Updated**: 2025-10-19 00:20
**Completion**: 90% (remaining: ~15 minutes)
**Context Used**: 61% (session ending due to context limit)
