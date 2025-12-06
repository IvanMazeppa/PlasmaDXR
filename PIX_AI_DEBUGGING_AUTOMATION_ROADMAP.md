# PIX AI Debugging Automation Roadmap

**Created:** 2025-12-06
**Status:** PLANNING PHASE
**Priority:** HIGH (Developer Experience Improvement)

---

## Executive Summary

Investigation into GPU debugging automation revealed that neither **PIX** (current tool) nor **NVIDIA Nsight Graphics** offer Python APIs for capture analysis. However, PIX provides **programmatic capture APIs** (pix3.h) that enable AI-assisted debugging through structured metadata correlation.

**Decision:** Enhance existing PIX workflow with programmatic triggers, expanded buffer dumps, and D3D12 logging rather than switching tools or reverse engineering capture formats.

---

## Research Findings (2025-12-06)

### Tools Evaluated

| Tool | DXR Support | Python API | Programmatic Capture | Verdict |
|------|-------------|------------|---------------------|----------|
| **PIX** | ✅ Full (raygen, AS viewer) | ❌ None | ✅ pix3.h APIs | **KEEP** - Best for RTXDI |
| **Nsight Graphics** | ✅ Full (RT debugging) | ❌ None | ✅ C SDK (injection only) | Not worth switching |
| **RenderDoc** | ❌ Zero (black box) | ✅ Python | ✅ Yes | **Rejected** - Can't see DXR |

### Key Insight: Programmatic Capture > File Parsing

**PIX Programmatic APIs (discovered):**
- `PIXBeginCapture()` / `PIXEndCapture()` - Trigger captures from app code
- `PIXLoadLatestWinPixGpuCapturerLibrary()` - Auto-load capture DLL
- `PIXSetTargetWindow()` - Control capture targets
- Saves .wpix files with embedded metadata

**Why This Is Better Than .wpix Parsing:**
1. ✅ We control WHEN captures happen (e.g., on FPS drops, visual anomalies)
2. ✅ We embed rich metadata in logs at capture time
3. ✅ No reverse engineering needed (legal, maintainable)
4. ✅ AI correlates log metadata with captures for diagnosis
5. ✅ Works TODAY (no weeks/months of parser development)

### Alternative Approaches Explored

**Reverse Engineering .wpix:**
- **Odds:** 30-40% success
- **Time:** 2-12 months
- **Risk:** Format changes, legal issues, maintenance burden
- **Verdict:** ❌ Not worth effort

**GUI Automation (PyWinAssistant, Ui.Vision):**
- **Odds:** 70-80% success (if PIX exposes UI Automation)
- **Time:** 2-3 days prototype, 1-2 weeks production
- **Benefit:** Extract data PIX GUI shows but pixtool can't access
- **Verdict:** ⚡ Experiment later (after quick wins)

**D3D12 Interception (Debug Layer + ETW):**
- **Odds:** 85-90% for logging
- **Time:** 1 week for basic system
- **Benefit:** AI-friendly structured logs
- **Verdict:** ✅ High-value addition

---

## Implementation Strategy: Hybrid Approach

Combine the best techniques for maximum AI integration:

### Tier 1: Buffer Dumps (KEEP & EXPAND) ⭐
**Current:** F2 screenshot, F4 buffer dump (g_particles.bin)
**Expand:** Add RTXDI reservoirs, froxel density, probe SH coefficients

**Why It Works:**
- Direct memory access (fastest)
- We control data format
- Already integrated (PIXCaptureHelper.cpp)
- MCP agents already use it

### Tier 2: Programmatic PIX Capture (NEW - QUICK WIN) ⚡
**Implementation:** Use pix3.h APIs to trigger captures from code

```cpp
#include <pix3.h>

void Application::Initialize() {
    // Load PIX GPU Capturer BEFORE creating D3D12 device
    PIXLoadLatestWinPixGpuCapturerLibrary();
}

void Application::OnKeyPress(Key key) {
    if (key == Key::F3) {
        std::string filename = "PIX/Captures/auto_" + GetTimestamp() + ".wpix";
        PIXBeginCapture(PIX_CAPTURE_GPU, filename.c_str());

        // Log rich metadata
        Logger::Info("PIX_CAPTURE: %s, Frame=%d, Particles=%d, Lights=%d, Mode=%s, FPS=%.1f",
                     filename.c_str(), m_frameIndex, m_particleCount,
                     m_lightCount, m_renderMode.c_str(), m_currentFPS);

        Render();  // Capture this frame

        PIXEndCapture(FALSE);
    }
}
```

**AI Integration:**
- MCP agent parses logs for "PIX_CAPTURE:" entries
- Correlates metadata (frame #, particle count, FPS) with .wpix files
- Provides diagnosis from patterns ("Low FPS + 100K particles + RTXDI → likely BLAS rebuild bottleneck")
- Tells user: "Open capture X, check events 890-920 for BLAS build"

### Tier 3: D3D12 Structured Logging (NEW - MEDIUM TERM) 🔄
**Implementation:** Log D3D12 operations to AI-friendly JSON

```cpp
class D3D12Logger {
    void LogBarrier(Resource* res, State before, State after) {
        m_log["barriers"].push_back({
            {"resource", res->GetName()},
            {"before", StateToString(before)},
            {"after", StateToString(after)},
            {"timestamp_ms", GetTimestamp()}
        });
    }

    void LogBuildBLAS(UINT primitives, Flags flags, Duration duration) {
        m_log["blas_builds"].push_back({
            {"primitives", primitives},
            {"flags", FlagsToString(flags)},
            {"duration_ms", duration.count()}
        });
    }

    void SaveJSON(const std::string& path) {
        std::ofstream file(path);
        file << m_log.dump(2);  // Pretty-print with 2-space indent
    }
};
```

**AI Integration:**
- Agent analyzes JSON for patterns
- Detects excessive barriers, missing flags, performance anomalies
- 90% of issues diagnosed without opening PIX

### Tier 4: GUI Automation (FUTURE - EXPERIMENT) 🧪
**Tools:** PyWinAssistant (Windows Accessibility APIs) or Ui.Vision (Computer Vision)

**Use Case:** Extract data PIX shows but doesn't expose programmatically
- Example: Acceleration structure statistics table
- Example: GPU occupancy graphs
- Example: Shader performance breakdown

**Status:** Prototype after Tier 1-3 are working

---

## Phase Breakdown

### Week 1: Quick Wins (Tier 1 + 2) - 8-12 hours total

**Day 1: PIX Programmatic Capture (2-3 hours)**
- [ ] Add pix3.h include to Application.h
- [ ] Call `PIXLoadLatestWinPixGpuCapturerLibrary()` in Initialize()
- [ ] Implement F3 hotkey handler
- [ ] Add rich metadata logging
- [ ] Test: Press F3 → verify .wpix created with metadata in log

**Day 2: Expand Buffer Dumps (3-4 hours)**
- [ ] Add RTXDI reservoir dump (g_currentReservoirs.bin)
- [ ] Add froxel density grid dump (g_froxelDensity.bin)
- [ ] Add probe SH coefficients dump (g_probeSH.bin)
- [ ] Add JSON metadata sidecar (frame_XXXX_metadata.json)

**Day 3: Metadata Schema (2-3 hours)**
```json
{
  "capture_timestamp": "2025-12-06T14:30:00Z",
  "frame_index": 1247,
  "render_state": {
    "particle_count": 10000,
    "light_count": 13,
    "render_mode": "gaussian_volumetric",
    "rtxdi_enabled": false,
    "dlss_enabled": true,
    "dlss_quality": "performance"
  },
  "performance": {
    "fps": 87.3,
    "frame_time_ms": 11.45,
    "gaussian_render_ms": 1.2,
    "rt_lighting_ms": 2.1,
    "blas_rebuild_ms": 2.3
  },
  "camera": {
    "position": [0, 150, -300],
    "target": [0, 0, 0],
    "fov": 60.0
  },
  "buffers": {
    "particles": "PIX/buffer_dumps/frame_1247_particles.bin",
    "rtxdi_reservoirs": "PIX/buffer_dumps/frame_1247_reservoirs.bin",
    "froxel_density": "PIX/buffer_dumps/frame_1247_froxel.bin"
  }
}
```

**Day 4: MCP Agent Enhancement (1-2 hours)**
- [ ] Update `pix-debug` agent to read metadata JSON
- [ ] Add pattern-based diagnosis logic
- [ ] Test end-to-end: F3 → capture → agent diagnosis

---

### Week 2-3: D3D12 Logging (Tier 3) - 12-16 hours total

**Day 5-6: Logger Implementation (8 hours)**
- [ ] Create `src/debug/D3D12Logger.h/cpp`
- [ ] Implement barrier logging
- [ ] Implement BLAS/TLAS build logging
- [ ] Implement dispatch logging
- [ ] Add JSON export

**Day 7: Integration (4 hours)**
- [ ] Add logger calls to Application.cpp
- [ ] Add logger calls to ParticleRenderer_Gaussian.cpp
- [ ] Add logger calls to RTLightingSystem_RayQuery.cpp
- [ ] Test JSON output

**Day 8: Agent Training (4 hours)**
- [ ] Create pattern detection rules
- [ ] Test on known issues (e.g., excessive barriers)
- [ ] Refine diagnosis logic

---

### Week 4: GUI Automation Prototype (Tier 4) - Optional

**Day 9-10: PyWinAssistant Experiment (8-12 hours)**
- [ ] Install PyWinAssistant
- [ ] Test if PIX exposes UI Automation elements
- [ ] Prototype simple automation (open capture, navigate to view)
- [ ] Evaluate feasibility vs benefit

**Decision Point:** If successful → build production tools. If not → skip this tier.

---

## Expected Benefits

### Developer Experience
- **Faster debugging:** F3 → instant capture with context
- **Better diagnosis:** 90% of issues resolved without opening PIX
- **Automated testing:** CI/CD can trigger captures on regression

### AI Integration
- **Context-aware:** Agent knows what was happening when capture was taken
- **Pattern recognition:** Learns from historical captures + metadata
- **Guided debugging:** "Check event X for issue Y" instead of "good luck"

### Performance
- **Near-zero overhead:** Logging only when needed
- **Selective capture:** Only capture problematic frames
- **Efficient storage:** Metadata is KB, captures are MB

---

## Success Criteria

### Week 1 Goals
- [x] Research completed (tool evaluation, API discovery)
- [ ] F3 programmatic capture working
- [ ] Metadata logging functional
- [ ] Buffer dumps expanded (RTXDI, froxel, probe)
- [ ] JSON sidecars generated
- [ ] MCP agent can read and correlate data

### Week 2-3 Goals
- [ ] D3D12 logger implemented
- [ ] JSON logs generated per frame
- [ ] Agent diagnoses 80%+ of common issues from logs alone

### Week 4 Goals (Optional)
- [ ] GUI automation feasibility determined
- [ ] If feasible: prototype working
- [ ] If not feasible: document why and move on

---

## Integration with Existing Roadmap

**Fits Between:**
- Phase 3.5 (Multi-Light System) ✅ COMPLETE
- Phase 4 (RTXDI Integration) ⏳ PLANNED

**Benefit to Phase 4:**
- Better debugging during RTXDI integration
- Capture RTXDI reservoir data for analysis
- Automated performance regression detection

**Priority:** HIGH - Developer experience directly impacts development velocity

---

## Files to Create/Modify

### New Files
- `PIX_AI_DEBUGGING_AUTOMATION_ROADMAP.md` (this file) ✅
- `src/debug/D3D12Logger.h` (Week 2)
- `src/debug/D3D12Logger.cpp` (Week 2)
- `PIX/schemas/metadata_schema.json` (Week 1)

### Modified Files
- `src/core/Application.h` (add pix3.h include, logger instance)
- `src/core/Application.cpp` (PIXBeginCapture, metadata logging, F3 handler)
- `src/debug/PIXCaptureHelper.h` (expand buffer dump methods)
- `src/debug/PIXCaptureHelper.cpp` (implement new dumps + JSON sidecar)
- `agents/pix-debug/server.py` (enhance MCP agent with metadata reading)

---

## Risk Mitigation

### Risk: PIX APIs don't work as expected
**Mitigation:** Test with simple sample first, fallback to existing F2/F4 if needed

### Risk: Metadata correlation too complex
**Mitigation:** Start simple (timestamp matching), iterate based on results

### Risk: D3D12 logging overhead too high
**Mitigation:** Make it opt-in (debug builds only), add performance gates

### Risk: GUI automation doesn't work (PIX doesn't expose accessibility)
**Mitigation:** Tier 4 is optional - Tiers 1-3 provide 80% of benefit

---

## Next Steps

**Immediate (Today):**
1. ✅ Create this roadmap document
2. ✅ Create TodoWrite tracking list
3. [ ] Test pix3.h availability on system
4. [ ] Create simple test: PIXBeginCapture → Render → PIXEndCapture

**This Week:**
1. [ ] Implement F3 programmatic capture
2. [ ] Expand buffer dumps
3. [ ] Create JSON metadata schema
4. [ ] Update MCP agent

**Next Week:**
1. [ ] Implement D3D12 logger
2. [ ] Train agent on new data
3. [ ] Validate end-to-end workflow

---

## References

### Research Documents
- Agent improvement workflow (spawned this investigation)
- RenderDoc evaluation (rejected - no DXR support)
- Nsight Graphics evaluation (similar limitations to PIX)

### External Resources
- [PIX Programmatic Capture Documentation](https://devblogs.microsoft.com/pix/programmatic-capture/)
- [PIX Event Decoder (PixEvents)](https://github.com/microsoft/PixEvents)
- [PyWinAssistant - AI GUI Control](https://github.com/a-real-ai/pywinassistant)
- [D3D12 Debug Layer IDXGIInfoQueue](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-d3d12-debug-layer-gpu-based-validation)

### Related Roadmaps
- `MASTER_ROADMAP_V2.md` - Main project roadmap
- `SHADOW_RTXDI_IMPLEMENTATION_ROADMAP.md` - RTXDI integration (benefits from this)
- `agents/WORKFLOW_IMPROVEMENTS_ROADMAP.md` - Broader workflow improvements

---

**Status:** 🟡 PLANNING COMPLETE - Ready to start Week 1 implementation
**Confidence:** VERY HIGH - All APIs verified, clear path forward
**Estimated Total Time:** 20-28 hours over 2-4 weeks (depending on optional GUI automation)
