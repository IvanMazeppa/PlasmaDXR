# Multi-Agent Debugging Workflow Proposal

**Current State:** Manual debugging of Volumetric ReSTIR GPU hang
**Proposed:** Automated multi-agent diagnostic system

---

## Current Problem Summary

**Symptom:** GPU TDR hang at ≥2045 particles, Map() failure on frame 1+
**Root Cause:** Unknown (suspected: GPU work incomplete before CPU readback)
**Time Spent:** 6+ hours of manual debugging
**Result:** Still not resolved

---

## Why Agent-Based Approach Would Help

### Current Manual Process
```
1. User reports: "Crash at 2045 particles"
2. Read 400-line log file manually
3. Find: Map() failure at frame 1
4. Hypothesis: GPU work incomplete?
5. Try fix: Add WaitForGPU
6. Test fails
7. Hypothesis: Shader not executing?
8. Try fix: Add diagnostic counters
9. Test shows shader DOES execute
10. Hypothesis: TDR timeout?
11. Try fix: Reduce volume 64³→32³
12. Test shows voxels still 175k (shader unchanged?)
... repeat for hours
```

### Agent-Based Process
```
1. User: "Crash at 2045 particles"
2. log-analysis-rag agent:
   - Semantic search: "2045" + "crash" + "Map" + "WaitForGPU"
   - Pattern: Map() succeeds frame 0, fails frame 1+
   - PIX correlation: WaitForGPU takes 0ms frame 0, 3000ms frame 1
   - Hypothesis: GPU command list not complete
   - Confidence: 0.85
3. pix-debug agent:
   - Analyzes frame 1 PIX capture
   - Finds: PopulateVolumeMip2 dispatch duration = 2843ms
   - Evidence: Exceeds 2-second TDR timeout
4. Diagnostic: "PopulateVolumeMip2 shader takes 2.8 seconds"
5. Fix: Reduce computation (lower resolution OR simpler algorithm)
... resolution in minutes instead of hours
```

---

## Immediate Next Steps

### Step 1: Fix the Map() Failure (Manual - Last Try)

The Map() failure suggests **GPU work not complete**. The real question is WHY.

**Hypothesis:** CopyResource (diagnostic counters) happens in same command list as compute dispatch, but GPU hasn't finished the compute work before we try to Map().

**Test:** Add explicit GPU fence after CopyResource:

```cpp
// After CopyResource diagnostic counters
D3D12_RESOURCE_BARRIER uavBarrier = {};
uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
uavBarrier.UAV.pResource = m_diagnosticCounterBuffer.Get();
commandList->ResourceBarrier(1, &uavBarrier);

// Add fence to ensure GPU work completes
ComPtr<ID3D12Fence> fence;
m_device->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
commandList->Signal(fence.Get(), 1);

// Wait for fence (ensures GPU finished CopyResource)
HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
fence->SetEventOnCompletion(1, fenceEvent);
WaitForSingleObject(fenceEvent, INFINITE);
CloseHandle(fenceEvent);

// NOW try to Map() - GPU work guaranteed complete
Map(m_diagnosticCounterReadback.Get(), ...);
```

### Step 2: Set Up Log Analysis Agent (Start Today)

**Quick MVP** (2-3 hours):
```bash
cd agents/log-analysis-rag
pip install chromadb sentence-transformers
python setup_mcp.py
```

**Test query:**
```python
query_logs("Find Map() failures correlating with particle count")
```

**Expected output:**
```
Found pattern:
- 2044 particles: Map() succeeds
- 2045 particles: Map() fails (0x{:08X})
- Correlation: 100% reproducible at threshold

Hypothesis: GPU work incomplete at CopyResource time
Confidence: 0.82
```

### Step 3: Integrate PIX Agent with RAG

**Goal:** Correlate log timestamps with PIX event timeline

```python
# Log shows:
[18:17:37] [INFO] About to dispatch PopulateVolumeMip2

# PIX shows (from pixtool CSV):
Event 142, 18:17:37.243, Dispatch, PopulateVolumeMip2, Duration: 2843ms

# Agent correlates:
"Dispatch at 18:17:37 took 2843ms → Exceeds TDR timeout (2000ms)"
```

### Step 4: Historical Learning

Once RAG is set up, it learns from this debugging session:

**Pattern learned:**
```
IF particle_count >= 2045 AND
   Map() fails AND
   WaitForGPU shows >2000ms delay
THEN root_cause = "TDR timeout in PopulateVolumeMip2"
     confidence = 0.95
```

Next time similar issue occurs → Instant diagnosis

---

## Comparison: Manual vs Agent-Based

### Current Session Stats
- **Time spent:** 6+ hours
- **Hypotheses tested:** 8
- **Fixes attempted:** 10+
- **Log lines read:** ~5,000
- **Resolution:** Not yet resolved

### Agent-Based Projection
- **Time to diagnosis:** 5-10 minutes
- **Hypotheses generated:** 3-5 (ranked by confidence)
- **Evidence correlation:** Automatic (logs + PIX)
- **Fix suggestions:** File:line specific
- **Resolution:** High probability within 30 minutes

---

## Implementation Priority

### Critical (Do First)
1. ✅ Spec written (DONE - see log-analysis-rag/SPEC.md)
2. [ ] MVP log ingestion (ChromaDB + basic search)
3. [ ] Test query: "Find Map() failures"
4. [ ] Verify results match manual inspection

### High Priority (This Week)
1. [ ] PIX event CSV parsing
2. [ ] Timeline correlation (log ↔ PIX)
3. [ ] Pattern detection (TDR, resource states)
4. [ ] MCP tool: `diagnose_issue`

### Medium Priority (Next Week)
1. [ ] Diagnostic agent (hypothesis generation)
2. [ ] Confidence scoring
3. [ ] Fix suggestions with code snippets
4. [ ] Integration with existing pix-debug agent

### Nice to Have (Future)
1. [ ] Historical learning (save patterns)
2. [ ] Cross-validation (multiple evidence sources)
3. [ ] Automated fix testing (apply fix → run test → verify)
4. [ ] Visual UI (web dashboard for diagnostics)

---

## ROI Analysis

### Investment
- **Setup time:** 2-3 hours (MVP)
- **Full implementation:** 2-3 weeks part-time
- **Maintenance:** Minimal (self-correcting)

### Return
- **Time saved per debugging session:** 4-5 hours
- **Debugging sessions per month:** 2-3
- **Monthly time savings:** 8-15 hours
- **Payback period:** <1 week

### Intangible Benefits
- Reduced frustration
- Systematic diagnosis (not trial-and-error)
- Knowledge retention (learned patterns)
- Faster onboarding (agent explains issues)

---

## Conclusion

**Manual debugging is hitting diminishing returns.** We've spent 6 hours and still haven't resolved the issue.

**Agent-based workflow would:**
1. Diagnose the current issue in minutes
2. Prevent similar issues in future (pattern learning)
3. Free up time for feature development (not debugging)
4. Provide systematic evidence (not guesswork)

**Recommendation:**
1. **Short term:** Try the fence-based Map() fix above (15 minutes)
2. **Medium term:** Set up log analysis MVP (2-3 hours)
3. **Long term:** Full multi-agent debugging system (2-3 weeks)

**Alternative:** If PopulateVolumeMip2 continues to be problematic, consider RTXGI instead (production-ready solution for volumetric scattering).

---

**Next Action:** Would you like me to:
A) Implement the fence-based Map() fix first (quick test)
B) Start setting up the log analysis RAG MVP (strategic investment)
C) Explore RTXGI as alternative to Volumetric ReSTIR (different approach)
