# Window 2 Handoff: MCP Connection Fixes & Production Testing

**Branch**: 0.17.0
**Date**: 2025-11-16
**Task**: Fix 2 MCP server connections + production test autonomous agent

---

## Context (Read This First)

**What happened**: Mission-control autonomous agent is working (agents/mission-control/autonomous_agent.py) but 2 specialist MCP servers have connection issues.

**What's working**: 4/6 specialist MCP servers connected (dxr-image-quality-analyst, gaussian-analyzer, material-system-engineer, path-and-probe)

**What's broken**: 2/6 specialist MCP servers failing to connect (log-analysis-rag, pix-debug)

**Your mission**: Fix the 2 broken connections, then production test mission-control on a real problem.

---

## Task 1: Fix log-analysis-rag Connection (30 min)

### Diagnosis

**Error symptom**: Connection timeout or failed to connect in mission-control startup logs

**Likely cause**: Virtual environment or dependency issue

### Fix Steps

1. **Check virtual environment**:
```bash
cd agents/log-analysis-rag
ls -la venv/  # Does it exist?
source venv/bin/activate  # Can you activate it?
which python  # Should be in venv/
```

2. **Validate dependencies**:
```bash
cd agents/log-analysis-rag
cat requirements.txt
pip list | grep -E "(langchain|faiss|anthropic)"
```

3. **Test server standalone**:
```bash
cd agents/log-analysis-rag
./run_server.sh  # Does it start without errors?
# Press Ctrl+C after confirming startup
```

4. **Check for errors in run_server.sh**:
```bash
cat run_server.sh
# Look for:
# - Correct python path
# - Correct server.py path
# - Environment variable issues
```

5. **If virtual env missing or broken**:
```bash
cd agents/log-analysis-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

6. **Validate fix**:
```bash
cd agents/mission-control
python autonomous_agent.py "What tools do you have from log-analysis-rag?"
# Should list 6 tools: ingest_logs, query_logs, diagnose_issue, etc.
```

---

## Task 2: Fix pix-debug Connection (30 min)

**Same pattern as log-analysis-rag**:

1. Check virtual env exists: `agents/pix-debug/venv/`
2. Validate dependencies: `pip list | grep -E "(anthropic|mcp)"`
3. Test standalone: `cd agents/pix-debug && ./run_server.sh`
4. Recreate venv if needed
5. Validate: Test in mission-control autonomous agent

---

## Task 3: Production Test - Real Problem (1-2 hours)

### Option A: Probe Grid Brightness Issue

**Problem**: Probe grid may have intensity or coverage issues

**Test workflow**:
```bash
cd agents/mission-control
python autonomous_agent.py "Analyze probe grid lighting system. Is brightness adequate? Any coverage gaps?"
```

**Expected autonomous behavior**:
1. Calls `path-and-probe__analyze_probe_grid`
2. Calls `path-and-probe__validate_probe_coverage`
3. Searches codebase for probe intensity settings
4. Synthesizes findings with specific metrics
5. Recommends fix if issues found

**Success criteria**:
- Agent autonomously coordinates 2+ tools
- Provides quantified analysis (grid size, coverage %, intensity multiplier)
- Makes evidence-based recommendation

### Option B: DLSS-SR Breakage Investigation

**Problem**: Per docs/DLSS_SR_BREAKAGE_2025-11-14.md, DLSS Super Resolution may have issues

**Test workflow**:
```bash
cd agents/mission-control
python autonomous_agent.py "Investigate DLSS Super Resolution breakage from 2025-11-14. Read the breakage doc and recommend fix."
```

**Expected autonomous behavior**:
1. Reads `docs/DLSS_SR_BREAKAGE_2025-11-14.md`
2. Searches codebase for DLSS integration code
3. Searches logs for DLSS errors
4. Synthesizes root cause
5. Recommends fix with confidence level

### Option C: Visual Regression from Nov 2-4 (From Test 3)

**Problem**: Test 3 LPIPS comparison showed 69.29% similarity (33.7% below 85% threshold) between Nov 2-4 screenshots

**Test workflow**:
```bash
cd agents/mission-control
python autonomous_agent.py "Investigate visual regression between Nov 2-4. LPIPS was 69.29% (failed quality gate). What broke?"
```

**Expected autonomous behavior**:
1. Calls `dxr-image-quality-analyst__list_recent_screenshots`
2. Calls `dxr-image-quality-analyst__compare_screenshots_ml`
3. Calls `log-analysis-rag__query_logs` to search Nov 2-4 changes
4. Calls `path-and-probe__analyze_probe_grid` (likely culprit)
5. Synthesizes root cause with evidence

**Success criteria**:
- Agent identifies specific change that caused regression
- Provides evidence (log entries, config changes, shader modifications)
- Recommends rollback or fix

---

## Success Criteria (Overall)

### Connection Fixes
- ✅ log-analysis-rag shows "connected" in mission-control startup
- ✅ pix-debug shows "connected" in mission-control startup
- ✅ 6/6 specialist MCP servers operational

### Production Test
- ✅ Autonomous agent completes workflow without manual intervention
- ✅ Agent coordinates 3+ specialist tools
- ✅ Provides quantified, evidence-based recommendation
- ✅ Demonstrates supervised autonomy (recommends, doesn't just execute)

---

## Communication Protocol

**Report back to Window 1** with:
1. Connection fix results (success/failure for each)
2. Production test chosen (A, B, or C)
3. Autonomous agent behavior observed
4. Any blockers or issues encountered

**If stuck**: Check Window 1 for strategic guidance.

---

## Key Files Reference

**Mission-Control**:
- Autonomous agent: `agents/mission-control/autonomous_agent.py`
- Virtual env: `agents/mission-control/venv/`
- Run script: `agents/mission-control/run_server.sh` (NOT USED - it's an autonomous agent, not server)

**Specialist MCP Servers**:
- log-analysis-rag: `agents/log-analysis-rag/server.py`, `run_server.sh`
- pix-debug: `agents/pix-debug/server.py`, `run_server.sh`

**Documentation**:
- Vision roadmap: `docs/VISION_REALIZATION_ROADMAP.md`
- Agent SDK modernization: `docs/AGENT_SDK_MODERNIZATION_PLAN.md`
- Current state summary: `docs/CURRENT_STATE_SUMMARY_2025-11-16.md` (Window 1 creating)

---

## Important Notes

1. **Don't modify mission-control autonomous agent** - it's working correctly
2. **Only fix specialist MCP server connections** - don't change their implementations
3. **Mission-control is NOT an MCP server** - it's an autonomous agent (no run_server.sh)
4. **Test in interactive mode**: `python autonomous_agent.py` then type queries
5. **Check startup logs**: Mission-control prints which MCP servers connected successfully

---

**Estimated time**: 1-2 hours total (30 min per connection + 1 hour production test)

**Good luck! Report findings back to Window 1.**
