# Archived Agent Architecture Documentation (2025-11-17)

**Why archived:** These documents represented planning and exploration phases before the final architecture decision was made.

## What Happened

### The Journey (2025-11-14 to 2025-11-17)

1. **2025-11-14:** Started exploring multi-agent architecture, created initial hierarchy
2. **2025-11-15:** Built MCP tool servers, created path-and-probe specialist
3. **2025-11-16:** Built Mission-Control Agent SDK agent, created Rendering-Council proof-of-concept
4. **2025-11-16:** Tested Agent SDK ($0.79 per session) vs Legacy Agents (£0)
5. **2025-11-17:** **DECISION:** Use Legacy Agents for daily work, Agent SDK for automation only

### The Final Decision

**Problem:** Agent SDK councils cost $0.79+ per session with no quality advantage over free legacy agents.

**Solution:**
- ✅ **Daily work:** Legacy agents (`.claude/agents/`) - £0 cost, full capabilities
- ✅ **Automation:** Agent SDK (mission-control) - Only when cost justified (CI/CD, nightly QA)
- ❌ **Council agents:** Not used for daily work (too expensive)

**Result:**
- ~85KB of documentation consolidated to ~30KB
- 8 overlapping docs → 1 definitive spec
- Clear architecture decision
- £0 daily cost maintained

## Archived Documents

### Planning & Exploration
- **AUTONOMOUS_MULTI_AGENT_RAG_GUIDE.md** - Comprehensive guide (superseded by AGENT_ARCHITECTURE_SPECIFICATION.md)
- **MULTI_AGENT_ROADMAP.md** - Development roadmap (superseded)
- **AGENT_HIERARCHY_AND_ROLES.md** - Tier structure (superseded)
- **CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md** - Long-term vision (superseded)
- **VISION_REALIZATION_ROADMAP.md** - 3-phase plan (superseded by final decision)

### Status Snapshots
- **CURRENT_STATE_SUMMARY_2025-11-16.md** - Pre-decision status
- **WINDOW_2_COMPLETION_REPORT.md** - Window 2 status
- **WINDOW_2_HANDOFF_MCP_FIXES.md** - Window 2 handoff

## What Was Kept

### Active Documentation
- **docs/AGENT_ARCHITECTURE_SPECIFICATION.md** - **DEFINITIVE SPEC** (created 2025-11-17)
- **docs/CLAUDE.md** - Project context and user preferences
- **docs/sessions/** - Historical session logs (kept for context)

### Active Agents
- **.claude/agents/gaussian-volumetric-rendering-specialist.md** - Legacy agent (£0 cost, active)
- **agents/mission-control/** - Agent SDK agent (automation only, not daily use)
- **agents/rendering-council/** - Agent SDK proof-of-concept (not used daily)
- **agents/[7 MCP tool servers]** - Specialists providing 38+ tools (£0 cost, active)

## Key Lessons

### What Worked
1. ✅ MCP tool servers (correct as tools-only, no AI reasoning)
2. ✅ Legacy agents (£0 cost, full capabilities, perfect for daily work)
3. ✅ Agent SDK for autonomy (works great, but not worth daily cost)

### What Didn't Work
1. ❌ Agent SDK councils for daily work (too expensive, no quality advantage)
2. ❌ Complex multi-agent hierarchies (premature optimization)
3. ❌ Overlapping documentation (85KB of confusion)

### The Winning Architecture
```
You (Supervised-Autonomous)
    ↓
Legacy Agents (.claude/agents/) - £0 cost
    ↓
MCP Tool Servers (38+ tools) - £0 cost

[Agent SDK only for automation - rare use]
```

## If You're Reading This in the Future

**Don't read these archived docs for current architecture.**

**Read instead:**
- `docs/AGENT_ARCHITECTURE_SPECIFICATION.md` - The definitive spec (2025-11-17)

**These archived docs are historical context only** - they show the exploration process but don't reflect the final decision.

---

**Archived:** 2025-11-17
**Reason:** Superseded by AGENT_ARCHITECTURE_SPECIFICATION.md
**Value:** Historical context of decision-making process
