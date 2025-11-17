# Agent SDK vs Legacy Agent Comparison

## Side-by-Side Comparison

| Feature | **Agent SDK** (Rendering Council) | **Legacy Agent** (gaussian-volumetric-rendering-specialist) |
|---------|-----------------------------------|-------------------------------------------------------------|
| **Location** | `agents/rendering-council/` | `.claude/agents/gaussian-volumetric-rendering-specialist.md` |
| **Type** | Python standalone service | Markdown prompt for Claude Code |
| **Execution** | Independent process | Within Claude Code session |
| **Context** | Persistent across runs | Fresh each invocation |
| **Autonomy** | True autonomy with API calls | Executed by Claude Code |
| **Cost** | $0.10-0.50 per session (API tokens) | £0 (included in Claude Code subscription) |
| **Resumability** | Yes (agent ID + session) | Yes (via resume parameter) |
| **MCP Tools** | Direct access via SDK | Via Claude Code's MCP integration |
| **Multi-turn** | Native (ClaudeSDKClient) | Native (subagent context) |

## What Agent SDK Adds

**1. True Independence**
- Runs as separate process
- Can be triggered by cron jobs, webhooks, CI/CD
- Not tied to Claude Code being open

**2. Programmatic Control**
- Python API for integration
- Can be called from other scripts
- Easier automation

**3. Custom Logic**
- Can add Python preprocessing
- Custom MCP server implementations
- More flexible tool integration

**4. Long-running Sessions**
- Maintains state across days/weeks
- Can handle multi-day analysis tasks
- Session persistence built-in

## What Agent SDK Costs

**1. API Tokens**
- Every interaction costs tokens
- Input + output charged
- Can add up for frequent use

**2. Setup Complexity**
- Requires Python environment
- API key management
- More moving parts

**3. No Built-in Claude Code Tools**
- Must explicitly integrate MCP servers
- Doesn't have native Read/Write/Bash access
- Requires more configuration

## Test Criteria

Track these metrics to determine if Agent SDK is worth the cost:

### Effectiveness
- [ ] Does it make better rendering decisions?
- [ ] Does it catch more bugs?
- [ ] Are its solutions more comprehensive?
- [ ] Does it research solutions more thoroughly?

### Autonomy
- [ ] Does it work more independently (less hand-holding)?
- [ ] Does it follow through on multi-step plans?
- [ ] Does it self-correct when wrong?
- [ ] Does it ask for help appropriately?

### Efficiency
- [ ] Is it faster to complete tasks?
- [ ] Does it require fewer iterations?
- [ ] Does it produce production-ready code first time?

### Value
- [ ] Cost per session: $______
- [ ] Value delivered per session: High / Medium / Low
- [ ] Would you pay for this over using legacy agents? Yes / No
- [ ] Break-even point: ______ sessions per month justified

## Recommendation Framework

**Use Agent SDK if:**
- ✅ Task requires multi-day context
- ✅ Need automation/CI integration
- ✅ Benefits justify $0.10-0.50 per session
- ✅ Need true autonomy outside Claude Code

**Use Legacy Agent if:**
- ✅ Working within Claude Code already
- ✅ Cost-sensitive (£0 extra)
- ✅ One-off analysis/debugging
- ✅ Don't need persistence across sessions

## Next Steps After Testing

**If Agent SDK is MORE effective:**
1. Create remaining 3 councils (Physics, Materials, Diagnostics)
2. Create Orchestrator to coordinate councils
3. Set monthly budget ($10-20/month)
4. Use for critical architectural decisions

**If Agent SDK is EQUIVALENT:**
- Use Agent SDK for automation/CI only
- Stick with legacy agents for daily work
- Reserve Agent SDK for long-running analysis

**If Agent SDK is LESS effective:**
- Identify why (cost constraints? setup issues?)
- Stick with legacy agent architecture
- Revisit when SDK matures or costs decrease
