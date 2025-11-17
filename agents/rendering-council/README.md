# Rendering Council - Agent SDK Implementation

Autonomous Agent SDK agent for rendering decisions and visual quality management in PlasmaDX-Clean.

## Purpose

This is a **proof-of-concept** comparing Agent SDK (autonomous) vs legacy agents (Claude Code subagents).

**Equivalent legacy agent:** `.claude/agents/gaussian-volumetric-rendering-specialist.md`

## Architecture Position

```
Top Tier (Agent SDK):    Rendering Council ← YOU ARE HERE
                         ↓ uses
Middle Tier (Legacy):    gaussian-volumetric-rendering-specialist
                         ↓ uses
Bottom Tier (MCP):       gaussian-analyzer, dxr-image-quality-analyst
```

## Capabilities

- **Autonomous rendering decisions** (not just tool execution)
- **Multi-session persistence** (maintains context across runs)
- **Strategic visual quality management**
- **Direct MCP tool access** (gaussian-analyzer, dxr-image-quality-analyst, pix-debug)

## Installation

```bash
cd agents/rendering-council
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python rendering_council_agent.py "Analyze Gaussian rendering bugs and propose fixes"
```

## Configuration

- **API Key**: Set `ANTHROPIC_API_KEY` environment variable
- **Model**: Uses Claude Sonnet 4.5 (medium-weight council agent)
- **MCP Servers**: Connects to gaussian-analyzer and dxr-image-quality-analyst

## Cost Estimates

- **Per session**: $0.10 - $0.50 (estimated, depends on task complexity)
- **Per month** (10 sessions): ~$1-5
- **Compared to**: Legacy agents use Claude Code subscription (£0 extra)

## Comparison Metrics

Track these to compare Agent SDK vs legacy:
1. **Autonomy**: Does it make better decisions independently?
2. **Quality**: Does it produce better solutions?
3. **Speed**: Is it faster or slower?
4. **Cost**: Is the API cost justified by improved effectiveness?

## Next Steps

If Rendering Council proves effective:
1. Create Physics Council
2. Create Materials Council
3. Create Diagnostics Council
4. Create Orchestrator to coordinate all councils
