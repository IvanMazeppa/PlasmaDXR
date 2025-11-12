# ⚠️ DEPRECATED - This Agent Has Been Renamed

**This directory is deprecated as of 2025-11-12.**

## New Location

This agent has been **renamed** and **moved** to:

```
agents/dxr-image-quality-analyst/
```

## Reason for Rename

The agent was originally named "RTXDI Quality Analyzer" but has evolved into a **general-purpose DirectX Raytracing (DXR) image quality analysis tool** that supports:

- Multi-light systems (not just RTXDI)
- PCSS soft shadows
- Material system analysis (Phase 5)
- Adaptive particle radius
- DLSS integration
- Dynamic emission
- PINN hybrid mode
- Variable refresh rate

The old name "rtxdi-quality-analyzer" implied RTXDI-specific functionality, which was confusing since the agent is now universal.

## New Name

**DXR Image Quality Analyst** - DirectX Raytracing Image Quality Analysis

This name better reflects the agent's capabilities as a comprehensive rendering quality analyzer for any DXR-based system.

## Migration Steps

If you're using this agent in your MCP configuration:

1. **Update your Claude Code MCP settings** (`~/.claude/config.json` or equivalent):
   ```json
   {
     "mcpServers": {
       "dxr-image-quality-analyst": {
         "command": "bash",
         "args": ["/path/to/PlasmaDX-Clean/agents/dxr-image-quality-analyst/run_server.sh"]
       }
     }
   }
   ```

2. **Remove the old configuration entry** for "rtxdi-quality-analyzer"

3. **Restart Claude Code** to pick up the new configuration

## Available Tools (Unchanged)

The agent's tools remain the same:

1. **`list_recent_screenshots`** - List recent screenshots with metadata
2. **`compare_performance`** - Compare performance metrics across rendering modes
3. **`analyze_pix_capture`** - Analyze PIX GPU captures for bottlenecks
4. **`compare_screenshots_ml`** - ML-powered LPIPS perceptual similarity comparison
5. **`assess_visual_quality`** - AI vision analysis against 7-dimension quality rubric

## Documentation

See the new location for updated documentation:
- `agents/dxr-image-quality-analyst/README.md`
- `RTXDI_QUALITY_ANALYZER_UPGRADE_SUMMARY.md` (project root)

---

**This directory will be removed in a future cleanup.**

Please update your references and MCP configurations.
