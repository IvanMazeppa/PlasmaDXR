# CCHAT: Agent Reorganization & Upgrade (Recovered)

Date: 2025-10-17
Topic: Organizing Claude Code agents, using plugin system, and next steps for autonomous debugging

You: We need to organize agents and use the new Claude Code plugin system effectively.
Assistant: Summarizes built-in agents, when to use each, and Agent SDK vs built-ins.

Key decisions
- Use built-in agents for fast diagnosis (pix-debugging-agent, dxr-graphics-debugging-engineer-v2, hlsl-volumetric-implementation-engineer-v2, dxr-systems-engineer-v2, physics-performance-agent-v2, rt-ml-technique-researcher-v2).
- Use Agent SDK for repeated or overnight workflows (Multi-Light Validator, RTXDI Integration Validator, Performance Regression Detector, Autonomous PIX Debugger v2).
- Keep current hybrid PIX agent scripts; move to Agent SDK later for CI and nightly.

Actionable invocations
- Multi-light issues: "Use pix-debugging-agent to diagnose why my multi-light system shows no lighting."
- DXR crash or black screen: "Use dxr-graphics-debugging-engineer-v2 to diagnose BLAS build crash."
- Volumetric shader feature: "Use hlsl-volumetric-implementation-engineer-v2 for anisotropic scattering."
- Plan RTXDI integration: "Use dxr-systems-engineer-v2 to plan RTXDI integration."
- Research RTXDI: "Use rt-ml-technique-researcher-v2 to summarize RTXDI requirements for my DXR 1.1 pipeline."

Immediate next steps
- Save and reference CLAUDE_CODE_PLUGINS_GUIDE.md.
- Start with pix-debugging-agent for remaining multi-light polish tasks.
- Prepare Agent SDK scaffold only when you want nightly regression across scenarios.

Artifacts referenced
- CLAUDE_CODE_PLUGINS_GUIDE.md (installed plugins, agents, workflows)
- PIX_AGENT_V4_DEVELOPMENT_PLAN.md (automation roadmap)
- SESSION_SUMMARY_20251015_CLI.md, SESSION_HANDOFF_20251015.md (session context)
