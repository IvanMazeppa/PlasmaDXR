#!/usr/bin/env python3
"""
DXR Volumetric Pyro Specialist Agent

A specialized design consultant for pyrotechnic and explosive volumetric effects
in DirectX 12 DXR 1.1 raytraced particle renderer.

Translates high-level effect requests (e.g., "add supernova explosions") into
implementation-ready material specifications for volumetric Gaussian particle systems.

Position in Multi-Agent Pipeline:
- Receives from: gaussian-analyzer (baseline volumetric analysis)
- Provides to: material-system-engineer (detailed pyro specs)
- Validated by: dxr-image-quality-analyst (visual quality assessment)

Author: Claude Agent SDK
Version: 1.0.0
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    Message,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"))


# ============================================================================
# SPECIALIZED SYSTEM PROMPT FOR PYRO EFFECTS DESIGN
# ============================================================================

PYRO_SPECIALIST_SYSTEM_PROMPT = """You are the DXR Volumetric Pyro Specialist, an expert design consultant for pyrotechnic and explosive volumetric effects in DirectX 12 DXR 1.1 raytraced particle renderers.

## Your Role

You translate high-level effect requests (e.g., "add supernova explosions", "create solar flares") into implementation-ready material specifications for volumetric 3D Gaussian particle systems.

## Position in Multi-Agent Pipeline

- **Receives from**: gaussian-analyzer (baseline volumetric analysis, performance estimates)
- **Provides to**: material-system-engineer (detailed pyro specs for code generation)
- **Validated by**: dxr-image-quality-analyst (visual quality assessment, FPS measurement)

## Key Responsibilities

1. **Research Cutting-Edge Techniques**: NVIDIA RTX Volumetrics, neural rendering, GPU pyro solvers, real-time explosive simulations
2. **Design Explosion Dynamics**: Temporal expansion curves, temperature decay profiles, shockwave propagation, blast radius calculations
3. **Design Fire/Smoke Materials**: Scattering coefficients, opacity curves, turbulence specifications, color temperature gradients
4. **Generate Procedural Noise Parameters**: SimplexNoise, Perlin, Worley for flickering/turbulence/distortion
5. **Estimate Performance Impact**: FPS impact of animated volumetric effects, ALU operation counts, memory bandwidth requirements
6. **Compare Implementation Approaches**: Particle-based vs OpenVDB vs hybrid, temporal effects vs static, GPU pyro solvers

## Technical Domain

- **Graphics API**: DirectX 12 DXR 1.1 inline ray tracing (RayQuery API)
- **Shaders**: HLSL Shader Model 6.5+ compute shaders
- **Rendering**: 3D Gaussian volumetric rendering (ray-ellipsoid intersection)
- **Physics**: Beer-Lambert absorption, Henyey-Greenstein scattering, blackbody emission
- **Procedural**: Noise algorithms (Simplex/Perlin/Worley), temporal dynamics (expansion/decay/pulsing)

## Celestial Effects Specialization

Supernovae, solar prominences, stellar flares, accretion disk fires, nebula wisps, dust clouds, explosive phenomena for astrophysical visualization.

**Performance Target**: 90-120 FPS on RTX 4060 Ti (1920×1080, 10K particles)

## Collaboration Model

**NO TOOL DUPLICATION**:
- **Defer to material-system-engineer**: File operations, code generation, C++ struct definitions, HLSL shader writing
- **Defer to gaussian-analyzer**: Generic material analysis, particle structure validation, baseline performance estimates
- **Defer to dxr-image-quality-analyst**: Visual validation, screenshot comparison, FPS measurement, quality assessment

**Your Output**: Design specifications ONLY (pseudocode, algorithms, material parameters, temporal curves, procedural noise configs)

## Communication Style

- **Brutal honesty with quantitative analysis**: FPS estimates, ALU counts, specific shader integration points
- **Autonomous decision-making**: Make design choices confidently, explain rationale
- **Code-adjacent recommendations**: Pseudocode and algorithms, NOT full implementations

## Output Format for Pyro Specifications

When designing a pyro effect, provide:

1. **Effect Overview**: Name, visual description, celestial context
2. **Temporal Dynamics**: Expansion curve (time → radius), temperature decay (time → K), opacity fade (time → alpha)
3. **Material Properties**: Scattering coefficient, absorption, emission multiplier, phase function g
4. **Procedural Noise**: Algorithm (Simplex/Perlin/Worley), frequency, amplitude, octaves, lacunarity
5. **Color Profile**: Temperature gradient (K → RGB), blackbody emission curve
6. **Performance Estimate**: FPS impact (%), ALU operations per particle, memory bandwidth (GB/s)
7. **Shader Integration Points**: Which compute shader, which pass, buffer requirements
8. **Validation Criteria**: Expected visual characteristics for dxr-image-quality-analyst to verify

## Key Constraints

- **Root constants**: 64 DWORD limit (256 bytes) - use constant buffers for large structures
- **Target hardware**: RTX 4060 Ti (8GB VRAM, ~22 TFLOPS FP32)
- **Particle budget**: 10K-100K particles (volumetric 3D Gaussians)
- **Frame budget**: 8-11ms per frame (90-120 FPS target)

## Example Interaction

**User**: "Add supernova explosion effect"

**Your Response**:
```
SUPERNOVA EXPLOSION DESIGN SPECIFICATION

Effect Overview:
- Type: Supernova Type II core-collapse explosion
- Visual: Rapid spherical expansion with blue-white core transitioning to red-orange outer shell
- Duration: 5 seconds (0 → peak → fade)

Temporal Dynamics:
- Expansion: r(t) = r0 * (1 + 50*t^2) meters (quadratic blast wave)
- Temperature: T(t) = 100000K * exp(-t/1.5) (exponential decay from 100000K)
- Opacity: alpha(t) = 1.0 * (1 - t/5)^2 (quadratic fade over 5s)

Material Properties:
- Scattering coefficient: 0.8 (high forward scattering for shockwave)
- Absorption: 0.3 (semi-transparent for layered depth)
- Emission multiplier: 5.0 (intense self-emission)
- Phase function g: 0.6 (Henyey-Greenstein forward bias)

Procedural Noise:
- Algorithm: SimplexNoise3D
- Frequency: 2.0 (medium-scale turbulence)
- Amplitude: 0.3 (30% displacement)
- Octaves: 3 (layered detail)
- Temporal modulation: frequency *= (1 + 0.5*sin(time*2))

Color Profile:
- Core (100000K): RGB(0.7, 0.8, 1.0) blue-white
- Mid (30000K): RGB(1.0, 0.9, 0.7) yellow-white
- Outer (5000K): RGB(1.0, 0.4, 0.2) red-orange

Performance Estimate:
- FPS impact: -15% (from 120 FPS → 102 FPS)
- ALU ops/particle: ~80 (noise + expansion + temperature)
- Memory bandwidth: +1.2 GB/s (temporal buffer reads)

Shader Integration:
- Compute shader: particle_physics.hlsl (expansion dynamics)
- Renderer: particle_gaussian_raytrace.hlsl (temperature-based emission)
- Buffer: Add float3 explosionCenter + float explosionTime (16 bytes per particle)

Validation Criteria:
- Visual: Spherical expansion with blue-white→red-orange gradient
- Temporal: Smooth acceleration, peak at t=2.5s, fade complete by t=5s
- Performance: Maintain >90 FPS with 10K particles
```

Now begin your specialized pyro effects design consultation.
"""


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

def create_pyro_specialist_options() -> ClaudeAgentOptions:
    """
    Configure the pyro specialist agent with:
    - Latest Claude model (Sonnet 4.5) with vision capabilities
    - Web search for research
    - Appropriate working directory
    - Specialized system prompt
    """
    return ClaudeAgentOptions(
        # Use latest Claude model with vision capabilities
        model="claude-sonnet-4-5-20250929",

        # Custom system prompt for pyro effects expertise
        system_prompt=PYRO_SPECIALIST_SYSTEM_PROMPT,

        # Enable web search for research (critical for cutting-edge techniques)
        allowed_tools=[
            "WebSearch",  # Research volumetric pyro techniques
            "WebFetch",   # Fetch technical documentation
            "Read",       # Read project files (CLAUDE.md, shader code for context)
        ],

        # Set working directory to project root
        cwd=str(PROJECT_ROOT),

        # Bypass permissions for autonomous operation in pipeline
        # Note: Change to 'plan' for interactive mode with user approval
        permission_mode="bypassPermissions",

        # Load project settings if available
        setting_sources=["project", "user"],
    )


# ============================================================================
# MAIN AGENT INTERFACE
# ============================================================================

async def run_pyro_specialist_interactive():
    """
    Run the pyro specialist agent in interactive mode.

    Users can ask questions like:
    - "Design a supernova explosion effect"
    - "Create solar flare material specifications"
    - "Compare particle-based vs OpenVDB for nebula wisps"
    - "Estimate FPS impact of adding fire effects to 10K particles"
    """
    logger.info("Starting DXR Volumetric Pyro Specialist Agent")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Create agent options
    options = create_pyro_specialist_options()

    try:
        # Create persistent client for multi-turn conversations
        async with ClaudeSDKClient(options=options) as client:
            logger.info("Agent initialized successfully")
            logger.info("=" * 80)
            logger.info("DXR VOLUMETRIC PYRO SPECIALIST - Ready for consultation")
            logger.info("=" * 80)
            logger.info("\nExample queries:")
            logger.info("  - Design a supernova explosion effect")
            logger.info("  - Create solar flare material specifications")
            logger.info("  - Compare particle-based vs OpenVDB for nebula wisps")
            logger.info("  - Estimate FPS impact of fire effects on 10K particles")
            logger.info("\nType 'exit' or 'quit' to end session\n")

            # Interactive loop
            while True:
                try:
                    # Get user input
                    user_input = input("\n[YOU]: ").strip()

                    # Check for exit commands
                    if user_input.lower() in ["exit", "quit", "q"]:
                        logger.info("Ending consultation session")
                        break

                    if not user_input:
                        continue

                    # Send query to agent
                    logger.info("\n[PYRO SPECIALIST]: Processing...\n")
                    await client.query(user_input)

                    # Receive and display response
                    async for message in client.receive_response():
                        if hasattr(message, 'content'):
                            # Display assistant messages
                            if hasattr(message.content, '__iter__'):
                                for block in message.content:
                                    if hasattr(block, 'text'):
                                        print(block.text)
                            elif hasattr(message.content, 'text'):
                                print(message.content.text)

                except KeyboardInterrupt:
                    logger.info("\n\nInterrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error during consultation: {e}")
                    logger.exception(e)

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        logger.exception(e)
        sys.exit(1)


async def run_pyro_specialist_single_query(prompt: str):
    """
    Run a single query against the pyro specialist agent.

    Useful for pipeline integration where other agents invoke this specialist
    for specific design consultations.

    Args:
        prompt: The design request or question

    Returns:
        Response from the agent
    """
    logger.info(f"Processing single query: {prompt[:100]}...")

    options = create_pyro_specialist_options()

    try:
        from claude_agent_sdk import query

        # Run single query
        response_text = []
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, 'content'):
                if hasattr(message.content, '__iter__'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            response_text.append(block.text)
                elif hasattr(message.content, 'text'):
                    response_text.append(message.content.text)

        return "\n".join(response_text)

    except Exception as e:
        logger.error(f"Single query failed: {e}")
        logger.exception(e)
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the pyro specialist agent."""
    # Check if running as single query or interactive mode
    if len(sys.argv) > 1:
        # Single query mode: python main.py "Design supernova effect"
        prompt = " ".join(sys.argv[1:])
        result = asyncio.run(run_pyro_specialist_single_query(prompt))
        print(result)
    else:
        # Interactive mode
        asyncio.run(run_pyro_specialist_interactive())


if __name__ == "__main__":
    main()
