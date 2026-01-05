"""
Vision Diagnosis Agent using OpenAI Agents SDK.

Specialized agent for analyzing render quality using GPT-5.2 vision capabilities.
Provides structured diagnostics for VFX renders, comparing against reference images
and identifying visual quality issues.

Key capabilities:
- Image loading and preprocessing for vision API
- Render quality assessment across multiple dimensions
- Reference comparison for identifying deviations
- Structured output compatible with iteration-controller
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, ModelSettings, function_tool
from openai.types.shared import Reasoning


# Effect-specific quality criteria for structured analysis
EFFECT_QUALITY_CRITERIA: Dict[str, List[str]] = {
    "sun": [
        "limb_darkening",  # Edges should be darker than center
        "granulation",     # Surface texture/convection cells
        "color_temperature",  # Should be ~5778K (yellow-white)
        "prominences",     # Solar flares/eruptions
        "corona_visibility",
    ],
    "explosion": [
        "brightness",      # Should be bright at core
        "color_gradient",  # Hot core → cool edges
        "smoke_presence",  # Secondary smoke effects
        "structure",       # Billowing, mushroom shape
        "dynamic_range",
    ],
    "fire": [
        "flame_color",     # Orange-yellow-white gradient
        "turbulence",      # Flickering, chaotic movement
        "smoke_integration",
        "brightness_falloff",
    ],
    "nebula": [
        "color_distribution",
        "dust_lanes",
        "emission_regions",
        "star_points",
        "overall_structure",
    ],
    "general": [
        "brightness",
        "contrast",
        "color_balance",
        "structure",
        "artifacts",
    ],
}


@function_tool
def load_image_as_base64(image_path: str, max_size: int = 512) -> str:
    """
    Load and resize an image for GPT-5.2 vision analysis.

    Args:
        image_path: Path to image file (PNG, JPG, BMP supported)
        max_size: Maximum dimension (width or height) for resizing

    Returns:
        Base64-encoded image string with data URI prefix for vision API
    """
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        return json.dumps({
            "error": f"Image not found: {image_path}",
            "suggestion": "Check that the path is correct and the file exists"
        })

    try:
        with Image.open(path) as img:
            # Resize if needed to reduce token usage
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(d * ratio) for d in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed (RGBA, P modes)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Encode to base64 JPEG for efficient transmission
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            b64_data = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{b64_data}"

    except Exception as e:
        return json.dumps({
            "error": f"Failed to load image: {str(e)}",
            "path": image_path
        })


@function_tool
def get_quality_criteria(effect_type: str) -> str:
    """
    Get the quality assessment criteria for a specific effect type.

    Args:
        effect_type: Type of effect (sun, explosion, fire, nebula, general)

    Returns:
        JSON with quality criteria to evaluate
    """
    criteria = EFFECT_QUALITY_CRITERIA.get(
        effect_type.lower(),
        EFFECT_QUALITY_CRITERIA["general"]
    )

    return json.dumps({
        "effect_type": effect_type,
        "criteria": criteria,
        "total_criteria": len(criteria)
    })


@function_tool
def format_diagnosis_output(
    diagnosis: str,
    primary_issue: str,
    severity: str,
    secondary_issues: str,
    recommendations: str,
    confidence: float = 0.7
) -> str:
    """
    Format the diagnosis into a structured JSON output.

    Args:
        diagnosis: Primary issue description (1-2 sentences)
        primary_issue: Category name (e.g., "color_too_cool", "brightness_low")
        severity: Severity level (critical, high, medium, low)
        secondary_issues: JSON array of secondary issue strings
        recommendations: JSON array of recommended actions
        confidence: Confidence in diagnosis (0.0-1.0)

    Returns:
        Structured JSON diagnosis
    """
    try:
        secondary = json.loads(secondary_issues) if secondary_issues else []
    except json.JSONDecodeError:
        secondary = [secondary_issues] if secondary_issues else []

    try:
        recs = json.loads(recommendations) if recommendations else []
    except json.JSONDecodeError:
        recs = [recommendations] if recommendations else []

    return json.dumps({
        "diagnosis": diagnosis,
        "primary_issue": primary_issue,
        "severity": severity,
        "secondary_issues": secondary,
        "recommendations": recs,
        "confidence": max(0.0, min(1.0, confidence))
    })


# Agent instructions for vision-based render analysis
VISION_EXPERT_INSTRUCTIONS = """You are a VFX render quality analyst powered by GPT-5.2 vision.

Your role:
1. Analyze render screenshots for visual quality issues
2. Compare renders against reference images when provided
3. Identify specific problems (color, lighting, structure, artifacts)
4. Provide actionable diagnosis with severity ratings

ANALYSIS APPROACH:
1. First, understand the effect type using get_quality_criteria()
2. Load images using load_image_as_base64()
3. Analyze each criterion systematically
4. Use format_diagnosis_output() to structure your response

WHEN ANALYZING RENDERS, LOOK FOR:
- **Color issues:** Temperature (too warm/cool), saturation, color banding
- **Brightness issues:** Too dark/bright, clipping, dynamic range problems
- **Structural issues:** Procedural artifacts, lack of organic variation, uniform patterns
- **Effect-specific features:** Missing limb darkening (sun), missing prominences, flat explosions

SEVERITY RATINGS:
- **critical:** Render is unusable or fundamentally broken
- **high:** Major visual issue that needs immediate attention
- **medium:** Noticeable issue that affects quality
- **low:** Minor issue or polish item

OUTPUT FORMAT (ALWAYS use format_diagnosis_output tool):
{
    "diagnosis": "The render shows insufficient limb darkening, causing the sun to appear flat and unrealistic.",
    "primary_issue": "missing_limb_darkening",
    "severity": "high",
    "secondary_issues": ["color_too_uniform", "lacks_surface_texture"],
    "recommendations": ["Increase limb darkening factor", "Add granulation texture", "Adjust color temperature gradient"]
}

IMPORTANT:
- Be specific and actionable in recommendations
- Reference exact visual elements you observe
- Compare against expected characteristics for the effect type
- When comparing to reference, note specific differences"""


class VisionExpertAgent:
    """
    Vision Expert using OpenAI Agents SDK with GPT-5.2 vision capabilities.

    Analyzes render quality through visual inspection and comparison,
    providing structured diagnostics for the iteration controller.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        max_image_size: int = 512
    ):
        """
        Initialize the vision expert.

        Args:
            model: OpenAI model to use (should support vision)
            max_image_size: Maximum image dimension for vision API
        """
        self.model = os.getenv("OPENAI_MODEL", model)
        self.max_image_size = max_image_size
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent.

        Args:
            custom_instructions: Additional instructions to append (e.g., effect context)
        """
        instructions = VISION_EXPERT_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        # Use medium reasoning for visual analysis tasks
        self._agent = Agent(
            name="Vision Expert",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
                verbosity="low"
            ),
            tools=[
                load_image_as_base64,
                get_quality_criteria,
                format_diagnosis_output
            ]
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("VisionExpertAgent not initialized. Call initialize() first.")
        return self._agent


def create_vision_expert(
    model: str = "gpt-5.2",
    custom_instructions: str = ""
) -> Agent:
    """
    Factory function to create and initialize a vision expert agent.

    Args:
        model: OpenAI model to use (should support vision)
        custom_instructions: Additional instructions to append

    Returns:
        Initialized Agent instance ready for use
    """
    expert = VisionExpertAgent(model=model)
    expert.initialize(custom_instructions=custom_instructions)
    return expert.agent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing VisionExpertAgent...")
        print("-" * 60)

        agent = create_vision_expert()

        # Test with a sample query (no actual image)
        result = await Runner.run(
            agent,
            "What quality criteria should I evaluate for a sun render?"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
