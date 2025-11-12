"""
Visual Quality Assessment Tool for PlasmaDX Volumetric Rendering

This tool provides qualitative AI analysis of rendering quality based on
the Visual Quality Rubric. It evaluates 7 key dimensions:

1. Volumetric Depth & Atmosphere
2. Lighting Quality & Rim Lighting
3. Temperature Gradient & Blackbody Emission
4. RTXDI Light Sampling Quality
5. Shadow Quality
6. Anisotropic Scattering & Phase Function
7. Performance & Temporal Stability

The tool is designed to be used by Claude Code's vision models (GPT-4V or
Claude 3.5 Sonnet with vision) to provide human-like qualitative assessment.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class VisualQualityAssessment:
    """
    Provides structured context for AI vision models to assess rendering quality.

    This class doesn't do the analysis itself - instead, it provides:
    1. The quality rubric to guide AI analysis
    2. Reference images for comparison
    3. Structured output format
    4. Historical tracking
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.rubric_path = project_root / "agents" / "rtxdi-quality-analyzer" / "VISUAL_QUALITY_RUBRIC.md"
        self.references_dir = project_root / "screenshots" / "reference"
        self.annotations_dir = project_root / "screenshots" / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def load_rubric(self) -> str:
        """Load the visual quality rubric markdown."""
        if self.rubric_path.exists():
            return self.rubric_path.read_text(encoding='utf-8')
        else:
            return "Error: Visual quality rubric not found. Please create VISUAL_QUALITY_RUBRIC.md"

    def load_reference_images(self) -> Dict[str, Path]:
        """Load paths to reference images (golden standard, good, issues, failures)."""
        references = {}

        if not self.references_dir.exists():
            return references

        # Scan reference subdirectories
        for category in ['golden_standard', 'good', 'issues', 'failures']:
            category_dir = self.references_dir / category
            if category_dir.exists():
                references[category] = list(category_dir.glob("*.png")) + list(category_dir.glob("*.bmp"))

        return references

    def load_annotations(self, screenshot_name: str) -> Optional[Dict[str, Any]]:
        """Load existing quality annotations for a screenshot."""
        annotation_file = self.annotations_dir / f"{screenshot_name}.json"

        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                return json.load(f)

        return None

    def save_annotation(self, screenshot_name: str, assessment: Dict[str, Any]):
        """Save quality assessment as JSON annotation."""
        annotation_file = self.annotations_dir / f"{screenshot_name}.json"

        # Add metadata
        assessment['timestamp'] = datetime.now().isoformat()
        assessment['screenshot'] = screenshot_name

        with open(annotation_file, 'w') as f:
            json.dump(assessment, f, indent=2)

    def validate_metadata(self, metadata: Dict[str, Any], screenshot_path: str) -> Dict[str, str]:
        """
        Validate metadata structure and detect common issues.

        Returns dict of warnings (empty if no issues).
        """
        warnings = {}

        # Check schema version
        if metadata.get('schema_version') != "2.0":
            warnings['schema'] = f"Metadata schema is {metadata.get('schema_version', 'unknown')}, expected 2.0"

        # Validate rendering section
        if 'rendering' in metadata:
            r = metadata['rendering']

            # Check lights (CRITICAL - prevents phantom "0 lights" bug)
            lights = r.get('lights', {})
            light_count = lights.get('count', 0)

            if light_count == 0:
                # Extra validation: check if light_list exists but count is wrong
                light_list = lights.get('light_list', [])
                if len(light_list) > 0:
                    warnings['lights_mismatch'] = f"CRITICAL: Metadata shows 0 lights but light_list has {len(light_list)} entries! Using light_list length instead."
                    # Auto-fix by returning corrected count
                    warnings['lights_corrected'] = str(len(light_list))
                else:
                    warnings['lights_zero'] = "WARNING: Metadata shows 0 lights. Verify if accurate or metadata bug."

            # Check nested structure presence (CRITICAL - prevents crashes)
            required_nested = ['rtxdi', 'shadows', 'lights']
            for key in required_nested:
                if key not in r:
                    warnings[f'missing_{key}'] = f"Missing required nested structure: rendering.{key}"

        # Validate performance section
        if 'performance' in metadata:
            p = metadata['performance']
            fps = p.get('fps', 0)

            # Sanity check: FPS too low or too high
            if fps < 1 and fps != 0:
                warnings['fps_suspicious'] = f"FPS is {fps:.1f} - suspiciously low, check if metadata is stale"
            elif fps > 300:
                warnings['fps_unrealistic'] = f"FPS is {fps:.1f} - suspiciously high, check if metadata is accurate"

        return warnings

    def prepare_analysis_context(self, screenshot_path: str,
                                  comparison_mode: bool = False,
                                  before_path: Optional[str] = None) -> str:
        """
        Prepare the context for AI vision model analysis.

        This returns a structured prompt that guides the AI to analyze
        the screenshot according to our quality rubric.
        """
        rubric = self.load_rubric()
        references = self.load_reference_images()

        # Build the analysis prompt
        context = f"""# PlasmaDX Volumetric Rendering Quality Assessment

## Your Task

You are analyzing a screenshot from PlasmaDX, a volumetric black hole accretion
disk renderer using DXR 1.1 ray tracing and 3D Gaussian splatting.

**Screenshot to analyze:** {screenshot_path}
"""

        if comparison_mode and before_path:
            context += f"""
**Comparison mode:** You are comparing TWO screenshots to assess improvements/regressions.
  - Before: {before_path}
  - After:  {screenshot_path}

Focus on identifying what changed and whether it's an improvement.
"""

        context += f"""

## Quality Rubric

{rubric}

## Available Reference Images

"""

        if references:
            for category, images in references.items():
                context += f"**{category.replace('_', ' ').title()}** ({len(images)} images):\n"
                for img in images[:3]:  # Show first 3 of each category
                    context += f"  - {img.name}\n"
                context += "\n"
        else:
            context += "No reference images available yet. Analysis based on rubric only.\n\n"

        context += """

## Required Output Format

Please provide your assessment in the following structure:

```
VOLUMETRIC RENDERING QUALITY ASSESSMENT

Screenshot: {filename}
Date: {date}
Overall Grade: {letter grade} ({score}/100)

=== CRITICAL DIMENSIONS ===

1. Volumetric Depth & Atmosphere: {EXCELLENT/GOOD/FAIR/POOR/MISSING} ({score}/100)
   ✅ {what's working well}
   ⚠️  {minor issues}
   ❌ {critical problems}

2. Lighting Quality & Rim Lighting: {rating} ({score}/100)
   {assessment}

=== HIGH PRIORITY ===

3. Temperature Gradient: {rating} ({score}/100)
   {assessment}

4. RTXDI Sampling Quality: {rating} ({score}/100)
   {assessment}

=== MEDIUM PRIORITY ===

5. Shadow Quality: {rating} ({score}/100)
   {assessment}

6. Anisotropic Scattering: {rating} ({score}/100)
   {assessment}

7. Temporal Stability: {rating} ({score}/100)
   {assessment}

=== KEY OBSERVATIONS ===

{2-3 paragraphs describing what you see}

=== COMPARISON TO GOALS ===

{How well does this match the "golden standard" aesthetic?}

=== ACTIONABLE RECOMMENDATIONS ===

1. {specific improvement suggestion}
2. {specific improvement suggestion}
3. {specific improvement suggestion}

=== TECHNICAL NOTES ===

{Any technical observations about RTXDI, RT lighting, etc.}
```

## Analysis Guidelines

1. **Focus on visual quality**, not technical metrics
2. **Compare to the rubric's "what we want" sections**
3. **Identify anti-patterns** listed in the rubric
4. **Be specific** - don't just say "lighting looks good", describe what makes it good
5. **Provide actionable feedback** - suggest concrete improvements

Begin your analysis:
"""

        return context


def assess_screenshot_quality(screenshot_path: str,
                               project_root: str,
                               comparison_before: Optional[str] = None,
                               save_annotation: bool = True) -> str:
    """
    Main entry point for visual quality assessment.

    This function prepares the context for AI analysis but does NOT
    perform the analysis itself. The actual analysis should be done
    by Claude Code's vision capabilities.

    Args:
        screenshot_path: Path to the screenshot to analyze
        project_root: Root directory of PlasmaDX project
        comparison_before: Optional path to "before" screenshot for comparison
        save_annotation: Whether to save the assessment as JSON

    Returns:
        Structured context prompt for AI vision model analysis
    """
    assessor = VisualQualityAssessment(Path(project_root))

    comparison_mode = comparison_before is not None

    context = assessor.prepare_analysis_context(
        screenshot_path,
        comparison_mode=comparison_mode,
        before_path=comparison_before
    )

    # Phase 1: Load metadata if available and append config-specific context
    metadata_path = Path(screenshot_path + ".json")
    if metadata_path.exists():
        try:
            metadata = json.load(open(metadata_path, 'r'))

            # CRITICAL: Validate metadata structure first
            validation_warnings = assessor.validate_metadata(metadata, screenshot_path)

            # Append metadata section to context
            context += "\n\n## SCREENSHOT METADATA (Phase 1 Enhancement)\n\n"

            # Show validation warnings if any (CRITICAL for phantom bug prevention)
            if validation_warnings:
                context += "**⚠️  METADATA VALIDATION WARNINGS:**\n"
                for warning_type, warning_msg in validation_warnings.items():
                    if 'lights_corrected' in warning_type:
                        # Auto-fix applied, use corrected value
                        context += f"  - Auto-fixed: Using light count = {warning_msg} (from light_list length)\n"
                    else:
                        context += f"  - {warning_msg}\n"
                context += "\n"

            context += "The screenshot was captured with the following configuration:\n\n"

            # Rendering configuration
            if 'rendering' in metadata:
                r = metadata['rendering']
                context += "**Rendering Configuration:**\n"

                # RTXDI status (nested structure)
                rtxdi = r.get('rtxdi', {})
                rtxdi_enabled = rtxdi.get('enabled', False)
                m4_enabled = rtxdi.get('m4_enabled', False)
                m5_enabled = rtxdi.get('m5_enabled', False)

                if m5_enabled:
                    rtxdi_status = "M5 ENABLED"
                elif m4_enabled:
                    rtxdi_status = "M4 ONLY (M5 disabled)"
                elif rtxdi_enabled:
                    rtxdi_status = "ENABLED (legacy)"
                else:
                    rtxdi_status = "DISABLED"

                context += f"- RTXDI Status: `{rtxdi_status}`\n"

                if rtxdi_enabled and not m5_enabled:
                    context += "  ⚠️ **CRITICAL:** M5 temporal accumulation is disabled - expect visible patchwork pattern!\n"

                context += f"- Temporal Blend Factor: `{rtxdi.get('temporal_blend_factor', 0.0):.3f}`\n"

                # Shadows (nested structure)
                shadows = r.get('shadows', {})
                shadow_rays_per_light = shadows.get('rays_per_light', 1)
                context += f"- Shadow Rays Per Light: `{shadow_rays_per_light}`\n"

                if shadow_rays_per_light == 1:
                    context += "  ℹ️ Using Performance preset (1-ray + temporal filtering)\n"

                # CRITICAL FIX: Lights are nested under r['lights']['count'], not r['light_count']
                lights = r.get('lights', {})
                light_count = lights.get('count', 0)
                context += f"- Light Count: `{light_count}` lights\n"

                # Validation: Warn if light count seems wrong
                if light_count == 0:
                    context += "  ⚠️ **CRITICAL VALIDATION WARNING:** Metadata shows 0 lights. Verify if this is accurate or a metadata bug!\n"
                    context += "  ⚠️ Check metadata JSON file directly before concluding lights are disabled.\n"

                # Physical effects (nested structure)
                physical_effects = metadata.get('physical_effects', {})
                phase_function = physical_effects.get('phase_function', {})
                context += f"- Phase Function: `{'ENABLED' if phase_function.get('enabled', False) else 'DISABLED'}`\n"

                # Feature status (nested structure)
                feature_status = metadata.get('feature_status', {}).get('working', {})
                context += f"- Shadow Rays: `{'ENABLED' if feature_status.get('shadow_rays', False) else 'DISABLED'}`\n"
                context += f"- In-Scattering: `{'ENABLED (deprecated)' if metadata.get('feature_status', {}).get('deprecated', {}).get('in_scattering', False) else 'DISABLED'}`\n\n"

            # Performance metrics
            if 'performance' in metadata:
                p = metadata['performance']
                context += "**Performance Metrics:**\n"
                context += f"- FPS: `{p.get('fps', 0):.1f}`\n"
                context += f"- Frame Time: `{p.get('frame_time_ms', 0):.2f}` ms\n\n"

            # Camera state
            if 'camera' in metadata:
                c = metadata['camera']
                context += "**Camera State:**\n"
                context += f"- Position: `({c.get('position', [0,0,0])[0]:.1f}, {c.get('position', [0,0,0])[1]:.1f}, {c.get('position', [0,0,0])[2]:.1f})`\n"
                context += f"- Distance: `{c.get('distance', 0):.1f}` units\n"
                context += f"- Height: `{c.get('height', 0):.1f}` units\n\n"

            # Configuration-specific recommendations
            context += "## CONFIG-SPECIFIC RECOMMENDATIONS\n\n"
            context += "Based on the captured metadata, provide SPECIFIC recommendations that reference:\n"
            context += "1. **Exact config values** (e.g., `rtxdi_m5_enabled: false`)\n"
            context += "2. **File locations** where changes should be made\n"
            context += "3. **Expected improvements** with quantitative estimates\n\n"

            context += "Example format:\n"
            context += "> Your RTXDI M5 is disabled (`rtxdi_m5_enabled: false` in metadata).\n"
            context += "> Enable via:\n"
            context += ">   - ImGui: Check 'RTXDI M5 Temporal Accumulation'\n"
            context += ">   - Config: Set `rtxdi_temporal_accumulation: true` in configs/builds/Debug.json\n"
            context += ">   - Expected improvement: Patchwork pattern disappears in ~67ms (8 frames @ 120 FPS)\n\n"

        except Exception as e:
            context += f"\n\n⚠️ Metadata file found but failed to parse: {str(e)}\n\n"

    return context


def list_reference_images(project_root: str) -> str:
    """List all available reference images for comparison."""
    assessor = VisualQualityAssessment(Path(project_root))
    references = assessor.load_reference_images()

    if not references:
        return """No reference images found.

To create a reference library:
1. Create directories: screenshots/reference/{golden_standard,good,issues,failures}/
2. Copy your best renders to golden_standard/
3. Organize other screenshots by quality level

This will enable comparison-based quality assessment.
"""

    result = "REFERENCE IMAGE LIBRARY\n"
    result += "=" * 80 + "\n\n"

    for category, images in references.items():
        result += f"{category.replace('_', ' ').title()}: {len(images)} images\n"
        result += "-" * 40 + "\n"

        for img in images:
            # Try to load annotation
            annotation_file = Path(project_root) / "screenshots" / "annotations" / f"{img.stem}.json"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    grade = data.get('grade', 'N/A')
                    overall = data.get('overall_score', 'N/A')
                    result += f"  {img.name} - Grade: {grade} ({overall}/100)\n"
            else:
                result += f"  {img.name} - (no annotation)\n"

        result += "\n"

    return result


def list_annotations(project_root: str, limit: int = 10) -> str:
    """List recent quality assessments with scores."""
    annotations_dir = Path(project_root) / "screenshots" / "annotations"

    if not annotations_dir.exists():
        return "No quality assessments recorded yet."

    # Get all JSON files
    annotations = sorted(
        annotations_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    if not annotations:
        return "No quality assessments recorded yet."

    result = f"RECENT QUALITY ASSESSMENTS (showing {len(annotations)} most recent)\n"
    result += "=" * 80 + "\n\n"

    for ann_file in annotations:
        with open(ann_file, 'r') as f:
            data = json.load(f)

        screenshot = data.get('screenshot', ann_file.stem)
        grade = data.get('grade', 'N/A')
        overall = data.get('overall_score', 'N/A')
        timestamp = data.get('timestamp', 'unknown')

        result += f"{screenshot}\n"
        result += f"  Grade: {grade} ({overall}/100)\n"
        result += f"  Date: {timestamp}\n"

        if 'scores' in data:
            scores = data['scores']
            result += "  Scores:\n"
            for dim, score in scores.items():
                result += f"    {dim}: {score}/100\n"

        result += "\n"

    return result
