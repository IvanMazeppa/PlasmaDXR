"""
Phase 2B-3: Deterministic render quality checks.

Pure-Python image validation that runs BEFORE LLM vision evaluation.
Critical failures short-circuit the evaluation pipeline at $0 cost.

Tier 1 in the multi-grader evaluation pipeline:
  Tier 1: Deterministic checks (this module) - $0
  Tier 2: ML metrics (LPIPS/TOPIQ) - $0 (deferred)
  Tier 3: LLM vision evaluation - $0.01-0.05
"""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# PNG magic bytes and JPEG magic bytes
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

# Minimum render dimensions (anything smaller is likely degenerate)
MIN_RENDER_WIDTH = 64
MIN_RENDER_HEIGHT = 64

# File size threshold for "likely blank" heuristic (bytes per pixel)
# A solid-color PNG compresses extremely well; real renders have texture.
BLANK_BYTES_PER_PIXEL_THRESHOLD = 0.05

# Minimum file size for a render at 720p+ (heuristic, in bytes)
MIN_RENDER_FILE_SIZE = 15_000

# Script regex patterns for lighting
LIGHT_PATTERNS = re.compile(
    r"""
    \b(?:POINT|SUN|AREA|SPOT)\b      # Light type enums
    | add_light                        # bpy.ops light creation
    | emission_strength                # Emission shader strength
    | \.energy\s*=                     # Light energy property
    """,
    re.VERBOSE,
)

# Script regex patterns for camera
CAMERA_PATTERNS = re.compile(
    r"""
    \bcamera\b                         # General camera reference
    | \bCAMERA\b                       # Camera type enum
    | set_camera                       # Camera setup function
    | camera_location                  # Camera position
    | scene\.camera                    # Scene camera assignment
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class DeterministicCheckResult:
    """Result of Tier 1 deterministic quality checks."""

    passed: bool  # True if no critical failures
    critical_issues: List[str] = field(default_factory=list)  # Auto-fail issues
    warnings: List[str] = field(default_factory=list)  # Non-critical concerns
    checks_run: int = 0
    checks_passed: int = 0
    evaluation_tier: str = "deterministic"

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"Tier1[{status} {self.checks_passed}/{self.checks_run}]"]
        if self.critical_issues:
            parts.append(f"critical={self.critical_issues}")
        if self.warnings:
            parts.append(f"warnings={self.warnings}")
        return " ".join(parts)


def _parse_png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """
    Parse width and height from a PNG IHDR chunk.

    PNG structure: 8-byte signature, then chunks.
    IHDR is always the first chunk: 4-byte length, 4-byte type, then data.
    IHDR data starts with 4-byte width, 4-byte height (big-endian uint32).
    """
    # Need at least: 8 (sig) + 4 (len) + 4 (type) + 8 (w+h) = 24 bytes
    if len(data) < 24:
        return None
    # Verify IHDR chunk type at offset 12
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (width, height)


def _check_valid_image(render_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that the file is a valid PNG or JPEG with readable headers."""
    if not render_path.exists():
        return False, "Render file does not exist"

    if render_path.stat().st_size == 0:
        return False, "Render file is empty (0 bytes)"

    with open(render_path, "rb") as f:
        header = f.read(16)

    if header[:8] == PNG_MAGIC:
        return True, None
    if header[:3] == JPEG_MAGIC:
        return True, None

    return False, f"Unrecognized image format (header: {header[:8].hex()})"


def _check_minimum_size(render_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that the render meets minimum dimension requirements."""
    with open(render_path, "rb") as f:
        header = f.read(32)

    if header[:8] == PNG_MAGIC:
        dims = _parse_png_dimensions(header)
        if dims is None:
            return False, "Could not parse PNG dimensions from IHDR"
        width, height = dims
        if width < MIN_RENDER_WIDTH or height < MIN_RENDER_HEIGHT:
            return False, f"Render too small: {width}x{height} (min {MIN_RENDER_WIDTH}x{MIN_RENDER_HEIGHT})"
        return True, None

    # For JPEG, we can't easily parse dimensions without more header parsing.
    # Fall back to file size heuristic: a 64x64 JPEG is at least a few KB.
    file_size = render_path.stat().st_size
    if file_size < 1000:  # Less than 1KB is suspiciously small for any render
        return False, f"Render suspiciously small: {file_size} bytes"
    return True, None


def _check_not_blank(render_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Check that the render is not a solid color (black/white).

    Uses file-size heuristic: solid-color PNGs compress to tiny sizes.
    If PIL is available, samples pixels for a more precise check.
    """
    file_size = render_path.stat().st_size

    with open(render_path, "rb") as f:
        header = f.read(32)

    # Get dimensions for bytes-per-pixel calculation
    width, height = 1920, 1080  # Conservative default
    if header[:8] == PNG_MAGIC:
        dims = _parse_png_dimensions(header)
        if dims:
            width, height = dims

    pixel_count = width * height
    bytes_per_pixel = file_size / pixel_count if pixel_count > 0 else 0

    if bytes_per_pixel < BLANK_BYTES_PER_PIXEL_THRESHOLD:
        # Try PIL for precise check before declaring critical failure
        try:
            from PIL import Image

            img = Image.open(render_path)
            # Sample corners and center
            w, h = img.size
            sample_points = [
                (0, 0),
                (w - 1, 0),
                (0, h - 1),
                (w - 1, h - 1),
                (w // 2, h // 2),
                (w // 4, h // 4),
                (3 * w // 4, 3 * h // 4),
            ]
            pixels = [img.getpixel(p) for p in sample_points]

            # Check if all sampled pixels are near-identical
            # Convert to tuples of RGB (ignore alpha)
            rgb_pixels = []
            for p in pixels:
                if isinstance(p, (int, float)):
                    rgb_pixels.append((p, p, p))
                elif len(p) >= 3:
                    rgb_pixels.append(p[:3])
                else:
                    rgb_pixels.append((p[0], p[0], p[0]))

            # All pixels within 5 of each other = blank
            first = rgb_pixels[0]
            all_same = all(
                abs(px[0] - first[0]) <= 5
                and abs(px[1] - first[1]) <= 5
                and abs(px[2] - first[2]) <= 5
                for px in rgb_pixels
            )

            if all_same:
                avg = sum(first) / len(first)
                if avg < 10:
                    return False, "Render is blank (all black)"
                elif avg > 245:
                    return False, "Render is blank (all white)"
                else:
                    return False, f"Render is solid color (RGB ~{first})"
            return True, None  # PIL says it's not blank despite small file

        except ImportError:
            # No PIL — trust the heuristic
            return False, f"Render likely blank (bytes/pixel={bytes_per_pixel:.4f}, threshold={BLANK_BYTES_PER_PIXEL_THRESHOLD})"

    return True, None


def _check_reasonable_file_size(render_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that the render file is large enough to contain meaningful content."""
    file_size = render_path.stat().st_size
    if file_size < MIN_RENDER_FILE_SIZE:
        return False, f"Render file unusually small: {file_size} bytes (expected >{MIN_RENDER_FILE_SIZE} for 720p+)"
    return True, None


def _check_lights_in_script(script_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that the script creates at least one light source."""
    try:
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return True, None  # Can't read script, don't block on it

    if LIGHT_PATTERNS.search(script_text):
        return True, None

    return False, "No light sources found in script (searched for POINT/SUN/AREA/SPOT/emission_strength)"


def _check_camera_in_script(script_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that the script sets up a camera."""
    try:
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return True, None  # Can't read script, don't block on it

    if CAMERA_PATTERNS.search(script_text):
        return True, None

    return False, "No camera setup found in script"


def run_deterministic_checks(
    render_path: str,
    script_path: Optional[str] = None,
    effect_type: str = "auto",
) -> DeterministicCheckResult:
    """
    Run Tier 1 deterministic quality checks on a render.

    Critical checks (auto-fail if any fails):
      - render_valid_image: Valid PNG/JPEG header
      - render_minimum_size: Image >= 64x64
      - render_not_blank: Not all-black or all-white

    Warning checks (logged but don't block LLM eval):
      - render_reasonable_file_size: File > 15KB
      - lights_in_script: Script creates at least one light
      - camera_in_script: Script has camera setup

    Args:
        render_path: Path to the rendered image file.
        script_path: Optional path to the Blender script (for script-level checks).
        effect_type: Effect type hint (unused for now, reserved for future per-type thresholds).

    Returns:
        DeterministicCheckResult with pass/fail status and details.
    """
    result = DeterministicCheckResult(passed=True)
    render_file = Path(render_path)

    # === Critical checks (fast-fail) ===
    critical_checks = [
        ("render_valid_image", lambda: _check_valid_image(render_file)),
        ("render_minimum_size", lambda: _check_minimum_size(render_file)),
        ("render_not_blank", lambda: _check_not_blank(render_file)),
    ]

    for check_name, check_fn in critical_checks:
        result.checks_run += 1
        try:
            passed, issue = check_fn()
        except Exception as e:
            passed, issue = False, f"{check_name} error: {e}"

        if passed:
            result.checks_passed += 1
        else:
            result.critical_issues.append(f"[{check_name}] {issue}")
            result.passed = False
            logger.warning(f"Tier 1 CRITICAL: {check_name} failed: {issue}")
            # Fast-fail on first critical issue — no need to check further
            return result

    # === Warning checks (don't block) ===
    warning_checks = [
        ("render_reasonable_file_size", lambda: _check_reasonable_file_size(render_file)),
    ]

    # Script-level checks only if script_path provided
    if script_path:
        script_file = Path(script_path)
        if script_file.exists():
            warning_checks.extend([
                ("lights_in_script", lambda: _check_lights_in_script(script_file)),
                ("camera_in_script", lambda: _check_camera_in_script(script_file)),
            ])

    for check_name, check_fn in warning_checks:
        result.checks_run += 1
        try:
            passed, issue = check_fn()
        except Exception as e:
            passed, issue = False, f"{check_name} error: {e}"

        if passed:
            result.checks_passed += 1
        else:
            result.warnings.append(f"[{check_name}] {issue}")
            logger.info(f"Tier 1 WARNING: {check_name}: {issue}")

    return result
