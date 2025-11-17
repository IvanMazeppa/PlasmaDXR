#!/usr/bin/env python3
"""
Screenshot Conversion Tool for PlasmaDX-Clean
Automatically converts F2 BMP screenshots to compressed PNG for agent analysis.

Usage:
    python tools/convert_screenshots.py                    # Convert all BMPs in build/bin/Debug/screenshots/
    python tools/convert_screenshots.py --watch             # Watch directory and auto-convert new screenshots
    python tools/convert_screenshots.py --quality 85        # JPEG quality (1-100)
    python tools/convert_screenshots.py --format jpg        # Output format (png or jpg)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed")
    print("Install with: pip install Pillow")
    sys.exit(1)


def convert_screenshot(bmp_path: Path, output_format: str = "png", quality: int = 85) -> Path:
    """Convert BMP screenshot to compressed format."""

    # Output path (same directory, different extension)
    output_path = bmp_path.with_suffix(f".{output_format}")

    # Skip if already converted and output is newer
    if output_path.exists() and output_path.stat().st_mtime > bmp_path.stat().st_mtime:
        return output_path

    try:
        # Load BMP
        img = Image.open(bmp_path)

        # Convert to RGB if needed (some BMPs have alpha)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Create white background
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save with compression
        if output_format == "png":
            img.save(output_path, "PNG", optimize=True, compress_level=9)
        elif output_format == "jpg":
            img.save(output_path, "JPEG", quality=quality, optimize=True)

        # Report compression ratio
        original_size = bmp_path.stat().st_size
        compressed_size = output_path.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100

        print(f"✅ {bmp_path.name} → {output_path.name} ({ratio:.1f}% smaller)")
        return output_path

    except Exception as e:
        print(f"❌ ERROR converting {bmp_path.name}: {e}")
        return None


def convert_all_screenshots(screenshot_dir: Path, output_format: str, quality: int) -> int:
    """Convert all BMP screenshots in directory."""

    bmp_files = list(screenshot_dir.glob("*.bmp"))

    if not bmp_files:
        print(f"No BMP files found in {screenshot_dir}")
        return 0

    print(f"\nFound {len(bmp_files)} BMP screenshot(s)")
    print(f"Output format: {output_format.upper()}")
    if output_format == "jpg":
        print(f"JPEG quality: {quality}")
    print()

    converted_count = 0
    for bmp_path in sorted(bmp_files):
        result = convert_screenshot(bmp_path, output_format, quality)
        if result:
            converted_count += 1

    print(f"\n✅ Converted {converted_count}/{len(bmp_files)} screenshot(s)")
    return converted_count


def watch_directory(screenshot_dir: Path, output_format: str, quality: int):
    """Watch directory for new BMP files and auto-convert."""

    print(f"👁️  Watching {screenshot_dir} for new screenshots...")
    print(f"Output format: {output_format.upper()}")
    if output_format == "jpg":
        print(f"JPEG quality: {quality}")
    print("Press Ctrl+C to stop\n")

    # Track already-seen files
    seen_files = set(screenshot_dir.glob("*.bmp"))

    try:
        while True:
            # Check for new files
            current_files = set(screenshot_dir.glob("*.bmp"))
            new_files = current_files - seen_files

            if new_files:
                for bmp_path in sorted(new_files):
                    # Wait a moment to ensure file is fully written
                    time.sleep(0.5)
                    convert_screenshot(bmp_path, output_format, quality)
                seen_files = current_files

            time.sleep(1)  # Poll every second

    except KeyboardInterrupt:
        print("\n\n👋 Stopped watching")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Convert PlasmaDX-Clean F2 screenshots from BMP to compressed formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/convert_screenshots.py                    # Convert all existing BMPs
  python tools/convert_screenshots.py --watch             # Auto-convert new screenshots
  python tools/convert_screenshots.py --format jpg --quality 90  # High-quality JPEG
        """
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="build/bin/Debug/screenshots",
        help="Screenshot directory (default: build/bin/Debug/screenshots)"
    )

    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output format (default: png)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85, only used for jpg format)"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch directory for new screenshots and auto-convert"
    )

    args = parser.parse_args()

    # Resolve screenshot directory (relative to project root)
    project_root = Path(__file__).parent.parent
    screenshot_dir = project_root / args.dir

    if not screenshot_dir.exists():
        print(f"ERROR: Screenshot directory not found: {screenshot_dir}")
        return 1

    print("=" * 80)
    print("PlasmaDX-Clean Screenshot Converter")
    print("=" * 80)
    print(f"Screenshot directory: {screenshot_dir}")

    if args.watch:
        watch_directory(screenshot_dir, args.format, args.quality)
    else:
        convert_all_screenshots(screenshot_dir, args.format, args.quality)

    return 0


if __name__ == "__main__":
    sys.exit(main())
