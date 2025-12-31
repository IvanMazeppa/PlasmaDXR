#!/bin/bash
# Extract frames from solar activity video for ML vision analysis
#
# Options for frame extraction:
#   ALL:     4196 frames @ ~1.5MB each = ~6.3GB
#   EVERY_5: 839 frames  @ ~1.5MB each = ~1.3GB (recommended)
#   EVERY_10: 420 frames @ ~1.5MB each = ~630MB
#   EVERY_30: 140 frames @ ~1.5MB each = ~210MB (1fps keyframes)

VIDEO_PATH="/home/maz3ppa/projects/PlasmaDXR/assets/reference_images/star/Eruptions_20241008_Activity_2048p30.mp4"
OUTPUT_DIR="/home/maz3ppa/projects/PlasmaDXR/assets/reference_images/star/Eruptions_20241008_Activity_2048p30"

# Frame skip (1 = all frames, 5 = every 5th frame, etc.)
FRAME_SKIP=${1:-5}

# JPEG quality (2 = highest quality, 31 = lowest)
# 2-5 is visually lossless for ML purposes
JPEG_QUALITY=2

echo "=============================================="
echo "Solar Activity Frame Extraction"
echo "=============================================="
echo "Input:       $VIDEO_PATH"
echo "Output:      $OUTPUT_DIR"
echo "Frame skip:  Every ${FRAME_SKIP}th frame"
echo "Quality:     JPEG qscale=$JPEG_QUALITY (highest)"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get video info
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$VIDEO_PATH" 2>/dev/null || echo "4196")
EXTRACT_FRAMES=$((TOTAL_FRAMES / FRAME_SKIP))

echo "Total frames in video: ~4196"
echo "Frames to extract:     ~$EXTRACT_FRAMES"
echo ""
echo "Starting extraction..."

# Extract frames with ffmpeg
# -qscale:v 2 = highest quality JPEG (near lossless)
# -vf "select=not(mod(n\,$FRAME_SKIP))" = extract every Nth frame
# -vsync vfr = variable frame rate (needed with select filter)

if [ "$FRAME_SKIP" -eq 1 ]; then
    # Extract all frames
    ffmpeg -i "$VIDEO_PATH" \
        -qscale:v $JPEG_QUALITY \
        "$OUTPUT_DIR/frame_%05d.jpg" \
        -hide_banner -loglevel info
else
    # Extract every Nth frame
    ffmpeg -i "$VIDEO_PATH" \
        -vf "select=not(mod(n\,$FRAME_SKIP))" \
        -qscale:v $JPEG_QUALITY \
        -vsync vfr \
        "$OUTPUT_DIR/frame_%05d.jpg" \
        -hide_banner -loglevel info
fi

# Count extracted frames
EXTRACTED=$(ls -1 "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)

echo ""
echo "=============================================="
echo "Extraction Complete!"
echo "=============================================="
echo "Frames extracted: $EXTRACTED"
echo "Total size:       $TOTAL_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Sample files:"
ls -la "$OUTPUT_DIR" | head -5
echo "..."
ls -la "$OUTPUT_DIR" | tail -3
