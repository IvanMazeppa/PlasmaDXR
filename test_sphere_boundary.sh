#!/bin/bash
# Sphere Boundary Test Suite
# Tests multi-light system at various distances to isolate the ~300 unit boundary issue
# Expected: Lights work up to 300 units, fail beyond

set -e  # Exit on error

# Configuration
BUILD_DIR="build/DebugPIX"
EXE_NAME="PlasmaDX-Clean-PIX.exe"
CONFIG_DIR="configs/scenarios"
RESULTS_DIR="test_results/sphere_boundary_$(date +%Y%m%d_%H%M%S)"
FRAME_COUNT=180  # 3 seconds at 60 FPS
CAPTURE_FRAME=120  # Capture at frame 120 (2 seconds in)

# Test distances (units)
DISTANCES=(50 100 200 300 400 500 1000)
LIGHT_COLORS=("RED" "ORANGE" "YELLOW" "GREEN" "CYAN" "BLUE" "MAGENTA")

# Create results directory structure
mkdir -p "$RESULTS_DIR/screenshots"
mkdir -p "$RESULTS_DIR/captures"
mkdir -p "$RESULTS_DIR/buffer_dumps"
mkdir -p "$RESULTS_DIR/logs"

echo "================================================================"
echo "SPHERE BOUNDARY TEST SUITE"
echo "================================================================"
echo "Testing multi-light system at distances: ${DISTANCES[@]} units"
echo "Results directory: $RESULTS_DIR"
echo "================================================================"
echo ""

# Summary file
SUMMARY_FILE="$RESULTS_DIR/test_summary.txt"
cat > "$SUMMARY_FILE" << EOF
SPHERE BOUNDARY TEST SUITE - $(date)
================================================================

TEST MATRIX:
EOF

for i in "${!DISTANCES[@]}"; do
    dist="${DISTANCES[$i]}"
    color="${LIGHT_COLORS[$i]}"
    printf "  %4d units - %7s light - " "$dist" "$color" >> "$SUMMARY_FILE"
    if [ "$dist" -le 200 ]; then
        echo "Expected: PASS" >> "$SUMMARY_FILE"
    elif [ "$dist" -eq 300 ]; then
        echo "Expected: BOUNDARY (critical test)" >> "$SUMMARY_FILE"
    else
        echo "Expected: FAIL" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << EOF

================================================================
TEST RESULTS:
================================================================

EOF

# Run each test
for i in "${!DISTANCES[@]}"; do
    dist="${DISTANCES[$i]}"
    color="${LIGHT_COLORS[$i]}"
    config_file="${CONFIG_DIR}/sphere_test_$(printf '%03d' $dist)u.json"

    echo ""
    echo "----------------------------------------------------------------"
    echo "TEST $((i+1))/7: ${color} LIGHT AT ${dist} UNITS"
    echo "----------------------------------------------------------------"
    echo "Config: $config_file"

    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        echo "SKIP" >> "$SUMMARY_FILE"
        continue
    fi

    # Run the application
    echo "Running application for 3 seconds..."
    start_time=$(date +%s)

    # Run in background with timeout
    timeout 10s "$BUILD_DIR/$EXE_NAME" --config="$config_file" > "$RESULTS_DIR/logs/sphere_${dist}u.log" 2>&1 &
    APP_PID=$!

    # Wait for completion or timeout
    wait $APP_PID 2>/dev/null
    EXIT_CODE=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Duration: ${duration}s (exit code: $EXIT_CODE)"

    # Move PIX capture if it exists
    if ls PIX/Captures/sphere_${dist}u_*.wpix 1> /dev/null 2>&1; then
        mv PIX/Captures/sphere_${dist}u_*.wpix "$RESULTS_DIR/captures/"
        echo "PIX capture saved"
    else
        echo "WARNING: No PIX capture found"
    fi

    # Move buffer dumps if they exist
    if [ -d "PIX/buffer_dumps" ]; then
        mkdir -p "$RESULTS_DIR/buffer_dumps/sphere_${dist}u"
        mv PIX/buffer_dumps/* "$RESULTS_DIR/buffer_dumps/sphere_${dist}u/" 2>/dev/null || true
        echo "Buffer dumps saved"
    fi

    # Record result in summary
    printf "  %4d units - %7s light - " "$dist" "$color" >> "$SUMMARY_FILE"
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "COMPLETED (review captures for visual confirmation)" >> "$SUMMARY_FILE"
    elif [ "$EXIT_CODE" -eq 124 ]; then
        echo "TIMEOUT (application hung)" >> "$SUMMARY_FILE"
    else
        echo "CRASHED (exit code: $EXIT_CODE)" >> "$SUMMARY_FILE"
    fi

    # Brief pause between tests
    echo "Cooling down for 2 seconds..."
    sleep 2
done

echo ""
echo "================================================================"
echo "TEST SUITE COMPLETE"
echo "================================================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "NEXT STEPS:"
echo "1. Review PIX captures in: $RESULTS_DIR/captures/"
echo "2. Check buffer dumps in: $RESULTS_DIR/buffer_dumps/"
echo "3. Review logs in: $RESULTS_DIR/logs/"
echo "4. Read summary: $RESULTS_DIR/test_summary.txt"
echo ""
echo "ANALYSIS CHECKLIST:"
echo "  [ ] Do 50-100 unit lights work correctly?"
echo "  [ ] Does 200 unit light show degradation?"
echo "  [ ] What happens at exactly 300 units (boundary)?"
echo "  [ ] Do 400-1000 unit lights fail completely?"
echo "  [ ] Is there a hard cutoff or gradual falloff?"
echo "  [ ] Are shadow rays working at all distances?"
echo "================================================================"

# Display summary
cat "$SUMMARY_FILE"
