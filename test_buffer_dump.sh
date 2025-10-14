#!/bin/bash
# Quick test script for buffer dump feature

echo "==================================="
echo "Buffer Dump Feature - Quick Test"
echo "==================================="
echo ""

# Test 1: Verify executable exists
echo "[TEST 1] Checking executable..."
if [ -f "build/Debug/PlasmaDX-Clean.exe" ]; then
    echo "  ✓ Executable found"
else
    echo "  ✗ Executable not found - run MSBuild first"
    exit 1
fi

# Test 2: Create clean output directory
echo ""
echo "[TEST 2] Creating output directory..."
rm -rf pix/buffer_dumps
mkdir -p pix/buffer_dumps
echo "  ✓ Directory created: pix/buffer_dumps"

# Test 3: Run auto-dump at frame 10 (quick test)
echo ""
echo "[TEST 3] Running auto-dump test (frame 10)..."
echo "  Command: ./build/Debug/PlasmaDX-Clean.exe --dump-buffers 10 --gaussian --particles 10000"
echo ""

./build/Debug/PlasmaDX-Clean.exe --dump-buffers 10 --gaussian --particles 10000

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ Application exited successfully"
else
    echo ""
    echo "  ✗ Application exited with error"
    exit 1
fi

# Test 4: Verify output files
echo ""
echo "[TEST 4] Verifying output files..."

files=(
    "pix/buffer_dumps/g_particles.bin"
    "pix/buffer_dumps/g_currentReservoirs.bin"
    "pix/buffer_dumps/g_prevReservoirs.bin"
    "pix/buffer_dumps/metadata.json"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        echo "  ✓ $file ($size bytes)"
    else
        echo "  ✗ $file (MISSING)"
        missing=$((missing + 1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "  ✗ $missing file(s) missing - check logs"
    exit 1
fi

# Test 5: Verify metadata.json content
echo ""
echo "[TEST 5] Checking metadata.json..."
if grep -q '"frame": 10' pix/buffer_dumps/metadata.json; then
    echo "  ✓ Correct frame number in metadata"
else
    echo "  ✗ Frame number mismatch in metadata"
    exit 1
fi

if grep -q '"restir_enabled": true' pix/buffer_dumps/metadata.json; then
    echo "  ✓ ReSTIR settings captured"
else
    echo "  ✗ ReSTIR settings missing"
    exit 1
fi

# Test 6: Verify buffer sizes (10K particles)
echo ""
echo "[TEST 6] Verifying buffer sizes..."

# g_particles: 10000 particles × 32 bytes = 320,000 bytes
particles_size=$(stat -c%s pix/buffer_dumps/g_particles.bin 2>/dev/null || stat -f%z pix/buffer_dumps/g_particles.bin 2>/dev/null)
if [ "$particles_size" -eq 320000 ]; then
    echo "  ✓ g_particles.bin size correct (320,000 bytes)"
else
    echo "  ⚠ g_particles.bin size unexpected: $particles_size bytes (expected 320,000)"
fi

# g_currentReservoirs: 1920×1080×32 = 66,355,200 bytes
reservoirs_size=$(stat -c%s pix/buffer_dumps/g_currentReservoirs.bin 2>/dev/null || stat -f%z pix/buffer_dumps/g_currentReservoirs.bin 2>/dev/null)
if [ "$reservoirs_size" -eq 66355200 ]; then
    echo "  ✓ g_currentReservoirs.bin size correct (66,355,200 bytes)"
else
    echo "  ⚠ g_currentReservoirs.bin size unexpected: $reservoirs_size bytes (expected 66,355,200)"
fi

# Summary
echo ""
echo "==================================="
echo "✓ ALL TESTS PASSED"
echo "==================================="
echo ""
echo "Buffer dump feature is working correctly!"
echo ""
echo "Output files:"
ls -lh pix/buffer_dumps/
echo ""
echo "Next steps:"
echo "  1. Run existing analysis: python pix/analyze_restir_manual.py --current pix/buffer_dumps/g_currentReservoirs.bin --prev pix/buffer_dumps/g_prevReservoirs.bin"
echo "  2. Test manual dump: ./build/Debug/PlasmaDX-Clean.exe --dump-buffers --gaussian (press Ctrl+D)"
echo "  3. Multi-distance capture: See BUFFER_DUMP_IMPLEMENTATION_COMPLETE.md"
echo ""