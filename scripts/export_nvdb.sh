#!/bin/bash
# ============================================================================
# NanoVDB Export Wrapper Script
# ============================================================================
#
# Easy-to-use wrapper for Blender NanoVDB export from the command line.
# Automatically finds Blender and passes arguments correctly.
#
# Usage:
#   ./scripts/export_nvdb.sh scene.blend --output ./volumes --resolution 300
#   ./scripts/export_nvdb.sh scene.blend --frames 1-50 --resolution 256
#
# Requirements:
#   - Blender 4.0+ installed (Linux version for WSL, or Windows via WSL path)
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find Blender executable
find_blender() {
    # Try common locations
    local candidates=(
        # WSL Linux Blender
        "/usr/bin/blender"
        "/usr/local/bin/blender"
        "$HOME/blender/blender"
        "/opt/blender/blender"
        # Windows Blender via WSL
        "/mnt/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
        "/mnt/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"
        "/mnt/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    # Check if 'blender' is in PATH
    if command -v blender &> /dev/null; then
        command -v blender
        return 0
    fi

    return 1
}

# Show usage
usage() {
    echo "Usage: $0 <scene.blend> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  scene.blend          Input Blender file with fluid simulation"
    echo ""
    echo "Options:"
    echo "  --output, -o DIR     Output directory for .nvdb files (default: ./nvdb_export)"
    echo "  --frames, -f RANGE   Frame range: 'current', number, or 'start-end' (default: current)"
    echo "  --resolution, -r N   Domain resolution (256-512 recommended, higher=smoother)"
    echo "  --grid, -g NAME      Grid name to export (default: density)"
    echo "  --bake-only          Only bake simulation, don't export"
    echo "  --no-bake            Skip baking, export existing cache"
    echo "  --quiet, -q          Suppress progress messages"
    echo ""
    echo "Examples:"
    echo "  $0 smoke_sim.blend --output ~/volumes/smoke --resolution 300"
    echo "  $0 explosion.blend --frames 1-100 --resolution 256"
    echo "  $0 clouds.blend --resolution 384 --bake-only"
    echo ""
    echo "Output:"
    echo "  Creates volume_XXXX.nvdb files in the output directory"
    echo "  Files are in NanoVDB format, compatible with PlasmaDX shader"
    echo ""
    echo "Notes:"
    echo "  - Resolution 256 is minimum for smooth results"
    echo "  - Resolution 384+ recommended for hero volumes"
    echo "  - Higher resolution = larger files and longer bake time"
    echo "  - See docs/NanoVDB/BLENDER_HIGH_RES_EXPORT_GUIDE.md for details"
}

# Main
main() {
    if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    # First argument is the blend file
    BLEND_FILE="$1"
    shift

    if [[ ! -f "$BLEND_FILE" ]]; then
        echo -e "${RED}ERROR: File not found: $BLEND_FILE${NC}"
        exit 1
    fi

    # Find Blender
    BLENDER=$(find_blender)
    if [[ -z "$BLENDER" ]]; then
        echo -e "${RED}ERROR: Blender not found!${NC}"
        echo ""
        echo "Please install Blender 4.0+ or set the path manually."
        echo "For WSL with Linux Blender:"
        echo "  sudo snap install blender --classic"
        echo "Or download from https://www.blender.org/download/"
        exit 1
    fi

    echo -e "${GREEN}Found Blender: $BLENDER${NC}"
    echo ""

    # Run Blender with our export script
    EXPORT_SCRIPT="$SCRIPT_DIR/blender_export_nvdb.py"

    if [[ ! -f "$EXPORT_SCRIPT" ]]; then
        echo -e "${RED}ERROR: Export script not found: $EXPORT_SCRIPT${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Starting Blender in background mode...${NC}"
    echo ""

    "$BLENDER" --background "$BLEND_FILE" --python "$EXPORT_SCRIPT" -- "$@"

    echo ""
    echo -e "${GREEN}Done!${NC}"
}

main "$@"
