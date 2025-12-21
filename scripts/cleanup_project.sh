#!/bin/bash
# PlasmaDX-Clean Project Cleanup Script
# Safe removal of regenerable artifacts and bloat
#
# Usage:
#   ./scripts/cleanup_project.sh --dry-run   # Preview what would be deleted
#   ./scripts/cleanup_project.sh --execute   # Actually delete

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DRY_RUN=true

if [[ "$1" == "--execute" ]]; then
    DRY_RUN=false
    echo -e "${RED}=== EXECUTE MODE - FILES WILL BE DELETED ===${NC}"
elif [[ "$1" == "--dry-run" ]]; then
    echo -e "${BLUE}=== DRY RUN MODE - No files will be deleted ===${NC}"
else
    echo "Usage: $0 [--dry-run|--execute]"
    echo "  --dry-run   Preview what would be deleted (default)"
    echo "  --execute   Actually delete files"
    exit 1
fi

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Function to safely remove with size reporting
safe_remove() {
    local path="$1"
    local description="$2"

    if [[ -e "$path" ]]; then
        local size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo -e "${YELLOW}[$size]${NC} $description: $path"

        if [[ "$DRY_RUN" == false ]]; then
            rm -rf "$path"
            echo -e "  ${GREEN}DELETED${NC}"
        fi
    fi
}

# Function to find and remove pattern
find_and_remove() {
    local pattern="$1"
    local description="$2"

    echo -e "\n${BLUE}=== $description ===${NC}"

    local found=$(find . -type d -name "$pattern" 2>/dev/null | wc -l)
    if [[ $found -gt 0 ]]; then
        local total_size=$(find . -type d -name "$pattern" -exec du -sh {} + 2>/dev/null | awk '{sum+=$1} END {print sum}')
        echo "Found $found directories matching '$pattern'"

        if [[ "$DRY_RUN" == false ]]; then
            find . -type d -name "$pattern" -exec rm -rf {} + 2>/dev/null || true
            echo -e "${GREEN}Removed all '$pattern' directories${NC}"
        else
            find . -type d -name "$pattern" 2>/dev/null | head -10
            if [[ $found -gt 10 ]]; then
                echo "  ... and $((found-10)) more"
            fi
        fi
    else
        echo "None found"
    fi
}

# ============================================================================
# PHASE 1: Python Virtual Environments (MASSIVE savings)
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 1: Python Virtual Environments${NC}"
echo -e "${RED}========================================${NC}"

safe_remove "venv" "Root virtual environment"
safe_remove "ml/venv" "ML virtual environment"
safe_remove ".venv" "Hidden virtual environment"
safe_remove "env" "Alternative venv name"

# ============================================================================
# PHASE 2: Python Cache Files
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 2: Python Cache Files${NC}"
echo -e "${RED}========================================${NC}"

find_and_remove "__pycache__" "Python bytecode cache"
find_and_remove ".ipynb_checkpoints" "Jupyter checkpoints"

echo -e "\n${BLUE}=== .pyc/.pyo files ===${NC}"
pyc_count=$(find . -name "*.pyc" -o -name "*.pyo" 2>/dev/null | wc -l)
echo "Found $pyc_count bytecode files"
if [[ "$DRY_RUN" == false ]]; then
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    echo -e "${GREEN}Removed all .pyc/.pyo files${NC}"
fi

# ============================================================================
# PHASE 3: Build Artifacts
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 3: Build Artifacts${NC}"
echo -e "${RED}========================================${NC}"

safe_remove "build" "CMake/MSBuild output"
safe_remove "cmake-build-debug" "CLion debug build"
safe_remove "cmake-build-release" "CLion release build"
safe_remove ".vs" "Visual Studio cache"

# ============================================================================
# PHASE 4: Node.js (if any)
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 4: Node.js Artifacts${NC}"
echo -e "${RED}========================================${NC}"

find_and_remove "node_modules" "Node.js dependencies"

# ============================================================================
# PHASE 5: PIX Captures (Optional - keep recent)
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 5: PIX Captures (Review manually)${NC}"
echo -e "${RED}========================================${NC}"

if [[ -d "PIX" ]]; then
    pix_size=$(du -sh PIX 2>/dev/null | cut -f1)
    echo -e "${YELLOW}PIX directory: $pix_size${NC}"
    echo "  Contains GPU captures - review manually"
    echo "  Consider keeping only recent/important captures"
    ls -la PIX/*.wpix 2>/dev/null | tail -5 || echo "  No .wpix files found"
fi

# ============================================================================
# PHASE 6: Git cleanup
# ============================================================================
echo -e "\n${RED}========================================${NC}"
echo -e "${RED}PHASE 6: Git Maintenance${NC}"
echo -e "${RED}========================================${NC}"

if [[ "$DRY_RUN" == false ]]; then
    echo "Running git gc --aggressive..."
    git gc --aggressive --prune=now 2>/dev/null || echo "Git gc skipped"
    echo "Running git prune..."
    git prune 2>/dev/null || echo "Git prune skipped"
else
    git_size=$(du -sh .git 2>/dev/null | cut -f1)
    echo "Git repository: $git_size"
    echo "Would run: git gc --aggressive --prune=now"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}CLEANUP SUMMARY${NC}"
echo -e "${GREEN}========================================${NC}"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}This was a DRY RUN - no files were deleted${NC}"
    echo ""
    echo "To actually clean up, run:"
    echo -e "  ${GREEN}./scripts/cleanup_project.sh --execute${NC}"
else
    echo -e "${GREEN}Cleanup complete!${NC}"
    echo ""
    echo "New project size:"
    du -sh . 2>/dev/null
    echo ""
    echo "File count:"
    find . -type f 2>/dev/null | wc -l
fi

echo ""
echo "Next steps:"
echo "1. Verify the build still works: cmake .. && make"
echo "2. Recreate venv if needed: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
echo "3. Consider migrating to WSL native filesystem for better performance"
