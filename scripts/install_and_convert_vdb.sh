#!/bin/bash
# OpenVDB to NanoVDB Conversion Script for PlasmaDX-Clean
# Run this script in WSL2 to convert .vdb files to .nvdb format

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} OpenVDB to NanoVDB Converter${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if nanovdb_convert is available
if ! command -v nanovdb_convert &> /dev/null; then
    echo -e "${YELLOW}nanovdb_convert not found. Installing libnanovdb-tools...${NC}"
    echo ""
    echo -e "${YELLOW}Please enter your sudo password:${NC}"
    sudo apt-get update
    sudo apt-get install -y libnanovdb-tools
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
else
    echo -e "${GREEN}nanovdb_convert is already installed.${NC}"
fi

echo ""

# Paths
PROJECT_DIR="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
VDB_DIR="${PROJECT_DIR}/VDBs/Clouds/CloudPackVDB"
OUTPUT_DIR="${PROJECT_DIR}/VDBs/NanoVDB"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check for VDB files
VDB_COUNT=$(find "${VDB_DIR}" -name "*.vdb" 2>/dev/null | wc -l)

if [ "$VDB_COUNT" -eq 0 ]; then
    echo -e "${RED}No .vdb files found in ${VDB_DIR}${NC}"
    exit 1
fi

echo -e "Found ${GREEN}${VDB_COUNT}${NC} VDB file(s) in ${VDB_DIR}"
echo -e "Output directory: ${OUTPUT_DIR}"
echo ""

# Convert files
SUCCESS_COUNT=0
FAIL_COUNT=0

for vdb_file in "${VDB_DIR}"/*.vdb; do
    if [ -f "$vdb_file" ]; then
        filename=$(basename "$vdb_file" .vdb)
        nvdb_file="${OUTPUT_DIR}/${filename}.nvdb"

        echo -n "Converting ${filename}.vdb... "

        if nanovdb_convert "$vdb_file" "$nvdb_file" 2>/dev/null; then
            size=$(ls -lh "$nvdb_file" | awk '{print $5}')
            echo -e "${GREEN}OK${NC} (${size})"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}FAILED${NC}"
            ((FAIL_COUNT++))
        fi
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "Conversion complete: ${GREEN}${SUCCESS_COUNT}${NC} succeeded, ${RED}${FAIL_COUNT}${NC} failed"
echo -e "${GREEN}========================================${NC}"

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/*.nvdb 2>/dev/null | awk '{print "  " $9 ": " $5}'
    echo ""
    echo -e "${GREEN}You can now load these .nvdb files in PlasmaDX-Clean!${NC}"
fi
