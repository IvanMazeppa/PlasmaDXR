#!/bin/bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
export PIX_AUTO_CAPTURE=1
export PIX_CAPTURE_FRAME=$1
./build/DebugPIX/PlasmaDX-Clean-PIX.exe
