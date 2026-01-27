#!/bin/bash
cd /home/maz3ppa/projects/PlasmaDXR
export PIX_AUTO_CAPTURE=1
export PIX_CAPTURE_FRAME=$1
./build/bin/Debug/PlasmaDXR.exe
