#!/usr/bin/env python3
"""
Updates diagnose_gpu_hang to use keyboard automation instead of config files
"""

import re

# Read current mcp_server.py
with open('mcp_server.py', 'r') as f:
    content = f.read()

# 1. Add imports
old_imports = """import asyncio
import json
import os
import subprocess
import struct
from datetime import datetime
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool"""

new_imports = """import asyncio
import json
import os
import subprocess
import struct
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pyautogui
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool"""

content = content.replace(old_imports, new_imports)

# 2. Replace the config file manipulation section with keyboard automation
old_launch_code = """        # Build command with config file
        config_path = os.path.join(PLASMA_DX_PATH, "configs/agents/volumetric_restir_test.json")
        cmd = [exe_path, f"--config={config_path}"]
        
        # Update config file with current particle count
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config["particles"]["count"] = count
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            except:
                pass
        
        try:
            # Launch process with timeout
            start_time = datetime.now()
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PLASMA_DX_PATH
            )"""

new_launch_code = """        # Build command (launches in Gaussian mode by default)
        cmd = [exe_path, f"--particles", str(count)]
        
        try:
            # Launch process
            start_time = datetime.now()
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PLASMA_DX_PATH
            )
            
            # Wait for window initialization
            time.sleep(2.0)
            
            # Press F7 to switch to Volumetric ReSTIR mode
            # This triggers the hotkey added to Application.cpp
            try:
                pyautogui.press('f7')
                time.sleep(0.5)  # Wait for mode switch
            except Exception as e:
                # Log keyboard automation failure but continue
                test_result["keyboard_automation_error"] = str(e)"""

content = content.replace(old_launch_code, new_launch_code)

# Write updated file
with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Updated mcp_server.py with keyboard automation")
print("Changes:")
print("  1. Added 'import time' and 'import pyautogui'")
print("  2. Replaced config file manipulation with F7 hotkey press")
print("  3. Added 2-second init delay + 0.5s mode switch delay")
