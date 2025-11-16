#!/usr/bin/env python3
"""
Fixes log directory to match exe working directory
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Fix log_dir to be relative to exe location, not project root
old_log_path = """    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")
    log_dir = os.path.join(PLASMA_DX_PATH, "logs")"""

new_log_path = """    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")
    exe_dir = os.path.dirname(exe_path)
    log_dir = os.path.join(exe_dir, "logs")  # Logs written relative to exe location"""

content = content.replace(old_log_path, new_log_path)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed log directory path")
print("Changes:")
print("  OLD: log_dir = PLASMA_DX_PATH/logs")
print("  NEW: log_dir = exe_dir/logs (build/bin/Debug/logs)")
