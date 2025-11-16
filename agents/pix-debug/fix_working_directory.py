#!/usr/bin/env python3
"""
Fixes working directory issue in diagnose_gpu_hang
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Fix 1: Update exe path to build/bin/Debug (latest build location)
content = content.replace(
    'exe_path = os.path.join(PLASMA_DX_PATH, "build/Debug/PlasmaDX-Clean.exe")',
    'exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")'
)

# Fix 2: Add exe_dir variable and use it as cwd
old_popen = """            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PLASMA_DX_PATH
            )"""

new_popen = """            # Get exe directory for correct working directory (shader paths)
            exe_dir = os.path.dirname(exe_path)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exe_dir  # Run from debug folder, not project root
            )"""

content = content.replace(old_popen, new_popen)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed working directory issue in diagnose_gpu_hang")
print("Changes:")
print("  1. Updated exe_path: build/Debug → build/bin/Debug")
print("  2. Changed cwd: PLASMA_DX_PATH → exe_dir (debug folder)")
