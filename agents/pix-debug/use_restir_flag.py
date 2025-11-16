#!/usr/bin/env python3
"""
Replace keyboard automation with --restir command-line flag
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Remove all PowerShell keyboard automation code
old_code = """        # Build command (launches in Gaussian mode by default)
        cmd = [exe_path, f"--particles", str(count)]
        
        try:
            # Get exe directory for correct working directory (shader paths)
            exe_dir = os.path.dirname(exe_path)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exe_dir  # Run from debug folder, not project root
            )
            
            # Wait for window initialization
            time.sleep(3.0)  # Increased for window focus
            
            # Press F7 using PowerShell SendKeys (WSL2 → Windows compatible)
            try:
                # PowerShell script to activate window and send F7
                ps_script = '''
$wshell = New-Object -ComObject wscript.shell
Start-Sleep -Milliseconds 500
if ($wshell.AppActivate("PlasmaDX-Clean")) {
    Start-Sleep -Milliseconds 300
    $wshell.SendKeys("{F7}")
} else {
    Write-Host "Failed to activate PlasmaDX window"
}
'''
                result = subprocess.run(
                    ["powershell.exe", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                time.sleep(1.0)  # Wait for mode switch
                if result.stderr:
                    test_result["keyboard_automation_warning"] = result.stderr
            except Exception as e:
                test_result["keyboard_automation_error"] = str(e)"""

new_code = """        # Build command with --restir flag to enable Volumetric ReSTIR
        cmd = [exe_path, "--particles", str(count), "--restir"]
        
        try:
            # Get exe directory for correct working directory (shader paths)
            exe_dir = os.path.dirname(exe_path)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exe_dir  # Run from debug folder, not project root
            )"""

content = content.replace(old_code, new_code)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Replaced keyboard automation with --restir flag")
print("Changes:")
print("  - Removed PowerShell SendKeys code")
print("  - Added --restir to command line")
print("  - Removed all timing delays (not needed)")
