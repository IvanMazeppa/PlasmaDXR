#!/usr/bin/env python3
"""
Fixes keyboard automation to work from WSL2 → Windows GUI
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Replace pyautogui (doesn't work WSL→Windows) with PowerShell SendKeys
old_code = """            # Wait for window initialization
            time.sleep(2.0)
            
            # Press F7 to switch to Volumetric ReSTIR mode
            # This triggers the hotkey added to Application.cpp
            try:
                pyautogui.press('f7')
                time.sleep(0.5)  # Wait for mode switch
            except Exception as e:
                # Log keyboard automation failure but continue
                test_result["keyboard_automation_error"] = str(e)"""

new_code = """            # Wait for window initialization
            time.sleep(3.0)  # Increased for window focus
            
            # Press F7 using PowerShell SendKeys (WSL2 → Windows compatible)
            try:
                # PowerShell script to activate window and send F7
                ps_script = '''
$wshell = New-Object -ComObject wscript.shell
Start-Sleep -Milliseconds 500
if ($wshell.AppActivate("PlasmaDX")) {
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

content = content.replace(old_code, new_code)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed keyboard automation for WSL2 → Windows")
print("Changes:")
print("  1. Replaced pyautogui with PowerShell SendKeys")
print("  2. Added AppActivate to focus PlasmaDX window")
print("  3. Increased delays for reliability")
