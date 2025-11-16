#!/usr/bin/env python3
"""
Improve process killing to handle Windows process tree
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

old_kill = """            except subprocess.TimeoutExpired:
                # Process exceeded timeout - kill it forcefully
                proc.kill()
                try:
                    proc.wait(timeout=2)  # Wait up to 2s for clean shutdown
                except subprocess.TimeoutExpired:
                    # Still running, terminate harder (Windows TASKKILL)
                    try:
                        subprocess.run(["taskkill.exe", "/F", "/IM", "PlasmaDX-Clean.exe"], 
                                     capture_output=True, timeout=2)
                    except:
                        pass  # Best effort
                
                end_time = datetime.now()
                test_result["runtime_seconds"] = (end_time - start_time).total_seconds()
                test_result["hang_detected"] = False  # Not a hang, just timeout
                test_result["status"] = "timeout"  # Changed from "hung" to "timeout"""

new_kill = """            except subprocess.TimeoutExpired:
                # Process exceeded timeout - kill it forcefully
                # Windows: Kill by image name to catch any child processes
                try:
                    kill_result = subprocess.run(
                        ["taskkill.exe", "/F", "/IM", "PlasmaDX-Clean.exe"], 
                        capture_output=True, 
                        text=True,
                        timeout=5
                    )
                    # Also kill the subprocess handle
                    proc.kill()
                    proc.wait(timeout=2)
                except:
                    # Best effort - process might already be dead
                    pass
                
                end_time = datetime.now()
                test_result["runtime_seconds"] = (end_time - start_time).total_seconds()
                test_result["hang_detected"] = False  # Not a hang, just timeout
                test_result["status"] = "timeout"""

content = content.replace(old_kill, new_kill)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Improved process killing")
print("  - Kill by image name first (catches all instances)")
print("  - Then kill subprocess handle")
print("  - More robust error handling")
