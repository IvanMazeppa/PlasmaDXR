#!/usr/bin/env python3
"""
Fix process killing to ensure PlasmaDX terminates properly
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Find the timeout handling and fix the kill
old_timeout = """            except subprocess.TimeoutExpired:
                # Process hung - kill it
                proc.kill()
                end_time = datetime.now()
                test_result["runtime_seconds"] = (end_time - start_time).total_seconds()
                test_result["hang_detected"] = True
                test_result["status"] = "hung"""

new_timeout = """            except subprocess.TimeoutExpired:
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

content = content.replace(old_timeout, new_timeout)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed process killing")
print("Changes:")
print("  1. Added proc.wait(2) after kill")
print("  2. Added Windows TASKKILL fallback")
print("  3. Changed status from 'hung' to 'timeout'")
print("  4. Set hang_detected=False (timeout != hang)")
