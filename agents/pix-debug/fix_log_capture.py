#!/usr/bin/env python3
"""
Fix log capture to include initialization logs (first 50 lines + last 100 lines)
"""

with open('mcp_server.py', 'r') as f:
    content = f.read()

# Find the log capture section and update it
old_capture = """                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read last 100 lines
                        lines = f.readlines()
                        last_lines = lines[-100:] if len(lines) > 100 else lines
                        test_result["logs_captured"] = [line.strip() for line in last_lines if line.strip()]"""

new_capture = """                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read first 50 lines (initialization) + last 100 lines (hang analysis)
                        lines = f.readlines()
                        first_lines = lines[:50] if len(lines) > 50 else lines
                        last_lines = lines[-100:] if len(lines) > 100 else lines
                        
                        # Combine first and last, avoiding duplicates if log is short
                        if len(lines) <= 150:
                            captured = lines
                        else:
                            captured = first_lines + ["... (middle section omitted) ..."] + last_lines
                        
                        test_result["logs_captured"] = [line.strip() for line in captured if line.strip()]"""

content = content.replace(old_capture, new_capture)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed log capture to include initialization logs")
print("Changes:")
print("  - Captures first 50 lines (for --restir verification)")
print("  - Captures last 100 lines (for hang analysis)")
print("  - Avoids duplicates for short logs")
