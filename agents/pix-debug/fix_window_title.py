#!/usr/bin/env python3
with open('mcp_server.py', 'r') as f:
    content = f.read()

# Fix window title to match actual PlasmaDX window
content = content.replace(
    'if ($wshell.AppActivate("PlasmaDX")) {',
    'if ($wshell.AppActivate("PlasmaDX-Clean")) {'
)

with open('mcp_server.py', 'w') as f:
    f.write(content)

print("✅ Fixed window title: PlasmaDX → PlasmaDX-Clean")
