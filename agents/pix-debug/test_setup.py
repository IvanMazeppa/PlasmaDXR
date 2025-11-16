#!/usr/bin/env python3
"""
Quick test to verify PIX Debugging Agent v4 setup
"""

import os
import sys

print("🔍 PIX Debugging Agent v4 - Setup Verification")
print("=" * 70)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✓ Virtual environment: Active")
else:
    print("⚠ Virtual environment: Not active (run 'source venv/bin/activate')")

# Check imports
try:
    import claude_agent_sdk
    print(f"✓ claude-agent-sdk: {claude_agent_sdk.__version__}")
except ImportError as e:
    print(f"✗ claude-agent-sdk: Not installed ({e})")
    sys.exit(1)

try:
    import numpy
    print(f"✓ numpy: {numpy.__version__}")
except ImportError:
    print("✗ numpy: Not installed")
    sys.exit(1)

try:
    import pandas
    print(f"✓ pandas: {pandas.__version__}")
except ImportError:
    print("✗ pandas: Not installed")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv: Installed")
except ImportError:
    print("✗ python-dotenv: Not installed")
    sys.exit(1)

# Check .env file
if os.path.exists('.env'):
    print("✓ .env file: Exists")
    load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("ANTHROPIC_API_KEY") == "your-api-key-here":
            print("⚠ ANTHROPIC_API_KEY: Set to placeholder (update with real key)")
        else:
            print("✓ ANTHROPIC_API_KEY: Configured")
    else:
        print("✗ ANTHROPIC_API_KEY: Not set in .env")
else:
    print("⚠ .env file: Not found (copy .env.example to .env)")

# Check main.py syntax
try:
    import main
    print("✓ main.py: Syntax valid")
except Exception as e:
    print(f"✗ main.py: Error - {e}")
    sys.exit(1)

# Check SDK tools
try:
    from claude_agent_sdk import tool, create_sdk_mcp_server, query, ClaudeAgentOptions
    print("✓ SDK imports: All available")
except ImportError as e:
    print(f"✗ SDK imports: Missing - {e}")
    sys.exit(1)

# Check paths
plasma_dx_path = os.getenv("PLASMA_DX_PATH", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
if os.path.exists(plasma_dx_path):
    print(f"✓ PlasmaDX-Clean path: {plasma_dx_path}")
else:
    print(f"⚠ PlasmaDX-Clean path: Not found at {plasma_dx_path}")

pixtool_path = os.getenv("PIXTOOL_PATH", "/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe")
if os.path.exists(pixtool_path):
    print(f"✓ PIX Tool path: {pixtool_path}")
else:
    print(f"⚠ PIX Tool path: Not found at {pixtool_path}")

print("\n" + "=" * 70)
print("✅ Setup verification complete!")
print("\nNext steps:")
print("  1. Update .env with your ANTHROPIC_API_KEY")
print("  2. Run 'python main.py' to start the agent")
print("=" * 70)
