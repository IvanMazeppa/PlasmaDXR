#!/usr/bin/env python3
"""
Test Hybrid Mode - Direct MCP Tool Calls

This test verifies that the orchestrator can call MCP tools directly,
bypassing the Claude Agent SDK's inner agent which doesn't execute tools.

The hybrid approach:
1. Imports MCP server modules directly
2. Calls tool functions as regular Python async functions
3. Bypasses JSON-RPC protocol and inner agent entirely

Usage:
    cd agents/blender-orchestrator
    source venv/bin/activate
    python test_hybrid.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Set environment
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_hybrid")


def load_server_module(server_name: str, server_path: Path):
    """Load a server module using importlib to avoid caching issues."""
    import importlib.util

    module_name = f"mcp_server_{server_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {server_name}")

    module = importlib.util.module_from_spec(spec)

    # Add server dir to path temporarily
    server_dir = str(server_path.parent)
    original_path = sys.path.copy()
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = original_path


# Global module cache for tests
_loaded_modules = {}


async def test_direct_imports():
    """Test that MCP server modules can be imported directly using importlib."""
    global _loaded_modules
    logger.info("=" * 60)
    logger.info("TEST 1: Direct Module Imports (using importlib)")
    logger.info("=" * 60)

    results = {}
    servers = [
        ("script-generator", PROJECT_ROOT / "agents" / "script-generator" / "server.py"),
        ("blender-executor", PROJECT_ROOT / "agents" / "blender-executor" / "server.py"),
        ("asset-evaluator", PROJECT_ROOT / "agents" / "asset-evaluator" / "server.py"),
    ]

    for server_name, server_path in servers:
        if not server_path.exists():
            results[server_name] = False
            logger.error(f"❌ {server_name} path not found: {server_path}")
            continue

        try:
            module = load_server_module(server_name, server_path)
            _loaded_modules[server_name] = module
            results[server_name] = True

            # Verify the module has expected tools
            tools = [name for name in dir(module)
                     if not name.startswith('_') and callable(getattr(module, name, None))]
            logger.info(f"✅ {server_name} module imported ({len(tools)} callable attributes)")
        except Exception as e:
            results[server_name] = False
            logger.error(f"❌ {server_name} import failed: {e}")

    return all(results.values()), results


async def test_direct_tool_call():
    """Test calling a tool function directly."""
    global _loaded_modules
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Direct Tool Call (list_techniques)")
    logger.info("=" * 60)

    try:
        # Use the module loaded in test 1
        if "script-generator" not in _loaded_modules:
            server_path = PROJECT_ROOT / "agents" / "script-generator" / "server.py"
            _loaded_modules["script-generator"] = load_server_module("script-generator", server_path)

        script_gen = _loaded_modules["script-generator"]

        # Call list_techniques directly
        logger.info("Calling list_techniques(effect_type='pyro')...")
        result = await script_gen.list_techniques(effect_type='pyro')

        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result preview: {str(result)[:500]}...")

        # Parse result
        data = json.loads(result) if isinstance(result, str) else result

        if "techniques" in data:
            techniques = data["techniques"]
            logger.info(f"✅ Found {len(techniques)} techniques:")
            for t in techniques[:5]:
                logger.info(f"   - {t.get('name', t)}")
            return True, data
        else:
            logger.error(f"❌ Unexpected result format: {data}")
            return False, data

    except Exception as e:
        logger.error(f"❌ Direct tool call failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def test_generate_script():
    """Test calling generate_script directly."""
    global _loaded_modules
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: Direct Tool Call (generate_script)")
    logger.info("=" * 60)

    try:
        # Use the module loaded in test 1
        if "script-generator" not in _loaded_modules:
            server_path = PROJECT_ROOT / "agents" / "script-generator" / "server.py"
            _loaded_modules["script-generator"] = load_server_module("script-generator", server_path)

        script_gen = _loaded_modules["script-generator"]

        # Call generate_script directly
        logger.info("Calling generate_script(effect_type='pyro', ...)...")
        result = await script_gen.generate_script(
            effect_type="pyro",
            description="Test explosion with bright orange fire",
            output_name="test_hybrid_explosion",
            resolution=64,  # Low res for quick test
            frame_end=10,   # Few frames for quick test
        )

        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result preview: {str(result)[:500]}...")

        # Parse result
        data = json.loads(result) if isinstance(result, str) else result

        if data.get("success") and data.get("script_path"):
            script_path = data["script_path"]
            logger.info(f"✅ Script generated: {script_path}")

            # Verify file exists
            if Path(script_path).exists():
                logger.info(f"✅ Script file exists")
                return True, data
            else:
                logger.error(f"❌ Script file NOT found: {script_path}")
                return False, data
        else:
            logger.error(f"❌ Script generation failed: {data}")
            return False, data

    except Exception as e:
        logger.error(f"❌ generate_script failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def test_orchestrator_init():
    """Test that the orchestrator can load direct tools."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: Orchestrator Direct Tool Loading")
    logger.info("=" * 60)

    try:
        # Import orchestrator module
        orchestrator_path = PROJECT_ROOT / "agents" / "blender-orchestrator"
        sys.path.insert(0, str(orchestrator_path))

        from orchestrator import BlenderOrchestratorAgent

        # Create orchestrator
        agent = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

        # Check direct tools were loaded
        logger.info(f"Direct mode enabled: {agent._direct_mode}")
        logger.info(f"Direct tools loaded: {list(agent._direct_tools.keys())}")

        if agent._direct_tools:
            logger.info(f"✅ Orchestrator loaded {len(agent._direct_tools)} direct tool modules")
            return True, list(agent._direct_tools.keys())
        else:
            logger.error("❌ No direct tools loaded")
            return False, []

    except Exception as e:
        logger.error(f"❌ Orchestrator init failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("HYBRID MODE TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("")

    results = {}

    # Test 1: Direct imports
    success, data = await test_direct_imports()
    results["imports"] = success

    # Test 2: Direct tool call (list_techniques)
    success, data = await test_direct_tool_call()
    results["list_techniques"] = success

    # Test 3: Direct tool call (generate_script)
    success, data = await test_generate_script()
    results["generate_script"] = success

    # Test 4: Orchestrator loading
    success, data = await test_orchestrator_init()
    results["orchestrator_init"] = success

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED - Hybrid mode is working!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("The orchestrator can now call MCP tools directly,")
        logger.info("bypassing the Claude Agent SDK's inner agent.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run test_explosion.py to test full workflow")
        logger.info("2. Monitor logs for [HYBRID] and [DIRECT] markers")
        return 0
    else:
        logger.info("=" * 60)
        logger.info("❌ SOME TESTS FAILED")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
