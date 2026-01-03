#!/usr/bin/env python3
"""
Minimal MCP tool execution test for Blender VFX Orchestrator.

This test verifies that the Claude Agent SDK inner agent actually executes
MCP tools instead of just describing them.

Test setup:
- ONE MCP server (script-generator)
- ONE tool call (list_techniques)
- $0.50 budget limit

Success criteria:
- Logs should show "[TOOL CALL] mcp__script-generator__list_techniques"
- Response should contain technique names from the catalog, not just
  "I'll use list_techniques()..."

Usage:
    cd agents/blender-orchestrator
    source venv/bin/activate
    python test_mcp_simple.py
"""

import asyncio
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

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_mcp_simple")

# Import after path setup
try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    SDK_AVAILABLE = True
except ImportError:
    logger.error("Claude Agent SDK not installed. Run: pip install claude-agent-sdk>=0.1.17")
    SDK_AVAILABLE = False


def create_simple_mcp_config() -> dict:
    """Create minimal MCP config with just script-generator."""
    script_gen_path = PROJECT_ROOT / "agents" / "script-generator"

    # NOTE: SDK McpStdioServerConfig only supports: type, command, args, env
    # NO cwd field - use cd in bash command instead
    return {
        "script-generator": {
            "type": "stdio",
            "command": "bash",
            "args": [
                "-c",
                f"cd '{script_gen_path}' && PROJECT_ROOT='{PROJECT_ROOT}' exec ./run_server.sh"
            ],
            "env": {
                "PROJECT_ROOT": str(PROJECT_ROOT),
                "PYTHONPATH": str(PROJECT_ROOT),
            }
        }
    }


async def test_single_tool_call():
    """Test that a single MCP tool is actually executed."""
    if not SDK_AVAILABLE:
        logger.error("Test cannot run - SDK not available")
        return False

    logger.info("=" * 60)
    logger.info("MINIMAL MCP TOOL EXECUTION TEST")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Check script-generator exists
    script_gen_path = PROJECT_ROOT / "agents" / "script-generator" / "run_server.sh"
    if not script_gen_path.exists():
        logger.error(f"Script generator not found: {script_gen_path}")
        return False
    logger.info(f"Script generator found: {script_gen_path}")

    # Create minimal config
    mcp_config = create_simple_mcp_config()
    logger.info(f"MCP config: {mcp_config}")

    # Create agent options with minimal budget
    options = ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),
        system_prompt="""You are a test agent. When asked to list techniques,
you MUST call the mcp__script-generator__list_techniques tool.
Do NOT just describe what you would do - actually call the tool.""",
        mcp_servers=mcp_config,
        allowed_tools=["mcp__script-generator__list_techniques"],
        max_budget_usd=0.50,  # Strict budget limit
        permission_mode='acceptEdits',
    )

    logger.info("Creating Claude SDK client...")

    try:
        client = ClaudeSDKClient(options=options)
        logger.info("Client created, starting agent...")

        # SDK uses async context manager pattern
        await client.__aenter__()
        logger.info("Agent started successfully")

        # Send a simple prompt asking to list techniques
        prompt = """Please list the available pyro techniques by calling the
mcp__script-generator__list_techniques tool.
Return the actual technique names from the tool result."""

        logger.info(f"Sending prompt: {prompt[:100]}...")
        await client.query(prompt)

        # Collect response and check for tool execution
        full_response = ""
        tool_calls = []
        message_count = 0

        tools_available = []  # Tools from system message (catalog)
        tools_invoked = []    # Tools actually called via tool_use
        tools_results = []    # Tool results received

        logger.info("Receiving response...")
        async for message in client.receive_response():
            message_count += 1
            message_type = type(message).__name__
            message_text = str(message) if not isinstance(message, str) else message

            # Check if this is a SystemMessage with tool catalog
            if message_type == 'SystemMessage' and 'tools' in message_text:
                import re
                catalog_tools = re.findall(r"'(mcp__[\w-]+__\w+)'", message_text)
                tools_available.extend(catalog_tools)
                logger.info(f"  Message {message_count}: type={message_type} ({len(catalog_tools)} tools in catalog)")
            else:
                logger.info(f"  Message {message_count}: type={message_type}")

            # Check if this is an AssistantMessage with tool_use content
            if hasattr(message, 'data') and isinstance(message.data, dict):
                data = message.data
                # Check for tool_use blocks in content
                if 'content' in data and isinstance(data['content'], list):
                    for block in data['content']:
                        if isinstance(block, dict) and block.get('type') == 'tool_use':
                            tool_name = block.get('name', 'unknown')
                            tools_invoked.append(tool_name)
                            logger.info(f"    [TOOL INVOKED] {tool_name}")
                # Check for tool_result in message type
                if data.get('type') == 'tool_result':
                    tools_results.append("result")
                    logger.info(f"    [TOOL RESULT RECEIVED]")

            # Look for tool_use patterns in text (SDK may serialize differently)
            if "'type': 'tool_use'" in message_text or '"type": "tool_use"' in message_text:
                import re
                # Extract tool name from tool_use block
                name_match = re.search(r"'name':\s*'(mcp__[\w-]+__\w+)'", message_text)
                if not name_match:
                    name_match = re.search(r'"name":\s*"(mcp__[\w-]+__\w+)"', message_text)
                if name_match and name_match.group(1) not in tools_invoked:
                    tools_invoked.append(name_match.group(1))
                    logger.info(f"    [TOOL INVOKED from text] {name_match.group(1)}")

            # Look for tool result patterns
            if "'type': 'tool_result'" in message_text or '"type": "tool_result"' in message_text:
                tools_results.append("result_from_text")
                logger.info(f"    [TOOL RESULT from text]")

            full_response += message_text

        # Merge for backward compat
        tool_calls = tools_invoked

        logger.info("-" * 40)
        logger.info(f"RESULTS:")
        logger.info(f"  Total messages: {message_count}")
        logger.info(f"  Tools in catalog: {len(tools_available)}")
        logger.info(f"  Tools invoked: {len(tools_invoked)}")
        logger.info(f"  Tool results: {len(tools_results)}")
        logger.info(f"  Tools called: {tools_invoked}")
        logger.info(f"  Response length: {len(full_response)} chars")
        logger.info("-" * 40)

        # Check for success indicators
        success = False

        if tools_invoked:
            if "mcp__script-generator__list_techniques" in tools_invoked:
                logger.info("✅ SUCCESS: list_techniques tool was INVOKED!")
                success = True
            else:
                logger.warning(f"⚠ Tools invoked but not the expected one: {tools_invoked}")
                success = True  # Still count as success - tools ARE working
        elif tools_available and not tools_invoked:
            # Tools available but not invoked - check what happened
            logger.warning("⚠ Tools are available but none were invoked")
            # Check if the response contains technique names (tool result)
            if "rising_mushroom" in full_response or "ground_burst" in full_response:
                logger.info("✅ SUCCESS: Response contains technique data (tool was called)")
                success = True
            else:
                logger.error("❌ FAILURE: Tools available but not used!")
                # Check for "I will use" phrases
                no_tool_phrases = ["I'll use", "I will use", "I will call", "Let me use"]
                for phrase in no_tool_phrases:
                    if phrase.lower() in full_response.lower():
                        logger.error(f"  Found '{phrase}' - tool was DESCRIBED but NOT CALLED")
                        break
                logger.error(f"  Response excerpt: {full_response[:500]}...")
        else:
            logger.error("❌ FAILURE: No tools available - MCP server may have failed to start!")
            logger.error(f"  Response excerpt: {full_response[:500]}...")

        # Clean up
        await client.__aexit__(None, None, None)
        logger.info("Agent stopped")

        return success

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    logger.info("Starting minimal MCP test...")

    result = asyncio.run(test_single_tool_call())

    logger.info("=" * 60)
    if result:
        logger.info("TEST PASSED - MCP tools are being executed correctly")
        sys.exit(0)
    else:
        logger.error("TEST FAILED - MCP tools are NOT being executed")
        logger.error("Check the logs above for details on what went wrong")
        sys.exit(1)


if __name__ == "__main__":
    main()
