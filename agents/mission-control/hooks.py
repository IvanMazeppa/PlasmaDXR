"""
Mission-Control Safety Hooks

Provides safety rails and guards for agent tool execution to prevent:
- Destructive file operations (rm -rf, etc.)
- Unsafe command execution
- Resource exhaustion
- Unauthorized operations

TODO: Integrate with ClaudeAgentOptions.hooks when implementing
      agent-driven tool execution.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("mission-control.hooks")


class SafetyHooks:
    """
    Safety hooks for guarding agent operations.

    Prevents destructive commands and enforces safe execution patterns.
    """

    # Dangerous command patterns to block
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",  # Recursive delete from root
        r"rm\s+-rf\s+\*",  # Delete all files
        r"dd\s+if=",  # Disk write operations
        r"mkfs\.",  # Format filesystem
        r":(){ :|:& };:",  # Fork bomb
        r"sudo\s+",  # Elevated privileges
        r"chmod\s+777",  # Unsafe permissions
    ]

    # Commands that should run in background only
    BACKGROUND_ONLY = [
        "PlasmaDX-Clean.exe",  # Main application
        "MSBuild.exe",  # Builds can be long-running
    ]

    @staticmethod
    def guard_bash_command(command: str) -> tuple[bool, str]:
        """
        Check if a bash command is safe to execute.

        Args:
            command: Shell command to validate

        Returns:
            Tuple of (is_safe, reason)
            - is_safe: True if command is allowed
            - reason: Explanation if blocked

        Example:
            >>> is_safe, reason = SafetyHooks.guard_bash_command("ls -la")
            >>> assert is_safe is True
            >>> is_safe, reason = SafetyHooks.guard_bash_command("rm -rf /")
            >>> assert is_safe is False
        """
        # Check for dangerous patterns
        for pattern in SafetyHooks.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                reason = f"Blocked: Command matches dangerous pattern '{pattern}'"
                logger.warning(f"guard_bash_command: {reason}")
                return False, reason

        # Check for background-only commands
        for bg_cmd in SafetyHooks.BACKGROUND_ONLY:
            if bg_cmd in command and "&" not in command:
                reason = (
                    f"Warning: '{bg_cmd}' should run in background (append ' &' or use run_in_background)"
                )
                logger.info(f"guard_bash_command: {reason}")
                # Don't block, just warn
                # Could enforce by returning (False, reason) here

        return True, "Command allowed"

    @staticmethod
    def guard_file_write(file_path: str, content: str) -> tuple[bool, str]:
        """
        Check if a file write operation is safe.

        Args:
            file_path: Path to file being written
            content: Content to write

        Returns:
            Tuple of (is_safe, reason)

        Prevents:
            - Writing to system directories
            - Overwriting critical project files without backup
            - Writing excessively large files
        """
        path = Path(file_path)

        # Block writes to system directories
        dangerous_dirs = ["/bin", "/usr", "/etc", "/sys", "/boot"]
        if any(str(path).startswith(d) for d in dangerous_dirs):
            reason = f"Blocked: Cannot write to system directory {path}"
            logger.warning(f"guard_file_write: {reason}")
            return False, reason

        # Check file size (block >100MB writes)
        content_size_mb = len(content) / (1024 * 1024)
        if content_size_mb > 100:
            reason = f"Blocked: Content too large ({content_size_mb:.1f} MB > 100 MB limit)"
            logger.warning(f"guard_file_write: {reason}")
            return False, reason

        # Warn if overwriting critical files
        critical_files = ["CLAUDE.md", "README.md", "CMakeLists.txt"]
        if path.name in critical_files:
            reason = f"Warning: Writing to critical file {path.name} - ensure backup exists"
            logger.info(f"guard_file_write: {reason}")
            # Don't block, just warn

        return True, "File write allowed"

    @staticmethod
    def guard_dispatch(council: str, priority: str) -> tuple[bool, str]:
        """
        Validate dispatch parameters before sending to council.

        Args:
            council: Target council name
            priority: Priority level

        Returns:
            Tuple of (is_valid, reason)
        """
        valid_councils = ["rendering", "materials", "physics", "diagnostics"]
        valid_priorities = ["critical", "high", "medium", "low"]

        if council not in valid_councils:
            return False, f"Invalid council '{council}'. Valid: {valid_councils}"

        if priority not in valid_priorities:
            return False, f"Invalid priority '{priority}'. Valid: {valid_priorities}"

        return True, "Dispatch parameters valid"


# Example integration with ClaudeAgentOptions (not yet implemented)
# TODO: Wire these hooks into the SDK when agent-driven execution is enabled
#
# def create_hooked_options() -> ClaudeAgentOptions:
#     """Create options with safety hooks enabled."""
#     hooks = {
#         "before_bash": SafetyHooks.guard_bash_command,
#         "before_write": SafetyHooks.guard_file_write,
#     }
#     return ClaudeAgentOptions(hooks=hooks)
