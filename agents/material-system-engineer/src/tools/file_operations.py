"""
File Operations Tools
Read, write, and search codebase files with automatic backup
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


async def read_codebase_file(
    project_root: str,
    file_path: str
) -> str:
    """
    Read any project file (shader/C++/header/JSON/config)

    Args:
        project_root: Path to PlasmaDX-Clean project
        file_path: Relative path from project root (e.g., "src/particles/ParticleSystem.h")

    Returns:
        File contents as string, or error message if file not found
    """
    try:
        full_path = Path(project_root) / file_path

        if not full_path.exists():
            return f"❌ Error: File not found: {file_path}\n\nFull path checked: {full_path}"

        if not full_path.is_file():
            return f"❌ Error: Path is not a file: {file_path}"

        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        file_size = len(content)
        line_count = content.count('\n') + 1

        return f"""✅ Successfully read file: {file_path}

File size: {file_size:,} bytes
Line count: {line_count:,} lines

=== FILE CONTENTS ===

{content}

=== END OF FILE ===
"""

    except Exception as e:
        return f"❌ Error reading file {file_path}: {str(e)}"


async def write_codebase_file(
    project_root: str,
    file_path: str,
    content: str,
    create_backup: bool = True
) -> str:
    """
    Write file with automatic backup to .backups/ directory

    Args:
        project_root: Path to PlasmaDX-Clean project
        file_path: Relative path from project root
        content: File content to write
        create_backup: Create timestamped backup before writing (default: True)

    Returns:
        Success/failure message with backup location
    """
    try:
        full_path = Path(project_root) / file_path

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        backup_path = None
        if create_backup and full_path.exists():
            backup_dir = Path(project_root) / ".backups"
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"{full_path.name}.{timestamp}.backup"
            backup_path = backup_dir / backup_filename

            shutil.copy2(full_path, backup_path)

        # Write new content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        file_size = len(content)
        line_count = content.count('\n') + 1

        result = f"✅ Successfully wrote file: {file_path}\n\n"
        result += f"File size: {file_size:,} bytes\n"
        result += f"Line count: {line_count:,} lines\n"

        if backup_path:
            result += f"\n📦 Backup created: .backups/{backup_path.name}"
        else:
            result += "\n(No backup created - file did not exist or backup disabled)"

        return result

    except Exception as e:
        return f"❌ Error writing file {file_path}: {str(e)}"


async def search_codebase(
    project_root: str,
    pattern: str,
    file_glob: str = "**/*",
    max_results: int = 100
) -> str:
    """
    Search for pattern in codebase files (grep-like functionality)

    Args:
        project_root: Path to PlasmaDX-Clean project
        pattern: Text pattern to search for (supports regex)
        file_glob: File pattern to search (default: all files)
                   Examples: "**/*.hlsl", "**/*.cpp", "src/**/*.h"
        max_results: Maximum number of matches to return

    Returns:
        Formatted search results with file paths and line numbers
    """
    import re

    try:
        project_path = Path(project_root)

        if not project_path.exists():
            return f"❌ Error: Project root not found: {project_root}"

        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"❌ Error: Invalid regex pattern '{pattern}': {str(e)}"

        # Search files
        matches = []
        files_searched = 0

        for file_path in project_path.glob(file_glob):
            if not file_path.is_file():
                continue

            # Skip binary files and certain directories
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'build', 'venv', '.dxil']):
                continue

            files_searched += 1

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, start=1):
                        if regex.search(line):
                            relative_path = file_path.relative_to(project_path)
                            matches.append({
                                'file': str(relative_path),
                                'line': line_num,
                                'content': line.rstrip()
                            })

                            if len(matches) >= max_results:
                                break

                if len(matches) >= max_results:
                    break

            except Exception:
                # Skip files that can't be read
                continue

        # Format results
        if not matches:
            return f"""🔍 Search Results: No matches found

Pattern: {pattern}
File glob: {file_glob}
Files searched: {files_searched:,}
"""

        result = f"""🔍 Search Results: {len(matches)} match(es) found

Pattern: {pattern}
File glob: {file_glob}
Files searched: {files_searched:,}

=== MATCHES ===

"""

        for match in matches:
            result += f"{match['file']}:{match['line']}\n"
            result += f"  {match['content']}\n\n"

        if len(matches) >= max_results:
            result += f"\n⚠️ Results truncated to {max_results} matches. Refine your search pattern for more specific results.\n"

        return result

    except Exception as e:
        return f"❌ Error searching codebase: {str(e)}"
