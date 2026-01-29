#!/usr/bin/env python3
"""Inject minimal bake validation prints into a Blender script.

This is a surgical helper for debugging: it only operates on a copy of a
generated script and inserts a small diagnostics block immediately after the
first bake call.
"""

from __future__ import annotations

from pathlib import Path


SNIPPET = """
    # ==== DEBUG: Bake Cache Validation ====
    try:
        _cache_dir = None
        for _obj in bpy.data.objects:
            for _mod in _obj.modifiers:
                if _mod.type == 'FLUID' and hasattr(_mod, 'domain_settings') and _mod.domain_settings:
                    _cache_dir = bpy.path.abspath(_mod.domain_settings.cache_directory)
                    break
            if _cache_dir:
                break
        print(f"[DEBUG] cache_directory: {_cache_dir}")
        if _cache_dir and os.path.isdir(_cache_dir):
            _counts = {}
            for _root, _dirs, _files in os.walk(_cache_dir):
                if _files:
                    _counts[_root] = len(_files)
            print(f"[DEBUG] cache file counts: {_counts}")
    except Exception as _e:
        print(f"[DEBUG] cache validation failed: {_e}")
    # ==== END DEBUG: Bake Cache Validation ====
"""


def inject_after_first_bake(content: str) -> tuple[str, bool]:
    needle = "bpy.ops.fluid.bake"
    idx = content.find(needle)
    if idx == -1:
        return content, False

    # Insert after the first bake call line.
    line_end = content.find("\n", idx)
    if line_end == -1:
        return content, False

    # Determine indent from the bake line.
    line_start = content.rfind("\n", 0, idx) + 1
    indent = ""
    while line_start < len(content) and content[line_start] in (" ", "\t"):
        indent += content[line_start]
        line_start += 1

    snippet = "\n".join(
        indent + line if line.strip() else line
        for line in SNIPPET.strip("\n").split("\n")
    )
    updated = content[: line_end + 1] + snippet + "\n" + content[line_end + 1 :]
    return updated, True


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: blender_add_bake_validation.py <script.py>")
        sys.exit(1)

    path = Path(sys.argv[1])
    content = path.read_text()
    updated, changed = inject_after_first_bake(content)
    if not changed:
        print("No bake call found; no changes made.")
        return
    path.write_text(updated)
    print(f"Injected bake validation into: {path}")


if __name__ == "__main__":
    main()
