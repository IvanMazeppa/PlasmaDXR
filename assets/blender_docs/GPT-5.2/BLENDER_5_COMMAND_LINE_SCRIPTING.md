# Blender 5 — Command-Line Python Scripting (and how we use it for rapid debugging)

This doc exists so we can run Blender scripts **headless**, get the **full traceback** immediately, and also get a **non-zero exit code** on failure (so you can batch-test assets without copy/pasting errors back into chat).

All flags referenced below are from the Blender 5.0 Manual and Python API pages surfaced via the repo’s **Blender Manual MCP server**.

## Core sources (MCP server)

- **Command-line arguments (authoritative flag list)**: `advanced/command_line/arguments.html`
- **Command-line overview**: `advanced/command_line/index.html`
- **Rendering from CLI (ordering matters)**: `advanced/command_line/render.html`
- **Python API “Use the Terminal” tips**: `info_tips_and_tricks.html`

## The 3 flags that make rapid debugging work

From `advanced/command_line/arguments.html` → **Python Options**:

- **`--background` / `-b`**: run Blender with no UI (much faster/safer for automation).
- **`--python <filepath>` / `-P <filepath>`**: run a Python script file.
- **`--python-exit-code <code>`**: *critical* — sets the process exit code (0–255) when a Python exception is raised by a command-line script. Set it to **1** for CI-style behavior.

### Canonical “run a script and fail on exception”

```bash
"/mnt/e/Program Files/Blender Foundation/Blender 5.0/blender.exe" \
  --background \
  --python-exit-code 1 \
  --python /path/to/script.py
echo "exit=$?"
```

If the script throws, you’ll get:

- The **full traceback** printed to the terminal (per `info_tips_and_tricks.html`).
- A non-zero **process exit code** (`1` in the example).

## Passing your own arguments to a script

From `advanced/command_line/arguments.html`:

- **`--`** ends Blender option processing.
- Everything after `--` is available to Python via `sys.argv`.

### Example

```bash
"/mnt/e/Program Files/Blender Foundation/Blender 5.0/blender.exe" \
  --background \
  --python-exit-code 1 \
  --python /path/to/script.py -- \
  --output_dir ./build/out --bake 0
```

Inside your script, the robust pattern is:

```python
import sys
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
```

All GPT‑5.2 asset scripts in `assets/blender_scripts/GPT-5.2/` follow this convention.

## Capturing logs: Blender logger + stdout/stderr

From `advanced/command_line/arguments.html` → **Logging Options**:

- **`--log-level <level>`**: `fatal|error|warning|info|debug|trace`
- **`--log <match>`**: enable categories (e.g. `"render,cycles"` or `"*"`).
- **`--log-file <filepath>`**: write Blender’s internal log output to a file.

You still want to capture `print()` output and tracebacks; easiest is shell redirection:

```bash
blender ... 2>&1 | tee run_stdout_stderr.txt
```

### WSL note (Windows Blender build)

If you’re calling a **Windows** `blender.exe` from **WSL**, avoid passing WSL-only paths like
`/mnt/d/...` to flags that expect a file path (notably `--log-file`). The Windows build won’t
resolve those paths and will print an error.

Use either:

- a **repo-relative path** (recommended): `--log-file build/blender_cli_logs/run/blender.log`
- or a **Windows path** like `D:\...` (escaping rules apply in shells)

The repo runner script below uses a relative `--log-file` automatically.

## Extra debug flags that help when Blender “hard-crashes”

From `advanced/command_line/arguments.html` → **Debug Options**:

- **`--debug-python`**: extra Python debug messages.
- **`--debug-wm`**: logs all operators (useful when an operator poll fails unexpectedly).
- **`--debug-exit-on-error`**: exit immediately when internal errors are detected.
- **`--disable-crash-handler`**: disables crash handler (useful when you want raw crash output for debugging).

These are noisy—use them on demand, not by default.

## Important gotcha: argument ordering

From `advanced/command_line/render.html`:

- **Arguments execute in the order they’re given**.
- Keep `-f` or `-a` last when rendering from CLI.

This matters less for pure scripting, but if you mix scripting + rendering flags, order can break expectations.

## Repo workflow: the “one command” runner we added

To make this painless, GPT‑5.2 added a WSL-friendly runner script:

- `assets/blender_scripts/GPT-5.2/run_blender_cli.sh`

It:

- writes **per-run** logs under `build/blender_cli_logs/…`
- passes `--python-exit-code 1`
- captures stdout/stderr with `tee`
- supports `--ui`, `--blend`, and log flags

### Setup (once)

```bash
chmod +x assets/blender_scripts/GPT-5.2/run_blender_cli.sh
export BLENDER_EXE="/mnt/e/Program Files/Blender Foundation/Blender 5.0/blender.exe"
```

### Examples (fast validation; no bake, no render)

```bash
assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
  assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
  --bake 0 --render_still 0 --render_anim 0
```

```bash
assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
  assets/blender_scripts/GPT-5.2/blender_bipolar_planetary_nebula.py -- \
  --bake 0 --render_still 0 --render_anim 0
```

### Reading failures

When something fails:

- First open the run folder under `build/blender_cli_logs/...`
- Check `stdout_stderr.txt` for the Python traceback
- If Blender logging is enabled, check `blender.log` for engine/device detail

This is the fastest way to iterate on scripts without copy/pasting errors into chat.


