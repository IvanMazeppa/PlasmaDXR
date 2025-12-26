#!/usr/bin/env bash
# GPT-5.2 — Blender 5 CLI runner (WSL-friendly)
#
# Why:
# - Run Blender Python scripts from the command line with:
#   - full stdout/stderr capture
#   - an explicit non-zero exit code on Python exceptions (--python-exit-code)
#   - Blender log output written to a per-run folder (--log-file)
#
# Sources (Blender 5.0 Manual / MCP):
# - advanced/command_line/arguments.html  (Python Options, Logging Options, Debug Options)
#
# Usage:
#   chmod +x assets/blender_scripts/GPT-5.2/run_blender_cli.sh
#   assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
#     assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
#     --bake 0 --render_still 0 --render_anim 0
#
# Optional flags:
#   --blender <path>         Override Blender executable (default uses BLENDER_EXE env or a common local path).
#   --blend <file.blend>     Open a .blend before executing the script.
#   --ui                     Run with UI (no --background).
#   --no-factory-startup     Don't pass --factory-startup (use your normal startup settings/addons).
#   --python-exit-code <n>   Exit code to use when Python raises (default: 1). Use 0 to disable.
#   --log-level <level>      fatal|error|warning|info|debug|trace (default: info).
#   --log <match>            Enable logging categories (e.g. "render,cycles" or "*").
#
# Everything after `--` is passed to your Python script as args (Blender places them in sys.argv).
#
# Output:
#   build/blender_cli_logs/<timestamp>_<script>/stdout_stderr.txt
#   build/blender_cli_logs/<timestamp>_<script>/blender.log
#
set -euo pipefail

SCRIPT=""
BLEND_FILE=""
BACKGROUND=1
FACTORY_STARTUP=1
PY_EXIT_CODE=1
LOG_LEVEL="${BLENDER_LOG_LEVEL:-info}"
LOG_MATCH="${BLENDER_LOG_MATCH:-}"

DEFAULT_BLENDER_EXE="/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender"
BLENDER_EXE="${BLENDER_EXE:-$DEFAULT_BLENDER_EXE}"

usage() {
  cat <<'EOF'
run_blender_cli.sh — run a Blender 5 Python script from CLI with logs + exit codes.

Usage:
  run_blender_cli.sh [options] <script.py> [-- <script args...>]

Options:
  --blender <path>         Blender executable (default: $BLENDER_EXE or /mnt/e/.../blender.exe)
  --blend <file.blend>     Open blend file before running the script
  --ui                     Run with UI (no --background)
  --no-factory-startup     Do not pass --factory-startup
  --python-exit-code <n>   Exit code for Python exceptions (default: 1, 0 disables)
  --log-level <level>      fatal|error|warning|info|debug|trace (default: info)
  --log <match>            Logging categories match, e.g. "render,cycles" or "*"
  -h, --help               Show this help

Examples:
  # Quick validate (no bake / no render):
  run_blender_cli.sh assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
    --bake 0 --render_still 0 --render_anim 0

  # With a .blend loaded first:
  run_blender_cli.sh --blend myscene.blend assets/.../script.py -- --some_arg 123
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blender)
      BLENDER_EXE="$2"
      shift 2
      ;;
    --blend)
      BLEND_FILE="$2"
      shift 2
      ;;
    --ui)
      BACKGROUND=0
      shift
      ;;
    --no-factory-startup)
      FACTORY_STARTUP=0
      shift
      ;;
    --python-exit-code)
      PY_EXIT_CODE="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --log)
      LOG_MATCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
      ;;
    *)
      if [[ -z "$SCRIPT" ]]; then
        SCRIPT="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        echo "Tip: put script args after --" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$SCRIPT" ]]; then
  echo "Missing <script.py>." >&2
  usage
  exit 2
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "Script not found: $SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$BLENDER_EXE" ]]; then
  echo "Blender executable not found: $BLENDER_EXE" >&2
  echo "Set BLENDER_EXE or pass --blender <path>." >&2
  exit 2
fi

if [[ -n "$BLEND_FILE" && ! -f "$BLEND_FILE" ]]; then
  echo "Blend file not found: $BLEND_FILE" >&2
  exit 2
fi

find_repo_root() {
  # Find the repo root by walking up until we hit a common marker.
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/.git" || -f "$d/CMakeLists.txt" || -f "$d/settings.gradle" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

ROOT="$(find_repo_root || true)"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: Could not locate repo root (expected .git/CMakeLists.txt/settings.gradle)." >&2
  exit 2
fi

LOG_DIR_REL="build/blender_cli_logs"
LOG_DIR="${ROOT}/${LOG_DIR_REL}"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
SCRIPT_BASE="$(basename "$SCRIPT")"
SCRIPT_BASE="${SCRIPT_BASE%.py}"
RUN_DIR_REL="${LOG_DIR_REL}/${TS}_${SCRIPT_BASE}"
RUN_DIR="${ROOT}/${RUN_DIR_REL}"
mkdir -p "$RUN_DIR"

STDOUT_LOG="${RUN_DIR}/stdout_stderr.txt"
# IMPORTANT (WSL → Windows blender.exe):
# - Blender (Windows) does not understand WSL paths like /mnt/d/... for --log-file.
# - Use a repo-relative path so Blender resolves it from the current working directory.
BLENDER_LOG_FILE_REL="${RUN_DIR_REL}/blender.log"
BLENDER_LOG_FILE="${RUN_DIR}/blender.log"

echo "[run_blender_cli] ROOT:      $ROOT"
echo "[run_blender_cli] Blender:   $BLENDER_EXE"
echo "[run_blender_cli] Script:    $SCRIPT"
echo "[run_blender_cli] Blend:     ${BLEND_FILE:-<none>}"
echo "[run_blender_cli] Run dir:   $RUN_DIR"
echo "[run_blender_cli] Log file:  $BLENDER_LOG_FILE_REL"
echo "[run_blender_cli] Stdout:    $STDOUT_LOG"

CMD=( "$BLENDER_EXE" )

if [[ "$BACKGROUND" -eq 1 ]]; then
  CMD+=( --background )
fi

if [[ "$FACTORY_STARTUP" -eq 1 ]]; then
  CMD+=( --factory-startup )
fi

# Logging (Blender's internal logger)
if [[ -n "$LOG_LEVEL" ]]; then
  CMD+=( --log-level "$LOG_LEVEL" )
fi
if [[ -n "$LOG_MATCH" ]]; then
  CMD+=( --log "$LOG_MATCH" )
fi
CMD+=( --log-file "$BLENDER_LOG_FILE_REL" )

# Non-zero exit on Python exception for command-line scripts.
CMD+=( --python-exit-code "$PY_EXIT_CODE" )

_abs_path() {
  # Print absolute path for an existing file/dir without relying on realpath(1).
  local p="$1"
  if [[ -d "$p" ]]; then (cd "$p" && pwd); return; fi
  if [[ -f "$p" ]]; then (cd "$(dirname "$p")" && printf '%s/%s\n' "$(pwd)" "$(basename "$p")"); return; fi
  echo "$p"
}

_rel_to_root_if_possible() {
  # If the path is inside $ROOT, print it as a relative path (so Windows Blender can open it).
  local p_abs="$1"
  case "$p_abs" in
    "$ROOT"/*) echo "${p_abs#$ROOT/}" ;;
    *) echo "$p_abs" ;;
  esac
}

# Script path:
# - Prefer repo-relative paths so Windows Blender can open the file.
SCRIPT_ABS="$(_abs_path "$SCRIPT")"
SCRIPT_FOR_BLENDER="$(_rel_to_root_if_possible "$SCRIPT_ABS")"

# Optional blend path:
BLEND_FOR_BLENDER=""
if [[ -n "$BLEND_FILE" ]]; then
  BLEND_ABS="$(_abs_path "$BLEND_FILE")"
  BLEND_FOR_BLENDER="$(_rel_to_root_if_possible "$BLEND_ABS")"
fi

# Optional .blend to open (use converted path).
if [[ -n "$BLEND_FOR_BLENDER" ]]; then
  CMD+=( "$BLEND_FOR_BLENDER" )
fi

# Script + script args (use converted path).
CMD+=( --python "$SCRIPT_FOR_BLENDER" -- )
CMD+=( "${SCRIPT_ARGS[@]}" )

echo
echo "[run_blender_cli] Running:"
printf '  %q' "${CMD[@]}"
echo
echo

set +e
( cd "$ROOT" && "${CMD[@]}" ) 2>&1 | tee "$STDOUT_LOG"
EC="${PIPESTATUS[0]}"
set -e

echo
echo "[run_blender_cli] Exit code: $EC"
echo "[run_blender_cli] Outputs in: $RUN_DIR"

exit "$EC"


