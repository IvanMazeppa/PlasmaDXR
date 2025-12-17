# Blender Scripts (canonical)

This folder is the canonical home for **Blender Python scripts** and helper runners used to generate/bake volumetric assets (OpenVDB → NanoVDB pipeline).

## Layout

- `GPT-5.2/`: Scripts authored/maintained by the GPT‑5.2 agent (single-file scripts + CLI runner).
- `shared/`: Cross-author scripts (or scripts maintained by multiple people/agents).
- `tools/`: Utility scripts for inspection/export/conversion that are intended to be run in Blender.

## Running scripts

All asset scripts support command-line execution:

```bash
blender -b -P assets/blender_scripts/GPT-5.2/<script>.py -- --help
```

For WSL + Windows `blender.exe`, use the runner in `assets/blender_scripts/GPT-5.2/run_blender_cli.sh`.


