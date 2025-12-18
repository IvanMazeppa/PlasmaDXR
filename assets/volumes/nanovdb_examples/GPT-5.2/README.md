# GPT-5.2 NanoVDB Examples

These `.nvdb` assets are generated via:

- Blender scripts: `assets/blender_scripts/GPT-5.2/`
- CLI runner: `assets/blender_scripts/GPT-5.2/run_blender_cli.sh`
- Conversion tool: `nanovdb_convert` (installed in WSL as `/usr/bin/nanovdb_convert`)

The goal is **maximum compatibility** with PlasmaDX-Clean’s file-loaded NanoVDB path:
- convert **only** the `density` grid (`nanovdb_convert -g density ...`)
- avoid quantization flags (`--fp16/--fp8/...`) unless engine support is confirmed


