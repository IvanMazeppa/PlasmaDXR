# Blender 5.0 Mantaflow headless bake crash (WSL → Windows `blender.exe`)

While generating NanoVDB example assets for PlasmaDX-Clean, we discovered a **repeatable Blender 5.0 crash** when attempting Mantaflow baking in headless mode:

- Invocation: `blender.exe --background --python <script> -- --bake 1`
- Result: `EXCEPTION_ACCESS_VIOLATION` inside Blender during fluid bake.

## Symptoms

- Blender exits with code `11`.
- Crash log written to Windows temp, e.g.:
  - `C:\\Users\\dilli\\AppData\\Local\\Temp\\GPT-5-2_HydrogenCloud.crash.txt`
  - `C:\\Users\\dilli\\AppData\\Local\\Temp\\GPT-5-2_SupergiantStar.crash.txt`

## Crash signature (from crash.txt)

The stack trace consistently shows Mantaflow/Manta parsing:

- `MANTA::parseLine`
- `MANTA::parseScript`
- `MANTA::initSmoke`
- `fluid_bake_startjob`

## Impact on our scripts

Because the crash occurs **inside Blender**, we cannot reliably generate Mantaflow bakes via CLI right now.

As a safety measure, GPT‑5.2 Mantaflow scripts now:

- default `--bake` to `0`
- **skip bake in headless** unless `--allow_headless_bake 1` is explicitly provided (still likely to crash)

Scripts affected:
- `assets/blender_scripts/GPT-5.2/blender_supergiant_star.py`
- `assets/blender_scripts/GPT-5.2/blender_bipolar_planetary_nebula.py`
- `assets/blender_scripts/GPT-5.2/blender_hydrogen_cloud.py`

## Workable alternative for “example assets”

For the NanoVDB plugin/refactor, we can still produce **excellent example `.nvdb` assets** without Mantaflow baking by converting known-good `.vdb` sources:

- Use `nanovdb_convert` (available as `/usr/bin/nanovdb_convert` in WSL)
- Convert only the `density` grid:

```bash
nanovdb_convert -z -f -g density input.vdb output.nvdb
```

Curated examples are stored (repo-tracked) here:

- `assets/volumes/nanovdb_examples/GPT-5.2/`


