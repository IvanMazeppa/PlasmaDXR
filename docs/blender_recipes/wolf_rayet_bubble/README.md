# Wolf-Rayet Bubble Nebula - Complete Recipe Package

**Created:** 2025-12-15
**Author:** Claude Code (Opus 4.5)
**Status:** ✅ Complete

---

## Contents

| File | Description |
|------|-------------|
| `blender_wolf_rayet_bubble.py` | Main automation script (Blender 5.0+) |
| `wolf_rayet_bubble.md` | Full recipe documentation |
| `README.md` | This file |

---

## What This Creates

A **Wolf-Rayet bubble nebula** - a multi-shell structure created by intense stellar winds from a Wolf-Rayet star interacting with material from previous mass-loss epochs.

### Visual Characteristics
- Bubble/shell morphology (like NGC 6888, Sharpless 308)
- Blue-green OIII emission color
- Asymmetric "break-out" structures
- Clumpy, filamentary detail

### Key Innovation: Three Wind Model
This recipe implements the astrophysically-accurate Three Wind Model:
1. **Inner fast wind** - Current WR star wind (high velocity, low density)
2. **Outer slow shell** - Previous RSG phase wind (low velocity, high density)
3. **Break-out jet** - Asymmetric structure where fast wind punches through

---

## Quick Start

```bash
# From repo root:
blender -b -P docs/blender_recipes/wolf_rayet_bubble/blender_wolf_rayet_bubble.py -- \
  --output_dir "/path/to/output" \
  --name "WolfRayetBubble" \
  --resolution 96 \
  --frame_end 120 \
  --bake 1
```

### Output Structure
```
/path/to/output/
├── WolfRayetBubble.blend      # Blender project file
├── vdb_cache/                  # OpenVDB sequence
│   ├── fluid_data_0001.vdb
│   ├── fluid_data_0002.vdb
│   └── ...
└── renders/                    # Optional rendered frames
    └── WolfRayetBubble_still.png
```

---

## Parameters

### Essential
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | (required) | Output directory path |
| `--resolution` | 96 | Voxel resolution (64-256) |
| `--frame_end` | 120 | Animation length |
| `--bake` | 1 | Run bake (0=skip) |

### Shape
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--domain_size` | 8.0 | Domain cube size |
| `--bubble_radius` | 2.5 | Emitter arrangement radius |
| `--enable_breakout` | 1 | Include asymmetric breakout |

### TDR Safety
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tdr_safe` | 1 | Apply TDR-safe render settings |
| `--render_still` | 0 | Render single frame (risky on GPU) |
| `--cycles_device` | CPU | Use CPU to avoid TDR |

---

## NanoVDB Conversion

After baking, convert to NanoVDB for PlasmaDX:

```bash
python scripts/convert_vdb_to_nvdb.py \
  "/path/to/output/vdb_cache/fluid_data_0080.vdb" \
  "/path/to/output/WolfRayetBubble.nvdb" \
  --grid density
```

---

## Recipe Comparison

| Feature | Supergiant Star | Bipolar Nebula | Wolf-Rayet Bubble |
|---------|----------------|----------------|-------------------|
| Structure | Solid sphere | Jets + ring | Hollow bubble |
| Emitters | 2 | 3 | 3 |
| Color | Orange-red | Cyan-magenta | Blue-green |
| Method | Turbulent interior | Bipolar jets | Three Wind Model |

---

## Astronomical Sources

- [Wolf-Rayet star - Wikipedia](https://en.wikipedia.org/wiki/Wolf%E2%80%93Rayet_star)
- [Wolf-Rayet nebula - Wikipedia](https://en.wikipedia.org/wiki/Wolf%E2%80%93Rayet_nebula)
- [WISE morphological study (A&A 2015)](https://www.aanda.org/articles/aa/full_html/2015/06/aa25706-15/aa25706-15.html)

---

## See Also

- [Full Recipe Documentation](wolf_rayet_bubble.md)
- [TDR Safe Workflow](../GPT-5-2_Scripts_Docs_Advice/TDR_SAFE_WORKFLOW.md)
- [Main Recipe Index](../README.md)
