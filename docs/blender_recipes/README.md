# Celestial Body VDB Recipe Library

**Purpose:** Production-ready recipes for creating volumetric celestial phenomena in Blender 5.0 for export to PlasmaDX-Clean.

**Target User:** Programmers familiar with code but new to Blender's interface.

---

## Quick Start

1. **Pick a recipe** from the categories below
2. **Follow the workflow** step-by-step in Blender
3. **Run the Python script** to automate creation
4. **Bake and export** VDB files
5. **Load in PlasmaDX** and render in real-time

---

## Recipe Categories

### Emission Nebulae
Glowing gas clouds that emit light (H-II regions, reflection nebulae).

| Recipe | Difficulty | Method | Status |
|--------|------------|--------|--------|
| [Hydrogen Cloud](emission_nebulae/hydrogen_cloud.md) | Beginner | Mantaflow | Planned |
| [Emission Pillar](emission_nebulae/emission_pillar.md) | Intermediate | Geometry Nodes | Planned |
| [Orion-Style Complex](emission_nebulae/orion_style.md) | Advanced | Hybrid | Planned |

### Explosions & Transients
Energetic events: supernovae, stellar flares, coronal mass ejections.

| Recipe | Difficulty | Method | Status |
|--------|------------|--------|--------|
| [Supernova Remnant](explosions/supernova_remnant.md) | Intermediate | Mantaflow | Planned |
| [Stellar Flare](explosions/stellar_flare.md) | Intermediate | Geometry Nodes | Planned |
| [Coronal Mass Ejection](explosions/coronal_ejection.md) | Advanced | Animated GeoNodes | Planned |

### Stellar Phenomena
Structures around and between stars: disks, coronae, jets.

| Recipe | Difficulty | Method | Status |
|--------|------------|--------|--------|
| [Protoplanetary Disk](stellar_phenomena/protoplanetary_disk.md) | Intermediate | Geometry Nodes | Planned |
| [Accretion Corona](stellar_phenomena/accretion_corona.md) | Advanced | Mantaflow + GeoNodes | Planned |
| [Planetary Nebula](stellar_phenomena/planetary_nebula.md) | Intermediate | Mantaflow | Planned |

### Dark Structures
Absorption-dominated volumes: dark nebulae, dust lanes.

| Recipe | Difficulty | Method | Status |
|--------|------------|--------|--------|
| [Dark Nebula](dark_structures/dark_nebula.md) | Beginner | Volume Object | Planned |
| [Dust Lane](dark_structures/dust_lane.md) | Intermediate | Geometry Nodes | Planned |

---

## Universal Scripts

Automation scripts that work across multiple recipe types.

| Script | Purpose |
|--------|---------|
| [quick_smoke_setup.py](scripts/quick_smoke_setup.py) | Create basic smoke domain + emitter |
| [vdb_export_batch.py](scripts/vdb_export_batch.py) | Batch export multiple simulations |
| [celestial_presets.py](scripts/celestial_presets.py) | Material presets for celestial bodies |

---

## VDB Export Quick Reference

### Required Settings for PlasmaDX Compatibility

```
Cache Type:        MODULAR or ALL
Data Format:       OpenVDB
Compression:       BLOSC (fastest) or ZIP (smallest)
Precision:         HALF (16-bit) recommended
```

### Grid Requirements by Body Type

| Body Type | Density | Temperature | Velocity | Notes |
|-----------|---------|-------------|----------|-------|
| Emission Nebula | Required | Optional | Optional | Temperature for color variation |
| Dark Nebula | Required | No | No | Absorption only |
| Supernova | Required | Required | Required | Full dynamics |
| Stellar Flare | Required | Required | Optional | High temperature |
| Disk | Required | Optional | Optional | Rotation useful |

### Resolution Guidelines

| Detail Level | Resolution | File Size (per frame) | Use Case |
|--------------|------------|----------------------|----------|
| Preview | 64³ | 2-5 MB | Testing workflow |
| Standard | 128³ | 10-30 MB | Most use cases |
| High Detail | 256³ | 50-150 MB | Hero shots |
| Maximum | 512³ | 200-500 MB | Close-ups only |

---

## Blender → PlasmaDX Material Mapping

How Blender volume properties translate to PlasmaDX material types:

| Blender Property | PlasmaDX Property | Mapping |
|------------------|-------------------|---------|
| Density | Opacity | Direct (0-1 range) |
| Blackbody Intensity | Emission Multiplier | Scale by 10x |
| Temperature | Temperature | Direct (Kelvin) |
| Anisotropy | Phase Function G | Direct (-1 to +1) |
| Absorption Color | Albedo RGB | Inverted |
| Scatter Color | Albedo RGB | Direct |

### Material Type Recommendations

| Celestial Body | PlasmaDX Material | Key Properties |
|----------------|-------------------|----------------|
| Hot Gas Cloud | GAS_CLOUD | Low opacity, backward scattering |
| Dark Nebula | DUST | High opacity, forward scattering |
| Stellar Corona | PLASMA | High emission, low scattering |
| Supernova Shell | PLASMA | High emission, outward motion |

---

## Coordinate System Notes

- **Blender:** Z-up, right-handed, meters
- **PlasmaDX:** Y-up (DirectX convention), right-handed

### Conversion Required
When loading Blender VDB in PlasmaDX:
- Rotate -90° around X axis, OR
- Swap Y and Z coordinates in loader

---

## Troubleshooting

### VDB Not Appearing in PlasmaDX
1. Check cache format is OpenVDB (not UniCache)
2. Verify VDB file exists in cache directory
3. Check grid names match expected (density, temperature)

### Simulation Takes Forever
1. Reduce resolution (try 64 first)
2. Enable Adaptive Domain
3. Reduce frame count for testing

### VDB File Too Large
1. Use BLOSC compression
2. Use HALF precision
3. Reduce resolution
4. Export only needed grids (density only for absorption)

### Colors Look Wrong in PlasmaDX
1. Check temperature grid is being read
2. Verify material type mapping
3. Adjust emission multiplier

---

## Contributing New Recipes

To add a new recipe:

1. Create markdown file in appropriate category folder
2. Follow the recipe template (see celestial-body-curator agent docs)
3. Include Python automation script
4. Test full workflow: Blender → VDB → PlasmaDX
5. Add entry to this README

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-07 | Initial library structure |

---

*Library maintained by: Claude Code celestial-body-curator agent*
*Last Updated: 2025-12-07*
