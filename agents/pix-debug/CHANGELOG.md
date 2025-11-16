# Changelog

All notable changes to the PIX Debugging Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.6] - 2025-11-01

### Added
- `analyze_dxil_root_signature` tool for detecting root signature mismatches
- `validate_shader_execution` tool for confirming shader execution via diagnostic counters
- Comprehensive documentation overhaul (README.md, TOOLS.md, KNOWN_ISSUES.md, CHANGELOG.md)

### Fixed
- GenerateCandidates root signature mismatch (Phase 1 stub code issue)
  - Simplified C++ root signature to match shader's actual Phase 1 needs (cb0 + u0 only)
  - Documented as temporary fix until Phase 2 random walk enabled

### Changed
- README.md restructured for clarity with separate TOOLS.md reference

### Known Issues
- PopulateVolumeMip2 not executing despite correct root signature (under investigation)
- analyze_restir_reservoirs outdated (32-byte vs 64-byte format)
- Duplicate tool registrations in mcp_server.py (cleanup needed)

---

## [0.1.5] - 2025-10-31

### Added
- `diagnose_gpu_hang` tool with threshold testing capability
- `--restir` command-line flag for PlasmaDX autonomous testing
- Autonomous crash threshold detection (e.g., "2044 works, 2045 crashes")

### Fixed
- Working directory issues causing shader loading failures
  - Tool now launches PlasmaDX from `build/bin/Debug/` directory
  - Shaders load correctly from relative paths

### Changed
- Switched from config-based control to command-line flags (more reliable)
- Added process termination via `taskkill.exe /F` (prevents orphaned processes)

---

## [0.1.4] - 2025-10-18

### Added
- Initial MCP server implementation with 6 basic tools:
  - `capture_buffers`
  - `analyze_restir_reservoirs`
  - `analyze_particle_buffers`
  - `pix_capture`
  - `pix_list_captures`
  - `diagnose_visual_artifact`

### Changed
- Upgraded from Claude Agent SDK v0.1.1 → v0.1.6
- Added explicit system prompt (required by SDK 0.1.6)

---

## [0.1.3] - 2025-10-13

### Added
- ReSTIR black dots bug analysis
  - Root cause: Low M values at far camera distances
  - Fix: Visibility scaling based on M value

### Fixed
- RTXDI weight calculation overflow
- ReSTIR temporal/spatial reuse bugs

---

## [0.1.2] - 2025-10-10

### Added
- PIX buffer dump parsing (binary struct analysis)
- Statistical analysis for ReSTIR reservoirs (W, M, weightSum)

---

## [0.1.1] - 2025-10-08

### Added
- Initial hybrid debugging approach (buffer dumps + PIX captures)
- Basic buffer validation tools

---

## [0.1.0] - 2025-10-05

### Added
- Project initialization
- MCP server scaffold
- Basic PIX integration

---

## Future Roadmap

### v0.2.0 (Planned)
- [ ] Update `analyze_restir_reservoirs` for 64-byte Volumetric ReSTIR format
- [ ] Fix duplicate tool registrations
- [ ] Automated PIX capture with PlasmaDX launch
- [ ] PIX .wpix file parsing (avoid manual GUI inspection)

### v0.3.0 (Planned)
- [ ] Real-time shader debugging (HLSL line-by-line)
- [ ] GPU performance profiling integration
- [ ] Automated regression testing suite

### v1.0.0 (Future)
- [ ] Complete Volumetric ReSTIR Phase 2-3 debugging support
- [ ] Multi-project DXR debugging support
- [ ] Web dashboard for debugging history

---

## Version Number Scheme

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes to tool interfaces or workflow
- **MINOR**: New tools, significant features, or major bug fixes
- **PATCH**: Small fixes, documentation updates, minor improvements

---

**Maintained by**: Claude Code automated debugging sessions  
**Repository**: Private (PlasmaDX-Clean development)
