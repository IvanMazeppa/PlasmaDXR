# PlasmaDX-Clean Build Guide

## Two Build Configurations

### Debug (Daily Development)
**Fast iteration, zero PIX overhead**

```batch
# Build
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

# Run
.\build\Debug\PlasmaDX-Clean.exe
```

**Features:**
- ✅ Full D3D12 debug layer
- ✅ Zero PIX overhead (code removed at compile time)
- ✅ Fast startup and runtime
- ❌ No PIX capture support

**Use for:**
- Normal development work
- Code debugging with Visual Studio
- Quick testing iterations
- Performance testing without PIX overhead

---

### DebugPIX (PIX Captures)
**For GPU debugging and agent automation**

```batch
# Build
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Run
.\build\DebugPIX\PlasmaDX-Clean-PIX.exe

# Or use PIX scripts
.\pix\quick_capture_test.bat
```

**Features:**
- ✅ Full PIX capture support
- ✅ Auto-capture at specified frame
- ✅ Same debug settings as Debug build
- ⚠️ Slight overhead from PIX integration

**Use for:**
- GPU debugging with PIX
- Performance profiling
- Automated captures via agent
- Analyzing render pipeline

---

## Quick Commands

```batch
# Build both configurations
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Clean rebuild
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Rebuild
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64 /t:Rebuild

# Build from Visual Studio
# Just select "Debug" or "DebugPIX" from the configuration dropdown
```

## PIX Capture Quick Start

```batch
# Easiest way - use the test script
.\pix\quick_capture_test.bat

# The script will:
# 1. Set PIX_AUTO_CAPTURE=1 and PIX_CAPTURE_FRAME=5
# 2. Run PlasmaDX-Clean-PIX.exe
# 3. Capture at frame 5 and exit
# 4. Check logs/ folder for confirmation
```

## Command-Line Arguments

Both builds support the same arguments:

```batch
# Default: 10K particles, Gaussian renderer
.\build\Debug\PlasmaDX-Clean.exe

# Specify particle count
.\build\Debug\PlasmaDX-Clean.exe --particles 50000

# Use billboard renderer instead
.\build\Debug\PlasmaDX-Clean.exe --billboard

# Combine options
.\build\Debug\PlasmaDX-Clean.exe --particles 20000 --gaussian
```

## Which Build Should I Use?

### Use Debug When:
- ✅ Writing new features
- ✅ Debugging C++ code in Visual Studio
- ✅ Testing physics or rendering changes
- ✅ You want maximum performance
- ✅ You don't need GPU profiling

### Use DebugPIX When:
- ✅ Analyzing GPU performance
- ✅ Debugging shader issues
- ✅ Need PIX timeline or graphics captures
- ✅ Running automated capture scripts
- ✅ Investigating rendering problems

## Troubleshooting

### "PIX support disabled" in Debug build
✅ **Expected behavior!** Debug build has zero PIX code for maximum performance.
Use DebugPIX build if you need PIX.

### "GPU Capturer DLL NOT loaded" in DebugPIX build
✅ **Expected when running manually.** PIX captures only work when:
- Using PIX launch commands, OR
- Running from PIX UI, OR
- Using environment variables (PIX_AUTO_CAPTURE=1)

Try: `.\pix\quick_capture_test.bat`

### Application fails to load shader
Check that shaders are compiled:
```batch
# Shaders should be in:
.\build\Debug\shaders\        # For Debug build
.\build\DebugPIX\shaders\     # For DebugPIX build
```

### Want to rebuild everything
```batch
# Clean all build artifacts
rmdir /s /q build

# Rebuild Debug
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Rebuild

# Rebuild DebugPIX
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64 /t:Rebuild
```

## Log Files

Both builds create timestamped logs:
```
.\build\Debug\logs\PlasmaDX-Clean_YYYYMMDD_HHMMSS.log
.\build\DebugPIX\logs\PlasmaDX-Clean_YYYYMMDD_HHMMSS.log
```

PIX-related messages in DebugPIX logs:
```
[PIX] Initializing PIX capture system...
[PIX] GPU Capturer DLL loaded successfully
[PIX] Auto-capture ENABLED - will capture at frame 120
[PIX] Frame 120: Starting capture...
[PIX] Capture saved to: Captures/...
```

## See Also

- [PIX_DUAL_BINARY_SETUP.md](PIX_DUAL_BINARY_SETUP.md) - Technical implementation details
- [PIX_AGENT_V1_STATUS.md](PIX_AGENT_V1_STATUS.md) - PIX agent workflow documentation
- [README.md](README.md) - Project overview
