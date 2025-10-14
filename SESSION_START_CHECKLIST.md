# Claude Session Start Checklist
**Purpose:** Quick reference for Claude at the start of each debugging session
**Last Updated:** 2025-10-14

---

## Critical: Read These Files First

Before starting any task, READ these files to understand the system:

1. **[configs/README.md](configs/README.md)** - Config system structure and usage
2. **[PIX/README.md](PIX/README.md)** - Autonomous buffer dumping system
3. **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Complete parameter reference
4. **This file** - Quick start checklist

---

## Project Structure Quick Reference

```
PlasmaDX-Clean/
│
├── configs/                              # Configuration system (NEW!)
│   ├── user/default.json                 # User's daily config
│   ├── builds/Debug.json                 # Debug build default
│   ├── builds/DebugPIX.json              # PIX agent build default
│   ├── agents/pix_agent.json             # PIX debugging agent config
│   └── scenarios/*.json                  # Test scenario configs
│
├── PIX/                                  # Autonomous debugging system
│   ├── buffer_dumps/                     # GPU buffer dumps (auto-generated)
│   ├── scripts/analysis/                 # Python analysis tools
│   └── docs/                             # PIX system documentation
│
├── src/                                  # C++ source code
│   ├── core/Application.cpp              # Main application loop
│   ├── config/Config.cpp                 # Config loading system
│   └── particles/                        # Particle rendering system
│
├── shaders/                              # HLSL shaders
│   └── particles/particle_gaussian_raytrace.hlsl  # Main ReSTIR shader
│
└── build/                                # Build outputs
    ├── Debug/                            # Daily development build
    └── DebugPIX/                         # PIX-instrumented build
```

---

## Key System Details

### 1. Configuration System

**⚠️ IMPORTANT:** As of 2025-10-14, configs are in `configs/` directory, NOT root!

**User's daily config:** `configs/user/default.json`
**PIX agent config:** `configs/agents/pix_agent.json`
**Debug build default:** `configs/builds/Debug.json`

**Config loading priority:**
1. Command-line: `--config=<path>`
2. Environment: `PLASMADX_CONFIG=<path>`
3. Build directory: `./config.json`
4. User default: `configs/user/default.json`

**When agents need to modify config:**
- ✅ **DO:** Modify `configs/agents/pix_agent.json` (agent's own config)
- ❌ **DON'T:** Modify `configs/user/default.json` (user's config)
- ❌ **DON'T:** Modify `configs/builds/*.json` (build defaults)

---

### 2. PIX Autonomous Debugging System

**Status:** ✅ Production-ready (as of 2025-10-13)

**How it works:**
- Application can dump GPU buffers directly (no PIX GUI needed)
- Python scripts analyze dumps automatically
- 2-second captures vs 5-10 minutes manual PIX extraction

**To capture buffers:**
```bash
./build/DebugPIX/PlasmaDX-Clean.exe --dump-buffers 120 --gaussian
```

**To analyze:**
```bash
python PIX/scripts/analysis/analyze_restir_manual.py \
    --current PIX/buffer_dumps/g_currentReservoirs.bin \
    --prev PIX/buffer_dumps/g_prevReservoirs.bin
```

**When debugging ReSTIR:**
1. User disables ReSTIR in config (it's broken)
2. Agent can re-enable it in `configs/agents/pix_agent.json`
3. Launch with agent config: `--config=configs/agents/pix_agent.json`

---

### 3. Build System

**Two build configurations:**

| Build | Purpose | Config | Features |
|-------|---------|--------|----------|
| **Debug** | Daily development | `configs/builds/Debug.json` | D3D12 debug ON, PIX OFF, ReSTIR OFF |
| **DebugPIX** | PIX agent debugging | `configs/builds/DebugPIX.json` | D3D12 debug OFF, PIX ON, ReSTIR ON |

**Build locations:**
- Debug: `build/Debug/PlasmaDX-Clean.exe`
- DebugPIX: `build/DebugPIX/PlasmaDX-Clean.exe`

**⚠️ Important:**
- User uses **Debug** build for daily work
- PIX agent uses **DebugPIX** build for analysis
- Don't mix them up!

---

### 4. ReSTIR Status (As of 2025-10-14)

**Current State:**
- ✅ Dots reduced (after 5 fixes)
- ⚠️ Other issues remain (RT lighting, control freezing)
- ❌ Not production-ready yet

**Known Issues:**
1. **RT Intensity Bug:** Increasing intensity makes particles MORE MUTED (line 673 clamp issue)
2. **Control Freezing:** Input freezes when ReSTIR enabled (line 576 blocking wait)
3. **ReSTIR Effect:** Doesn't create intended effect yet

**Fixes Applied:**
1. ✅ MIS weight formula (W × M)
2. ✅ Distance-adaptive temporal weight
3. ✅ Increased clamp to 10.0
4. ✅ Weaker attenuation (linear falloff)
5. ✅ 100× boost to misWeight

**Files Modified:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` (lines 357, 486-490, 645, 655, 659)

**Agent Findings:**
- Agent 2: RT intensity bug at line 673 (clamp to 10.0 needs increase to 100.0)
- Agent 3: Control freezing at line 576 of Application.cpp (blocking WaitForGPU)
- Agent 4: Needs new captures with ReSTIR enabled for analysis

---

## Common Tasks

### Task: Debug ReSTIR Issue

1. **Read:** `configs/README.md` and `PIX/README.md`
2. **Check:** Current ReSTIR status (see section 4 above)
3. **Verify:** Agent config has ReSTIR enabled (`configs/agents/pix_agent.json`)
4. **Capture:** Use `--dump-buffers 120` with DebugPIX build
5. **Analyze:** Run Python analysis scripts on dumps
6. **Fix:** Modify shader, recompile, test

### Task: Test Multiple Scenarios

1. **Use scenario configs:** `configs/scenarios/*.json`
2. **Batch test:**
   ```bash
   for scenario in configs/scenarios/*.json; do
       ./build/Debug/PlasmaDX-Clean.exe --config=$scenario --dump-buffers 120
   done
   ```
3. **Compare results:** Use `analyze_5_captures.py`

### Task: Agent Autonomous Analysis

1. **Agent modifies:** `configs/agents/pix_agent.json`
2. **Agent launches:** `--config=configs/agents/pix_agent.json --dump-buffers 120`
3. **Agent analyzes:** Run Python scripts on dumps
4. **Agent reports:** Findings and recommended fixes

---

## Critical Don'ts

### ❌ DON'T: Modify User's Config

User's config is sacred! If agent needs different settings:
- ✅ Copy to `configs/agents/pix_agent.json`
- ✅ Modify agent config
- ✅ Launch with `--config=configs/agents/pix_agent.json`

### ❌ DON'T: Assume ReSTIR Works

ReSTIR is currently disabled by default because it's broken. If you need to test it:
- Enable in agent config
- Use DebugPIX build
- Capture buffers for analysis

### ❌ DON'T: Mix Build Configs

Debug and DebugPIX have different configs and purposes:
- Debug: User's daily driver (D3D12 debug, no PIX)
- DebugPIX: Agent debugging (PIX instrumentation, no D3D12 debug)

### ❌ DON'T: Forget to Read Docs

Before starting any task:
1. Read `configs/README.md` (config system)
2. Read `PIX/README.md` (buffer dumping)
3. Read this checklist
4. Check current ReSTIR status (section 4)

---

## Shader Recompilation

**When modifying shaders, ALWAYS recompile:**

```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
    -T cs_6_5 -E main \
    "shaders/particles/particle_gaussian_raytrace.hlsl" \
    -Fo "shaders/particles/particle_gaussian_raytrace.dxil"
```

**Verify compilation:**
```bash
ls -lh shaders/particles/particle_gaussian_raytrace.dxil
# Check timestamp is recent
```

**Common mistake:** Testing without recompiling → changes don't take effect!

---

## Agent Deployment (Parallel)

When deploying multiple agents, use single message with multiple Task calls:

```
User: "Deploy agents in parallel to analyze ReSTIR"

Claude: I'll deploy 3 agents in parallel...
<function_calls>
  <invoke name="Task">...</invoke>
  <invoke name="Task">...</invoke>
  <invoke name="Task">...</invoke>