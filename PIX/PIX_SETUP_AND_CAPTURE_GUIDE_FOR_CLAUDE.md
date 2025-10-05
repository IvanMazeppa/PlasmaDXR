## PIX on Windows: Complete Setup and Capture Guide (for Claude to instruct Dilli)

Audience: Claude Code will read this and guide Dilli step‑by‑step to capture the right GPU frames in PIX v2507.11 for PlasmaDX.

Targets: Windows 11, NVIDIA RTX 4060 Ti, D3D12 + compute path (DXR disabled for this phase), Agility SDK 1.616 runtime (stable). The same flow applies to future Agility versions when re‑enabled.

Key external reference:
- Getting Started with the Agility SDK (requirements, pitfalls, D3D12SDKVersion/Path, header ordering, known issues) — https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/

---

### 1) Verify prerequisites

Claude: Confirm these items with Dilli before capture.

1. Operating system supports Agility model
   - Windows 10 1909+ or Windows 11. Confirm `D3D12Core.dll` exists in `C:\Windows\System32`.
   - See “OS with Agility SDK support” in the Agility guide: https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
2. GPU driver up to date (NVIDIA 466.11+ recommended).
3. Visual Studio installed (Community is fine).
4. PIX for Windows v2507.11 installed.
5. DirectX Shader Compiler (dxcompiler/dxil DLLs) present next to the exe (CMake copies from `bin/`).

Notes about Agility integration (read carefully):
- PlasmaDX exports `D3D12SDKVersion` and `D3D12SDKPath` in `src/core/D3D12AgilitySDK.cpp`.
- The Agility redist (D3D12Core.dll and optional d3d12SDKLayers.dll) must be in a subfolder next to the exe matching `D3D12SDKPath` (we use `D3D12\`). Do not place them next to the exe itself. Mismatch between `D3D12Core.dll` and SDKLayers can cause device creation to fail. Source: Agility “Known issues” — https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/

For this session we use the stable setup:
- `D3D12SDKVersion = 616` and `D3D12SDKPath = "D3D12\\"` (already configured).

---

### 2) Build location and runtime files

Claude: Ensure Dilli has a Debug build at:
- `D:\Users\dilli\AndroidStudioProjects\PlasmaDX\build-vs2022b\Debug\PlasmaDX.exe`
- The folder `build-vs2022b\Debug\D3D12\` contains Agility DLLs (D3D12Core.dll, d3d12SDKLayers.dll).
- `build-vs2022b\Debug\dxcompiler.dll` and `dxil.dll` exist.

If missing, rebuild from the repo root:
```
cmake --build build-vs2022b --config Debug
```

---

### 3) Recommended environment variables for this phase

Claude: In PIX “Launch Process”, set these environment variables.

- `PLASMADX_NO_DEBUG=1`  // disable D3D12 debug layer for clean runs (enable later if needed)
- `PLASMADX_NO_QUIT_ON_REMOVAL=1`  // keep window open on device removal (optional)
- `PLASMADX_DISABLE_DXR=1`  // force compute-only path (we’re validating VOL_0003)

Rationale: We’re debugging compute marching and density fill without DXR variability.

---

### 4) Launching a GPU capture (PIX v2507.11 UI)

Claude: Walk Dilli through these exact clicks in PIX.

1. Open PIX → Home.
2. Click “Launch Process”. In “Win32” tab set:
   - Executable Path: `...\build-vs2022b\Debug\PlasmaDX.exe`
   - Working Directory: `...\PlasmaDX`
   - Environment Variables: add the three above.
   - Capture Type: “GPU Capture”.
   - Options: check “Capture at Launch”; “Enable DRED Logging” optional.
3. Click “Next”. In “Capture Options” set:
   - Frame Count: `1` (fast iteration). Delimiter: `Present‑to‑Present`.
4. Click “Launch”. PlasmaDX opens, PIX attaches, and a capture is taken immediately.

Tip: If you prefer to capture live, uncheck “Capture at Launch”, click the camera icon (or F12) after the app starts.

---

### 5) What to capture for VOL_0003 diagnostics

Claude: We need four 1‑frame captures, each with a different debug mode. Dilli will press F4 in the app to cycle modes.

Make one capture for each of:
1. Off (normal ray march)
2. RayDir visualization
3. Bounds/AABB visualization
4. DensityProbe visualization

Between captures, instruct Dilli to tap F4 once to advance to the next mode, then click the capture button.

---

### 6) How to inspect the capture

Open the `.wpix` capture tab that appears. Use the three panes: Events (left), State (right), Resources (bottom‑left).

Identify the ray march dispatch:
- In Events → filter “Dispatch”. Look for the dispatch where threadgroups ≈ screen/16 (e.g., `(120, 68, 1)` for 1920×1080 @ 16×16 threads). That’s `RayMarcher::March`.
- Another dispatch with `(16, 16, 16)` is the 3D density fill (analytic sphere baseline).

Inspect constants bound to the marcher:
1. Select the marcher Dispatch in Events.
2. In State pane → under RootSignature (Compute), click CBV 0 (CameraConstants) and CBV 1 (VolumeConstants).
3. Verify key fields in CBV 1:
   - `g_volumeMin ≈ (-1, -1, -1)`, `g_volumeMax ≈ (1, 1, 1)`
   - `g_screenSize ≈ (width, height)`
   - `densityScale`, `absorption`, `stepSize`, `maxSteps`, `exposure` look reasonable
   - `g_debugMode` matches the mode for this capture: Off=0, RayDir=1, Bounds=2, DensityProbe=3

Inspect resources:
1. In Resources List (bottom‑left), under CS, confirm:
   - `SRV Texture 0 : g_density` (3D texture)
   - `UAV Texture 0 : g_hdrTarget` (R16G16B16A16_FLOAT)
   - `Sampler 0 : g_trilinearSampler`
2. Right‑click `g_density` → View History. You should see an earlier compute dispatch that writes it (sphere fill). Open the texture viewer and scrub slices to confirm a solid sphere when using the analytic baseline.
3. Right‑click `g_hdrTarget` → View History to see the marcher write.

Check barriers:
- Select the events immediately before/after the marcher dispatch. Confirm a UAV barrier for `g_hdrTarget` and transitions around HDR (UAV↔SRV) and density as needed. The debug layer message “Before state ... does not match ...” indicates a missing/incorrect transition earlier in the frame.

Interpreting the debug modes:
- RayDir (1): RGB encodes normalized ray direction; should rotate smoothly with camera.
- Bounds (2): magenta = miss; green = hit when starting outside; yellow = ray starts inside volume.
- DensityProbe (3): grayscale based on a single density sample at entry point; for the analytic sphere you should see a clean circular silhouette.

---

### 7) Saving and sharing captures

Claude: Ask Dilli to save each capture via File → Save As → `.wpix` and name them `raydir.wpix`, `bounds.wpix`, `probe.wpix`, `off.wpix`.

Optional: Export Event List text (Tools → Export) if you want a lightweight diff.

---

### 8) Common pitfalls and quick fixes

Device creation fails (Agility):
- Ensure `D3D12SDKVersion` export matches the DLL in `D3D12\`. Keep SDKLayers and Core from the same package. Trailing slash required in `D3D12SDKPath`. Do not place the Agility DLLs next to the exe — use a subfolder. Source: https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/

Capture shows “deprecated – use pix3.h instead” markers:
- Harmless; our code uses legacy `pix.h` markers. We can migrate to `pix3.h` later to clean up labels.

No 3D density writes in history:
- The analytic sphere fill didn’t run that frame. Ensure the code path is executed or trigger a capture right after switching modes.

Uniform fog output:
- Use Bounds (2) to verify you’re not inside the AABB; back up with `S` or shrink the volume.
- Use DensityProbe (3) to confirm non‑uniform density at entry.

PIX capture not triggering / app not starting:
- Run PIX as Administrator.
- Close overlays (GeForce Experience, screen recorders).
- If Agility was just swapped, reboot once; stale handles can cause CreateDevice failure.

---

### 9) Minimal report Claude should request from each capture

Claude: For each `.wpix`, record these yes/no items plus short notes:
1. Marcher dispatch present with expected threadgroups? (Y/N)
2. CBV 1 values sane and `g_debugMode` matches mode? (Y/N)
3. `g_density` history shows a compute write this frame? (Y/N)
4. `g_hdrTarget` has a UAV write from marcher? (Y/N)
5. Any barrier warnings in the Events around marcher? (paste the message)
6. Visual outcome: RayDir smooth? Bounds mostly green/yellow? Probe shows a circle?

---

### 10) Appendix: Re‑enabling Agility 1.717.1 later (optional)

If we attempt the newer Agility again:
1. Update `src/core/D3D12AgilitySDK.cpp` → `D3D12SDKVersion = 717` and ensure `D3D12SDKPath = "D3D12\\"`.
2. Place `D3D12Core.dll` and `d3d12SDKLayers.dll` from the 1.717.1 NuGet into `build‑vs2022b\Debug\D3D12\` (and `bin\D3D12\` for CMake copy step).
3. Replace headers with the NuGet include folder to avoid ABI/define mismatches.
4. Rebuild, try first with `PLASMADX_NO_DEBUG=1`. If it boots, re‑enable the debug layer.
5. If `CreateDevice` fails, confirm no architecture mismatch (x64 vs ARM64) and reboot once.

Reference and known issues: https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/

---

### 11) One‑page quickstart (Claude can read aloud)

1. Open PIX → Launch Process.
2. Set exe to `build‑vs2022b\Debug\PlasmaDX.exe`; working dir = repo root.
3. Add env: `PLASMADX_NO_DEBUG=1`, `PLASMADX_NO_QUIT_ON_REMOVAL=1`, `PLASMADX_DISABLE_DXR=1`.
4. GPU Capture → Next → Frame Count 1 → Launch.
5. Take four captures while pressing F4 between them: Off, RayDir, Bounds, DensityProbe.
6. For each, select the marcher Dispatch, inspect CBV 1 fields, verify `g_debugMode` and `screenSize`.
7. View History on `g_density` (SRV0) and `g_hdrTarget` (UAV0).
8. Save as `.wpix` files and send.

---

End of guide.


