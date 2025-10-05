# Capture all debug modes using PIXTool.exe
param(
    [string]$PIXTool = "C:/Program Files/Microsoft PIX/2507.11/pixtool.exe",
    [string]$App = "build-vs2022b/Debug/PlasmaDX.exe",
    [string]$OutDir = "pix/Captures"
)

$ErrorActionPreference = "Stop"

# Resolve absolute paths
if (-not (Test-Path $PIXTool)) { $PIXTool = Join-Path (Get-Location) "pix/pixtool.exe" }
if (-not (Test-Path $PIXTool)) { Write-Host "ERROR: pixtool.exe not found under pix/" -ForegroundColor Red; exit 1 }
$PIXTool = (Resolve-Path $PIXTool).Path

if (-not (Test-Path $App)) { $App = Join-Path (Get-Location) $App }
if (-not (Test-Path $App)) { Write-Host "ERROR: PlasmaDX.exe not found at $App" -ForegroundColor Red; exit 1 }
$AppAbs = (Resolve-Path $App).Path
$AppDir = Split-Path -Parent $AppAbs

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDirAbs = (Resolve-Path $OutDir).Path
New-Item -ItemType Directory -Force -Path "pix/Reports" | Out-Null

$envs = @{
    "PLASMADX_DISABLE_DXR" = "1"
    "PLASMADX_NO_DEBUG" = "1"
    "PLASMADX_FILL_SPHERE" = "1"
}

foreach ($mode in 0,1,2,3,4,5) {
    $capture = Join-Path $OutDirAbs ("mode_${mode}.wpix")
    $outLog = Join-Path "pix/Reports" ("mode_${mode}.out.log")
    $errLog = Join-Path "pix/Reports" ("mode_${mode}.err.log")

    # Use official verbs: launch → take-capture → save-capture
    # Use parent environment injection via Start-Process -Environment so the launched app inherits variables
    $pixArgs = @("launch", $AppAbs, "take-capture", "save-capture", $capture)
    $envForChild = @{
        "PLASMADX_DEBUG_MODE" = "$mode";
        "PLASMADX_DISABLE_DXR" = $envs['PLASMADX_DISABLE_DXR'];
        "PLASMADX_NO_DEBUG" = $envs['PLASMADX_NO_DEBUG']
        "PLASMADX_FILL_SPHERE" = $envs['PLASMADX_FILL_SPHERE']
    }

    Write-Host "Capturing mode $mode to $capture" -ForegroundColor Cyan
    $p = Start-Process -FilePath $PIXTool -ArgumentList $pixArgs -NoNewWindow -PassThru -WorkingDirectory $AppDir -RedirectStandardOutput $outLog -RedirectStandardError $errLog -Environment $envForChild
    $p.WaitForExit()

    if (-not (Test-Path $capture)) {
        Write-Host "WARNING: Capture for mode $mode did not produce a .wpix file. See $outLog / $errLog" -ForegroundColor Yellow
    }
}

Write-Host "All captures complete: $OutDir" -ForegroundColor Green
