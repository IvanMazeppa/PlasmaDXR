# Extract and analyze key data from PIX captures for PlasmaDX
param(
    [string]$PIXTool = "C:/Program Files/Microsoft PIX/2507.11/pixtool.exe",
    [string]$InDir = "pix/Captures",
    [string]$Out = "pix/workflow_claude/pix_analysis_report.md"
)

$ErrorActionPreference = "Stop"

# Ensure PIXTool path
if (-not (Test-Path $PIXTool)) { $PIXTool = Join-Path (Get-Location) "pix/pixtool.exe" }
if (-not (Test-Path $PIXTool)) {
    Write-Host "ERROR: PIXTool.exe not found" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$caps = @("off.wpix","RayDir.wpix","Bounds.wpix","DensityProbe.wpix",
          "mode_0.wpix","mode_1.wpix","mode_2.wpix","mode_3.wpix","mode_4.wpix","mode_5.wpix")

$report = @()
$report += "# PIX Capture Analysis Report for PlasmaDX (PIXTool)"
$report += "Generated: $(Get-Date)"
$report += ""

foreach ($c in $caps) {
    $file = Join-Path $InDir $c
    if (-not (Test-Path $file)) { continue }

    $report += "## Capture: $c"

    # Use documented commands: open-capture, save-event-list, save-screenshot
    $base = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $csv = Join-Path (Split-Path $Out) ("${base}_events.csv")
    $png = Join-Path (Split-Path $Out) ("${base}.png")

    & $PIXTool "open-capture" "$file" "save-event-list" "$csv" "--counters=*" "save-screenshot" "$png" | Out-Null

    $csvPath = (Resolve-Path $csv).Path
    $pngPath = (Resolve-Path $png).Path
    $report += ("- Saved event list: ``{0}``" -f $csvPath)
    $report += ("- Saved screenshot:  ``{0}``" -f $pngPath)
    $report += ""
}

$report | Out-File -FilePath $Out -Encoding UTF8
Write-Host "Analysis written to $Out" -ForegroundColor Green
