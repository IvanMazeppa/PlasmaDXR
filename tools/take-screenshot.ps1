<#
.SYNOPSIS
    PlasmaDX Screenshot Helper for Windows/WSL

.DESCRIPTION
    Captures full-screen screenshot and outputs WSL-compatible path for Claude Code.
    Screenshots saved to: C:\Users\dilli\Pictures\PlasmaDX-Screenshots\

.PARAMETER Description
    Optional description to include in filename (default: "screenshot")

.EXAMPLE
    powershell.exe -File tools/take-screenshot.ps1 "rtxdi-sphere-preset"
    # Outputs: /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi-sphere-preset_20251019_192345.png

.EXAMPLE
    powershell.exe -File tools/take-screenshot.ps1
    # Outputs: /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/screenshot_20251019_192345.png
#>

param(
    [Parameter(Position=0)]
    [string]$Description = "screenshot"
)

# Configuration
$BaseDir = "C:\Users\dilli\Pictures\PlasmaDX-Screenshots"
$Username = "dilli"

# Ensure directory exists
if (-not (Test-Path $BaseDir)) {
    New-Item -ItemType Directory -Path $BaseDir | Out-Null
    Write-Host "Created directory: $BaseDir"
}

# Generate filename with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeDescription = $Description -replace '[\\/:*?"<>|]', '_'  # Remove invalid chars
$Filename = "${SafeDescription}_${Timestamp}.png"
$FullPath = Join-Path $BaseDir $Filename

# Load required assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

try {
    # Capture entire primary screen
    $Screen = [System.Windows.Forms.Screen]::PrimaryScreen
    $Bounds = $Screen.Bounds
    $Bitmap = New-Object System.Drawing.Bitmap $Bounds.Width, $Bounds.Height
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)

    # Copy screen to bitmap
    $Graphics.CopyFromScreen($Bounds.Location, [System.Drawing.Point]::Empty, $Bounds.Size)

    # Save to file
    $Bitmap.Save($FullPath, [System.Drawing.Imaging.ImageFormat]::Png)

    # Cleanup
    $Graphics.Dispose()
    $Bitmap.Dispose()

    # Output results
    $WSLPath = "/mnt/c/Users/$Username/Pictures/PlasmaDX-Screenshots/$Filename"

    Write-Host ""
    Write-Host "Screenshot saved successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Windows Path:" -ForegroundColor Cyan
    Write-Host "  $FullPath"
    Write-Host ""
    Write-Host "WSL Path (for Claude Code):" -ForegroundColor Yellow
    Write-Host "  $WSLPath"
    Write-Host ""
    Write-Host "Copy the WSL path above and paste into Claude Code chat."
    Write-Host ""

} catch {
    Write-Host "Error capturing screenshot:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}
