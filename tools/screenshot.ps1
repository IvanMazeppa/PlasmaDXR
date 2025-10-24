# Screenshot capture script for PlasmaDX
# Saves screenshots to project directory with timestamp
# Bind to keyboard shortcut: Ctrl+Shift+S

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Project screenshot directory
$projectDir = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\screenshots"

# Create directory if it doesn't exist
if (-not (Test-Path $projectDir)) {
    New-Item -ItemType Directory -Path $projectDir | Out-Null
}

# Get all screens (multi-monitor support)
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds

# Create bitmap
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

# Capture screen
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)

# Generate filename with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$filename = "screenshot_$timestamp.png"
$filepath = Join-Path $projectDir $filename

# Save screenshot
$bitmap.Save($filepath, [System.Drawing.Imaging.ImageFormat]::Png)

# Cleanup
$graphics.Dispose()
$bitmap.Dispose()

# Show notification
Add-Type -AssemblyName System.Windows.Forms
$notification = New-Object System.Windows.Forms.NotifyIcon
$notification.Icon = [System.Drawing.SystemIcons]::Information
$notification.BalloonTipText = "Screenshot saved to $filename"
$notification.BalloonTipTitle = "PlasmaDX Screenshot"
$notification.Visible = $true
$notification.ShowBalloonTip(2000)

Start-Sleep -Seconds 2
$notification.Dispose()

Write-Host "Screenshot saved: $filepath"
