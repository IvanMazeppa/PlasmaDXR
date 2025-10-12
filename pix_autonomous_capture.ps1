#
# PIX Autonomous Capture Script
# Launches app, waits for warmup, captures frame, analyzes
#

param(
    [int]$WarmupSeconds = 2,
    [int]$Frames = 1,
    [string]$OutputName = "autonomous_capture"
)

$ErrorActionPreference = "Stop"

$projectRoot = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"
$pixTool = "C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"
$appExe = "$projectRoot\build\Debug\PlasmaDX-Clean.exe"
$captureFile = "$projectRoot\pix\Captures\$OutputName.wpix"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PIX Autonomous Capture v1.0" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Warmup: $WarmupSeconds seconds"
Write-Host "  Frames to capture: $Frames"
Write-Host "  Output: $OutputName.wpix"
Write-Host "  Renderer: Gaussian (10K particles)"
Write-Host ""

# Step 1: Launch app under PIX (keeps running)
Write-Host "Step 1: Launching app under PIX..." -ForegroundColor Yellow
$launchArgs = @(
    "launch", "`"$appExe`"",
    "--command-line=`"--particles 10000 --gaussian`"",
    "--working-directory=`"$projectRoot`""
)

$pixJob = Start-Job -ScriptBlock {
    param($tool, $args)
    & $tool $args
} -ArgumentList $pixTool, $launchArgs

Write-Host "  App launched (Job ID: $($pixJob.Id))" -ForegroundColor Green

# Give app time to start
Start-Sleep -Milliseconds 500

# Step 2: Wait for warmup
Write-Host ""
Write-Host "Step 2: Waiting for warmup ($WarmupSeconds seconds)..." -ForegroundColor Yellow
for ($i = 1; $i -le $WarmupSeconds; $i++) {
    Write-Host "  $i..." -NoNewline
    Start-Sleep -Seconds 1
}
Write-Host " Done!" -ForegroundColor Green

# Step 3: Find the app process
Write-Host ""
Write-Host "Step 3: Finding app process..." -ForegroundColor Yellow
$appProcess = Get-Process -Name "PlasmaDX-Clean" -ErrorAction SilentlyContinue
if ($appProcess) {
    $pid = $appProcess.Id
    Write-Host "  Found PID: $pid" -ForegroundColor Green
} else {
    Write-Host "  ERROR: App process not found!" -ForegroundColor Red
    Stop-Job $pixJob
    Remove-Job $pixJob
    exit 1
}

# Step 4: Attach and capture
Write-Host ""
Write-Host "Step 4: Capturing frame..." -ForegroundColor Yellow
$captureArgs = @(
    "attach", "$pid",
    "take-capture", "--frames=$Frames",
    "save-capture", "`"$captureFile`""
)

& $pixTool $captureArgs

# Step 5: Clean up - stop the app
Write-Host ""
Write-Host "Step 5: Stopping app..." -ForegroundColor Yellow
Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
Stop-Job $pixJob -ErrorAction SilentlyContinue
Remove-Job $pixJob -ErrorAction SilentlyContinue
Write-Host "  App stopped" -ForegroundColor Green

# Step 6: Verify capture
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Results" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $captureFile) {
    $fileSize = (Get-Item $captureFile).Length
    $fileSizeMB = [math]::Round($fileSize / 1MB, 2)

    Write-Host "SUCCESS! Capture created!" -ForegroundColor Green
    Write-Host "  File: $captureFile" -ForegroundColor White
    Write-Host "  Size: $fileSizeMB MB ($fileSize bytes)" -ForegroundColor White

    if ($fileSize -lt 100KB) {
        Write-Host "  WARNING: File is very small (<100KB)" -ForegroundColor Yellow
    } elseif ($fileSize -lt 1MB) {
        Write-Host "  WARNING: File is small (<1MB)" -ForegroundColor Yellow
    } else {
        Write-Host "  File size looks good!" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Open in PIX:" -ForegroundColor Cyan
    Write-Host "  WinPix.exe `"$captureFile`"" -ForegroundColor White

    Write-Host ""
    Write-Host "Analyze with pixtool:" -ForegroundColor Cyan
    Write-Host "  pixtool.exe open-capture `"$captureFile`" list-counters" -ForegroundColor White
    Write-Host "  pixtool.exe open-capture `"$captureFile`" save-event-list events.csv" -ForegroundColor White

} else {
    Write-Host "FAILED: No capture file created" -ForegroundColor Red

    # Check logs
    $latestLog = Get-ChildItem -Path "$projectRoot\logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        Write-Host ""
        Write-Host "Latest log: $($latestLog.Name)" -ForegroundColor Yellow
        Write-Host "Last 10 lines:" -ForegroundColor Yellow
        Get-Content $latestLog.FullName -Tail 10 | ForEach-Object { Write-Host "  $_" }
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
Write-Host ""
