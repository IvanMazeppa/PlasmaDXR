Write-Host "======================================" -ForegroundColor Cyan
Write-Host "PIX Capture - Attach Method" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$workDir = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"
$exe = "$workDir\build\Debug\PlasmaDX-Clean.exe"
$captureFile = "$workDir\pix\Captures\attach_test.wpix"
$pixtool = "C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"

Write-Host "Strategy: Launch app normally, wait for warmup, then attach PIX and capture" -ForegroundColor Yellow
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Executable: $exe"
Write-Host "  Working Dir: $workDir"
Write-Host "  Warmup: 3 seconds (~180 frames @ 60fps)"
Write-Host "  Capture: 1 frame"
Write-Host ""

# Check files exist
if (-not (Test-Path $exe)) {
    Write-Host "ERROR: Executable not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $pixtool)) {
    Write-Host "ERROR: pixtool.exe not found!" -ForegroundColor Red
    exit 1
}

# Delete old capture if exists
if (Test-Path $captureFile) {
    Remove-Item $captureFile -Force
    Write-Host "Deleted old capture file" -ForegroundColor Gray
}

Write-Host "Step 1: Launching app normally..." -ForegroundColor Green
Set-Location $workDir

try {
    $app = Start-Process -FilePath $exe `
                        -ArgumentList "--particles 10000 --gaussian" `
                        -WorkingDirectory $workDir `
                        -PassThru `
                        -NoNewWindow

    Write-Host "  App launched! PID: $($app.Id)" -ForegroundColor Green
    Write-Host ""

    Write-Host "Step 2: Waiting 3 seconds for warmup..." -ForegroundColor Green
    Write-Host "  This allows ~180 frames to render (well past frame 120)" -ForegroundColor Gray

    for ($i = 3; $i -gt 0; $i--) {
        Write-Host "  $i..." -NoNewline -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
    Write-Host " Ready!" -ForegroundColor Green
    Write-Host ""

    Write-Host "Step 3: Attaching PIX and capturing frame..." -ForegroundColor Green
    Write-Host "  Running: pixtool attach $($app.Id) take-capture --frames=1 --open save-capture" -ForegroundColor Gray

    $pixArgs = @(
        "attach", $app.Id,
        "take-capture", "--frames=1", "--open",
        "save-capture", "`"$captureFile`""
    )

    $pixOutput = & $pixtool $pixArgs 2>&1
    Write-Host $pixOutput
    Write-Host ""

    Write-Host "Step 4: Waiting for capture to complete..." -ForegroundColor Green
    Start-Sleep -Seconds 2

    Write-Host "Step 5: Stopping app..." -ForegroundColor Green
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  App stopped" -ForegroundColor Green
    Write-Host ""

} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($app -and !$app.HasExited) {
        Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    }
    exit 1
}

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Results" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $captureFile) {
    $fileSize = (Get-Item $captureFile).Length
    Write-Host "SUCCESS! Capture file created!" -ForegroundColor Green
    Write-Host "  File: $captureFile" -ForegroundColor Green
    Write-Host "  Size: $fileSize bytes ($([math]::Round($fileSize/1MB, 2)) MB)" -ForegroundColor Green

    if ($fileSize -lt 100000) {
        Write-Host ""
        Write-Host "  WARNING: File is small (<100KB) - may be incomplete" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "  File size looks good!" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Open in PIX with:" -ForegroundColor Cyan
    Write-Host "  `"C:\Program Files\Microsoft PIX\2509.25\WinPix.exe`" `"$captureFile`"" -ForegroundColor White

    # Check latest log
    $latestLog = Get-ChildItem -Path "$workDir\logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        Write-Host ""
        Write-Host "Latest log file: $($latestLog.Name)" -ForegroundColor Cyan
    }
} else {
    Write-Host "FAILED: No capture file created" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if app ran successfully (check logs folder)" -ForegroundColor Gray
    Write-Host "  2. Verify PIX can attach to running processes" -ForegroundColor Gray
    Write-Host "  3. Try running this script as Administrator" -ForegroundColor Gray

    $latestLog = Get-ChildItem -Path "$workDir\logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        Write-Host ""
        Write-Host "Latest log file: $($latestLog.Name)" -ForegroundColor Yellow
        Write-Host "Last 10 lines:" -ForegroundColor Yellow
        Get-Content $latestLog.FullName -Tail 10
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan