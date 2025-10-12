Write-Host "======================================" -ForegroundColor Cyan
Write-Host "PIX Capture - PowerShell Version" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$pixTool = "C:\Program Files\Microsoft PIX\2509.25\pixtool.exe"
$appExe = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe"
$workDir = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"
$captureFile = "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\test.wpix"

Write-Host "Step 1: Launching app with PIX..." -ForegroundColor Yellow
$launchArgs = @(
    "launch", "`"$appExe`"",
    "--command-line=`"--particles 10000 --gaussian`"",
    "--working-directory=`"$workDir`""
)

# Start pixtool launch in background
$pixProcess = Start-Process -FilePath $pixTool -ArgumentList $launchArgs -PassThru -NoNewWindow

Write-Host "App launching (PID: $($pixProcess.Id))..." -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Waiting 3 seconds for warmup..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Step 3: Taking capture..." -ForegroundColor Yellow
$captureArgs = @(
    "take-capture",
    "--frames=1",
    "--open",
    "save-capture", "`"$captureFile`""
)

& $pixTool $captureArgs

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Checking results..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

if (Test-Path $captureFile) {
    $fileSize = (Get-Item $captureFile).Length
    Write-Host "SUCCESS! Capture created!" -ForegroundColor Green
    Write-Host "  File: $captureFile" -ForegroundColor Green
    Write-Host "  Size: $fileSize bytes" -ForegroundColor Green

    if ($fileSize -lt 100000) {
        Write-Host "  WARNING: File is small (<100KB)" -ForegroundColor Yellow
    } else {
        Write-Host "  File size looks good!" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Open in PIX with:" -ForegroundColor Cyan
    Write-Host "  `"C:\Program Files\Microsoft PIX\2509.25\WinPix.exe`" `"$captureFile`"" -ForegroundColor White
} else {
    Write-Host "FAILED: No capture file created" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs\ folder for error messages" -ForegroundColor Yellow

    $latestLog = Get-ChildItem -Path "$workDir\logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        Write-Host "Latest log: $($latestLog.Name)" -ForegroundColor Yellow
    }
}

# Kill the app if still running
if (!$pixProcess.HasExited) {
    Write-Host ""
    Write-Host "Stopping app..." -ForegroundColor Yellow
    Stop-Process -Id $pixProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
