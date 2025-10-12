@echo off
echo ==========================================
echo PIX Capture with Warmup
echo ==========================================
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

set WARMUP_SECONDS=3
set CAPTURE_FRAMES=1

echo Configuration:
echo   Warmup: %WARMUP_SECONDS% seconds
echo   Frames: %CAPTURE_FRAMES%
echo   Renderer: Gaussian (10K particles)
echo.

REM Step 1: Launch app under PIX (in background)
echo Step 1: Launching app under PIX...
start /B "" "C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"

REM Give app time to start
timeout /t 1 /nobreak >nul

REM Step 2: Wait for warmup
echo Step 2: Waiting for warmup (%WARMUP_SECONDS% seconds)...
timeout /t %WARMUP_SECONDS% /nobreak

REM Step 3: Find process ID
echo Step 3: Finding app process...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq PlasmaDX-Clean.exe" /FO LIST ^| findstr /C:"PID:"') do set APP_PID=%%i

if "%APP_PID%"=="" (
    echo ERROR: App process not found!
    goto :error
)

echo   Found PID: %APP_PID%

REM Step 4: Attach and capture
echo Step 4: Capturing frame...
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" attach %APP_PID% take-capture --frames=%CAPTURE_FRAMES% save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\warmup_capture.wpix"

REM Step 5: Kill app
echo Step 5: Stopping app...
taskkill /PID %APP_PID% /F >nul 2>&1

echo.
echo ==========================================
echo Results
echo ==========================================
echo.

if exist "pix\Captures\warmup_capture.wpix" (
    for %%A in ("pix\Captures\warmup_capture.wpix") do (
        echo SUCCESS! Capture created!
        echo   File: pix\Captures\warmup_capture.wpix
        echo   Size: %%~zA bytes
        echo.
        if %%~zA LSS 1000000 (
            echo   WARNING: File is small (^<1MB^)
        ) else (
            echo   File size looks good!
        )
    )
    echo.
    echo Open in PIX:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\warmup_capture.wpix
) else (
    echo FAILED: No capture file created
    echo.
    echo Check latest log:
    dir /b /o-d logs\*.log | findstr /n "^" | findstr "^1:"
)

echo.
pause
exit /b 0

:error
echo.
pause
exit /b 1
