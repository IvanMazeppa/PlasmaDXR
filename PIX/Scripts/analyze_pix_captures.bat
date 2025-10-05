@echo off
setlocal EnableDelayedExpansion

REM PIX Analysis Script for PlasmaDX Volume Rendering Debug
REM This script uses PIXTool.exe to extract key data from GPU captures

set PIX_PATH=C:\Program Files\Microsoft PIX
set PIXTOOL="%PIX_PATH%\PIXTool.exe"
set "SCRIPT_DIR=%~dp0"
set "CAPTURE_DIR=%SCRIPT_DIR%..\Captures"
set "OUTPUT_DIR=%SCRIPT_DIR%..\workflow_claude"

REM Check if PIXTool exists
if not exist %PIXTOOL% (
    echo ERROR: PIXTool.exe not found at %PIX_PATH%
    echo Please ensure PIX for Windows is installed
    echo Download from: https://devblogs.microsoft.com/pix/download/
    exit /b 1
)

echo ========================================
echo PIX Capture Analysis for PlasmaDX
echo ========================================
echo.

REM Create output directory if it doesn't exist
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Analyze each capture
set CAPTURES=off RayDir Bounds DensityProbe
set OUTPUT_FILE=%OUTPUT_DIR%\pix_analysis_report.txt

echo PIX Capture Analysis Report > %OUTPUT_FILE%
echo Generated: %date% %time% >> %OUTPUT_FILE%
echo ======================================== >> %OUTPUT_FILE%
echo. >> %OUTPUT_FILE%

for %%C in (%CAPTURES%) do (
    echo Analyzing %%C.wpix...

    set WPIX_FILE=%CAPTURE_DIR%\%%C.wpix
    if exist !WPIX_FILE! (
        echo. >> %OUTPUT_FILE%
        echo ======================================== >> %OUTPUT_FILE%
        echo Capture: %%C.wpix >> %OUTPUT_FILE%
        echo ======================================== >> %OUTPUT_FILE%

        REM Export timing data
        echo. >> %OUTPUT_FILE%
        echo GPU Timing Summary: >> %OUTPUT_FILE%
        %PIXTOOL% timing --capture "!WPIX_FILE!" >> %OUTPUT_FILE% 2>nul

        REM Export events list
        echo. >> %OUTPUT_FILE%
        echo Event Summary: >> %OUTPUT_FILE%
        %PIXTOOL% events --capture "!WPIX_FILE!" --filter "Dispatch" >> %OUTPUT_FILE% 2>nul

        REM Export resource list
        echo. >> %OUTPUT_FILE%
        echo Resource Bindings: >> %OUTPUT_FILE%
        %PIXTOOL% resources --capture "!WPIX_FILE!" >> %OUTPUT_FILE% 2>nul

        REM Export shader info
        echo. >> %OUTPUT_FILE%
        echo Shader Information: >> %OUTPUT_FILE%
        %PIXTOOL% shaders --capture "!WPIX_FILE!" >> %OUTPUT_FILE% 2>nul

        REM Export warnings/errors
        echo. >> %OUTPUT_FILE%
        echo Warnings and Errors: >> %OUTPUT_FILE%
        %PIXTOOL% warnings --capture "!WPIX_FILE!" >> %OUTPUT_FILE% 2>nul

    ) else (
        echo WARNING: !WPIX_FILE! not found
        echo WARNING: !WPIX_FILE! not found >> %OUTPUT_FILE%
    )
)

echo.
echo ========================================
echo Analysis complete!
echo Output saved to: %OUTPUT_FILE%
echo ========================================

REM Also try to export specific dispatch details
echo. >> %OUTPUT_FILE%
echo ======================================== >> %OUTPUT_FILE%
echo Detailed Dispatch Analysis >> %OUTPUT_FILE%
echo ======================================== >> %OUTPUT_FILE%

for %%C in (%CAPTURES%) do (
    echo. >> %OUTPUT_FILE%
    echo %%C.wpix Dispatches: >> %OUTPUT_FILE%
    set WPIX_FILE=%CAPTURE_DIR%\%%C.wpix
    %PIXTOOL% events --capture "!WPIX_FILE!" --filter "Dispatch" >> %OUTPUT_FILE% 2>nul
)

endlocal