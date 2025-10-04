@echo off
echo ========================================
echo PlasmaDX-Clean Build Script
echo Clean architecture, automatic fallbacks
echo ========================================

REM Create build directory
if not exist build-vs2022 mkdir build-vs2022
cd build-vs2022

REM Generate Visual Studio solution
echo Generating Visual Studio 2022 solution...
cmake -G "Visual Studio 17 2022" -A x64 ..
if errorlevel 1 (
    echo CMake generation failed!
    pause
    exit /b 1
)

REM Build the project
echo Building Debug configuration...
cmake --build . --config Debug
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

REM Copy Agility SDK files
echo Copying Agility SDK files...
if not exist bin\Debug\D3D12 mkdir bin\Debug\D3D12
copy ..\external\D3D12\*.dll bin\Debug\D3D12\ >nul 2>&1

echo ========================================
echo Build complete!
echo Executable: bin\Debug\PlasmaDX-Clean.exe
echo ========================================

REM Ask to run
choice /M "Run PlasmaDX-Clean now?"
if errorlevel 2 goto :end
if errorlevel 1 goto :run

:run
echo Starting PlasmaDX-Clean...
cd bin\Debug
PlasmaDX-Clean.exe
cd ..\..

:end
pause