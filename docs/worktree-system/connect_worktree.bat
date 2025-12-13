@echo off
echo.
echo === PlasmaDX Worktree Connector (Terminal) ===
echo.
echo 1. plasmadx-server (Clean/Main)
echo 2. nanovdb
echo 3. blender
echo 4. multi-agent
echo 5. pinn-v4
echo 6. gaussianq
echo.
set /p choice="Enter number (or host name): "

if "%choice%"=="1" goto plasmadx
if "%choice%"=="2" goto nanovdb
if "%choice%"=="3" goto blender
if "%choice%"=="4" goto multiagent
if "%choice%"=="5" goto pinn
if "%choice%"=="6" goto gaussianq

REM Allow direct host name entry
ssh %choice%
goto end

:plasmadx
ssh plasmadx-server
goto end

:nanovdb
ssh nanovdb
goto end

:blender
ssh blender
goto end

:multiagent
ssh multi-agent
goto end

:pinn
ssh pinn-v4
goto end

:gaussianq
ssh gaussianq
goto end

:end
