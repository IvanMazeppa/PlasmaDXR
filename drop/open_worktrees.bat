@echo off
setlocal enabledelayedexpansion

REM === PlasmaDX Worktree Opener for Cursor ===
REM
REM Usage:
REM   open_worktrees.bat           - Opens ALL worktrees
REM   open_worktrees.bat nanovdb   - Opens only nanovdb
REM   open_worktrees.bat nanovdb blender - Opens nanovdb and blender
REM
REM Available worktrees:
REM   main, nanovdb, blender, multiagent, pinn, gaussianq

set "BASE=/mnt/d/Users/dilli/AndroidStudioProjects"

REM Define worktree mappings
set "main_host=plasmadx-server"
set "main_path=%BASE%/PlasmaDX-Clean"

set "nanovdb_host=nanovdb"
set "nanovdb_path=%BASE%/PlasmaDX-NanoVDB"

set "blender_host=blender"
set "blender_path=%BASE%/PlasmaDX-Blender"

set "multiagent_host=multi-agent"
set "multiagent_path=%BASE%/PlasmaDX-MultiAgent"

set "pinn_host=pinn-v4"
set "pinn_path=%BASE%/PlasmaDX-PINN-v4"

set "gaussianq_host=gaussianq"
set "gaussianq_path=%BASE%/PlasmaDX-GaussianImageQ"

REM Check if any arguments provided
if "%~1"=="" goto open_all

REM Open specific worktrees from arguments
:parse_args
if "%~1"=="" goto end

call :open_worktree %1
shift
goto parse_args

:open_all
echo.
echo === Opening ALL PlasmaDX Worktrees in Cursor ===
echo.
call :open_worktree main
timeout /t 2 /nobreak >nul
call :open_worktree nanovdb
timeout /t 2 /nobreak >nul
call :open_worktree blender
timeout /t 2 /nobreak >nul
call :open_worktree multiagent
timeout /t 2 /nobreak >nul
call :open_worktree pinn
timeout /t 2 /nobreak >nul
call :open_worktree gaussianq
goto end

:open_worktree
set "name=%~1"
set "host=!%name%_host!"
set "path=!%name%_path!"

if "!host!"=="" (
    echo Unknown worktree: %name%
    echo Available: main, nanovdb, blender, multiagent, pinn, gaussianq
    goto :eof
)

echo Opening %name% (%host%)...
start "" cursor --remote ssh-remote+%host% "%path%"
goto :eof

:end
echo.
echo Done!
