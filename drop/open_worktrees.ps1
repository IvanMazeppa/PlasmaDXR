# PlasmaDX Worktree Opener for Cursor (PowerShell version)
#
# Usage:
#   .\open_worktrees.ps1              - Opens ALL worktrees
#   .\open_worktrees.ps1 nanovdb      - Opens only nanovdb
#   .\open_worktrees.ps1 nanovdb blender - Opens nanovdb and blender
#   .\open_worktrees.ps1 -List        - List available worktrees
#
# Available worktrees:
#   main, nanovdb, blender, multiagent, pinn, gaussianq

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Worktrees,
    [switch]$List,
    [switch]$Help
)

$BASE = "/mnt/d/Users/dilli/AndroidStudioProjects"

$WorktreeMap = @{
    "main" = @{
        Host = "plasmadx-server"
        Path = "$BASE/PlasmaDX-Clean"
        Desc = "Main development (Clean)"
    }
    "nanovdb" = @{
        Host = "nanovdb"
        Path = "$BASE/PlasmaDX-NanoVDB"
        Desc = "NanoVDB volumetric rendering"
    }
    "blender" = @{
        Host = "blender"
        Path = "$BASE/PlasmaDX-Blender"
        Desc = "Blender VDB asset creation"
    }
    "multiagent" = @{
        Host = "multi-agent"
        Path = "$BASE/PlasmaDX-MultiAgent"
        Desc = "Multi-agent development"
    }
    "pinn" = @{
        Host = "pinn-v4"
        Path = "$BASE/PlasmaDX-PINN-v4"
        Desc = "Physics-Informed Neural Networks"
    }
    "gaussianq" = @{
        Host = "gaussianq"
        Path = "$BASE/PlasmaDX-GaussianImageQ"
        Desc = "Gaussian Image Quality analysis"
    }
}

function Show-Help {
    Write-Host ""
    Write-Host "=== PlasmaDX Worktree Opener ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\open_worktrees.ps1              - Opens ALL worktrees"
    Write-Host "  .\open_worktrees.ps1 nanovdb      - Opens only nanovdb"
    Write-Host "  .\open_worktrees.ps1 nanovdb blender - Opens multiple"
    Write-Host "  .\open_worktrees.ps1 -List        - List available worktrees"
    Write-Host ""
}

function Show-List {
    Write-Host ""
    Write-Host "=== Available Worktrees ===" -ForegroundColor Cyan
    Write-Host ""
    foreach ($key in $WorktreeMap.Keys | Sort-Object) {
        $wt = $WorktreeMap[$key]
        Write-Host "  $key" -ForegroundColor Green -NoNewline
        Write-Host " - $($wt.Desc)" -ForegroundColor Gray
    }
    Write-Host ""
}

function Open-Worktree {
    param([string]$Name)

    if (-not $WorktreeMap.ContainsKey($Name)) {
        Write-Host "Unknown worktree: $Name" -ForegroundColor Red
        Write-Host "Use -List to see available worktrees"
        return
    }

    $wt = $WorktreeMap[$Name]
    Write-Host "Opening $Name ($($wt.Host))..." -ForegroundColor Green

    $remoteUri = "ssh-remote+$($wt.Host)"
    Start-Process "cursor" -ArgumentList "--remote", $remoteUri, $wt.Path
}

# Main logic
if ($Help) {
    Show-Help
    exit
}

if ($List) {
    Show-List
    exit
}

if ($Worktrees.Count -eq 0) {
    # Open all worktrees
    Write-Host ""
    Write-Host "=== Opening ALL PlasmaDX Worktrees ===" -ForegroundColor Cyan
    Write-Host ""

    foreach ($key in @("main", "nanovdb", "blender", "multiagent", "pinn", "gaussianq")) {
        Open-Worktree -Name $key
        Start-Sleep -Seconds 2
    }
} else {
    # Open specified worktrees
    foreach ($wt in $Worktrees) {
        Open-Worktree -Name $wt
        Start-Sleep -Seconds 1
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
