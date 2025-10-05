# PIX Detailed Analysis Script for PlasmaDX
# This script extracts comprehensive data from PIX captures for GPT-5 analysis

$PIXPath = "C:\Program Files\Microsoft PIX"
$PIXTool = "$PIXPath\PIXTool.exe"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CaptureDir = Join-Path $ScriptDir "..\Captures"
$OutputDir = Join-Path $ScriptDir "..\workflow_claude"

# Check PIXTool exists
if (-not (Test-Path $PIXTool)) {
    Write-Host "ERROR: PIXTool.exe not found at $PIXPath" -ForegroundColor Red
    Write-Host "Please ensure PIX for Windows is installed"
    Write-Host "Download from: https://devblogs.microsoft.com/pix/download/"
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PIX Capture Analysis for PlasmaDX" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Captures = @("off", "RayDir", "Bounds", "DensityProbe")
$Report = @()
$Report += "# PIX Capture Analysis Report for PlasmaDX Volume Rendering"
$Report += "Generated: $(Get-Date)"
$Report += ""
$Report += "## Summary"
$Report += "This report analyzes 4 PIX GPU captures to diagnose volume rendering visibility issues."
$Report += ""

# Function to run PIXTool and capture output
function Invoke-PIXTool {
    param(
        [string]$Arguments
    )

    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $PIXTool
    $pinfo.Arguments = $Arguments
    $pinfo.UseShellExecute = $false
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $pinfo
    $process.Start() | Out-Null

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    return $stdout
}

# Analyze each capture
foreach ($capture in $Captures) {
    $wpixFile = Join-Path $CaptureDir "$capture.wpix"

    Write-Host "Analyzing $capture.wpix..." -ForegroundColor Yellow

    if (Test-Path $wpixFile) {
        $Report += "## Capture: $capture"
        $Report += ""

        # Get timing summary
        Write-Host "  - Extracting timing data..."
        $timing = Invoke-PIXTool "timing --capture `"$wpixFile`""
        if ($timing) {
            $Report += "### GPU Timing"
            $Report += "``````"
            $Report += $timing
            $Report += "``````"
            $Report += ""
        }

        # Get dispatch events
        Write-Host "  - Extracting dispatch events..."
        $events = Invoke-PIXTool "events --capture `"$wpixFile`" --filter Dispatch"
        if ($events) {
            # Parse for ray march dispatch (large thread groups)
            $rayMarchPattern = "Dispatch\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)"
            $matches = [regex]::Matches($events, $rayMarchPattern)

            $Report += "### Dispatch Events"
            $Report += "Found $($matches.Count) dispatch calls:"
            $Report += ""

            foreach ($match in $matches) {
                $x = [int]$match.Groups[1].Value
                $y = [int]$match.Groups[2].Value
                $z = [int]$match.Groups[3].Value

                # Identify dispatch type by thread group size
                if ($x -gt 100 -and $y -gt 60) {
                    $Report += "- **Ray March Dispatch**: ($x, $y, $z) - Screen-space ray marching"
                }
                elseif ($x -eq 16 -and $y -eq 16 -and $z -eq 16) {
                    $Report += "- **Density Fill Dispatch**: ($x, $y, $z) - 3D volume density generation"
                }
                else {
                    $Report += "- Other Dispatch: ($x, $y, $z)"
                }
            }
            $Report += ""
        }

        # Get resource bindings
        Write-Host "  - Extracting resource bindings..."
        $resources = Invoke-PIXTool "resources --capture `"$wpixFile`""
        if ($resources) {
            $Report += "### Resource Bindings"
            $Report += "``````"
            $Report += ($resources | Select-String -Pattern "(SRV|UAV|CBV|Sampler)" | Out-String)
            $Report += "``````"
            $Report += ""
        }

        # Get warnings/errors
        Write-Host "  - Checking for warnings/errors..."
        $warnings = Invoke-PIXTool "warnings --capture `"$wpixFile`""
        if ($warnings) {
            $Report += "### Warnings/Errors"
            $Report += "``````"
            $Report += $warnings
            $Report += "``````"
            $Report += ""
        }

    } else {
        Write-Host "  WARNING: $wpixFile not found" -ForegroundColor Red
        $Report += "## ERROR: $capture.wpix not found"
        $Report += ""
    }
}

# Add diagnostic conclusions
$Report += "## Key Diagnostic Questions for GPT-5"
$Report += ""
$Report += "1. **Debug Mode Values**: Are g_debugMode values different in each capture (0,1,2,3)?"
$Report += "2. **Ray March Dispatch**: Is there a consistent ~(120,68,1) dispatch in all captures?"
$Report += "3. **Density Fill**: Is there a (16,16,16) dispatch before ray marching?"
$Report += "4. **Resource Barriers**: Any UAV/SRV transition warnings?"
$Report += "5. **Visual Similarity**: User reports Off mode looks identical to DensityProbe mode"
$Report += ""
$Report += "## User-Reported Issue"
$Report += "- Seeing 'homogeneous colored fog' instead of coherent 3D volumetric shapes"
$Report += "- Previously had working blue 'mouth-like' volumetric shape"
$Report += "- Off mode (normal rendering) visually identical to DensityProbe mode (single sample)"
$Report += "- Using compute-only path with PLASMADX_DISABLE_DXR=1"
$Report += ""

# Save report
$OutputFile = Join-Path $OutputDir "pix_analysis_report.md"
$Report | Out-File -FilePath $OutputFile -Encoding UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host "Output saved to: $OutputFile" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Display summary
Write-Host ""
Write-Host "Quick Summary:" -ForegroundColor Cyan
Write-Host "  - Analyzed $($Captures.Count) captures"
Write-Host "  - Report contains timing, dispatches, resources, and warnings"
Write-Host "  - Ready to share with GPT-5 for debugging"