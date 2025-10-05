# Process PIX event-list CSVs (local or UNC) and generate a concise analysis report

param(
    [string[]]$CsvFiles = @(),
    [string]$CsvDir = "",
    [string]$Output = ""
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ScriptDir "..\workflow_claude\pix_analysis_report.md"
}

# Resolve input CSVs
$resolvedCsvs = @()
if ($CsvFiles -and $CsvFiles.Count -gt 0) {
    foreach ($p in $CsvFiles) {
        $resolvedCsvs += (Resolve-Path -Path $p).Path
    }
} elseif (-not [string]::IsNullOrWhiteSpace($CsvDir)) {
    if (Test-Path $CsvDir) {
        $resolvedCsvs = Get-ChildItem -Path $CsvDir -Filter *.csv | Sort-Object Name | ForEach-Object { $_.FullName }
    }
} else {
    $defaultDir = Join-Path $ScriptDir "..\Analysis"
    if (Test-Path $defaultDir) {
        $resolvedCsvs = Get-ChildItem -Path $defaultDir -Filter *.csv | Sort-Object Name | ForEach-Object { $_.FullName }
    }
}

if (-not $resolvedCsvs -or $resolvedCsvs.Count -eq 0) {
    Write-Host "No CSV files found. Provide -CsvFiles or -CsvDir." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

function Convert-ToMilliseconds {
    param([string]$ns)
    if ([string]::IsNullOrWhiteSpace($ns)) { return 0.0 }
    try { return [double]$ns / 1000000.0 } catch { return 0.0 }
}

function Get-Number {
    param([string]$value)
    if ([string]::IsNullOrWhiteSpace($value)) { return 0 }
    try { return [double]$value } catch { return 0 }
}

$report = @()
$report += "# PIX Event CSV Analysis"
$report += "Generated: $(Get-Date)"
$report += ""

foreach ($csvPath in $resolvedCsvs) {
    $name = Split-Path -Leaf $csvPath
    $modeLabel = ($name -replace "\.csv$", "")
    $report += "## File: $name"
    $report += "Path: $csvPath"
    $report += ""

    try {
        $rows = Import-Csv -Path $csvPath
    } catch {
        $report += "Failed to read CSV ($_)"
        $report += ""
        continue
    }

    if (-not $rows -or $rows.Count -eq 0) {
        $report += "No rows in CSV."
        $report += ""
        continue
    }

    # Find candidate compute dispatches and raymarch markers
    $dispatchRows = $rows | Where-Object { $_.Name -match '^Dispatch' -or $_.Name -match 'RayMarch' -or $_.Name -match 'RayMarcher' }
    $drawRows = $rows | Where-Object { $_.Name -match 'DrawInstanced' }

    $dominant = $null
    $dominantMs = -1.0
    foreach ($r in $dispatchRows) {
        $durMs = Convert-ToMilliseconds $r.'TOP to EOP Duration (ns)'
        if ($durMs -gt $dominantMs) { $dominant = $r; $dominantMs = $durMs }
    }

    $totalDispatches = ($dispatchRows | Measure-Object).Count
    $totalDraws = ($drawRows | Measure-Object).Count

    $report += "- **dispatch_count**: $totalDispatches"
    $report += "- **draw_count**: $totalDraws"

    if ($dominant) {
        $csInv = Get-Number $dominant.'CS Invocations'
        $threads = Get-Number $dominant.CsThreads
        $report += "- **dominant_compute**: $($dominant.Name)"
        $report += "- **dominant_duration_ms**: {0:N3}" -f $dominantMs
        $report += "- **cs_invocations**: $csInv"
        $report += "- **cs_threads**: $threads"
    } else {
        $report += "- **dominant_compute**: not found"
    }

    # Quick presence checks for pipeline order
    $hasRootSig = ($rows | Where-Object { $_.Name -match 'SetComputeRootSignature' } | Measure-Object).Count -gt 0
    $hasPipeline = ($rows | Where-Object { $_.Name -match 'SetPipelineState' } | Measure-Object).Count -gt 0
    $hasHeaps = ($rows | Where-Object { $_.Name -match 'SetDescriptorHeaps' } | Measure-Object).Count -gt 0
    $hasRTs = ($rows | Where-Object { $_.Name -match 'OMSetRenderTargets' } | Measure-Object).Count -gt 0
    $hasPresent = ($rows | Where-Object { $_.Name -match '^Present$' } | Measure-Object).Count -gt 0

    $report += "- **binds_root_signature**: $hasRootSig"
    $report += "- **binds_pipeline_state**: $hasPipeline"
    $report += "- **binds_descriptor_heaps**: $hasHeaps"
    $report += "- **sets_render_targets**: $hasRTs"
    $report += "- **present_called**: $hasPresent"
    $report += ""

    # Show the top 10 longest events for a quick hotspot view
    $withDur = @()
    foreach ($r in $rows) {
        $ms = Convert-ToMilliseconds $r.'TOP to EOP Duration (ns)'
        $withDur += [pscustomobject]@{ Name=$r.Name; DurationMs=$ms }
    }
    $top = $withDur | Sort-Object -Property DurationMs -Descending | Select-Object -First 10
    $report += "### Top 10 events by duration (ms)"
    foreach ($t in $top) {
        $report += "- {0,8:N3} ms  {1}" -f $t.DurationMs, $t.Name
    }
    $report += ""
}

$report | Out-File -FilePath $Output -Encoding UTF8
Write-Host "Wrote analysis to: $Output" -ForegroundColor Green


