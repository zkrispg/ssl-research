# Watcher: launches #1 (Transformer-only architecture replication) once
# #2c (FOA GCA ablation) completes. Sentinel = the last expected test_log
# of #2c: dcase2024_131_ablate_seed3_test.log.

$ErrorActionPreference = "Stop"
$RunsDir = "D:\ssl-research\runs"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$Sentinel = Join-Path $RunsDir "dcase2024_131_ablate_seed3_test.log"
$LogFile = Join-Path $RunsDir "dcase2024_chain_1.log"

function Write-Log {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Write-Log "watcher started; waiting for sentinel: $Sentinel"

while ($true) {
    if (Test-Path $Sentinel) {
        $c = Get-Content $Sentinel -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($c -match "F 20") {
            Write-Log "sentinel found and contains 'F 20' (#2c completed). Launching #1..."
            break
        }
    }
    Start-Sleep -Seconds 300
}

$argList = @('-NoLogo','-NoProfile','-WindowStyle','Hidden','-File',
             (Join-Path $DcaseDir '_run_task_1_xfm_only.ps1'))
Start-Process -FilePath "powershell.exe" -ArgumentList $argList `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $RunsDir 'dcase2024_1_orchestrator.out') `
    -RedirectStandardError  (Join-Path $RunsDir 'dcase2024_1_orchestrator.err')
Write-Log "#1 orchestrator launched. watcher exiting."
