# Watcher: launches #2c (FOA GCA ablation) once #3 (D supplemental seeds) completes.
#
# #3 finishes when the last expected test_log appears (currently
# dcase2024_121_frac_seed4_test.log). Polls every 5 minutes; on detection,
# launches _run_task_2c_foa_gca.ps1 detached and exits.

$ErrorActionPreference = "Stop"
$RunsDir = "D:\ssl-research\runs"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$Sentinel = Join-Path $RunsDir "dcase2024_121_frac_seed4_test.log"
$LogFile = Join-Path $RunsDir "dcase2024_chain_2c.log"

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
            Write-Log "sentinel found and contains 'F 20' (training successfully completed). Launching #2c..."
            break
        }
    }
    Start-Sleep -Seconds 300  # 5-minute poll
}

# Launch #2c
$argList = @('-NoLogo','-NoProfile','-WindowStyle','Hidden','-File',
             (Join-Path $DcaseDir '_run_task_2c_foa_gca.ps1'))
Start-Process -FilePath "powershell.exe" -ArgumentList $argList `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $RunsDir 'dcase2024_2c_orchestrator.out') `
    -RedirectStandardError  (Join-Path $RunsDir 'dcase2024_2c_orchestrator.err')
Write-Log "#2c orchestrator launched. watcher exiting."
