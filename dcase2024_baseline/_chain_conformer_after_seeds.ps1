# Watcher: wait for Tranche 1 (seed completion) to finish, then launch
# Tranche 2 (Conformer). Keeps the single GPU from being double-booked.

$LogDir = "D:\ssl-research\runs"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$sentinel = Join-Path $LogDir "seed_completion_master.log"
$confLog  = Join-Path $LogDir "conformer_master.log"

Write-Host ("[{0}] Chain watcher up. Waiting for Tranche 1 to finish ..." -f (Get-Date -Format "HH:mm:ss"))
while ($true) {
    if (Test-Path $sentinel) {
        $c = Get-Content $sentinel -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($c -match "SEED COMPLETION ALL DONE") { break }
    }
    Start-Sleep -Seconds 120
}
Write-Host ("[{0}] Tranche 1 done. Launching Conformer (Tranche 2) ..." -f (Get-Date -Format "HH:mm:ss"))

Push-Location $DcaseDir
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $DcaseDir "_run_conformer.ps1") *> $confLog
}
finally { Pop-Location }
Write-Host ("[{0}] Conformer orchestrator returned." -f (Get-Date -Format "HH:mm:ss"))
