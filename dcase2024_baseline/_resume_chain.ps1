# One-line helper to resume the Path C chain after any disconnection.
#
# What it does:
#   1. If a chain orchestrator is already running, do nothing.
#   2. Otherwise launch the chain orchestrator detached. Resume logic in
#      _run_path_c_full.ps1 skips any cell whose test_log exists, and
#      cleans up partial model files from aborted prior cells.
#
# Usage from any shell:
#   powershell -ExecutionPolicy Bypass -File D:\ssl-research\dcase2024_baseline\_resume_chain.ps1

$pwshOrch = Get-Process powershell -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -like "*_run_path_c_full.ps1*" -or
        # Fallback: a python child running our train script also implies orchestrator is up.
        $false
    }
$pythonOrch = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*train_seldnet.py*" -or $_.CommandLine -like "*test_only.py*" -or $_.CommandLine -like "*batch_feature_extraction.py*" }

if ($pythonOrch) {
    Write-Host "Chain already running:"
    $pythonOrch | Format-Table Id, StartTime, @{N='cpu_s';E={[int]$_.CPU}}, @{N='mem_MB';E={[int]($_.WorkingSet/1MB)}}
    return
}

# Spawn detached
$pwsh = (Get-Command powershell.exe).Source
$proc = Start-Process -FilePath $pwsh `
    -ArgumentList "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", "D:\ssl-research\dcase2024_baseline\_run_path_c_full.ps1" `
    -WindowStyle Hidden -PassThru
Write-Host ("[{0}] chain orchestrator (re)started, PID: {1}" -f (Get-Date -Format "HH:mm:ss"), $proc.Id)
