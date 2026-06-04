# Detached launcher for DCASE 2024 baseline reproduce.
# Runs FOA seeds 1..4 sequentially (60 epochs each). Survives parent shell exit.
# Usage:
#   powershell -ExecutionPolicy Bypass -File _run_repro_seeds.ps1
# Logs are written to D:\ssl-research\week11_starss23\runs\dcase2024_repro_foa_seed{S}.log

$ErrorActionPreference = "Stop"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$LogDir   = "D:\ssl-research\week11_starss23\runs"
$Python   = "D:\ssl-research\venv\Scripts\python.exe"
$TaskId   = "100"  # FOA + multi-ACCDDOA, 60 epochs, batch 32, finetune from synthetic

$SeedsToRun = @(1, 2, 3, 4)

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

foreach ($seed in $SeedsToRun) {
    $log = Join-Path $LogDir "dcase2024_repro_foa_seed$seed.log"
    $job = "repro_seed$seed"

    Write-Host ("[{0}] starting seed {1}, log -> {2}" -f (Get-Date -Format "HH:mm:ss"), $seed, $log)

    $env:SSL_SEED = "$seed"
    # Use python -u (unbuffered) and Tee-Object to ensure logs are
    # flushed to disk in real time. Pipe also surfaces the output to
    # stdout (in this hidden shell that's a no-op but harmless).
    Push-Location $DcaseDir
    try {
        & $Python "-u" "train_seldnet.py" $TaskId $job "$seed" 2>&1 |
            Tee-Object -FilePath $log

        $testLog = Join-Path $LogDir "dcase2024_test_only_seed$seed.log"
        & $Python "-u" "test_only.py" $TaskId $job 2>&1 |
            Tee-Object -FilePath $testLog
    }
    finally {
        Pop-Location
    }

    Write-Host ("[{0}] seed {1} done." -f (Get-Date -Format "HH:mm:ss"), $seed)
}

Write-Host ("[{0}] all reproduce seeds finished." -f (Get-Date -Format "HH:mm:ss"))
