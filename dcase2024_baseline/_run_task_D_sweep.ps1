# Path C / Tier V (D): training-data fraction sweep orchestrator.
#
# Trains 4 cells x 3 seeds = 12 ckpts comparing GCA full (110-clone)
# vs no-GCA (112-clone) at 50% and 25% of dev-train data. The 100%
# point reuses the existing Stage 3 results (task 110 vs 112 x 5 seeds).
#
# Tasks:
#   120 = GCA full,  train_data_fraction = 0.50
#   121 = no-GCA,    train_data_fraction = 0.50
#   122 = GCA full,  train_data_fraction = 0.25
#   123 = no-GCA,    train_data_fraction = 0.25
#
# Order per seed: 122/123 first (fast 0.5h each), then 120/121 (1h each).
# Total wall time ~9 GPU hours sequential.
#
# Resume-safe: skips a (task, seed) pair if its test_log already exists.

$ErrorActionPreference = "Stop"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$LogDir   = "D:\ssl-research\runs"
$Python   = "D:\ssl-research\venv\Scripts\python.exe"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Test-CellComplete {
    param([string]$TaskId, [string]$Job)
    $testlog = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    return Test-Path $testlog
}

function Invoke-Train {
    param([string]$TaskId, [string]$Job, [int]$Seed)
    $log       = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    $modelGlob = Join-Path $DcaseDir ("models_audio\{0}_{1}_dev_split0_multiaccdoa_*_model.h5" -f $TaskId, $Job)

    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-Host ("[{0}] SKIP train (test_log exists): task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
        return
    }

    Get-ChildItem $modelGlob -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    Write-Host ("[{0}] TRAIN start: task={1} job={2} seed={3}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job, $Seed)
    Push-Location $DcaseDir
    try {
        $env:SSL_SEED = "$Seed"
        $cmd = "`"$Python`" -u train_seldnet.py $TaskId $Job $Seed > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally {
        Pop-Location
    }
    Write-Host ("[{0}] TRAIN done : task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

function Invoke-Test {
    param([string]$TaskId, [string]$Job)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (Test-Path $log) {
        Write-Host ("[{0}] SKIP test  (test_log exists): task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
        return
    }
    Write-Host ("[{0}] TEST  start: task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u test_only.py $TaskId $Job > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally {
        Pop-Location
    }
    Write-Host ("[{0}] TEST  done : task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

# ============== Sweep ==============
Write-Host "===================== TIER V (D) data-fraction sweep ====================="
foreach ($seed in 0, 1, 2) {
    foreach ($task in '122', '123', '120', '121') {
        # 122/123 = 25% pair (fast) first, then 120/121 = 50% pair.
        $env:SSL_SEED = "$seed"
        $job = "frac_seed$seed"
        Invoke-Train -TaskId $task -Job $job -Seed $seed
        Invoke-Test  -TaskId $task -Job $job
    }
}

Write-Host ("[{0}] TIER V (D) ALL DONE." -f (Get-Date -Format "HH:mm:ss"))

# ----- Auto-run analysis after the sweep -----
Push-Location $DcaseDir
try {
    $cmd = "`"$Python`" -u _path_c_data_fraction_analyze.py > `"$LogDir\dcase2024_data_fraction_analyze.log`" 2>&1"
    & cmd.exe /c $cmd
    $cmd2 = "`"$Python`" -u _build_progress_doc_v2.py > `"$LogDir\dcase2024_doc_v3_build.log`" 2>&1"
    & cmd.exe /c $cmd2
}
finally {
    Pop-Location
}
Write-Host ("[{0}] ANALYZE + DOC REBUILD DONE." -f (Get-Date -Format "HH:mm:ss"))
