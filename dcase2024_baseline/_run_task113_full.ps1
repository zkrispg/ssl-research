# Task 113 (Vanilla SE-block) ablation: 5 seeds, sequential, resume-safe.
#
# Each cell: 60 epochs train (~110 min) + test_only (~9 min) ~ 2h.
# Total wall time: ~10 GPU hours.
#
# Copies the I/O patterns proven robust during Stage 3:
# cmd.exe /c "python ... > log 2>&1" to avoid PowerShell pipeline deadlocks
# in Hidden-window mode.

$ErrorActionPreference = "Stop"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$LogDir   = "D:\ssl-research\week11_starss23\runs"
$Python   = "D:\ssl-research\venv\Scripts\python.exe"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Test-CellComplete {
    param([string]$TaskId, [string]$Job)
    return Test-Path (Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log")
}

function Invoke-Train {
    param([string]$TaskId, [string]$Job, [int]$Seed)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    $modelGlob = Join-Path $DcaseDir ("models_audio\{0}_{1}_dev_split0_multiaccdoa_*_model.h5" -f $TaskId, $Job)

    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-Host ("[{0}] SKIP train (test_log exists): {1} {2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
        return
    }
    Get-ChildItem $modelGlob -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host ("[{0}] TRAIN start: {1} {2} seed={3}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job, $Seed)
    Push-Location $DcaseDir
    try {
        $env:SSL_SEED = "$Seed"
        $cmd = "`"$Python`" -u train_seldnet.py $TaskId $Job $Seed > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally { Pop-Location }
    Write-Host ("[{0}] TRAIN done : {1} {2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

function Invoke-Test {
    param([string]$TaskId, [string]$Job)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (Test-Path $log) {
        Write-Host ("[{0}] SKIP test  (test_log exists): {1} {2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
        return
    }
    Write-Host ("[{0}] TEST  start: {1} {2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u test_only.py $TaskId $Job > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally { Pop-Location }
    Write-Host ("[{0}] TEST  done : {1} {2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

Write-Host "=== Task 113 (Vanilla SE-block) detached run starting ==="
foreach ($seed in 0..4) {
    $job = "ablate_seed$seed"
    Invoke-Train -TaskId "113" -Job $job -Seed $seed
    Invoke-Test  -TaskId "113" -Job $job
}
Write-Host "=== Task 113 detached run finished ==="

# After all 5 cells done, run analysis and rebuild progress doc
Write-Host "[$(Get-Date -Format HH:mm:ss)] re-running final n=5 analysis (now with 113)"
Push-Location $DcaseDir
try {
    & cmd.exe /c "`"$Python`" -u _path_c_analyze.py > `"$LogDir\path_c_analyze_with113.log`" 2>&1"
    & cmd.exe /c "`"$Python`" -u _path_c_cross_starss22.py --cells 113 > `"$LogDir\path_c_cross_113.log`" 2>&1"
    & cmd.exe /c "`"$Python`" -u _path_c_probe.py --cells 113 > `"$LogDir\path_c_probe_113.log`" 2>&1"
    & cmd.exe /c "`"$Python`" -u _build_progress_doc_v2.py > `"$LogDir\path_c_progress_v3.log`" 2>&1"
}
finally { Pop-Location }

# Final marker
Set-Content "D:\ssl-research\paper\_task113_DONE.txt" -Value ("done at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "=== ALL 113 work complete ==="
