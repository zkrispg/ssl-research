# Full Path C orchestrator -- DCASE 2024 baseline reproduce + GCA ablation.
#
# Stages:
#   STAGE 1: FOA reproduce, seeds 1..4   (task 100, finetune init)
#                  -> 4 x ~110 min train + ~9 min test
#   STAGE 2: MIC feature extraction      (task 102 setup, CPU only ~10 min)
#   STAGE 3: GCA ablation, seeds 0..4    (task 110, 111, 112)
#                  -> 15 x ~110 min train + ~9 min test
#
# Total wall time: ~36 GPU hours. Survives PowerShell/IDE restarts because
# we are spawned by Start-Process -WindowStyle Hidden and each child uses
# python -u + Tee-Object -FilePath for line-buffered logs.
#
# Resume: each train step skips if its model file already exists in the
# expected output dir; each test step skips if dcase output_dir is non-empty.

$ErrorActionPreference = "Stop"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$LogDir   = "D:\ssl-research\week11_starss23\runs"
$Python   = "D:\ssl-research\venv\Scripts\python.exe"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Test-CellComplete {
    # A cell is complete only when its test_log has been written. This is
    # safer than checking for the model file alone because best.pt is saved
    # after every val epoch, including epoch 0 of an aborted run.
    param([string]$TaskId, [string]$Job)
    $testlog = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    return Test-Path $testlog
}

function Invoke-Train {
    param([string]$TaskId, [string]$Job, [int]$Seed)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    $modelFile = Join-Path $DcaseDir ("models_audio\{0}_{1}_dev_split0_multiaccdoa_*_model.h5" -f $TaskId, $Job)

    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-Host ("[{0}] SKIP train (test_log exists): task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
        return
    }

    # Clean up any partial model from an aborted previous run so we
    # always retrain from scratch on resume (the alternative is a model
    # trained for 1-2 epochs that we'd silently treat as final).
    Get-ChildItem $modelFile -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    Write-Host ("[{0}] TRAIN start: task={1} job={2} seed={3}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job, $Seed)
    Push-Location $DcaseDir
    try {
        # Launch via cmd.exe with native > redirection. Avoids PowerShell
        # Tee-Object pipeline deadlocks we observed in -WindowStyle Hidden
        # mode when stderr+stdout writes overlap on a small Windows pipe.
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

function Invoke-FeatureExtraction {
    param([string]$TaskId)
    $log = Join-Path $LogDir "dcase2024_feat_extract_task${TaskId}.log"
    Write-Host ("[{0}] FEATEXTRACT start: task={1}" -f (Get-Date -Format "HH:mm:ss"), $TaskId)
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u batch_feature_extraction.py $TaskId > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally {
        Pop-Location
    }
    Write-Host ("[{0}] FEATEXTRACT done : task={1}" -f (Get-Date -Format "HH:mm:ss"), $TaskId)
}

# ============== STAGE 1: FOA reproduce, seeds 1..4 ==============
Write-Host "===================== STAGE 1: FOA reproduce seeds 1..4 ====================="
foreach ($seed in 1, 2, 3, 4) {
    $env:SSL_SEED = "$seed"
    Invoke-Train -TaskId "100" -Job "repro_seed$seed" -Seed $seed
    Invoke-Test  -TaskId "100" -Job "repro_seed$seed"
}

# ============== STAGE 2: MIC feature extraction ==============
# task 102 reuses task 6's MIC GCC features. We run task 6 once to build
# the cache (since task 102 inherits feat-label-dir layout).
Write-Host "===================== STAGE 2: MIC GCC feature extraction ====================="
$micFeatDir = "D:\ssl-research\dcase2024_baseline\..\DCASE2024_SELD_dataset\seld_feat_label\mic_dev_norm"
if (-not (Test-Path $micFeatDir)) {
    Invoke-FeatureExtraction -TaskId "102"
}
else {
    Write-Host ("[{0}] SKIP MIC feature extraction (cache exists)." -f (Get-Date -Format "HH:mm:ss"))
}

# ============== STAGE 3: GCA ablation, seeds 0..4 x {110, 111, 112} ==============
Write-Host "===================== STAGE 3: GCA ablation, 5 seeds x 3 cells ====================="
# Order chosen so the most informative paired contrast (full vs no_geom) lands
# first. Cell ordering inside each seed: 110 (full), 111 (no_geom), 112 (control).
foreach ($seed in 0, 1, 2, 3, 4) {
    foreach ($task in '110', '111', '112') {
        $env:SSL_SEED = "$seed"
        $job = "ablate_seed$seed"
        Invoke-Train -TaskId $task -Job $job -Seed $seed
        Invoke-Test  -TaskId $task -Job $job
    }
}

Write-Host ("[{0}] PATH C ALL STAGES COMPLETE." -f (Get-Date -Format "HH:mm:ss"))
