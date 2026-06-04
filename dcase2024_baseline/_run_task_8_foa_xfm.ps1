# Tier VIII: FOA + Transformer-only architecture (closes the 2x2
# modality x architecture dissociation matrix).
#
# Cells (3 cells x 3 seeds = 9 ckpts):
#   150 = FOA + Transformer-only + no GCA
#   151 = FOA + Transformer-only + GCA full      (gca_modality=foa)
#   152 = FOA + Transformer-only + GCA no_geom   (gca_modality=foa)

$ErrorActionPreference = "Stop"
$DcaseDir = "D:\ssl-research\dcase2024_baseline"
$LogDir   = "D:\ssl-research\runs"
$Python   = "D:\ssl-research\venv\Scripts\python.exe"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Test-CellComplete {
    param([string]$TaskId, [string]$Job)
    $testlog = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (-not (Test-Path $testlog)) { return $false }
    $content = Get-Content $testlog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    return ($content -match "F 20")
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
    finally { Pop-Location }
    Write-Host ("[{0}] TRAIN done : task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

function Invoke-Test {
    param([string]$TaskId, [string]$Job)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (Test-Path $log) {
        $c = Get-Content $log -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($c -match "F 20") {
            Write-Host ("[{0}] SKIP test (test_log ok): task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
            return
        }
        Remove-Item $log -Force
    }
    Write-Host ("[{0}] TEST  start: task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u test_only.py $TaskId $Job > `"$log`" 2>&1"
        & cmd.exe /c $cmd
    }
    finally { Pop-Location }
    Write-Host ("[{0}] TEST  done : task={1} job={2}" -f (Get-Date -Format "HH:mm:ss"), $TaskId, $Job)
}

# ============== Sweep ==============
Write-Host "===================== Tier VIII: FOA + Transformer-only ====================="
foreach ($seed in 0, 1, 2) {
    foreach ($task in '150', '151', '152') {
        $env:SSL_SEED = "$seed"
        $job = "ablate_seed$seed"
        Invoke-Train -TaskId $task -Job $job -Seed $seed
        Invoke-Test  -TaskId $task -Job $job
    }
}

Write-Host ("[{0}] Tier VIII (FOA + Xfm) ALL DONE." -f (Get-Date -Format "HH:mm:ss"))

# ----- Auto-rerun the unified 2x2 dissociation analyzer + doc rebuild -----
Push-Location $DcaseDir
try {
    & cmd.exe /c "`"$Python`" -u _path_c_2x2_dissociation.py > `"$LogDir\dcase2024_2x2_analyze.log`" 2>&1"
    & cmd.exe /c "`"$Python`" -u _build_progress_doc_v2.py > `"$LogDir\dcase2024_doc_v4.log`" 2>&1"
}
finally { Pop-Location }
Write-Host ("[{0}] 2x2 ANALYZE + DOC REBUILD DONE." -f (Get-Date -Format "HH:mm:ss"))
