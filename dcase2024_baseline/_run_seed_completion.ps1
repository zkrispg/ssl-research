# Journal Tranche 1: complete every cell to n=5 seeds {0,1,2,3,4}.
#
# Existing seeds:
#   100 (FOA no_gca, repro_seed)        : 1,2,3,4   -> add 0
#   130/131 (FOA+CRNN GCA, ablate_seed) : 1,2,3     -> add 0,4
#   140/141/142 (MIC+Xfm, ablate_seed)  : 0,1,2     -> add 3,4
#   150/151/152 (FOA+Xfm, ablate_seed)  : 0,1,2     -> add 3,4
#
# Each run: train_seldnet.py <task> <job> <seed> (60 ep) + test_only.py.
# Skips automatically if the *_test.log already contains "F 20".

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

# (task, jobPrefix, seed) triples to fill. jobPrefix = "repro" or "ablate".
$Jobs = @(
    @('100','repro',0),
    @('130','ablate',0), @('130','ablate',4),
    @('131','ablate',0), @('131','ablate',4),
    @('140','ablate',3), @('140','ablate',4),
    @('141','ablate',3), @('141','ablate',4),
    @('142','ablate',3), @('142','ablate',4),
    @('150','ablate',3), @('150','ablate',4),
    @('151','ablate',3), @('151','ablate',4),
    @('152','ablate',3), @('152','ablate',4)
)

Write-Host "===================== Journal Tranche 1: seed completion (n->5) ====================="
Write-Host ("[{0}] {1} runs queued." -f (Get-Date -Format "HH:mm:ss"), $Jobs.Count)
$i = 0
foreach ($t in $Jobs) {
    $i++
    $task = $t[0]; $job = ("{0}_seed{1}" -f $t[1], $t[2]); $seed = [int]$t[2]
    Write-Host ("--- [{0}/{1}] task={2} job={3} ---" -f $i, $Jobs.Count, $task, $job)
    Invoke-Train -TaskId $task -Job $job -Seed $seed
    Invoke-Test  -TaskId $task -Job $job
}
Write-Host ("[{0}] SEED COMPLETION ALL DONE." -f (Get-Date -Format "HH:mm:ss"))
