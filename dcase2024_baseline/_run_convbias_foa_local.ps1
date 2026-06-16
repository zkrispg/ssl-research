# Local Windows runner for this Codex workspace.
# Mirrors _run_convbias_foa.ps1 with machine-specific paths only.
$DcaseDir = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\dcase2024_baseline"
$Python   = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\venv\Scripts\python.exe"

$ErrorActionPreference = "Stop"
$LogDir = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\runs"
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
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
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

Write-Host "===================== FOA convbias supplement (Conformer + Transformer), n=3 ====================="
$tasks = '184','185','186','187'
$total = $tasks.Count * 3
$i = 0
foreach ($seed in 0,1,2) {
    foreach ($task in $tasks) {
        $i++
        $job = "ablate_seed$seed"
        Write-Host ("--- [{0}/{1}] task={2} job={3} ---" -f $i, $total, $task, $job)
        Invoke-Train -TaskId $task -Job $job -Seed $seed
        Invoke-Test  -TaskId $task -Job $job
    }
}
Write-Host ("[{0}] FOA convbias supplement ALL DONE." -f (Get-Date -Format "HH:mm:ss"))
