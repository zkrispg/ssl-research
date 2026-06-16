param(
    [string[]]$Tasks = @('184', '185', '186', '187'),
    [int[]]$Seeds = @(0, 1, 2),
    [string]$JobPrefix = 'det_seld'
)

$ErrorActionPreference = "Stop"

$DcaseDir = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\dcase2024_baseline"
$Python = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\venv\Scripts\python.exe"
$LogDir = "C:\Users\Administrator\Documents\Codex\2026-06-08\1\work\ssl-research\runs"
New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$DriverLog = Join-Path $LogDir "convbias_foa_${JobPrefix}_${RunId}_driver.log"
$NvidiaLog = Join-Path $LogDir "convbias_foa_${JobPrefix}_${RunId}_nvidia_smi.txt"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:SSL_DETERMINISTIC = "1"
$env:SSL_DETERMINISTIC_STRICT = "1"
$env:SSL_DISABLE_TF32 = "0"
$env:SSL_BEST_METRIC = "seld"
$env:SSL_BEST_TIE = "earlier"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $DriverLog -Value $line -Encoding UTF8
}

function Test-CellComplete {
    param([string]$TaskId, [string]$Job)
    $testLog = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (-not (Test-Path $testLog)) { return $false }
    $content = Get-Content $testLog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    return (($content -match "SELD score") -and ($content -match "DOAE_CD"))
}

function Invoke-Train {
    param([string]$TaskId, [string]$Job, [int]$Seed)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    $modelGlob = Join-Path $DcaseDir ("models_audio\{0}_{1}_dev_split0_multiaccdoa_*_model.h5" -f $TaskId, $Job)
    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-RunLog "SKIP train: task=$TaskId job=$Job test log already complete"
        return
    }
    Get-ChildItem $modelGlob -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    $env:SSL_SEED = "$Seed"
    Write-RunLog "CMD train: $Python -u train_seldnet.py $TaskId $Job $Seed > $log 2>&1"
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u train_seldnet.py $TaskId $Job $Seed > `"$log`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "train failed with exit code $LASTEXITCODE; see $log"
        }
    }
    finally {
        Pop-Location
    }
    Write-RunLog "DONE train: task=$TaskId job=$Job seed=$Seed log=$log"
}

function Invoke-Test {
    param([string]$TaskId, [string]$Job)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-RunLog "SKIP test: task=$TaskId job=$Job test log already complete"
        return
    }
    if (Test-Path $log) {
        Remove-Item $log -Force
    }
    Write-RunLog "CMD test: $Python -u test_only.py $TaskId $Job > $log 2>&1"
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u test_only.py $TaskId $Job > `"$log`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "test failed with exit code $LASTEXITCODE; see $log"
        }
    }
    finally {
        Pop-Location
    }
    Write-RunLog "DONE test: task=$TaskId job=$Job log=$log"
}

Write-RunLog "FOA convbias deterministic run start"
Write-RunLog "tasks=$($Tasks -join ',') seeds=$($Seeds -join ',') job_prefix=$JobPrefix"
Write-RunLog "env CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES SSL_DETERMINISTIC=$env:SSL_DETERMINISTIC SSL_BEST_METRIC=$env:SSL_BEST_METRIC SSL_BEST_TIE=$env:SSL_BEST_TIE"

try {
    & nvidia-smi | Out-File -FilePath $NvidiaLog -Encoding UTF8
    Write-RunLog "nvidia-smi snapshot: $NvidiaLog"
}
catch {
    Write-RunLog "WARNING nvidia-smi snapshot failed: $($_.Exception.Message)"
}

$total = $Tasks.Count * $Seeds.Count
$i = 0
foreach ($seed in $Seeds) {
    foreach ($task in $Tasks) {
        $i++
        $job = "${JobPrefix}_seed$seed"
        Write-RunLog "CELL [$i/$total] task=$task job=$job seed=$seed"
        Invoke-Train -TaskId $task -Job $job -Seed $seed
        Invoke-Test -TaskId $task -Job $job
    }
}

Write-RunLog "FOA convbias deterministic run complete"
