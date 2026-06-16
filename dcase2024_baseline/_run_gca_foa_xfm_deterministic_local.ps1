$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DcaseDir = $PSScriptRoot
$LogDir = Join-Path $Root "runs"
$Python = Join-Path $Root "venv\Scripts\python.exe"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$DriverLog = Join-Path $LogDir "gca_foa_xfm_det_seld_${RunStamp}_driver.log"
$StatusFile = Join-Path $LogDir "gca_foa_xfm_det_seld_${RunStamp}_status.txt"
$GpuLog = Join-Path $LogDir "gca_foa_xfm_det_seld_${RunStamp}_nvidia_smi_live.csv"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

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
    return ($content -match "F\s*20" -and $content -match "DOAE_CD" -and $content -match "SELD score")
}

function Invoke-Train {
    param([string]$TaskId, [string]$Job, [int]$Seed)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-RunLog "SKIP train/test complete: task=$TaskId job=$Job"
        return
    }

    Write-RunLog "TRAIN start: task=$TaskId job=$Job seed=$Seed"
    Push-Location $DcaseDir
    try {
        $env:SSL_SEED = "$Seed"
        $env:PYTHONHASHSEED = "$Seed"
        $env:SSL_DETERMINISTIC = "1"
        $env:SSL_DETERMINISTIC_STRICT = "1"
        $env:SSL_DISABLE_TF32 = "1"
        $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        $env:SSL_BEST_METRIC = "seld"
        $env:SSL_BEST_TIE = "earlier"
        $cmd = "`"$Python`" -u train_seldnet.py $TaskId $Job $Seed > `"$log`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "train failed: task=$TaskId job=$Job exit=$LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-RunLog "TRAIN done : task=$TaskId job=$Job"
}

function Invoke-Test {
    param([string]$TaskId, [string]$Job)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}_test.log"
    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-RunLog "SKIP test complete: task=$TaskId job=$Job"
        return
    }

    Write-RunLog "TEST start : task=$TaskId job=$Job"
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u test_only.py $TaskId $Job > `"$log`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "test failed: task=$TaskId job=$Job exit=$LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-RunLog "TEST done  : task=$TaskId job=$Job"
}

Write-RunLog "GCA FOA+Transformer deterministic SELD rerun starts"
Write-RunLog "driver_log=$DriverLog"
Write-RunLog "gpu_log=$GpuLog"
Write-RunLog "tasks=151,152 seeds=0,1,2,3,4 best_metric=seld tie=earlier deterministic=strict"

$monitor = Start-Job -ScriptBlock {
    param($OutPath)
    "timestamp,utilization.gpu,memory.used,power.draw,power.limit" | Set-Content -Path $OutPath -Encoding UTF8
    while ($true) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,power.limit --format=csv,noheader,nounits 2>$null
        if ($line) {
            Add-Content -Path $OutPath -Value "$ts,$line" -Encoding UTF8
        }
        Start-Sleep -Seconds 60
    }
} -ArgumentList $GpuLog

try {
    foreach ($seed in 0, 1, 2, 3, 4) {
        foreach ($task in "151", "152") {
            $job = "det_gca_seed$seed"
            Set-Content -Path $StatusFile -Value "running task=$task job=$job seed=$seed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
            Invoke-Train -TaskId $task -Job $job -Seed $seed
            Invoke-Test -TaskId $task -Job $job
        }
    }
    Set-Content -Path $StatusFile -Value "complete at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
    Write-RunLog "GCA FOA+Transformer deterministic SELD rerun complete"
}
catch {
    Set-Content -Path $StatusFile -Value "failed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $_" -Encoding UTF8
    Write-RunLog "FAILED: $_"
    throw
}
finally {
    Stop-Job $monitor -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $monitor -ErrorAction SilentlyContinue | Out-Null
}
