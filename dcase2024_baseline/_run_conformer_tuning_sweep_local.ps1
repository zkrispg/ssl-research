$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DcaseDir = $PSScriptRoot
$LogDir = Join-Path $Root "runs"
$Python = Join-Path $Root "venv\Scripts\python.exe"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$DriverLog = Join-Path $LogDir "conformer_tuning_sweep_${RunStamp}_driver.log"
$StatusFile = Join-Path $LogDir "conformer_tuning_sweep_${RunStamp}_status.txt"
$GpuLog = Join-Path $LogDir "conformer_tuning_sweep_${RunStamp}_nvidia_smi_live.csv"

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
    param([string]$TaskId, [string]$Job, [int]$Seed, [string]$BestMetric, [string]$Lr, [string]$Dropout)
    $log = Join-Path $LogDir "dcase2024_${TaskId}_${Job}.log"
    if (Test-CellComplete -TaskId $TaskId -Job $Job) {
        Write-RunLog "SKIP train/test complete: task=$TaskId job=$Job"
        return
    }

    Write-RunLog "TRAIN start: task=$TaskId job=$Job seed=$Seed best=$BestMetric lr=$Lr dropout=$Dropout"
    Push-Location $DcaseDir
    try {
        $env:SSL_SEED = "$Seed"
        $env:PYTHONHASHSEED = "$Seed"
        $env:SSL_DETERMINISTIC = "1"
        $env:SSL_DETERMINISTIC_STRICT = "1"
        $env:SSL_DISABLE_TF32 = "1"
        $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        $env:SSL_BEST_METRIC = "$BestMetric"
        $env:SSL_BEST_TIE = "earlier"
        $env:SSL_LR = "$Lr"
        $env:SSL_DROPOUT = "$Dropout"
        $cmd = "`"$Python`" -u train_seldnet.py $TaskId $Job $Seed > `"$log`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "train failed: task=$TaskId job=$Job exit=$LASTEXITCODE"
        }
    }
    finally {
        Remove-Item Env:\SSL_LR -ErrorAction SilentlyContinue
        Remove-Item Env:\SSL_DROPOUT -ErrorAction SilentlyContinue
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
            throw "test failed: task=$TaskId job=$Job"
        }
    }
    finally {
        Pop-Location
    }
    Write-RunLog "TEST done  : task=$TaskId job=$Job"
}

Write-RunLog "Conformer tuning sweep starts"
Write-RunLog "driver_log=$DriverLog"
Write-RunLog "gpu_log=$GpuLog"
Write-RunLog "pilot tasks=171,172 seeds=0,1 lr=1e-3,5e-4,3e-4 best=seld,doae dropout=0.05"
Write-RunLog "Purpose: choose a better Conformer operating point before any larger n=5 rerun."

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
    foreach ($metric in "seld", "doae") {
        foreach ($lrName in "1em3", "5em4", "3em4") {
            if ($lrName -eq "1em3") { $lr = "1e-3" }
            elseif ($lrName -eq "5em4") { $lr = "5e-4" }
            else { $lr = "3e-4" }
            foreach ($seed in 0, 1) {
                foreach ($task in "171", "172") {
                    $job = "tune_${metric}_lr${lrName}_seed$seed"
                    Set-Content -Path $StatusFile -Value "running task=$task job=$job seed=$seed metric=$metric lr=$lr at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
                    Invoke-Train -TaskId $task -Job $job -Seed $seed -BestMetric $metric -Lr $lr -Dropout "0.05"
                    Invoke-Test -TaskId $task -Job $job
                }
            }
        }
    }
    Set-Content -Path $StatusFile -Value "complete at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
    Write-RunLog "Conformer tuning sweep complete"
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

Push-Location $DcaseDir
try {
    foreach ($metric in "seld", "doae") {
        foreach ($lrName in "1em3", "5em4", "3em4") {
            & $Python _summarize_seld_logs.py `
                --tasks 171,172 `
                --seeds 0,1 `
                --job-template "tune_${metric}_lr${lrName}_seed{seed}" `
                --pairs "FOA_Conformer_${metric}_lr${lrName}:171-172" `
                --out-prefix "conformer_tune_${metric}_lr${lrName}_summary" `
                --title "Conformer tuning ${metric} lr${lrName} Summary"
        }
    }
}
finally {
    Pop-Location
}
