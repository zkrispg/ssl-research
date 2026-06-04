# Quick status snapshot for the Path C chain orchestrator.
# Run any time to see what stage we're in and how each cell is doing.

$LogDir = "D:\ssl-research\week11_starss23\runs"
$ModelDir = "D:\ssl-research\dcase2024_baseline\models_audio"

Write-Host "=================================================================="
Write-Host (" Path C status snapshot  --  {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "=================================================================="

# ---- orchestrator alive? ----
$orchestrator = Get-Process powershell -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -eq "" -and $_.StartTime -lt (Get-Date) } |
    Sort-Object StartTime
$python = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.WorkingSet -gt 100MB }

if ($python) {
    foreach ($p in $python) {
        $ageMin = [math]::Round((((Get-Date) - $p.StartTime).TotalMinutes), 1)
        Write-Host (" python PID {0}: alive {1} min, {2} MB, {3} CPU-s" -f $p.Id, $ageMin, [int]($p.WorkingSet/1MB), [int]$p.CPU)
    }
}
else {
    Write-Host " no active python training process"
}

# ---- GPU ----
$gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null) -join ' '
Write-Host (" gpu: {0}" -f $gpu)
Write-Host ""

# ---- per-cell state ----
function Get-LastEpoch {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $lines = Get-Content $Path -ErrorAction SilentlyContinue
    if ($null -eq $lines -or $lines.Count -eq 0) { return $null }
    # @(...) preserves array semantics even when a single line matches.
    $epLines = @($lines | Where-Object { $_ -match "^epoch:\s*\d+" })
    if ($epLines.Count -eq 0) { return $null }
    $last = $epLines[-1]
    if ($last -match "^epoch:\s*(\d+),.*F/AE.*?:\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)") {
        return [pscustomobject]@{
            epoch = [int]$matches[1]
            F1    = [double]$matches[2]
            AE    = [double]$matches[3]
            Dist  = [double]$matches[4]
            RDE   = [double]$matches[5]
            SELD  = [double]$matches[6]
        }
    }
    return $null
}

function Test-CellModel {
    param([string]$TaskId, [string]$Job)
    $glob = Join-Path $ModelDir "$($TaskId)_*$Job*model.h5"
    return @(Get-ChildItem $glob -ErrorAction SilentlyContinue).Count -gt 0
}

$cells = @(
    @{ task='100'; job='repro_seed0'; descr='Stage1 FOA repro seed0 (test-evaled)' }
    @{ task='100'; job='repro_seed1'; descr='Stage1 FOA repro seed1' }
    @{ task='100'; job='repro_seed2'; descr='Stage1 FOA repro seed2' }
    @{ task='100'; job='repro_seed3'; descr='Stage1 FOA repro seed3' }
    @{ task='100'; job='repro_seed4'; descr='Stage1 FOA repro seed4' }
    @{ task='110'; job='ablate_seed0'; descr='Stage3 110 GCA full   seed0' }
    @{ task='110'; job='ablate_seed1'; descr='Stage3 110 GCA full   seed1' }
    @{ task='110'; job='ablate_seed2'; descr='Stage3 110 GCA full   seed2' }
    @{ task='110'; job='ablate_seed3'; descr='Stage3 110 GCA full   seed3' }
    @{ task='110'; job='ablate_seed4'; descr='Stage3 110 GCA full   seed4' }
    @{ task='111'; job='ablate_seed0'; descr='Stage3 111 GCA nogeom seed0' }
    @{ task='111'; job='ablate_seed1'; descr='Stage3 111 GCA nogeom seed1' }
    @{ task='111'; job='ablate_seed2'; descr='Stage3 111 GCA nogeom seed2' }
    @{ task='111'; job='ablate_seed3'; descr='Stage3 111 GCA nogeom seed3' }
    @{ task='111'; job='ablate_seed4'; descr='Stage3 111 GCA nogeom seed4' }
    @{ task='112'; job='ablate_seed0'; descr='Stage3 112 no-GCA     seed0' }
    @{ task='112'; job='ablate_seed1'; descr='Stage3 112 no-GCA     seed1' }
    @{ task='112'; job='ablate_seed2'; descr='Stage3 112 no-GCA     seed2' }
    @{ task='112'; job='ablate_seed3'; descr='Stage3 112 no-GCA     seed3' }
    @{ task='112'; job='ablate_seed4'; descr='Stage3 112 no-GCA     seed4' }
)

Write-Host (" {0,-40} {1,-13} {2,-12} {3}" -f "cell", "train", "test_eval", "last epoch / metrics")
Write-Host (" " + ("-" * 100))
foreach ($c in $cells) {
    $log = Join-Path $LogDir ("dcase2024_{0}_{1}.log" -f $c.task, $c.job)
    $testlog = Join-Path $LogDir ("dcase2024_{0}_{1}_test.log" -f $c.task, $c.job)

    # train state. The DCASE script writes best.pt as soon as epoch 0 finishes,
    # so a checkpoint alone doesn't mean training is complete. We declare DONE
    # only when the orchestrator's TEST log exists (it is written after train
    # exits cleanly).
    if (Test-Path $testlog) {
        $trainState = "DONE"
    }
    elseif (Test-CellModel -TaskId $c.task -Job $c.job) {
        $ep = Get-LastEpoch -Path $log
        if ($null -ne $ep -and $ep.epoch -ge 59) {
            $trainState = "trn-done"  # 60 epochs reached, awaiting test
        }
        else {
            $trainState = "running"
        }
    }
    elseif (Test-Path $log) {
        $trainState = "running"
    }
    else {
        $trainState = "queued"
    }

    # test eval state
    if (Test-Path $testlog) {
        $testState = "DONE"
    }
    else {
        $testState = "(pending)"
    }

    # last epoch info
    $ep = Get-LastEpoch -Path $log
    if ($null -eq $ep) {
        if (Test-Path $log) { $epStr = "(no epoch yet)" } else { $epStr = "" }
    }
    else {
        $epStr = "ep {0}/60  F1={1:N2}%  AE={2:N1}  SELD={3:N3}" -f ($ep.epoch + 1), ($ep.F1 * 100), $ep.AE, $ep.SELD
    }

    Write-Host (" {0,-40} {1,-13} {2,-12} {3}" -f $c.descr, $trainState, $testState, $epStr)
}
Write-Host ""

# ---- analyzer summary ----
Write-Host "==== analyzer summary (recomputed) ===="
Push-Location D:\ssl-research\dcase2024_baseline
try {
    & D:\ssl-research\venv\Scripts\python.exe _path_c_analyze.py 2>&1 | Select-Object -Last 8
}
finally {
    Pop-Location
}
