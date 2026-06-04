# TASLP rush: status across #3 (D supplement seeds 3,4), #2c (FOA GCA),
# and #1 (Transformer-only arch). Reads only the test_logs to determine
# completion.

$RunsDir = "D:\ssl-research\runs"

Write-Host "==================== TASLP rush status ===================="
Write-Host ""

$activePy = Get-Process -Name python -ErrorAction SilentlyContinue
$activePs = Get-Process -Name powershell -ErrorAction SilentlyContinue
Write-Host ("python.exe PIDs: " + (@($activePy | ForEach-Object Id) -join ', '))
Write-Host ("powershell.exe count: " + ($activePs.Count))
Write-Host ""

function Show-Stage {
    param([string]$Title, [array]$Tasks, [array]$Seeds, [string]$JobPattern, [int]$RoughMin)
    Write-Host ("===== {0} =====" -f $Title)
    $done = 0
    $total = $Tasks.Count * $Seeds.Count
    foreach ($s in $Seeds) {
        foreach ($t in $Tasks) {
            $job = $JobPattern -replace '\{seed\}', $s
            $tl = Join-Path $RunsDir ("dcase2024_{0}_{1}_test.log" -f $t, $job)
            $tl_exists = Test-Path $tl
            $tl_ok = $false
            if ($tl_exists) {
                $c = Get-Content $tl -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
                if ($c -match "F 20") { $tl_ok = $true; $done++ }
            }
            $status = if ($tl_ok) { "DONE" } elseif ($tl_exists) { "FAILED" } else { "pending" }
            Write-Host ("  task={0} {1}: {2}" -f $t, $job, $status)
        }
    }
    $remain = $total - $done
    $h = [math]::Round($remain * $RoughMin / 60.0, 1)
    Write-Host ("  -> {0}/{1} done, ~{2} h remaining" -f $done, $total, $h)
    Write-Host ""
}

Show-Stage -Title "#3: D supplement seeds 3, 4 (~14 GPU-h)" `
           -Tasks @('122','123','120','121') -Seeds @(3, 4) `
           -JobPattern "frac_seed{seed}" -RoughMin 95

Show-Stage -Title "#2c: FOA GCA ablation (~11 GPU-h)" `
           -Tasks @('130','131') -Seeds @(1, 2, 3) `
           -JobPattern "ablate_seed{seed}" -RoughMin 110

Show-Stage -Title "#1: Transformer-only arch (~18 GPU-h)" `
           -Tasks @('140','141','142') -Seeds @(0, 1, 2) `
           -JobPattern "ablate_seed{seed}" -RoughMin 120

Show-Stage -Title "Tier VIII: FOA + Transformer-only (~18-20 GPU-h)" `
           -Tasks @('150','151','152') -Seeds @(0, 1, 2) `
           -JobPattern "ablate_seed{seed}" -RoughMin 130

Write-Host "===== Latest active train log tail ====="
$latest = Get-ChildItem $RunsDir -Filter "dcase2024_15*.log" -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -notmatch "_test\.log$" -and $_.Name -notmatch "orchestrator|chain|analyze|doc" } |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) {
    $latest = Get-ChildItem $RunsDir -Filter "dcase2024_1*.log" -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -notmatch "_test\.log$" -and $_.Name -notmatch "orchestrator|chain|analyze|doc" } |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if ($latest) {
    Write-Host ("file: {0} (mtime={1})" -f $latest.Name, $latest.LastWriteTime)
    Get-Content $latest.FullName -Tail 4 -Encoding UTF8
} else {
    Write-Host "(no active train log yet)"
}
Write-Host ""

Write-Host "===== Watcher log heads ====="
foreach ($f in @('dcase2024_chain_2c.log', 'dcase2024_chain_1.log')) {
    $p = Join-Path $RunsDir $f
    if (Test-Path $p) {
        Write-Host ("[{0}]" -f $f)
        Get-Content $p -Tail 2 -Encoding UTF8
        Write-Host ""
    }
}
