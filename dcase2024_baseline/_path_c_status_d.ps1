# Quick snapshot of the Tier V (D) data-fraction sweep status.
$RunsDir = "D:\ssl-research\runs"

Write-Host "===== Tier V (D) data-fraction sweep status ====="
Write-Host ""

$orchestratorRunning = $false
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*_run_task_D_sweep*"
} | ForEach-Object {
    $orchestratorRunning = $true
    Write-Host ("Orchestrator PID {0}: RUNNING (started {1})" -f $_.ProcessId, $_.CreationDate)
}
if (-not $orchestratorRunning) {
    Write-Host "Orchestrator: NOT running"
}

$pyCount = (Get-Process -Name python -ErrorAction SilentlyContinue).Count
Write-Host ("python.exe processes: {0}" -f $pyCount)
Write-Host ""

Write-Host "===== Per-cell completion (test_log existence) ====="
foreach ($s in 0, 1, 2) {
    foreach ($task in '122', '123', '120', '121') {
        $tlog = Join-Path $RunsDir ("dcase2024_{0}_frac_seed{1}_test.log" -f $task, $s)
        $tlog_exists = Test-Path $tlog
        $tlog_ok = $false
        if ($tlog_exists) {
            $content = Get-Content $tlog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($content -match "F 20") { $tlog_ok = $true }
        }
        $status = if ($tlog_ok) { "DONE" } elseif ($tlog_exists) { "FAILED" } else { "pending" }
        Write-Host ("  {0} seed {1}: {2}" -f $task, $s, $status)
    }
}
Write-Host ""

Write-Host "===== Latest train log tail ====="
$latest = Get-ChildItem $RunsDir -Filter "dcase2024_12*_frac_seed*.log" -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -notmatch "_test\.log$" } |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) {
    Write-Host ("file: {0} (mtime={1})" -f $latest.Name, $latest.LastWriteTime)
    Get-Content $latest.FullName -Tail 5 -Encoding UTF8
} else {
    Write-Host "(no train logs yet)"
}
Write-Host ""

Write-Host "===== ETA (very rough) ====="
$done = 0
$total = 12
foreach ($s in 0, 1, 2) {
    foreach ($task in '122', '123', '120', '121') {
        $tlog = Join-Path $RunsDir ("dcase2024_{0}_frac_seed{1}_test.log" -f $task, $s)
        if (Test-Path $tlog) {
            $content = Get-Content $tlog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($content -match "F 20") { $done++ }
        }
    }
}
$remain = $total - $done
$remain_h = [math]::Round($remain * 1.55, 1)
Write-Host ("done {0}/{1}, remaining ~{2} h ({3} ckpts at ~93 min each)" -f $done, $total, $remain_h, $remain)
