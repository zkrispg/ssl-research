# Wait for Tier VIII orchestrator to finish, then run B (polish) + E (synth)
# analyses in sequence. Designed to run unattended overnight.
#
# Sentinels watched:
#   - "Tier VIII done." marker in dcase2024_8_orchestrator.out
#   - All 9 ckpts (150/151/152 x seeds 0..2) trained and tested.
#
# Pipeline once Tier VIII is done:
#   1. STARSS22 cross-eval for new cells   (~30 min GPU + feat extract)
#   2. Linear probing on 4 cells           (~30-60 min GPU)
#   3. 2x2 ANOVA + within-cell paired      (CPU, ~10s)
#   4. Per-class breakdown                 (CPU, ~5 min)  [run anyway in case re-run needed]
#   5. 2x2 dissociation analyzer           (CPU)
#   6. (E) TAU-NIGENS-SSE-2021 zero-shot   (~30-60 min GPU; only if data downloaded)
#   7. Build progress doc v3
#
# Output sentinel: D:\ssl-research\runs\after_tier8_done.txt

$ErrorActionPreference = "Stop"
$repo  = "D:\ssl-research\dcase2024_baseline"
$runs  = "D:\ssl-research\runs"
$paper = "D:\ssl-research\paper"
$logf  = Join-Path $runs "after_tier8_orchestrator.log"
"$([DateTime]::Now.ToString('o')) START chain after Tier VIII" | Out-File $logf -Encoding utf8

function Log($msg) {
    "$([DateTime]::Now.ToString('HH:mm:ss')) $msg" | Out-File $logf -Append -Encoding utf8
    Write-Host "$([DateTime]::Now.ToString('HH:mm:ss')) $msg"
}

function Wait-ForTier8 {
    $orchOut = Join-Path $runs "dcase2024_8_orchestrator.out"
    Log "watching for Tier VIII completion at $orchOut ..."
    while ($true) {
        if (Test-Path $orchOut) {
            $tail = Get-Content $orchOut -Raw -Encoding UTF8
            # Orchestrator emits this string when training+test+analyze loop is done.
            if ($tail -match "Tier VIII \(FOA \+ Xfm\) ALL DONE" -or
                $tail -match "2x2 ANALYZE \+ DOC REBUILD DONE") {
                Log "found Tier VIII completion marker"
                return $true
            }
            $ckpt_count = 0
            foreach ($t in '150', '151', '152') {
                foreach ($s in 0, 1, 2) {
                    $logTest = Join-Path $runs "dcase2024_${t}_ablate_seed${s}_test.log"
                    if (Test-Path $logTest) {
                        $c = Get-Content $logTest -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
                        if ($c -match "F 20") { $ckpt_count++ }
                    }
                }
            }
            if ($ckpt_count -ge 9) {
                Log "all 9 Tier VIII ckpts have F 20 metrics in test log"
                return $true
            }
        }
        Start-Sleep -Seconds 300
    }
}

function Run-Step($name, $cmd, $argsList) {
    Log "===== STEP $name : $cmd $argsList ====="
    $stepLog = Join-Path $runs "after_tier8_step_$($name).log"
    & cmd.exe /c "cd /d $repo && $cmd $argsList > `"$stepLog`" 2>&1"
    $rc = $LASTEXITCODE
    Log "STEP $name exit=$rc (log: $stepLog)"
    if ($rc -ne 0) {
        Log "WARN step $name returned $rc; continuing with remaining steps"
    }
}

# 0. Wait for Tier VIII to finish
$ok = Wait-ForTier8
if (-not $ok) { Log "Wait-ForTier8 returned false; aborting"; exit 1 }
Log "Tier VIII complete -- proceeding with B + E pipeline"

$py = "D:\ssl-research\venv\Scripts\python.exe -u"

# 1. STARSS22 cross-eval for new cells (130/131/140-142/150-152). MIC features
#    already extracted from previous run; FOA features will be extracted in
#    this step as needed (idempotent).
Run-Step "1_cross22"  $py "_path_c_cross_starss22.py --cells 100 130 131 140 141 142 150 151 152"

# 2. Linear probing for the 4-cell dissociation (CRNN+MIC, CRNN+FOA, Xfm+MIC,
#    Xfm+FOA). The MIC+CRNN cells (110/111/112) already have probed values;
#    we re-probe everything for consistency and to populate JSON.
Run-Step "2_probe"    $py "_path_c_probe.py"

# 3. 2x2 ANOVA + within-cell paired stats (depends on test logs being present)
Run-Step "3_anova"    $py "_path_c_2x2_anova.py"

# 4. Per-class breakdown (also re-runs to catch FOA + Xfm cells)
Run-Step "4_perclass" $py "_path_c_per_class.py"

# 5. 2x2 dissociation analyzer + plot
Run-Step "5_dissoc"   $py "_path_c_2x2_dissociation.py"

# 6. (E) TAU-NIGENS-SSE-2021 zero-shot evaluation -- only run if TAU data ready.
$tauReady = Test-Path "D:\ssl-research\TAU_NIGENS_SSE_2021\foa_dev.zip"
$tauExtracted = Test-Path "D:\ssl-research\TAU_NIGENS_SSE_2021\foa_dev"
$tauScript = Join-Path $repo "_path_c_synth_nigens.py"
if ((Test-Path $tauScript) -and ($tauReady -or $tauExtracted)) {
    Run-Step "6_synth" $py "_path_c_synth_nigens.py"
} else {
    Log "STEP 6 skipped (TAU-NIGENS data or analyzer script not ready)"
}

# 7. Build progress doc v3
Run-Step "7_doc" $py "_build_progress_doc_v2.py"

# Final sentinel
$sent = Join-Path $runs "after_tier8_done.txt"
"$([DateTime]::Now.ToString('o')) all post-Tier-VIII steps complete" | Out-File $sent -Encoding utf8
Log "ALL DONE -- sentinel written to $sent"
