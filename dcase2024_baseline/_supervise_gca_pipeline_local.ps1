$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DcaseDir = $PSScriptRoot
$LogDir = Join-Path $Root "runs"
$DownloadZip = Join-Path $Root "downloads\mic_dev.zip"
$DownloadState = "$DownloadZip.aria2"
$MicFeatureDirs = @(
    (Join-Path $Root "DCASE2024_SELD_dataset\seld_feat_label\mic_dev_norm"),
    (Join-Path $Root "DCASE2024_SELD_dataset\seld_feat_label\mic_dev_gcc_norm")
)
$MicLabelDir = Join-Path $Root "DCASE2024_SELD_dataset\seld_feat_label\mic_dev_adpit_label"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SupervisorLog = Join-Path $LogDir "supervise_gca_pipeline_${RunStamp}.log"
$SupervisorStatus = Join-Path $LogDir "supervise_gca_pipeline_${RunStamp}_status.txt"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Write-SupervisorLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $SupervisorLog -Value $line -Encoding UTF8
    Set-Content -Path $SupervisorStatus -Value $Message -Encoding UTF8
}

function Get-MatchingProcess {
    param([string]$Pattern)
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $Pattern }
}

function Test-ZipReadable {
    if (-not (Test-Path $DownloadZip)) { return $false }
    $cmd = "tar -tf `"$DownloadZip`" > NUL"
    & cmd.exe /d /c $cmd
    return ($LASTEXITCODE -eq 0)
}

function Test-CompleteLog {
    param([string]$TaskId, [int]$Seed)
    $path = Join-Path $LogDir "dcase2024_${TaskId}_det_gca_seed${Seed}_test.log"
    if (-not (Test-Path $path)) { return $false }
    $content = Get-Content $path -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    return ($content -match "F\s*20" -and $content -match "DOAE_CD" -and $content -match "SELD score")
}

function Get-CompleteCount {
    param([string[]]$TaskIds)
    $count = 0
    foreach ($seed in 0, 1, 2, 3, 4) {
        foreach ($task in $TaskIds) {
            if (Test-CompleteLog -TaskId $task -Seed $seed) {
                $count += 1
            }
        }
    }
    return $count
}

function Get-MicFeatureCount {
    $best = 0
    foreach ($dir in $MicFeatureDirs) {
        if (-not (Test-Path $dir)) { continue }
        $count = (Get-ChildItem -Path $dir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -gt $best) {
            $best = $count
        }
    }
    return $best
}

function Test-MicFeaturesReady {
    $featCount = Get-MicFeatureCount
    $labelCount = 0
    if (Test-Path $MicLabelDir) {
        $labelCount = (Get-ChildItem -Path $MicLabelDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    return ($featCount -ge 160 -and $labelCount -ge 160)
}

function Start-MicDownload {
    $url = "https://zenodo.org/records/7709052/files/mic_dev.zip?download=1"
    $outDir = Join-Path $Root "downloads"
    New-Item -Path $outDir -ItemType Directory -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LogDir "download_mic_dev_${stamp}_supervisor.log"
    $err = Join-Path $LogDir "download_mic_dev_${stamp}_supervisor.err.log"
    $args = @(
        "-x", "8", "-s", "8", "-c",
        "--file-allocation=none",
        "--summary-interval=60",
        "--console-log-level=notice",
        "--retry-wait=30",
        "--max-tries=0",
        "-d", $outDir,
        "-o", "mic_dev.zip",
        $url
    )
    $proc = Start-Process -FilePath "aria2c.exe" -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
    Write-SupervisorLog "restarted MIC download pid=$($proc.Id) log=$log"
}

function Start-HiddenPowerShell {
    param([string]$ScriptName, [string]$Prefix)
    $script = Join-Path $DcaseDir $ScriptName
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $LogDir "${Prefix}_${stamp}.out.log"
    $err = Join-Path $LogDir "${Prefix}_${stamp}.err.log"
    $proc = Start-Process -FilePath powershell.exe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    Write-SupervisorLog "started $ScriptName pid=$($proc.Id)"
}

Write-SupervisorLog "GCA pipeline supervisor starts"

while ($true) {
    try {
        $downloadComplete = ((Test-Path $DownloadZip) -and -not (Test-Path $DownloadState) -and (Test-ZipReadable))
        $downloadProc = Get-MatchingProcess "aria2c.*mic_dev\.zip"
        if (-not $downloadComplete -and -not $downloadProc) {
            Write-SupervisorLog "MIC download incomplete and aria2c not running; restarting"
            Start-MicDownload
        }

        $watcherProc = Get-MatchingProcess "_watch_mic_download_extract_run\.ps1"
        if (-not (Test-MicFeaturesReady) -and -not $watcherProc) {
            Write-SupervisorLog "MIC watcher not running and features not ready; restarting watcher"
            Start-HiddenPowerShell -ScriptName "_watch_mic_download_extract_run.ps1" -Prefix "watch_mic_supervisor"
        }

        $foaDone = Get-CompleteCount -TaskIds @("130", "131")
        $foaProc = Get-MatchingProcess "(_run_gca_foa_crnn_deterministic_local\.ps1|train_seldnet\.py\s+(130|131)\s+det_gca_seed|test_only\.py\s+(130|131)\s+det_gca_seed)"
        if ($foaDone -lt 10 -and -not $foaProc) {
            Write-SupervisorLog "FOA rerun incomplete ($foaDone/10) and not running; restarting FOA runner"
            Start-HiddenPowerShell -ScriptName "_run_gca_foa_crnn_deterministic_local.ps1" -Prefix "gca_foa_supervisor"
        }

        $micFeaturesReady = Test-MicFeaturesReady
        $micDone = Get-CompleteCount -TaskIds @("141", "142")
        $micProc = Get-MatchingProcess "(_run_gca_mic_xfm_deterministic_local\.ps1|train_seldnet\.py\s+(141|142)\s+det_gca_seed|test_only\.py\s+(141|142)\s+det_gca_seed)"
        if ($micFeaturesReady -and $foaDone -eq 10 -and $micDone -lt 10 -and -not $micProc) {
            Write-SupervisorLog "MIC rerun ready but incomplete ($micDone/10); starting MIC runner"
            Start-HiddenPowerShell -ScriptName "_run_gca_mic_xfm_deterministic_local.ps1" -Prefix "gca_mic_supervisor"
        }

        Write-SupervisorLog "heartbeat: download_complete=$downloadComplete foa=$foaDone/10 mic_features=$micFeaturesReady mic=$micDone/10"
    }
    catch {
        Write-SupervisorLog "supervisor loop error: $_"
    }
    Start-Sleep -Seconds 300
}
