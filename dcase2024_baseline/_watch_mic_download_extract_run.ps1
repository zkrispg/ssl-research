$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DcaseDir = $PSScriptRoot
$DataRoot = Join-Path $Root "DCASE2024_SELD_dataset"
$DownloadZip = Join-Path $Root "downloads\mic_dev.zip"
$DownloadState = "$DownloadZip.aria2"
$LogDir = Join-Path $Root "runs"
$Python = Join-Path $Root "venv\Scripts\python.exe"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$DriverLog = Join-Path $LogDir "watch_mic_download_extract_run_${RunStamp}.log"
$StatusFile = Join-Path $LogDir "watch_mic_download_extract_run_${RunStamp}_status.txt"
$FeatureLog = Join-Path $LogDir "feature_extraction_mic_141_${RunStamp}.log"

New-Item -Path $LogDir -ItemType Directory -Force | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $DriverLog -Value $line -Encoding UTF8
    Set-Content -Path $StatusFile -Value $Message -Encoding UTF8
}

function Test-ZipReadable {
    if (-not (Test-Path $DownloadZip)) { return $false }
    $cmd = "tar -tf `"$DownloadZip`" > NUL"
    & cmd.exe /d /c $cmd
    return ($LASTEXITCODE -eq 0)
}

function Wait-ForDownload {
    Write-WatchLog "waiting for $DownloadZip"
    while ($true) {
        if ((Test-Path $DownloadZip) -and -not (Test-Path $DownloadState)) {
            Write-WatchLog "download state file gone; testing zip"
            if (Test-ZipReadable) {
                Write-WatchLog "zip test passed"
                return
            }
            Write-WatchLog "zip test failed; waiting for a complete readable archive"
        }
        Start-Sleep -Seconds 300
    }
}

function Ensure-MicExtracted {
    $micDir = Join-Path $DataRoot "mic_dev"
    $wavCount = 0
    if (Test-Path $micDir) {
        $wavCount = (Get-ChildItem -Path $micDir -Recurse -Filter "*.wav" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if ($wavCount -ge 160) {
        Write-WatchLog "mic_dev already extracted: wav_count=$wavCount"
        return
    }

    Write-WatchLog "extracting mic_dev.zip to $DataRoot"
    Expand-Archive -Path $DownloadZip -DestinationPath $DataRoot -Force
    $wavCount = (Get-ChildItem -Path $micDir -Recurse -Filter "*.wav" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($wavCount -lt 160) {
        throw "mic_dev extraction incomplete: wav_count=$wavCount"
    }
    Write-WatchLog "mic_dev extraction complete: wav_count=$wavCount"
}

function Ensure-MicFeatures {
    $featDirs = @(
        (Join-Path $DataRoot "seld_feat_label\mic_dev_norm"),
        (Join-Path $DataRoot "seld_feat_label\mic_dev_gcc_norm")
    )
    $labelDir = Join-Path $DataRoot "seld_feat_label\mic_dev_adpit_label"
    $featCount = 0
    $labelCount = 0
    foreach ($featDir in $featDirs) {
        if (-not (Test-Path $featDir)) { continue }
        $count = (Get-ChildItem -Path $featDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -gt $featCount) {
            $featCount = $count
        }
    }
    if (Test-Path $labelDir) {
        $labelCount = (Get-ChildItem -Path $labelDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if ($featCount -ge 160 -and $labelCount -ge 160) {
        Write-WatchLog "MIC features already ready: features=$featCount labels=$labelCount"
        return
    }

    Write-WatchLog "extracting MIC-GCC features with task 141"
    Push-Location $DcaseDir
    try {
        $cmd = "`"$Python`" -u batch_feature_extraction.py 141 > `"$FeatureLog`" 2>&1"
        & cmd.exe /d /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "feature extraction failed exit=$LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    $featCount = 0
    foreach ($featDir in $featDirs) {
        if (-not (Test-Path $featDir)) { continue }
        $count = (Get-ChildItem -Path $featDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -gt $featCount) {
            $featCount = $count
        }
    }
    $labelCount = (Get-ChildItem -Path $labelDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($featCount -lt 160 -or $labelCount -lt 160) {
        throw "MIC features incomplete: features=$featCount labels=$labelCount"
    }
    Write-WatchLog "MIC features ready: features=$featCount labels=$labelCount"
}

function Wait-ForFoaRunner {
    Write-WatchLog "waiting for FOA runner to finish before starting MIC training"
    while ($true) {
        $running = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "python|powershell" -and
                ($_.CommandLine -match "dcase2024_130_det_gca_seed|dcase2024_131_det_gca_seed|_run_gca_foa_crnn_deterministic_local.ps1")
            }
        if (-not $running) {
            Write-WatchLog "FOA runner no longer active"
            return
        }
        Start-Sleep -Seconds 300
    }
}

try {
    Write-WatchLog "MIC watcher starts"
    Wait-ForDownload
    Ensure-MicExtracted
    Ensure-MicFeatures
    Wait-ForFoaRunner
    Write-WatchLog "starting MIC+Transformer deterministic rerun"
    $script = Join-Path $DcaseDir "_run_gca_mic_xfm_deterministic_local.ps1"
    $out = Join-Path $LogDir "gca_mic_xfm_det_seld_launcher_auto.out.log"
    $err = Join-Path $LogDir "gca_mic_xfm_det_seld_launcher_auto.err.log"
    $proc = Start-Process -FilePath powershell.exe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    Write-WatchLog "MIC+Transformer runner launched pid=$($proc.Id)"
}
catch {
    Write-WatchLog "FAILED: $_"
    throw
}
