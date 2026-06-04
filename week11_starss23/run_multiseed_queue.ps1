# Sequential queue runner for the multi-seed paired runs.
# Runs no_geom and full back-to-back for each seed in $seeds.
# All console output is appended to one log so a single tail call shows the
# combined progress of the whole queue.

param(
    [int[]] $Seeds = @(1, 2),
    [string[]] $Variants = @("no_geom", "full"),
    [int]    $Epochs = 30,
    [int]    $BatchSize = 32,
    [int]    $CropsPerClip = 8,
    [string] $OutSuffix = "mc8_inmem",
    [string] $QueueLog = "D:\ssl-research\week11_starss23\runs\queue_multiseed.log"
)

$ErrorActionPreference = "Stop"
$pyExe = "D:\ssl-research\venv\Scripts\python.exe"

function Append-Log($line) {
    $line | Out-File -FilePath $QueueLog -Append -Encoding UTF8
}

if (Test-Path $QueueLog) { Remove-Item $QueueLog }

Append-Log "=== queue start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Append-Log "seeds=$($Seeds -join ',')  variants=$($Variants -join ',')  epochs=$Epochs"

$jobIdx = 0
$totalJobs = $Seeds.Count * $Variants.Count

foreach ($seed in $Seeds) {
    foreach ($variant in $Variants) {
        $jobIdx++
        $stamp = Get-Date -Format 'HH:mm:ss'
        $banner = "`n=== [$jobIdx/$totalJobs] $stamp  variant=$variant seed=$seed  starting ==="
        Write-Host $banner -ForegroundColor Cyan
        Append-Log $banner

        $argList = @(
            "-u",
            "-m", "week11_starss23.train_seld",
            "--variant", $variant,
            "--epochs", $Epochs,
            "--batch-size", $BatchSize,
            "--train-crops-per-clip", $CropsPerClip,
            "--in-memory",
            "--seed", $seed,
            "--out-suffix", $OutSuffix
        )

        $tmpStdout = "$env:TEMP\queue_${variant}_seed${seed}.stdout"
        $tmpStderr = "$env:TEMP\queue_${variant}_seed${seed}.stderr"

        $proc = Start-Process -FilePath $pyExe -ArgumentList $argList `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $tmpStdout `
            -RedirectStandardError $tmpStderr

        $proc.WaitForExit()

        if (Test-Path $tmpStdout) {
            $bytes = Get-Content -Path $tmpStdout -Encoding Byte -ReadCount 0
            $text = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "\x00",""
            $text | Out-File -FilePath $QueueLog -Append -Encoding UTF8
        }
        if (Test-Path $tmpStderr) {
            $err = Get-Content -Path $tmpStderr -Raw -ErrorAction SilentlyContinue
            if ($err -and $err.Trim().Length -gt 0) {
                Append-Log "--- stderr ($variant seed=$seed) ---"
                $err | Out-File -FilePath $QueueLog -Append -Encoding UTF8
            }
        }

        $stamp = Get-Date -Format 'HH:mm:ss'
        $exitMsg = "=== [$jobIdx/$totalJobs] $stamp  variant=$variant seed=$seed  exit=$($proc.ExitCode) ==="
        Write-Host $exitMsg -ForegroundColor Cyan
        Append-Log $exitMsg
    }
}

Append-Log "=== queue end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "[QUEUE DONE]" -ForegroundColor Green
