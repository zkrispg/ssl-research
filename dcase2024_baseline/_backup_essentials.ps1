# Back up the IRREPLACEABLE core of the research project (~8 GB): all code,
# trained checkpoints, pretrained weights, the paper, and every experiment log.
# Public datasets (STARSS23/22, TAU-NIGENS, raw audio) and venv/.wheels are
# EXCLUDED on purpose -- they are re-downloadable / re-buildable on the new machine.
#
# Usage (run on the OLD 3050Ti machine, BEFORE selling it):
#   powershell -ExecutionPolicy Bypass -File _backup_essentials.ps1 -Dest E:\backup
#   (Dest = external drive or a synced cloud folder, e.g. OneDrive\backup)
param([Parameter(Mandatory=$true)][string]$Dest)

$ErrorActionPreference = "Stop"
$Src  = "D:\ssl-research"
$Root = Join-Path $Dest "ssl-research-essentials"
Write-Host "Backing up core -> $Root"

# 1) baseline code + ALL trained checkpoints (models_audio) + pretrained weights
robocopy "$Src\dcase2024_baseline" "$Root\dcase2024_baseline" /E `
    /XD __pycache__ .ipynb_checkpoints /NFL /NDL /NJH

# 2) the paper (tex, pdf, figs, json/md artifacts)
robocopy "$Src\paper" "$Root\paper" /E /XD __pycache__ /NFL /NDL /NJH

# 3) every experiment log (the raw results behind every table)
robocopy "$Src\runs" "$Root\runs" /E /NFL /NDL /NJH
if (Test-Path "$Src\week11_starss23\runs") {
    robocopy "$Src\week11_starss23\runs" "$Root\week11_starss23\runs" /E /NFL /NDL /NJH
}

# 4) all week* research code (small, but it's your work)
Get-ChildItem $Src -Directory -Filter "week*" -ErrorAction SilentlyContinue | ForEach-Object {
    robocopy $_.FullName (Join-Path $Root $_.Name) /E /XD __pycache__ models results /NFL /NDL /NJH | Out-Null
}

# 5) tools (e.g. tectonic for compiling the paper)
if (Test-Path "$Src\tools") {
    robocopy "$Src\tools" "$Root\tools" /E /NFL /NDL /NJH | Out-Null
}

Write-Host ""
$sz = (Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
"DONE. Core backup = {0:N1} GB at {1}" -f ($sz/1GB), $Root
Write-Host "Keep TWO copies of this (one external drive + one cloud) before wiping the old PC."
