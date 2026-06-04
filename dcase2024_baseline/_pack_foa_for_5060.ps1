# Pack the ~5.3 GB FOA-only subset needed to run the convbias supplement on the
# 5060 machine. Copies into <Dest>\ssl-research preserving the D:\ssl-research
# layout, so on the 5060 you can drop it straight into D:\ssl-research (zero path
# edits). Excludes big/uneeded dirs (trained ckpts, results, caches, MIC feats).
#
# Usage (run on THIS machine):
#   powershell -ExecutionPolicy Bypass -File _pack_foa_for_5060.ps1 -Dest E:\transfer
#   (replace E:\transfer with your USB / external drive / share root)
param([Parameter(Mandatory=$true)][string]$Dest)

$ErrorActionPreference = "Stop"
$Src  = "D:\ssl-research"
$Root = Join-Path $Dest "ssl-research"
Write-Host "Packing FOA subset -> $Root"

# code (exclude trained ckpts / results / caches / git)
robocopy "$Src\dcase2024_baseline" "$Root\dcase2024_baseline" /E `
    /XD models_audio results_audio __pycache__ .git .ipynb_checkpoints /NFL /NDL /NJH
robocopy "$Src\week09_geometry_attn" "$Root\week09_geometry_attn" /E /XD __pycache__ /NFL /NDL /NJH

# FOA features + labels + normalization weights
$FL = "$Src\DCASE2024_SELD_dataset\seld_feat_label"
$FLd = "$Root\DCASE2024_SELD_dataset\seld_feat_label"
robocopy "$FL\foa_dev_norm"        "$FLd\foa_dev_norm"        /E /NFL /NDL /NJH
robocopy "$FL\foa_dev_adpit_label" "$FLd\foa_dev_adpit_label" /E /NFL /NDL /NJH
robocopy "$FL" "$FLd" foa_wts /NFL /NDL /NJH

Write-Host ""
$sz = (Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
"DONE. Packed {0:N1} GB into {1}" -f ($sz/1GB), $Root
Write-Host "On the 5060: copy '$Root' contents into D:\ssl-research (merge), then follow setup steps."
