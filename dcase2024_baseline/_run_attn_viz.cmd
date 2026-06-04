@echo off
cd /d D:\ssl-research\dcase2024_baseline
set CUDA_VISIBLE_DEVICES=
"D:\ssl-research\venv\Scripts\python.exe" -u _path_c_attn_viz.py > "D:\ssl-research\week11_starss23\runs\path_c_attn_viz_full.log" 2>&1
