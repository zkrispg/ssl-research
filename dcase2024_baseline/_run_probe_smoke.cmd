@echo off
cd /d D:\ssl-research\dcase2024_baseline
set CUDA_VISIBLE_DEVICES=
"D:\ssl-research\venv\Scripts\python.exe" -u _path_c_probe.py --cells 110 --seeds 0 --max-files 5 > "D:\ssl-research\week11_starss23\runs\path_c_probe_smoke.log" 2>&1
