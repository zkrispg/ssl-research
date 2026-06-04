"""Smoke test: verify train_data_fraction subsamples train files and not val/test."""
from __future__ import annotations
import os, sys
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
os.chdir(r"D:\ssl-research\dcase2024_baseline")
import parameters
import cls_data_generator

for task in ("110", "120", "122"):
    p = parameters.get_params(task).copy()
    print(f"\n=== task {task} (train_data_fraction={p.get('train_data_fraction', 1.0)}) ===")
    dg_train = cls_data_generator.DataGenerator(params=p, split=[1, 2, 3], shuffle=False)
    dg_val   = cls_data_generator.DataGenerator(params=p, split=[4],       shuffle=False)
    print(f"  train files: {len(dg_train.get_filelist())}")
    print(f"  val files:   {len(dg_val.get_filelist())}")
    if hasattr(dg_train, "_filenames_list"):
        print(f"  first train file: {dg_train._filenames_list[0]}")
        print(f"  last  train file: {dg_train._filenames_list[-1]}")
