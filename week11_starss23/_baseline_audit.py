"""Compare absolute performance of every variant cell at thr=0.30 (best per spec).

Lists each variant's macro F1, micro F1, macro SELD aggregated over its 5 seeds.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

R = Path("D:/ssl-research/week11_starss23/runs")

# (label, dir-prefix, suffix)
GROUPS = [
    ("no_geom         (vanilla, 30ep)", "no_geom",          "mc8_inmem"),
    ("full            (vanilla, 30ep)", "full",             "mc8_inmem"),
    ("seldnet         (vanilla, 30ep)", "seldnet_official", "baseline_mc8_inmem"),
    ("no_geom + spec  (strong, 30ep)",  "no_geom",          "mc8_inmem_specaug"),
    ("full + spec     (strong, 30ep)",  "full",             "mc8_inmem_specaug"),
    ("seldnet + spec  (strong, 30ep)",  "seldnet_official", "baseline_mc8_inmem_specaug"),
    ("no_geom + spec_w(weak  , 30ep)",  "no_geom",          "mc8_inmem_specaug_weak"),
    ("full + spec_w   (weak  , 30ep)",  "full",             "mc8_inmem_specaug_weak"),
    ("no_geom_xl      (cap, 30ep)",     "no_geom_xl",       "cap_xl_mc8_inmem"),
    ("full_xl         (cap, 30ep)",     "full_xl",          "cap_xl_mc8_inmem"),
    ("seldnet_FOA     (vanilla, 30ep)", "seldnet_official", "foa_baseline_mc8_inmem"),
]

print("=" * 96)
print(f'{"variant":<38}{"thr":<6}{"macro_F1":>10}{"micro_F1":>10}{"macro_SELD":>12}{"params":>10}')
print("=" * 96)

for label, prefix, suffix in GROUPS:
    rows_macro_f1, rows_micro_f1, rows_seld, params = [], [], [], None
    for seed in range(5):
        cell = R / f"{prefix}_seed{seed}_{suffix}"
        sweep = cell / "eval_threshold_sweep.json"
        if not sweep.exists():
            continue
        d = json.loads(sweep.read_text(encoding="utf-8"))
        if params is None:
            sm = cell / "summary.json"
            if sm.exists():
                params = json.loads(sm.read_text(encoding="utf-8")).get("n_params")
        thr = "0.30"
        block = d.get("thresholds", {}).get(thr)
        if block is None:
            continue
        rows_macro_f1.append(block["macro"]["f1"])
        rows_micro_f1.append(block["micro"]["f1"])
        rows_seld.append(block["macro"]["seld"])
    if not rows_macro_f1:
        print(f"{label:<38} (no data)")
        continue
    mf = statistics.mean(rows_macro_f1)
    sf = statistics.stdev(rows_macro_f1) if len(rows_macro_f1) > 1 else 0.0
    mif = statistics.mean(rows_micro_f1)
    sif = statistics.stdev(rows_micro_f1) if len(rows_micro_f1) > 1 else 0.0
    sd = statistics.mean(rows_seld)
    sd_s = statistics.stdev(rows_seld) if len(rows_seld) > 1 else 0.0
    n = len(rows_macro_f1)
    print(
        f"{label:<38}"
        f"{'0.30':<6}"
        f"{mf*100:>6.2f}±{sf*100:<3.2f}"
        f"  {mif*100:>6.2f}±{sif*100:<3.2f}"
        f"  {sd:>5.3f}±{sd_s:<5.3f}"
        f"{(params or 0):>10}"
    )

print("=" * 96)
print("note: F1 in % at thr=0.30 (best on average), SELD = (er + (1-f1) + le_norm + (1-lr))/4")
print("ref : DCASE 2023 official MIC AO baseline ~30% macro F1 @ STARSS23 dev-test (100 epochs)")
print("ref : DCASE 2024 official AO baseline ~16-18% macro F1 (2024 metric folds in distance)")
