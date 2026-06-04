"""T1d -- per-class F1 / LE_CD / LR_CD breakdown for the N = 5 geometry pair.

For each seed in {0..4} and each variant in {no_geom, full}, load the
best.pt, run full STARSS23 dev-test inference, and compute per-class
DCASE Task 3 metrics at thr = 0.18. Then pair across seeds and run a
paired t-test for every class on every metric.

This answers the reviewer question "which classes drive the +19.6 %
LE_CD-macro penalty?" — Section 4.2 of the paper currently claims it is
the *rare* classes; this script provides the empirical evidence.

Output: ``runs/analysis_perclass.json`` plus a printed Markdown-ready
table.

Cost: ~10 minutes on GPU (10 cells × 78 clips × full-clip inference).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.seld_labels import DCASE2023_CLASSES
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.starss_dataset import StarssDataset

RUNS = Path("D:/ssl-research/week11_starss23/runs")
OUT = RUNS / "analysis_perclass.json"
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD = 0.18
NMS_TOL = 15.0
TOL_DEG = 20.0
N_CLASSES = len(DCASE2023_CLASSES)


def _evaluate_per_class(run_dir: Path) -> dict[str, np.ndarray]:
    """Run full-clip inference and return per-class DCASE metric arrays.

    Caches the raw counters (tp/fp/fn/...) per seed/variant so that aggregation
    failures don't force a re-run of the GPU inference.
    """
    cache = run_dir / "perclass_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return {
            "f1": z["f1"], "le_cd": z["le_cd"], "lr_cd": z["lr_cd"],
            "er_cd": z["er_cd"], "precision": z["precision"], "recall": z["recall"],
            "n_ref_per_class": z["n_ref_per_class"],
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = run_dir / "best.pt"
    model, feat_cfg, blob = load_checkpoint(ckpt, device)
    n_classes = blob["model_cfg"]["n_classes"]
    ds = StarssDataset(
        audio_dir=Path("D:/ssl-research/data/STARSS23/mic_dev"),
        metadata_dir=Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev"),
        split="test",
        feature_config=feat_cfg,
        clip_seconds=None,
        random_crop=False,
        cache_dir=Path("D:/ssl-research/data/STARSS23/feat_cache"),
    )
    stats_obj = DcaseSeldStats(tolerance_deg=TOL_DEG, n_classes=n_classes)
    with torch.no_grad():
        for k in range(len(ds)):
            sample = ds[k]
            feats = sample["features"].unsqueeze(0).to(device)
            target = sample["target"].numpy()
            out_flat = model(feats)["accdoa"][0].cpu().numpy()
            T = out_flat.shape[0]
            n_tracks = out_flat.shape[1] // (3 * n_classes)
            pred = out_flat.reshape(T, n_tracks, 3, n_classes)
            pred_frames = decode_pred_to_events(
                pred, activity_threshold=THRESHOLD, nms_tol_deg=NMS_TOL
            )
            gt_frames = target_to_events(target)
            for p, g in zip(pred_frames, gt_frames):
                stats_obj.add_frame(p, g)
    pc = stats_obj.per_class_metrics()
    out = {k: np.asarray(v, dtype=np.float64) for k, v in pc.items()}
    out["n_ref_per_class"] = stats_obj.n_ref_per_class.astype(np.int64)
    np.savez(cache, **out)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _paired_t(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test plus Wilcoxon signed-rank (non-parametric companion)."""
    delta = b - a
    out: dict[str, float] = {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
    }
    if len(delta) < 2 or np.allclose(delta.std(ddof=1), 0):
        out.update({
            "t": float("nan"), "p": float("nan"),
            "wilcoxon_W": float("nan"), "wilcoxon_p": float("nan"),
        })
        return out
    res = stats.ttest_rel(b, a)
    out["t"] = float(res.statistic)
    out["p"] = float(res.pvalue)
    # Wilcoxon: requires nonzero deltas; with N=5 the smallest possible p-value
    # is 1/2^4 = 0.0625, so this NEVER survives Bonferroni for K=13 classes.
    # We still report it as a robustness check on the parametric assumptions.
    nz = delta[delta != 0]
    if len(nz) >= 1:
        try:
            wres = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_W"] = float(wres.statistic)
            out["wilcoxon_p"] = float(wres.pvalue)
        except ValueError:
            out["wilcoxon_W"] = float("nan")
            out["wilcoxon_p"] = float("nan")
    else:
        out["wilcoxon_W"] = float("nan")
        out["wilcoxon_p"] = float("nan")
    return out


def main() -> None:
    print(f"[analysis] per-class breakdown -> {OUT}", flush=True)
    print(f"[analysis] seeds={SEEDS}  thr={THRESHOLD}  classes={N_CLASSES}\n", flush=True)

    no_geom_runs: dict[int, dict[str, np.ndarray]] = {}
    full_runs: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        for variant, store in (("no_geom", no_geom_runs), ("full", full_runs)):
            run_dir = RUNS / f"{variant}_seed{seed}_mc8_inmem"
            print(f"  [{variant} seed={seed}] evaluating per-class metrics...",
                  flush=True)
            store[seed] = _evaluate_per_class(run_dir)

    # ---------- aggregate per-class ----------
    rows: list[dict[str, Any]] = []
    metric_names = ("f1", "er_cd", "le_cd", "lr_cd")
    for c in range(N_CLASSES):
        row: dict[str, Any] = {
            "class_id": c,
            "class_name": DCASE2023_CLASSES[c],
            "mean_n_ref": float(np.mean(
                [no_geom_runs[s]["n_ref_per_class"][c] for s in SEEDS]
            )),
        }
        for m in metric_names:
            ng_vec = np.array(
                [no_geom_runs[s][m][c] for s in SEEDS], dtype=np.float64
            )
            ft_vec = np.array(
                [full_runs[s][m][c] for s in SEEDS], dtype=np.float64
            )
            row[f"{m}_no_geom_mean"] = float(np.mean(ng_vec))
            row[f"{m}_full_mean"] = float(np.mean(ft_vec))
            row[f"{m}_paired"] = _paired_t(ng_vec, ft_vec)
        rows.append(row)

    # ---------- multiple-comparison correction ----------
    # We test 4 metrics × N_CLASSES classes = 4*13 = 52 hypotheses; Bonferroni
    # alpha is 0.05/52 ≈ 0.00096. Per-metric Bonferroni (within a metric, K=13)
    # gives 0.05/13 ≈ 0.00385.
    n_classes_active = sum(1 for r in rows if r["mean_n_ref"] > 0)
    bonf_within_metric = 0.05 / max(n_classes_active, 1)
    bonf_global = 0.05 / max(len(metric_names) * n_classes_active, 1)
    bonferroni = {
        "within_metric_alpha": float(bonf_within_metric),
        "global_alpha": float(bonf_global),
        "n_classes_active": n_classes_active,
        "n_metrics": len(metric_names),
    }

    out = {
        "seeds": list(SEEDS),
        "threshold": THRESHOLD,
        "bonferroni": bonferroni,
        "classes": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------- print Markdown table ----------
    print(
        "\nPer-class paired t-test (full - no_geom) at thr=0.18, N=5\n"
        f"{'cls':<3} {'name':<30} {'n_ref':>6} "
        f"{'F1_ng':>7} {'F1_ft':>7} {'F1_p':>6}  "
        f"{'LE_ng':>7} {'LE_ft':>7} {'LE_p':>6}  "
        f"{'LR_ng':>7} {'LR_ft':>7} {'LR_p':>6}",
        flush=True,
    )
    print("-" * 130)
    for row in rows:
        f1 = row["f1_paired"]
        le = row["le_cd_paired"]
        lr = row["lr_cd_paired"]
        print(
            f"{row['class_id']:<3} {row['class_name'][:29]:<30} "
            f"{row['mean_n_ref']:>6.1f} "
            f"{row['f1_no_geom_mean']:>7.3f} {row['f1_full_mean']:>7.3f} "
            f"{f1['p']:>6.3f}  "
            f"{row['le_cd_no_geom_mean']:>7.1f} {row['le_cd_full_mean']:>7.1f} "
            f"{le['p']:>6.3f}  "
            f"{row['lr_cd_no_geom_mean']:>7.3f} {row['lr_cd_full_mean']:>7.3f} "
            f"{lr['p']:>6.3f}"
        )

    # Identify top-3 classes driving the LE_CD effect.
    print("\nTop classes by |Δ LE_CD| (full - no_geom):", flush=True)
    sorted_le = sorted(rows, key=lambda r: -abs(r["le_cd_paired"]["mean_delta"]))
    for r in sorted_le[:3]:
        d = r["le_cd_paired"]
        print(f"  class {r['class_id']:>2} {r['class_name'][:30]:<30}  "
              f"Δ={d['mean_delta']:+.2f}°  p={d['p']:.3f}  "
              f"(n_ref={r['mean_n_ref']:.0f})")

    # ---------- Bonferroni summary ----------
    print(
        f"\nMultiple-comparison correction: {n_classes_active} active classes, "
        f"{len(metric_names)} metrics."
    )
    print(f"  Per-metric Bonferroni alpha = 0.05/K = {bonf_within_metric:.4f}")
    print(f"  Global    Bonferroni alpha = 0.05/(M*K) = {bonf_global:.4f}")
    n_t_signif_metric = sum(
        1 for r in rows for m in metric_names
        if not np.isnan(r[f"{m}_paired"]["p"])
        and r[f"{m}_paired"]["p"] < bonf_within_metric
    )
    n_w_signif_metric = sum(
        1 for r in rows for m in metric_names
        if not np.isnan(r[f"{m}_paired"]["wilcoxon_p"])
        and r[f"{m}_paired"]["wilcoxon_p"] < bonf_within_metric
    )
    print(f"  classes surviving per-metric Bonferroni (parametric t):  {n_t_signif_metric}")
    print(f"  classes surviving per-metric Bonferroni (Wilcoxon SR):   {n_w_signif_metric}")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
