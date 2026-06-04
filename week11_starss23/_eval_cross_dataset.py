"""Path B / B1: zero-shot cross-dataset DCASE SELD evaluation.

Trains were on STARSS23 dev-train. We re-evaluate the existing best.pt
checkpoints on a *different* test set whose ground truth is publicly
available, *without* fine-tuning. This addresses the single-dataset
critique on the ICASSP submission.

Currently supported test sets:
    * "starss22-test"  -- STARSS22 dev-test (54 clips, 13-class taxonomy
       identical to STARSS23, 4-channel mic format).
    * "starss23-test"  -- the canonical in-distribution comparison
       (78 clips); useful as a sanity-check baseline.

Usage::

    python -m week11_starss23._eval_cross_dataset \
        --variants no_geom full \
        --seeds 0 1 2 3 4 \
        --suffix mc8_inmem \
        --testset starss22-test

Output: ``runs/cross_dataset_<testset>_<variant>_<suffix>_seed<i>.json``
plus an aggregate ``runs/cross_dataset_<testset>_summary.json`` with
paired t-tests across the variant pair (no_geom vs full, etc.).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.starss_dataset import StarssDataset

REPO = Path("D:/ssl-research")
RUNS = REPO / "week11_starss23" / "runs"
EVAL_THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)
NMS_TOL = 15.0
TOL_DEG = 20.0


TESTSET_PATHS = {
    "starss22-test": {
        "audio_dir": REPO / "data" / "STARSS22" / "mic_dev",
        "metadata_dir": REPO / "data" / "STARSS22" / "metadata_dev",
        "split": "test",
        "cache_dir": REPO / "data" / "STARSS22" / "feat_cache",
    },
    "starss23-test": {
        "audio_dir": REPO / "data" / "STARSS23" / "mic_dev",
        "metadata_dir": REPO / "data" / "STARSS23" / "metadata_dev" / "metadata_dev",
        "split": "test",
        "cache_dir": REPO / "data" / "STARSS23" / "feat_cache",
    },
}


def _eval_one_checkpoint(ckpt: Path, testset_cfg: dict) -> dict:
    """Run inference + DCASE threshold sweep on ``ckpt``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(ckpt, device)
    n_classes = blob["model_cfg"]["n_classes"]
    ds = StarssDataset(
        audio_dir=testset_cfg["audio_dir"],
        metadata_dir=testset_cfg["metadata_dir"],
        split=testset_cfg["split"],
        feature_config=feat_cfg,
        clip_seconds=None,
        random_crop=False,
        cache_dir=testset_cfg["cache_dir"],
    )
    if len(ds) == 0:
        raise RuntimeError(f"no clips found under {testset_cfg['metadata_dir']}")

    t0 = time.time()
    cached_preds: list[np.ndarray] = []
    cached_targets: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(len(ds)):
            sample = ds[k]
            feats = sample["features"].unsqueeze(0).to(device)
            target = sample["target"].numpy()
            out_flat = model(feats)["accdoa"][0].cpu().numpy()
            T = out_flat.shape[0]
            n_tracks = out_flat.shape[1] // (3 * n_classes)
            cached_preds.append(out_flat.reshape(T, n_tracks, 3, n_classes))
            cached_targets.append(target)
    inference_s = time.time() - t0

    per_threshold: dict[str, dict] = {}
    for thr in EVAL_THRESHOLDS:
        st = DcaseSeldStats(tolerance_deg=TOL_DEG, n_classes=n_classes)
        for pred, tgt in zip(cached_preds, cached_targets):
            for p, g in zip(
                decode_pred_to_events(pred, activity_threshold=thr, nms_tol_deg=NMS_TOL),
                target_to_events(tgt),
            ):
                st.add_frame(p, g)
        per_threshold[f"{thr:.2f}"] = {
            "macro": st.summary("macro"),
            "micro": st.summary("micro"),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "n_clips": len(ds),
        "inference_s": inference_s,
        "thresholds": per_threshold,
    }


def _paired_t(a: list[float], b: list[float]) -> dict:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    delta = b_arr - a_arr
    if len(delta) < 2 or np.allclose(delta.std(ddof=1), 0):
        return {
            "n": len(delta),
            "mean_a": float(a_arr.mean()),
            "mean_b": float(b_arr.mean()),
            "mean_delta": float(delta.mean()),
            "t": float("nan"), "p": float("nan"),
        }
    res = stats.ttest_rel(b_arr, a_arr)
    return {
        "n": len(delta),
        "mean_a": float(a_arr.mean()),
        "mean_b": float(b_arr.mean()),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "t": float(res.statistic),
        "p": float(res.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", default=["no_geom", "full"],
        help="variant names matching run-dir prefix"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--suffix", default="mc8_inmem",
        help="run-dir suffix (e.g. mc8_inmem, baseline_mc8_inmem for SELDnet)"
    )
    parser.add_argument(
        "--testset", required=True, choices=list(TESTSET_PATHS.keys()),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run even if per-cell JSON already exists"
    )
    args = parser.parse_args()

    cfg = TESTSET_PATHS[args.testset]
    print(f"[cross-dataset] testset = {args.testset}")
    print(f"  audio: {cfg['audio_dir']}")
    print(f"  metadata: {cfg['metadata_dir']}")
    print(f"  variants: {args.variants}  seeds: {args.seeds}  suffix: {args.suffix}")

    cells: dict[tuple[str, int], dict] = {}
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = RUNS / f"{variant}_seed{seed}_{args.suffix}"
            ckpt = run_dir / "best.pt"
            if not ckpt.exists():
                print(f"  [skip] {variant} seed={seed} -- best.pt missing in {run_dir}")
                continue
            out_path = (
                RUNS
                / f"cross_dataset_{args.testset}_{variant}_{args.suffix}_seed{seed}.json"
            )
            if out_path.exists() and not args.force:
                print(f"  [reuse] {variant} seed={seed}", flush=True)
                cells[(variant, seed)] = json.loads(
                    out_path.read_text(encoding="utf-8")
                )
                continue
            print(f"  [eval]  {variant} seed={seed}  ckpt={ckpt.name}", flush=True)
            t0 = time.time()
            result = _eval_one_checkpoint(ckpt, cfg)
            elapsed = time.time() - t0
            result["variant"] = variant
            result["seed"] = seed
            result["suffix"] = args.suffix
            result["testset"] = args.testset
            result["wall_s"] = elapsed
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            cells[(variant, seed)] = result
            print(
                f"     -> {len(result['thresholds'])} thr × 78 clips "
                f"in {elapsed:.0f}s", flush=True
            )

    # ---------- aggregate paired t-tests across variant pairs ----------
    if len(args.variants) < 2:
        return
    summary: dict = {
        "testset": args.testset,
        "suffix": args.suffix,
        "seeds": args.seeds,
        "comparisons": [],
    }
    for ia, va in enumerate(args.variants):
        for vb in args.variants[ia + 1 :]:
            for thr in EVAL_THRESHOLDS:
                thr_key = f"{thr:.2f}"
                for avg in ("macro", "micro"):
                    for metric in ("f1", "er", "le_cd", "lr_cd", "seld"):
                        a_vals: list[float] = []
                        b_vals: list[float] = []
                        seeds_used: list[int] = []
                        for seed in args.seeds:
                            ca = cells.get((va, seed))
                            cb = cells.get((vb, seed))
                            if ca is None or cb is None:
                                continue
                            a_vals.append(ca["thresholds"][thr_key][avg][metric])
                            b_vals.append(cb["thresholds"][thr_key][avg][metric])
                            seeds_used.append(seed)
                        if len(a_vals) < 2:
                            continue
                        summary["comparisons"].append({
                            "a_variant": va, "b_variant": vb,
                            "threshold": thr_key, "average": avg,
                            "metric": metric,
                            "seeds": seeds_used,
                            **_paired_t(a_vals, b_vals),
                        })

    summary_path = RUNS / f"cross_dataset_{args.testset}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {summary_path}")
    # Print headline numbers (macro SELD at 0.18 and 0.30)
    print("\nHeadline (macro SELD, lower is better):")
    for thr in ("0.18", "0.30"):
        print(f"  thr={thr}")
        for c in summary["comparisons"]:
            if c["threshold"] != thr or c["average"] != "macro" or c["metric"] != "seld":
                continue
            sig = ""
            if not np.isnan(c.get("p", float("nan"))):
                if c["p"] < 0.001: sig = " ***"
                elif c["p"] < 0.01: sig = " **"
                elif c["p"] < 0.05: sig = " *"
            print(
                f"    {c['a_variant']:>8}={c['mean_a']:.3f}  "
                f"{c['b_variant']:>8}={c['mean_b']:.3f}  "
                f"Δ={c['mean_delta']:+.3f}  p={c.get('p', float('nan')):.3f}{sig}"
            )


if __name__ == "__main__":
    main()
