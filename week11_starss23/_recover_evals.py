"""Run DCASE threshold sweep on the 3 ckpts whose queue eval was lost.

Also writes a synthetic summary.json for full_seed2 (queue died before
the [done] block in train_seld.py wrote it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.starss_dataset import StarssDataset

THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)
RUNS = Path("D:/ssl-research/week11_starss23/runs")

# (variant, seed, manual_best_eval_loss_for_summary_json_or_None)
TARGETS = [
    ("no_geom", 0, None),
    ("full", 0, None),
    ("full", 2, 0.02039),  # epoch 26, see queue_full_seed2.log
]


def evaluate_ckpt(ckpt_path: Path, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(ckpt_path, device)
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
    cached_preds: list[np.ndarray] = []
    cached_targets: list[np.ndarray] = []
    print(f"  forward over {len(ds)} clips ...", flush=True)
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

    per_threshold: dict[str, dict] = {}
    for thr in THRESHOLDS:
        stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=n_classes)
        for pred, tgt in zip(cached_preds, cached_targets):
            for p, g in zip(
                decode_pred_to_events(pred, activity_threshold=thr, nms_tol_deg=15.0),
                target_to_events(tgt),
            ):
                stats.add_frame(p, g)
        per_threshold[f"{thr:.2f}"] = {
            "macro": stats.summary("macro"),
            "micro": stats.summary("micro"),
        }
    out = {"thresholds": per_threshold, "n_clips_eval": len(ds)}
    (run_dir / "eval_threshold_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    del model, cached_preds, cached_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    for variant, seed, manual_best in TARGETS:
        run_dir = RUNS / f"{variant}_seed{seed}_mc8_inmem"
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {variant} seed={seed}: no best.pt", flush=True)
            continue
        print(f"\n[eval] {variant} seed={seed}", flush=True)
        evaluate_ckpt(ckpt, run_dir)

        if manual_best is not None and not (run_dir / "summary.json").exists():
            summary = {
                "best_eval_loss": manual_best,
                "n_params": 590966,
                "note": "training stopped at epoch 26 (queue runner died); "
                        "best_eval_loss recovered from queue_full_seed2.log",
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"  wrote synthetic summary.json (best_eval_loss={manual_best})")
    print("\n[done] recovery evals complete.")


if __name__ == "__main__":
    main()
