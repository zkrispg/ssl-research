"""Multi-seed queue for the strict DCASE 2023 SELDnet baseline reproduction.

Trains :class:`SeldNetOfficial` on STARSS23 dev with the same recipe used
for the ``no_geom`` and ``full`` cells: 30 epochs, 8 random crops per
clip per epoch, in-memory feature cache, optionally SpecAugment.

Output dirs are suffixed ``baseline_mc8_inmem`` (vanilla) or
``baseline_mc8_inmem_specaug`` (with augmentation). The script is
resume-aware: skips cells whose ``best.pt`` + ``summary.json`` +
``eval_threshold_sweep.json`` already exist; writes a partial
``multiseed_summary_seldnet_baseline.json`` after every cell.

Usage:
    # Plain SELDnet baseline (no SpecAug):
    python -m week11_starss23.run_seldnet_baseline_queue

    # With SpecAug (matched to the augmented cells of full / no_geom):
    python -m week11_starss23.run_seldnet_baseline_queue --specaug
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
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

REPO = Path("D:/ssl-research")
PYEXE = REPO / "venv" / "Scripts" / "python.exe"
RUNS_DIR = REPO / "week11_starss23" / "runs"
EVAL_THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)
TRAIN_KWARGS = dict(epochs=30, batch_size=32, train_crops_per_clip=8, in_memory=True)


@dataclass
class RunSpec:
    seed: int
    out_suffix: str
    specaug: bool

    @property
    def variant(self) -> str:
        return "seldnet_official"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / f"{self.variant}_seed{self.seed}_{self.out_suffix}"

    @property
    def best_ckpt(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def log_path(self) -> Path:
        tag = "specaug" if self.specaug else "vanilla"
        return RUNS_DIR / f"queue_seldnet_baseline_{tag}_seed{self.seed}.log"


def _train_one(spec: RunSpec) -> dict:
    cmd = [
        str(PYEXE), "-u", "-m", "week11_starss23.train_seld",
        "--variant", spec.variant,
        "--epochs", str(TRAIN_KWARGS["epochs"]),
        "--batch-size", str(TRAIN_KWARGS["batch_size"]),
        "--train-crops-per-clip", str(TRAIN_KWARGS["train_crops_per_clip"]),
        "--seed", str(spec.seed),
        "--out-suffix", spec.out_suffix,
    ]
    if TRAIN_KWARGS["in_memory"]:
        cmd.append("--in-memory")
    if spec.specaug:
        cmd.append("--specaug")

    print(
        f"\n{'='*72}\n[queue/seldnet] training {spec.variant} seed={spec.seed} "
        f"specaug={spec.specaug}\n  cmd: {' '.join(cmd)}\n{'='*72}",
        flush=True,
    )
    t0 = time.time()
    with spec.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.time() - t0
    summary_path = spec.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "spec": {"variant": spec.variant, "seed": spec.seed, "specaug": spec.specaug},
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "best_eval_loss": summary.get("best_eval_loss"),
        "n_params": summary.get("n_params"),
    }


def _eval_one(spec: RunSpec) -> dict:
    print(f"\n[queue/seldnet] eval seed={spec.seed} specaug={spec.specaug}", flush=True)
    if not spec.best_ckpt.exists():
        return {"error": f"best.pt missing for seed={spec.seed}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(spec.best_ckpt, device)
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
    for thr in EVAL_THRESHOLDS:
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
    out = {
        "spec": {"variant": spec.variant, "seed": spec.seed, "specaug": spec.specaug},
        "n_clips_eval": len(ds),
        "thresholds": per_threshold,
    }
    (spec.run_dir / "eval_threshold_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    del model, cached_preds, cached_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="seeds to run (default 0 1 2 = N=3 minimum for paired t-test)",
    )
    parser.add_argument(
        "--specaug", action="store_true",
        help="enable SpecAug for the baseline (matches ablated cells)",
    )
    args = parser.parse_args()

    out_suffix = "baseline_mc8_inmem_specaug" if args.specaug else "baseline_mc8_inmem"
    progress_file = (
        RUNS_DIR / f"multiseed_progress_seldnet_baseline_{'specaug' if args.specaug else 'vanilla'}.json"
    )
    summary_file = (
        RUNS_DIR / f"multiseed_summary_seldnet_baseline_{'specaug' if args.specaug else 'vanilla'}.json"
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[queue/seldnet] PYEXE={PYEXE}", flush=True)
    print(f"[queue/seldnet] cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"[queue/seldnet] seeds: {args.seeds}  specaug={args.specaug}", flush=True)
    print(f"[queue/seldnet] out_suffix: {out_suffix}", flush=True)

    progress = {
        "seeds": args.seeds,
        "specaug": args.specaug,
        "out_suffix": out_suffix,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": [],
        "current": None,
        "errors": [],
    }
    progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    all_trainings: list[dict] = []
    all_evals: list[dict] = []

    for seed in args.seeds:
        spec = RunSpec(seed=seed, out_suffix=out_suffix, specaug=args.specaug)
        ckpt_ok = spec.best_ckpt.exists()
        summary_ok = (spec.run_dir / "summary.json").exists()
        sweep_ok = (spec.run_dir / "eval_threshold_sweep.json").exists()
        if ckpt_ok and summary_ok and sweep_ok:
            print(f"\n[queue/seldnet] SKIP seed={seed} (artifacts complete)", flush=True)
            sj = json.loads((spec.run_dir / "summary.json").read_text(encoding="utf-8"))
            ej = json.loads((spec.run_dir / "eval_threshold_sweep.json").read_text(encoding="utf-8"))
            all_trainings.append({
                "spec": {"variant": spec.variant, "seed": seed, "specaug": args.specaug},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            })
            all_evals.append({
                "spec": {"variant": spec.variant, "seed": seed, "specaug": args.specaug},
                **ej,
            })
            progress["completed"].append({
                "seed": seed, "best_eval_loss": sj.get("best_eval_loss"),
                "elapsed_s": None, "resumed": True,
            })
            progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            continue

        if ckpt_ok and summary_ok:
            print(f"\n[queue/seldnet] TRAIN-SKIP seed={seed}", flush=True)
            sj = json.loads((spec.run_dir / "summary.json").read_text(encoding="utf-8"))
            train_summary = {
                "spec": {"variant": spec.variant, "seed": seed, "specaug": args.specaug},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            }
        else:
            progress["current"] = {"seed": seed, "stage": "training", "since": time.strftime("%H:%M:%S")}
            progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            train_summary = _train_one(spec)
            if train_summary["returncode"] != 0:
                progress["errors"].append({
                    "stage": "train", "seed": seed,
                    "returncode": train_summary["returncode"],
                })
        all_trainings.append(train_summary)

        progress["current"] = {"seed": seed, "stage": "eval", "since": time.strftime("%H:%M:%S")}
        progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        try:
            eval_summary = _eval_one(spec)
            all_evals.append(eval_summary)
        except Exception as exc:
            progress["errors"].append({
                "stage": "eval", "seed": seed, "exception": repr(exc),
            })

        progress["completed"].append({
            "seed": seed,
            "best_eval_loss": train_summary.get("best_eval_loss"),
            "elapsed_s": train_summary.get("elapsed_s"),
        })
        progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        # Persist intermediate summary after every cell
        partial = {
            "trainings": all_trainings,
            "evals": [e for e in all_evals if "spec" in e],
            "progress": progress,
            "status": "in_progress",
        }
        summary_file.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = {
        "trainings": all_trainings,
        "evals": [e for e in all_evals if "spec" in e],
        "progress": progress,
        "status": "complete",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_file.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[queue/seldnet] DONE. summary -> {summary_file}", flush=True)


if __name__ == "__main__":
    main()
