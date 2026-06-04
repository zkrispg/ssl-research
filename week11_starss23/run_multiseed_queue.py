"""Sequential queue runner for STARSS23 multi-seed paired training.

Trains ``(variant, seed)`` cells in series so that there is never GPU
contention. After each training finishes, evaluates the saved ``best.pt``
on the dev-test split at a fixed operating threshold and writes a
per-cell ``eval_{thr}.json``. Once all four cells are done, aggregates
into a single ``multiseed_summary.json``.

Designed to be launched once and left running for ~4 h. Logs progress
both to stdout (line-buffered, ``-u``) and to a structured progress
file the parent shell can poll.

Usage:
    python -u -m week11_starss23.run_multiseed_queue
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
PROGRESS_FILE = RUNS_DIR / "multiseed_progress.json"
SUMMARY_FILE = RUNS_DIR / "multiseed_summary.json"

import argparse

DEFAULT_QUEUE: list[tuple[str, int]] = [
    ("no_geom", 1),
    ("full", 1),
    ("no_geom", 2),
    ("full", 2),
    ("no_geom", 3),
    ("full", 3),
    ("no_geom", 4),
    ("full", 4),
]
QUEUE: list[tuple[str, int]] = DEFAULT_QUEUE  # filled in main() if --seeds is passed
TRAIN_KWARGS = dict(epochs=30, batch_size=32, train_crops_per_clip=8, in_memory=True)
EVAL_THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)


@dataclass
class RunSpec:
    variant: str
    seed: int

    @property
    def out_suffix(self) -> str:
        return "mc8_inmem"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / f"{self.variant}_seed{self.seed}_{self.out_suffix}"

    @property
    def best_ckpt(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def log_path(self) -> Path:
        return RUNS_DIR / f"queue_{self.variant}_seed{self.seed}.log"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _write_progress(payload: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_one(spec: RunSpec) -> dict:
    """Run a single training as a blocking subprocess, return summary dict."""
    cmd = [
        str(PYEXE),
        "-u",
        "-m",
        "week11_starss23.train_seld",
        "--variant", spec.variant,
        "--epochs", str(TRAIN_KWARGS["epochs"]),
        "--batch-size", str(TRAIN_KWARGS["batch_size"]),
        "--train-crops-per-clip", str(TRAIN_KWARGS["train_crops_per_clip"]),
        "--seed", str(spec.seed),
        "--out-suffix", spec.out_suffix,
    ]
    if TRAIN_KWARGS["in_memory"]:
        cmd.append("--in-memory")

    print(f"\n{'='*72}\n[queue] training {spec.variant} seed={spec.seed}\n  cmd: {' '.join(cmd)}\n{'='*72}", flush=True)
    t0 = time.time()
    with spec.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - t0
    summary_path = spec.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "spec": {"variant": spec.variant, "seed": spec.seed},
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "best_eval_loss": summary.get("best_eval_loss"),
        "n_params": summary.get("n_params"),
    }


# ---------------------------------------------------------------------------
# Evaluation (in-process; reuses the just-trained model)
# ---------------------------------------------------------------------------


def _eval_one(spec: RunSpec) -> dict:
    """Run threshold-sweep DCASE evaluation on ``spec.best_ckpt``."""
    print(f"\n[queue] eval {spec.variant} seed={spec.seed}", flush=True)
    if not spec.best_ckpt.exists():
        return {"error": f"best.pt missing for {spec.variant} seed={spec.seed}"}

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
        "spec": {"variant": spec.variant, "seed": spec.seed},
        "n_clips_eval": len(ds),
        "thresholds": per_threshold,
    }
    (spec.run_dir / "eval_threshold_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    # Free GPU memory before next training
    del model, cached_preds, cached_targets
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _aggregate(all_evals: list[dict], all_trainings: list[dict]) -> dict:
    """Build per-(variant, seed) and per-(variant, threshold) tables for paper."""
    by_cell = {}
    for e in all_evals:
        s = e["spec"]
        by_cell[(s["variant"], s["seed"])] = e

    rows = []
    for variant, seed in QUEUE:
        cell = by_cell.get((variant, seed))
        train_cell = next(
            (t for t in all_trainings if t["spec"]["variant"] == variant
             and t["spec"]["seed"] == seed), None
        )
        if cell is None:
            continue
        for thr_str, m in cell["thresholds"].items():
            rows.append({
                "variant": variant,
                "seed": seed,
                "threshold": float(thr_str),
                "f1_macro": m["macro"]["f1"],
                "f1_micro": m["micro"]["f1"],
                "le_macro": m["macro"]["le_cd"],
                "le_micro": m["micro"]["le_cd"],
                "lr_macro": m["macro"]["lr_cd"],
                "lr_micro": m["micro"]["lr_cd"],
                "er_macro": m["macro"]["er"],
                "seld_micro": m["micro"]["seld"],
                "best_eval_loss": train_cell["best_eval_loss"] if train_cell else None,
                "elapsed_s": train_cell["elapsed_s"] if train_cell else None,
            })
    return {"trainings": all_trainings, "rows": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds to enqueue (default: 1 2). variants are no_geom + full.")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["no_geom", "full"])
    parser.add_argument("--summary-suffix", type=str, default="",
                        help="appended to output filenames so multiple queues coexist")
    args = parser.parse_args()
    global QUEUE, PROGRESS_FILE, SUMMARY_FILE
    if args.seeds is not None:
        QUEUE = [(v, s) for s in args.seeds for v in args.variants]
    if args.summary_suffix:
        PROGRESS_FILE = RUNS_DIR / f"multiseed_progress_{args.summary_suffix}.json"
        SUMMARY_FILE = RUNS_DIR / f"multiseed_summary_{args.summary_suffix}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[queue] PYEXE={PYEXE}", flush=True)
    print(f"[queue] cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"[queue] queue: {QUEUE}", flush=True)
    print(f"[queue] progress -> {PROGRESS_FILE}", flush=True)
    print(f"[queue] summary  -> {SUMMARY_FILE}", flush=True)

    progress = {
        "queue": QUEUE,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": [],
        "current": None,
        "errors": [],
    }
    _write_progress(progress)

    all_trainings: list[dict] = []
    all_evals: list[dict] = []

    for variant, seed in QUEUE:
        spec = RunSpec(variant=variant, seed=seed)

        # ---- Resume: skip cells whose artifacts already exist -------------
        ckpt_ok = spec.best_ckpt.exists()
        summary_ok = (spec.run_dir / "summary.json").exists()
        sweep_ok = (spec.run_dir / "eval_threshold_sweep.json").exists()
        if ckpt_ok and summary_ok and sweep_ok:
            print(f"\n[queue] SKIP {variant} seed={seed} (artifacts already complete)", flush=True)
            with (spec.run_dir / "summary.json").open("r", encoding="utf-8") as f:
                sj = json.load(f)
            with (spec.run_dir / "eval_threshold_sweep.json").open("r", encoding="utf-8") as f:
                ej = json.load(f)
            all_trainings.append({
                "spec": {"variant": variant, "seed": seed},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            })
            all_evals.append({"spec": {"variant": variant, "seed": seed}, **ej})
            progress["completed"].append({
                "variant": variant, "seed": seed,
                "best_eval_loss": sj.get("best_eval_loss"),
                "elapsed_s": None,
                "resumed": True,
            })
            _write_progress(progress)
            continue

        # ---- Train (skip if best.pt + summary.json already exist) ---------
        if ckpt_ok and summary_ok:
            print(f"\n[queue] TRAIN-SKIP {variant} seed={seed} (best.pt exists)", flush=True)
            with (spec.run_dir / "summary.json").open("r", encoding="utf-8") as f:
                sj = json.load(f)
            train_summary = {
                "spec": {"variant": variant, "seed": seed},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            }
        else:
            progress["current"] = {
                "variant": variant, "seed": seed,
                "stage": "training", "since": time.strftime("%H:%M:%S"),
            }
            _write_progress(progress)
            train_summary = _train_one(spec)
            if train_summary["returncode"] != 0:
                progress["errors"].append({
                    "stage": "train", **train_summary["spec"],
                    "returncode": train_summary["returncode"],
                })
        all_trainings.append(train_summary)

        progress["current"] = {
            "variant": variant, "seed": seed,
            "stage": "eval", "since": time.strftime("%H:%M:%S"),
        }
        _write_progress(progress)
        try:
            eval_summary = _eval_one(spec)
            all_evals.append(eval_summary)
        except Exception as exc:
            progress["errors"].append({
                "stage": "eval", **{"variant": variant, "seed": seed},
                "exception": repr(exc),
            })

        progress["completed"].append({
            "variant": variant, "seed": seed,
            "best_eval_loss": train_summary.get("best_eval_loss"),
            "elapsed_s": train_summary.get("elapsed_s"),
        })
        _write_progress(progress)

        # Persist intermediate summary after every cell so we never lose
        # progress if the runner dies again.
        partial = _aggregate(all_evals, all_trainings)
        partial["progress"] = progress
        partial["status"] = "in_progress"
        SUMMARY_FILE.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = _aggregate(all_evals, all_trainings)
    final["progress"] = progress
    final["status"] = "complete"
    final["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SUMMARY_FILE.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[queue] DONE. summary -> {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
