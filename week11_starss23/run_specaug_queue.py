"""SpecAug ablation queue: re-train all 10 multi-seed cells with SpecAugment.

Output dirs are suffixed ``mc8_inmem_specaug`` so they don't clobber the
vanilla N = 5 runs. Each cell: 30 epochs, 8 random crops/clip/epoch,
in-memory feature cache, ``--specaug`` flag enabled.

Resume-aware: skips cells whose ``best.pt`` + ``summary.json`` +
``eval_threshold_sweep.json`` already exist. Writes a partial
``multiseed_summary_specaug.json`` after every cell so the data is never
lost if the host process dies.
"""
from __future__ import annotations

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
PROGRESS_FILE = RUNS_DIR / "multiseed_progress_specaug.json"
SUMMARY_FILE = RUNS_DIR / "multiseed_summary_specaug.json"

QUEUE: list[tuple[str, int]] = [
    ("no_geom", s) for s in (0, 1, 2, 3, 4)
] + [
    ("full", s) for s in (0, 1, 2, 3, 4)
]
TRAIN_KWARGS = dict(epochs=30, batch_size=32, train_crops_per_clip=8, in_memory=True)
OUT_SUFFIX = "mc8_inmem_specaug"
EVAL_THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)


@dataclass
class RunSpec:
    variant: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / f"{self.variant}_seed{self.seed}_{OUT_SUFFIX}"

    @property
    def best_ckpt(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def log_path(self) -> Path:
        return RUNS_DIR / f"queue_specaug_{self.variant}_seed{self.seed}.log"


def _write_progress(payload: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_one(spec: RunSpec) -> dict:
    cmd = [
        str(PYEXE), "-u", "-m", "week11_starss23.train_seld",
        "--variant", spec.variant,
        "--epochs", str(TRAIN_KWARGS["epochs"]),
        "--batch-size", str(TRAIN_KWARGS["batch_size"]),
        "--train-crops-per-clip", str(TRAIN_KWARGS["train_crops_per_clip"]),
        "--seed", str(spec.seed),
        "--out-suffix", OUT_SUFFIX,
        "--specaug",
    ]
    if TRAIN_KWARGS["in_memory"]:
        cmd.append("--in-memory")

    print(f"\n{'='*72}\n[queue/specaug] training {spec.variant} seed={spec.seed}\n  cmd: {' '.join(cmd)}\n{'='*72}", flush=True)
    t0 = time.time()
    with spec.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, check=False,
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


def _eval_one(spec: RunSpec) -> dict:
    print(f"\n[queue/specaug] eval {spec.variant} seed={spec.seed}", flush=True)
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
    del model, cached_preds, cached_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[queue/specaug] PYEXE={PYEXE}", flush=True)
    print(f"[queue/specaug] cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"[queue/specaug] queue: {QUEUE}", flush=True)

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
        ckpt_ok = spec.best_ckpt.exists()
        summary_ok = (spec.run_dir / "summary.json").exists()
        sweep_ok = (spec.run_dir / "eval_threshold_sweep.json").exists()
        if ckpt_ok and summary_ok and sweep_ok:
            print(f"\n[queue/specaug] SKIP {variant} seed={seed} (artifacts complete)", flush=True)
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

        if ckpt_ok and summary_ok:
            print(f"\n[queue/specaug] TRAIN-SKIP {variant} seed={seed}", flush=True)
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
                "stage": "eval", "variant": variant, "seed": seed,
                "exception": repr(exc),
            })

        progress["completed"].append({
            "variant": variant, "seed": seed,
            "best_eval_loss": train_summary.get("best_eval_loss"),
            "elapsed_s": train_summary.get("elapsed_s"),
        })
        _write_progress(progress)

        # Persist intermediate summary after every cell
        partial = {
            "trainings": all_trainings,
            "evals": [e for e in all_evals if "spec" in e],
            "progress": progress,
            "status": "in_progress",
        }
        SUMMARY_FILE.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = {
        "trainings": all_trainings,
        "evals": [e for e in all_evals if "spec" in e],
        "progress": progress,
        "status": "complete",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SUMMARY_FILE.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[queue/specaug] DONE. summary -> {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
