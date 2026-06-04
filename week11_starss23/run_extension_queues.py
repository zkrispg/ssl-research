"""All Tier 1 + Tier 2 extension queues for the ICASSP paper, in one file.

Each queue is a function ``run_<name>()`` that trains its cells, runs the
DCASE threshold sweep evaluation, writes a partial summary after every
cell, and returns when the queue's summary JSON is marked complete. All
queues are resume-aware: they skip cells whose ``best.pt`` +
``summary.json`` + ``eval_threshold_sweep.json`` already exist.

Queues:
    G1: ``seldnet_n5_extension``
        Adds SELDnet baseline seeds 3, 4 (vanilla and +SpecAug) to bring
        the SELDnet comparison to N = 5 paired (matches the geometry
        ablation).
    G2: ``weak_specaug``
        no_geom and full at N = 5 with WEAK SpecAug (mask widths halved).
        Verifies that the catastrophic SpecAug result is not
        strength-cherry-picked.
    G3: ``capacity_sweep``
        no_geom and full at three new capacity points (xs / l / xl), each
        N = 3 seeds. Combined with the existing N = 5 medium runs this
        gives a 4-point capacity curve.

Each function is invoked via ``run_extension_queues.py --queue G1`` or
``--queue all`` for the supervisor.
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
RUNS = REPO / "week11_starss23" / "runs"
EVAL_THRESHOLDS = (0.10, 0.15, 0.18, 0.22, 0.30)


@dataclass
class CellSpec:
    variant: str
    seed: int
    out_suffix: str
    extra_train_args: tuple[str, ...] = ()  # e.g. ("--specaug", "--specaug-strength", "weak")
    epochs: int = 30
    batch_size: int = 32
    crops_per_clip: int = 8
    in_memory: bool = True

    @property
    def run_dir(self) -> Path:
        return RUNS / f"{self.variant}_seed{self.seed}_{self.out_suffix}"

    @property
    def best_ckpt(self) -> Path:
        return self.run_dir / "best.pt"


# ----------------------------------------------------------------------------
# Generic train/eval helpers (shared across queues)
# ----------------------------------------------------------------------------

def _train_one(spec: CellSpec, queue_tag: str) -> dict:
    cmd = [
        str(PYEXE), "-u", "-m", "week11_starss23.train_seld",
        "--variant", spec.variant,
        "--epochs", str(spec.epochs),
        "--batch-size", str(spec.batch_size),
        "--train-crops-per-clip", str(spec.crops_per_clip),
        "--seed", str(spec.seed),
        "--out-suffix", spec.out_suffix,
    ]
    if spec.in_memory:
        cmd.append("--in-memory")
    cmd.extend(spec.extra_train_args)

    log_path = RUNS / f"queue_{queue_tag}_{spec.variant}_seed{spec.seed}.log"
    print(f"\n{'='*72}\n[{queue_tag}] training {spec.variant} seed={spec.seed} "
          f"suffix={spec.out_suffix}\n  cmd: {' '.join(cmd)}\n{'='*72}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.time() - t0
    summary_path = spec.run_dir / "summary.json"
    sj = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "spec": {"variant": spec.variant, "seed": spec.seed, "suffix": spec.out_suffix},
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "best_eval_loss": sj.get("best_eval_loss"),
        "n_params": sj.get("n_params"),
    }


def _eval_one(spec: CellSpec, queue_tag: str) -> dict:
    print(f"\n[{queue_tag}] eval {spec.variant} seed={spec.seed}", flush=True)
    if not spec.best_ckpt.exists():
        return {"error": f"best.pt missing for {spec.variant} seed={spec.seed}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(spec.best_ckpt, device)
    n_classes = blob["model_cfg"]["n_classes"]
    # Pick the right audio + cache directory based on the feature array type
    # stored in the checkpoint (mic vs FOA share the metadata folder).
    if getattr(feat_cfg, "array_type", "mic") == "foa":
        audio_dir = Path("D:/ssl-research/data/STARSS23/foa_dev")
        cache_dir = Path("D:/ssl-research/data/STARSS23/foa_feat_cache")
    else:
        audio_dir = Path("D:/ssl-research/data/STARSS23/mic_dev")
        cache_dir = Path("D:/ssl-research/data/STARSS23/feat_cache")
    ds = StarssDataset(
        audio_dir=audio_dir,
        metadata_dir=Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev"),
        split="test",
        feature_config=feat_cfg,
        clip_seconds=None,
        random_crop=False,
        cache_dir=cache_dir,
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
        "spec": {"variant": spec.variant, "seed": spec.seed, "suffix": spec.out_suffix},
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


def _run_queue(specs: list[CellSpec], queue_tag: str) -> None:
    """Generic runner: skip-complete-resume + per-cell partial-summary save."""
    progress_path = RUNS / f"multiseed_progress_{queue_tag}.json"
    summary_path = RUNS / f"multiseed_summary_{queue_tag}.json"

    progress = {
        "tag": queue_tag,
        "specs": [{"variant": s.variant, "seed": s.seed, "suffix": s.out_suffix} for s in specs],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": [],
        "current": None,
        "errors": [],
    }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    all_trainings: list[dict] = []
    all_evals: list[dict] = []

    for spec in specs:
        ckpt_ok = spec.best_ckpt.exists()
        summary_ok = (spec.run_dir / "summary.json").exists()
        sweep_ok = (spec.run_dir / "eval_threshold_sweep.json").exists()
        if ckpt_ok and summary_ok and sweep_ok:
            print(f"\n[{queue_tag}] SKIP {spec.variant} seed={spec.seed}", flush=True)
            sj = json.loads((spec.run_dir / "summary.json").read_text(encoding="utf-8"))
            ej = json.loads((spec.run_dir / "eval_threshold_sweep.json").read_text(encoding="utf-8"))
            all_trainings.append({
                "spec": {"variant": spec.variant, "seed": spec.seed, "suffix": spec.out_suffix},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            })
            all_evals.append({
                "spec": {"variant": spec.variant, "seed": spec.seed, "suffix": spec.out_suffix},
                **ej,
            })
            progress["completed"].append({
                "variant": spec.variant, "seed": spec.seed,
                "best_eval_loss": sj.get("best_eval_loss"),
                "elapsed_s": None, "resumed": True,
            })
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            continue

        if ckpt_ok and summary_ok:
            print(f"\n[{queue_tag}] TRAIN-SKIP {spec.variant} seed={spec.seed}", flush=True)
            sj = json.loads((spec.run_dir / "summary.json").read_text(encoding="utf-8"))
            train_summary = {
                "spec": {"variant": spec.variant, "seed": spec.seed, "suffix": spec.out_suffix},
                "returncode": 0, "elapsed_s": None,
                "best_eval_loss": sj.get("best_eval_loss"),
                "n_params": sj.get("n_params"),
            }
        else:
            progress["current"] = {
                "variant": spec.variant, "seed": spec.seed,
                "stage": "training", "since": time.strftime("%H:%M:%S"),
            }
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            train_summary = _train_one(spec, queue_tag)
            if train_summary["returncode"] != 0:
                progress["errors"].append({
                    "stage": "train", "variant": spec.variant, "seed": spec.seed,
                    "returncode": train_summary["returncode"],
                })
        all_trainings.append(train_summary)

        progress["current"] = {
            "variant": spec.variant, "seed": spec.seed,
            "stage": "eval", "since": time.strftime("%H:%M:%S"),
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        try:
            eval_summary = _eval_one(spec, queue_tag)
            all_evals.append(eval_summary)
        except Exception as exc:
            progress["errors"].append({
                "stage": "eval", "variant": spec.variant, "seed": spec.seed,
                "exception": repr(exc),
            })

        progress["completed"].append({
            "variant": spec.variant, "seed": spec.seed,
            "best_eval_loss": train_summary.get("best_eval_loss"),
            "elapsed_s": train_summary.get("elapsed_s"),
        })
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        partial = {
            "trainings": all_trainings,
            "evals": [e for e in all_evals if "spec" in e],
            "progress": progress,
            "status": "in_progress",
        }
        summary_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = {
        "trainings": all_trainings,
        "evals": [e for e in all_evals if "spec" in e],
        "progress": progress,
        "status": "complete",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[{queue_tag}] DONE. summary -> {summary_path}", flush=True)


# ----------------------------------------------------------------------------
# Queue G1: SELDnet baseline N = 5 extension (seeds 3, 4)
# ----------------------------------------------------------------------------

def queue_g1_seldnet_n5_extension() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for seed in (3, 4):
        specs.append(CellSpec(
            variant="seldnet_official", seed=seed,
            out_suffix="baseline_mc8_inmem",
        ))
    for seed in (3, 4):
        specs.append(CellSpec(
            variant="seldnet_official", seed=seed,
            out_suffix="baseline_mc8_inmem_specaug",
            extra_train_args=("--specaug",),  # strong by default
        ))
    return specs


# ----------------------------------------------------------------------------
# Queue G2: Weak SpecAug control (10 cells, no_geom + full, seeds 0..4)
# ----------------------------------------------------------------------------

def queue_g2_weak_specaug() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for variant in ("no_geom", "full"):
        for seed in range(5):
            specs.append(CellSpec(
                variant=variant, seed=seed,
                out_suffix="mc8_inmem_specaug_weak",
                extra_train_args=("--specaug", "--specaug-strength", "weak"),
            ))
    return specs


# ----------------------------------------------------------------------------
# Queue G3: Capacity sweep (3 sizes x 2 variants x 3 seeds = 18 cells)
# ----------------------------------------------------------------------------

def queue_g3_capacity_sweep() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for size_tag in ("xs", "l", "xl"):
        for variant_base in ("no_geom", "full"):
            variant = f"{variant_base}_{size_tag}"
            for seed in (0, 1, 2):
                specs.append(CellSpec(
                    variant=variant, seed=seed,
                    out_suffix=f"cap_{size_tag}_mc8_inmem",
                ))
    return specs


# ----------------------------------------------------------------------------
# Queue G5: Capacity sweep N=5 extension (seeds 3, 4 for all 6 capacity points)
# ----------------------------------------------------------------------------
# Path B Phase A1: lifts each capacity cell from N=3 to N=5 to address the
# low-power critique on the geometry-by-capacity ablation.

def queue_g5_capacity_sweep_n5_extension() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for size_tag in ("xs", "l", "xl"):
        for variant_base in ("no_geom", "full"):
            variant = f"{variant_base}_{size_tag}"
            for seed in (3, 4):
                specs.append(CellSpec(
                    variant=variant, seed=seed,
                    out_suffix=f"cap_{size_tag}_mc8_inmem",
                ))
    return specs


# ----------------------------------------------------------------------------
# Queue G6: FOA SELDnet (Path B / B2 -- format ablation on the canonical baseline)
# ----------------------------------------------------------------------------
# Five seeds of the strict DCASE 2023 SELDnet baseline trained on the FOA
# (Ambisonics) version of STARSS23 dev-train. Pairs seed-by-seed with the
# existing MIC seldnet_official runs to test whether the array format
# significantly changes performance.

def queue_g6_seldnet_foa() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for seed in range(5):
        specs.append(CellSpec(
            variant="seldnet_official", seed=seed,
            out_suffix="foa_baseline_mc8_inmem",
            extra_train_args=("--array-type", "foa"),
        ))
    return specs


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

QUEUE_REGISTRY = {
    "G1": ("seldnet_n5_extension", queue_g1_seldnet_n5_extension),
    "G2": ("weak_specaug", queue_g2_weak_specaug),
    "G3": ("capacity_sweep", queue_g3_capacity_sweep),
    "G5": ("capacity_sweep_n5_extension", queue_g5_capacity_sweep_n5_extension),
    "G6": ("seldnet_foa", queue_g6_seldnet_foa),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, choices=list(QUEUE_REGISTRY) + ["all"])
    args = parser.parse_args()

    queues = list(QUEUE_REGISTRY) if args.queue == "all" else [args.queue]
    for q in queues:
        tag, factory = QUEUE_REGISTRY[q]
        specs = factory()
        print(f"\n[run_extension_queues] launching queue {q} = {tag} "
              f"({len(specs)} cells)\n", flush=True)
        _run_queue(specs, queue_tag=tag)


if __name__ == "__main__":
    main()
