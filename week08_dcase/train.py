"""Train MultiAccdoaCRNN with ADPIT loss + auxiliary count cross-entropy.

Reuses the W5/W6 multi-source dataset (feat, soft-label, az_padded)
because the simulator and feature pipeline are unchanged; we just
discard the soft-label and use ``az_padded`` directly for ADPIT
supervision. Validation uses DCASE-style metrics implemented in
:mod:`dcase_metrics`, with peak picking driven by the auxiliary count
head -- we take exactly the ``K_pred`` highest-magnitude active tracks
after NMS.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))
sys.path.insert(0, str(Path(__file__).parent.parent / "week05_multi_source"))

from dcase_metrics import DcaseSeldStats, format_summary, overall_seld_score  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_accdoa_model import MultiAccdoaCRNN, count_parameters  # noqa: E402
from multi_accdoa import (  # noqa: E402
    adpit_loss_batch,
    decode_multi_accdoa,
    make_gt_xy,
)
from multi_dataset import MultiSourceConfig, precompute_multi_source_dataset  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    train_samples: int = 5000
    val_samples: int = 800
    duration: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_count: float = 1.0
    activity_threshold: float = 0.5
    nms_tol_deg: float = 25.0
    tolerance_deg: float = 20.0
    rt60_range: tuple[float, float] | None = (0.15, 0.5)
    reverb_prob: float = 0.5
    snr_range_db: tuple[float, float] = (-5.0, 30.0)
    out_dir: Path = Path(__file__).parent / "checkpoints"
    plot_path: Path = Path(__file__).parent / "training.png"
    resume_from: Path | None = None
    resume_lr_scale: float = 0.3


def evaluate_dcase(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    ce: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_adpit = 0.0
    total_ce = 0.0
    n_seen = 0
    head_correct = 0
    stats = DcaseSeldStats(tolerance_deg=cfg.tolerance_deg)
    with torch.no_grad():
        for x, _label, az_padded in loader:
            x = x.to(device)
            az_padded = az_padded.to(device)
            gt_xy = make_gt_xy(az_padded)
            k_target = (~torch.isnan(az_padded[:, :, 0]) if az_padded.dim() == 3
                        else ~torch.isnan(az_padded)).sum(dim=-1) - 1

            out = model(x)
            loss_adpit = adpit_loss_batch(out["accdoa"], gt_xy)
            loss_ce = ce(out["count"], k_target)
            total_adpit += float(loss_adpit.item()) * x.size(0)
            total_ce += float(loss_ce.item()) * x.size(0)
            n_seen += x.size(0)

            pred_k = out["count"].argmax(dim=-1).cpu().numpy() + 1
            head_correct += int(((pred_k - 1) == k_target.cpu().numpy()).sum())
            decoded = decode_multi_accdoa(
                out["accdoa"],
                activity_threshold=cfg.activity_threshold,
                nms_tol_deg=cfg.nms_tol_deg,
            )
            # Trim each prediction to the count-head's predicted K.
            for b in range(x.size(0)):
                preds = decoded[b]
                if len(preds) > pred_k[b]:
                    preds = preds[: pred_k[b]]
                az = az_padded[b].cpu().numpy()
                gts = az[~np.isnan(az)]
                stats.add_sample(preds, gts)

    summary = stats.summary()
    summary["loss_adpit"] = total_adpit / max(n_seen, 1)
    summary["loss_count"] = total_ce / max(n_seen, 1)
    summary["seld_score"] = overall_seld_score(summary)
    summary["head_count_acc"] = head_correct / max(n_seen, 1)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from. "
                        "Loads weights, then runs --epochs more epochs at "
                        "reduced LR (resume_lr_scale).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override TrainConfig.epochs.")
    parser.add_argument("--lr-scale", type=float, default=None,
                        help="Multiplier on base lr when resuming.")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for checkpoint and plot filenames.")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = TrainConfig()
    if args.resume:
        cfg.resume_from = Path(args.resume)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr_scale is not None:
        cfg.resume_lr_scale = args.lr_scale
    if args.out_suffix:
        cfg.plot_path = cfg.plot_path.with_name(
            f"{cfg.plot_path.stem}_{args.out_suffix}{cfg.plot_path.suffix}"
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    print(f"[train] target metrics: DCASE SELD (F1, ER, LE_CD, LR_CD, SELD)", flush=True)

    train_ds_cfg = MultiSourceConfig(
        mic_positions=mics,
        n_samples=cfg.train_samples,
        duration=cfg.duration,
        rt60_range=cfg.rt60_range,
        reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db,
        seed_base=0,
    )
    val_ds_cfg = MultiSourceConfig(
        mic_positions=mics,
        n_samples=cfg.val_samples,
        duration=cfg.duration,
        rt60_range=cfg.rt60_range,
        reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db,
        seed_base=10_000_000,
    )
    train_ds = precompute_multi_source_dataset(train_ds_cfg, desc="precompute train")
    val_ds = precompute_multi_source_dataset(val_ds_cfg, desc="precompute val  ")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    model = MultiAccdoaCRNN(n_mics=4, n_freq=257, n_tracks=3, max_k=3).to(device)
    print(f"[train] model parameters: {count_parameters(model):,}", flush=True)

    base_lr = cfg.lr
    best_seld = float("inf")
    if cfg.resume_from is not None and cfg.resume_from.exists():
        ckpt_resume = torch.load(cfg.resume_from, map_location=device,
                                 weights_only=False)
        model.load_state_dict(ckpt_resume["model_state"])
        prev_seld = ckpt_resume.get("seld_score", float("inf"))
        if isinstance(prev_seld, (int, float)) and prev_seld == prev_seld:
            best_seld = float(prev_seld)
        base_lr = cfg.lr * cfg.resume_lr_scale
        print(f"[train] resumed from {cfg.resume_from}  "
              f"(prev SELD={best_seld:.3f}, lr={base_lr:.2e})", flush=True)

    ce = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history: dict[str, list[float]] = {
        "train_total": [], "val_F1": [], "val_ER": [],
        "val_LE": [], "val_LR": [], "val_SELD": [], "val_head_acc": [],
    }
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    best_path = cfg.out_dir / f"best{suffix}.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False, mininterval=1.0)
        for x, _label, az_padded in pbar:
            x = x.to(device)
            az_padded = az_padded.to(device)
            gt_xy = make_gt_xy(az_padded)
            k_target = (~torch.isnan(az_padded)).sum(dim=-1) - 1

            out = model(x)
            loss_adpit = adpit_loss_batch(out["accdoa"], gt_xy)
            loss_count = ce(out["count"], k_target)
            loss = loss_adpit + cfg.lambda_count * loss_count

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"adpit": f"{loss_adpit.item():.4f}",
                              "count": f"{loss_count.item():.4f}"})
        train_loss = total / max(n_seen, 1)

        val = evaluate_dcase(model, val_loader, cfg, device, ce)
        sched.step()

        history["train_total"].append(train_loss)
        history["val_F1"].append(val["F1"])
        history["val_ER"].append(val["ER"])
        history["val_LE"].append(val["LE_CD"] if np.isfinite(val["LE_CD"]) else 180.0)
        history["val_LR"].append(val["LR_CD"])
        history["val_SELD"].append(val["seld_score"] if np.isfinite(val["seld_score"]) else 1.0)
        history["val_head_acc"].append(val["head_count_acc"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{cfg.epochs}] "
            f"train={train_loss:.4f}  "
            + format_summary("val", val)
            + f"  SELD={val['seld_score']:.3f}  head_acc={val['head_count_acc']:.3f}  "
            + f"({elapsed:.1f}s)",
            flush=True,
        )

        if val["seld_score"] < best_seld:
            best_seld = val["seld_score"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mics": 4, "n_freq": 257, "n_tracks": 3, "max_k": 3,
                    "epoch": epoch, "val_summary": val, "seld_score": best_seld,
                },
                best_path,
            )
            print(f"  -> saved {best_path} (SELD {best_seld:.3f})", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(history["train_total"], label="train_total")
    axes[0, 0].set_title("Training loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(history["val_F1"], label="F1", color="C2")
    axes[0, 1].plot(history["val_LR"], label="LR", color="C0")
    axes[0, 1].set_title("Validation F1 / LR"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].plot(history["val_LE"], color="C3")
    axes[1, 0].set_title("Validation LE_CD (deg)"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(history["val_SELD"], color="C4", label="SELD score (lower=better)")
    axes[1, 1].plot(history["val_head_acc"], color="C5", label="count head acc")
    axes[1, 1].set_title("SELD score / count acc"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved {cfg.plot_path}", flush=True)
    print(f"[train] best SELD = {best_seld:.3f}, checkpoint -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
