"""Train the multi-source CRNN with BCE on the spatial pseudo-spectrum.

Loss: per-frame BCE between the model's sigmoid output and the soft
multi-hot label. The static-source label is broadcast across time. We
weight class-positive bins (active source) higher than negatives via
``pos_weight`` to compensate for the strong class imbalance (~3 / 72
positive bins per sample).

Validation tracks F1 at a 20-degree tolerance after peak-picking, which
matches the metric reported in the W5 evaluation table.
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

from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import find_peaks_circular  # noqa: E402
from multi_dataset import (  # noqa: E402
    MultiSourceConfig,
    make_grid,
    precompute_multi_source_dataset,
)
from multi_eval import LocalizationStats  # noqa: E402
from multi_model import MultiSourceCRNN, count_parameters  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    train_samples: int = 2000
    val_samples: int = 400
    duration: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 12.0  # ~ (72 - 3) / 3 to balance the BCE
    tolerance_deg: float = 20.0
    rt60_range: tuple[float, float] | None = (0.15, 0.5)
    reverb_prob: float = 0.5
    snr_range_db: tuple[float, float] = (-5.0, 30.0)
    out_dir: Path = Path(__file__).parent / "checkpoints"
    plot_path: Path = Path(__file__).parent / "training.png"


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    grid: np.ndarray,
    device: torch.device,
    criterion: nn.Module,
    tolerance_deg: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    stats = LocalizationStats()
    with torch.no_grad():
        for x, y, az_padded in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # (B, T, C)
            T = logits.shape[1]
            target = y.unsqueeze(1).expand(-1, T, -1)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

            probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()  # (B, C)
            for b in range(probs.shape[0]):
                preds = find_peaks_circular(
                    probs[b], grid, n_peaks=None, rel_threshold=0.5,
                    min_separation_deg=25.0,
                )
                gts = az_padded[b].numpy()
                gts = gts[~np.isnan(gts)]
                stats.add_sample(preds, gts, tolerance_deg=tolerance_deg)

    summary = stats.summary()
    summary["loss"] = total_loss / max(n_seen, 1)
    return summary


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = TrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    print("[train] mic positions:", flush=True)
    print(mics, flush=True)
    grid = make_grid(-180, 180, 5)

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

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=len(grid)).to(device)
    print(f"[train] model parameters: {count_parameters(model):,}", flush=True)

    pos_weight = torch.full((len(grid),), cfg.pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_mae": [], "val_count_acc": []}
    best_f1 = -1.0
    best_path = cfg.out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False, mininterval=1.0)
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # (B, T, C)
            T = logits.shape[1]
            target = y.unsqueeze(1).expand(-1, T, -1)
            loss = criterion(logits, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = total / max(n_seen, 1)

        val = evaluate(model, val_loader, grid, device, criterion, cfg.tolerance_deg)
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_f1"].append(val["f1"])
        history["val_mae"].append(val["mae_tp_deg"])
        history["val_count_acc"].append(val["count_acc"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
            f"val_F1={val['f1']:.3f}  val_P={val['precision']:.3f}  "
            f"val_R={val['recall']:.3f}  val_MAE={val['mae_tp_deg']:.2f}  "
            f"count_acc={val['count_acc']:.3f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mics": 4,
                    "n_freq": 257,
                    "n_classes": len(grid),
                    "epoch": epoch,
                    "val_f1": val["f1"],
                    "val_mae": val["mae_tp_deg"],
                },
                best_path,
            )
            print(f"  -> saved {best_path} (F1 {best_f1:.3f})", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_f1"], color="C2", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation F1 (tol=20 deg)")
    axes[1].set_title("F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(history["val_mae"], color="C0", marker="s", label="MAE (TP)")
    ax_count = axes[2].twinx()
    ax_count.plot(history["val_count_acc"], color="C3", marker="^", linestyle="--", label="count acc")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MAE on TP (deg)", color="C0")
    ax_count.set_ylabel("count accuracy", color="C3")
    axes[2].set_title("Localization quality")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved {cfg.plot_path}", flush=True)
    print(f"[train] best val F1 = {best_f1:.3f}, checkpoint -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
