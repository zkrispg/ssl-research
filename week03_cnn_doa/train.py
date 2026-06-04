"""Train the phase-map CNN on synthetic free-field + reverb data.

CPU-friendly defaults: ~40K-parameter model, batch size 64, 200 batches per
epoch, 12 epochs. Total wall time on a recent x86 laptop is ~15-20 minutes.

Outputs:
    week03_cnn_doa/checkpoints/best.pt    -- best validation checkpoint
    week03_cnn_doa/training.png           -- loss + accuracy curves
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from dataset import (  # noqa: E402
    DatasetConfig,
    azimuth_classes,
    precompute_phase_dataset,
)
from geometry import uniform_circular_array  # noqa: E402
from model import PhaseMapCNN, count_parameters  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    train_samples: int = 8000
    val_samples: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Free-field only. Adding reverb to a 40k-parameter single-frame CNN with
    # only ~2k samples does not help on CPU; the data:parameter ratio is too
    # low and the multipath signal cannot be resolved from a single frame.
    # We instead document the limitation and address it in W4 with a
    # multi-frame CRNN.
    rt60_range: tuple[float, float] | None = None
    snr_range_db: tuple[float, float] = (-5.0, 30.0)
    n_workers: int = 0
    out_dir: Path = Path(__file__).parent / "checkpoints"
    plot_path: Path = Path(__file__).parent / "training.png"


def angular_error_deg(pred_classes: np.ndarray, true_classes: np.ndarray, grid: np.ndarray) -> np.ndarray:
    pred_az = grid[pred_classes]
    true_az = grid[true_classes]
    diff = ((pred_az - true_az + 180.0) % 360.0) - 180.0
    return np.abs(diff)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    grid: np.ndarray,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds_np = np.concatenate(preds)
    targets_np = np.concatenate(targets)
    err = angular_error_deg(preds_np, targets_np, grid)
    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(preds_np == targets_np)),
        "mae_deg": float(np.mean(err)),
        "median_err_deg": float(np.median(err)),
    }


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    train_cfg = TrainConfig()
    train_cfg.out_dir.mkdir(parents=True, exist_ok=True)

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    train_ds_cfg = DatasetConfig(
        mic_positions=mics,
        n_samples=train_cfg.train_samples,
        rt60_range=train_cfg.rt60_range,
        snr_range_db=train_cfg.snr_range_db,
        seed_base=0,
    )
    val_ds_cfg = DatasetConfig(
        mic_positions=mics,
        n_samples=train_cfg.val_samples,
        rt60_range=train_cfg.rt60_range,
        snr_range_db=train_cfg.snr_range_db,
        seed_base=10_000_000,
    )

    grid = azimuth_classes(*train_ds_cfg.azimuth_grid_deg)
    print(f"[train] mic positions:\n{mics}")
    print(f"[train] azimuth grid: {len(grid)} classes, {grid[0]:.0f}..{grid[-1]:.0f} deg")

    train_ds = precompute_phase_dataset(train_ds_cfg, desc="precompute train")
    val_ds = precompute_phase_dataset(val_ds_cfg, desc="precompute val  ")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.n_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.n_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    model = PhaseMapCNN(
        n_mics=4,
        n_freq=train_ds_cfg.n_fft // 2 + 1,
        n_classes=len(grid),
    ).to(device)
    print(f"[train] model parameters: {count_parameters(model):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}
    best_mae = float("inf")
    best_path = train_cfg.out_dir / "best.pt"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, grid, device, criterion)
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_mae"].append(val_metrics["mae_deg"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{train_cfg.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.3f}  val_mae={val_metrics['mae_deg']:.2f} deg  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["mae_deg"] < best_mae:
            best_mae = val_metrics["mae_deg"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "grid": grid.tolist(),
                    "n_mics": 4,
                    "n_freq": train_ds_cfg.n_fft // 2 + 1,
                    "n_classes": len(grid),
                    "epoch": epoch,
                    "val_mae": val_metrics["mae_deg"],
                },
                best_path,
            )
            print(f"  -> saved {best_path} (val MAE {best_mae:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history["val_mae"], color="C2", marker="o")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation MAE (deg)", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_title("Validation angular error")
    ax2.grid(True, alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(history["val_acc"], color="C3", marker="s", linestyle="--")
    ax3.set_ylabel("validation accuracy", color="C3")
    ax3.tick_params(axis="y", labelcolor="C3")
    fig.tight_layout()
    fig.savefig(train_cfg.plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved curves -> {train_cfg.plot_path}")
    print(f"[train] best val MAE = {best_mae:.2f} deg, checkpoint -> {best_path}")


if __name__ == "__main__":
    main()
