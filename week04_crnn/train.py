"""Train the CRNN with ACCDOA-style azimuth regression.

CPU-friendly: ~70K-parameter model, 2000 training samples, 20 epochs.
Total wall time on a recent x86 laptop is ~15-20 minutes. The bottleneck
is pyroomacoustics during the one-time precompute phase.
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

from crnn_dataset import (  # noqa: E402
    MultiFrameConfig,
    az_to_xy,
    precompute_multi_frame_dataset,
    xy_to_az_deg,
)
from crnn_model import CRNNDoa, count_parameters  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    train_samples: int = 1500
    val_samples: int = 300
    duration: float = 0.5  # halves T to ~30 frames -> ~2x faster per batch
    lr: float = 1e-3
    weight_decay: float = 1e-4
    rt60_range: tuple[float, float] | None = (0.15, 0.5)
    reverb_prob: float = 0.5
    snr_range_db: tuple[float, float] = (-5.0, 30.0)
    out_dir: Path = Path(__file__).parent / "checkpoints"
    plot_path: Path = Path(__file__).parent / "training.png"


def angular_error_deg(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Per-sample angular error in degrees, expecting (..., 2) tensors."""
    pred_az = xy_to_az_deg(pred_xy)
    targ_az = xy_to_az_deg(target_xy)
    diff = ((pred_az - targ_az + 180.0) % 360.0) - 180.0
    return diff.abs()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    errs = []
    with torch.no_grad():
        for x, az_rad in loader:
            x = x.to(device)
            az_rad = az_rad.to(device)
            pred = model(x)  # (B, T, 2)
            T = pred.shape[1]
            target_xy = az_to_xy(az_rad).unsqueeze(1).expand(-1, T, -1)
            loss = torch.mean((pred - target_xy) ** 2)
            total_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            # Aggregate per-frame predictions to one azimuth estimate per sample.
            mean_xy = pred.mean(dim=1)  # (B, 2)
            target_xy_per = az_to_xy(az_rad)
            err_deg = angular_error_deg(mean_xy, target_xy_per)
            errs.append(err_deg.cpu().numpy())
    err_np = np.concatenate(errs)
    return {
        "loss": total_loss / max(n_seen, 1),
        "mae_deg": float(np.mean(err_np)),
        "median_deg": float(np.median(err_np)),
        "p90_deg": float(np.percentile(err_np, 90)),
    }


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = TrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    print("[train] mic positions:")
    print(mics)

    train_ds_cfg = MultiFrameConfig(
        mic_positions=mics,
        n_samples=cfg.train_samples,
        duration=cfg.duration,
        rt60_range=cfg.rt60_range,
        reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db,
        seed_base=0,
    )
    val_ds_cfg = MultiFrameConfig(
        mic_positions=mics,
        n_samples=cfg.val_samples,
        duration=cfg.duration,
        rt60_range=cfg.rt60_range,
        reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db,
        seed_base=10_000_000,
    )

    train_ds = precompute_multi_frame_dataset(train_ds_cfg, desc="precompute train")
    val_ds = precompute_multi_frame_dataset(val_ds_cfg, desc="precompute val  ")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    model = CRNNDoa(n_mics=4, n_freq=257).to(device)
    print(f"[train] model parameters: {count_parameters(model):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_p90": []}
    best_mae = float("inf")
    best_path = cfg.out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n_seen = 0
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch:02d}",
            leave=False,
            mininterval=1.0,
        )
        for x, az_rad in pbar:
            x = x.to(device)
            az_rad = az_rad.to(device)
            pred = model(x)  # (B, T, 2)
            T = pred.shape[1]
            target_xy = az_to_xy(az_rad).unsqueeze(1).expand(-1, T, -1)
            loss = torch.mean((pred - target_xy) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = total / max(n_seen, 1)
        val = evaluate(model, val_loader, device)
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mae"].append(val["mae_deg"])
        history["val_p90"].append(val["p90_deg"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
            f"val_mae={val['mae_deg']:.2f}  median={val['median_deg']:.2f}  "
            f"p90={val['p90_deg']:.2f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if val["mae_deg"] < best_mae:
            best_mae = val["mae_deg"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mics": 4,
                    "n_freq": 257,
                    "epoch": epoch,
                    "val_mae": val["mae_deg"],
                },
                best_path,
            )
            print(f"  -> saved {best_path} (val MAE {best_mae:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE on ACCDOA")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"], color="C2", marker="o", label="MAE")
    axes[1].plot(history["val_p90"], color="C3", marker="s", label="P90")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation angular error (deg)")
    axes[1].set_title("Validation angular error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved {cfg.plot_path}")
    print(f"[train] best val MAE = {best_mae:.2f} deg, checkpoint -> {best_path}")


if __name__ == "__main__":
    main()
