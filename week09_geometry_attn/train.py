"""Train W9 GCA-augmented multi-task CRNN with ablation variants.

Variants
--------
* ``full``    : GCA with geometry bias + count head + augmentation (W9 main).
* ``no_geom`` : GCA without geometry bias (plain channel attention) +
                count head + augmentation -- isolates the contribution
                of the geometry token vs the attention itself.
* ``no_aug``  : full GCA + count head, no augmentation -- isolates the
                interaction between GCA and channel-rotation augmentation.

The training pipeline (data, optimiser, schedule, loss) is identical to
``week06_method/train.py`` so the only intentional difference between W6
``best_full.pt`` and W9 ``best_full.pt`` is the GCA module.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week06_method"))

from gca_model import GCAMultiTaskCRNN, count_parameters  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import find_peaks_circular  # noqa: E402
from multi_dataset import MultiSourceConfig, make_grid  # noqa: E402
from multi_eval import LocalizationStats  # noqa: E402
from multi_task_dataset import AugConfig, build_datasets  # noqa: E402

VARIANTS = {
    "full":    {"geometry_bias": True,  "use_aug": True},
    "no_geom": {"geometry_bias": False, "use_aug": True},
    "no_aug":  {"geometry_bias": True,  "use_aug": False},
}


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    train_samples: int = 2000
    val_samples: int = 400
    duration: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 12.0
    lambda_count: float = 1.0
    tolerance_deg: float = 20.0
    rt60_range: tuple[float, float] | None = (0.15, 0.5)
    reverb_prob: float = 0.5
    snr_range_db: tuple[float, float] = (-5.0, 30.0)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    grid: np.ndarray,
    device: torch.device,
    bce: nn.Module,
    ce: nn.Module,
    tolerance_deg: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    correct_count = 0
    stats = LocalizationStats()
    with torch.no_grad():
        for x, y, az_padded, k in loader:
            x = x.to(device)
            y = y.to(device)
            k = k.to(device)
            out = model(x)
            spec = out["spectrum"]
            T = spec.shape[1]
            target = y.unsqueeze(1).expand(-1, T, -1)
            loss_spec = bce(spec, target)
            loss_count = ce(out["count"], k)
            loss = loss_spec + loss_count
            total_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

            probs = torch.sigmoid(spec).mean(dim=1).cpu().numpy()
            pred_k = out["count"].argmax(dim=-1).cpu().numpy() + 1
            correct_count += int(((torch.tensor(pred_k) - 1) == k.cpu()).sum().item())

            for b in range(probs.shape[0]):
                preds = find_peaks_circular(
                    probs[b], grid, n_peaks=int(pred_k[b]),
                    rel_threshold=0.0, min_separation_deg=25.0,
                )
                gts = az_padded[b].numpy()
                gts = gts[~np.isnan(gts)]
                stats.add_sample(preds, gts, tolerance_deg=tolerance_deg)

    summary = stats.summary()
    summary["loss"] = total_loss / max(n_seen, 1)
    summary["count_acc_head"] = correct_count / max(n_seen, 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to warm-start the model from. "
                        "The optimizer is reset; lr can be scaled with --lr-scale.")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="Multiplier on base lr (useful for resume finetune).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed: shifts torch/numpy seeds and the "
                        "dataset seed_base for multi-seed significance runs.")
    parser.add_argument("--speech", action="store_true",
                        help="Replace band-limited noise sources with "
                        "synthetic formant-based speech-like signals "
                        "(W10 robustness experiment).")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Append to checkpoint and plot filenames.")
    args = parser.parse_args()
    settings = VARIANTS[args.variant]

    if args.speech:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "week10_significance"))
        from speech_source import make_source_speech_like
        from multi_source_data import set_source_generator
        set_source_generator(make_source_speech_like)
        print("[train] using synthetic speech-like sources", flush=True)

    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = []
    if args.seed != 0:
        suffix_parts.append(f"seed{args.seed}")
    if args.speech:
        suffix_parts.append("speech")
    if args.out_suffix:
        suffix_parts.append(args.out_suffix)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    plot_path = Path(__file__).parent / f"training_{args.variant}{suffix}.png"
    ckpt_path = out_dir / f"best_{args.variant}{suffix}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    print(f"[train] variant = {args.variant}  geometry_bias={settings['geometry_bias']}  "
          f"use_aug={settings['use_aug']}", flush=True)

    grid = make_grid(-180, 180, 5)

    train_ds_cfg = MultiSourceConfig(
        mic_positions=mics, n_samples=cfg.train_samples, duration=cfg.duration,
        rt60_range=cfg.rt60_range, reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db, seed_base=args.seed * 1_000_000,
    )
    val_ds_cfg = MultiSourceConfig(
        mic_positions=mics, n_samples=cfg.val_samples, duration=cfg.duration,
        rt60_range=cfg.rt60_range, reverb_prob=cfg.reverb_prob,
        snr_range_db=cfg.snr_range_db,
        seed_base=10_000_000 + args.seed * 1_000_000,
    )

    aug_cfg = AugConfig() if settings["use_aug"] else AugConfig(False, False)
    train_ds, val_ds = build_datasets(train_ds_cfg, val_ds_cfg, aug=aug_cfg)
    if not settings["use_aug"]:
        train_ds.train_mode = False

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    model = GCAMultiTaskCRNN(
        mic_positions=mics, n_classes=len(grid), max_k=3,
        gca_embed_dim=16, geometry_bias=settings["geometry_bias"],
    ).to(device)
    print(f"[train] model parameters: {count_parameters(model):,}", flush=True)

    best_f1 = -1.0
    if args.resume:
        ckpt_resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume["model_state"])
        prev_f1 = ckpt_resume.get("val_f1", -1.0)
        if isinstance(prev_f1, (int, float)) and prev_f1 == prev_f1:
            best_f1 = float(prev_f1)
        print(f"[train] resumed from {args.resume}  (prev val F1 {best_f1:.3f})",
              flush=True)

    pos_weight = torch.full((len(grid),), cfg.pos_weight, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr * args.lr_scale, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_count_head": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}",
                    leave=False, mininterval=1.0)
        for x, y, _, k in pbar:
            x = x.to(device)
            y = y.to(device)
            k = k.to(device)
            out = model(x)
            spec = out["spectrum"]
            T = spec.shape[1]
            target = y.unsqueeze(1).expand(-1, T, -1)
            loss_spec = bce(spec, target)
            loss_count = ce(out["count"], k)
            loss = loss_spec + cfg.lambda_count * loss_count
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = total / max(n_seen, 1)

        val = evaluate(
            model, val_loader, grid, device, bce, ce,
            tolerance_deg=cfg.tolerance_deg,
        )
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_f1"].append(val["f1"])
        history["val_count_head"].append(val["count_acc_head"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
            f"val_F1={val['f1']:.3f}  val_P={val['precision']:.3f}  "
            f"val_R={val['recall']:.3f}  val_MAE={val['mae_tp_deg']:.2f}  "
            f"head_count_acc={val['count_acc_head']:.3f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mics": 4, "n_freq": 257, "n_classes": len(grid),
                    "max_k": 3,                     "epoch": epoch,
                    "val_f1": val["f1"],
                    "val_count_head": val["count_acc_head"],
                    "variant": args.variant,
                    "geometry_bias": settings["geometry_bias"],
                    "seed": args.seed,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path} (F1 {best_f1:.3f})", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss ({args.variant})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_f1"], color="C2", marker="o")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation F1")
    axes[1].set_title("F1"); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(history["val_count_head"], color="C3", marker="^")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("count head accuracy")
    axes[2].set_title("Count head"); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved {plot_path}", flush=True)
    print(f"[train] best val F1 = {best_f1:.3f}, checkpoint -> {ckpt_path}",
          flush=True)


if __name__ == "__main__":
    main()
