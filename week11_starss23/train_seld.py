"""SELD training script for STARSS23.

Trains a :class:`SeldCRNN` (with optional GCA) on the cached STARSS23
dev-train split, evaluates loss on dev-test, and writes a checkpoint
plus per-epoch metrics. Designed to be re-runnable with different
seeds for the multi-seed paired-test protocol.

Usage:
    python -m week11_starss23.train_seld \
        --variant no_geom \
        --epochs 15 \
        --seed 0 \
        --out-suffix run0
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from week11_starss23.seld_augment import (
    SpecAugment,
    SpecAugmentConfig,
    specaug_config_for_strength,
)
from week11_starss23.seld_features import SeldFeatureConfig
from week11_starss23.seld_loss import ClassCoupledAdpitLoss
from week11_starss23.seld_model import SeldCRNN, SeldModelConfig, default_uca4_positions
from week11_starss23.starss_dataset import StarssDataset

DEFAULT_AUDIO_DIR = Path("D:/ssl-research/data/STARSS23/mic_dev")
DEFAULT_META_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")
DEFAULT_CACHE_DIR = Path("D:/ssl-research/data/STARSS23/feat_cache")
DEFAULT_OUT_DIR = Path("D:/ssl-research/week11_starss23/runs")

# FOA mode swaps audio dir and feat cache (different feature stack -> different cache).
DEFAULT_FOA_AUDIO_DIR = Path("D:/ssl-research/data/STARSS23/foa_dev")
DEFAULT_FOA_CACHE_DIR = Path("D:/ssl-research/data/STARSS23/foa_feat_cache")


VARIANTS = {
    # Each entry produces a SeldModelConfig override -- except
    # "seldnet_official" which dispatches to the strict DCASE 2023
    # baseline (see week11_starss23.seldnet_official).
    "baseline": dict(use_gca=False),
    "full": dict(use_gca=True, gca_geometry_bias=True),
    "no_geom": dict(use_gca=True, gca_geometry_bias=False),
    "seldnet_official": dict(),  # uses SeldNetOfficial; no SeldModelConfig overrides

    # ---- Capacity sweep variants (Tier 2 ablation) ----
    # Defaults: cnn_filters=64, rnn_hidden=128 -> ~590 K params with use_gca=True.
    # We hold use_gca=True and gca_geometry_bias on/off to keep the geometry
    # ablation orthogonal at every capacity point.
    "no_geom_xs": dict(
        use_gca=True, gca_geometry_bias=False,
        cnn_filters=32, rnn_hidden=96,
    ),  # ~250 K params
    "full_xs": dict(
        use_gca=True, gca_geometry_bias=True,
        cnn_filters=32, rnn_hidden=96,
    ),
    "no_geom_l": dict(
        use_gca=True, gca_geometry_bias=False,
        cnn_filters=96, rnn_hidden=192,
    ),  # ~1.5 M params
    "full_l": dict(
        use_gca=True, gca_geometry_bias=True,
        cnn_filters=96, rnn_hidden=192,
    ),
    "no_geom_xl": dict(
        use_gca=True, gca_geometry_bias=False,
        cnn_filters=128, rnn_hidden=256,
    ),  # ~3 M params
    "full_xl": dict(
        use_gca=True, gca_geometry_bias=True,
        cnn_filters=128, rnn_hidden=256,
    ),
}

# Variants that route through SeldNetOfficial instead of SeldCRNN.
SELDNET_OFFICIAL_VARIANTS = {"seldnet_official"}

# Mapping from variant name to a short capacity tag for tables/plots.
CAPACITY_TAG = {
    "no_geom_xs": "xs", "full_xs": "xs",
    "no_geom":    "m",  "full":    "m",
    "no_geom_l":  "l",  "full_l":  "l",
    "no_geom_xl": "xl", "full_xl": "xl",
}


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    audio_dir: Path,
    metadata_dir: Path,
    cache_dir: Path,
    feat_cfg: SeldFeatureConfig,
    train_clip_seconds: float,
    eval_clip_seconds: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_crops_per_clip: int = 1,
    in_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="train",
        feature_config=feat_cfg,
        clip_seconds=train_clip_seconds,
        random_crop=True,
        cache_dir=cache_dir,
        seed=seed,
        crops_per_clip=train_crops_per_clip,
        in_memory=in_memory,
    )
    eval_ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="test",
        feature_config=feat_cfg,
        clip_seconds=eval_clip_seconds,
        random_crop=False,
        cache_dir=cache_dir,
        seed=seed,
        in_memory=in_memory,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, eval_loader


def train_one_epoch(
    model: SeldCRNN,
    loader: DataLoader,
    loss_fn: ClassCoupledAdpitLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augment: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    if augment is not None:
        augment.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if augment is not None:
            feats = augment(feats)
        optimizer.zero_grad(set_to_none=True)
        out = model(feats)["accdoa"]
        loss = loss_fn(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = feats.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs
    return {"loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def evaluate(
    model: SeldCRNN,
    loader: DataLoader,
    loss_fn: ClassCoupledAdpitLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        out = model(feats)["accdoa"]
        loss = loss_fn(out, target)
        bs = feats.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs
    return {"loss": total_loss / max(n_samples, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), default="no_geom")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-clip-seconds", type=float, default=5.0)
    parser.add_argument("--eval-clip-seconds", type=float, default=10.0)
    parser.add_argument(
        "--train-crops-per-clip",
        type=int,
        default=1,
        help="number of independent random crops per clip per epoch (1 = old behaviour)",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="cache decoded features in RAM (~3-5 GB for STARSS23) for faster epochs",
    )
    parser.add_argument(
        "--specaug",
        action="store_true",
        help="enable SpecAugment on training inputs (DCASE 2023 baseline defaults)",
    )
    parser.add_argument(
        "--specaug-strength",
        type=str,
        default="strong",
        choices=["strong", "weak", "off"],
        help="SpecAug strength preset; 'strong' = DCASE 2023 default, 'weak' = "
             "half-strength control, 'off' = no masks (identity).",
    )
    parser.add_argument(
        "--array-type", choices=["mic", "foa"], default="mic",
        help="audio array format. 'mic' uses the 4-mic GCC-PHAT stack (10 ch); "
             "'foa' uses W,X,Y,Z + intensity vector (7 ch). When set to 'foa' "
             "audio-dir and cache-dir default to the FOA versions if not given.",
    )
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--out-suffix", type=str, default="", help="appended to run name (run id, etc.)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="3-batch toy run for smoke-testing the pipeline",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Resolve audio/cache dirs based on array type, unless user overrode.
    if args.array_type == "foa":
        if args.audio_dir is None:
            args.audio_dir = DEFAULT_FOA_AUDIO_DIR
        if args.cache_dir is None:
            args.cache_dir = DEFAULT_FOA_CACHE_DIR
    else:
        if args.audio_dir is None:
            args.audio_dir = DEFAULT_AUDIO_DIR
        if args.cache_dir is None:
            args.cache_dir = DEFAULT_CACHE_DIR
    print(f"[audio] {args.audio_dir}")
    print(f"[cache] {args.cache_dir}")

    feat_cfg = SeldFeatureConfig(array_type=args.array_type)
    in_channels = feat_cfg.n_feature_channels()
    if args.variant in SELDNET_OFFICIAL_VARIANTS:
        from week11_starss23.seldnet_official import (
            SeldNetOfficial,
            SeldNetOfficialConfig,
        )
        model_cfg = SeldNetOfficialConfig(in_channels=in_channels)
        model = SeldNetOfficial(model_cfg).to(device)
        model_type = "seldnet_official"
    else:
        model_kwargs = VARIANTS[args.variant].copy()
        model_kwargs["in_channels"] = in_channels
        model_cfg = SeldModelConfig(**model_kwargs)
        model = SeldCRNN(model_cfg, mic_positions=default_uca4_positions()).to(device)
        model_type = "seld_crnn"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] variant={args.variant}  type={model_type}  "
        f"in_channels={in_channels}  params={n_params:,}"
    )

    train_loader, eval_loader = make_loaders(
        audio_dir=args.audio_dir,
        metadata_dir=args.metadata_dir,
        cache_dir=args.cache_dir,
        feat_cfg=feat_cfg,
        train_clip_seconds=args.train_clip_seconds,
        eval_clip_seconds=args.eval_clip_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_crops_per_clip=args.train_crops_per_clip,
        in_memory=args.in_memory,
    )
    print(
        f"[data] train_batches={len(train_loader)}  "
        f"eval_batches={len(eval_loader)}  bs={args.batch_size}"
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = ClassCoupledAdpitLoss().to(device)
    augment: torch.nn.Module | None = None
    if args.specaug:
        augment = SpecAugment(specaug_config_for_strength(args.specaug_strength)).to(device)
        print(f"[augment] strength={args.specaug_strength}  {augment}", flush=True)

    run_name = f"{args.variant}_seed{args.seed}"
    if args.out_suffix:
        run_name += f"_{args.out_suffix}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    history = []
    best_eval = float("inf")
    overall_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        if args.quick_test:
            # Truncate iterators to a few batches for sanity-only run.
            train_metrics = _quick_pass(model, train_loader, loss_fn, optimizer, device, n=3)
            eval_metrics = _quick_pass(model, eval_loader, loss_fn, None, device, n=2)
        else:
            train_metrics = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, augment=augment
            )
            eval_metrics = evaluate(model, eval_loader, loss_fn, device)
        scheduler.step()
        dt = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "eval_loss": eval_metrics["loss"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": dt,
        }
        history.append(row)
        print(
            f"  epoch {epoch:3d}/{args.epochs}  "
            f"train={row['train_loss']:.5f}  eval={row['eval_loss']:.5f}  "
            f"lr={row['lr']:.2e}  dt={dt:.1f}s",
            flush=True,
        )

        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "feat_cfg": asdict(feat_cfg),
                    "model_type": model_type,
                    "args": vars(args),
                    "history": history,
                },
                out_dir / "best.pt",
            )

    total_t = time.perf_counter() - overall_t0
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "best_eval_loss": best_eval,
        "n_params": n_params,
        "history": history,
        "total_seconds": total_t,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] best_eval={best_eval:.5f}  total_time={total_t:.1f}s  -> {out_dir}")


def _quick_pass(
    model: SeldCRNN,
    loader: DataLoader,
    loss_fn: ClassCoupledAdpitLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    n: int,
) -> dict[str, float]:
    """Run ``n`` batches for smoke testing."""
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total = 0.0
    count = 0
    it = iter(loader)
    for _ in range(n):
        try:
            batch = next(it)
        except StopIteration:
            break
        feats = batch["features"].to(device)
        target = batch["target"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            out = model(feats)["accdoa"]
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(feats)["accdoa"]
                loss = loss_fn(out, target)
        total += loss.item() * feats.shape[0]
        count += feats.shape[0]
    return {"loss": total / max(count, 1)}


if __name__ == "__main__":
    main()
