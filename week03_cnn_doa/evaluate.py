"""Evaluate the trained CNN against SRP-PHAT and MUSIC on identical test sets.

Two evaluation regimes:

1. **SNR sweep, free-field**: -10 .. +20 dB, UCA4, single source.
2. **RT60 sweep, fixed SNR=10dB**: rt60 in 0 / 0.2 / 0.4 / 0.6 / 0.9 s,
   shoebox room.

For the CNN, we aggregate per-frame softmax predictions across all STFT
frames of a test signal and take the argmax of the mean. Ground truth is
snapped to the nearest class on the training azimuth grid before computing
the angular error so that CNN, SRP-PHAT, and MUSIC are compared on the
same target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week01_gcc_phat"))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from dataset import azimuth_classes  # noqa: E402
from features import phase_features  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from model import PhaseMapCNN  # noqa: E402
from music import music  # noqa: E402
from simulate_array import simulate_freefield, simulate_room  # noqa: E402
from srp_phat import srp_phat  # noqa: E402

CHECKPOINT = Path(__file__).parent / "checkpoints" / "best.pt"


def _wrap_deg(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def predict_cnn(
    signals: np.ndarray,
    model: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> float:
    """CNN inference by averaging per-frame softmax across all STFT frames."""
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    # feat: (2, M, F, T) -> (T, 2, M, F)
    x = torch.from_numpy(np.moveaxis(feat, -1, 0).copy()).float()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    return float(grid[int(np.argmax(probs))])


def run_one_trial(
    signals: np.ndarray,
    mics: np.ndarray,
    fs: int,
    model: PhaseMapCNN,
    grid: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> dict[str, float]:
    """Run CNN, SRP-PHAT, MUSIC on the same signals; return predicted azimuths."""
    cnn_pred = predict_cnn(signals, model, grid, fs=fs, n_fft=n_fft, hop_length=hop_length)
    srp_pred, _, _ = srp_phat(signals, mic_positions=mics, fs=fs)
    mus_pred, _, _ = music(signals, mic_positions=mics, fs=fs, n_sources=1)
    return {"cnn": cnn_pred, "srp": srp_pred, "music": mus_pred}


def snap_to_grid(az_deg: float, grid: np.ndarray) -> float:
    """Snap to nearest azimuth on the training grid (wrapped distance)."""
    diff = ((grid - az_deg + 180.0) % 360.0) - 180.0
    return float(grid[int(np.argmin(np.abs(diff)))])


def eval_snr_sweep(
    snrs: list[float],
    angles: np.ndarray,
    n_seeds: int,
    mics: np.ndarray,
    model: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, list[float]]:
    print("\n=== SNR sweep, free-field, UCA4 ===")
    results: dict[str, list[float]] = {"cnn": [], "srp": [], "music": []}

    for snr in snrs:
        errs = {"cnn": [], "srp": [], "music": []}
        total = len(angles) * n_seeds
        with tqdm(total=total, desc=f"SNR={snr:+.1f}", leave=False) as pbar:
            for ang in angles:
                ang_snapped = snap_to_grid(float(ang), grid)
                for seed in range(n_seeds):
                    signals, _ = simulate_freefield(
                        mic_positions=mics,
                        azimuth_deg=float(ang),
                        fs=fs,
                        duration=1.0,
                        snr_db=float(snr),
                        seed=int(abs(seed) * 10000 + abs(int(ang)) + 1),
                    )
                    preds = run_one_trial(
                        signals, mics, fs, model, grid, n_fft, hop_length
                    )
                    for k, v in preds.items():
                        errs[k].append(abs(_wrap_deg(v - ang_snapped)))
                    pbar.update(1)
        for k in results:
            results[k].append(float(np.mean(errs[k])))
        print(
            f"  SNR={snr:+6.1f} dB    "
            f"CNN={results['cnn'][-1]:6.2f}    "
            f"SRP={results['srp'][-1]:6.2f}    "
            f"MUSIC={results['music'][-1]:6.2f}"
        )
    return results


def eval_rt60_sweep(
    rt60_list: list[float],
    snr_db: float,
    angles: np.ndarray,
    n_seeds: int,
    mics: np.ndarray,
    model: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, list[float]]:
    print(f"\n=== RT60 sweep, SNR={snr_db} dB, UCA4 ===")
    results: dict[str, list[float]] = {"cnn": [], "srp": [], "music": []}

    for rt60 in rt60_list:
        errs = {"cnn": [], "srp": [], "music": []}
        label = "anechoic" if rt60 == 0.0 else f"RT60={rt60:.1f}"
        total = len(angles) * n_seeds
        with tqdm(total=total, desc=label, leave=False) as pbar:
            for ang in angles:
                ang_snapped = snap_to_grid(float(ang), grid)
                for seed in range(n_seeds):
                    if rt60 == 0.0:
                        signals, _ = simulate_freefield(
                            mic_positions=mics,
                            azimuth_deg=float(ang),
                            fs=fs,
                            duration=1.0,
                            snr_db=float(snr_db),
                            seed=int(abs(seed) * 10000 + abs(int(ang)) + 1),
                        )
                    else:
                        signals, _ = simulate_room(
                            mic_positions=mics,
                            azimuth_deg=float(ang),
                            rt60=rt60,
                            fs=fs,
                            duration=1.0,
                            snr_db=float(snr_db),
                            seed=int(abs(seed) * 10000 + abs(int(ang)) + 1),
                        )
                    preds = run_one_trial(
                        signals, mics, fs, model, grid, n_fft, hop_length
                    )
                    for k, v in preds.items():
                        errs[k].append(abs(_wrap_deg(v - ang_snapped)))
                    pbar.update(1)
        for k in results:
            results[k].append(float(np.mean(errs[k])))
        print(
            f"  {label:<12}    "
            f"CNN={results['cnn'][-1]:6.2f}    "
            f"SRP={results['srp'][-1]:6.2f}    "
            f"MUSIC={results['music'][-1]:6.2f}"
        )
    return results


def plot_snr_sweep(snrs: list[float], results: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style = {"cnn": ("PhaseMap CNN", "C0", "o"), "srp": ("SRP-PHAT", "C1", "s"), "music": ("MUSIC", "C2", "^")}
    for k, (label, color, marker) in style.items():
        ax.plot(snrs, results[k], color=color, marker=marker, label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (deg)")
    ax.set_yscale("log")
    ax.set_title("Free-field DOA estimation, UCA4, single speaker")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_rt60_sweep(rt60s: list[float], results: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rt60s))
    width = 0.27
    style = {"cnn": ("PhaseMap CNN", "C0"), "srp": ("SRP-PHAT", "C1"), "music": ("MUSIC", "C2")}
    for i, (k, (label, color)) in enumerate(style.items()):
        ax.bar(x + (i - 1) * width, results[k], width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"RT60={t:.1f}" if t > 0 else "anechoic" for t in rt60s])
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Reverberation robustness, UCA4, SNR=10 dB")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}; run train.py first")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = azimuth_classes(-180, 180, 5)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = PhaseMapCNN(n_mics=4, n_freq=n_fft // 2 + 1, n_classes=len(grid))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded checkpoint from epoch {ckpt['epoch']} (val MAE {ckpt['val_mae']:.2f})")

    angles = np.arange(-150, 151, 30, dtype=float)

    snrs = [20.0, 10.0, 0.0, -5.0, -10.0]
    snr_results = eval_snr_sweep(
        snrs=snrs,
        angles=angles,
        n_seeds=3,
        mics=mics,
        model=model,
        grid=grid,
        fs=fs,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    rt60_list = [0.0, 0.2, 0.4, 0.6, 0.9]
    rt60_results = eval_rt60_sweep(
        rt60_list=rt60_list,
        snr_db=10.0,
        angles=angles,
        n_seeds=3,
        mics=mics,
        model=model,
        grid=grid,
        fs=fs,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    plot_snr_sweep(snrs, snr_results, Path(__file__).parent / "eval_snr.png")
    plot_rt60_sweep(rt60_list, rt60_results, Path(__file__).parent / "eval_rt60.png")


if __name__ == "__main__":
    main()
