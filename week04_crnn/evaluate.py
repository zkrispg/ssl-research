"""Evaluate W4 CRNN against W3 CNN, SRP-PHAT, and MUSIC on identical tests.

Same SNR-sweep and RT60-sweep test sets as ``week03_cnn_doa/evaluate.py``,
so the results directly extend the W3 baseline table with the multi-frame
CRNN method.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week03_cnn_doa"))

from crnn_dataset import az_to_xy, xy_to_az_deg  # noqa: E402
from crnn_model import CRNNDoa  # noqa: E402
from dataset import azimuth_classes  # noqa: E402
from features import phase_features  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from model import PhaseMapCNN  # noqa: E402
from music import music  # noqa: E402
from simulate_array import simulate_freefield, simulate_room  # noqa: E402
from srp_phat import srp_phat  # noqa: E402

CRNN_CKPT = Path(__file__).parent / "checkpoints" / "best.pt"
CNN_CKPT = Path(__file__).parent.parent / "week03_cnn_doa" / "checkpoints" / "best.pt"


def _wrap_deg(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def predict_crnn(
    signals: np.ndarray,
    model: CRNNDoa,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> float:
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)  # (1, 2, M, F, T)
    with torch.no_grad():
        pred = model(x)  # (1, T, 2)
        mean_xy = pred.mean(dim=1).squeeze(0)  # (2,)
    return float(xy_to_az_deg(mean_xy).item())


def predict_cnn(
    signals: np.ndarray,
    model: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> float:
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(np.moveaxis(feat, -1, 0).copy()).float()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    return float(grid[int(np.argmax(probs))])


def all_methods_predict(
    signals: np.ndarray,
    mics: np.ndarray,
    fs: int,
    crnn_model: CRNNDoa,
    cnn_model: PhaseMapCNN,
    grid: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> dict[str, float]:
    return {
        "crnn": predict_crnn(signals, crnn_model, fs=fs, n_fft=n_fft, hop_length=hop_length),
        "cnn": predict_cnn(signals, cnn_model, grid, fs=fs, n_fft=n_fft, hop_length=hop_length),
        "srp": srp_phat(signals, mic_positions=mics, fs=fs)[0],
        "music": music(signals, mic_positions=mics, fs=fs, n_sources=1)[0],
    }


def eval_snr_sweep(
    snrs: list[float],
    angles: np.ndarray,
    n_seeds: int,
    mics: np.ndarray,
    crnn: CRNNDoa,
    cnn: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
) -> dict[str, list[float]]:
    print("\n=== SNR sweep, free-field, UCA4 ===", flush=True)
    results: dict[str, list[float]] = {"crnn": [], "cnn": [], "srp": [], "music": []}
    for snr in snrs:
        errs = {k: [] for k in results}
        total = len(angles) * n_seeds
        with tqdm(total=total, desc=f"SNR={snr:+.1f}", leave=False) as pbar:
            for ang in angles:
                for seed in range(n_seeds):
                    signals, _ = simulate_freefield(
                        mic_positions=mics,
                        azimuth_deg=float(ang),
                        fs=fs,
                        duration=1.0,
                        snr_db=float(snr),
                        seed=int(abs(seed) * 10000 + abs(int(ang)) + 1),
                    )
                    preds = all_methods_predict(signals, mics, fs, crnn, cnn, grid)
                    for k, v in preds.items():
                        errs[k].append(abs(_wrap_deg(v - float(ang))))
                    pbar.update(1)
        for k in results:
            results[k].append(float(np.mean(errs[k])))
        print(
            f"  SNR={snr:+6.1f} dB    "
            f"CRNN={results['crnn'][-1]:6.2f}    "
            f"CNN={results['cnn'][-1]:6.2f}    "
            f"SRP={results['srp'][-1]:6.2f}    "
            f"MUSIC={results['music'][-1]:6.2f}",
            flush=True,
        )
    return results


def eval_rt60_sweep(
    rt60_list: list[float],
    snr_db: float,
    angles: np.ndarray,
    n_seeds: int,
    mics: np.ndarray,
    crnn: CRNNDoa,
    cnn: PhaseMapCNN,
    grid: np.ndarray,
    fs: int,
) -> dict[str, list[float]]:
    print(f"\n=== RT60 sweep, SNR={snr_db} dB, UCA4 ===", flush=True)
    results: dict[str, list[float]] = {"crnn": [], "cnn": [], "srp": [], "music": []}
    for rt60 in rt60_list:
        errs = {k: [] for k in results}
        label = "anechoic" if rt60 == 0.0 else f"RT60={rt60:.1f}"
        total = len(angles) * n_seeds
        with tqdm(total=total, desc=label, leave=False) as pbar:
            for ang in angles:
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
                    preds = all_methods_predict(signals, mics, fs, crnn, cnn, grid)
                    for k, v in preds.items():
                        errs[k].append(abs(_wrap_deg(v - float(ang))))
                    pbar.update(1)
        for k in results:
            results[k].append(float(np.mean(errs[k])))
        print(
            f"  {label:<12}    "
            f"CRNN={results['crnn'][-1]:6.2f}    "
            f"CNN={results['cnn'][-1]:6.2f}    "
            f"SRP={results['srp'][-1]:6.2f}    "
            f"MUSIC={results['music'][-1]:6.2f}",
            flush=True,
        )
    return results


def plot_snr(snrs: list[float], results: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    style = {
        "music": ("MUSIC", "C2", "^"),
        "srp": ("SRP-PHAT", "C1", "s"),
        "cnn": ("PhaseMap CNN (W3)", "C3", "d"),
        "crnn": ("Multi-frame CRNN (W4)", "C0", "o"),
    }
    for k, (label, color, marker) in style.items():
        ax.plot(snrs, results[k], color=color, marker=marker, label=label, linewidth=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (deg)")
    ax.set_yscale("log")
    ax.set_title("Free-field DOA estimation, UCA4")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def plot_rt60(rt60s: list[float], results: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(rt60s))
    width = 0.2
    style = {
        "music": ("MUSIC", "C2"),
        "srp": ("SRP-PHAT", "C1"),
        "cnn": ("PhaseMap CNN (W3)", "C3"),
        "crnn": ("Multi-frame CRNN (W4)", "C0"),
    }
    for i, (k, (label, color)) in enumerate(style.items()):
        ax.bar(x + (i - 1.5) * width, results[k], width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"RT60={t:.1f}" if t > 0 else "anechoic" for t in rt60s])
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Reverberation robustness, UCA4, SNR=10 dB")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def main() -> None:
    if not CRNN_CKPT.exists():
        raise SystemExit(f"CRNN checkpoint not found: {CRNN_CKPT}")
    if not CNN_CKPT.exists():
        raise SystemExit(f"CNN checkpoint not found: {CNN_CKPT}")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = azimuth_classes(-180, 180, 5)

    crnn = CRNNDoa(n_mics=4, n_freq=257)
    crnn.load_state_dict(torch.load(CRNN_CKPT, map_location="cpu", weights_only=False)["model_state"])
    crnn.eval()

    cnn = PhaseMapCNN(n_mics=4, n_freq=257, n_classes=72)
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location="cpu", weights_only=False)["model_state"])
    cnn.eval()

    angles = np.arange(-150, 151, 30, dtype=float)

    snrs = [20.0, 10.0, 0.0, -5.0, -10.0]
    snr_results = eval_snr_sweep(
        snrs=snrs, angles=angles, n_seeds=3, mics=mics, crnn=crnn, cnn=cnn, grid=grid, fs=fs
    )

    rt60_list = [0.0, 0.2, 0.4, 0.6, 0.9]
    rt60_results = eval_rt60_sweep(
        rt60_list=rt60_list, snr_db=10.0, angles=angles, n_seeds=3,
        mics=mics, crnn=crnn, cnn=cnn, grid=grid, fs=fs,
    )

    plot_snr(snrs, snr_results, Path(__file__).parent / "eval_snr.png")
    plot_rt60(rt60_list, rt60_results, Path(__file__).parent / "eval_rt60.png")


if __name__ == "__main__":
    main()
