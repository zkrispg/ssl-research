"""Compare W5 multi-source CRNN to SRP-PHAT and MUSIC on identical tests.

Two evaluation regimes:

1. **Oracle K**: each method is told the true number of sources K.
   The CRNN takes the top-K peaks by sigmoid response, SRP-PHAT and
   MUSIC take the top-K peaks of their respective spatial spectra. This
   isolates the localization quality of each method.

2. **Auto K**: each method must decide how many sources are present.
   The CRNN uses a tunable sigmoid threshold; SRP-PHAT/MUSIC use a
   relative-threshold rule on the spatial spectrum. This measures the
   joint counting + localization performance.

We sweep the test set across (k, snr, rt60) factor combinations and
report SELD-style precision/recall/F1 with a 20-degree tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import (  # noqa: E402
    find_peaks_circular,
    music_multi,
    srp_phat_multi,
)
from multi_dataset import make_grid  # noqa: E402
from multi_eval import LocalizationStats  # noqa: E402
from multi_model import MultiSourceCRNN  # noqa: E402
from multi_source_data import (  # noqa: E402
    sample_distinct_azimuths,
    simulate_freefield_multi,
    simulate_room_multi,
)

CHECKPOINT = Path(__file__).parent / "checkpoints" / "best.pt"
TOLERANCE_DEG = 20.0


def predict_crnn_oracle(
    signals: np.ndarray,
    model: MultiSourceCRNN,
    grid: np.ndarray,
    n_sources: int,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).parent.parent / "week03_cnn_doa"))
    from features import phase_features

    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).mean(dim=1).squeeze(0).cpu().numpy()
    return find_peaks_circular(
        probs, grid, n_peaks=n_sources, rel_threshold=0.0, min_separation_deg=25.0
    )


def predict_crnn_auto(
    signals: np.ndarray,
    model: MultiSourceCRNN,
    grid: np.ndarray,
    abs_threshold: float,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).parent.parent / "week03_cnn_doa"))
    from features import phase_features

    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).mean(dim=1).squeeze(0).cpu().numpy()
    # Use absolute threshold on sigmoid probability for "auto K".
    if probs.max() < abs_threshold:
        return np.empty(0, dtype=np.float32)
    rel = abs_threshold / max(probs.max(), 1e-6)
    return find_peaks_circular(
        probs, grid, n_peaks=None, rel_threshold=rel, min_separation_deg=25.0
    )


def gen_test_signals(
    mics: np.ndarray,
    azimuths: np.ndarray,
    snr_db: float,
    rt60: float,
    fs: int,
    seed: int,
):
    if rt60 == 0.0:
        return simulate_freefield_multi(
            mic_positions=mics,
            azimuths_deg=azimuths,
            fs=fs,
            duration=1.0,
            snr_db=snr_db,
            seed=seed,
        )
    return simulate_room_multi(
        mic_positions=mics,
        azimuths_deg=azimuths,
        rt60=rt60,
        fs=fs,
        duration=1.0,
        snr_db=snr_db,
        seed=seed,
    )


def eval_factor(
    *,
    name: str,
    k_values: list[int],
    snr_db: float,
    rt60: float,
    n_per_k: int,
    mics: np.ndarray,
    grid: np.ndarray,
    model: MultiSourceCRNN,
    fs: int,
    n_fft: int,
    hop_length: int,
    crnn_threshold: float,
) -> dict[str, dict[str, float]]:
    """Run all 6 method/K-mode combinations on the same test mixtures."""
    methods = {
        "srp_oracleK": LocalizationStats(),
        "music_oracleK": LocalizationStats(),
        "crnn_oracleK": LocalizationStats(),
        "srp_autoK": LocalizationStats(),
        "music_autoK": LocalizationStats(),
        "crnn_autoK": LocalizationStats(),
    }
    rng = np.random.default_rng(hash((name, snr_db, rt60)) & 0xFFFFFFFF)

    total = sum(n_per_k for _ in k_values)
    pbar = tqdm(total=total, desc=name, leave=False)
    for k in k_values:
        for trial in range(n_per_k):
            local_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            azs = sample_distinct_azimuths(local_rng, k, min_separation_deg=30.0)
            seed = int(local_rng.integers(0, 2**31))
            signals, _ = gen_test_signals(
                mics, azs, snr_db=snr_db, rt60=rt60, fs=fs, seed=seed
            )

            srp_oracle = srp_phat_multi(
                signals, mics, fs=fs, n_sources=k, rel_threshold=0.0
            )
            mus_oracle = music_multi(
                signals, mics, fs=fs, n_sources=k, rel_threshold=0.0
            )
            crnn_oracle = predict_crnn_oracle(
                signals, model, grid, n_sources=k, fs=fs,
                n_fft=n_fft, hop_length=hop_length,
            )

            srp_auto = srp_phat_multi(
                signals, mics, fs=fs, n_sources=None, rel_threshold=0.5
            )
            mus_auto = music_multi(
                signals, mics, fs=fs, n_sources=3, rel_threshold=0.5
            )
            crnn_auto = predict_crnn_auto(
                signals, model, grid, abs_threshold=crnn_threshold, fs=fs,
                n_fft=n_fft, hop_length=hop_length,
            )

            methods["srp_oracleK"].add_sample(srp_oracle, azs, TOLERANCE_DEG)
            methods["music_oracleK"].add_sample(mus_oracle, azs, TOLERANCE_DEG)
            methods["crnn_oracleK"].add_sample(crnn_oracle, azs, TOLERANCE_DEG)
            methods["srp_autoK"].add_sample(srp_auto, azs, TOLERANCE_DEG)
            methods["music_autoK"].add_sample(mus_auto, azs, TOLERANCE_DEG)
            methods["crnn_autoK"].add_sample(crnn_auto, azs, TOLERANCE_DEG)
            pbar.update(1)
    pbar.close()
    return {name: stats.summary() for name, stats in methods.items()}


def print_table(title: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(
        f"{'method':<18}  {'F1':>5}  {'P':>5}  {'R':>5}  "
        f"{'MAE_TP':>7}  {'count':>6}",
        flush=True,
    )
    for name, m in results.items():
        print(
            f"{name:<18}  {m['f1']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  "
            f"{m['mae_tp_deg']:7.2f}  {m['count_acc']:.3f}",
            flush=True,
        )


def plot_oracle_vs_auto(
    rt60_results: dict[float, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    rt60s = sorted(rt60_results.keys())
    methods_oracle = ["srp_oracleK", "music_oracleK", "crnn_oracleK"]
    methods_auto = ["srp_autoK", "music_autoK", "crnn_autoK"]
    label_map = {
        "srp_oracleK": "SRP-PHAT",
        "music_oracleK": "MUSIC",
        "crnn_oracleK": "CRNN (W5)",
        "srp_autoK": "SRP-PHAT",
        "music_autoK": "MUSIC",
        "crnn_autoK": "CRNN (W5)",
    }
    color_map = {"srp": "C1", "music": "C2", "crnn": "C0"}

    for ax, methods, title in (
        (axes[0], methods_oracle, "Oracle K (true source count given)"),
        (axes[1], methods_auto, "Auto K (each method counts sources)"),
    ):
        for m in methods:
            f1s = [rt60_results[r][m]["f1"] for r in rt60s]
            color_key = m.split("_")[0]
            marker = {"srp": "s", "music": "^", "crnn": "o"}[color_key]
            ax.plot(rt60s, f1s, color=color_map[color_key], marker=marker,
                    label=label_map[m], linewidth=2)
        labels = ["anechoic" if r == 0.0 else f"{r:.1f}" for r in rt60s]
        ax.set_xticks(rt60s)
        ax.set_xticklabels(labels)
        ax.set_xlabel("RT60 (s)")
        ax.set_ylabel("F1 (tol=20 deg)")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}; run train.py first")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=len(grid))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"[eval] loaded checkpoint from epoch {ckpt['epoch']} (val F1 {ckpt['val_f1']:.3f})",
        flush=True,
    )

    crnn_threshold = 0.5  # absolute sigmoid threshold for autoK

    rt60_results = {}
    for rt60 in [0.0, 0.3, 0.6]:
        snr = 10.0
        name = f"RT60={rt60:.1f}, SNR={snr}"
        rt60_results[rt60] = eval_factor(
            name=name,
            k_values=[1, 2, 3],
            snr_db=snr,
            rt60=rt60,
            n_per_k=15,
            mics=mics,
            grid=grid,
            model=model,
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            crnn_threshold=crnn_threshold,
        )
        print_table(name, rt60_results[rt60])

    plot_oracle_vs_auto(rt60_results, Path(__file__).parent / "eval_rt60_f1.png")

    # SNR sweep, anechoic, K varies
    print("\n=== SNR sweep, anechoic ===", flush=True)
    snr_results = {}
    for snr in [20.0, 0.0, -10.0]:
        name = f"SNR={snr}, anechoic"
        snr_results[snr] = eval_factor(
            name=name,
            k_values=[1, 2, 3],
            snr_db=snr,
            rt60=0.0,
            n_per_k=15,
            mics=mics,
            grid=grid,
            model=model,
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            crnn_threshold=crnn_threshold,
        )
        print_table(name, snr_results[snr])


if __name__ == "__main__":
    main()
