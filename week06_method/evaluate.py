"""Compare W6 full method against W5 CRNN, SRP-PHAT, and MUSIC.

Uses the same RT60/SNR test grid as ``week05_multi_source/evaluate.py`` so
the numbers extend directly into the W5 baseline table. The W6 model
uses its count head to decide K and then takes top-K peaks; W5 uses
relative-threshold peak picking (auto-K).
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week03_cnn_doa"))
sys.path.insert(0, str(Path(__file__).parent.parent / "week05_multi_source"))

from features import phase_features  # noqa: E402
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
from multi_task_model import MultiTaskCRNN  # noqa: E402

W5_CKPT = (
    Path(__file__).parent.parent / "week05_multi_source" / "checkpoints" / "best.pt"
)
W6_CKPT = Path(__file__).parent / "checkpoints" / "best_full.pt"
TOLERANCE_DEG = 20.0


def predict_w5(
    signals: np.ndarray,
    model: MultiSourceCRNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
    threshold: float = 0.5,
) -> np.ndarray:
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).mean(dim=1).squeeze(0).cpu().numpy()
    if probs.max() < threshold:
        return np.empty(0, dtype=np.float32)
    rel = threshold / max(probs.max(), 1e-6)
    return find_peaks_circular(
        probs, grid, n_peaks=None, rel_threshold=rel, min_separation_deg=25.0
    )


def predict_w6(
    signals: np.ndarray,
    model: MultiTaskCRNN,
    grid: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(
        probs, grid, n_peaks=pred_k, rel_threshold=0.0, min_separation_deg=25.0
    )


def gen_test_signals(mics, azimuths, snr_db, rt60, fs, seed):
    if rt60 == 0.0:
        return simulate_freefield_multi(
            mic_positions=mics, azimuths_deg=azimuths,
            fs=fs, duration=1.0, snr_db=snr_db, seed=seed,
        )
    return simulate_room_multi(
        mic_positions=mics, azimuths_deg=azimuths, rt60=rt60,
        fs=fs, duration=1.0, snr_db=snr_db, seed=seed,
    )


def eval_factor(
    *, name, k_values, snr_db, rt60, n_per_k,
    mics, grid, w5_model, w6_model, fs, n_fft, hop_length,
):
    methods = {
        "srp_oracleK": LocalizationStats(),
        "music_oracleK": LocalizationStats(),
        "w5_autoK": LocalizationStats(),
        "w6_full": LocalizationStats(),
    }
    rng = np.random.default_rng(hash((name, snr_db, rt60)) & 0xFFFFFFFF)

    total = sum(n_per_k for _ in k_values)
    pbar = tqdm(total=total, desc=name, leave=False)
    for k in k_values:
        for trial in range(n_per_k):
            local_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
            azs = sample_distinct_azimuths(local_rng, k, min_separation_deg=30.0)
            seed = int(local_rng.integers(0, 2 ** 31))
            signals, _ = gen_test_signals(mics, azs, snr_db, rt60, fs, seed)

            srp_oracle = srp_phat_multi(signals, mics, fs=fs, n_sources=k, rel_threshold=0.0)
            mus_oracle = music_multi(signals, mics, fs=fs, n_sources=k, rel_threshold=0.0)
            w5_pred = predict_w5(signals, w5_model, grid, fs, n_fft, hop_length, threshold=0.5)
            w6_pred = predict_w6(signals, w6_model, grid, fs, n_fft, hop_length)

            methods["srp_oracleK"].add_sample(srp_oracle, azs, TOLERANCE_DEG)
            methods["music_oracleK"].add_sample(mus_oracle, azs, TOLERANCE_DEG)
            methods["w5_autoK"].add_sample(w5_pred, azs, TOLERANCE_DEG)
            methods["w6_full"].add_sample(w6_pred, azs, TOLERANCE_DEG)
            pbar.update(1)
    pbar.close()
    return {n: stats.summary() for n, stats in methods.items()}


def print_table(title, results):
    print(f"\n=== {title} ===", flush=True)
    print(
        f"{'method':<15}  {'F1':>5}  {'P':>5}  {'R':>5}  {'MAE_TP':>7}  {'count':>6}",
        flush=True,
    )
    for name, m in results.items():
        print(
            f"{name:<15}  {m['f1']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  "
            f"{m['mae_tp_deg']:7.2f}  {m['count_acc']:.3f}",
            flush=True,
        )


def plot_summary(rt60_results, snr_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    methods = ["srp_oracleK", "music_oracleK", "w5_autoK", "w6_full"]
    labels = {
        "srp_oracleK": "SRP-PHAT (oracle K)",
        "music_oracleK": "MUSIC (oracle K)",
        "w5_autoK": "W5 CRNN (auto K)",
        "w6_full": "W6 full (count head)",
    }
    colors = {
        "srp_oracleK": "C1",
        "music_oracleK": "C2",
        "w5_autoK": "C0",
        "w6_full": "C3",
    }
    markers = {
        "srp_oracleK": "s",
        "music_oracleK": "^",
        "w5_autoK": "o",
        "w6_full": "*",
    }

    rt60s = sorted(rt60_results.keys())
    for m in methods:
        f1s = [rt60_results[r][m]["f1"] for r in rt60s]
        axes[0].plot(rt60s, f1s, color=colors[m], marker=markers[m],
                     label=labels[m], linewidth=2, markersize=8)
    axes[0].set_xticks(rt60s)
    axes[0].set_xticklabels(["anechoic" if r == 0.0 else f"{r:.1f}" for r in rt60s])
    axes[0].set_xlabel("RT60 (s)")
    axes[0].set_ylabel("F1 (tol=20 deg)")
    axes[0].set_title("Reverberation sweep, SNR=10 dB, K∈{1,2,3}")
    axes[0].set_ylim(0.3, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    snrs = sorted(snr_results.keys())
    for m in methods:
        f1s = [snr_results[s][m]["f1"] for s in snrs]
        axes[1].plot(snrs, f1s, color=colors[m], marker=markers[m],
                     label=labels[m], linewidth=2, markersize=8)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("F1 (tol=20 deg)")
    axes[1].set_title("SNR sweep, anechoic, K∈{1,2,3}")
    axes[1].set_ylim(0.3, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    if not W5_CKPT.exists():
        raise SystemExit(f"W5 checkpoint not found: {W5_CKPT}")
    if not W6_CKPT.exists():
        raise SystemExit(f"W6 checkpoint not found: {W6_CKPT}")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    w5_ckpt = torch.load(W5_CKPT, map_location="cpu", weights_only=False)
    w5_model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=len(grid))
    w5_model.load_state_dict(w5_ckpt["model_state"])
    w5_model.eval()
    print(f"[eval] W5 checkpoint epoch {w5_ckpt['epoch']} val F1 {w5_ckpt['val_f1']:.3f}",
          flush=True)

    w6_ckpt = torch.load(W6_CKPT, map_location="cpu", weights_only=False)
    w6_model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=len(grid), max_k=3)
    w6_model.load_state_dict(w6_ckpt["model_state"])
    w6_model.eval()
    print(f"[eval] W6 checkpoint epoch {w6_ckpt['epoch']} val F1 {w6_ckpt['val_f1']:.3f}  "
          f"head_acc {w6_ckpt['val_count_head']:.3f}", flush=True)

    rt60_results = {}
    for rt60 in [0.0, 0.3, 0.6]:
        rt60_results[rt60] = eval_factor(
            name=f"RT60={rt60:.1f}, SNR=10",
            k_values=[1, 2, 3], snr_db=10.0, rt60=rt60, n_per_k=15,
            mics=mics, grid=grid, w5_model=w5_model, w6_model=w6_model,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"RT60={rt60:.1f}, SNR=10", rt60_results[rt60])

    snr_results = {}
    for snr in [20.0, 0.0, -10.0]:
        snr_results[snr] = eval_factor(
            name=f"SNR={snr}, anechoic",
            k_values=[1, 2, 3], snr_db=snr, rt60=0.0, n_per_k=15,
            mics=mics, grid=grid, w5_model=w5_model, w6_model=w6_model,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"SNR={snr}, anechoic", snr_results[snr])

    plot_summary(rt60_results, snr_results, Path(__file__).parent / "eval_summary.png")


if __name__ == "__main__":
    main()
