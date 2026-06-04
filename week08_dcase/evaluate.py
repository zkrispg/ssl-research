"""Compare W8 Multi-ACCDOA against W6 sigmoid-spectrum, W5 CRNN, SRP-PHAT,
and MUSIC under DCASE-style metrics.

The comparison uses identical RT60/SNR grids and the same fixed seeds as
:mod:`week06_method.evaluate`, but every method is now scored with the
four DCASE Task 3 metrics (F1, ER, LE_CD, LR_CD) plus the combined SELD
score. This is the core experimental table for the W8 chapter.

The key change relative to W6 is the W8 prediction path: the model
outputs Multi-ACCDOA tracks instead of a 72-bin sigmoid spectrum, and
the auxiliary count head is used to clip predictions to the top-K most
active tracks after NMS.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week06_method"))

from dcase_metrics import DcaseSeldStats, format_summary, overall_seld_score  # noqa: E402
from features import phase_features  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_accdoa_model import MultiAccdoaCRNN  # noqa: E402
from multi_accdoa import decode_multi_accdoa  # noqa: E402
from multi_baselines import (  # noqa: E402
    find_peaks_circular,
    music_multi,
    srp_phat_multi,
)
from multi_dataset import make_grid  # noqa: E402
from multi_model import MultiSourceCRNN  # noqa: E402
from multi_source_data import (  # noqa: E402
    sample_distinct_azimuths,
    simulate_freefield_multi,
    simulate_room_multi,
)
from multi_task_model import MultiTaskCRNN  # noqa: E402

W5_CKPT = Path(__file__).parent.parent / "week05_multi_source" / "checkpoints" / "best.pt"
W6_CKPT = Path(__file__).parent.parent / "week06_method" / "checkpoints" / "best_full.pt"
W8_CKPT = Path(__file__).parent / "checkpoints" / "best_v2.pt"
TOLERANCE_DEG = 20.0


def predict_w5(signals, model, grid, fs, n_fft, hop_length, threshold=0.5):
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


def predict_w6(signals, model, grid, fs, n_fft, hop_length):
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(
        probs, grid, n_peaks=pred_k, rel_threshold=0.0, min_separation_deg=25.0
    )


def predict_w8(signals, model, fs, n_fft, hop_length, *, activity_threshold=0.5,
               nms_tol_deg=25.0):
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    decoded = decode_multi_accdoa(
        out["accdoa"], activity_threshold=activity_threshold, nms_tol_deg=nms_tol_deg,
    )[0]
    if len(decoded) > pred_k:
        decoded = decoded[:pred_k]
    return decoded


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


def eval_factor(*, name, k_values, snr_db, rt60, n_per_k, mics, grid,
                w5_model, w6_model, w8_model, fs, n_fft, hop_length):
    method_names = ["srp_oracleK", "music_oracleK", "w5_autoK", "w6_full",
                    "w8_accdoa"]
    methods = {n: DcaseSeldStats(tolerance_deg=TOLERANCE_DEG) for n in method_names}
    rng = np.random.default_rng(hash((name, snr_db, rt60)) & 0xFFFFFFFF)

    total = sum(n_per_k for _ in k_values)
    pbar = tqdm(total=total, desc=name, leave=False)
    for k in k_values:
        for _ in range(n_per_k):
            local_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
            azs = sample_distinct_azimuths(local_rng, k, min_separation_deg=30.0)
            seed = int(local_rng.integers(0, 2 ** 31))
            signals, _ = gen_test_signals(mics, azs, snr_db, rt60, fs, seed)

            srp_oracle = srp_phat_multi(signals, mics, fs=fs, n_sources=k,
                                        rel_threshold=0.0)
            mus_oracle = music_multi(signals, mics, fs=fs, n_sources=k,
                                     rel_threshold=0.0)
            w5_pred = predict_w5(signals, w5_model, grid, fs, n_fft, hop_length,
                                 threshold=0.5)
            w6_pred = predict_w6(signals, w6_model, grid, fs, n_fft, hop_length)
            w8_pred = predict_w8(signals, w8_model, fs, n_fft, hop_length)

            methods["srp_oracleK"].add_sample(srp_oracle, azs)
            methods["music_oracleK"].add_sample(mus_oracle, azs)
            methods["w5_autoK"].add_sample(w5_pred, azs)
            methods["w6_full"].add_sample(w6_pred, azs)
            methods["w8_accdoa"].add_sample(w8_pred, azs)
            pbar.update(1)
    pbar.close()
    return {n: stats.summary() for n, stats in methods.items()}


def print_table(title, results):
    print(f"\n=== {title} ===", flush=True)
    print(
        f"{'method':<15}  {'F1':>5}  {'ER':>5}  {'LE_CD':>6}  {'LR_CD':>5}  "
        f"{'count':>5}  {'SELD':>5}",
        flush=True,
    )
    for name, m in results.items():
        seld = overall_seld_score(m)
        print(
            f"{name:<15}  {m['F1']:.3f}  {m['ER']:.3f}  {m['LE_CD']:6.2f}  "
            f"{m['LR_CD']:.3f}  {m['count_acc']:.3f}  {seld:.3f}",
            flush=True,
        )


def plot_summary(rt60_results, snr_results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    methods = ["srp_oracleK", "music_oracleK", "w5_autoK", "w6_full", "w8_accdoa"]
    labels = {
        "srp_oracleK": "SRP-PHAT (oracle K)",
        "music_oracleK": "MUSIC (oracle K)",
        "w5_autoK": "W5 CRNN (auto K)",
        "w6_full": "W6 sigmoid (count head)",
        "w8_accdoa": "W8 Multi-ACCDOA + ADPIT",
    }
    colors = {
        "srp_oracleK": "C1", "music_oracleK": "C2", "w5_autoK": "C0",
        "w6_full": "C3", "w8_accdoa": "C4",
    }
    markers = {
        "srp_oracleK": "s", "music_oracleK": "^", "w5_autoK": "o",
        "w6_full": "*", "w8_accdoa": "D",
    }

    rt60s = sorted(rt60_results.keys())
    for m in methods:
        f1s = [rt60_results[r][m]["F1"] for r in rt60s]
        selds = [overall_seld_score(rt60_results[r][m]) for r in rt60s]
        axes[0, 0].plot(rt60s, f1s, color=colors[m], marker=markers[m],
                        label=labels[m], linewidth=2, markersize=8)
        axes[1, 0].plot(rt60s, selds, color=colors[m], marker=markers[m],
                        label=labels[m], linewidth=2, markersize=8)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks(rt60s)
        ax.set_xticklabels(["anechoic" if r == 0.0 else f"{r:.1f}" for r in rt60s])
        ax.set_xlabel("RT60 (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel("F1 (tol=20 deg)")
    axes[0, 0].set_title("Reverberation sweep -- higher is better")
    axes[0, 0].set_ylim(0.3, 1.05)
    axes[1, 0].set_ylabel("SELD score (lower is better)")
    axes[1, 0].set_title("Reverberation sweep -- DCASE SELD score")

    snrs = sorted(snr_results.keys())
    for m in methods:
        f1s = [snr_results[s][m]["F1"] for s in snrs]
        selds = [overall_seld_score(snr_results[s][m]) for s in snrs]
        axes[0, 1].plot(snrs, f1s, color=colors[m], marker=markers[m],
                        label=labels[m], linewidth=2, markersize=8)
        axes[1, 1].plot(snrs, selds, color=colors[m], marker=markers[m],
                        label=labels[m], linewidth=2, markersize=8)
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0, 1].set_ylabel("F1 (tol=20 deg)")
    axes[0, 1].set_title("SNR sweep, anechoic")
    axes[0, 1].set_ylim(0.3, 1.05)
    axes[1, 1].set_ylabel("SELD score (lower is better)")
    axes[1, 1].set_title("SNR sweep -- DCASE SELD score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    for ckpt_path in (W5_CKPT, W6_CKPT, W8_CKPT):
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    w5_ckpt = torch.load(W5_CKPT, map_location="cpu", weights_only=False)
    w5_model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=len(grid))
    w5_model.load_state_dict(w5_ckpt["model_state"])
    w5_model.eval()
    print(f"[eval] W5 epoch={w5_ckpt['epoch']}  val F1={w5_ckpt['val_f1']:.3f}",
          flush=True)

    w6_ckpt = torch.load(W6_CKPT, map_location="cpu", weights_only=False)
    w6_model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=len(grid), max_k=3)
    w6_model.load_state_dict(w6_ckpt["model_state"])
    w6_model.eval()
    print(f"[eval] W6 epoch={w6_ckpt['epoch']}  val F1={w6_ckpt['val_f1']:.3f}  "
          f"head_acc={w6_ckpt['val_count_head']:.3f}", flush=True)

    w8_ckpt = torch.load(W8_CKPT, map_location="cpu", weights_only=False)
    w8_model = MultiAccdoaCRNN(
        n_mics=w8_ckpt["n_mics"], n_freq=w8_ckpt["n_freq"],
        n_tracks=w8_ckpt["n_tracks"], max_k=w8_ckpt["max_k"],
    )
    w8_model.load_state_dict(w8_ckpt["model_state"])
    w8_model.eval()
    print(f"[eval] W8 epoch={w8_ckpt['epoch']}  "
          f"val SELD={w8_ckpt['seld_score']:.3f}  "
          f"head_acc={w8_ckpt['val_summary']['head_count_acc']:.3f}", flush=True)

    rt60_results = {}
    for rt60 in [0.0, 0.3, 0.6]:
        rt60_results[rt60] = eval_factor(
            name=f"RT60={rt60:.1f}, SNR=10",
            k_values=[1, 2, 3], snr_db=10.0, rt60=rt60, n_per_k=15,
            mics=mics, grid=grid,
            w5_model=w5_model, w6_model=w6_model, w8_model=w8_model,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"RT60={rt60:.1f}, SNR=10", rt60_results[rt60])

    snr_results = {}
    for snr in [20.0, 0.0, -10.0]:
        snr_results[snr] = eval_factor(
            name=f"SNR={snr}, anechoic",
            k_values=[1, 2, 3], snr_db=snr, rt60=0.0, n_per_k=15,
            mics=mics, grid=grid,
            w5_model=w5_model, w6_model=w6_model, w8_model=w8_model,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"SNR={snr}, anechoic", snr_results[snr])

    plot_summary(rt60_results, snr_results,
                 Path(__file__).parent / "eval_summary.png")


if __name__ == "__main__":
    main()
