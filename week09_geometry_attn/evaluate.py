"""Compare W9 (W6 backbone + GCA) against W6, W5, MUSIC, SRP-PHAT and W8.

Uses the **same** RT60/SNR test grid as :mod:`week08_dcase.evaluate`, with
identical seeds and number of trials per condition. All methods are
scored under DCASE Task 3 metrics (F1, ER, LE_CD, LR_CD) plus the
combined SELD score, so the resulting numbers extend the W8 table by a
single column ("W9 GCA").

By default the script evaluates all available W9 variant checkpoints
(``best_full.pt``, ``best_no_geom.pt``, ``best_no_aug.pt``) so a single
run produces the full ablation table.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week08_dcase"))

from dcase_metrics import DcaseSeldStats, format_summary, overall_seld_score  # noqa: E402
from features import phase_features  # noqa: E402
from gca_model import GCAMultiTaskCRNN  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import (  # noqa: E402
    find_peaks_circular,
    music_multi,
    srp_phat_multi,
)
from multi_dataset import make_grid  # noqa: E402
from multi_source_data import (  # noqa: E402
    sample_distinct_azimuths,
    simulate_freefield_multi,
    simulate_room_multi,
)
from multi_task_model import MultiTaskCRNN  # noqa: E402

W6_CKPT = Path(__file__).parent.parent / "week06_method" / "checkpoints" / "best_full.pt"
W9_DIR = Path(__file__).parent / "checkpoints"
TOLERANCE_DEG = 20.0


def predict_w6(signals, model, grid, fs, n_fft, hop_length):
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(
        probs, grid, n_peaks=pred_k, rel_threshold=0.0, min_separation_deg=25.0,
    )


def predict_w9(signals, model, grid, fs, n_fft, hop_length):
    """Identical decoding to W6 (same head topology); only feature path differs."""
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(
        probs, grid, n_peaks=pred_k, rel_threshold=0.0, min_separation_deg=25.0,
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


def eval_factor(*, name, k_values, snr_db, rt60, n_per_k, mics, grid,
                w6_model, w9_models, fs, n_fft, hop_length):
    method_names = ["srp_oracleK", "music_oracleK", "w6_full"]
    method_names += [f"w9_{v}" for v in w9_models]
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

            srp_oracle = srp_phat_multi(signals, mics, fs=fs, n_sources=k, rel_threshold=0.0)
            mus_oracle = music_multi(signals, mics, fs=fs, n_sources=k, rel_threshold=0.0)
            w6_pred = predict_w6(signals, w6_model, grid, fs, n_fft, hop_length)

            methods["srp_oracleK"].add_sample(srp_oracle, azs)
            methods["music_oracleK"].add_sample(mus_oracle, azs)
            methods["w6_full"].add_sample(w6_pred, azs)
            for variant, model in w9_models.items():
                w9_pred = predict_w9(signals, model, grid, fs, n_fft, hop_length)
                methods[f"w9_{variant}"].add_sample(w9_pred, azs)
            pbar.update(1)
    pbar.close()
    return {n: stats.summary() for n, stats in methods.items()}


def print_table(title, results):
    print(f"\n=== {title} ===", flush=True)
    print(
        f"{'method':<16}  {'F1':>5}  {'ER':>5}  {'LE_CD':>6}  {'LR_CD':>5}  "
        f"{'count':>5}  {'SELD':>5}",
        flush=True,
    )
    for name, m in results.items():
        seld = overall_seld_score(m)
        print(
            f"{name:<16}  {m['F1']:.3f}  {m['ER']:.3f}  {m['LE_CD']:6.2f}  "
            f"{m['LR_CD']:.3f}  {m['count_acc']:.3f}  {seld:.3f}",
            flush=True,
        )


def plot_summary(rt60_results, snr_results, methods, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = {
        "srp_oracleK":  "SRP-PHAT (oracle K)",
        "music_oracleK":"MUSIC (oracle K)",
        "w6_full":      "W6 sigmoid + count",
        "w9_full":      "W9 GCA full",
        "w9_no_geom":   "W9 plain attention",
        "w9_no_aug":    "W9 GCA no aug",
    }
    color_palette = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    marker_palette = ["s", "^", "*", "D", "P", "X", "v"]
    colors = {m: color_palette[i % len(color_palette)] for i, m in enumerate(methods)}
    markers = {m: marker_palette[i % len(marker_palette)] for i, m in enumerate(methods)}

    rt60s = sorted(rt60_results.keys())
    for m in methods:
        f1s = [rt60_results[r][m]["F1"] for r in rt60s]
        selds = [overall_seld_score(rt60_results[r][m]) for r in rt60s]
        axes[0, 0].plot(rt60s, f1s, color=colors[m], marker=markers[m],
                        label=labels.get(m, m), linewidth=2, markersize=8)
        axes[1, 0].plot(rt60s, selds, color=colors[m], marker=markers[m],
                        label=labels.get(m, m), linewidth=2, markersize=8)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks(rt60s)
        ax.set_xticklabels(["anechoic" if r == 0.0 else f"{r:.1f}" for r in rt60s])
        ax.set_xlabel("RT60 (s)"); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0, 0].set_ylabel("F1 (tol=20 deg)")
    axes[0, 0].set_title("RT60 sweep -- F1 (higher is better)")
    axes[0, 0].set_ylim(0.3, 1.05)
    axes[1, 0].set_ylabel("SELD score (lower is better)")
    axes[1, 0].set_title("RT60 sweep -- SELD score")

    snrs = sorted(snr_results.keys())
    for m in methods:
        f1s = [snr_results[s][m]["F1"] for s in snrs]
        selds = [overall_seld_score(snr_results[s][m]) for s in snrs]
        axes[0, 1].plot(snrs, f1s, color=colors[m], marker=markers[m],
                        label=labels.get(m, m), linewidth=2, markersize=8)
        axes[1, 1].plot(snrs, selds, color=colors[m], marker=markers[m],
                        label=labels.get(m, m), linewidth=2, markersize=8)
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_xlabel("SNR (dB)"); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0, 1].set_ylabel("F1 (tol=20 deg)")
    axes[0, 1].set_title("SNR sweep, anechoic")
    axes[0, 1].set_ylim(0.3, 1.05)
    axes[1, 1].set_ylabel("SELD score (lower is better)")
    axes[1, 1].set_title("SNR sweep -- SELD score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    if not W6_CKPT.exists():
        raise SystemExit(f"missing W6 checkpoint: {W6_CKPT}")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    w6_ckpt = torch.load(W6_CKPT, map_location="cpu", weights_only=False)
    w6_model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=len(grid), max_k=3)
    w6_model.load_state_dict(w6_ckpt["model_state"])
    w6_model.eval()
    print(f"[eval] W6 epoch={w6_ckpt['epoch']}  val F1={w6_ckpt['val_f1']:.3f}",
          flush=True)

    # Prefer resumed (longer-trained) checkpoints when available.
    variant_candidates = {
        "full": ["best_full_resumed.pt", "best_full.pt"],
        "no_geom": ["best_no_geom.pt"],
        "no_aug": ["best_no_aug.pt"],
    }
    w9_models: dict[str, GCAMultiTaskCRNN] = {}
    for variant, candidates in variant_candidates.items():
        ckpt_path = next((W9_DIR / c for c in candidates if (W9_DIR / c).exists()), None)
        if ckpt_path is None:
            print(f"[eval] skip W9 {variant}: no checkpoint among {candidates}",
                  flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = GCAMultiTaskCRNN(
            mic_positions=mics, n_freq=ckpt["n_freq"], n_classes=ckpt["n_classes"],
            max_k=ckpt["max_k"], geometry_bias=ckpt["geometry_bias"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        w9_models[variant] = model
        print(f"[eval] W9 {variant}: epoch={ckpt['epoch']}  val F1={ckpt['val_f1']:.3f}",
              flush=True)

    if not w9_models:
        raise SystemExit("no W9 checkpoints found; run train.py for at least one variant.")

    methods = ["srp_oracleK", "music_oracleK", "w6_full"] + [
        f"w9_{v}" for v in w9_models
    ]

    rt60_results = {}
    for rt60 in [0.0, 0.3, 0.6]:
        rt60_results[rt60] = eval_factor(
            name=f"RT60={rt60:.1f}, SNR=10",
            k_values=[1, 2, 3], snr_db=10.0, rt60=rt60, n_per_k=15,
            mics=mics, grid=grid,
            w6_model=w6_model, w9_models=w9_models,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"RT60={rt60:.1f}, SNR=10", rt60_results[rt60])

    snr_results = {}
    for snr in [20.0, 0.0, -10.0]:
        snr_results[snr] = eval_factor(
            name=f"SNR={snr}, anechoic",
            k_values=[1, 2, 3], snr_db=snr, rt60=0.0, n_per_k=15,
            mics=mics, grid=grid,
            w6_model=w6_model, w9_models=w9_models,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print_table(f"SNR={snr}, anechoic", snr_results[snr])

    plot_summary(rt60_results, snr_results, methods,
                 Path(__file__).parent / "eval_summary.png")


if __name__ == "__main__":
    main()
