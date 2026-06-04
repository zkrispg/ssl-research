"""Out-of-distribution evaluation for the W6 model.

Each test sample is drawn from a different random acoustic environment
(see :class:`DiverseRoomSampler`). We slice the test set by RT60 (low /
mid / high) to expose how each method's performance scales with
reverberation outside of the narrow training distribution. The same
test set is used for SRP-PHAT, MUSIC, the W5 single-task CRNN, and the
W6 multi-task CRNN, so the resulting numbers extend the W6 results
table directly.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "week09_geometry_attn"))

from diverse_simulator import DiverseRoomSampler, simulate_diverse  # noqa: E402
from features import phase_features  # noqa: E402
from gca_model import GCAMultiTaskCRNN  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import (  # noqa: E402
    find_peaks_circular,
    music_multi,
    srp_phat_multi,
)
from multi_dataset import make_grid  # noqa: E402
from multi_eval import LocalizationStats  # noqa: E402
from multi_model import MultiSourceCRNN  # noqa: E402
from multi_source_data import sample_distinct_azimuths  # noqa: E402
from multi_task_model import MultiTaskCRNN  # noqa: E402

TOLERANCE_DEG = 20.0
W5_CKPT = Path(__file__).parent.parent / "week05_multi_source" / "checkpoints" / "best.pt"
W6_CKPT = Path(__file__).parent.parent / "week06_method" / "checkpoints" / "best_full.pt"
W9_CKPT = Path(__file__).parent.parent / "week09_geometry_attn" / "checkpoints" / "best_no_geom.pt"


def predict_w5(signals, model, grid, fs, n_fft, hop_length, threshold=0.5):
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).mean(dim=1).squeeze(0).cpu().numpy()
    if probs.max() < threshold:
        return np.empty(0, dtype=np.float32)
    rel = threshold / max(probs.max(), 1e-6)
    return find_peaks_circular(probs, grid, n_peaks=None, rel_threshold=rel,
                               min_separation_deg=25.0)


def predict_w6(signals, model, grid, fs, n_fft, hop_length):
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(probs, grid, n_peaks=pred_k, rel_threshold=0.0,
                               min_separation_deg=25.0)


def predict_w9(signals, model, grid, fs, n_fft, hop_length):
    """Identical decoding path to W6; only the preceding feature transform differs."""
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(probs, grid, n_peaks=pred_k, rel_threshold=0.0,
                               min_separation_deg=25.0)


def stratify_by_rt60(rt60: float) -> str:
    if rt60 < 0.35:
        return "low"
    if rt60 < 0.65:
        return "mid"
    return "high"


def evaluate(
    *,
    n_samples: int,
    snr_db: float,
    rt60_range: tuple[float, float],
    mics: np.ndarray,
    grid: np.ndarray,
    w5_model: MultiSourceCRNN,
    w6_model: MultiTaskCRNN,
    w9_model: GCAMultiTaskCRNN | None,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run all 4-5 methods on ``n_samples`` mixtures with random env params.

    Returns nested dict ``stratum -> method -> metric``.
    """
    sampler = DiverseRoomSampler(rt60_range=rt60_range)
    methods = ["srp_oracleK", "music_oracleK", "w5_autoK", "w6_full"]
    if w9_model is not None:
        methods.append("w9_no_geom")
    strata = ["low", "mid", "high"]
    stats = {s: {m: LocalizationStats() for m in methods} for s in strata}
    rng = np.random.default_rng(0)

    pbar = tqdm(total=n_samples, desc=f"snr={snr_db}")
    for i in range(n_samples):
        local_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
        K = int(local_rng.integers(1, 4))
        azs = sample_distinct_azimuths(local_rng, K, min_separation_deg=30.0)
        seed = int(local_rng.integers(0, 2 ** 31))
        signals, info = simulate_diverse(
            mic_positions=mics,
            azimuths_deg=azs,
            sampler=sampler,
            fs=fs,
            duration=1.0,
            snr_db=snr_db,
            seed=seed,
        )
        bucket = stratify_by_rt60(info["rt60"])

        srp_oracle = srp_phat_multi(signals, mics, fs=fs, n_sources=K, rel_threshold=0.0)
        mus_oracle = music_multi(signals, mics, fs=fs, n_sources=K, rel_threshold=0.0)
        w5_pred = predict_w5(signals, w5_model, grid, fs, n_fft, hop_length)
        w6_pred = predict_w6(signals, w6_model, grid, fs, n_fft, hop_length)

        stats[bucket]["srp_oracleK"].add_sample(srp_oracle, azs, TOLERANCE_DEG)
        stats[bucket]["music_oracleK"].add_sample(mus_oracle, azs, TOLERANCE_DEG)
        stats[bucket]["w5_autoK"].add_sample(w5_pred, azs, TOLERANCE_DEG)
        stats[bucket]["w6_full"].add_sample(w6_pred, azs, TOLERANCE_DEG)
        if w9_model is not None:
            w9_pred = predict_w9(signals, w9_model, grid, fs, n_fft, hop_length)
            stats[bucket]["w9_no_geom"].add_sample(w9_pred, azs, TOLERANCE_DEG)
        pbar.update(1)
    pbar.close()

    return {s: {m: stats[s][m].summary() for m in methods} for s in strata}


def print_results(snr_db: float, results: dict) -> None:
    print(f"\n=== OOD test, SNR={snr_db} dB ===", flush=True)
    for bucket in ["low", "mid", "high"]:
        rows = results[bucket]
        if rows["srp_oracleK"]["n_samples"] == 0:
            print(f"  {bucket:<5}  (no samples)", flush=True)
            continue
        print(
            f"  {bucket:<5} (RT60 {'<0.35' if bucket=='low' else '0.35-0.65' if bucket=='mid' else '>0.65'} s, "
            f"n={rows['srp_oracleK']['n_samples']})",
            flush=True,
        )
        print(
            f"    {'method':<14}  {'F1':>5}  {'P':>5}  {'R':>5}  {'MAE_TP':>7}  {'count':>6}",
            flush=True,
        )
        for name in rows.keys():
            m = rows[name]
            print(
                f"    {name:<14}  {m['f1']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  "
                f"{m['mae_tp_deg']:7.2f}  {m['count_acc']:.3f}",
                flush=True,
            )


def plot_summary(snr_results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(snr_results), figsize=(5 * len(snr_results), 4.5))
    if len(snr_results) == 1:
        axes = [axes]
    methods = list(next(iter(snr_results.values()))["low"].keys())
    labels = {
        "srp_oracleK": "SRP-PHAT (oracle K)",
        "music_oracleK": "MUSIC (oracle K)",
        "w5_autoK": "W5 CRNN (auto K)",
        "w6_full": "W6 full (count head)",
        "w9_no_geom": "W9 no_geom (channel attn.)",
    }
    colors = {"srp_oracleK": "C1", "music_oracleK": "C2",
              "w5_autoK": "C0", "w6_full": "C3", "w9_no_geom": "C4"}
    markers = {"srp_oracleK": "s", "music_oracleK": "^",
               "w5_autoK": "o", "w6_full": "*", "w9_no_geom": "D"}

    buckets = ["low", "mid", "high"]
    bucket_labels = ["RT60<0.35", "0.35-0.65", ">0.65"]
    x = np.arange(len(buckets))

    for ax, (snr, results) in zip(axes, sorted(snr_results.items())):
        for m in methods:
            f1s = [results[b][m]["f1"] for b in buckets]
            ax.plot(x, f1s, color=colors[m], marker=markers[m], markersize=10,
                    label=labels[m], linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels)
        ax.set_xlabel("RT60 stratum")
        ax.set_ylabel("F1 (tol=20 deg)")
        ax.set_title(f"OOD test, SNR={snr:.0f} dB, K∈{{1,2,3}}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main() -> None:
    if not W5_CKPT.exists() or not W6_CKPT.exists():
        raise SystemExit("missing W5/W6 checkpoints; run their train scripts first")

    fs = 16000
    n_fft = 512
    hop_length = 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    w5_ckpt = torch.load(W5_CKPT, map_location="cpu", weights_only=False)
    w5_model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=len(grid))
    w5_model.load_state_dict(w5_ckpt["model_state"])
    w5_model.eval()

    w6_ckpt = torch.load(W6_CKPT, map_location="cpu", weights_only=False)
    w6_model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=len(grid), max_k=3)
    w6_model.load_state_dict(w6_ckpt["model_state"])
    w6_model.eval()
    print(
        f"[eval] W5 epoch {w5_ckpt['epoch']} val F1 {w5_ckpt['val_f1']:.3f}; "
        f"W6 epoch {w6_ckpt['epoch']} val F1 {w6_ckpt['val_f1']:.3f}",
        flush=True,
    )

    w9_model: GCAMultiTaskCRNN | None = None
    if W9_CKPT.exists():
        w9_ckpt = torch.load(W9_CKPT, map_location="cpu", weights_only=False)
        w9_model = GCAMultiTaskCRNN(
            mic_positions=mics, n_freq=257, n_classes=len(grid),
            max_k=3, geometry_bias=bool(w9_ckpt.get("geometry_bias", False)),
        )
        w9_model.load_state_dict(w9_ckpt["model_state"])
        w9_model.eval()
        print(
            f"[eval] W9 no_geom epoch {w9_ckpt['epoch']} val F1 {w9_ckpt['val_f1']:.3f}",
            flush=True,
        )

    snr_results = {}
    for snr in [10.0, 0.0]:
        snr_results[snr] = evaluate(
            n_samples=120,
            snr_db=snr,
            rt60_range=(0.20, 1.0),
            mics=mics,
            grid=grid,
            w5_model=w5_model,
            w6_model=w6_model,
            w9_model=w9_model,
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        print_results(snr, snr_results[snr])

    plot_summary(snr_results, Path(__file__).parent / "ood_eval.png")


if __name__ == "__main__":
    main()
