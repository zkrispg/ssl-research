"""End-to-end GCC-PHAT demo.

Sweeps a ground-truth azimuth from -80 to +80 degrees, simulates a two-mic
recording at each angle, runs GCC-PHAT, and reports the estimation error.
A correct implementation should yield mean absolute error well below 2 deg
at SNR >= 20 dB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from gcc_phat import gcc_phat, tdoa_to_doa
from simulate import simulate_two_mic


def evaluate_sweep(
    angles: np.ndarray,
    mic_distance: float = 0.1,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 20.0,
    n_seeds: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GCC-PHAT on a range of ground-truth angles with multiple seeds.

    Returns ``(angles, mean_estimates)`` where each angle's estimate is the
    mean over ``n_seeds`` independent noise/source realizations. NaN values
    (which occur when the GCC peak is outside the physical TDOA range under
    very low SNR) are filtered out before averaging.
    """
    estimates = np.zeros_like(angles, dtype=float)
    max_tau = mic_distance / 343.0 * 1.05

    for i, theta in enumerate(angles):
        per_seed = []
        for s in range(n_seeds):
            sig1, sig2, _ = simulate_two_mic(
                azimuth_deg=float(theta),
                mic_distance=mic_distance,
                fs=fs,
                duration=duration,
                snr_db=snr_db,
                seed=int(i * 1000 + s),
            )
            tau_hat, _ = gcc_phat(sig1, sig2, fs=fs, max_tau=max_tau, interp=16)
            est = tdoa_to_doa(tau_hat, mic_distance)
            if np.isfinite(est):
                per_seed.append(est)
        estimates[i] = float(np.mean(per_seed)) if per_seed else float("nan")

    return angles, estimates


def plot_single_example(
    azimuth_deg: float = 30.0,
    mic_distance: float = 0.1,
    fs: int = 16000,
    snr_db: float = 20.0,
    out_path: str = "week01_gcc_phat/demo_cc.png",
) -> None:
    """Plot the cross-correlation function for one example."""
    sig1, sig2, true_tau = simulate_two_mic(
        azimuth_deg=azimuth_deg,
        mic_distance=mic_distance,
        fs=fs,
        snr_db=snr_db,
    )
    max_tau = mic_distance / 343.0 * 1.05
    tau_hat, cc = gcc_phat(sig1, sig2, fs=fs, max_tau=max_tau, interp=16)
    est = tdoa_to_doa(tau_hat, mic_distance)

    interp = 16
    lags = np.arange(-(len(cc) // 2), len(cc) // 2 + 1) / (interp * fs)
    if len(lags) != len(cc):
        lags = lags[: len(cc)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags * 1e6, cc)
    ax.axvline(true_tau * 1e6, color="g", linestyle="--", label=f"true tau = {true_tau*1e6:.1f} us")
    ax.axvline(tau_hat * 1e6, color="r", linestyle=":", label=f"est. tau = {tau_hat*1e6:.1f} us")
    ax.set_xlabel("Lag (us)")
    ax.set_ylabel("GCC-PHAT")
    ax.set_title(f"GCC-PHAT  true={azimuth_deg:.1f} deg  est={est:.1f} deg  SNR={snr_db} dB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[demo] saved cross-correlation plot to {out_path}")


def main() -> None:
    fs = 16000
    mic_distance = 0.1

    print("=" * 60)
    print("GCC-PHAT sanity check")
    print(f"  fs = {fs} Hz, mic distance = {mic_distance*100:.1f} cm")
    print("=" * 60)

    angles = np.arange(-80, 81, 10, dtype=float)

    for snr_db in [30.0, 20.0, 10.0, 0.0, -5.0]:
        gt, est = evaluate_sweep(
            angles, mic_distance=mic_distance, fs=fs, snr_db=snr_db
        )
        err = np.abs(gt - est)
        valid = np.isfinite(err)
        n_valid = int(valid.sum())
        mae = float(np.mean(err[valid])) if n_valid else float("nan")
        max_err = float(np.max(err[valid])) if n_valid else float("nan")
        print(
            f"SNR = {snr_db:5.1f} dB   MAE = {mae:5.2f} deg   "
            f"max err = {max_err:5.2f} deg   valid = {n_valid}/{len(angles)}"
        )

    plot_single_example(
        azimuth_deg=30.0,
        mic_distance=mic_distance,
        fs=fs,
        snr_db=20.0,
        out_path=str(Path(__file__).parent / "demo_cc.png"),
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    for snr_db in [30.0, 20.0, 10.0, 0.0]:
        gt, est = evaluate_sweep(angles, mic_distance=mic_distance, fs=fs, snr_db=snr_db)
        ax.plot(gt, est, marker="o", label=f"SNR={snr_db:.0f} dB")
    ax.plot([-90, 90], [-90, 90], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("True azimuth (deg)")
    ax.set_ylabel("Estimated azimuth (deg)")
    ax.set_title("GCC-PHAT DOA estimation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = str(Path(__file__).parent / "demo_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[demo] saved sweep plot to {out}")


if __name__ == "__main__":
    main()
