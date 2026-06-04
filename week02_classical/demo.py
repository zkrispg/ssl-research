"""W2 demo: compare GCC-PHAT (2-mic), SRP-PHAT, and MUSIC on the same data.

Three experiments are reported:

1. **Free-field SNR sweep** (UCA4, no reverb):
   sweep azimuths -150..+150 deg, average MAE per SNR.
2. **Mic-count effect**: SRP-PHAT with 2 / 4 / 8 mics under low SNR.
3. **Reverberation**: free-field vs. RT60 = 0.3 / 0.6 s in a shoebox room.

Outputs PNG figures next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week01_gcc_phat"))

from gcc_phat import gcc_phat, tdoa_to_doa  # noqa: E402

from geometry import array_aperture, uniform_circular_array, uniform_linear_array  # noqa: E402
from music import music  # noqa: E402
from simulate_array import simulate_freefield, simulate_room  # noqa: E402
from srp_phat import srp_phat  # noqa: E402

OUT_DIR = Path(__file__).parent


def _wrap(deg: float) -> float:
    return ((deg + 180) % 360) - 180


def _eval_mae(errors: list[float]) -> float:
    return float(np.mean([abs(e) for e in errors]))


def _seed(seed: int, ang: float) -> int:
    """Build a non-negative seed combining a base index and an azimuth."""
    return int(abs(seed) * 10000 + abs(int(ang)) + (1 if ang < 0 else 0))


def _gcc_phat_2mic_estimate(signals: np.ndarray, mic_distance: float, fs: int) -> float:
    """Use the W1 2-mic GCC-PHAT on the first two channels of a linear-pair sub-array.

    The 2-mic solution can only resolve azimuth in [-90, 90] (front-back
    ambiguity). For comparison purposes we report the front-half angle.
    """
    max_tau = mic_distance / 343.0 * 1.05
    tau_hat, _ = gcc_phat(signals[0], signals[1], fs=fs, max_tau=max_tau, interp=16)
    return tdoa_to_doa(tau_hat, mic_distance)


def experiment_snr_sweep(
    snrs: list[float],
    fs: int = 16000,
    n_seeds: int = 4,
    out_path: Path = OUT_DIR / "snr_sweep.png",
) -> None:
    print("\n=== Experiment 1: SNR sweep, free-field, UCA4 ===")
    uca4 = uniform_circular_array(n_mics=4, radius=0.04)
    angles = np.arange(-150, 151, 30, dtype=float)

    results = {"GCC-PHAT (2 mic)": [], "SRP-PHAT (UCA4)": [], "MUSIC (UCA4)": []}

    # 2-mic linear pair: mics 1 and 3 of the UCA lie on the y-axis, so their
    # broadside is +x, which matches the azimuth convention (azimuth 0 = +x).
    pair_indices = [1, 3]
    pair_distance = 2.0 * 0.04

    for snr in snrs:
        errs_g, errs_s, errs_m = [], [], []
        for ang in angles:
            for seed in range(n_seeds):
                signals, _ = simulate_freefield(
                    mic_positions=uca4,
                    azimuth_deg=float(ang),
                    fs=fs,
                    duration=1.0,
                    snr_db=snr,
                    seed=_seed(seed, ang),
                )
                # 2-mic GCC-PHAT uses mics with broadside aligned to +x.
                pair = signals[pair_indices]
                gcc_est = _gcc_phat_2mic_estimate(pair, pair_distance, fs)
                if np.isfinite(gcc_est):
                    # ULA can only cover [-90, 90]. Fold ground truth.
                    ang_folded = ang
                    if ang > 90:
                        ang_folded = 180 - ang
                    elif ang < -90:
                        ang_folded = -180 - ang
                    errs_g.append(_wrap(gcc_est - ang_folded))

                srp_est, _, _ = srp_phat(signals, mic_positions=uca4, fs=fs)
                errs_s.append(_wrap(srp_est - ang))

                mus_est, _, _ = music(signals, mic_positions=uca4, fs=fs, n_sources=1)
                errs_m.append(_wrap(mus_est - ang))

        results["GCC-PHAT (2 mic)"].append(_eval_mae(errs_g))
        results["SRP-PHAT (UCA4)"].append(_eval_mae(errs_s))
        results["MUSIC (UCA4)"].append(_eval_mae(errs_m))
        print(
            f"  SNR={snr:+6.1f} dB    GCC={_eval_mae(errs_g):5.2f}    "
            f"SRP={_eval_mae(errs_s):5.2f}    MUSIC={_eval_mae(errs_m):5.2f}"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, vals in results.items():
        ax.plot(snrs, vals, marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (deg)")
    ax.set_yscale("log")
    ax.set_title("Free-field azimuth estimation, speech-band source")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def experiment_mic_count(
    snrs: list[float],
    fs: int = 16000,
    n_seeds: int = 4,
    out_path: Path = OUT_DIR / "mic_count.png",
) -> None:
    print("\n=== Experiment 2: SRP-PHAT mic count ===")
    angles = np.arange(-150, 151, 30, dtype=float)

    arrays = {
        "UCA2": uniform_circular_array(n_mics=2, radius=0.04),
        "UCA4": uniform_circular_array(n_mics=4, radius=0.04),
        "UCA8": uniform_circular_array(n_mics=8, radius=0.04),
    }

    results: dict[str, list[float]] = {name: [] for name in arrays}
    for snr in snrs:
        for name, arr in arrays.items():
            errs = []
            for ang in angles:
                for seed in range(n_seeds):
                    signals, _ = simulate_freefield(
                        mic_positions=arr,
                        azimuth_deg=float(ang),
                        fs=fs,
                        duration=1.0,
                        snr_db=snr,
                        seed=_seed(seed, ang),
                    )
                    est, _, _ = srp_phat(signals, mic_positions=arr, fs=fs)
                    errs.append(_wrap(est - ang))
            results[name].append(_eval_mae(errs))
        msg = "  SNR={:+6.1f} dB    ".format(snr) + "    ".join(
            f"{n}={results[n][-1]:5.2f}" for n in arrays
        )
        print(msg)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, vals in results.items():
        ax.plot(snrs, vals, marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (deg)")
    ax.set_yscale("log")
    ax.set_title("SRP-PHAT: more mics => more robust at low SNR")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def experiment_reverb(
    rt60_list: list[float],
    fs: int = 16000,
    n_seeds: int = 3,
    out_path: Path = OUT_DIR / "reverb.png",
) -> None:
    print("\n=== Experiment 3: Reverberation effect ===")
    uca4 = uniform_circular_array(n_mics=4, radius=0.04)
    angles = np.arange(-150, 151, 30, dtype=float)

    results = {"SRP-PHAT": [], "MUSIC": []}
    for rt60 in rt60_list:
        errs_s, errs_m = [], []
        for ang in angles:
            for seed in range(n_seeds):
                if rt60 == 0.0:
                    signals, _ = simulate_freefield(
                        mic_positions=uca4,
                        azimuth_deg=float(ang),
                        fs=fs,
                        duration=1.0,
                        snr_db=20.0,
                        seed=_seed(seed, ang),
                    )
                else:
                    signals, _ = simulate_room(
                        mic_positions=uca4,
                        azimuth_deg=float(ang),
                        rt60=rt60,
                        fs=fs,
                        duration=1.0,
                        snr_db=20.0,
                        seed=_seed(seed, ang),
                    )
                est_s, _, _ = srp_phat(signals, mic_positions=uca4, fs=fs)
                errs_s.append(_wrap(est_s - ang))

                est_m, _, _ = music(signals, mic_positions=uca4, fs=fs, n_sources=1)
                errs_m.append(_wrap(est_m - ang))
        results["SRP-PHAT"].append(_eval_mae(errs_s))
        results["MUSIC"].append(_eval_mae(errs_m))
        rt_label = "anechoic" if rt60 == 0.0 else f"RT60={rt60:.1f}s"
        print(
            f"  {rt_label:<12}    SRP={_eval_mae(errs_s):6.2f}    "
            f"MUSIC={_eval_mae(errs_m):6.2f}"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rt60_list))
    width = 0.35
    ax.bar(x - width / 2, results["SRP-PHAT"], width=width, label="SRP-PHAT")
    ax.bar(x + width / 2, results["MUSIC"], width=width, label="MUSIC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"RT60={t:.1f}" if t > 0 else "anechoic" for t in rt60_list])
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Reverberation degrades classical SSL methods (UCA4, SNR=20 dB)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    experiment_snr_sweep(snrs=[30.0, 20.0, 10.0, 0.0, -5.0])
    experiment_mic_count(snrs=[20.0, 10.0, 0.0, -5.0, -10.0])
    experiment_reverb(rt60_list=[0.0, 0.2, 0.4, 0.6, 0.9])


if __name__ == "__main__":
    main()
