"""Multi-seed DCASE evaluation + paired t-test for W6 vs W9 ``no_geom``.

The script discovers all checkpoints of the form

    week06_method/checkpoints/best_full[_seed<N>].pt        -> W6 seeds
    week09_geometry_attn/checkpoints/best_no_geom[_seed<N>].pt -> W9 seeds

Each seed is run on the **identical** 6 grid points used in
``week09_geometry_attn/evaluate.py`` (RT60 = 0, 0.3, 0.6 at SNR = 10; SNR =
20, 0, -10 at RT60 = 0). For every condition we then form **N seeds**
samples per method, compute mean +/- std, and run a paired t-test on the
W9 - W6 differences across the 6 conditions (one test per seed pair).

The output table is exactly the one referenced as ``TODO W10`` in the
paper draft.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps  # paired t-test
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "week02_classical"))
sys.path.insert(0, str(ROOT / "week03_cnn_doa"))
sys.path.insert(0, str(ROOT / "week05_multi_source"))
sys.path.insert(0, str(ROOT / "week06_method"))
sys.path.insert(0, str(ROOT / "week08_dcase"))
sys.path.insert(0, str(ROOT / "week09_geometry_attn"))

from dcase_metrics import DcaseSeldStats, overall_seld_score  # noqa: E402
from features import phase_features  # noqa: E402
from gca_model import GCAMultiTaskCRNN  # noqa: E402
from geometry import uniform_circular_array  # noqa: E402
from multi_baselines import find_peaks_circular  # noqa: E402
from multi_dataset import make_grid  # noqa: E402
from multi_source_data import (  # noqa: E402
    sample_distinct_azimuths,
    simulate_freefield_multi,
    simulate_room_multi,
)
from multi_task_model import MultiTaskCRNN  # noqa: E402

W6_DIR = ROOT / "week06_method" / "checkpoints"
W9_DIR = ROOT / "week09_geometry_attn" / "checkpoints"
TOL_DEG = 20.0

CONDITIONS: list[tuple[str, float, float]] = [
    ("RT60=0.0,SNR=10",  0.0, 10.0),
    ("RT60=0.3,SNR=10",  0.3, 10.0),
    ("RT60=0.6,SNR=10",  0.6, 10.0),
    ("RT60=0.0,SNR=20",  0.0, 20.0),
    ("RT60=0.0,SNR=0",   0.0,  0.0),
    ("RT60=0.0,SNR=-10", 0.0, -10.0),
]


def parse_seed_from_path(p: Path, prefix: str) -> int:
    """``best_full.pt`` -> seed 0; ``best_full_seed3.pt`` -> seed 3."""
    name = p.stem
    if name == prefix:
        return 0
    m = re.match(rf"^{re.escape(prefix)}_seed(\d+)$", name)
    if m is None:
        raise ValueError(f"cannot parse seed from filename {p.name}")
    return int(m.group(1))


def discover_w6_checkpoints() -> dict[int, Path]:
    """Return ``{seed: path}`` for all W6 full-variant checkpoints found."""
    out: dict[int, Path] = {}
    for p in W6_DIR.glob("best_full*.pt"):
        if "resumed" in p.stem or "v2" in p.stem:
            continue
        seed = parse_seed_from_path(p, "best_full")
        out[seed] = p
    return out


def discover_w9_no_geom_checkpoints() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in W9_DIR.glob("best_no_geom*.pt"):
        if "resumed" in p.stem or "v2" in p.stem:
            continue
        seed = parse_seed_from_path(p, "best_no_geom")
        out[seed] = p
    return out


def predict_w6(signals, model, grid, fs, n_fft, hop_length) -> np.ndarray:
    feat = phase_features(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["spectrum"]).mean(dim=1).squeeze(0).cpu().numpy()
        pred_k = int(out["count"].argmax(dim=-1).item()) + 1
    return find_peaks_circular(probs, grid, n_peaks=pred_k,
                               rel_threshold=0.0, min_separation_deg=25.0)


def predict_w9(signals, model, grid, fs, n_fft, hop_length) -> np.ndarray:
    """Identical decoding head to W6."""
    return predict_w6(signals, model, grid, fs, n_fft, hop_length)


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


def evaluate_condition(*, condition_name, rt60, snr_db, n_per_k, mics, grid,
                       models, fs, n_fft, hop_length) -> dict[str, float]:
    """Return mean SELD score (lower=better) per model id on a single grid point."""
    stats = {mid: DcaseSeldStats(tolerance_deg=TOL_DEG) for mid in models}
    rng = np.random.default_rng(hash((condition_name, snr_db, rt60)) & 0xFFFFFFFF)

    total = 3 * n_per_k
    pbar = tqdm(total=total, desc=condition_name, leave=False)
    for k in (1, 2, 3):
        for _ in range(n_per_k):
            local_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
            azs = sample_distinct_azimuths(local_rng, k, min_separation_deg=30.0)
            seed = int(local_rng.integers(0, 2 ** 31))
            signals, _ = gen_test_signals(mics, azs, snr_db, rt60, fs, seed)
            for mid, (kind, model) in models.items():
                if kind == "w6":
                    pred = predict_w6(signals, model, grid, fs, n_fft, hop_length)
                elif kind == "w9":
                    pred = predict_w9(signals, model, grid, fs, n_fft, hop_length)
                else:
                    raise ValueError(kind)
                stats[mid].add_sample(pred, azs)
            pbar.update(1)
    pbar.close()
    return {mid: overall_seld_score(s.summary()) for mid, s in stats.items()}


def main() -> None:
    fs, n_fft, hop_length = 16000, 512, 256
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = make_grid(-180, 180, 5)

    w6_paths = discover_w6_checkpoints()
    w9_paths = discover_w9_no_geom_checkpoints()
    if not w6_paths or not w9_paths:
        raise SystemExit("missing checkpoints; train both W6 and W9 first.")
    print(f"[multi_seed] W6 seeds: {sorted(w6_paths)}", flush=True)
    print(f"[multi_seed] W9 seeds: {sorted(w9_paths)}", flush=True)

    models: dict[str, tuple[str, torch.nn.Module]] = {}
    for s, p in sorted(w6_paths.items()):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=len(grid), max_k=3)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models[f"w6_s{s}"] = ("w6", m)
        print(f"[multi_seed] W6 s{s}  val F1 {ck['val_f1']:.3f}  epoch {ck['epoch']}",
              flush=True)
    for s, p in sorted(w9_paths.items()):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = GCAMultiTaskCRNN(
            mic_positions=mics, n_freq=ck["n_freq"], n_classes=ck["n_classes"],
            max_k=ck["max_k"], geometry_bias=bool(ck.get("geometry_bias", False)),
        )
        m.load_state_dict(ck["model_state"])
        m.eval()
        models[f"w9_s{s}"] = ("w9", m)
        print(f"[multi_seed] W9 s{s}  val F1 {ck['val_f1']:.3f}  epoch {ck['epoch']}",
              flush=True)

    seld_per_model_per_cond: dict[str, list[float]] = {mid: [] for mid in models}
    for name, rt60, snr_db in CONDITIONS:
        scores = evaluate_condition(
            condition_name=name, rt60=rt60, snr_db=snr_db, n_per_k=15,
            mics=mics, grid=grid, models=models,
            fs=fs, n_fft=n_fft, hop_length=hop_length,
        )
        print(f"\n=== {name} ===", flush=True)
        for mid in sorted(scores):
            print(f"  {mid:<8}  SELD = {scores[mid]:.4f}", flush=True)
        for mid, v in scores.items():
            seld_per_model_per_cond[mid].append(v)

    w6_seeds = sorted(w6_paths)
    w9_seeds = sorted(w9_paths)
    w6_matrix = np.array(
        [seld_per_model_per_cond[f"w6_s{s}"] for s in w6_seeds]
    )
    w9_matrix = np.array(
        [seld_per_model_per_cond[f"w9_s{s}"] for s in w9_seeds]
    )
    cond_names = [c[0] for c in CONDITIONS]

    print("\n=== Multi-seed summary (mean +/- std across seeds) ===", flush=True)
    header = f"{'condition':<22}  {'W6 mean':>8}  {'W6 std':>7}  {'W9 mean':>8}  {'W9 std':>7}  {'delta':>7}  {'rel':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rel_changes = []
    for i, name in enumerate(cond_names):
        w6_mean, w6_std = w6_matrix[:, i].mean(), w6_matrix[:, i].std(ddof=1) if len(w6_seeds) > 1 else 0.0
        w9_mean, w9_std = w9_matrix[:, i].mean(), w9_matrix[:, i].std(ddof=1) if len(w9_seeds) > 1 else 0.0
        delta = w9_mean - w6_mean
        rel = delta / w6_mean * 100.0 if w6_mean > 1e-9 else float("nan")
        rel_changes.append(rel)
        print(
            f"{name:<22}  {w6_mean:8.4f}  {w6_std:7.4f}  {w9_mean:8.4f}  "
            f"{w9_std:7.4f}  {delta:+7.4f}  {rel:+6.1f}%",
            flush=True,
        )
    overall_w6 = w6_matrix.mean()
    overall_w9 = w9_matrix.mean()
    print("-" * len(header), flush=True)
    print(
        f"{'overall (6 cond)':<22}  {overall_w6:8.4f}  {' ':>7}  {overall_w9:8.4f}  "
        f"{' ':>7}  {overall_w9 - overall_w6:+7.4f}  {(overall_w9 - overall_w6) / overall_w6 * 100:+6.1f}%",
        flush=True,
    )

    n_w6_seeds = len(w6_seeds)
    n_w9_seeds = len(w9_seeds)
    if n_w6_seeds == n_w9_seeds and n_w6_seeds >= 2:
        print("\n=== Paired t-test (per-condition mean, paired by seed) ===", flush=True)
        for i, name in enumerate(cond_names):
            diffs = w9_matrix[:, i] - w6_matrix[:, i]
            if np.allclose(diffs, diffs[0]):
                print(f"  {name:<22}  diffs constant ({diffs[0]:+.4f}); t-test skipped",
                      flush=True)
                continue
            t, p = sps.ttest_rel(w9_matrix[:, i], w6_matrix[:, i])
            sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
            print(f"  {name:<22}  t={t:+.3f}  p={p:.4f} {sig}", flush=True)

        diff_per_cond_w9_minus_w6 = w9_matrix.mean(axis=0) - w6_matrix.mean(axis=0)
        t, p = sps.ttest_1samp(diff_per_cond_w9_minus_w6, 0.0)
        sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
        print(f"\n  (W9 - W6) mean diff across 6 conditions: "
              f"{diff_per_cond_w9_minus_w6.mean():+.4f}  "
              f"t={t:+.3f}  p={p:.4f} {sig}",
              flush=True)
    else:
        print("\n[multi_seed] Need >=2 paired seeds per model for paired t-test; "
              f"got W6 n={n_w6_seeds} W9 n={n_w9_seeds}", flush=True)

    summary = {
        "conditions": cond_names,
        "w6_seeds": w6_seeds,
        "w9_seeds": w9_seeds,
        "w6_matrix": w6_matrix.tolist(),
        "w9_matrix": w9_matrix.tolist(),
        "overall_w6": float(overall_w6),
        "overall_w9": float(overall_w9),
    }
    out_path = Path(__file__).parent / "multi_seed_summary.json"
    import json
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[multi_seed] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
