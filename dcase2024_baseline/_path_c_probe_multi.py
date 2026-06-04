"""Path C / Tier V (B): linear probing on MULTI-SOURCE frames.

Companion to `_path_c_probe.py` which only used frames with EXACTLY ONE
active source. Here we probe frames with EXACTLY TWO active sources to
test whether the geometry prior also affects how multi-source spatial
information is encoded in the conv stack.

Method
------
For each ckpt:
  1. Same hook + extraction as the single-source probe (post-conv pooled features
     of shape (T_label_total, D=2*C_red)).
  2. For each frame with EXACTLY TWO sources, sort the two ground-truth
     (azimuth, elevation) pairs by ASCENDING azimuth (a canonical ordering)
     and concatenate to form an 8-d sin/cos target:
        [sin az0, cos az0, sin el0, cos el0,
         sin az1, cos az1, sin el1, cos el1]
  3. Fit a Ridge regressor (alpha=1, standardized X) per fold (5-fold CV
     across files), predicting 8 outputs jointly.
  4. Evaluate with HUNGARIAN-matched angular error (assignment between the
     two predicted (az, el) pairs and the two ground-truth pairs that
     minimises total angular error). Report mean angular MAE in degrees.

Outputs:
    D:\\ssl-research\\paper\\path_c_probe_multi.json
    D:\\ssl-research\\paper\\path_c_probe_multi.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
import parameters
import seldnet_model
# Reuse helpers from the single-source probe so we don't duplicate forward logic.
from _path_c_probe import (
    list_dev_test_files,
    extract_post_conv_features,
    build_and_load,
    META_DIR,
    FEAT_DIR,
)

OUT_PATH = Path(r"D:\ssl-research\paper")
OUT_PATH.mkdir(parents=True, exist_ok=True)

CELLS = [
    {"task": "110", "name": "110_gca_full"},
    {"task": "111", "name": "111_gca_nogeom"},
    {"task": "112", "name": "112_no_gca"},
    {"task": "113", "name": "113_vanilla_se"},
]
SEEDS = [0, 1, 2, 3, 4]


def load_two_source_targets(stem: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y8, mask) where mask[t]=True if frame t has EXACTLY 2 active
    sources. y8[t] is 8-d: two (sin az, cos az, sin el, cos el) tuples sorted
    by ascending GT azimuth.
    """
    y8 = np.zeros((n_frames, 8), dtype=np.float32)
    mask = np.zeros(n_frames, dtype=bool)
    counts: dict[int, list[tuple[float, float]]] = {}
    sub_dirs = ("dev-test-sony", "dev-test-tau")
    for sub in sub_dirs:
        cand = META_DIR / sub / f"{stem}.csv"
        if cand.is_file():
            csvpath = cand; break
    else:
        return y8, mask
    with open(csvpath, "r") as fh:
        for row in csv.reader(fh):
            try:
                fr = int(row[0]); az = float(row[3]); el = float(row[4])
            except (ValueError, IndexError):
                continue
            counts.setdefault(fr, []).append((az, el))
    for fr, items in counts.items():
        if not (0 <= fr < n_frames): continue
        if len(items) != 2: continue
        items_sorted = sorted(items, key=lambda x: x[0])  # by ascending azimuth
        for j, (az, el) in enumerate(items_sorted):
            azr, elr = np.deg2rad(az), np.deg2rad(el)
            y8[fr, 4 * j + 0] = np.sin(azr)
            y8[fr, 4 * j + 1] = np.cos(azr)
            y8[fr, 4 * j + 2] = np.sin(elr)
            y8[fr, 4 * j + 3] = np.cos(elr)
        mask[fr] = True
    return y8, mask


def angular_mae_hungarian(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-row Hungarian-matched mean angular error in degrees.
    Both pred and gt are (N, 8): two 4-d (sin az, cos az, sin el, cos el) tuples.
    """
    N = pred.shape[0]
    out = np.zeros(N, dtype=np.float64)
    # Normalize sin/cos
    def norm_pair(a):
        s, c = a[..., 0], a[..., 1]
        n = np.sqrt(s ** 2 + c ** 2 + 1e-9)
        return s / n, c / n

    for n in range(N):
        # pred has 2 pred sources at idx 0 and 1 (each 4-d)
        # gt has 2 GT sources at idx 0 and 1 (each 4-d)
        cost = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            p_az_s, p_az_c = norm_pair(pred[n, 4 * i + 0:4 * i + 2])
            p_el_s, p_el_c = norm_pair(pred[n, 4 * i + 2:4 * i + 4])
            ang_p_az = np.arctan2(p_az_s, p_az_c)
            ang_p_el = np.arctan2(p_el_s, p_el_c)
            for j in range(2):
                g_az = np.arctan2(gt[n, 4 * j + 0], gt[n, 4 * j + 1])
                g_el = np.arctan2(gt[n, 4 * j + 2], gt[n, 4 * j + 3])
                d_az = np.angle(np.exp(1j * (ang_p_az - g_az)))
                d_el = np.angle(np.exp(1j * (ang_p_el - g_el)))
                # mean of |az-error| and |el-error| in deg, treated as scalar cost
                cost[i, j] = np.rad2deg(np.abs(d_az)) + np.rad2deg(np.abs(d_el))
        # Hungarian for 2x2: just compare two assignments
        diag = cost[0, 0] + cost[1, 1]
        anti = cost[0, 1] + cost[1, 0]
        if diag <= anti:
            err = (cost[0, 0] + cost[1, 1]) / 4.0  # / (2 sources * 2 angles)
        else:
            err = (cost[0, 1] + cost[1, 0]) / 4.0
        out[n] = err
    return out


def fit_probe_kfold_multi(X: np.ndarray, y8: np.ndarray, file_index: np.ndarray, k: int = 5) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    unique_files = np.unique(file_index)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_files)
    folds = np.array_split(unique_files, k)
    fold_maes = []
    for fold_i, val_files in enumerate(folds):
        val_mask = np.isin(file_index, val_files)
        Xt, yt = X[~val_mask], y8[~val_mask]
        Xv, yv = X[val_mask],  y8[val_mask]
        if len(Xt) < 32 or len(Xv) == 0:
            continue
        sc = StandardScaler().fit(Xt)
        reg = Ridge(alpha=1.0).fit(sc.transform(Xt), yt)
        pred = reg.predict(sc.transform(Xv))
        errs = angular_mae_hungarian(pred, yv)
        fold_maes.append(float(errs.mean()))
    if not fold_maes:
        return {"mae_per_fold": [], "mae_mean": float("nan"), "mae_std": 0.0,
                "note": "no usable folds (insufficient train/val)"}
    return {
        "mae_per_fold": fold_maes,
        "mae_mean":     float(np.mean(fold_maes)),
        "mae_std":      float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0,
    }


def probe_one_ckpt(task_id: str, seed: int, file_stems: list[str], device) -> dict | None:
    p = parameters.get_params(task_id).copy()
    try:
        model = build_and_load(task_id, seed).to(device)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    Xs, ys, fis = [], [], []
    n_frames_total = 0; n_frames_used = 0
    for fi, stem in enumerate(file_stems):
        npy = FEAT_DIR / f"{stem}.npy"
        feat_post = extract_post_conv_features(model, npy, p, device)  # (T_label_total, D)
        T_label = feat_post.shape[0]
        y8, mask = load_two_source_targets(stem, T_label)
        n_frames_total += T_label
        n_frames_used  += int(mask.sum())
        if mask.sum() == 0: continue
        Xf = feat_post[mask]; yf = y8[mask]
        Xs.append(Xf); ys.append(yf); fis.append(np.full(Xf.shape[0], fi, dtype=np.int32))
    del model; torch.cuda.empty_cache()
    if not Xs:
        return {"note": "no two-source frames found", "n_frames_total": n_frames_total}
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0); fi = np.concatenate(fis)
    print(f"    probe X: {X.shape}, y: {y.shape}, files used: {len(np.unique(fi))}, "
          f"2-src frames: {n_frames_used}/{n_frames_total}")
    res = fit_probe_kfold_multi(X, y, fi, k=5)
    res["n_frames_total"] = n_frames_total
    res["n_frames_two_src"] = n_frames_used
    return res


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) < 2: return float("nan")
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def aggregate(per_ckpt: dict) -> dict:
    cells: dict[str, dict[int, float]] = {c["name"]: {} for c in CELLS}
    for cell in CELLS:
        for s in SEEDS:
            key = f"{cell['task']}_seed{s}"
            if key in per_ckpt:
                v = per_ckpt[key].get("mae_mean")
                if v is not None and not np.isnan(v):
                    cells[cell["name"]][s] = v
    summary = {}
    for cname, by_seed in cells.items():
        vals = list(by_seed.values())
        summary[cname] = {
            "n_seeds":    len(vals),
            "mae_per_seed": by_seed,
            "mae_mean":   float(np.mean(vals)) if vals else None,
            "mae_std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    contrasts = {}
    pair_specs = [
        ("110_gca_full",   "112_no_gca",     "GCA full vs no-GCA (multi-src probing)"),
        ("110_gca_full",   "111_gca_nogeom", "GCA full vs no_geom (multi-src probing)"),
        ("111_gca_nogeom", "112_no_gca",     "no_geom GCA vs no-GCA (multi-src probing)"),
        ("113_vanilla_se", "112_no_gca",     "Vanilla SE vs no-GCA (multi-src probing)"),
        ("113_vanilla_se", "111_gca_nogeom", "Vanilla SE vs GCA no_geom (multi-src probing)"),
        ("110_gca_full",   "113_vanilla_se", "GCA full vs Vanilla SE (multi-src probing)"),
    ]
    for a, b, descr in pair_specs:
        shared = sorted(set(cells[a].keys()) & set(cells[b].keys()))
        if len(shared) < 2:
            contrasts[f"{a}__vs__{b}"] = {"n": len(shared), "note": "insufficient pairs"}; continue
        a_vals = np.array([cells[a][s] for s in shared])
        b_vals = np.array([cells[b][s] for s in shared])
        deltas = (a_vals - b_vals).tolist()
        t_stat, p_t = stats.ttest_rel(a_vals, b_vals)
        try:
            w_stat, p_w = stats.wilcoxon(a_vals, b_vals)
        except Exception:
            w_stat, p_w = float("nan"), float("nan")
        contrasts[f"{a}__vs__{b}"] = {
            "description":     descr,
            "n":               len(shared),
            "a_per_seed":      a_vals.tolist(),
            "b_per_seed":      b_vals.tolist(),
            "delta_per_seed":  deltas,
            "delta_mean":      float(np.mean(deltas)),
            "delta_std":       float(np.std(deltas, ddof=1)),
            "ttest_rel":       {"t": float(t_stat), "p": float(p_t)},
            "wilcoxon":        {"w": float(w_stat), "p": float(p_w)},
            "cohens_dz":       cohens_dz(deltas),
        }
    return {"per_cell": summary, "contrasts": contrasts}


def write_md(payload: dict, path: Path) -> None:
    L = ["# Path C / multi-source linear probing",
         "",
         "Probe target: 2 sources -> 8-d sin/cos vector (sorted by GT azimuth).",
         "Eval: Hungarian-matched mean angular error (deg).",
         ""]
    L.append("## Per-cell")
    L.append("| cell | n | MAE mean (deg) | MAE std (deg) |")
    L.append("| ---- | - | -------------- | ------------- |")
    for cname, c in payload["per_cell"].items():
        m = c["mae_mean"]
        if m is None:
            L.append(f"| {cname} | {c['n_seeds']} | n/a | n/a |"); continue
        L.append(f"| {cname} | {c['n_seeds']} | {m:.2f} | {c['mae_std']:.2f} |")
    L += ["", "## Paired contrasts"]
    for cname, c in payload["contrasts"].items():
        L.append(f"### {cname}")
        if c.get("n", 0) < 2:
            L.append("- insufficient pairs"); L.append(""); continue
        L.append(f"- _{c['description']}_")
        L.append(f"- n={c['n']}, mean delta = {c['delta_mean']:+.2f} +/- {c['delta_std']:.2f} deg")
        L.append(f"- t = {c['ttest_rel']['t']:+.2f} (p_t = {c['ttest_rel']['p']:.4f}, p_w = {c['wilcoxon']['p']:.4f})")
        L.append(f"- Cohen's d_z = {c['cohens_dz']:+.2f}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    cells = args.cells or [c["task"] for c in CELLS]
    seeds = args.seeds or SEEDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_stems = list_dev_test_files()
    if args.max_files:
        file_stems = file_stems[: args.max_files]
    print(f"using {len(file_stems)} dev-test files")

    per_ckpt: dict[str, dict] = {}
    for task_id in cells:
        for s in seeds:
            print(f"\n=== {task_id} seed {s} ===")
            r = probe_one_ckpt(task_id, s, file_stems, device)
            if r is None: continue
            per_ckpt[f"{task_id}_seed{s}"] = r
            if "mae_mean" in r:
                print(f"    multi-src MAE = {r['mae_mean']:.2f} +/- {r['mae_std']:.2f} deg")

    payload = aggregate(per_ckpt)
    payload["per_ckpt"] = per_ckpt
    out_json = OUT_PATH / "path_c_probe_multi.json"
    out_md   = OUT_PATH / "path_c_probe_multi.md"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, out_md)
    print(f"\n[saved] {out_json}\n[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
