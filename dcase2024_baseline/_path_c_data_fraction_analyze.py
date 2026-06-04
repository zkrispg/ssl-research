"""Path C / Tier V (D): analyze the training-data fraction sweep.

Reads per-seed DCASE 2024 test_logs for tasks 110/112 (100%, from Stage 3,
job=ablate_seed{seed}) and 120/121/122/123 (50%/25%, job=frac_seed{seed}).

For each fraction f in {0.25, 0.50, 1.00}, computes:
  - per-cell mean F1, DOAE_CD, SELD score (across 3 or 5 seeds)
  - paired delta (GCA full - no GCA) per seed
  - paired t-test on delta, Cohen's d_z, bootstrap CI

Outputs:
    D:\\ssl-research\\paper\\path_c_data_fraction.json
    D:\\ssl-research\\paper\\path_c_data_fraction.md
    D:\\ssl-research\\paper\\figs\\path_c_data_fraction_F1.png
    D:\\ssl-research\\paper\\figs\\path_c_data_fraction_delta.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
from _path_c_analyze import parse_log, _read_text_any_encoding  # type: ignore

LOG_DIR  = Path(r"D:\ssl-research\runs")
LEGACY_LOG_DIRS = [Path(r"D:\ssl-research\week11_starss23\runs")]


def _resolve_log(name: str) -> Path:
    """Search the primary LOG_DIR first, then any legacy locations."""
    cand = LOG_DIR / name
    if cand.is_file():
        return cand
    for d in LEGACY_LOG_DIRS:
        c2 = d / name
        if c2.is_file():
            return c2
    return cand  # fall back to default (parse_log will treat as missing)
OUT_PATH = Path(r"D:\ssl-research\paper")
FIG_DIR  = OUT_PATH / "figs"
OUT_PATH.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fraction -> {gca_task, no_gca_task, job_pattern, seeds}
# 25%/50% had supplemental seeds 3, 4 added in the TASLP rush (Tier V D suppl.)
# so all three rows now potentially have n=5 (subject to test_log presence).
FRACTIONS = [
    {"frac": 0.25, "gca": "122", "no_gca": "123", "job": "frac_seed{seed}",   "seeds": [0, 1, 2, 3, 4]},
    {"frac": 0.50, "gca": "120", "no_gca": "121", "job": "frac_seed{seed}",   "seeds": [0, 1, 2, 3, 4]},
    {"frac": 1.00, "gca": "110", "no_gca": "112", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
]


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    if len(a) < 2: return float("nan")
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def bootstrap_ci(deltas, n_boot=5000, alpha=0.05, rng_seed=0):
    if len(deltas) < 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(deltas, dtype=np.float64)
    bm = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(bm, 100*alpha/2)),
            float(np.percentile(bm, 100*(1-alpha/2))))


def collect() -> dict:
    rows = []
    for spec in FRACTIONS:
        f = spec["frac"]
        for seed in spec["seeds"]:
            job = spec["job"].format(seed=seed)
            for cell_kind, cell_task in (("gca", spec["gca"]), ("no_gca", spec["no_gca"])):
                log = _resolve_log(f"dcase2024_{cell_task}_{job}_test.log")
                m = parse_log(str(log))
                rows.append({
                    "fraction":  f,
                    "task":      cell_task,
                    "job":       job,
                    "seed":      seed,
                    "cell_kind": cell_kind,
                    "log":       str(log),
                    "log_exists": log.is_file(),
                    "metrics":   m,
                })
    return {"rows": rows}


def aggregate(payload: dict) -> dict:
    """Aggregate per-fraction × cell_kind, plus paired delta(gca - no_gca)."""
    rows = payload["rows"]
    out = {"per_fraction": {}, "summary": []}

    for spec in FRACTIONS:
        f = spec["frac"]
        f_key = f"frac_{int(f*100)}pct"
        # collect for this fraction
        gca_seeds, no_seeds = {}, {}
        for r in rows:
            if r["fraction"] != f or r["metrics"] is None:
                continue
            target = gca_seeds if r["cell_kind"] == "gca" else no_seeds
            target[r["seed"]] = r["metrics"]
        shared = sorted(set(gca_seeds.keys()) & set(no_seeds.keys()))
        # per-metric aggregate
        agg = {"n_seeds_gca":   len(gca_seeds),
               "n_seeds_no_gca": len(no_seeds),
               "shared_seeds":  shared}
        for metric in ("F1", "LE", "DE", "RDE", "SELD"):
            gca_vals = np.array([gca_seeds[s][metric] for s in gca_seeds.keys()])
            no_vals  = np.array([no_seeds[s][metric] for s in no_seeds.keys()])
            agg[metric] = {
                "gca_mean": float(np.mean(gca_vals)) if len(gca_vals) else None,
                "gca_std":  float(np.std(gca_vals, ddof=1)) if len(gca_vals) > 1 else 0.0,
                "no_gca_mean": float(np.mean(no_vals)) if len(no_vals) else None,
                "no_gca_std":  float(np.std(no_vals, ddof=1)) if len(no_vals) > 1 else 0.0,
            }
            # paired delta on shared seeds
            if len(shared) >= 2:
                d = np.array([gca_seeds[s][metric] - no_seeds[s][metric] for s in shared])
                t_stat, p_t = stats.ttest_rel(
                    np.array([gca_seeds[s][metric] for s in shared]),
                    np.array([no_seeds[s][metric]  for s in shared]),
                )
                try:
                    w_stat, p_w = stats.wilcoxon(
                        np.array([gca_seeds[s][metric] for s in shared]),
                        np.array([no_seeds[s][metric]  for s in shared]),
                    )
                except Exception:
                    w_stat, p_w = float("nan"), float("nan")
                agg[metric]["delta_per_seed"] = d.tolist()
                agg[metric]["delta_mean"]     = float(d.mean())
                agg[metric]["delta_std"]      = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                agg[metric]["t_stat"]         = float(t_stat)
                agg[metric]["p_t"]            = float(p_t)
                agg[metric]["p_w"]            = float(p_w)
                agg[metric]["cohens_dz"]      = cohens_dz(d.tolist())
                agg[metric]["boot_ci_95"]     = list(bootstrap_ci(d.tolist()))
        out["per_fraction"][f_key] = agg
        out["summary"].append({
            "frac": f,
            "n_pairs": len(shared),
            "F1_delta": agg.get("F1", {}).get("delta_mean"),
            "SELD_delta": agg.get("SELD", {}).get("delta_mean"),
            "DOAE_delta": agg.get("LE", {}).get("delta_mean"),
        })
    return out


def write_md(payload: dict, agg: dict, path: Path) -> None:
    L = ["# Path C / data-fraction sweep: when does the geometry prior help?",
         "",
         "Compares 110 (GCA full, geometry prior) vs 112 (no GCA, matched control)",
         "trained on subsets of STARSS23 dev-train at three fractions:",
         "100% (existing Stage 3), 50% (tasks 120/121), 25% (tasks 122/123).",
         "",
         "Hypothesis test: at low data, the geometry prior should regularize and",
         "*help* the model; at high data, the prior should over-constrain and *hurt*",
         "the model. A positive *interaction* between data-fraction and prior would",
         "manifest as a non-zero slope in delta = (GCA - no-GCA) vs fraction.",
         "",
         "## Per-fraction summary"]
    L.append("| fraction | n pairs | F1 GCA (%) | F1 no-GCA (%) | delta F1 (pp) | t | p_t | d_z |")
    L.append("| -------- | ------- | ---------- | ------------- | ------------- | - | --- | --- |")
    for spec in FRACTIONS:
        f = spec["frac"]
        f_key = f"frac_{int(f*100)}pct"
        a = agg["per_fraction"].get(f_key, {})
        if not a:
            L.append(f"| {int(f*100)}% | 0 | n/a | n/a | n/a | - | - | - |"); continue
        nm = len(a.get("shared_seeds", []))
        f1 = a.get("F1", {})
        gm = f1.get("gca_mean");      gs = f1.get("gca_std", 0.0)
        nm_v = f1.get("no_gca_mean"); ns = f1.get("no_gca_std", 0.0)
        dm = f1.get("delta_mean");    t = f1.get("t_stat"); pt = f1.get("p_t"); dz = f1.get("cohens_dz")
        if gm is None:
            L.append(f"| {int(f*100)}% | {nm} | n/a | n/a | n/a | - | - | - |"); continue
        L.append(f"| {int(f*100)}% | {nm} | {gm:.2f} +/- {gs:.2f} | {nm_v:.2f} +/- {ns:.2f} | "
                 f"{dm:+.2f}{' (?)' if dm is None else ''} | {t:+.2f} | {pt:.4f} | {dz:+.2f} |")
    L.append("")

    L.append("## Per-fraction SELD score (lower is better)")
    L.append("| fraction | n pairs | SELD GCA | SELD no-GCA | delta SELD | t | p_t | d_z |")
    L.append("| -------- | ------- | -------- | ----------- | ---------- | - | --- | --- |")
    for spec in FRACTIONS:
        f = spec["frac"]
        f_key = f"frac_{int(f*100)}pct"
        a = agg["per_fraction"].get(f_key, {})
        if not a:
            L.append(f"| {int(f*100)}% | 0 | n/a | n/a | n/a | - | - | - |"); continue
        nm = len(a.get("shared_seeds", []))
        s = a.get("SELD", {})
        gm = s.get("gca_mean"); gs = s.get("gca_std", 0.0)
        nm_v = s.get("no_gca_mean"); ns = s.get("no_gca_std", 0.0)
        dm = s.get("delta_mean"); t = s.get("t_stat"); pt = s.get("p_t"); dz = s.get("cohens_dz")
        if gm is None:
            L.append(f"| {int(f*100)}% | {nm} | n/a | n/a | n/a | - | - | - |"); continue
        L.append(f"| {int(f*100)}% | {nm} | {gm:.3f} +/- {gs:.3f} | {nm_v:.3f} +/- {ns:.3f} | "
                 f"{dm:+.3f} | {t:+.2f} | {pt:.4f} | {dz:+.2f} |")
    L.append("")

    L.append("## Interpretation")
    L.append("- **Positive delta F1** at fraction f means the geometry prior helps at f.")
    L.append("- **Negative delta F1** means the geometry prior hurts at f.")
    L.append("- A monotonically increasing delta as fraction decreases would support the")
    L.append("  'prior helps when data is scarce' hypothesis (expected behavior of an")
    L.append("  inductive bias acting as regularizer).")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def plot(agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = [spec["frac"] for spec in FRACTIONS]
    f1_gca, f1_gca_std, f1_no, f1_no_std = [], [], [], []
    seld_gca, seld_no = [], []
    delta_f1, delta_f1_ci_lo, delta_f1_ci_hi, delta_p = [], [], [], []
    delta_seld = []
    for spec in FRACTIONS:
        f_key = f"frac_{int(spec['frac']*100)}pct"
        a = agg["per_fraction"].get(f_key, {})
        f1 = a.get("F1", {})
        s  = a.get("SELD", {})
        f1_gca.append(f1.get("gca_mean", np.nan)); f1_gca_std.append(f1.get("gca_std", 0.0))
        f1_no.append(f1.get("no_gca_mean", np.nan)); f1_no_std.append(f1.get("no_gca_std", 0.0))
        seld_gca.append(s.get("gca_mean", np.nan));  seld_no.append(s.get("no_gca_mean", np.nan))
        d_mean = f1.get("delta_mean", np.nan)
        ci = f1.get("boot_ci_95", [np.nan, np.nan])
        delta_f1.append(d_mean); delta_f1_ci_lo.append(ci[0]); delta_f1_ci_hi.append(ci[1])
        delta_p.append(f1.get("p_t", np.nan))
        delta_seld.append(s.get("delta_mean", np.nan))

    # F1 vs fraction
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fracs_pct = [100*f for f in fracs]
    f1_gca = np.array(f1_gca, dtype=np.float64); f1_no = np.array(f1_no, dtype=np.float64)
    f1_gca_std = np.array(f1_gca_std); f1_no_std = np.array(f1_no_std)
    axes[0].errorbar(fracs_pct, f1_gca, yerr=f1_gca_std, marker="o", color="tab:red",   capsize=4, label="GCA full")
    axes[0].errorbar(fracs_pct, f1_no,  yerr=f1_no_std,  marker="s", color="tab:gray",  capsize=4, label="no GCA")
    axes[0].set_xlabel("training data fraction (%)")
    axes[0].set_ylabel("F 20° (%)")
    axes[0].set_title("F1 vs training-data fraction (STARSS23 dev-test)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # Delta F1 with CI
    delta_f1 = np.array(delta_f1, dtype=np.float64)
    lo = np.array(delta_f1_ci_lo, dtype=np.float64); hi = np.array(delta_f1_ci_hi, dtype=np.float64)
    yerr = np.stack([delta_f1 - lo, hi - delta_f1])
    axes[1].errorbar(fracs_pct, delta_f1, yerr=yerr, marker="D", color="tab:blue", capsize=4)
    axes[1].axhline(0, color="black", linestyle=":", lw=0.8)
    axes[1].set_xlabel("training data fraction (%)")
    axes[1].set_ylabel("delta F1 (GCA - no GCA, pp)")
    axes[1].set_title("Geometry prior contribution vs data fraction (95% bootstrap CI)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "path_c_data_fraction_F1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")

    # SELD score vs fraction (similar plot)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    seld_gca = np.array(seld_gca, dtype=np.float64); seld_no = np.array(seld_no, dtype=np.float64)
    ax.plot(fracs_pct, seld_gca, marker="o", color="tab:red", label="GCA full")
    ax.plot(fracs_pct, seld_no,  marker="s", color="tab:gray", label="no GCA")
    ax.set_xlabel("training data fraction (%)")
    ax.set_ylabel("SELD score (lower is better)")
    ax.set_title("SELD score vs training-data fraction")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "path_c_data_fraction_SELD.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> int:
    payload = collect()
    agg = aggregate(payload)
    out_json = OUT_PATH / "path_c_data_fraction.json"
    out_json.write_text(json.dumps({"raw": payload, "agg": agg}, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {out_json}")
    write_md(payload, agg, OUT_PATH / "path_c_data_fraction.md")
    print(f"[saved] {OUT_PATH / 'path_c_data_fraction.md'}")
    try:
        plot(agg)
    except Exception as e:
        print(f"[warn] plotting failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
