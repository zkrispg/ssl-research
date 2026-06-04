"""Path C / 2x2 dissociation: formal mixed-effects ANOVA.

Tests the (modality x architecture x prior) interaction in a single statistic.
The dependent variables are the per-seed test metrics (DOAE_CD, F1, SELD,
RDE, Dist_err); the independent variables are:

  modality:     MIC / FOA           (within-subjects via paired ckpts)
  architecture: CRNN / Xfm          (within-subjects via paired ckpts)
  prior:        gca_full / no_geom  (within-subjects via paired ckpts)

Subjects are seed indices.

We fit two models with statsmodels OLS:
  1. Within-cell paired test: prior contribution per cell (already in 2x2.json)
  2. Across-cell ANOVA: 2x2x2 between (modality, arch, prior) with seed as
     repeated measure. Tests the 3-way interaction (i.e. whether the
     prior's effect direction depends on (modality, arch)).

Note: the 4 cells use partially overlapping seed sets, so we use a
mixed-effects ANOVA with seed as random effect (not a strict balanced
within-subjects design).

Outputs:
    D:\\ssl-research\\paper\\path_c_2x2_anova.json
    D:\\ssl-research\\paper\\path_c_2x2_anova.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
from _path_c_analyze import parse_log  # type: ignore

LOG_DIRS = [
    Path(r"D:\ssl-research\runs"),
    Path(r"D:\ssl-research\week11_starss23\runs"),
]
OUT_PATH = Path(r"D:\ssl-research\paper")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# (cell_key, modality, arch, prior, task_id, job_pattern, seeds)
CELLS = [
    # MIC + CRNN
    ("MIC_CRNN_full",   "MIC", "CRNN", "full",   "110", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("MIC_CRNN_nogeom", "MIC", "CRNN", "nogeom", "111", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    # FOA + CRNN
    ("FOA_CRNN_full",   "FOA", "CRNN", "full",   "130", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("FOA_CRNN_nogeom", "FOA", "CRNN", "nogeom", "131", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    # MIC + Xfm
    ("MIC_XFM_full",    "MIC", "Xfm",  "full",   "141", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("MIC_XFM_nogeom",  "MIC", "Xfm",  "nogeom", "142", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    # FOA + Xfm (Tier VIII)
    ("FOA_XFM_full",    "FOA", "Xfm",  "full",   "151", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("FOA_XFM_nogeom",  "FOA", "Xfm",  "nogeom", "152", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    # MIC + Conformer (journal Tranche 2)
    ("MIC_CONF_full",   "MIC", "Conformer", "full",   "161", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("MIC_CONF_nogeom", "MIC", "Conformer", "nogeom", "162", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    # FOA + Conformer (journal Tranche 2)
    ("FOA_CONF_full",   "FOA", "Conformer", "full",   "171", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
    ("FOA_CONF_nogeom", "FOA", "Conformer", "nogeom", "172", "ablate_seed{seed}", [0, 1, 2, 3, 4]),
]


def _resolve_log(name: str) -> Path:
    for d in LOG_DIRS:
        p = d / name
        if p.is_file(): return p
    return LOG_DIRS[0] / name


def collect_long_format() -> pd.DataFrame:
    """Build long-format DataFrame: one row per (ckpt) with columns
    cell_key, modality, arch, prior, seed, F1, LE, DE, RDE, SELD."""
    rows = []
    for cell_key, mod, arch, prior, task, jobpat, seeds in CELLS:
        for seed in seeds:
            log = _resolve_log(f"dcase2024_{task}_{jobpat.format(seed=seed)}_test.log")
            m = parse_log(str(log))
            if m is None: continue
            rows.append({
                "cell_key":  cell_key, "modality": mod, "arch": arch, "prior": prior,
                "task": task, "seed": seed,
                "F1": m["F1"], "LE": m["LE"], "DE": m["DE"], "RDE": m["RDE"], "SELD": m["SELD"],
            })
    return pd.DataFrame(rows)


def fit_anova(df: pd.DataFrame, dv: str) -> dict:
    """Fit OLS model with full 2x2x2 factorial + seed random effect."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    sub = df[["modality", "arch", "prior", "seed", dv]].copy()
    # Use treatment coding with reference levels: MIC, CRNN, nogeom.
    formula = f"{dv} ~ C(modality, Treatment('MIC')) * C(arch, Treatment('CRNN')) * C(prior, Treatment('nogeom'))"
    try:
        model = ols(formula, data=sub).fit()
    except Exception as e:
        return {"error": str(e)}
    aov = anova_lm(model, typ=2)
    return {
        "n_obs":      int(model.nobs),
        "rsquared":   float(model.rsquared),
        "aov":        aov.reset_index().to_dict(orient="records"),
        "coef":       {k: float(v) for k, v in model.params.items()},
        "p_values":   {k: float(v) for k, v in model.pvalues.items()},
    }


def fit_paired_within_arch_modality(df: pd.DataFrame, dv: str, arch: str, modality: str) -> dict:
    """For one (arch, modality) cell, paired-test on seed-matched (full vs nogeom)."""
    from scipy import stats
    sub = df[(df["arch"] == arch) & (df["modality"] == modality)]
    full = sub[sub["prior"] == "full"].set_index("seed")[dv]
    nogm = sub[sub["prior"] == "nogeom"].set_index("seed")[dv]
    shared = sorted(set(full.index) & set(nogm.index))
    if len(shared) < 2: return {"n": len(shared), "note": "insufficient pairs"}
    a = full.loc[shared].to_numpy(); b = nogm.loc[shared].to_numpy()
    delta = a - b
    t_stat, p_t = stats.ttest_rel(a, b)
    s = delta.std(ddof=1)
    dz = float(delta.mean() / s) if s > 0 else float("nan")
    return {"n": len(shared), "shared_seeds": shared, "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=1)), "t": float(t_stat), "p_t": float(p_t),
            "cohens_dz": dz}


def main() -> int:
    df = collect_long_format()
    print(f"[info] long-format DataFrame: {len(df)} rows")
    print(df.groupby(['modality', 'arch', 'prior']).size().to_string())

    # Save raw long-format data
    long_csv = OUT_PATH / "path_c_2x2_anova_long.csv"
    df.to_csv(long_csv, index=False)
    print(f"[saved] {long_csv}")

    payload = {"n_rows_total": int(len(df)),
               "anovas": {}, "within_cell_pairs": {}}
    for dv in ("F1", "LE", "DE", "RDE", "SELD"):
        a = fit_anova(df, dv)
        payload["anovas"][dv] = a
        print(f"\n--- ANOVA on {dv} ---")
        if "aov" in a:
            for row in a["aov"]:
                print(f"  {row.get('index', row.get('source', '?'))}: F={row.get('F', float('nan')):.2f}, "
                      f"PR(>F)={row.get('PR(>F)', float('nan')):.4f}")

        # Within-cell paired contrasts
        cells = []
        for mod in ("MIC", "FOA"):
            for arch in ("CRNN", "Xfm", "Conformer"):
                r = fit_paired_within_arch_modality(df, dv, arch, mod)
                r["modality"] = mod; r["arch"] = arch
                cells.append(r)
        payload["within_cell_pairs"][dv] = cells

    out_json = OUT_PATH / "path_c_2x2_anova.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # Markdown report
    L = ["# Path C / 2x2 ANOVA: formal interaction test for (modality, arch, prior)",
         "",
         "OLS factorial ANOVA on per-seed metrics with three between-cell",
         "categorical factors (modality, arch, prior) and seed as the unit of",
         "replication. Type-II SS. Reference levels: modality=MIC, arch=CRNN, prior=nogeom.",
         "",
         "Note: cells have 3-5 seeds each (unbalanced). The key statistic is",
         "the **3-way interaction term** modality:arch:prior, which tests whether",
         "the prior's effect direction depends on (modality, arch).",
         ""]
    for dv in ("LE", "F1", "SELD", "RDE", "DE"):
        a = payload["anovas"][dv]
        L.append(f"## {dv} (DOAE_CD)" if dv == "LE" else f"## {dv}")
        if "error" in a:
            L.append(f"- error: {a['error']}"); L.append(""); continue
        L.append(f"- n_obs = {a['n_obs']}, R^2 = {a['rsquared']:.3f}")
        L.append("")
        L.append("| factor | F | PR(>F) |")
        L.append("| ------ | - | ------ |")
        for row in a["aov"]:
            f = row.get("F"); p = row.get("PR(>F)")
            name = row.get("index", row.get("source", "?"))
            if f is None or (isinstance(f, float) and np.isnan(f)):
                continue
            star = ""
            if p is not None and not np.isnan(p):
                if p < 0.001: star = " ***"
                elif p < 0.01: star = " **"
                elif p < 0.05: star = " *"
                elif p < 0.1: star = " ."
            L.append(f"| {name} | {f:.2f} | {p:.4f}{star} |")
        L.append("")

        L.append(f"### Within-cell paired contrasts (full - nogeom) on {dv}")
        L.append("| cell | n | delta mean | t (p_t) | d_z |")
        L.append("| ---- | - | ---------- | ------- | --- |")
        for r in payload["within_cell_pairs"][dv]:
            label = f"{r.get('modality')}+{r.get('arch')}"
            if "delta_mean" not in r:
                L.append(f"| {label} | {r.get('n', 0)} | n/a | - | - |"); continue
            L.append(f"| {label} | {r['n']} | {r['delta_mean']:+.3f} \u00b1 {r['delta_std']:.3f} | "
                     f"t={r['t']:+.2f} (p={r['p_t']:.3f}) | {r['cohens_dz']:+.2f} |")
        L.append("")

    out_md = OUT_PATH / "path_c_2x2_anova.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
