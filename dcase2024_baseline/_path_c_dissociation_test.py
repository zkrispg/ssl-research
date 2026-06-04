"""Path C / P1-3: formal test of the direction-vs-distance double dissociation.

The paper claims a double dissociation:
  * DIRECTION (DOAE_CD / LE) is modulated by ARCHITECTURE (not modality);
  * DISTANCE  (RDE)          is modulated by MODALITY     (not architecture).

A double dissociation is, statistically, a pair of crossed interactions. We test
each interaction as a within-seed paired SECOND DIFFERENCE (the seeds 0-4 are
matched across cells), which absorbs the seed pairing exactly:

  delta(modality, arch, seed) = metric_full - metric_nogeom              (the prior effect)

  ARCH contrast  (CRNN vs Transformer), pooled over modality:
      D_arch = delta_CRNN - delta_Xfm    per matched (modality, seed)     -> tests arch x prior
  MODALITY contrast (FOA vs MIC), pooled over architecture:
      D_mod  = delta_FOA  - delta_MIC    per matched (arch, seed)         -> tests modality x prior

Filling the 2x2 {LE, RDE} x {arch contrast, modality contrast} with these paired
tests shows exactly which interaction each metric's effect rests on, and how
strong the evidence is at n=5. We also echo the Type-II factorial interaction
F/p for completeness.

Outputs (regenerable):
    D:\\ssl-research\\paper\\path_c_dissociation_test.json
    D:\\ssl-research\\paper\\path_c_dissociation_test.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
from _path_c_2x2_anova import collect_long_format  # type: ignore

OUT = Path(r"D:\ssl-research\paper")


def per_seed_deltas(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    rows = []
    for mod in ("MIC", "FOA"):
        for arch in ("CRNN", "Conformer", "Xfm"):
            cell = df[(df["modality"] == mod) & (df["arch"] == arch)]
            full = cell[cell["prior"] == "full"].set_index("seed")[dv]
            nog = cell[cell["prior"] == "nogeom"].set_index("seed")[dv]
            shared = sorted(set(full.index) & set(nog.index))
            for s in shared:
                rows.append({"modality": mod, "arch": arch, "seed": int(s),
                             "delta": float(full.loc[s] - nog.loc[s])})
    return pd.DataFrame(rows)


def paired_t(D: np.ndarray) -> dict:
    D = np.asarray(D, dtype=float)
    t, p = stats.ttest_1samp(D, 0.0)
    sd = D.std(ddof=1)
    dz = float(D.mean() / sd) if sd > 0 else float("nan")
    return {"n": int(len(D)), "mean": float(D.mean()), "t": float(t),
            "p": float(p), "cohens_dz": dz}


def arch_contrast(deltas: pd.DataFrame) -> dict:
    """D = delta_CRNN - delta_Xfm per matched (modality, seed); pooled over modality."""
    w = deltas[deltas["arch"].isin(["CRNN", "Xfm"])].pivot_table(
        index=["modality", "seed"], columns="arch", values="delta").dropna(subset=["CRNN", "Xfm"])
    return paired_t((w["CRNN"] - w["Xfm"]).to_numpy())


def modality_contrast(deltas: pd.DataFrame) -> dict:
    """D = delta_FOA - delta_MIC per matched (arch, seed); pooled over all three archs."""
    w = deltas.pivot_table(index=["arch", "seed"], columns="modality",
                           values="delta").dropna(subset=["FOA", "MIC"])
    return paired_t((w["FOA"] - w["MIC"]).to_numpy())


def factorial_interactions(df: pd.DataFrame, dv: str) -> dict:
    """Type-II factorial F/p for the two interactions, on the full 3-level design."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = (f"{dv} ~ C(modality, Treatment('MIC')) * C(arch, Treatment('CRNN')) "
               f"* C(prior, Treatment('nogeom'))")
    aov = anova_lm(ols(formula, data=df).fit(), typ=2).reset_index()
    out = {}
    for _, r in aov.iterrows():
        name = str(r["index"])
        if pd.isna(r.get("F")):
            continue
        if "modality" in name and "prior" in name and "arch" not in name:
            out["modality_x_prior"] = {"F": float(r["F"]), "p": float(r["PR(>F)"])}
        elif "arch" in name and "prior" in name and "modality" not in name:
            out["arch_x_prior"] = {"F": float(r["F"]), "p": float(r["PR(>F)"])}
    return out


def main() -> int:
    df = collect_long_format()
    print(f"[info] long-format rows: {len(df)}")

    res = {}
    for dv, nice in (("LE", "DOAE_CD (direction)"), ("RDE", "RDE (distance)")):
        d = per_seed_deltas(df, dv)
        res[dv] = {
            "metric": nice,
            "arch_contrast_CRNN_minus_Xfm": arch_contrast(d),
            "modality_contrast_FOA_minus_MIC": modality_contrast(d),
            "factorial": factorial_interactions(df, dv),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "path_c_dissociation_test.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {OUT / 'path_c_dissociation_test.json'}")

    def fmt(c):
        return f"mean={c['mean']:+.3f}, t={c['t']:+.2f}, p={c['p']:.3f}, d_z={c['cohens_dz']:+.2f} (n={c['n']})"

    L = ["# Path C / P1-3: direction-vs-distance double-dissociation test",
         "",
         "Each cell is a within-seed paired SECOND DIFFERENCE on the geometry effect",
         "delta = full - nogeom. The ARCH contrast (delta_CRNN - delta_Xfm, pooled over",
         "modality) tests architecture x prior; the MODALITY contrast",
         "(delta_FOA - delta_MIC, pooled over the three backbones) tests modality x prior.",
         "A clean double dissociation predicts: DIRECTION significant under the ARCH",
         "contrast only; DISTANCE significant under the MODALITY contrast only.",
         "",
         "## Paired second-difference interactions",
         "",
         "| metric | ARCH contrast (CRNN-Xfm) | MODALITY contrast (FOA-MIC) |",
         "| ------ | ------------------------ | --------------------------- |"]
    for dv in ("LE", "RDE"):
        a = res[dv]["arch_contrast_CRNN_minus_Xfm"]
        m = res[dv]["modality_contrast_FOA_minus_MIC"]
        L.append(f"| {res[dv]['metric']} | {fmt(a)} | {fmt(m)} |")
    L += ["",
          "## Type-II factorial interaction F/p (3-level design, n_obs=60)",
          "",
          "| metric | arch x prior | modality x prior |",
          "| ------ | ------------ | ---------------- |"]
    for dv in ("LE", "RDE"):
        fa = res[dv]["factorial"].get("arch_x_prior", {})
        fm = res[dv]["factorial"].get("modality_x_prior", {})
        L.append(f"| {res[dv]['metric']} | F={fa.get('F', float('nan')):.2f}, p={fa.get('p', float('nan')):.3f} "
                 f"| F={fm.get('F', float('nan')):.2f}, p={fm.get('p', float('nan')):.3f} |")
    L.append("")

    (OUT / "path_c_dissociation_test.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {OUT / 'path_c_dissociation_test.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
