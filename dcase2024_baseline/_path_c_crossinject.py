"""Cross-injection robustness: does the architecture-graded geometry-prior
effect survive a SECOND injection mechanism?

The main paper measures the geometry prior with GCA (geometry biases the
channel-attention keys). Here we re-measure the SAME contrast with a
mechanistically different injection -- ``convbias`` (geometry mapped by a
learned linear projection and added as a per-filter bias to the first conv
feature maps) -- on the two EXTREME cells of the architecture axis:

  FOA + CRNN        -> 180 (full) vs 181 (no_geom)   [GCA HELPS:  delta < 0]
  MIC + Transformer -> 182 (full) vs 183 (no_geom)   [GCA HURTS:  delta > 0]

If the sign of the paired DOAE_CD contrast matches the GCA result in both
cells, the helps->harms ordering is not an artifact of one injection
mechanism. Reuses the same paired-difference statistics as the main 2x2 table.

Outputs:
    D:\\ssl-research\\paper\\path_c_crossinject.json
    D:\\ssl-research\\paper\\path_c_crossinject.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
from _path_c_analyze import parse_log  # type: ignore
from _path_c_2x2_dissociation import cohens_dz, bootstrap_ci, _resolve_log  # type: ignore

OUT_PATH = Path(r"D:\ssl-research\paper")
OUT_PATH.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]

CELLS = [
    {"key": "FOA_CRNN", "label": "FOA + CRNN", "arch": "CRNN",
     "expect": "helps", "gca_delta_LE": -5.02,
     "full": "180", "nogeom": "181"},
    {"key": "MIC_XFM", "label": "MIC + Transformer", "arch": "Transformer",
     "expect": "harms", "gca_delta_LE": +5.70,
     "full": "182", "nogeom": "183"},
]


def collect(task: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for seed in SEEDS:
        log = _resolve_log(f"dcase2024_{task}_ablate_seed{seed}_test.log")
        m = parse_log(str(log))
        if m is not None:
            out[seed] = m
    return out


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except Exception:
        w_stat, p_w = float("nan"), float("nan")
    return {
        "delta_mean": float(d.mean()),
        "delta_std": float(d.std(ddof=1)),
        "t": float(t_stat), "p_t": float(p_t),
        "w": float(w_stat), "p_w": float(p_w),
        "cohens_dz": cohens_dz(d.tolist()),
        "boot_ci_95": list(bootstrap_ci(d.tolist())),
    }


def main() -> int:
    payload: dict[str, dict] = {}
    for spec in CELLS:
        full = collect(spec["full"])
        ngm = collect(spec["nogeom"])
        shared = sorted(set(full) & set(ngm))
        c = {"label": spec["label"], "arch": spec["arch"],
             "expect": spec["expect"], "gca_delta_LE": spec["gca_delta_LE"],
             "n_full": len(full), "n_nogeom": len(ngm), "shared_seeds": shared}
        for metric in ("F1", "LE", "DE", "RDE", "SELD"):
            f_vals = np.array([full[s][metric] for s in full]) if full else np.array([])
            g_vals = np.array([ngm[s][metric] for s in ngm]) if ngm else np.array([])
            c[f"{metric}_full_mean"] = float(np.mean(f_vals)) if len(f_vals) else None
            c[f"{metric}_full_std"] = float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else 0.0
            c[f"{metric}_nogeom_mean"] = float(np.mean(g_vals)) if len(g_vals) else None
            c[f"{metric}_nogeom_std"] = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else 0.0
            if len(shared) >= 2:
                a = np.array([full[s][metric] for s in shared])
                b = np.array([ngm[s][metric] for s in shared])
                c[metric] = paired_stats(a, b)
            else:
                c[metric] = {"note": f"insufficient pairs (n={len(shared)})"}
        # ordering verdict on the headline metric (LE = DOAE_CD)
        if isinstance(c["LE"], dict) and c["LE"].get("delta_mean") is not None:
            obs = c["LE"]["delta_mean"]
            gca = spec["gca_delta_LE"]
            c["sign_matches_gca"] = bool(np.sign(obs) == np.sign(gca))
        else:
            c["sign_matches_gca"] = None
        payload[spec["key"]] = c
        print(f"[{spec['label']}] n={len(shared)} expect={spec['expect']} "
              f"convbias dDOAE={c['LE'].get('delta_mean','-') if isinstance(c['LE'],dict) else '-'} "
              f"(GCA {spec['gca_delta_LE']:+.2f})  match={c['sign_matches_gca']}")

    out_json = OUT_PATH / "path_c_crossinject.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    L = ["# Cross-injection robustness: convbias geometry prior on the two extreme cells",
         "",
         "A *second* geometry-injection mechanism (`convbias`: geometry as a learned ",
         "per-filter conv-feature bias) re-measured on the recurrent (helps) and ",
         "pure-attention (harms) cells. Paired contrast = full - no_geom on matched seeds; ",
         "`full` and `no_geom` have identical parameter counts. The GCA column repeats the ",
         "main-paper result for the same cell.",
         "",
         "## DOAE_CD (deg) -- headline metric (negative = geometry helps)",
         "| cell | n | convbias full | convbias no_geom | delta DOAE (convbias) | t (p) | d_z | 95% CI | GCA delta | sign matches? |",
         "| ---- | - | ------------- | ---------------- | --------------------- | ----- | --- | ------ | --------- | ------------- |"]
    for spec in CELLS:
        c = payload[spec["key"]]
        if not (isinstance(c["LE"], dict) and c["LE"].get("delta_mean") is not None):
            L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | n/a | n/a | n/a | - | - | - | {spec['gca_delta_LE']:+.2f} | - |")
            continue
        d = c["LE"]; ci = d["boot_ci_95"]
        match = "yes" if c["sign_matches_gca"] else "NO"
        L.append(f"| **{spec['label']}** | {len(c['shared_seeds'])} | "
                 f"{c['LE_full_mean']:.2f} \u00b1 {c['LE_full_std']:.2f} | "
                 f"{c['LE_nogeom_mean']:.2f} \u00b1 {c['LE_nogeom_std']:.2f} | "
                 f"**{d['delta_mean']:+.2f}** | t={d['t']:+.2f} (p={d['p_t']:.3f}) | "
                 f"**{d['cohens_dz']:+.2f}** | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                 f"{spec['gca_delta_LE']:+.2f} | **{match}** |")
    L.append("")
    L += ["## F1 (%) and SELD score (paired delta = full - no_geom)",
          "| cell | delta F1 | t (p) | delta SELD | t (p) |",
          "| ---- | -------- | ----- | ---------- | ----- |"]
    for spec in CELLS:
        c = payload[spec["key"]]
        f1 = c["F1"]; sd = c["SELD"]
        if isinstance(f1, dict) and f1.get("delta_mean") is not None:
            L.append(f"| {spec['label']} | {f1['delta_mean']:+.2f} | t={f1['t']:+.2f} (p={f1['p_t']:.3f}) | "
                     f"{sd['delta_mean']:+.3f} | t={sd['t']:+.2f} (p={sd['p_t']:.3f}) |")
        else:
            L.append(f"| {spec['label']} | n/a | - | n/a | - |")
    L.append("")
    # verdict
    verdicts = [c.get("sign_matches_gca") for c in payload.values()]
    if all(v is True for v in verdicts):
        L.append("**Verdict:** the helps->harms ordering is preserved under a second, "
                 "mechanistically distinct injection -- the architecture-graded effect is "
                 "not an artifact of GCA-style injection.")
    elif any(v is None for v in verdicts):
        L.append("**Verdict:** incomplete -- some cells still missing seeds.")
    else:
        L.append("**Verdict:** the ordering does NOT fully replicate under convbias; "
                 "interpret the geometry-prior effect as injection-dependent.")
    out_md = OUT_PATH / "path_c_crossinject.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
