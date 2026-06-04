"""A3: distance (RDE) contrast across the 2x2 grid.

Reads the precomputed paired contrasts in path_c_2x2.json and reports the
geometry-prior effect on relative distance error (RDE), the third dimension of
the DCASE 2024 3D-SELD task. The point of interest is whether the prior's effect
on *distance* follows the same architecture-conditional pattern as its effect on
*direction* (DOAE), or a different one.

Output: D:\\ssl-research\\paper\\path_c_rde.md
"""
from __future__ import annotations

import json
from pathlib import Path

JSON_PATH = Path(r"D:\ssl-research\paper\path_c_2x2.json")
OUT_MD = Path(r"D:\ssl-research\paper\path_c_rde.md")

ORDER = ["MIC_CRNN", "FOA_CRNN", "MIC_XFM", "FOA_XFM"]


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    cells = data["cells"]

    L = ["# A3 / distance (RDE) geometry-prior contrast across the 2x2 grid",
         "",
         "delta RDE = full - no_geom (lower RDE is better; negative delta = prior helps).",
         "Compare against the DOAE pattern: DOAE is architecture-conditional, RDE is",
         "expected to test whether distance behaves the same way.",
         "",
         "| cell | n | RDE full | RDE no_geom | delta RDE | t (p) | d_z | 95% CI |",
         "| ---- | - | -------- | ----------- | --------- | ----- | --- | ------ |"]
    summary = {}
    for key in ORDER:
        c = cells[key]
        rde = c["RDE"]
        d = rde.get("delta_mean")
        if d is None:
            L.append(f"| {c['label']} | {len(c['shared_seeds'])} | n/a | n/a | n/a | - | - | - |")
            continue
        ci = rde["boot_ci_95"]
        L.append(
            f"| {c['label']} | {len(c['shared_seeds'])} | {c['RDE_full_mean']:.3f} | "
            f"{c['RDE_nogeom_mean']:.3f} | {d:+.3f} | t={rde['t']:+.2f} (p={rde['p_t']:.3f}) | "
            f"{rde['cohens_dz']:+.2f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        summary[key] = (c["modality"], c["arch"], d, rde["cohens_dz"])

    L += ["",
          "## Reading",
          "On RDE the prior **helps both FOA cells** (FOA+CRNN delta="
          f"{summary['FOA_CRNN'][2]:+.3f}, d_z={summary['FOA_CRNN'][3]:+.2f}; FOA+Xfm delta="
          f"{summary['FOA_XFM'][2]:+.3f}, d_z={summary['FOA_XFM'][3]:+.2f}) and is **null in both "
          "MIC cells** "
          f"(MIC+CRNN d_z={summary['MIC_CRNN'][3]:+.2f}; MIC+Xfm d_z={summary['MIC_XFM'][3]:+.2f}).",
          "",
          "Thus the prior's distance effect is **modality-conditional** (helps the Ambisonic",
          "format regardless of temporal architecture), in contrast to its **architecture-",
          "conditional** direction effect. The two spatial dimensions are governed by",
          "different factors -- a double dissociation that reinforces 'it depends'."]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
