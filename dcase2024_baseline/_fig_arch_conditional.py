"""Render the architecture-conditional figure for the paper.

Reads the precomputed paired contrasts in path_c_2x2.json and draws a single
column-width panel of Delta DOAE_CD, ordered and grouped by temporal
architecture so the sign boundary is visually explicit:

    [ MIC+CRNN  FOA+CRNN ]  |  [ MIC+Xfm  FOA+Xfm ]
       recurrent: helps/neutral   transformer: hurts/neutral

Output: D:\\ssl-research\\paper\\figs\\path_c_2x2_dissociation.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

JSON_PATH = Path(r"D:\ssl-research\paper\path_c_2x2.json")
OUT_PNG = Path(r"D:\ssl-research\paper\figs\path_c_2x2_dissociation.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# left-to-right order, grouped by architecture
ORDER = ["MIC_CRNN", "FOA_CRNN", "MIC_XFM", "FOA_XFM"]
GROUP_SPLIT = 2  # first 2 are recurrent, last 2 are transformer

HELP = "#2c6fbb"    # blue
HURT = "#c0392b"    # red
NULL = "#b0b0b0"    # gray


def classify(delta: float, dz: float) -> str:
    if delta is None or np.isnan(dz):
        return "null"
    if delta <= -2 and dz <= -1:
        return "help"
    if delta >= 2 and dz >= 1:
        return "hurt"
    return "null"


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    cells = data["cells"]

    labels, deltas, lo, hi, colors = [], [], [], [], []
    for key in ORDER:
        c = cells[key]
        le = c["LE"]
        d = le.get("delta_mean")
        dz = le.get("cohens_dz", float("nan"))
        ci = le.get("boot_ci_95", [np.nan, np.nan])
        labels.append(c["label"].replace(" + ", "+"))
        deltas.append(d if d is not None else np.nan)
        lo.append(ci[0]); hi.append(ci[1])
        kind = classify(d, dz)
        colors.append({"help": HELP, "hurt": HURT, "null": NULL}[kind])

    deltas = np.array(deltas, float)
    yerr = np.stack([deltas - np.array(lo, float), np.array(hi, float) - deltas])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.5, 3.1))

    # shaded architecture regions
    ax.axvspan(-0.55, GROUP_SPLIT - 0.5, color=HELP, alpha=0.06, zorder=0)
    ax.axvspan(GROUP_SPLIT - 0.5, len(labels) - 0.45, color=HURT, alpha=0.06, zorder=0)
    ax.axvline(GROUP_SPLIT - 0.5, color="0.5", lw=0.8, ls="-", zorder=1)
    ax.axhline(0, color="black", lw=0.8, ls=":", zorder=2)

    ax.bar(x, deltas, width=0.62, color=colors, edgecolor="black",
           linewidth=0.7, yerr=yerr, capsize=4, zorder=3,
           error_kw={"lw": 1.0})

    # value labels
    for xi, d in zip(x, deltas):
        if np.isnan(d):
            continue
        off = 0.9 if d >= 0 else -0.9
        va = "bottom" if d >= 0 else "top"
        ax.text(xi, d + off, f"{d:+.1f}", ha="center", va=va, fontsize=8.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=12)
    ax.set_ylabel(r"$\Delta$DOAE$_\mathrm{CD}$ (deg): full $-$ no_geom", fontsize=9)

    ymax = np.nanmax(np.abs(hi) if np.nanmax(np.abs(hi)) else 1)
    lim = max(13, np.nanmax(np.abs(np.concatenate([lo, hi]))) + 2)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.55, len(labels) - 0.45)

    # architecture group annotations (with direction hint folded in)
    ax.text(0.5, lim * 0.95, "Recurrent (CRNN+MHSA)", ha="center", va="top",
            fontsize=8.5, color=HELP, fontweight="bold")
    ax.text(0.5, lim * 0.95 - lim * 0.10, "helps / neutral", ha="center",
            va="top", fontsize=7.5, style="italic", color=HELP)
    ax.text(2.5, -lim * 0.80, "Transformer", ha="center", va="bottom",
            fontsize=8.5, color=HURT, fontweight="bold")
    ax.text(2.5, -lim * 0.80 - lim * 0.02, "hurts / neutral", ha="center",
            va="top", fontsize=7.5, style="italic", color=HURT)

    ax.set_title("Geometry prior: effect sign tracks architecture",
                 fontsize=9.5, pad=6)

    legend = [Patch(fc=HELP, ec="black", label="helps ($d_z\\!\\leq\\!-1$)"),
              Patch(fc=HURT, ec="black", label="hurts ($d_z\\!\\geq\\!+1$)"),
              Patch(fc=NULL, ec="black", label="null")]
    ax.legend(handles=legend, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, framealpha=0.9,
              handlelength=1.1, borderpad=0.4, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
