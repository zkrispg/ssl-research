"""Path B / final pairwise t-test results."""
from __future__ import annotations

import json
from pathlib import Path

R = Path("D:/ssl-research/week11_starss23/runs")


def _sig(p: float) -> str:
    if p != p:
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _show(d: dict, label: str, thresholds: tuple[str, ...] = ("0.18", "0.30")) -> None:
    n = d.get("n_seeds", "?")
    seeds = d.get("seeds", "?")
    print(f"\n  {label}  (N={n}, seeds={seeds})")
    print(f'    {"thr":<6}{"avg":<8}{"metric":<8}{"a":>10}{"b":>10}{"delta":>10}{"p":>10}')
    print("    " + "-" * 56)
    dcase = d.get("dcase", {})
    for thr in thresholds:
        block = dcase.get(thr, {})
        for avg in ("macro", "micro"):
            for m in ("seld", "f1"):
                if avg not in block or m not in block[avg]:
                    continue
                c = block[avg][m]
                p = c.get("p", float("nan"))
                a = sum(c["a_per_seed"]) / len(c["a_per_seed"])
                b = sum(c["b_per_seed"]) / len(c["b_per_seed"])
                print(
                    f'    {thr:<6}{avg:<8}{m:<8}'
                    f'{a:>10.4f}{b:>10.4f}'
                    f'{c["mean_delta"]:>+10.4f}{p:>10.4f}{_sig(p)}'
                )


print("=" * 80)
print("PATH B FINAL HEADLINE  (full = with geometry; no_geom = ablated)")
print("=" * 80)

print("\n[A1] Capacity x geometry sweep at N=5 (paired t-test)")
print("     delta = full - no_geom; positive => geometry HURTS that size")
for size, base in [
    ("xs", "pairwise_no_geom_xs_cap_xs_mc8_inmem_vs_full_xs_cap_xs_mc8_inmem_n5"),
    ("m",  "pairwise_no_geom_mc8_inmem_vs_full_mc8_inmem_n5"),
    ("l",  "pairwise_no_geom_l_cap_l_mc8_inmem_vs_full_l_cap_l_mc8_inmem_n5"),
    ("xl", "pairwise_no_geom_xl_cap_xl_mc8_inmem_vs_full_xl_cap_xl_mc8_inmem_n5"),
]:
    fp = R / f"{base}.json"
    if not fp.exists():
        print(f"\n  size={size}: MISSING ({fp.name})")
        continue
    d = json.loads(fp.read_text(encoding="utf-8"))
    _show(d, f"size={size}")

print("\n[B2] SELDnet MIC vs FOA at N=5")
fp = R / "pairwise_seldnet_official_baseline_mc8_inmem_vs_seldnet_official_foa_baseline_mc8_inmem_n5.json"
if fp.exists():
    d = json.loads(fp.read_text(encoding="utf-8"))
    _show(d, "SELDnet  MIC vs FOA")
else:
    print("\n  MISSING:", fp.name)

# B1: cross-dataset
print("\n[B1] Zero-shot cross-dataset on STARSS22 dev-test (paired t-tests)")
fp = R / "cross_dataset_starss22-test_summary.json"
if fp.exists():
    d = json.loads(fp.read_text(encoding="utf-8"))
    print(f'    {"comparison":<32}{"thr":<6}{"avg":<6}{"metric":<8}{"a":>10}{"b":>10}{"delta":>10}{"p":>10}')
    print("    " + "-" * 92)
    for c in d.get("comparisons", []):
        if c["metric"] == "seld" and c["average"] == "macro":
            label = f'{c["a_variant"]} vs {c["b_variant"]}'
            p = c.get("p", float("nan"))
            print(
                f'    {label:<32}{c["threshold"]:<6}{c["average"]:<6}{c["metric"]:<8}'
                f'{c["mean_a"]:>10.4f}{c["mean_b"]:>10.4f}'
                f'{c["mean_delta"]:>+10.4f}{p:>10.4f}{_sig(p)}'
            )
else:
    print("    MISSING:", fp.name)

# In-distribution reminder is already shown in [A1] under size=m

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
