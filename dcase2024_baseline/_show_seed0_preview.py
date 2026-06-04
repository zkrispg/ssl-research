"""Print n=1 (seed 0) cross-cell preview of Stage 3 ablation."""
import json

d = json.load(open(r"D:\ssl-research\paper\path_c_results.json"))
cells = [("100_foa_repro",   "FOA reproduce (Stage 1 ref, n=5)"),
         ("110_gca_full",    "MIC + GCA full       (seed 0)"),
         ("111_gca_nogeom",  "MIC + GCA no_geom    (seed 0)"),
         ("112_no_gca",      "MIC + no GCA control (seed 0)")]

print(f"{'cell':<40} {'F 20deg %':>10} {'AE deg':>10} {'RDE':>7} {'SELD':>8}")
print("-" * 76)
for c, label in cells:
    e = d["per_cell"][c]
    if e["n_seeds"] == 0:
        print(f"{label:<40}   (no data)")
        continue
    if e["n_seeds"] == 1:
        f1   = e["F1"]["values"][0]
        le   = e["LE"]["values"][0]
        rde  = e["RDE"]["values"][0]
        seld = e["SELD"]["values"][0]
    else:
        f1   = e["F1"]["mean"]
        le   = e["LE"]["mean"]
        rde  = e["RDE"]["mean"]
        seld = e["SELD"]["mean"]
    print(f"{label:<40} {f1:>10.2f} {le:>10.2f} {rde:>7.2f} {seld:>8.3f}")
print()
print("Note: 100_foa_repro is FOA modality (Stage 1 reference). 110/111/112 are MIC modality (Stage 3).")
print("Per-seed-0 deltas (110-112, 111-112):")
g_full = d["per_cell"]["110_gca_full"]
g_no   = d["per_cell"]["111_gca_nogeom"]
ctrl   = d["per_cell"]["112_no_gca"]
if g_full["n_seeds"] >= 1 and ctrl["n_seeds"] >= 1:
    for m in ("F1", "LE", "RDE", "SELD"):
        df = g_full[m]["values"][0] - ctrl[m]["values"][0]
        dn = g_no[m]["values"][0]   - ctrl[m]["values"][0]
        print(f"  {m:5}   GCA_full - no_GCA = {df:+.3f}     GCA_nogeom - no_GCA = {dn:+.3f}")
