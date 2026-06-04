"""Print Stage 1 (5-seed FOA reproduce) summary in a clean table."""
import json

d = json.load(open(r"D:\ssl-research\paper\path_c_results.json"))
foa = d["per_cell"]["100_foa_repro"]

print(f"Stage 1 -- FOA reproduce  (DCASE 2024 baseline + finetune from synthetic, 60 ep, n=5):")
print()
print(f"  {'Metric':<22} {'mean':>8} {'std':>8}     per-seed values")
print(f"  {'-'*22} {'-'*8} {'-'*8}     {'-'*40}")
for m, label in (("F1", "F 20deg (%)"),
                 ("LE", "DOAE_CD (deg)"),
                 ("RDE", "RDE_CD (rel)"),
                 ("DE", "Dist_err (m)"),
                 ("SELD", "SELD score")):
    e = foa[m]
    vals = ", ".join(f"{v:.2f}" for v in e["values"])
    print(f"  {label:<22} {e['mean']:>8.3f} {e['std']:>8.3f}     [{vals}]")
print()
print("DCASE 2024 official (Krause et al., EUSIPCO 2024):")
print("  F 20deg = 13.1 %      DOAE_CD = 36.9 deg     RDE = 0.33     Dist_err ~ 0.5")
print()
import statistics as s
fs = foa["F1"]["values"]
print(f"95% CI on F 20deg via t-dist: {s.mean(fs):.2f} +/- {1.96*s.stdev(fs)/(len(fs)**0.5):.2f} %")
print(f"  -> contains official 13.1%? {abs(s.mean(fs)-13.1) <= 1.96*s.stdev(fs)/(len(fs)**0.5)}")
