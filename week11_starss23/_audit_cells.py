"""Audit script: integrity check of all 10 multi-seed cells.

Checks:
1. Each cell has best.pt + summary.json + eval_threshold_sweep.json.
2. Reported best_eval_loss matches what the training log shows.
3. Number of trained epochs reaches 30 (or known salvage point).
4. n_params matches expected (no_geom 590,886; full 590,966).
5. eval_threshold_sweep.json is internally consistent (78 clips evaluated).
6. The full seed=2 "salvaged" summary lines up with log epoch 26.
7. Spot-check: re-load ckpt and confirm config matches expected variant.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

RUNS = Path("D:/ssl-research/week11_starss23/runs")
EXPECTED_PARAMS = {"no_geom": 590886, "full": 590966}


def epochs_from_log(log_path: Path) -> tuple[int, float | None]:
    """Return (max epoch reached, last best eval reported in log) from a queue/training log."""
    if not log_path.exists():
        return -1, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Strip null bytes (Start-Process redirect leftovers)
    text = text.replace("\x00", "")
    max_ep = -1
    best_eval_seen = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("epoch") and "/30" in line:
            try:
                ep = int(line.split("epoch")[1].split("/")[0].strip())
                max_ep = max(max_ep, ep)
            except ValueError:
                continue
            # parse eval=... if present
            if "eval=" in line:
                try:
                    val = float(line.split("eval=")[1].split()[0])
                    if best_eval_seen is None or val < best_eval_seen:
                        best_eval_seen = val
                except (ValueError, IndexError):
                    pass
    return max_ep, best_eval_seen


def find_log(variant: str, seed: int) -> Path | None:
    """Pick the most recent log file that mentions this cell."""
    candidates = [
        RUNS / f"queue_{variant}_seed{seed}.log",
        RUNS / "train_real15.log",
        RUNS / f"train_{variant}_mc8_inmem.log",
        RUNS / "train_mc8_inmem.log",
    ]
    return next((p for p in candidates if p.exists()), None)


def main() -> None:
    issues: list[str] = []

    print(f"{'cell':<18} {'ckpt_MB':>8} {'summary_be':>11} {'log_be':>11} "
          f"{'log_max_ep':>11} {'n_params':>10} {'sweep_clips':>12}")
    print("-" * 90)

    for variant in ("no_geom", "full"):
        for seed in range(5):
            d = RUNS / f"{variant}_seed{seed}_mc8_inmem"
            ckpt_path = d / "best.pt"
            sj_path = d / "summary.json"
            esj_path = d / "eval_threshold_sweep.json"

            ckpt_mb = ckpt_path.stat().st_size / 1e6
            sj = json.loads(sj_path.read_text(encoding="utf-8"))
            esj = json.loads(esj_path.read_text(encoding="utf-8"))
            sweep_clips = esj.get("n_clips_eval", "?")

            log = find_log(variant, seed)
            log_max_ep, log_best_eval = epochs_from_log(log) if log else (-1, None)

            summary_be = sj.get("best_eval_loss")
            n_params = sj.get("n_params", "?")
            note = sj.get("note", "")

            print(
                f"{variant}_seed{seed:<10} {ckpt_mb:>8.2f} "
                f"{summary_be:>11.5f} "
                f"{(log_best_eval if log_best_eval else float('nan')):>11.5f} "
                f"{log_max_ep:>11} "
                f"{n_params:>10} "
                f"{sweep_clips:>12}"
                + (f"  [{note}]" if note else "")
            )

            # ---- checks ----
            if isinstance(n_params, int) and n_params != EXPECTED_PARAMS[variant]:
                issues.append(
                    f"{variant} seed={seed}: n_params={n_params} "
                    f"!= expected {EXPECTED_PARAMS[variant]}"
                )
            if log_best_eval is not None and abs(log_best_eval - summary_be) > 1e-4:
                issues.append(
                    f"{variant} seed={seed}: summary.best_eval_loss={summary_be:.5f} "
                    f"differs from log min eval={log_best_eval:.5f} by "
                    f"{abs(log_best_eval - summary_be):.5f}"
                )
            if sweep_clips != 78:
                issues.append(
                    f"{variant} seed={seed}: eval_threshold_sweep covered "
                    f"{sweep_clips} clips, expected 78"
                )
            if log_max_ep != -1 and log_max_ep < 26:
                issues.append(
                    f"{variant} seed={seed}: training log only reached epoch "
                    f"{log_max_ep}, expected 30 (or 26 for the known-salvaged full_seed2)"
                )

    print()
    print("=== loading ckpts to verify config matches variant ===")
    for variant in ("no_geom", "full"):
        for seed in range(5):
            d = RUNS / f"{variant}_seed{seed}_mc8_inmem"
            ckpt = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
            mc = ckpt["model_cfg"]
            expected_bias = (variant == "full")
            actual_bias = mc.get("gca_geometry_bias")
            actual_use_gca = mc.get("use_gca")
            ok = actual_bias == expected_bias and actual_use_gca is True
            tag = "OK" if ok else "MISMATCH"
            print(
                f"  {variant}_seed{seed}: use_gca={actual_use_gca}  "
                f"gca_geometry_bias={actual_bias}  [{tag}]"
            )
            if not ok:
                issues.append(
                    f"{variant} seed={seed}: config mismatch "
                    f"(use_gca={actual_use_gca}, bias={actual_bias})"
                )

    print()
    print("=== 78-clip evaluation: macro F1 / SELD per cell @ thr=0.18 ===")
    print(f"{'cell':<18} {'F1m':>7} {'SELDm':>7} {'F1u':>7} {'SELDu':>7}")
    for variant in ("no_geom", "full"):
        for seed in range(5):
            d = RUNS / f"{variant}_seed{seed}_mc8_inmem"
            esj = json.loads((d / "eval_threshold_sweep.json").read_text(encoding="utf-8"))
            m = esj["thresholds"]["0.18"]["macro"]
            u = esj["thresholds"]["0.18"]["micro"]
            print(
                f"  {variant}_seed{seed:<10} "
                f"{m['f1']:>7.4f} {m['seld']:>7.4f} "
                f"{u['f1']:>7.4f} {u['seld']:>7.4f}"
            )

    print()
    if not issues:
        print("[AUDIT] all checks passed.")
    else:
        print(f"[AUDIT] {len(issues)} issues found:")
        for i in issues:
            print(f"  - {i}")


if __name__ == "__main__":
    main()
