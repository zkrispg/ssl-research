"""T1c -- bootstrap CIs, Cohen's d_z, and post-hoc power for paired tests.

For every paired comparison saved under ``runs/pairwise_*.json`` we
compute:
    * 10K-iteration percentile bootstrap 95 % CI on the paired delta mean.
    * Cohen's d_z = mean(delta) / std(delta, ddof=1) — the effect size
      for a paired-samples t-test.
    * Post-hoc power at observed effect size via Monte Carlo:
        - draw new samples from N(d_z, 1) of the original sample size
        - run paired t-test, record fraction with p < 0.05
        - 5K iterations.

The script reads each pairwise file, augments it with the new
statistics, and writes all augmented data to
``runs/analysis_bootstrap.json``. Each per-comparison entry is keyed by
the original filename's stem.

Why we report d_z and post-hoc power: ICASSP reviewers in 2025+
increasingly expect *both* a p-value and an effect size with a power
estimate, especially for null-leaning results. Without these, our null
geometry result on the loss metric ("p = 0.51") is hard to defend.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RUNS = Path("D:/ssl-research/week11_starss23/runs")
OUT = RUNS / "analysis_bootstrap.json"
B = 10_000
POWER_SIMS = 5_000
RNG = np.random.default_rng(20260518)


def _bootstrap_mean_ci(delta: np.ndarray, n_boot: int = B) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI on the mean of paired deltas."""
    if len(delta) < 2:
        return (float("nan"), float("nan"))
    boot = RNG.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _cohens_dz(delta: np.ndarray) -> float:
    if len(delta) < 2:
        return float("nan")
    s = delta.std(ddof=1)
    if s == 0:
        return float("inf") if delta.mean() != 0 else 0.0
    return float(delta.mean() / s)


def _post_hoc_power(d_z: float, n: int, n_sims: int = POWER_SIMS, alpha: float = 0.05) -> float:
    """Monte-Carlo post-hoc power estimate at the observed d_z and N."""
    if not np.isfinite(d_z) or n < 2:
        return float("nan")
    sims = RNG.normal(loc=d_z, scale=1.0, size=(n_sims, n))
    mean = sims.mean(axis=1)
    std = sims.std(axis=1, ddof=1)
    safe_std = np.where(std == 0, 1e-9, std)
    t = mean / (safe_std / np.sqrt(n))
    df = n - 1
    p_two_sided = 2 * (1 - stats.t.cdf(np.abs(t), df=df))
    return float((p_two_sided < alpha).mean())


def _required_n_for_power_80(d_z: float, max_n: int = 50) -> int | None:
    """Smallest paired-sample size at which observed |d_z| reaches 80 % power."""
    if not np.isfinite(d_z) or d_z == 0:
        return None
    abs_d = abs(d_z)
    for n in range(2, max_n + 1):
        if _post_hoc_power(abs_d, n, n_sims=2_000) >= 0.80:
            return n
    return max_n + 1


def _augment_block(stat_block: dict) -> dict:
    """Given a paired_t-style dict (with a/b per-seed lists), add bootstrap+power."""
    a = np.array(stat_block.get("a_per_seed", []), dtype=np.float64)
    b = np.array(stat_block.get("b_per_seed", []), dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return {**stat_block, "bootstrap": None, "cohens_dz": None, "power_at_observed": None}
    delta = b - a
    ci_lo, ci_hi = _bootstrap_mean_ci(delta)
    d_z = _cohens_dz(delta)
    power = _post_hoc_power(d_z, len(delta))
    n_for_80 = _required_n_for_power_80(d_z)
    return {
        **stat_block,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "cohens_dz": d_z,
        "power_at_observed": power,
        "required_n_for_80pct_power": n_for_80,
    }


def main() -> None:
    print(f"[analysis] bootstrap + Cohen's d_z + power -> {OUT}\n", flush=True)
    out: dict[str, Any] = {}
    seen = 0
    for path in sorted(RUNS.glob("pairwise_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        comp_name = path.stem.replace("pairwise_", "")
        comp_out: dict[str, Any] = {
            "a": data.get("a"),
            "b": data.get("b"),
            "n_seeds": data.get("n_seeds"),
            "seeds": data.get("seeds"),
        }
        # ADPIT eval loss block
        bel = data.get("best_eval_loss")
        if bel:
            comp_out["best_eval_loss"] = _augment_block(bel)
        # DCASE per-threshold per-avg per-metric
        if "dcase" in data:
            comp_out["dcase"] = {}
            for thr, avg_block in data["dcase"].items():
                comp_out["dcase"][thr] = {}
                for avg, metric_block in avg_block.items():
                    comp_out["dcase"][thr][avg] = {
                        m: _augment_block(b) for m, b in metric_block.items()
                    }
        out[comp_name] = comp_out
        seen += 1

        # Print a one-liner highlighting the @0.18 macro-SELD effect.
        if "dcase" in data and "0.18" in data["dcase"]:
            d = data["dcase"]["0.18"].get("macro", {}).get("seld")
            if d is not None:
                aug = _augment_block(d)
                ci = aug["bootstrap_ci_95"]
                print(
                    f"[{comp_name:>50}]  N={data.get('n_seeds')}  "
                    f"Δ={d['mean_delta']:+.4f}  p={d['p']:.3f}  "
                    f"d_z={aug['cohens_dz']:+.2f}  "
                    f"95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                    f"power={aug['power_at_observed']:.2f}",
                    flush=True,
                )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}  ({seen} comparisons)\n")


if __name__ == "__main__":
    main()
