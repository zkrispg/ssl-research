from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRIC_PATTERNS = {
    "seld": re.compile(r"SELD score.*?:\s*([0-9.]+)"),
    "f20": re.compile(r"F\s*20.*?:\s*([0-9.]+)\s*%"),
    "doae": re.compile(r"DOAE_CD.*?:\s*([0-9.]+)"),
    "dist": re.compile(r"Dist_err.*?:\s*([0-9.]+)"),
    "rde": re.compile(r"RDE_CD.*?:\s*([0-9.]+)"),
}


def _parse_csv(value: str, cast):
    if not value:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _float_match(pattern: re.Pattern[str], text: str):
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def _last_int_match(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def _last_str_match(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _sd(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def parse_cell(run_dir: Path, task: str, job: str, seed: int):
    train_log = run_dir / f"dcase2024_{task}_{job}.log"
    test_log = run_dir / f"dcase2024_{task}_{job}_test.log"
    row = {
        "task": task,
        "job": job,
        "seed": seed,
        "train_log": str(train_log),
        "test_log": str(test_log),
        "status": "missing",
        "best_epoch": None,
        "best_metric": None,
        "data_digest": None,
    }

    if train_log.exists():
        train_text = train_log.read_text(encoding="utf-8", errors="replace")
        row["best_epoch"] = _last_int_match(r"best_val_epoch:\s*([0-9]+)", train_text)
        row["best_metric"] = _last_str_match(r"\[checkpoint\] best_metric = ([a-z]+)", train_text)
        row["data_digest"] = _last_str_match(r"\[data\] test files=\d+ digest=([0-9a-f]+)", train_text)

    if not test_log.exists():
        return row

    text = test_log.read_text(encoding="utf-8", errors="replace")
    if "SELD score" not in text or "DOAE_CD" not in text:
        row["status"] = "incomplete"
        return row

    row["status"] = "complete"
    for name, pattern in METRIC_PATTERNS.items():
        row[name] = _float_match(pattern, text)
    if row["data_digest"] is None:
        row["data_digest"] = _last_str_match(r"\[test_only\] test files = \d+ digest = ([0-9a-f]+)", text)
    return row


def summarize_tasks(rows):
    out = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and row["status"] == "complete"]
        item = {"task": task, "n": len(task_rows)}
        for metric in ("seld", "f20", "doae", "dist", "rde"):
            vals = [row.get(metric) for row in task_rows]
            item[f"{metric}_mean"] = _mean(vals)
            item[f"{metric}_sd"] = _sd(vals)
        out.append(item)
    return out


def paired_deltas(rows, full_task: str, control_task: str):
    deltas = []
    full_by_seed = {
        row["seed"]: row for row in rows
        if row["task"] == full_task and row["status"] == "complete"
    }
    control_by_seed = {
        row["seed"]: row for row in rows
        if row["task"] == control_task and row["status"] == "complete"
    }
    for seed in sorted(set(full_by_seed) & set(control_by_seed)):
        full = full_by_seed[seed]
        control = control_by_seed[seed]
        item = {"pair": f"{full_task}-{control_task}", "seed": seed}
        for metric in ("seld", "f20", "doae", "dist", "rde"):
            if full.get(metric) is None or control.get(metric) is None:
                item[f"delta_{metric}"] = None
            else:
                item[f"delta_{metric}"] = full[metric] - control[metric]
        deltas.append(item)
    return deltas


def write_markdown(path: Path, rows, summaries, deltas):
    lines = []
    lines.append("# Convbias FOA Run Summary")
    lines.append("")
    lines.append("Positive delta means full-geometry task minus matched no-geometry control.")
    lines.append("For DOAE/SELD/RDE lower is better; for F20 higher is better.")
    lines.append("")
    lines.append("## Per Task")
    lines.append("")
    lines.append("| task | n | SELD mean | F20 mean | DOAE mean | RDE mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summaries:
        lines.append(
            "| {task} | {n} | {seld} | {f20} | {doae} | {rde} |".format(
                task=item["task"],
                n=item["n"],
                seld=_fmt(item.get("seld_mean")),
                f20=_fmt(item.get("f20_mean")),
                doae=_fmt(item.get("doae_mean")),
                rde=_fmt(item.get("rde_mean")),
            )
        )
    lines.append("")
    lines.append("## Paired Deltas")
    lines.append("")
    lines.append("| pair | n | delta SELD | delta F20 | delta DOAE | delta RDE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for pair in sorted({row["pair"] for row in deltas}):
        pair_rows = [row for row in deltas if row["pair"] == pair]
        lines.append(
            "| {pair} | {n} | {seld} | {f20} | {doae} | {rde} |".format(
                pair=pair,
                n=len(pair_rows),
                seld=_fmt(_mean([row.get("delta_seld") for row in pair_rows])),
                f20=_fmt(_mean([row.get("delta_f20") for row in pair_rows])),
                doae=_fmt(_mean([row.get("delta_doae") for row in pair_rows])),
                rde=_fmt(_mean([row.get("delta_rde") for row in pair_rows])),
            )
        )
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | digest |")
    lines.append("|---|---:|---|---|---:|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {task} | {seed} | {job} | {status} | {epoch} | {metric} | {seld} | {f20} | {doae} | {digest} |".format(
                task=row["task"],
                seed=row["seed"],
                job=row["job"],
                status=row["status"],
                epoch=row["best_epoch"] if row["best_epoch"] is not None else "NA",
                metric=row["best_metric"] or "NA",
                seld=_fmt(row.get("seld")),
                f20=_fmt(row.get("f20")),
                doae=_fmt(row.get("doae")),
                digest=row.get("data_digest") or "NA",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--run-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--tasks", default="184,185,186,187")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--job-prefix", default="det_seld")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    tasks = _parse_csv(args.tasks, str)
    seeds = _parse_csv(args.seeds, int)
    out_prefix = args.out_prefix or f"convbias_foa_{args.job_prefix}_summary"

    rows = []
    for seed in seeds:
        for task in tasks:
            rows.append(parse_cell(args.run_dir, task, f"{args.job_prefix}_seed{seed}", seed))

    summaries = summarize_tasks(rows)
    deltas = []
    deltas.extend(paired_deltas(rows, "184", "185"))
    deltas.extend(paired_deltas(rows, "186", "187"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{out_prefix}.json"
    csv_path = args.out_dir / f"{out_prefix}.csv"
    md_path = args.out_dir / f"{out_prefix}.md"

    payload = {"rows": rows, "summaries": summaries, "deltas": deltas}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(md_path, rows, summaries, deltas)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
